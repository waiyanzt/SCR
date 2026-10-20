"""Shared frozen-SCR inference for portable link-prediction datasets.

By default each graph is evaluated in a separate process, including a reference
repeat per task. Use --limit 64 for a smoke run; see the dataset guide in docs/.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

from lp_common import (DATASETS, ROOT, array_hash, file_hash, full_target_negatives,
                       ranks_and_metrics, relation_count_features, run_directory,
                       validate_run_tag, write_json)


def parser(dataset):
    profile = DATASETS[dataset]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / f"rai-datasets/scr_{dataset}_lp")
    ap.add_argument("--output", type=Path, default=ROOT / "results", help="Results root (default: <SCR>/results)")
    ap.add_argument("--run-tag", type=validate_run_tag, default=None,
                    help="Optional run-name suffix; defaults to smokeN when --limit N is used")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "ckpts/scr_3g.pth")
    ap.add_argument("--config", type=Path, default=ROOT / "config/eval_all.yaml")
    ap.add_argument("--task", choices=["all", *profile["variants"]], default="all")
    variants = sorted({v for values in profile["variants"].values() for v in values})
    ap.add_argument("--variant", choices=["all", *variants], default="all")
    ap.add_argument("--view", choices=["all", "physical", "compiled"], default="all")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1024)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--p", type=float, default=0.8)
    ap.add_argument("--limit", type=int, default=0, help="0 = all queries; otherwise a fixed keyed subset")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--no-repeat", action="store_true")
    ap.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--repeat", action="store_true", help=argparse.SUPPRESS)
    ap.set_defaults(dataset=dataset)
    return ap


def load_inputs(args):
    root = json.loads((args.data / "manifest.json").read_text())
    directory = args.data / args.task
    manifest_path = directory / "manifest.json"
    if root["format"] != DATASETS[args.dataset]["format"] or file_hash(manifest_path) != root["tasks"][args.task]:
        raise ValueError("Invalid or changed export manifest")
    meta = json.loads(manifest_path.read_text())
    for name, expected in meta["files"].items():
        if file_hash(directory / name) != expected:
            raise ValueError(f"Changed prepared input: {directory / name}")
    with np.load(directory / "queries.npz") as z:
        pos = z[f"{args.split}_pos"].copy()
        neg = (full_target_negatives(pos, z["candidate_pool"]) if "candidate_pool" in z.files
               else z[f"{args.split}_neg"].copy())
    with np.load(directory / f"{args.variant}.npz") as z:
        edges = z[args.view].copy()
    features = None
    if meta["feature_policy"] != "scr_graph_relation_counts":
        with np.load(directory / "features.npz") as z:
            features = z["x"].copy()
    indices = np.arange(len(pos))
    if args.limit:
        # Identity-keyed subset, shared by all variants and independent of file order.
        import hashlib
        keys = [hashlib.sha256(f"{args.seed}:{int(h)}:{int(t)}".encode()).digest() for h, t in pos]
        indices = np.sort(sorted(indices, key=lambda i: keys[i])[:args.limit])
        pos, neg = pos[indices], neg[indices]
    return meta, edges, features, pos, neg, indices


def parameter_hash(model):
    import hashlib
    h = hashlib.sha256()
    for name, value in model.named_parameters():
        h.update(name.encode())
        h.update(array_hash(value.detach().cpu().numpy()).encode())
    return h.hexdigest()


def evaluate(args):
    # No dependency on the ignored INV-GNN checkout after export.
    sys.path.insert(0, str(ROOT))
    import random
    import torch
    import yaml
    from torch_geometric.data import Data
    from ultra import tasks
    from ultra.models import SCR, GraphEncoding

    if args.task == "all" or args.view == "all" or args.variant == "all":
        raise ValueError("Single-process evaluation requires one task, view, and variant")
    run = run_directory(args.output, args.dataset, args.task, args.view,
                        args.variant, args.repeat, args.run_tag)
    if run.exists():
        raise FileExistsError(f"Refusing to overwrite {run}; choose a new --output")
    meta, edges, raw, pos, neg, indices = load_inputs(args)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use your configured SCR GPU environment")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))["model"]
    model = SCR(rel_model_cfg=cfg["relation_model"], entity_model_cfg=cfg["entity_model"])
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.requires_grad_(False)
    model.eval()
    before = parameter_hash(model)
    model = model.to(device)
    started = time.perf_counter()

    with torch.no_grad():
        encoder = GraphEncoding(cfg["entity_model"]["feat_dim"], "", {}, k=args.k, p=args.p)
        # Match SCR's supplied-feature mean fill, then project ONCE per task/seed
        # on CPU. Reset the RNG to make this independent of model construction.
        derived_features = meta["feature_policy"] == "scr_graph_relation_counts"
        if derived_features:
            raw = relation_count_features(edges, meta["num_nodes"], len(meta["relation_names"]),
                                          self_loop=cfg["entity_model"].get("self_loop", False))
        features = torch.as_tensor(raw, dtype=torch.float32)
        if not derived_features:
            features[features.sum(1) == 0] = features.mean(0)
        torch.manual_seed(args.seed)
        projected = encoder.svd_proj_2(features, encoder.hidden_dim) if features.shape[1] != encoder.hidden_dim else features
        projected_hash = array_hash(projected.numpy())
        projected = projected.to(device)
        del raw, features
        r = len(meta["relation_names"])
        forward = torch.as_tensor(edges, device=device)
        edge_index = torch.cat((forward[:, :2].T, forward[:, :2].T.flip(0)), dim=1)
        edge_type = torch.cat((forward[:, 2], forward[:, 2] + r))
        graph = Data(edge_index=edge_index, edge_type=edge_type,
                     num_nodes=meta["num_nodes"], num_relations=torch.tensor(2 * r, device=device))
        graph = tasks.build_relation_graph(graph)
        graph = tasks.build_relation_types(graph)
        if cfg["entity_model"].get("self_loop", False):
            graph = tasks.build_selfloop_edges(graph)
        # Use SCR's own semantic-neighbor construction without reading or writing
        # its dataset-name caches. The projection is shared across representations.
        node_embeds, triples, neighbors = encoder.build_faiss(graph, projected, k=args.k, p=args.p)
        if triples is None:
            raise RuntimeError("SCR produced no semantic neighbors; upstream core_process cannot handle this input")
        extended = copy.deepcopy(graph)
        extended.edge_index = torch.cat((extended.edge_index, triples[:, (0, 2)].T), dim=1)
        extended.edge_type = torch.cat((extended.edge_type, triples[:, 1] + extended.num_relations))
        extended.num_relations = extended.num_relations + 2
        extended = tasks.build_relation_graph(extended)
        graph.ent_embeds = node_embeds
        graph.topk_indices = neighbors
        graph.ex_edge_index, graph.ex_edge_type = extended.edge_index, extended.edge_type
        graph.ex_num_relations, graph.ex_relation_graph = extended.num_relations, extended.relation_graph
        candidates = np.column_stack((pos[:, 1], neg))
        scores = np.empty(candidates.shape, dtype=np.float32)
        for start in range(0, len(pos), args.batch_size):
            end = min(start + args.batch_size, len(pos))
            batch = torch.empty((end - start, 3), dtype=torch.long, device=device)
            batch[:, :2] = torch.as_tensor(pos[start:end], device=device)
            batch[:, 2] = meta["target_relation"]
            # Preserve the upstream SCR all-tail forward path, then restrict the
            # ranking to the benchmark candidates. Never average head prediction.
            t_batch, unused_heads = tasks.all_negative(graph, batch)
            del unused_heads
            predictions = model(graph, t_batch)
            kept = torch.as_tensor(candidates[start:end], device=device)
            scores[start:end] = predictions.gather(1, kept).cpu().numpy()
            if start == 0 or end == len(pos) or end % 200 == 0:
                print(f"{args.task}/{args.view}/{args.variant}: {end}/{len(pos)}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    after = parameter_hash(model)
    if before != after:
        raise RuntimeError("Model parameters changed during frozen inference")
    ranks, pess, metrics = ranks_and_metrics(scores)
    contract = {
        "checkpoint_sha256": file_hash(args.checkpoint), "config_sha256": file_hash(args.config),
        "task_manifest_sha256": file_hash(args.data / args.task / "manifest.json"),
        "queries_sha256": array_hash(pos, candidates), "feature_policy": meta["feature_policy"],
        "parameters_sha256": before, "seed": args.seed, "k": args.k, "p": args.p,
        "split": args.split, "batch_size": args.batch_size,
        "torch": torch.__version__, "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "adapter_sha256": file_hash(Path(__file__)),
        "helpers_sha256": file_hash(ROOT / "script/lp_common.py"),
    }
    if not derived_features:
        contract["projected_features_sha256"] = projected_hash
    metrics.update({"task": args.task, "variant": args.variant, "view": args.view,
                    "repeat": args.repeat, "num_queries": len(pos), "num_candidates": 20,
                    "dataset": args.dataset, "run_tag": args.run_tag,
                    "protocol": meta["protocol"], "comparison_contract": contract,
                    "projected_features_sha256": projected_hash,
                    "graph_sha256": array_hash(edges), "weights_unchanged": True,
                    "seconds": elapsed, "device": str(device), "torch": torch.__version__,
                    "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0})
    run.mkdir(parents=True)
    np.savez_compressed(run / "scores.npz", query_index=indices, positives=pos,
                        candidates=candidates, scores=scores, rank_tie=ranks, rank_pess=pess)
    write_json(run / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)


def main(dataset="imdb"):
    ap = parser(dataset)
    args = ap.parse_args()
    profile = DATASETS[dataset]
    task_variants = profile["variants"]
    if args.batch_size < 1 or args.k < 1 or args.limit < 0 or not -1 <= args.p <= 1:
        ap.error("Require batch-size/k > 0, limit >= 0, and -1 <= p <= 1")
    if args.run_tag is None:
        args.run_tag = f"smoke{args.limit}" if args.limit else ""
    if args.single:
        evaluate(args)
        return
    tasks = list(task_variants) if args.task == "all" else [args.task]
    views = ["physical", "compiled"] if args.view == "all" else [args.view]
    jobs = []
    for task in tasks:
        variants = task_variants[task] if args.variant == "all" else [args.variant]
        if any(v not in task_variants[task] for v in variants):
            ap.error(f"{task} only defines {task_variants[task]}")
        jobs.extend((task, view, v, False) for view in views for v in variants)
        if not args.no_repeat and "physical" in views and "v1" in variants:
            jobs.append((task, "physical", "v1", True))
    # Preflight before launching CUDA imports or expensive jobs.
    if not (args.data / "manifest.json").is_file():
        ap.error(f"Prepared {profile['label']} inputs missing; see docs/{profile['doc']}")
    exported = json.loads((args.data / "manifest.json").read_text())
    for task in tasks:
        if task not in exported.get("tasks", {}):
            ap.error(f"Task {task} was not exported; prepare it or select --task explicitly")
    for path in (args.checkpoint, args.config):
        if not path.is_file():
            ap.error(f"Missing {path}")
    for task, view, variant, repeat in jobs:
        run = run_directory(args.output, dataset, task, view, variant, repeat, args.run_tag)
        if run.exists():
            ap.error(f"Output already exists: {run}. Choose a new --output.")
    for task, view, variant, repeat in jobs:
        entrypoint = ROOT / "script" / f"eval_{dataset}_lp.py"
        cmd = [sys.executable, str(entrypoint), "--single", "--task", task,
               "--view", view, "--variant", variant]
        for name in ("data", "output", "checkpoint", "config", "device", "batch_size", "seed", "k", "p", "limit", "split", "run_tag"):
            cmd.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        if repeat:
            cmd.append("--repeat")
        print(f"Launching {task}/{view}/{variant}" + (" repeat" if repeat else ""), flush=True)
        subprocess.run(cmd, check=True, env={**os.environ, "PYTHONHASHSEED": str(args.seed)})


if __name__ == "__main__":
    main()
