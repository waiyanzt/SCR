"""Export INV-GNN IMDb bundles to portable NumPy inputs for frozen SCR inference.

Run the source repository's split and graph preprocessing first (see
docs/IMDB_ZERO_SHOT.md). Only this exporter needs the source checkout.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from lp_common import (ROOT, TARGET_NAMES, TASK_VARIANTS, array_hash,
                            audit_heldout, file_hash, forward_edges,
                            validate_queries, write_json)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-repo", type=Path, default=ROOT / "INV-GNN-12D9")
    ap.add_argument("--bundle-root", type=Path, default=ROOT / "rai-datasets/preprocessed/RGCN_structural_v10_year_canonfeat/IMDB")
    ap.add_argument("--output", type=Path, default=ROOT / "rai-datasets/scr_imdb_lp")
    ap.add_argument("--tasks", nargs="+", choices=TASK_VARIANTS, default=list(TASK_VARIANTS))
    args = ap.parse_args()
    if args.output.exists():
        ap.error(f"Output already exists: {args.output}. Choose a new --output to avoid mixing exports.")
    source = args.source_repo.resolve()
    if not (source / "common/inv_rgcn/data.py").is_file():
        ap.error(f"Missing INV-GNN source checkout: {source}")
    for task in args.tasks:
        for variant in TASK_VARIANTS[task]:
            path = args.bundle_root / task / variant / "graph.pt"
            if not path.is_file():
                ap.error(f"Missing {path}. Run the preprocessing commands in docs/IMDB_ZERO_SHOT.md first.")

    sys.path.insert(0, str(source / "common"))
    from inv_rgcn.data import load_graph_data

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="imdb-export-", dir=args.output.parent) as temp:
        stage = Path(temp) / "export"
        stage.mkdir()
        source_files = [
            "models/RGCN/preprocess_IMDB_rgcn_structural.py",
            "dataset_construction/imdb/reconstruct_IMDB_md_ml_shared_splits.py",
            "dataset_construction/imdb/build_IMDB_my_shared_splits.py",
            "common/imdb_canonical_features.py", "common/inv_rgcn/data.py",
        ]
        manifest = {"format": "scr_imdb_lp_v1", "source_code": {
            name: file_hash(source / name) for name in source_files}, "tasks": {}}
        for task in args.tasks:
            out = stage / task
            out.mkdir()
            reference, records = None, {}
            for variant in TASK_VARIANTS[task]:
                path = args.bundle_root / task / variant / "graph.pt"
                data = load_graph_data(path)
                if data.x is None or data.metadata.get("feature_mode") != "pooled_bow":
                    raise ValueError("Expected canonical pooled_bow features; regenerate the source bundles")
                if data.metadata.get("task") != task or data.variant != variant:
                    raise ValueError(f"Wrong task/variant in {path}")
                if not data.metadata.get("lp_target_type_pooled_train_only"):
                    raise ValueError("Missing train-only target feature provenance")
                if task == "movie_year" and data.metadata.get("movie_year_neg_policy") != "freq_matched":
                    raise ValueError("Expected frequency-matched Movie-Year negatives")
                queries = {k: data.task_data[k].cpu().numpy() for k in (
                    "train_pos", "val_pos", "test_pos", "train_neg", "val_neg", "test_neg")}
                types = data.node_type.cpu().numpy()
                validate_queries(queries, types, 4 if task == "movie_year" else 1)
                current = (data.feature_hash, data.supervision_hash, data.semantic_hash,
                           array_hash(types), data.num_nodes)
                if reference is None:
                    reference = current
                    np.savez_compressed(out / "features.npz", x=data.x.cpu().numpy())
                    np.savez_compressed(out / "queries.npz", node_type=types, **queries)
                elif current != reference:
                    raise ValueError(f"Features, supervision, node ids, or compiled program differ: {task}/{variant}")
                arrays, effective = {}, {}
                names = None
                for view, prefix in (("physical", "physical"), ("compiled", "semantic")):
                    edges, vocab = forward_edges(
                        getattr(data, f"{prefix}_edge_index").cpu().numpy(),
                        getattr(data, f"{prefix}_edge_type").cpu().numpy(),
                        getattr(data, f"{prefix}_relation_names"))
                    if names is not None and vocab != names:
                        raise ValueError("Physical/compiled relation vocabularies differ")
                    names = vocab
                    offsets = data.metadata["offsets"]
                    audit_heldout(edges, vocab, queries, task, offsets["movie"], offsets["link"])
                    arrays[view] = edges
                    effective[view] = array_hash(edges)
                if TARGET_NAMES[task] not in names:
                    raise ValueError("Task relation missing from SCR vocabulary")
                np.savez_compressed(out / f"{variant}.npz", **arrays)
                records[variant] = {
                    "source_bundle_sha256": file_hash(path),
                    "source_physical_hash": data.physical_hash,
                    "source_semantic_hash": data.semantic_hash,
                    "edges_sha256": effective,
                    "edges": {k: len(v) for k, v in arrays.items()},
                }
                print(f"Exported {task}/{variant}", flush=True)
            if len({r["edges_sha256"]["compiled"] for r in records.values()}) != 1:
                raise ValueError(f"Compiled SCR graphs differ for {task}")
            task_meta = {
                "num_nodes": reference[4], "relation_names": names,
                "target_relation": names.index(TARGET_NAMES[task]),
                "feature_hash": reference[0], "supervision_hash": reference[1],
                "source_semantic_hash": reference[2], "variants": records,
                "protocol": "tail_only_positive_plus_19_fixed_negatives_average_ties",
                "message_semantics": "unit_weight_sets_with_internal_inverse_edges",
                "feature_policy": "canonical_pooled_bow_target_type_training_pairs_only",
                "negative_policy": "freq_matched" if task == "movie_year" else "uniform",
                "split_counts": {s: len(queries[f"{s}_pos"]) for s in ("train", "val", "test")},
            }
            task_meta["files"] = {p.name: file_hash(p) for p in out.glob("*.npz")}
            write_json(out / "manifest.json", task_meta)
            manifest["tasks"][task] = file_hash(out / "manifest.json")
        write_json(stage / "manifest.json", manifest)
        shutil.move(str(stage), str(args.output))
    print(f"Portable SCR inputs: {args.output.resolve()}")


if __name__ == "__main__":
    main()
