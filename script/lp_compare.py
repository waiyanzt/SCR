"""Compare frozen SCR candidate scores across graph representations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lp_common import (DATASETS, ROOT, ranks_and_metrics, run_directory,
                       summary_stem, validate_run_tag, write_json)


def compare_scores(reference, other, bootstrap=1000):
    if reference.shape != other.shape:
        raise ValueError("Candidate-score shapes differ")
    a, _, _ = ranks_and_metrics(reference)
    b, _, _ = ranks_and_metrics(other)
    # Pilot convention: identical vectors -> 1, undefined nonidentical vectors -> 0.
    tau = []
    for x, y in zip(reference, other):
        if np.array_equal(x, y):
            tau.append(1.0)
        elif np.ptp(x) == 0 or np.ptp(y) == 0:
            tau.append(0.0)
        else:
            from scipy.stats import kendalltau
            value = kendalltau(x, y).statistic
            tau.append(float(value) if np.isfinite(value) else 0.0)
    delta = 1 / b - 1 / a
    result = {
        "tau_b": float(np.mean(tau)), "delta_mrr": float(delta.mean()),
        "hit1_flip_rate": float(np.mean((a <= 1) != (b <= 1))),
        "hit10_flip_rate": float(np.mean((a <= 10) != (b <= 10))),
        "bitwise_identical": bool(np.array_equal(reference, other)),
        "num_queries": len(a),
    }
    if bootstrap:
        rng = np.random.default_rng(0)
        draws = [delta[rng.integers(len(delta), size=len(delta))].mean() for _ in range(bootstrap)]
        result["delta_mrr_95ci"] = np.quantile(draws, [0.025, 0.975]).tolist()
    return result


def main(dataset="imdb"):
    profile = DATASETS[dataset]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--run-tag", type=validate_run_tag, default="", help="Select an evaluation run-name suffix")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()
    if args.bootstrap < 0:
        ap.error("bootstrap must be nonnegative")
    paths = [run_directory(args.results, dataset, task, "physical", "v1", run_tag=args.run_tag) / "metrics.json"
             for task in profile["variants"]]
    paths = [path for path in paths if path.is_file()]
    if not paths:
        ap.error("No physical/v1 reference results found")
    summary = {}
    lines = [f"# SCR {profile['label']} zero-shot results", "", "20 candidate tails; average ranks for ties.", ""]
    for reference_path in paths:
        task_name = reference_path.parent.name
        ref_meta = json.loads(reference_path.read_text())
        with np.load(reference_path.with_name("scores.npz")) as z:
            ref = {key: z[key].copy() for key in z.files}
        records, compiled = {}, []
        lines.extend([f"## {task_name}", "", "| Input | MRR | Hits@10 | tau vs physical v1 | delta MRR |", "|---|---:|---:|---:|---:|"])
        repeat_ok = None
        runs = [(view, variant, False) for view in ("physical", "compiled")
                for variant in profile["variants"][task_name]] + [("physical", "v1", True)]
        for view, variant, repeat in runs:
            path = run_directory(args.results, dataset, task_name, view, variant, repeat, args.run_tag) / "metrics.json"
            if not path.is_file():
                continue
            meta = json.loads(path.read_text())
            if meta["comparison_contract"] != ref_meta["comparison_contract"]:
                raise ValueError(f"Incompatible checkpoint, features, queries, or settings: {path}")
            with np.load(path.with_name("scores.npz")) as z:
                for key in ("query_index", "positives", "candidates"):
                    if not np.array_equal(ref[key], z[key]):
                        raise ValueError(f"Query/candidate order differs: {path}")
                scores = z["scores"].copy()
            _, _, calculated = ranks_and_metrics(scores)
            if not np.isclose(calculated["mrr"], meta["mrr"], atol=1e-12, rtol=0):
                raise ValueError(f"Stored metrics do not match candidate scores: {path}")
            comparison = compare_scores(ref["scores"], scores, args.bootstrap)
            name = f"{view}/{variant}" + ("__repeat" if repeat else "")
            records[name] = {"metrics": meta, "comparison": comparison}
            lines.append(f"| {name} | {meta['mrr']:.4f} | {meta['hits@10']:.4f} | {comparison['tau_b']:.4f} | {comparison['delta_mrr']:+.4f} |")
            if meta["view"] == "compiled":
                compiled.append(((meta["graph_sha256"], meta.get("projected_features_sha256",
                                 meta["comparison_contract"].get("projected_features_sha256"))), scores))
            if meta["repeat"]:
                repeat_ok = comparison["bitwise_identical"]
        expected = {f"{view}/{variant}" for view in ("physical", "compiled")
                    for variant in profile["variants"][task_name]}
        expected.add("physical/v1__repeat")
        missing = sorted(expected - records.keys())
        compiled_complete = all(f"compiled/{v}" in records for v in profile["variants"][task_name])
        compiled_ok = all(h == compiled[0][0] and np.array_equal(s, compiled[0][1]) for h, s in compiled) if compiled_complete else None
        summary[task_name] = {
            "runs": records, "reference_repeat_bitwise_identical": repeat_ok,
            "compiled_views_bitwise_identical": compiled_ok,
            "missing_standard_runs": missing,
        }
        lines.extend(["", f"Reference repeat bitwise identical: {repeat_ok}. Compiled views bitwise identical: {compiled_ok}.", ""])
        if missing:
            lines.extend(["Partial suite. Missing: " + ", ".join(missing) + ".", ""])
    stem = summary_stem(dataset, args.run_tag)
    write_json(args.results / f"{stem}.json", summary)
    (args.results / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.results / (stem + '.md')}")


if __name__ == "__main__":
    main()
