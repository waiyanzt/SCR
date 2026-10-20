"""Portable data checks and candidate-ranking metrics for frozen SCR pilots."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

TASK_VARIANTS = {
    "movie_year": ("v1", "v2", "v3", "v4", "universal"),
    "movie_director": ("v1", "v3", "universal"),
}
TARGET_NAMES = {"movie_year": "movie_to_year", "movie_director": "movie_to_director"}
ROOT = Path(__file__).resolve().parents[1]
DBLP_TASK_VARIANTS = {"paper_conference": ("v1", "v2", "v3", "universal")}
DATASETS = {
    "imdb": {"label": "IMDb", "format": "scr_imdb_lp_v1", "variants": TASK_VARIANTS,
             "doc": "IMDB_ZERO_SHOT.md"},
    "dblp": {"label": "DBLP", "format": "scr_dblp_lp_v1", "variants": DBLP_TASK_VARIANTS,
             "doc": "DBLP_ZERO_SHOT.md"},
}


def validate_run_tag(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]*", value):
        raise ValueError("Run tags may contain only letters, digits, underscores, and hyphens")
    return value


def run_directory(output, dataset, task, view, variant, repeat=False, run_tag=""):
    name = f"scr_zeroshot_{dataset}{variant}_lp"
    if view == "compiled":
        name += "_compiled"
    if repeat:
        name += "_repeat"
    if run_tag:
        name += f"_{validate_run_tag(run_tag)}"
    return Path(output) / name / task


def summary_stem(dataset, run_tag=""):
    tag = f"_{validate_run_tag(run_tag)}" if run_tag else ""
    return f"scr_zeroshot_{dataset}_lp{tag}_summary"


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def forward_edges(edge_index, edge_type, names):
    """Verify mirror pairs, retain one direction, and remap ids for SCR inverses.

    Returns [E, 3] in SCR's (head, tail, relation) order, including empty relation
    slots in the vocabulary. Repeated facts are rejected, not silently weighted.
    """
    rows = np.column_stack((edge_index.T, edge_type)).astype(np.int64)
    if len(np.unique(rows, axis=0)) != len(rows):
        raise ValueError("Duplicate typed edges in source graph")
    kept, pieces = [], []
    seen = set()
    for rid, name in enumerate(names):
        if name in seen:
            continue
        left, right = name.split("_to_")
        reverse = f"{right}_to_{left}"
        if reverse not in names or reverse == name:
            raise ValueError(f"Missing distinct reverse relation for {name}")
        rev_id = names.index(reverse)
        a = set(map(tuple, rows[rows[:, 2] == rid, :2]))
        b = set(map(tuple, rows[rows[:, 2] == rev_id, :2][:, ::-1]))
        if a != b:
            raise ValueError(f"Reverse edges do not match: {name}/{reverse}")
        new_id = len(kept)
        kept.append(name)
        part = rows[rows[:, 2] == rid].copy()
        part[:, 2] = new_id
        pieces.append(part)
        seen.update((name, reverse))
    out = np.concatenate(pieces)
    order = np.lexsort((out[:, 0], out[:, 1], out[:, 2]))
    return out[order], kept


def validate_queries(queries, node_type, target_type, head_type=0):
    """Check head-disjoint supervision and the exact 20-candidate contract."""
    heads = []
    for split in ("train", "val", "test"):
        pos, neg = queries[f"{split}_pos"], queries[f"{split}_neg"]
        if pos.ndim != 2 or pos.shape[1] != 2 or neg.shape != (len(pos), 19):
            raise ValueError(f"Invalid {split} positive/negative dimensions")
        if not len(pos) or len(np.unique(pos[:, 0])) != len(pos):
            raise ValueError(f"Empty or repeated head queries in {split}")
        if min(pos.min(), neg.min()) < 0 or max(pos.max(), neg.max()) >= len(node_type):
            raise ValueError("Query node id out of bounds")
        if np.any(node_type[pos[:, 0]] != head_type):
            raise ValueError("Query head is outside the task head node type")
        if np.any(node_type[pos[:, 1]] != target_type) or np.any(node_type[neg] != target_type):
            raise ValueError("Candidate is outside the target node type")
        if np.any(neg == pos[:, 1, None]):
            raise ValueError("A negative equals the positive target")
        heads.extend(pos[:, 0].tolist())
    if len(set(heads)) != len(heads):
        raise ValueError("Heads overlap across training/validation/test")


def full_target_negatives(positives, pool):
    """Every other target, in fixed pool order; positive is scored first."""
    pool = np.asarray(pool, dtype=np.int64)
    if pool.ndim != 1 or len(pool) != 20 or len(np.unique(pool)) != 20:
        raise ValueError("Expected exactly 20 distinct conference candidates")
    if not np.isin(positives[:, 1], pool).all():
        raise ValueError("Positive target missing from conference pool")
    all_targets = np.broadcast_to(pool, (len(positives), len(pool)))
    return all_targets[all_targets != positives[:, 1, None]].reshape(len(positives), 19)


def validate_full_target_queries(queries, node_type, pool):
    """Validate DBLP's flattened negatives without sampling or trusting row order."""
    normalized = {}
    for split in ("train", "val", "test"):
        pos, flat = queries[f"{split}_pos"], queries[f"{split}_neg"]
        if pos.ndim != 2 or pos.shape[1] != 2 or not len(pos):
            raise ValueError(f"Invalid {split} positive pairs")
        neg = full_target_negatives(pos, pool)
        expected = {(int(h), int(t)) for h, row in zip(pos[:, 0], neg) for t in row}
        if flat.shape != (len(pos) * 19, 2) or set(map(tuple, flat.tolist())) != expected:
            raise ValueError(f"{split} source negatives are not the full target complement")
        normalized[f"{split}_pos"], normalized[f"{split}_neg"] = pos, neg
    validate_queries(normalized, node_type, target_type=3, head_type=1)
    return normalized


def audit_dblp_heldout(edges, names, queries):
    """Reject held-out P-C edges and paper-specific Area context in train scope."""
    heldout = np.concatenate([queries[f"{s}_pos"] for s in ("val", "test")])
    pairs = set(map(tuple, heldout.tolist()))
    for relation, head_column, tail_column in (("paper_to_conference", 0, 1),
                                               ("conference_to_paper", 1, 0)):
        if relation in names:
            rows = edges[edges[:, 2] == names.index(relation)]
            if any((int(row[head_column]), int(row[tail_column])) in pairs for row in rows):
                raise ValueError(f"Held-out label leaked via {relation}")
    papers = set(heldout[:, 0].tolist())
    for relation, column in (("paper_to_area", 0), ("area_to_paper", 1)):
        if relation in names and any(int(row[column]) in papers for row in edges[edges[:, 2] == names.index(relation)]):
            raise ValueError("Held-out paper has training-only Area context")


def relation_count_features(edges, num_nodes, num_forward_relations, self_loop=True):
    """SCR's top/ontology fallback, including internal inverses and identity edges."""
    r = num_forward_relations
    out = np.zeros((num_nodes, 2 * r + 1), dtype=np.float32)
    np.add.at(out, (edges[:, 0], edges[:, 2]), 1)
    np.add.at(out, (edges[:, 1], edges[:, 2] + r), 1)
    if self_loop:
        out[np.unique(edges[:, :2]), 2 * r] = 1
    return out


def audit_heldout(edges, names, queries, task, movie_offset, link_offset):
    """Reject held-out target edges on movies AND their declared Link copies."""
    triples = set(map(tuple, edges.tolist()))
    suffix = "year" if task == "movie_year" else "director"
    for split in ("val", "test"):
        for movie, target in queries[f"{split}_pos"]:
            for kind, head in (("movie", movie), ("link", movie - movie_offset + link_offset)):
                name = f"{kind}_to_{suffix}"
                if name in names and (head, target, names.index(name)) in triples:
                    raise ValueError(f"Held-out label leaked via {name}: {movie}->{target}")


def ranks_and_metrics(scores):
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[1] < 2 or not np.isfinite(scores).all():
        raise ValueError("Expected finite [queries, positive + negatives] scores")
    better = (scores[:, 1:] > scores[:, :1]).sum(axis=1)
    tied = (scores[:, 1:] == scores[:, :1]).sum(axis=1)
    ranks = 1.0 + better + tied / 2.0
    pess = 1 + better + tied
    metrics = {"mrr": float(np.mean(1 / ranks)), "mr": float(ranks.mean())}
    metrics.update({f"hits@{k}": float(np.mean(ranks <= k)) for k in (1, 3, 10)})
    metrics["mrr_pessimistic"] = float(np.mean(1 / pess))
    return ranks, pess, metrics
