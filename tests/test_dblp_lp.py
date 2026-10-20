"""Synthetic DBLP contract/dispatch checks; no preprocessing or GPU experiments."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "script"))
from lp_common import (audit_dblp_heldout, file_hash, full_target_negatives,
                       ranks_and_metrics, relation_count_features, run_directory,
                       summary_stem, validate_full_target_queries, write_json)
from lp_compare import main as compare_main
from lp_zero_shot import load_inputs, main, parser


def query_fixture():
    pool = np.arange(3, 23)
    types = np.array([1, 1, 1] + [3] * 20 + [4, 4])
    q = {}
    for head, split in enumerate(("train", "val", "test")):
        pos = np.array([[head, head + 3]])
        q[f"{split}_pos"] = pos
        neg = full_target_negatives(pos, pool)
        # Source flattened pair ordering is deliberately different from pool order.
        q[f"{split}_neg"] = np.column_stack((np.repeat(head, 19), neg[0, ::-1]))
    return q, types, pool


class DblpContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_root = Path(__file__).resolve().parents[1] / "tmp"
        cls.temp_root.mkdir(exist_ok=True)

    def test_candidate_pool_is_complete_and_positive_is_excluded(self):
        pos = np.array([[0, 5], [1, 12]])
        pool = np.arange(3, 23)
        neg = full_target_negatives(pos, pool)
        self.assertEqual(neg.shape, (2, 19))
        for positive, negatives in zip(pos[:, 1], neg):
            self.assertEqual(set(negatives) | {positive}, set(pool))
            self.assertNotIn(positive, negatives)

    def test_duplicate_or_missing_conferences_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "20 distinct"):
            full_target_negatives(np.array([[0, 3]]), np.full(20, 3))
        with self.assertRaisesRegex(ValueError, "missing"):
            full_target_negatives(np.array([[0, 100]]), np.arange(20))

    def test_flattened_source_order_does_not_change_candidate_protocol(self):
        q, types, pool = query_fixture()
        normalized = validate_full_target_queries(q, types, pool)
        np.testing.assert_array_equal(normalized["val_neg"], full_target_negatives(q["val_pos"], pool))

    def test_sampled_or_repeated_source_negatives_are_rejected(self):
        q, types, pool = query_fixture()
        q["test_neg"][0] = q["test_neg"][1]
        with self.assertRaisesRegex(ValueError, "full target complement"):
            validate_full_target_queries(q, types, pool)

    def test_paper_overlap_is_rejected(self):
        q, types, pool = query_fixture()
        q["test_pos"][0, 0] = 1
        q["test_neg"][:, 0] = 1
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_full_target_queries(q, types, pool)

    def test_heldout_conference_and_area_edges_are_rejected(self):
        q, _, _ = query_fixture()
        names = ["paper_to_conference", "paper_to_area", "author_to_area"]
        with self.assertRaisesRegex(ValueError, "label leaked"):
            audit_dblp_heldout(np.array([[1, 4, 0]]), names, q)
        with self.assertRaisesRegex(ValueError, "Area context"):
            audit_dblp_heldout(np.array([[2, 23, 1]]), names, q)
        # A shared author's contextual Area edge is allowed.
        audit_dblp_heldout(np.array([[0, 23, 2]]), names, q)

    def test_structural_features_include_inverses_and_only_active_selfloops(self):
        edges = np.array([[0, 1, 0], [1, 2, 1]])
        features = relation_count_features(edges, 4, 2)
        np.testing.assert_array_equal(features, [
            [1, 0, 0, 0, 1], [0, 1, 1, 0, 1],
            [0, 0, 0, 1, 1], [0, 0, 0, 0, 0]])
        self.assertEqual(features.dtype, np.float32)
        self.assertEqual(relation_count_features(edges, 4, 2, self_loop=False)[:, -1].sum(), 0)

    def test_profiles_preserve_imdb_defaults_and_add_dblp_defaults(self):
        self.assertEqual(parser("imdb").parse_args([]).data.name, "scr_imdb_lp")
        db = parser("dblp").parse_args([])
        self.assertEqual(db.data.name, "scr_dblp_lp")
        self.assertEqual(db.dataset, "dblp")

    def test_portable_dblp_load_requires_no_feature_file(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            root = Path(temp)
            task = root / "paper_conference"
            task.mkdir()
            q, _, pool = query_fixture()
            np.savez(task / "queries.npz", candidate_pool=pool, test_pos=q["test_pos"])
            np.savez(task / "v1.npz", physical=np.array([[0, 3, 0]]))
            meta = {"feature_policy": "scr_graph_relation_counts",
                    "files": {p.name: file_hash(p) for p in task.glob("*.npz")}}
            write_json(task / "manifest.json", meta)
            write_json(root / "manifest.json", {"format": "scr_dblp_lp_v1",
                       "tasks": {"paper_conference": file_hash(task / "manifest.json")}})
            args = SimpleNamespace(dataset="dblp", data=root, task="paper_conference",
                                   variant="v1", view="physical", split="test", limit=0, seed=1024)
            _, _, features, pos, neg, indices = load_inputs(args)
            self.assertIsNone(features)
            self.assertEqual(neg.shape, (1, 19))
            np.testing.assert_array_equal(pos, q["test_pos"])
            # A changed portable file is caught before model inference.
            (task / "v1.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Changed prepared input"):
                load_inputs(args)

    def test_dblp_suite_dispatches_nine_dblp_processes_without_running_them(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            root = Path(temp)
            data = root / "data"
            data.mkdir()
            write_json(data / "manifest.json", {"tasks": {"paper_conference": "stub"}})
            with patch.object(sys, "argv", ["eval_dblp_lp.py", "--data", str(data),
                                            "--output", str(root / "results")]), \
                 patch("lp_zero_shot.subprocess.run") as dispatch:
                main("dblp")
            self.assertEqual(dispatch.call_count, 9)
            commands = [call.args[0] for call in dispatch.call_args_list]
            self.assertTrue(all(Path(c[1]).name == "eval_dblp_lp.py" for c in commands))
            self.assertEqual(sum("--repeat" in c for c in commands), 1)
            self.assertTrue(all("--single" in c for c in commands))

    def test_dblp_summary_checks_full_suite_and_rejects_incompatible_runs(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            root = Path(temp)
            scores = np.ones((1, 20), dtype=np.float32)
            _, _, calculated = ranks_and_metrics(scores)
            q, _, pool = query_fixture()
            pos = q["test_pos"]
            candidates = np.column_stack((pos[:, 1], full_target_negatives(pos, pool)))
            paths = []
            for view in ("physical", "compiled"):
                variants = ["v1", "v2", "v3", "universal"] + (["v1__repeat"] if view == "physical" else [])
                for variant in variants:
                    run = run_directory(root, "dblp", "paper_conference", view,
                                        variant.replace("__repeat", ""), variant.endswith("__repeat"))
                    run.mkdir(parents=True)
                    meta = {**calculated, "comparison_contract": {"feature_policy": "scr_graph_relation_counts"},
                            "view": view, "repeat": variant.endswith("__repeat"),
                            "graph_sha256": "same-compiled" if view == "compiled" else variant,
                            "projected_features_sha256": "same-features" if view == "compiled" else variant}
                    write_json(run / "metrics.json", meta)
                    np.savez(run / "scores.npz", scores=scores, positives=pos,
                             candidates=candidates, query_index=np.array([0]))
                    paths.append(run / "metrics.json")
            argv = ["compare_dblp_lp.py", "--results", str(root), "--bootstrap", "0"]
            with patch.object(sys, "argv", argv):
                compare_main("dblp")
            result = json.loads((root / f"{summary_stem('dblp')}.json").read_text())["paper_conference"]
            self.assertTrue(result["compiled_views_bitwise_identical"])
            self.assertTrue(result["reference_repeat_bitwise_identical"])
            self.assertEqual(result["missing_standard_runs"], [])
            changed = json.loads(paths[-1].read_text())
            changed["comparison_contract"]["different_checkpoint"] = True
            write_json(paths[-1], changed)
            with patch.object(sys, "argv", argv), self.assertRaisesRegex(ValueError, "Incompatible"):
                compare_main("dblp")


if __name__ == "__main__":
    unittest.main()
