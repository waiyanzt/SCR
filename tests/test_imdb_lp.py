"""CPU-only checks of the experimental contract; no SCR experiment is run."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "script"))
from lp_common import audit_heldout, forward_edges, ranks_and_metrics, validate_queries
from lp_compare import compare_scores


class ImdbContractTests(unittest.TestCase):
    def test_forward_selection_preserves_empty_relation_slots(self):
        names = ["movie_to_year", "year_to_movie", "movie_to_actor", "actor_to_movie"]
        edges, vocab = forward_edges(np.array([[0, 3], [3, 0]]), np.array([0, 1]), names)
        np.testing.assert_array_equal(edges, [[0, 3, 0]])
        self.assertEqual(vocab, ["movie_to_year", "movie_to_actor"])

    def test_asymmetric_reverse_pair_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Reverse edges"):
            forward_edges(np.array([[0], [3]]), np.array([0]), ["movie_to_year", "year_to_movie"])

    def test_duplicate_edges_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            forward_edges(np.array([[0, 0], [3, 3]]), np.array([0, 0]), ["movie_to_year", "year_to_movie"])

    def test_average_ties_and_repeated_negative_candidates(self):
        # Repeated sampled years remain repeated; do not deduplicate the benchmark.
        ranks, pess, metrics = ranks_and_metrics(np.array([[2, 3, 2, 2], [4, 0, 1, 2]]))
        np.testing.assert_array_equal(ranks, [3, 1])
        np.testing.assert_array_equal(pess, [4, 1])
        self.assertAlmostEqual(metrics["mrr"], 2 / 3)
        self.assertEqual(metrics["hits@1"], 0.5)

    def test_nonfinite_scores_are_rejected(self):
        with self.assertRaises(ValueError):
            ranks_and_metrics([[1, float("nan")]])

    def test_identical_constant_scores_define_repeat_floor(self):
        scores = np.ones((3, 20))
        result = compare_scores(scores, scores.copy(), bootstrap=20)
        self.assertEqual(result["tau_b"], 1)
        self.assertTrue(result["bitwise_identical"])
        self.assertEqual(result["delta_mrr_95ci"], [0, 0])

    def test_nonidentical_constant_scores_have_zero_tau(self):
        result = compare_scores(np.ones((1, 20)), np.zeros((1, 20)), bootstrap=0)
        self.assertEqual(result["tau_b"], 0)
        self.assertFalse(result["bitwise_identical"])

    def test_link_copy_of_heldout_label_is_rejected(self):
        queries = {"val_pos": np.array([[0, 6]]), "test_pos": np.array([[1, 7]])}
        names = ["movie_to_year", "link_to_year"]
        with self.assertRaisesRegex(ValueError, "link_to_year"):
            audit_heldout(np.array([[3, 6, 1]]), names, queries, "movie_year", 0, 3)
        audit_heldout(np.array([[2, 6, 0]]), names, queries, "movie_year", 0, 3)

    def test_same_movie_cannot_appear_in_two_splits(self):
        q = {}
        for split, head in zip(("train", "val", "test"), (0, 1, 2)):
            q[f"{split}_pos"] = np.array([[head, 3]])
            q[f"{split}_neg"] = np.full((1, 19), 4)
        types = np.array([0, 0, 0, 4, 4])
        validate_queries(q, types, 4)
        q["test_pos"][0, 0] = 1
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_queries(q, types, 4)

    def test_wrong_candidate_type_and_true_negatives_are_rejected(self):
        q = {}
        for split, head in zip(("train", "val", "test"), (0, 1, 2)):
            q[f"{split}_pos"] = np.array([[head, 3]])
            q[f"{split}_neg"] = np.full((1, 19), 4)
        types = np.array([0, 0, 0, 4, 4])
        q["val_neg"][0, 0] = 0
        with self.assertRaisesRegex(ValueError, "target node type"):
            validate_queries(q, types, 4)
        q["val_neg"][0, 0] = 3
        with self.assertRaisesRegex(ValueError, "positive target"):
            validate_queries(q, types, 4)


if __name__ == "__main__":
    unittest.main()
