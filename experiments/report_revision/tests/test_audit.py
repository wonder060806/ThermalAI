import sys
import unittest
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from revision.audit import (
    array_fingerprint,
    cross_split_audit,
    dataset_temperature_summary,
    duplicate_groups,
    nearest_neighbor_distances,
    numeric_parameter_summary,
    predict_global_mean,
    total_power_regression_predictions,
)


class AuditTests(unittest.TestCase):
    def test_cross_split_audit_finds_duplicate_and_nearest_train_case(self):
        fields = {
            "train_a": np.array([[1.0, 2.0]]),
            "train_b": np.array([[10.0, 20.0]]),
            "test_x": np.array([[1.0, 2.0]]),
            "test_y": np.array([[9.0, 19.0]]),
        }
        result = cross_split_audit(fields, ["train_a", "train_b"], ["test_x", "test_y"])
        self.assertEqual(result["exact_cross_split_duplicates"], [["train_a", "test_x"]])
        self.assertEqual(result["nearest_train_by_test"]["test_y"]["train_case"], "train_b")

    def test_numeric_parameter_summary_exposes_narrow_ranges(self):
        result = numeric_parameter_summary([
            {"id": 0, "htc": 100.0, "power": 5.0},
            {"id": 1, "htc": 100.0, "power": 7.0},
        ])
        self.assertEqual(result["htc"]["range"], 0.0)
        self.assertEqual(result["power"]["range"], 2.0)

    def test_fingerprint_distinguishes_exact_and_rounded_duplicates(self):
        first = np.array([1.0001, 2.0001])
        near = np.array([1.0002, 2.0002])

        self.assertNotEqual(array_fingerprint(first), array_fingerprint(near))
        self.assertEqual(
            array_fingerprint(first, decimals=3),
            array_fingerprint(near, decimals=3),
        )

    def test_duplicate_groups_reports_only_shared_fingerprints(self):
        arrays = {
            "a": np.array([1.0, 2.0]),
            "b": np.array([1.0, 2.0]),
            "c": np.array([3.0, 4.0]),
        }
        self.assertEqual(duplicate_groups(arrays), [["a", "b"]])

    def test_nearest_neighbor_distance_is_normalized_per_element(self):
        arrays = {
            "a": np.array([0.0, 0.0]),
            "b": np.array([3.0, 4.0]),
            "c": np.array([0.0, 2.0]),
        }
        result = nearest_neighbor_distances(arrays)
        self.assertEqual(result["a"]["neighbor"], "c")
        self.assertAlmostEqual(result["a"]["rmse"], np.sqrt(2.0))

    def test_temperature_summary_exposes_degenerate_dynamic_range(self):
        fields = {
            "cold": np.array([[300.0, 300.5], [300.2, 300.1]]),
            "hot": np.array([[300.0, 310.0], [305.0, 302.0]]),
        }
        summary = dataset_temperature_summary(fields, ambient_k=293.0)
        self.assertAlmostEqual(summary["cases"]["cold"]["range_k"], 0.5)
        self.assertAlmostEqual(summary["cases"]["hot"]["max_rise_k"], 17.0)
        self.assertEqual(summary["n_cases"], 2)

    def test_trivial_baselines_have_expected_predictions(self):
        train = np.array([[300.0, 302.0], [304.0, 306.0]])
        self.assertTrue(np.allclose(predict_global_mean(train, (2, 2)), 303.0))

        powers = np.array([1.0, 2.0, 3.0])
        temperatures = np.array([301.0, 303.0, 305.0])
        predicted = total_power_regression_predictions(
            powers[:2], temperatures[:2], powers[2:]
        )
        self.assertTrue(np.allclose(predicted, [305.0]))


if __name__ == "__main__":
    unittest.main()
