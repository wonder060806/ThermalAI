import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from revision.aggregate import (
    _exact_sign_flip_p_value,
    aggregate_paired_results,
    aggregate_physics_results,
    aggregate_seeded_metrics,
)


class AggregateTests(unittest.TestCase):
    def test_paired_aggregation_reads_current_training_mae_key(self):
        records = []
        for method, mae in (("pinn", 1.5), ("random", 2.0)):
            records.append({
                "status": "completed",
                "config": {
                    "method": method, "train_size": 10,
                    "init_seed": 0, "data_seed": 0,
                },
                "test": {"aggregate": {"mae_k": mae}},
            })

        report = aggregate_paired_results(records, {10: [0]})

        self.assertEqual(report["status"], "complete")
        self.assertAlmostEqual(
            report["groups"]["10"]["pinn_minus_random_mae_k"]["mean"],
            -0.5,
        )

    def test_exact_sign_flip_test_exposes_five_pair_resolution_limit(self):
        self.assertEqual(_exact_sign_flip_p_value([-1.0] * 5), 0.0625)
        self.assertEqual(_exact_sign_flip_p_value([0.0, 0.0]), 1.0)
    def test_paired_aggregation_rejects_unpaired_data_subsets(self):
        records = [
            {"status": "completed", "config": {"method": "pinn", "train_size": 10,
             "init_seed": 0, "data_seed": 0}, "test": {"aggregate": {"mae_k_mean": 1.0}}},
            {"status": "completed", "config": {"method": "random", "train_size": 10,
             "init_seed": 0, "data_seed": 1}, "test": {"aggregate": {"mae_k_mean": 2.0}}},
        ]
        result = aggregate_paired_results(records, {10: [0]})
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["groups"]["10"]["missing_seeds"], [0])

    def test_small_seed_confidence_interval_uses_student_t(self):
        records = []
        for seed, difference in enumerate((1.0, 2.0, 3.0)):
            records.extend([
                {"status": "completed", "config": {"method": "random", "train_size": 10, "init_seed": seed},
                 "test": {"aggregate": {"mae_k_mean": 10.0}}},
                {"status": "completed", "config": {"method": "pinn", "train_size": 10, "init_seed": seed},
                 "test": {"aggregate": {"mae_k_mean": 10.0 + difference}}},
            ])
        interval = aggregate_paired_results(records, {10: [0, 1, 2]})["groups"]["10"]["pinn_minus_random_mae_k"]
        self.assertAlmostEqual(interval["ci95_low"], -0.4841, places=3)
        self.assertAlmostEqual(interval["ci95_high"], 4.4841, places=3)

    def test_seeded_metric_aggregation_detects_missing_seed(self):
        records = [{"status": "completed", "config": {"init_seed": seed},
                    "test": {"aggregate": {"mae_k": 1.0 + seed,
                                             "hotspot_location_error_um": 10.0}}}
                   for seed in (0, 1)]
        result = aggregate_seeded_metrics(records, expected_seeds=(0, 1, 2))
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["missing_seeds"], [2])
        self.assertAlmostEqual(result["metrics"]["mae_k"]["mean"], 1.5)

    def test_physics_aggregation_pairs_each_setting_with_supervised_seed(self):
        records = []
        for seed, baseline, corrected in [(0, 3.0, 2.0), (1, 2.0, 2.5)]:
            for variant, weight, mae in (("none", 0.0, baseline), ("volumetric_source", 0.1, corrected)):
                records.append({"status": "completed", "config": {
                    "init_seed": seed, "physics_variant": variant, "physics_weight": weight,
                }, "test": {"aggregate": {"mae_k_mean": mae}}})
        result = aggregate_physics_results(
            records, variants=("volumetric_source",), weights=(0.1,), seeds=(0, 1)
        )
        setting = result["settings"]["volumetric_source:w=0.1"]
        self.assertEqual(result["status"], "complete")
        self.assertAlmostEqual(setting["mae_delta_vs_supervised_k"]["mean"], -0.25)
        self.assertEqual(setting["win_rate_vs_supervised"], 0.5)
        self.assertIn("paired_t_p_value_two_sided", setting)
        self.assertIn("exact_sign_flip_p_value_two_sided", setting)
        self.assertIn("holm_adjusted_p_value", setting)
        self.assertIn("significant_after_holm_0p05", setting)

    def test_paired_aggregation_reports_direction_and_missing_runs(self):
        records = []
        for seed, pinn, random in [(0, 2.0, 3.0), (1, 2.5, 2.0), (2, 1.0, 2.0)]:
            for method, mae in (("pinn", pinn), ("random", random)):
                records.append({"status": "completed", "config": {
                    "method": method, "train_size": 10, "init_seed": seed,
                }, "test": {"aggregate": {"mae_k_mean": mae}}})
        report = aggregate_paired_results(records, expected_seeds={10: [0, 1, 2, 3]})
        group = report["groups"]["10"]
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(group["paired_count"], 3)
        self.assertEqual(group["missing_seeds"], [3])
        self.assertAlmostEqual(group["pinn_minus_random_mae_k"]["mean"], -0.5)
        self.assertAlmostEqual(group["pinn_win_rate"], 2 / 3)
        self.assertIn("paired_t_p_value_two_sided", group)
        self.assertIn("holm_adjusted_p_value", group)


if __name__ == "__main__":
    unittest.main()
