import json
import sys
import tempfile
import unittest
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from revision.training_config import load_training_selection, validate_training_config


class TrainingConfigTests(unittest.TestCase):
    def test_selection_reads_exact_nested_subset_and_fixed_test(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "split.json"
            path.write_text(json.dumps({"split": {
                "train_subsets": {"10": [str(i) for i in range(10)]},
                "test": ["20", "21"],
            }}), encoding="utf-8")
            train_ids, test_ids = load_training_selection(path, 10)
            self.assertEqual(train_ids, list(range(10)))
            self.assertEqual(test_ids, [20, 21])

    def test_selection_rejects_overlap_even_if_manifest_was_edited(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "split.json"
            path.write_text(json.dumps({"split": {
                "train_subsets": {"1": ["7"]}, "test": ["7"]
            }}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_training_selection(path, 1)

    def test_selection_reads_requested_data_replicate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "split.json"
            path.write_text(json.dumps({
                "split": {"train_subsets": {"2": ["0", "1"]}, "test": ["9"]},
                "replicates": {
                    "0": {"train_subsets": {"2": ["0", "1"]}},
                    "1": {"train_subsets": {"2": ["2", "3"]}},
                },
            }), encoding="utf-8")
            train_ids, test_ids = load_training_selection(path, 2, data_seed=1)
            self.assertEqual(train_ids, [2, 3])
            self.assertEqual(test_ids, [9])

    def test_random_method_rejects_checkpoint_and_pinn_requires_it(self):
        with self.assertRaisesRegex(ValueError, "must not"):
            validate_training_config({
                "method": "random", "checkpoint": "model.pth",
                "epochs": 10, "learning_rate": 1e-4,
            })
        with self.assertRaisesRegex(ValueError, "requires"):
            validate_training_config({
                "method": "pinn", "checkpoint": None,
                "epochs": 10, "learning_rate": 1e-4,
            })


if __name__ == "__main__":
    unittest.main()
