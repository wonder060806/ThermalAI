import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


class TrainCliTests(unittest.TestCase):
    def test_validate_only_reports_selection_without_importing_gpu_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            split = root / "split.json"
            split.write_text(json.dumps({"split": {
                "train_subsets": {"2": ["0", "1"]}, "test": ["2"]
            }}), encoding="utf-8")
            output = root / "result.json"
            completed = subprocess.run([
                sys.executable, str(EXPERIMENT_ROOT / "train_revision.py"),
                "--split", str(split), "--train-size", "2",
                "--method", "random", "--epochs", "1",
                "--learning-rate", "0.0001", "--init-seed", "7",
                "--data-dir", str(root), "--output", str(output),
                "--validate-only",
            ], text=True, capture_output=True)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "validated")
            self.assertEqual(result["train_ids"], [0, 1])
            self.assertEqual(result["test_ids"], [2])
            self.assertEqual(result["config"]["method"], "random")
            self.assertEqual(result["config"]["source_scale_per_beta"], 200.0)
            self.assertEqual(result["config"]["surface_flux_scale_per_beta"], 1.0)
            self.assertEqual(len(result["config_hash"]), 64)


if __name__ == "__main__":
    unittest.main()
