import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = EXPERIMENT_ROOT.parents[2]


class TrainSmokeTests(unittest.TestCase):
    def test_one_case_one_epoch_physics_training_logs_diagnostics(self):
        data_dir = WORKSPACE_ROOT / "3d-ice" / "data"
        checkpoint = (
            WORKSPACE_ROOT / "DeepOHeat" / "DeepOHeat" / "2d_power_map" /
            "log" / "pinn_pretrain" / "checkpoints" / "model_epoch_2000.pth"
        )
        if not (data_dir / "case_0_temp.txt").exists() or not checkpoint.exists():
            self.skipTest("local data or PINN checkpoint is unavailable")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            split = root / "split.json"
            split.write_text(json.dumps({"split": {
                "train_subsets": {"1": ["0"]}, "test": ["1"]
            }}), encoding="utf-8")
            output = root / "physics.json"
            completed = subprocess.run([
                sys.executable, str(EXPERIMENT_ROOT / "train_revision.py"),
                "--split", str(split), "--train-size", "1", "--method", "pinn",
                "--checkpoint", str(checkpoint), "--epochs", "1",
                "--learning-rate", "0.0001", "--init-seed", "3",
                "--data-dir", str(data_dir), "--output", str(output), "--device", "cpu",
                "--physics-variant", "source_free", "--physics-weight", "0.1",
                "--physics-points", "2", "--boundary-points", "2",
            ], text=True, capture_output=True, timeout=180)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(result["history"]["physics"]), 1)
            epoch = result["history"]["physics"][0]
            self.assertIn("gradient_cosine_similarity", epoch["gradient_diagnostics_mean"])
            self.assertIn("gradient_conflict_fraction", epoch)
            self.assertEqual(epoch["case_count"], 1)

    def test_one_case_one_epoch_random_cpu_training_writes_metrics(self):
        data_dir = WORKSPACE_ROOT / "3d-ice" / "data"
        if not (data_dir / "case_0_temp.txt").exists():
            self.skipTest("local 3D-ICE smoke data is unavailable")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            split = root / "split.json"
            split.write_text(json.dumps({"split": {
                "train_subsets": {"1": ["0"]}, "test": ["1"]
            }}), encoding="utf-8")
            output = root / "result.json"
            completed = subprocess.run([
                sys.executable, str(EXPERIMENT_ROOT / "train_revision.py"),
                "--split", str(split), "--train-size", "1",
                "--method", "random", "--epochs", "1",
                "--learning-rate", "0.0001", "--init-seed", "7",
                "--data-dir", str(data_dir), "--output", str(output),
                "--device", "cpu",
            ], text=True, capture_output=True, timeout=180)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["config"]["method"], "random")
            self.assertEqual(len(result["history"]["train_mse"]), 1)
            self.assertIn("relative_temperature_rise_error", result["test"]["aggregate"])
            self.assertTrue(Path(result["checkpoint_path"]).exists())

    def test_one_case_one_epoch_pinn_cpu_training_loads_checkpoint(self):
        data_dir = WORKSPACE_ROOT / "3d-ice" / "data"
        checkpoint = (
            WORKSPACE_ROOT / "DeepOHeat" / "DeepOHeat" / "2d_power_map" /
            "log" / "pinn_pretrain" / "checkpoints" / "model_epoch_2000.pth"
        )
        if not (data_dir / "case_0_temp.txt").exists() or not checkpoint.exists():
            self.skipTest("local data or PINN checkpoint is unavailable")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            split = root / "split.json"
            split.write_text(json.dumps({"split": {
                "train_subsets": {"1": ["0"]}, "test": ["1"]
            }}), encoding="utf-8")
            output = root / "result.json"
            completed = subprocess.run([
                sys.executable, str(EXPERIMENT_ROOT / "train_revision.py"),
                "--split", str(split), "--train-size", "1",
                "--method", "pinn", "--checkpoint", str(checkpoint),
                "--epochs", "1", "--learning-rate", "0.0001",
                "--init-seed", "7", "--data-dir", str(data_dir),
                "--output", str(output), "--device", "cpu",
            ], text=True, capture_output=True, timeout=180)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["config"]["method"], "pinn")
            self.assertEqual(result["train_ids"], [0])


if __name__ == "__main__":
    unittest.main()
