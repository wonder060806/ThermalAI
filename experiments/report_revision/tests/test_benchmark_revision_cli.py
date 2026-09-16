import subprocess
import sys
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]


class BenchmarkRevisionCliTests(unittest.TestCase):
    def test_surface_checkpoint_cpu_smoke(self):
        checkpoint = WORKSPACE / "DeepOHeat" / "DeepOHeat" / "2d_power_map" / "log" / "pinn_pretrain" / "checkpoints" / "model_epoch_2000.pth"
        data = WORKSPACE / "3d-ice" / "data"
        if not checkpoint.exists() or not (data / "case_0_temp.txt").exists():
            self.skipTest("local checkpoint/data unavailable")
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "benchmark.json"
            split = Path(temp_dir) / "split.json"
            split.write_text(json.dumps({"split": {"test": ["0"]}}), encoding="utf-8")
            completed = subprocess.run([
                sys.executable, str(ROOT / "benchmark_revision.py"),
                "--kind", "surface", "--checkpoint", str(checkpoint),
                "--data-dir", str(data), "--case-id", "0", "--device", "cpu",
                "--split", str(split),
                "--warmups", "0", "--repeats", "1", "--load-repeats", "1",
                "--output", str(output),
            ], capture_output=True, text=True, timeout=180)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output.exists())

    def test_rejects_case_not_in_frozen_test_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            split = Path(temp_dir) / "split.json"
            split.write_text(json.dumps({"split": {"test": ["9"]}}), encoding="utf-8")
            completed = subprocess.run([
                sys.executable, str(ROOT / "benchmark_revision.py"),
                "--kind", "surface", "--checkpoint", str(Path(temp_dir) / "missing.pth"),
                "--data-dir", temp_dir, "--case-id", "0", "--device", "cpu",
                "--split", str(split), "--output", str(Path(temp_dir) / "out.json"),
            ], capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("not in the frozen test split", completed.stderr)


if __name__ == "__main__":
    unittest.main()
