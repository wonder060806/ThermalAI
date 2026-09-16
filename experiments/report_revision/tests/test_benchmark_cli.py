import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = EXPERIMENT_ROOT.parents[2]


class BenchmarkCliTests(unittest.TestCase):
    def test_cpu_model_benchmark_writes_cold_and_warm_timings(self):
        model_dir = WORKSPACE_ROOT / "ThermalAI"
        if not (model_dir / "model_final.pth").exists():
            self.skipTest("deployment model unavailable")
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "benchmark.json"
            completed = subprocess.run([
                sys.executable, str(EXPERIMENT_ROOT / "benchmark.py"),
                "--model-dir", str(model_dir), "--device", "cpu",
                "--warmups", "1", "--repeats", "2", "--load-repeats", "1",
                "--output", str(output),
            ], text=True, capture_output=True, timeout=180)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["device_requested"], "cpu")
            self.assertEqual(result["model_load"]["summary"]["count"], 1)
            self.assertEqual(result["warm_inference"]["summary"]["count"], 2)
            self.assertIsNone(result["speedup_vs_external_median"])


if __name__ == "__main__":
    unittest.main()
