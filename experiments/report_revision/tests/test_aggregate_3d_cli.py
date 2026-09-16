import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Aggregate3dCliTests(unittest.TestCase):
    def test_cli_aggregates_five_seed_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            for seed in range(5):
                (folder / f"seed_{seed}.json").write_text(json.dumps({
                    "status": "completed", "config": {"init_seed": seed},
                    "test": {"aggregate": {"mae_k": 1.0 + seed,
                        "hotspot_location_error_um": 100.0 + seed}},
                }), encoding="utf-8")
            output = folder / "summary.json"
            completed = subprocess.run([
                sys.executable, str(ROOT / "aggregate_3d_results.py"),
                "--input-dir", str(folder), "--output", str(output),
            ], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["metrics"]["mae_k"]["mean"], 3.0)


if __name__ == "__main__":
    unittest.main()
