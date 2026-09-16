import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class AggregateCliTests(unittest.TestCase):
    def test_cli_discovers_result_json_and_writes_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for method, mae in (("pinn", 1.0), ("random", 2.0)):
                (root / f"{method}.json").write_text(json.dumps({
                    "status": "completed",
                    "config": {"method": method, "train_size": 10, "init_seed": 0},
                    "test": {"aggregate": {"mae_k_mean": mae}},
                }), encoding="utf-8")
            output = root / "summary.json"
            completed = subprocess.run([
                sys.executable, str(ROOT / "aggregate_results.py"),
                "--input-dir", str(root), "--train-sizes", "10",
                "--seeds", "0", "--output", str(output),
            ], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["status"], "complete")


if __name__ == "__main__":
    unittest.main()
