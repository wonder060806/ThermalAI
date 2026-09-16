import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class AggregatePhysicsCliTests(unittest.TestCase):
    def test_cli_reports_complete_full_matrix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            index = 0
            for seed in range(10):
                settings = [("none", 0.0)] + [
                    (variant, weight)
                    for variant in ("surface_flux", "volumetric_source")
                    for weight in (0.01, 0.05, 0.1, 0.25, 0.5)
                ]
                for variant, weight in settings:
                    (folder / f"run_{index}.json").write_text(json.dumps({
                        "status": "completed",
                        "config": {"init_seed": seed, "physics_variant": variant,
                                   "physics_weight": weight},
                        "test": {"aggregate": {"mae_k_mean": 2.0 + weight}},
                    }), encoding="utf-8")
                    index += 1
            output = folder / "summary.json"
            completed = subprocess.run([
                sys.executable, str(ROOT / "aggregate_physics_results.py"),
                "--input-dir", str(folder), "--output", str(output),
            ], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["status"], "complete")


if __name__ == "__main__":
    unittest.main()
