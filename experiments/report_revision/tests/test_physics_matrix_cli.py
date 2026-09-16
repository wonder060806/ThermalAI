import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PhysicsMatrixCliTests(unittest.TestCase):
    def test_cli_writes_one_hundred_ten_jobs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "matrix.json"
            completed = subprocess.run([
                sys.executable, str(ROOT / "generate_physics_matrix.py"),
                "--python", sys.executable, "--train-script", str(ROOT / "train_revision.py"),
                "--split", "split.json", "--data-dir", "data",
                "--checkpoint", "pinn.pth", "--output-dir", "results",
                "--matrix-output", str(output),
            ], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            matrix = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(matrix["jobs"]), 110)
            self.assertEqual(matrix["paired_by"], ["physics_variant", "physics_weight", "data_seed", "init_seed"])


if __name__ == "__main__":
    unittest.main()
