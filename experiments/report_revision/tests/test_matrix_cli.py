import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


class MatrixCliTests(unittest.TestCase):
    def test_cli_generates_seventy_job_formal_matrix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "matrix.json"
            completed = subprocess.run([
                sys.executable, str(EXPERIMENT_ROOT / "generate_small_sample_matrix.py"),
                "--python", "python", "--split", "split.json",
                "--data-dir", "data", "--checkpoint", "pinn.pth",
                "--output-dir", "results", "--matrix-output", str(output),
            ], text=True, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            matrix = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(matrix["schema_version"], 2)
            self.assertEqual(matrix["paired_by"], ["train_size", "data_seed", "init_seed"])
            self.assertEqual(len(matrix["jobs"]), 70)


if __name__ == "__main__":
    unittest.main()
