import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Matrix3dCliTests(unittest.TestCase):
    def test_cli_writes_five_jobs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "matrix.json"
            completed = subprocess.run([
                sys.executable, str(ROOT / "generate_3d_matrix.py"),
                "--python", sys.executable, "--split", "split.json",
                "--data-dir", "data", "--output-dir", "results",
                "--matrix-output", str(output),
            ], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            matrix = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(matrix["schema_version"], 2)
            self.assertEqual(matrix["paired_by"], ["data_seed", "init_seed"])
            self.assertEqual(len(matrix["jobs"]), 5)


if __name__ == "__main__":
    unittest.main()
