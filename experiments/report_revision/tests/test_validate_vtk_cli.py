import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


class ValidateVtkCliTests(unittest.TestCase):
    def test_cli_writes_layer_summary_and_gate_result(self):
        vtk = """# vtk DataFile Version 3.0
test
ASCII
DATASET UNSTRUCTURED_GRID
POINTS 16 float
0 0 0
1 0 0
1 1 0
0 1 0
0 0 1
1 0 1
1 1 1
0 1 1
0 0 2
1 0 2
1 1 2
0 1 2
0 0 3
1 0 3
1 1 3
0 1 3
CELLS 3 27
8 0 1 2 3 4 5 6 7
8 4 5 6 7 8 9 10 11
8 8 9 10 11 12 13 14 15
CELL_TYPES 3
12
12
12
CELL_DATA 3
SCALARS Temperature float 1
LOOKUP_TABLE default
310
315
325
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "truth.vtk"
            output = root / "validation.json"
            source.write_text(vtk, encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(EXPERIMENT_ROOT / "validate_vtk.py"),
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                    "--min-layers",
                    "3",
                    "--min-mean-span-k",
                    "1",
                    "--max-k",
                    "500",
                ],
                cwd=EXPERIMENT_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["gate"]["n_layers"], 3)
        self.assertEqual(len(report["layers"]), 3)
        self.assertEqual(report["source_vtk"], str(source.resolve()))


if __name__ == "__main__":
    unittest.main()
