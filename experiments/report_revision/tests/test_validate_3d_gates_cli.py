import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]


class Validate3dGatesCliTests(unittest.TestCase):
    def test_cli_accepts_known_zero_and_multilayer_controls(self):
        zero = WORKSPACE / "3d-ice" / "bin" / "multilayer_zero.vtk"
        powered = WORKSPACE / "3d-ice" / "bin" / "multilayer_smoke.vtk"
        if not zero.exists() or not powered.exists():
            self.skipTest("3D control VTKs unavailable")
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "gates.json"
            completed = subprocess.run([
                sys.executable, str(ROOT / "validate_3d_gates.py"),
                "--zero", str(zero), "--medium", str(powered), "--fine", str(powered),
                "--output", str(output),
            ], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["status"], "pass")


if __name__ == "__main__":
    unittest.main()
