import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PrepareMicrochannelTruthTests(unittest.TestCase):
    def test_cli_rebuilds_inputs_from_existing_parameters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            params = root / "params.json"
            params.write_text(json.dumps([{
                "id": 7, "total_power_mw": 20.0,
                "blocks": [{"x": 0, "y": 0, "w": 500, "h": 1000, "power_mw": 20.0}],
            }]), encoding="utf-8")
            output = root / "prepared"
            completed = subprocess.run([
                sys.executable, str(ROOT / "prepare_microchannel_truth.py"),
                "--params", str(params), "--output-dir", str(output),
            ], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("0.020000000000", (output / "case_microchannel_7.flp").read_text())
            self.assertIn("case_microchannel_7_temp.txt", (output / "stacks" / "case_microchannel_7.stk").read_text())
            manifest = json.loads((output / "generation_manifest.json").read_text())
            self.assertEqual(manifest["scientific_label"], "equivalent_strong_convection")


if __name__ == "__main__":
    unittest.main()
