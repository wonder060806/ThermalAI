import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class GenerateSampleEfficiencyCasesTests(unittest.TestCase):
    def test_cli_generates_legacy_domain_without_overlaps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir)
            completed = subprocess.run([
                sys.executable, str(ROOT / "generate_sample_efficiency_cases.py"),
                "--output-dir", str(target), "--num-cases", "12", "--seed", "8",
            ], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            cases = json.loads((target / "cases_params.json").read_text(encoding="utf-8"))
            self.assertEqual(len(cases), 12)
            self.assertTrue(all(case["chip_length"] == 1000 for case in cases))
            self.assertTrue(all(0.5 <= case["total_power_mw"] <= 5.0 for case in cases))
            self.assertEqual(len({case["layout_group"] for case in cases}), 12)
            stack = (target / "stacks" / "case_0.stk").read_text()
            self.assertIn("case_0_temp.txt", stack)
            self.assertLess(stack.index("layer 495 TOY"), stack.index("source 5 TOY"))


if __name__ == "__main__":
    unittest.main()
