import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class GenerateRealisticCasesTests(unittest.TestCase):
    def test_cli_generates_reproducible_realistic_definitions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir)
            completed = subprocess.run([
                sys.executable, str(ROOT / "generate_realistic_cases.py"),
                "--output-dir", str(target), "--num-cases", "4", "--seed", "7",
                "--chip-size-mm", "20", "--power-min-w", "150", "--power-max-w", "500",
            ], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            cases = json.loads((target / "cases_params.json").read_text(encoding="utf-8"))
            self.assertEqual(len(cases), 4)
            self.assertEqual(cases[0]["chip_length"], 20_000)
            self.assertGreaterEqual(cases[0]["total_power_w"], 150.0)
            self.assertLessEqual(cases[0]["total_power_w"], 500.0)
            self.assertTrue((target / "case_0.flp").is_file())
            self.assertIn("discretization 5, 10", (target / "case_0.flp").read_text())
            stack = (target / "stacks" / "case_0.stk").read_text(encoding="utf-8")
            self.assertIn("chip length 20000", stack)
            self.assertIn("cell length 1000", stack)
            self.assertIn("non-uniform true", stack)
            self.assertIn('case_0_temp.txt', stack)
            self.assertIn('case_0.vtk', stack)
            layouts = {json.dumps(case["blocks"], sort_keys=True) for case in cases}
            self.assertEqual(len(layouts), 4)
            self.assertTrue((target / "validation" / "zero.flp").is_file())
            self.assertTrue((target / "validation" / "stacks" / "fine.stk").is_file())
            self.assertIn("discretization 10, 20", (target / "validation" / "fine.flp").read_text())
            self.assertTrue((target / "validation" / "stacks" / "htc_low.stk").is_file())
            self.assertTrue((target / "validation" / "stacks" / "silicon_thick.stk").is_file())
            self.assertIn("5.0e-8", (target / "validation" / "stacks" / "htc_low.stk").read_text())
            self.assertIn("layer 100 SILICON", (target / "validation" / "stacks" / "silicon_thick.stk").read_text())
            manifest = json.loads((target / "generation_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["scope"], "representative_high_htc_liquid_boundary")
            self.assertFalse(manifest["claims_exact_commercial_package_geometry"])
            self.assertAlmostEqual(manifest["derived"]["power_density_range_w_cm2"][0], 37.5)
            self.assertAlmostEqual(manifest["physical_parameters_si"]["top_htc_w_m2_k"], 100000.0)
            self.assertGreaterEqual(len(manifest["evidence"]), 4)
            self.assertEqual(set(manifest["sensitivity_cases"]), {"htc_low", "htc_high", "silicon_thin", "silicon_thick"})


if __name__ == "__main__":
    unittest.main()
