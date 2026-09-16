import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


class AuditCliTests(unittest.TestCase):
    def test_solid_mode_reads_generated_generic_case_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "cases_params.json").write_text(
                json.dumps([
                    {"id": 0, "total_power_mw": 10.0},
                    {"id": 1, "total_power_mw": 20.0},
                ]),
                encoding="utf-8",
            )
            (root / "case_0_temp.txt").write_text("300 301\n302 303\n", encoding="utf-8")
            (root / "case_1_temp.txt").write_text("301 302\n303 304\n", encoding="utf-8")
            output = root / "audit.json"

            completed = subprocess.run([
                sys.executable, str(EXPERIMENT_ROOT / "audit_dataset.py"),
                "--data-dir", str(root), "--mode", "solid",
                "--output", str(output), "--require-complete",
            ], capture_output=True, text=True)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["coverage"]["parameter_rows"], 2)
            self.assertEqual(report["coverage"]["temperature_maps"], 2)
            self.assertTrue(report["coverage"]["complete"])

    def test_cli_reads_maps_and_writes_scientifically_labeled_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "cases_microchannel_params.json").write_text(
                json.dumps([
                    {"id": 0, "total_power_mw": 10.0},
                    {"id": 1, "total_power_mw": 20.0},
                    {"id": 2, "total_power_mw": 30.0},
                ]),
                encoding="utf-8",
            )
            for case_id, rows in enumerate((
                "300 301\n302 303\n",
                "301 302\n303 304\n",
                "302 303\n304 305\n",
            )):
                (root / f"case_microchannel_{case_id}_temp.txt").write_text(
                    "% map\n" + rows, encoding="utf-8"
                )
            output = root / "audit.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(EXPERIMENT_ROOT / "audit_dataset.py"),
                    "--data-dir", str(root),
                    "--mode", "microchannel",
                    "--ambient-k", "293",
                    "--output", str(output),
                ],
                text=True,
                capture_output=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["scientific_label"], "equivalent_strong_convection")
            self.assertEqual(report["temperature_summary"]["n_cases"], 3)
            self.assertTrue(report["coverage"]["complete"])
            self.assertEqual(report["numeric_parameter_summary"]["total_power_mw"]["range"], 20.0)
            self.assertIn("total_power_regression", report["leave_one_out_trivial_baselines"]["0"])

    def test_require_complete_rejects_missing_temperature_maps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "cases_microchannel_params.json").write_text(
                json.dumps([{"id": 0, "total_power_mw": 10.0}, {"id": 1, "total_power_mw": 20.0}]),
                encoding="utf-8",
            )
            (root / "case_microchannel_0_temp.txt").write_text("300 301\n302 303\n", encoding="utf-8")
            completed = subprocess.run([
                sys.executable, str(EXPERIMENT_ROOT / "audit_dataset.py"),
                "--data-dir", str(root), "--mode", "microchannel",
                "--output", str(root / "audit.json"), "--require-complete",
            ], capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("incomplete", completed.stderr)


if __name__ == "__main__":
    unittest.main()
