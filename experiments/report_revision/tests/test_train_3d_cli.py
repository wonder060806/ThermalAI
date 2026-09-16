import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parents[2]


class Train3dCliTests(unittest.TestCase):
    def test_one_epoch_real_multilayer_vtk_cpu_smoke(self):
        vtk = WORKSPACE_ROOT / "3d-ice" / "bin" / "multilayer_smoke.vtk"
        floorplan = WORKSPACE_ROOT / "3d-ice" / "bin" / "report_revision_realistic.flp"
        if not vtk.exists() or not floorplan.exists():
            self.skipTest("multilayer smoke truth unavailable")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for case_id in (0, 1):
                shutil.copyfile(vtk, root / f"case_{case_id}.vtk")
                shutil.copyfile(floorplan, root / f"case_{case_id}.flp")
            (root / "cases_params.json").write_text(json.dumps([
                {"id": 0, "chip_length": 10000}, {"id": 1, "chip_length": 10000}
            ]), encoding="utf-8")
            split = root / "split.json"
            split.write_text(json.dumps({"split": {
                "train_subsets": {"1": ["0"]}, "test": ["1"]
            }}), encoding="utf-8")
            output = root / "result.json"
            completed = subprocess.run([
                sys.executable, str(ROOT / "train_3d_revision.py"),
                "--split", str(split), "--train-size", "1", "--data-dir", str(root),
                "--epochs", "1", "--learning-rate", "0.0001", "--init-seed", "2",
                "--output", str(output), "--device", "cpu",
            ], capture_output=True, text=True, timeout=180)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "completed")
            self.assertIn("hotspot_location_error_um", result["test"]["aggregate"])

    def test_validate_only_checks_selection_without_loading_vtk(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            split = root / "split.json"
            split.write_text(json.dumps({"split": {
                "train_subsets": {"2": ["0", "1"]}, "test": ["2"]
            }}), encoding="utf-8")
            output = root / "result.json"
            completed = subprocess.run([
                sys.executable, str(ROOT / "train_3d_revision.py"),
                "--split", str(split), "--train-size", "2",
                "--data-dir", str(root / "not-yet-generated"),
                "--epochs", "1", "--learning-rate", "0.001",
                "--init-seed", "4", "--output", str(output),
                "--validate-only",
            ], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "validated")
            self.assertEqual(result["train_ids"], [0, 1])
            self.assertTrue(result["config"]["deterministic_algorithms"])
            self.assertEqual(result["config"]["study"], "representative_high_htc_full_volume_3d")


if __name__ == "__main__":
    unittest.main()
