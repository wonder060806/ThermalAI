import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from revision.preflight import validate_3d_dataset_files, validate_dataset_files, validate_matrix_paths


class PreflightTests(unittest.TestCase):
    def test_3d_dataset_gate_accepts_real_multilayer_fixture(self):
        workspace = ROOT.parents[2]
        vtk = workspace / "3d-ice" / "bin" / "multilayer_smoke.vtk"
        floorplan = workspace / "3d-ice" / "bin" / "report_revision_realistic.flp"
        if not vtk.exists() or not floorplan.exists():
            self.skipTest("multilayer fixture unavailable")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "cases_params.json").write_text(json.dumps([{"id": 0}]), encoding="utf-8")
            shutil.copyfile(vtk, root / "case_0.vtk")
            shutil.copyfile(floorplan, root / "case_0.flp")
            result = validate_3d_dataset_files(root, minimum_cases=1)
            self.assertEqual(result["case_count"], 1)
            self.assertGreater(result["minimum_layer_mean_span_k"], 0.1)

    def test_dataset_requires_params_and_every_case_pair(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data = Path(temp_dir)
            (data / "cases_params.json").write_text(
                json.dumps([{"id": 0}, {"id": 1}]), encoding="utf-8"
            )
            for case_id in (0, 1):
                (data / f"case_{case_id}.flp").write_text("x", encoding="utf-8")
                (data / f"case_{case_id}_temp.txt").write_text("300 301\n302 303\n", encoding="utf-8")
            result = validate_dataset_files(data, minimum_cases=2)
            self.assertEqual(result["case_count"], 2)
            (data / "case_1_temp.txt").unlink()
            with self.assertRaisesRegex(ValueError, "case_1_temp"):
                validate_dataset_files(data, minimum_cases=2)

    def test_3d_preflight_can_numeric_check_nonuniform_tmap_without_claiming_spatial_map(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data = Path(temp_dir)
            (data / "cases_params.json").write_text(json.dumps([{"id": 0}]), encoding="utf-8")
            (data / "case_0.flp").write_text("x", encoding="utf-8")
            (data / "case_0_temp.txt").write_text(
                '% Thermal map (please find axis information in "xyaxis_CHIP.txt")\n300 301 302 303\n',
                encoding="utf-8",
            )
            result = validate_dataset_files(data, minimum_cases=1,
                                            allow_nonuniform_tmap=True)
            self.assertFalse(result["spatial_temperature_maps_validated"])
            self.assertEqual(result["nonuniform_temperature_map_count"], 1)

    def test_matrix_rejects_missing_input_but_allows_future_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            existing = root / "split.json"
            existing.write_text("{}", encoding="utf-8")
            train_script = root / "train.py"
            train_script.write_text("", encoding="utf-8")
            matrix = {"jobs": [{
                "job_id": "one",
                "command": [sys.executable, str(train_script), "--split", str(existing),
                            "--output", str(root / "future" / "one.json")],
            }]}
            result = validate_matrix_paths(matrix)
            self.assertEqual(result["job_count"], 1)
            matrix["jobs"][0]["command"][3] = str(root / "missing.json")
            with self.assertRaisesRegex(ValueError, "missing input"):
                validate_matrix_paths(matrix)

    def test_matrix_rejects_duplicate_output_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script = root / "train.py"
            script.write_text("", encoding="utf-8")
            output = root / "same.json"
            matrix = {"jobs": [
                {"job_id": "one", "command": [sys.executable, str(script), "--output", str(output)]},
                {"job_id": "two", "command": [sys.executable, str(script), "--output", str(output)]},
            ]}
            with self.assertRaisesRegex(ValueError, "duplicate output"):
                validate_matrix_paths(matrix)


if __name__ == "__main__":
    unittest.main()
