import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from revision.dataset import load_3d_case, load_solid_case, parse_temperature_map


class DatasetTests(unittest.TestCase):
    def test_3d_loader_returns_normalized_cell_centers_and_temperature(self):
        vtk = """# vtk DataFile Version 3.0
x
ASCII
DATASET UNSTRUCTURED_GRID
POINTS 8 float
0 0 0
1000 0 0
1000 1000 0
0 1000 0
0 0 100
1000 0 100
1000 1000 100
0 1000 100
CELLS 1 9
8 0 1 2 3 4 5 6 7
CELL_TYPES 1
12
CELL_DATA 1
SCALARS Temperature float 1
LOOKUP_TABLE default
310
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "case_0.vtk").write_text(vtk, encoding="utf-8")
            (root / "case_0.flp").write_text(
                "hot:\n position 0, 0;\n dimension 1000, 1000;\n power values 0.1;\n",
                encoding="utf-8",
            )
            case = load_3d_case(root, 0, chip_um=1000, model_grid_size=3,
                                power_per_unit_mw=10, temperature_ref_k=300,
                                temperature_scale_k=10)
        self.assertTrue(np.allclose(case["coords"], [[0.5, 0.5, 0.05]]))
        self.assertTrue(np.allclose(case["temperature_u"], [1.0]))
        self.assertAlmostEqual(case["floorplan_total_power_mw"], 100.0)
        self.assertTrue(np.allclose(case["power_dimless"], 2.5))

    def test_temperature_parser_accepts_3d_ice_comment_header(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "map.txt"
            path.write_text("% header\n300 301\n302 303\n", encoding="utf-8")
            field = parse_temperature_map(path)
            self.assertTrue(np.array_equal(field, [[300.0, 301.0], [302.0, 303.0]]))

    def test_temperature_parser_restores_single_line_square_map(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "map.txt"
            path.write_text("% flattened uniform output\n300 301 302 303\n", encoding="utf-8")
            field = parse_temperature_map(path)
            self.assertEqual(field.shape, (2, 2))

    def test_temperature_parser_rejects_nonuniform_map_without_axis_coordinates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "map.txt"
            path.write_text(
                '% Thermal map (please find axis information in "xyaxis_CHIP.txt")\n'
                "300 301 302 303\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "non-uniform.*axis"):
                parse_temperature_map(path)

    def test_case_loader_conserves_total_floorplan_power(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "case_0_temp.txt").write_text(
                "% map\n300 301\n302 303\n", encoding="utf-8"
            )
            (root / "case_0.flp").write_text(
                "hotspot:\n"
                " position 0, 0;\n"
                " dimension 500, 1000;\n"
                " power values 0.002;\n",
                encoding="utf-8",
            )
            case = load_solid_case(
                root, case_id=0, chip_um=1000.0, model_grid_size=3,
                power_per_unit_mw=0.01, temperature_ref_k=293.0,
                temperature_scale_k=10.0,
            )
            self.assertEqual(case["temperature_k"].shape, (3, 3))
            self.assertAlmostEqual(case["floorplan_total_power_mw"], 2.0)
            expected = np.array([[100.0, 100.0, 100.0],
                                 [50.0, 50.0, 50.0],
                                 [0.0, 0.0, 0.0]])
            self.assertTrue(np.allclose(case["power_dimless"], expected))
            self.assertTrue(np.allclose(
                case["temperature_u"], (case["temperature_k"] - 293.0) / 10.0
            ))


if __name__ == "__main__":
    unittest.main()
