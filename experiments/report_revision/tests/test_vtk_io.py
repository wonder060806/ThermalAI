import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from revision.vtk_io import (
    layer_temperature_summary,
    parse_3dice_vtk,
    validate_multilayer_truth,
    validate_zero_power_truth,
    grid_convergence_summary,
)


class VtkIoTests(unittest.TestCase):
    def test_zero_power_gate_requires_ambient_field(self):
        result = validate_zero_power_truth(np.array([298.14, 298.16]), 298.15, 0.02)
        self.assertAlmostEqual(result["maximum_ambient_deviation_k"], 0.01, places=6)
        with self.assertRaisesRegex(ValueError, "zero-power"):
            validate_zero_power_truth(np.array([298.15, 300.0]), 298.15, 0.1)

    def test_grid_convergence_compares_medium_against_fine(self):
        medium = [{"z_center_um": 5.0, "mean_k": 320.0, "max_k": 330.0}]
        fine = [{"z_center_um": 5.0, "mean_k": 320.2, "max_k": 330.3}]
        result = grid_convergence_summary(medium, fine, max_difference_k=0.5)
        self.assertAlmostEqual(result["maximum_temperature_difference_k"], 0.3)
        with self.assertRaisesRegex(ValueError, "grid convergence"):
            grid_convergence_summary(medium, fine, max_difference_k=0.1)

    def test_parser_returns_hexahedron_centers_and_temperatures(self):
        vtk = """# vtk DataFile Version 3.0
test
ASCII
DATASET UNSTRUCTURED_GRID
POINTS 8 float
0 0 0
2 0 0
2 4 0
0 4 0
0 0 6
2 0 6
2 4 6
0 4 6
CELLS 1 9
8 0 1 2 3 4 5 6 7
CELL_TYPES 1
12
CELL_DATA 1
SCALARS Temperature float 1
LOOKUP_TABLE default
321.5
SCALARS Labels int 1
LOOKUP_TABLE default
0
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "field.vtk"
            path.write_text(vtk, encoding="utf-8")
            result = parse_3dice_vtk(path)
        self.assertTrue(np.allclose(result["centers_um"], [[1.0, 2.0, 3.0]]))
        self.assertTrue(np.allclose(result["temperature_k"], [321.5]))

    def test_layer_summary_groups_cell_centers_by_z(self):
        result = layer_temperature_summary(
            np.array([[0, 0, 1], [1, 0, 1], [0, 0, 3]], dtype=float),
            np.array([300.0, 302.0, 310.0]),
        )
        self.assertEqual(result[0]["z_center_um"], 1.0)
        self.assertEqual(result[0]["n_cells"], 2)
        self.assertEqual(result[0]["mean_k"], 301.0)
        self.assertEqual(result[1]["mean_k"], 310.0)

    def test_multilayer_gate_rejects_repeated_planes_and_extreme_temperature(self):
        repeated = [
            {"z_center_um": 1.0, "mean_k": 310.0, "max_k": 311.0},
            {"z_center_um": 2.0, "mean_k": 310.0, "max_k": 311.0},
            {"z_center_um": 3.0, "mean_k": 310.0, "max_k": 311.0},
        ]
        with self.assertRaisesRegex(ValueError, "layer-to-layer"):
            validate_multilayer_truth(repeated, min_layers=3, min_mean_span_k=0.1, max_k=500)
        extreme = [
            {"z_center_um": 1.0, "mean_k": 310.0, "max_k": 700.0},
            {"z_center_um": 2.0, "mean_k": 320.0, "max_k": 700.0},
            {"z_center_um": 3.0, "mean_k": 330.0, "max_k": 700.0},
        ]
        with self.assertRaisesRegex(ValueError, "maximum temperature"):
            validate_multilayer_truth(extreme, min_layers=3, min_mean_span_k=0.1, max_k=500)

    def test_multilayer_gate_accepts_distinct_physical_layers(self):
        layers = [
            {"z_center_um": 2.5, "mean_k": 324.7, "max_k": 333.6},
            {"z_center_um": 30.0, "mean_k": 324.5, "max_k": 333.3},
            {"z_center_um": 105.0, "mean_k": 315.0, "max_k": 320.7},
        ]
        result = validate_multilayer_truth(
            layers, min_layers=3, min_mean_span_k=0.1, max_k=500
        )
        self.assertAlmostEqual(result["layer_mean_span_k"], 9.7)


if __name__ == "__main__":
    unittest.main()
