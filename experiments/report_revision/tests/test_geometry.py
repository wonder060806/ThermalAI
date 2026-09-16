import sys
import unittest
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from revision.geometry import (
    build_heterogeneous_floorplan,
    render_3dice_floorplan,
    validate_floorplan,
)


class GeometryTests(unittest.TestCase):
    def test_realistic_floorplan_covers_die_and_conserves_power(self):
        blocks = build_heterogeneous_floorplan(
            chip_width_um=10_000,
            chip_height_um=10_000,
            total_power_w=75.0,
        )
        result = validate_floorplan(blocks, 10_000, 10_000, expected_power_w=75.0)
        self.assertAlmostEqual(result["covered_area_um2"], 100_000_000.0)
        self.assertAlmostEqual(result["total_power_w"], 75.0)
        self.assertGreaterEqual(result["n_blocks"], 8)
        self.assertGreater(result["max_power_density_w_per_mm2"], result["min_power_density_w_per_mm2"])
        rendered = render_3dice_floorplan(blocks, discretization=(5, 10))
        self.assertEqual(rendered.count("power values"), 8)
        self.assertIn("compute_0 :", rendered)
        self.assertIn("discretization 5, 10 ;", rendered)

    def test_floorplan_rejects_overlap_and_out_of_bounds(self):
        overlap = [
            {"name": "a", "x_um": 0, "y_um": 0, "width_um": 6000, "height_um": 10000, "power_w": 10},
            {"name": "b", "x_um": 5000, "y_um": 0, "width_um": 5000, "height_um": 10000, "power_w": 10},
        ]
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_floorplan(overlap, 10_000, 10_000, expected_power_w=20)
        outside = [
            {"name": "a", "x_um": 0, "y_um": 0, "width_um": 11000, "height_um": 10000, "power_w": 10}
        ]
        with self.assertRaisesRegex(ValueError, "bounds"):
            validate_floorplan(outside, 10_000, 10_000, expected_power_w=10)


if __name__ == "__main__":
    unittest.main()
