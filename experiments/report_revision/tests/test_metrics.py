import math
import sys
import unittest
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from revision.metrics import thermal_metrics, thermal_metrics_3d


class ThermalMetricsTests(unittest.TestCase):
    def test_3d_metrics_report_internal_hotspot_distance_and_layer_errors(self):
        coords = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 2]], dtype=float)
        true = np.array([300.0, 310.0, 305.0])
        pred = np.array([300.0, 304.0, 311.0])
        result = thermal_metrics_3d(true, pred, coords, ambient_k=295.0)
        self.assertAlmostEqual(result["mae_k"], 4.0)
        self.assertAlmostEqual(result["hotspot_location_error_um"], 5**0.5)
        self.assertEqual(result["true_hotspot_z_um"], 0.0)
        self.assertEqual(result["predicted_hotspot_z_um"], 2.0)
        self.assertAlmostEqual(result["layer_mae_k_mean"], 4.5)
        self.assertAlmostEqual(result["layer_mae_k_max"], 6.0)

    def test_metrics_use_temperature_rise_and_report_hotspot_location(self):
        true = np.array([[300.0, 310.0], [305.0, 320.0]])
        pred = np.array([[302.0, 307.0], [306.0, 316.0]])

        result = thermal_metrics(true, pred, ambient_k=295.0, spacing=(2.0, 3.0))

        self.assertAlmostEqual(result["mae_k"], 2.5)
        self.assertAlmostEqual(result["rmse_k"], math.sqrt(7.5))
        self.assertAlmostEqual(result["max_abs_error_k"], 4.0)
        self.assertAlmostEqual(result["hotspot_temperature_error_k"], 4.0)
        self.assertAlmostEqual(result["hotspot_location_error"], 0.0)
        self.assertAlmostEqual(result["relative_temperature_rise_error"], 2.5 / 13.75)
        self.assertAlmostEqual(result["relative_hotspot_temperature_rise_error"], 4.0 / 25.0)
        self.assertAlmostEqual(result["nmae_by_range"], 2.5 / 20.0)
        self.assertAlmostEqual(result["absolute_temperature_mape"],
                               np.mean(np.array([2/300, 3/310, 1/305, 4/320])))

    def test_hotspot_location_uses_physical_spacing(self):
        true = np.array([[10.0, 20.0], [30.0, 40.0]])
        pred = np.array([[50.0, 20.0], [30.0, 35.0]])

        result = thermal_metrics(true, pred, ambient_k=0.0, spacing=(2.0, 3.0))

        self.assertAlmostEqual(result["hotspot_location_error"], math.sqrt(13.0))

    def test_zero_temperature_range_returns_null_nmae(self):
        result = thermal_metrics(
            np.full((2, 2), 300.0),
            np.full((2, 2), 301.0),
            ambient_k=295.0,
        )

        self.assertIsNone(result["nmae_by_range"])

    def test_invalid_shapes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            thermal_metrics(np.zeros((2, 2)), np.zeros((4,)), ambient_k=0.0)


if __name__ == "__main__":
    unittest.main()
