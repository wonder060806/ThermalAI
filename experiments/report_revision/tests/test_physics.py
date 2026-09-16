import sys
import unittest
from pathlib import Path

import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from revision.physics import (
    dimensionless_volumetric_source,
    dimensionless_surface_flux,
    gradient_diagnostics,
    normalized_weighted_loss,
    validate_physics_config,
)


class PhysicsTests(unittest.TestCase):
    def test_legacy_beta_unit_surface_flux_scale_is_one(self):
        value = dimensionless_surface_flux(
            cell_power_w=6.25e-6,
            cell_area_m2=(1e-3 / 20) ** 2,
            length_scale_m=1e-3,
            conductivity_w_mk=0.1,
            temperature_scale_k=25.0,
        )
        self.assertAlmostEqual(value, 1.0)

    def test_gradient_diagnostics_detects_orthogonal_losses(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        data_loss = (parameter[0] + parameter[1]) ** 2
        physics_loss = (parameter[0] - parameter[1]) ** 2
        result = gradient_diagnostics(data_loss, physics_loss, [parameter])
        self.assertAlmostEqual(result["data_gradient_norm"], 6 * 2**0.5, places=5)
        self.assertAlmostEqual(result["physics_gradient_norm"], 2 * 2**0.5, places=5)
        self.assertAlmostEqual(result["gradient_cosine_similarity"], 0.0, places=5)

    def test_dimensionless_source_matches_heat_equation_scaling(self):
        # 1 W in a 1 mm² cell and 0.1 mm source layer gives q'''=1e10 W/m³.
        # q* = q''' L² / (k ΔT) = 1e10*1e-6/(100*10) = 10.
        result = dimensionless_volumetric_source(
            cell_power_w=1.0,
            cell_area_m2=1e-6,
            source_thickness_m=1e-4,
            length_scale_m=1e-3,
            conductivity_w_mk=100.0,
            temperature_scale_k=10.0,
        )
        self.assertAlmostEqual(result, 10.0)

    def test_legacy_beta_unit_volumetric_source_scale_is_200(self):
        result = dimensionless_volumetric_source(
            cell_power_w=0.00625e-3,
            cell_area_m2=(1e-3 / 20) ** 2,
            source_thickness_m=5e-6,
            length_scale_m=1e-3,
            conductivity_w_mk=0.1,
            temperature_scale_k=25.0,
        )
        self.assertAlmostEqual(result, 200.0)

    def test_normalized_loss_uses_reference_scales_not_raw_magnitudes(self):
        result = normalized_weighted_loss(
            losses={"data": 4.0, "pde": 100.0},
            weights={"data": 0.75, "pde": 0.25},
            reference_scales={"data": 2.0, "pde": 50.0},
        )
        self.assertAlmostEqual(result, 2.0)

    def test_physics_config_rejects_weights_that_do_not_sum_to_one(self):
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            validate_physics_config({
                "variant": "volumetric_source",
                "weights": {"data": 0.8, "pde": 0.4},
            })

    def test_source_variant_requires_physical_scales(self):
        with self.assertRaisesRegex(ValueError, "source_thickness_m"):
            validate_physics_config({
                "variant": "volumetric_source",
                "weights": {"data": 0.5, "pde": 0.5},
                "length_scale_m": 1e-3,
            })


if __name__ == "__main__":
    unittest.main()
