import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from revision.physics_training import (
    _source_beta,
    physics_loss_components,
    sample_physics_coordinates,
)


class QuadraticModel(torch.nn.Module):
    def forward(self, coords, beta):
        return (coords**2).sum(dim=1, keepdim=True)


class LinearZModel(torch.nn.Module):
    def forward(self, coords, beta):
        return 2.5 * coords[:, 2:3]


class PhysicsTrainingTests(unittest.TestCase):
    def test_prescribed_power_sampling_does_not_join_coordinate_gradient_graph(self):
        power = torch.tensor([[0.0, 1.0], [1.0, 2.0]])
        coords = torch.tensor([[0.5, 0.5, 0.5]], requires_grad=True)

        sampled = _source_beta(power, coords)

        self.assertAlmostEqual(sampled.item(), 1.0, places=6)
        self.assertFalse(sampled.requires_grad)

    def test_sampling_is_seeded_and_marks_source_layer(self):
        first = sample_physics_coordinates(10, 4, seed=3, source_top_z=0.5, source_thickness_z=0.005)
        second = sample_physics_coordinates(10, 4, seed=3, source_top_z=0.5, source_thickness_z=0.005)
        self.assertTrue(torch.equal(first["bulk"], second["bulk"]))
        self.assertTrue(torch.all(first["source"][:, 2] >= 0.495))
        self.assertTrue(torch.all(first["source"][:, 2] <= 0.5))
        self.assertTrue(torch.all(first["front"][:, 1] == 0.0))
        self.assertTrue(torch.all(first["back"][:, 1] == 1.0))

    def test_quadratic_field_has_expected_source_free_pde_mse(self):
        coords = {"bulk": torch.tensor([[0.2, 0.3, 0.4], [0.7, 0.1, 0.2]])}
        beta_map = torch.ones((3, 3))
        losses = physics_loss_components(
            QuadraticModel(), beta_map, coords, variant="source_free",
            source_scale_per_beta=0.0, biot=5.0, ambient_u=0.2,
        )
        self.assertAlmostEqual(losses["pde"].item(), 36.0, places=4)

    def test_volumetric_source_changes_source_residual(self):
        coords = {"source": torch.tensor([[0.5, 0.5, 0.499]])}
        losses = physics_loss_components(
            QuadraticModel(), torch.ones((3, 3)), coords,
            variant="volumetric_source", source_scale_per_beta=2.0,
            biot=5.0, ambient_u=0.2,
        )
        self.assertAlmostEqual(losses["source_pde"].item(), 64.0, places=4)

    def test_surface_flux_variant_matches_deepoheat_top_power_boundary(self):
        coords = {
            "source": torch.tensor([[0.5, 0.5, 0.499]]),
            "top": torch.tensor([[0.5, 0.5, 0.5]]),
        }
        losses = physics_loss_components(
            QuadraticModel(), torch.ones((3, 3)), coords,
            variant="surface_flux", source_scale_per_beta=220.5,
            biot=5.0, ambient_u=0.2, surface_flux_scale_per_beta=1.0,
        )
        self.assertAlmostEqual(losses["source_pde"].item(), 36.0, places=4)
        self.assertAlmostEqual(losses["top_surface_flux"].item(), 0.0, places=6)

    def test_surface_flux_bilinearly_samples_nodal_power_map(self):
        losses = physics_loss_components(
            QuadraticModel(), torch.tensor([[0.0, 1.0], [1.0, 2.0]]),
            {"top": torch.tensor([[0.5, 0.5, 0.5]])},
            variant="surface_flux", source_scale_per_beta=200.0,
            biot=5.0, ambient_u=0.2, surface_flux_scale_per_beta=1.0,
        )
        self.assertAlmostEqual(losses["top_surface_flux"].item(), 0.0, places=6)

    def test_source_free_variant_enforces_zero_source_in_source_layer(self):
        coords = {"source": torch.tensor([[0.5, 0.5, 0.499]])}
        losses = physics_loss_components(
            QuadraticModel(), torch.ones((3, 3)), coords,
            variant="source_free", source_scale_per_beta=2.0,
            biot=5.0, ambient_u=0.2,
        )
        self.assertAlmostEqual(losses["source_pde"].item(), 36.0, places=4)

    def test_front_and_back_are_adiabatic(self):
        coords = {
            "front": torch.tensor([[0.2, 0.0, 0.3]]),
            "back": torch.tensor([[0.2, 1.0, 0.3]]),
        }
        losses = physics_loss_components(
            QuadraticModel(), torch.ones((3, 3)), coords,
            variant="source_free", source_scale_per_beta=0.0,
            biot=5.0, ambient_u=0.2,
        )
        self.assertIn("front_adiabatic", losses)
        self.assertIn("back_adiabatic", losses)


if __name__ == "__main__":
    unittest.main()
