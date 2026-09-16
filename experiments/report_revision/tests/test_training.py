import sys
import unittest
from pathlib import Path

import numpy as np
import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from revision.training import (
    aggregate_physics_steps,
    calibrate_physics_reference_scales,
    evaluate_3d_cases,
    evaluate_cases,
    mixed_physics_step,
    physics_raw_losses,
    supervised_3d_epoch,
    supervised_epoch,
)


class ScalarFieldModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, coords, beta):
        return self.bias.expand(coords.shape[0], 1)


class QuadraticScaleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, coords, beta):
        return self.scale * (coords**2).sum(dim=1, keepdim=True)


class TrainingTests(unittest.TestCase):
    def test_reference_scale_calibration_uses_all_cases_not_first(self):
        scales = calibrate_physics_reference_scales([
            {"data_loss_raw": 1.0, "physics_loss_raw": 9.0},
            {"data_loss_raw": 5.0, "physics_loss_raw": 3.0},
        ])
        reverse = calibrate_physics_reference_scales(list(reversed([
            {"data_loss_raw": 1.0, "physics_loss_raw": 9.0},
            {"data_loss_raw": 5.0, "physics_loss_raw": 3.0},
        ])))
        self.assertEqual(scales, {"data": 3.0, "physics": 6.0})
        self.assertEqual(scales, reverse)

    def test_raw_physics_measurement_does_not_update_model(self):
        model = QuadraticScaleModel()
        top = torch.tensor([[0.0, 0.0, 0.5], [1.0, 0.0, 0.5]])
        sample = {"power_dimless": np.ones((2, 2)),
                  "temperature_u": np.array([[0.25, 1.25]])}
        before = model.scale.detach().clone()
        result = physics_raw_losses(
            model, sample, top, {"bulk": torch.tensor([[0.2, 0.3, 0.4]])},
            "source_free", 0.0, 5.0, 0.2,
        )
        self.assertAlmostEqual(result["data_loss_raw"], 0.0, places=6)
        self.assertAlmostEqual(result["physics_loss_raw"], 36.0, places=4)
        self.assertTrue(torch.equal(before, model.scale.detach()))

    def test_physics_epoch_diagnostics_cover_every_case(self):
        steps = [
            {"data_loss_raw": 1.0, "physics_loss_raw": 2.0, "total_loss": 3.0,
             "gradient_diagnostics": {"gradient_cosine_similarity": -0.5, "data_gradient_norm": 2.0},
             "physics_components": {"bulk_pde": 4.0, "left_adiabatic": 1.0}},
            {"data_loss_raw": 3.0, "physics_loss_raw": 4.0, "total_loss": 5.0,
             "gradient_diagnostics": {"gradient_cosine_similarity": 0.5, "data_gradient_norm": 4.0},
             "physics_components": {"bulk_pde": 8.0, "left_adiabatic": 3.0}},
        ]
        result = aggregate_physics_steps(steps)
        self.assertEqual(result["case_count"], 2)
        self.assertAlmostEqual(result["gradient_diagnostics_mean"]["gradient_cosine_similarity"], 0.0)
        self.assertAlmostEqual(result["gradient_conflict_fraction"], 0.5)
        self.assertAlmostEqual(result["physics_components_mean"]["bulk_pde"], 6.0)

    def test_3d_epoch_and_evaluation_use_case_coordinates(self):
        model = ScalarFieldModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        samples = [{
            "case_id": 1,
            "coords": np.array([[0, 0, 0.1], [1, 0, 0.2]]),
            "centers_um": np.array([[0, 0, 100.0], [1000.0, 0, 200.0]]),
            "power_dimless": np.ones((2, 2)),
            "temperature_u": np.ones(2),
            "temperature_k": np.array([300.0, 301.0]),
        }]
        loss = supervised_3d_epoch(model, optimizer, samples, torch.device("cpu"))
        self.assertAlmostEqual(loss, 1.0)
        model.bias.data.fill_(0.7)
        result = evaluate_3d_cases(model, samples, torch.device("cpu"),
                                   temperature_ref_k=293.0,
                                   temperature_scale_k=10.0,
                                   ambient_k=293.0)
        self.assertEqual(result["cases"][0]["case_id"], 1)
        self.assertIn("hotspot_location_error_um", result["aggregate"])

    def test_mixed_step_logs_normalized_components_and_gradients(self):
        model = QuadraticScaleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        top = torch.tensor([[0.0, 0.0, 0.5], [1.0, 0.0, 0.5]])
        sample = {
            "power_dimless": np.ones((2, 2)),
            "temperature_u": np.array([[0.25, 1.25]]),
        }
        result = mixed_physics_step(
            model, optimizer, sample, top,
            coordinate_groups={"bulk": torch.tensor([[0.2, 0.3, 0.4]])},
            variant="source_free", physics_weight=0.25,
            reference_scales={"data": 2.0, "physics": 18.0},
            source_scale_per_beta=0.0, biot=5.0, ambient_u=0.2,
        )
        self.assertAlmostEqual(result["data_loss_raw"], 0.0, places=6)
        self.assertAlmostEqual(result["physics_loss_raw"], 36.0, places=4)
        self.assertAlmostEqual(result["total_loss"], 0.5, places=4)
        self.assertIn("gradient_cosine_similarity", result["gradient_diagnostics"])

    def test_supervised_epoch_updates_real_model_toward_target(self):
        model = ScalarFieldModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        coords = torch.zeros((4, 3))
        samples = [{
            "power_dimless": np.ones((2, 2)),
            "temperature_u": np.ones((2, 2)),
        }]

        loss = supervised_epoch(model, optimizer, samples, coords)

        self.assertAlmostEqual(loss, 1.0)
        self.assertGreater(model.bias.item(), 0.0)

    def test_evaluation_reports_full_field_metrics_for_each_case(self):
        model = ScalarFieldModel()
        model.bias.data.fill_(1.0)
        coords = torch.zeros((4, 3))
        samples = [{
            "case_id": 4,
            "power_dimless": np.ones((2, 2)),
            "temperature_k": np.array([[300.0, 301.0], [302.0, 303.0]]),
        }]
        result = evaluate_cases(
            model, samples, coords, temperature_ref_k=293.0,
            temperature_scale_k=10.0, ambient_k=293.0,
        )
        self.assertEqual(result["cases"][0]["case_id"], 4)
        self.assertAlmostEqual(result["cases"][0]["metrics"]["mae_k"], 1.5)
        self.assertAlmostEqual(result["aggregate"]["mae_k"], 1.5)

    def test_3d_evaluation_uses_each_samples_physical_centers(self):
        model = ScalarFieldModel()
        samples = [{
            "case_id": 8,
            "power_dimless": np.ones((2, 2)),
            "coords": np.array([[0.25, 0.25, 0.25], [0.75, 0.25, 0.25]]),
            "centers_um": np.array([[25.0, 25.0, 5.0], [75.0, 25.0, 5.0]]),
            "temperature_k": np.array([300.0, 310.0]),
        }]
        result = evaluate_3d_cases(
            model, samples, torch.device("cpu"), temperature_ref_k=300.0,
            temperature_scale_k=1.0, ambient_k=298.15,
        )
        self.assertAlmostEqual(
            result["cases"][0]["metrics"]["hotspot_location_error_um"], 50.0
        )


if __name__ == "__main__":
    unittest.main()
