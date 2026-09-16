"""Shared supervised loop used by all initialization comparisons."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from .metrics import thermal_metrics, thermal_metrics_3d
from .physics import gradient_diagnostics
from .physics_training import physics_loss_components


def calibrate_physics_reference_scales(raw_losses: Sequence[dict]) -> dict[str, float]:
    """Use the full calibration set so loss scales do not depend on case order."""
    if not raw_losses:
        raise ValueError("reference-scale calibration requires losses")
    return {
        "data": max(float(np.mean([item["data_loss_raw"] for item in raw_losses])), 1e-12),
        "physics": max(float(np.mean([item["physics_loss_raw"] for item in raw_losses])), 1e-12),
    }


def _physics_loss_tensors(model, sample, top_coords, coordinate_groups, variant,
                          source_scale_per_beta, biot, ambient_u,
                          surface_flux_scale_per_beta=1.0):
    beta_map = torch.as_tensor(
        np.asarray(sample["power_dimless"]), dtype=torch.float32,
        device=top_coords.device,
    )
    beta = beta_map.reshape(1, -1).repeat(top_coords.shape[0], 1)
    target = torch.as_tensor(
        np.asarray(sample["temperature_u"]).reshape(-1, 1),
        dtype=torch.float32, device=top_coords.device,
    )
    prediction = model(top_coords, beta)
    data_loss = torch.nn.functional.mse_loss(prediction.reshape_as(target), target)
    component_tensors = physics_loss_components(
        model, beta_map, coordinate_groups, variant, source_scale_per_beta,
        biot, ambient_u, surface_flux_scale_per_beta,
    )
    physics_loss = torch.stack(list(component_tensors.values())).mean()
    return data_loss, physics_loss, component_tensors


def physics_raw_losses(model, sample, top_coords, coordinate_groups, variant,
                       source_scale_per_beta, biot, ambient_u,
                       surface_flux_scale_per_beta=1.0) -> dict[str, float]:
    """Measure initial raw losses without changing parameters or optimizer state."""
    data_loss, physics_loss, _ = _physics_loss_tensors(
        model, sample, top_coords, coordinate_groups, variant,
        source_scale_per_beta, biot, ambient_u, surface_flux_scale_per_beta,
    )
    return {"data_loss_raw": float(data_loss.detach().cpu()),
            "physics_loss_raw": float(physics_loss.detach().cpu())}


def aggregate_physics_steps(steps: Sequence[dict]) -> dict[str, object]:
    """Summarize all cases in an epoch without privileging the first case."""
    if not steps:
        raise ValueError("physics epoch requires at least one step")
    diagnostics = sorted(steps[0]["gradient_diagnostics"])
    components = sorted(steps[0]["physics_components"])
    for step in steps:
        if sorted(step["gradient_diagnostics"]) != diagnostics:
            raise ValueError("inconsistent gradient diagnostic fields")
        if sorted(step["physics_components"]) != components:
            raise ValueError("inconsistent physics component fields")
    cosine_values = [float(step["gradient_diagnostics"]["gradient_cosine_similarity"])
                     for step in steps]
    return {
        "case_count": len(steps),
        "data_loss_raw_mean": float(np.mean([step["data_loss_raw"] for step in steps])),
        "physics_loss_raw_mean": float(np.mean([step["physics_loss_raw"] for step in steps])),
        "total_loss_mean": float(np.mean([step["total_loss"] for step in steps])),
        "gradient_diagnostics_mean": {
            name: float(np.mean([step["gradient_diagnostics"][name] for step in steps]))
            for name in diagnostics
        },
        "gradient_conflict_fraction": float(np.mean([value < 0.0 for value in cosine_values])),
        "physics_components_mean": {
            name: float(np.mean([step["physics_components"][name] for step in steps]))
            for name in components
        },
    }


def _sample_tensors(sample, coords):
    device = coords.device
    power = torch.as_tensor(
        np.asarray(sample["power_dimless"]).reshape(-1),
        dtype=torch.float32,
        device=device,
    )
    beta = power.unsqueeze(0).repeat(coords.shape[0], 1)
    return beta


def supervised_epoch(model, optimizer, samples: Sequence[dict], coords) -> float:
    """Train one epoch over every case with per-case optimizer steps."""
    if not samples:
        raise ValueError("supervised_epoch requires at least one sample")
    model.train()
    losses = []
    for sample in samples:
        beta = _sample_tensors(sample, coords)
        target = torch.as_tensor(
            np.asarray(sample["temperature_u"]).reshape(-1, 1),
            dtype=torch.float32,
            device=coords.device,
        )
        if target.shape[0] != coords.shape[0]:
            raise ValueError("temperature target and coordinates have different sizes")
        optimizer.zero_grad(set_to_none=True)
        prediction = model(coords, beta)
        loss = torch.nn.functional.mse_loss(prediction.reshape_as(target), target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def mixed_physics_step(
    model,
    optimizer,
    sample: dict,
    top_coords,
    coordinate_groups,
    variant: str,
    physics_weight: float,
    reference_scales: dict[str, float] | None,
    source_scale_per_beta: float,
    biot: float,
    ambient_u: float,
    surface_flux_scale_per_beta: float = 1.0,
) -> dict[str, object]:
    """Take one normalized mixed-loss step and retain conflict diagnostics."""
    if not 0.0 <= physics_weight <= 1.0:
        raise ValueError("physics_weight must be between zero and one")
    data_loss, physics_loss, component_tensors = _physics_loss_tensors(
        model, sample, top_coords, coordinate_groups, variant,
        source_scale_per_beta, biot, ambient_u, surface_flux_scale_per_beta,
    )
    if reference_scales is None:
        reference_scales = {
            "data": max(float(data_loss.detach().cpu()), 1e-12),
            "physics": max(float(physics_loss.detach().cpu()), 1e-12),
        }
    if reference_scales.get("data", 0) <= 0 or reference_scales.get("physics", 0) <= 0:
        raise ValueError("reference scales must be positive")
    normalized_data = data_loss / reference_scales["data"]
    normalized_physics = physics_loss / reference_scales["physics"]
    diagnostics = gradient_diagnostics(
        normalized_data, normalized_physics, model.parameters()
    )
    total_loss = (1.0 - physics_weight) * normalized_data + physics_weight * normalized_physics
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    optimizer.step()
    return {
        "data_loss_raw": float(data_loss.detach().cpu()),
        "physics_loss_raw": float(physics_loss.detach().cpu()),
        "physics_components": {
            name: float(value.detach().cpu()) for name, value in component_tensors.items()
        },
        "total_loss": float(total_loss.detach().cpu()),
        "reference_scales": dict(reference_scales),
        "gradient_diagnostics": diagnostics,
    }


def supervised_3d_epoch(model, optimizer, samples: Sequence[dict], device) -> float:
    """Train on every physical cell center in each full-volume case."""
    if not samples:
        raise ValueError("supervised_3d_epoch requires samples")
    model.train()
    losses = []
    for sample in samples:
        coords = torch.as_tensor(sample["coords"], dtype=torch.float32, device=device)
        beta_map = torch.as_tensor(sample["power_dimless"], dtype=torch.float32, device=device)
        beta = beta_map.reshape(1, -1).repeat(coords.shape[0], 1)
        target = torch.as_tensor(sample["temperature_u"], dtype=torch.float32, device=device).reshape(-1, 1)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(coords, beta)
        loss = torch.nn.functional.mse_loss(prediction.reshape_as(target), target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def evaluate_3d_cases(
    model, samples: Sequence[dict], device, temperature_ref_k: float,
    temperature_scale_k: float, ambient_k: float,
) -> dict[str, object]:
    model.eval()
    cases = []
    for sample in samples:
        coords = torch.as_tensor(sample["coords"], dtype=torch.float32, device=device)
        beta_map = torch.as_tensor(sample["power_dimless"], dtype=torch.float32, device=device)
        beta = beta_map.reshape(1, -1).repeat(coords.shape[0], 1)
        prediction_u = model(coords, beta).detach().cpu().numpy().reshape(-1)
        prediction_k = temperature_ref_k + temperature_scale_k * prediction_u
        metrics = thermal_metrics_3d(
            np.asarray(sample["temperature_k"]), prediction_k,
            np.asarray(sample["centers_um"]), ambient_k,
        )
        cases.append({"case_id": int(sample["case_id"]), "metrics": metrics})
    aggregate = {}
    if cases:
        for name in cases[0]["metrics"]:
            values = [case["metrics"][name] for case in cases]
            finite = [float(value) for value in values if value is not None and np.isfinite(value)]
            aggregate[name] = float(np.mean(finite)) if finite else None
    return {"cases": cases, "aggregate": aggregate}


@torch.no_grad()
def evaluate_cases(
    model,
    samples: Sequence[dict],
    coords,
    temperature_ref_k: float,
    temperature_scale_k: float,
    ambient_k: float,
) -> dict[str, object]:
    """Evaluate complete fields and aggregate each metric across cases."""
    model.eval()
    cases = []
    for sample in samples:
        beta = _sample_tensors(sample, coords)
        prediction_u = model(coords, beta).detach().cpu().numpy().reshape(
            np.asarray(sample["temperature_k"]).shape
        )
        prediction_k = temperature_ref_k + temperature_scale_k * prediction_u
        metrics = thermal_metrics(
            np.asarray(sample["temperature_k"]), prediction_k, ambient_k
        )
        cases.append({"case_id": int(sample["case_id"]), "metrics": metrics})
    aggregate = {}
    if cases:
        for metric_name in cases[0]["metrics"]:
            values = [case["metrics"][metric_name] for case in cases]
            finite_values = [value for value in values if value is not None]
            aggregate[metric_name] = (
                float(np.mean(finite_values)) if finite_values else None
            )
    return {"cases": cases, "aggregate": aggregate}
