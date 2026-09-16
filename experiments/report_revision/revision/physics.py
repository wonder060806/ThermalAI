"""Physics-loss scaling and configuration validation."""

from __future__ import annotations

import math
from collections.abc import Mapping


def gradient_diagnostics(data_loss, physics_loss, parameters) -> dict[str, float]:
    """Measure scale and alignment of two loss gradients without mutating `.grad`."""
    import torch

    params = [parameter for parameter in parameters if parameter.requires_grad]
    if not params:
        raise ValueError("gradient diagnostics requires trainable parameters")
    data_grads = torch.autograd.grad(
        data_loss, params, retain_graph=True, allow_unused=True
    )
    physics_grads = torch.autograd.grad(
        physics_loss, params, retain_graph=True, allow_unused=True
    )

    def flattened(grads):
        return torch.cat([
            (torch.zeros_like(parameter) if grad is None else grad).reshape(-1)
            for parameter, grad in zip(params, grads)
        ])

    data_vector = flattened(data_grads)
    physics_vector = flattened(physics_grads)
    data_norm = torch.linalg.vector_norm(data_vector)
    physics_norm = torch.linalg.vector_norm(physics_vector)
    denominator = data_norm * physics_norm
    cosine = (
        torch.dot(data_vector, physics_vector) / denominator
        if float(denominator.detach().cpu()) > 0.0
        else torch.tensor(float("nan"), device=data_vector.device)
    )
    ratio = physics_norm / data_norm if float(data_norm.detach().cpu()) > 0.0 else torch.tensor(float("inf"))
    return {
        "data_gradient_norm": float(data_norm.detach().cpu()),
        "physics_gradient_norm": float(physics_norm.detach().cpu()),
        "physics_to_data_gradient_ratio": float(ratio.detach().cpu()),
        "gradient_cosine_similarity": float(cosine.detach().cpu()),
    }


SOURCE_REQUIRED_FIELDS = (
    "length_scale_m",
    "source_thickness_m",
    "conductivity_w_mk",
    "temperature_scale_k",
)


def dimensionless_surface_flux(
    cell_power_w: float,
    cell_area_m2: float,
    length_scale_m: float,
    conductivity_w_mk: float,
    temperature_scale_k: float,
) -> float:
    """Return q'' L/(k ΔT) for a nondimensional Neumann heat-flux BC."""
    values = {
        "cell_area_m2": cell_area_m2,
        "length_scale_m": length_scale_m,
        "conductivity_w_mk": conductivity_w_mk,
        "temperature_scale_k": temperature_scale_k,
    }
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if not math.isfinite(cell_power_w) or cell_power_w < 0.0:
        raise ValueError("cell_power_w must be non-negative and finite")
    return cell_power_w / cell_area_m2 * length_scale_m / (
        conductivity_w_mk * temperature_scale_k
    )


def dimensionless_volumetric_source(
    cell_power_w: float,
    cell_area_m2: float,
    source_thickness_m: float,
    length_scale_m: float,
    conductivity_w_mk: float,
    temperature_scale_k: float,
) -> float:
    """Return q''' L²/(k ΔT) for the nondimensional steady heat equation."""
    values = {
        "cell_area_m2": cell_area_m2,
        "source_thickness_m": source_thickness_m,
        "length_scale_m": length_scale_m,
        "conductivity_w_mk": conductivity_w_mk,
        "temperature_scale_k": temperature_scale_k,
    }
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if not math.isfinite(cell_power_w) or cell_power_w < 0.0:
        raise ValueError("cell_power_w must be non-negative and finite")
    volumetric_heat_w_m3 = cell_power_w / (cell_area_m2 * source_thickness_m)
    return (
        volumetric_heat_w_m3
        * length_scale_m**2
        / (conductivity_w_mk * temperature_scale_k)
    )


def normalized_weighted_loss(
    losses: Mapping[str, float],
    weights: Mapping[str, float],
    reference_scales: Mapping[str, float],
) -> float:
    """Combine loss components after division by fixed positive reference scales."""
    if set(losses) != set(weights) or set(losses) != set(reference_scales):
        raise ValueError("losses, weights, and reference_scales must share keys")
    total = 0.0
    for name, loss in losses.items():
        scale = float(reference_scales[name])
        weight = float(weights[name])
        if scale <= 0.0 or not math.isfinite(scale):
            raise ValueError(f"reference scale for {name} must be positive")
        if weight < 0.0 or not math.isfinite(weight):
            raise ValueError(f"weight for {name} must be non-negative")
        total += weight * float(loss) / scale
    return total


def validate_physics_config(config: Mapping[str, object]) -> None:
    """Reject ambiguous or dimensionally incomplete physics configurations."""
    variant = config.get("variant")
    if variant not in {"none", "source_free", "surface_flux", "volumetric_source"}:
        raise ValueError("variant must be none, source_free, surface_flux, or volumetric_source")
    weights = config.get("weights")
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError("weights must be a non-empty mapping")
    total_weight = sum(float(value) for value in weights.values())
    if not math.isclose(total_weight, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("physics loss weights must sum to 1")
    if variant == "volumetric_source":
        for field in SOURCE_REQUIRED_FIELDS:
            if field not in config:
                raise ValueError(f"volumetric_source requires {field}")
            value = float(config[field])
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be positive and finite")
