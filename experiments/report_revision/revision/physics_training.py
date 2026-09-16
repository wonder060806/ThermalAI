"""Differentiable, mean-square physics residuals for DeepONet fine-tuning."""

from __future__ import annotations

import torch


def sample_physics_coordinates(
    n_interior: int,
    n_boundary: int,
    seed: int,
    source_top_z: float = 0.5,
    source_thickness_z: float = 0.005,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    if n_interior <= 0 or n_boundary <= 0 or source_thickness_z <= 0:
        raise ValueError("physics sample counts and source thickness must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def rand(rows, columns):
        return torch.rand((rows, columns), generator=generator).to(device)

    bulk = rand(n_interior, 3)
    bulk[:, 2] *= source_top_z - source_thickness_z
    source = rand(n_interior, 3)
    source[:, 2] = source_top_z - source_thickness_z + source[:, 2] * source_thickness_z
    xy = rand(n_boundary, 2)
    bottom = torch.column_stack([xy, torch.zeros(n_boundary, device=device)])
    top = torch.column_stack([rand(n_boundary, 2), torch.full((n_boundary,), source_top_z, device=device)])
    side_a = rand(n_boundary, 2)
    side_a[:, 1] *= source_top_z
    side_b = rand(n_boundary, 2)
    side_b[:, 1] *= source_top_z
    return {
        "bulk": bulk, "source": source, "bottom": bottom, "top": top,
        "left": torch.column_stack([torch.zeros(n_boundary, device=device), side_a]),
        "right": torch.column_stack([torch.ones(n_boundary, device=device), side_a]),
        "front": torch.column_stack([
            side_b[:, 0], torch.zeros(n_boundary, device=device), side_b[:, 1]
        ]),
        "back": torch.column_stack([
            side_b[:, 0], torch.ones(n_boundary, device=device), side_b[:, 1]
        ]),
    }


def _prediction_and_derivatives(model, beta_map, raw_coords):
    coords = raw_coords.detach().clone().requires_grad_(True)
    beta = beta_map.reshape(1, -1).repeat(coords.shape[0], 1)
    if hasattr(model, "physics_forward"):
        prediction, coords = model.physics_forward(coords, beta)
    else:
        prediction = model(coords, beta)
    gradient = torch.autograd.grad(
        prediction, coords, torch.ones_like(prediction), create_graph=True
    )[0]
    laplacian = torch.zeros_like(prediction)
    for dimension in range(3):
        second = torch.autograd.grad(
            gradient[:, dimension], coords,
            torch.ones_like(gradient[:, dimension]), create_graph=True,
        )[0][:, dimension : dimension + 1]
        laplacian = laplacian + second
    return prediction, gradient, laplacian, coords


def _source_beta(beta_map, coords):
    if beta_map.ndim != 2:
        raise ValueError("power map must be a 2D nodal field")
    # The power field is prescribed input data, not a trainable quantity.  The
    # physics coordinates require gradients only for temperature derivatives;
    # allowing grid_sample to differentiate its lookup grid creates an
    # irrelevant CUDA backward path that is not deterministic.
    beta_map = beta_map.detach()
    coords = coords.detach()
    # beta_map is stored [x, y]; grid_sample expects [N, C, y, x].
    image = beta_map.transpose(0, 1).unsqueeze(0).unsqueeze(0)
    grid = (coords[:, :2] * 2.0 - 1.0).reshape(1, -1, 1, 2)
    sampled = torch.nn.functional.grid_sample(
        image, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return sampled.reshape(-1, 1)


def physics_loss_components(
    model,
    beta_map: torch.Tensor,
    coordinate_groups: dict[str, torch.Tensor],
    variant: str,
    source_scale_per_beta: float,
    biot: float,
    ambient_u: float,
    surface_flux_scale_per_beta: float = 1.0,
) -> dict[str, torch.Tensor]:
    if variant not in {"source_free", "surface_flux", "volumetric_source"}:
        raise ValueError("physics variant must be source_free, surface_flux, or volumetric_source")
    losses = {}
    for name, raw_coords in coordinate_groups.items():
        prediction, gradient, laplacian, coords = _prediction_and_derivatives(
            model, beta_map, raw_coords
        )
        if name == "bulk":
            losses["pde"] = torch.mean(laplacian.square())
        elif name == "source":
            source = (
                source_scale_per_beta * _source_beta(beta_map, coords)
                if variant == "volumetric_source" else 0.0
            )
            residual = laplacian + source
            losses["source_pde"] = torch.mean(residual.square())
        elif name == "bottom":
            residual = gradient[:, 2:3] - biot * (prediction - ambient_u)
            losses["bottom_robin"] = torch.mean(residual.square())
        elif name == "top":
            if variant == "surface_flux":
                residual = gradient[:, 2:3] - (
                    surface_flux_scale_per_beta * _source_beta(beta_map, coords)
                )
                losses["top_surface_flux"] = torch.mean(residual.square())
            else:
                losses["top_adiabatic"] = torch.mean(gradient[:, 2:3].square())
        elif name == "left":
            losses["left_adiabatic"] = torch.mean(gradient[:, 0:1].square())
        elif name == "right":
            losses["right_adiabatic"] = torch.mean(gradient[:, 0:1].square())
        elif name == "front":
            losses["front_adiabatic"] = torch.mean(gradient[:, 1:2].square())
        elif name == "back":
            losses["back_adiabatic"] = torch.mean(gradient[:, 1:2].square())
    if not losses:
        raise ValueError("no recognized physics coordinate groups")
    return losses
