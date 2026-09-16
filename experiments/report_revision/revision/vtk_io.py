"""Parser for ASCII legacy VTK files emitted by 3D-ICE T3d."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def validate_zero_power_truth(
    temperature_k: np.ndarray, ambient_k: float, tolerance_k: float,
) -> dict[str, float]:
    values = np.asarray(temperature_k, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all() or tolerance_k <= 0:
        raise ValueError("invalid zero-power gate inputs")
    deviation = float(np.max(np.abs(values - ambient_k)))
    if deviation > tolerance_k:
        raise ValueError("zero-power field does not return to ambient temperature")
    return {"ambient_k": float(ambient_k), "maximum_ambient_deviation_k": deviation}


def grid_convergence_summary(
    medium_layers: list[dict[str, float | int]],
    fine_layers: list[dict[str, float | int]],
    max_difference_k: float,
) -> dict[str, float | int]:
    if len(medium_layers) != len(fine_layers) or not medium_layers:
        raise ValueError("grid convergence requires matching physical layers")
    medium_z = np.asarray([float(layer["z_center_um"]) for layer in medium_layers])
    fine_z = np.asarray([float(layer["z_center_um"]) for layer in fine_layers])
    if not np.allclose(medium_z, fine_z, atol=1e-6, rtol=0.0):
        raise ValueError("grid convergence layers have different z locations")
    mean_differences = np.abs(
        np.asarray([float(layer["mean_k"]) for layer in medium_layers])
        - np.asarray([float(layer["mean_k"]) for layer in fine_layers])
    )
    max_temperature_difference = abs(
        max(float(layer["max_k"]) for layer in medium_layers)
        - max(float(layer["max_k"]) for layer in fine_layers)
    )
    observed = max(max_temperature_difference, float(np.max(mean_differences)))
    if observed > max_difference_k:
        raise ValueError("grid convergence difference exceeds configured tolerance")
    return {
        "n_layers": len(medium_layers),
        "maximum_layer_mean_difference_k": float(np.max(mean_differences)),
        "maximum_temperature_difference_k": float(max_temperature_difference),
        "gate_max_difference_k": float(max_difference_k),
    }


def _find_line(lines: list[str], prefix: str) -> int:
    for index, line in enumerate(lines):
        if line.strip().startswith(prefix):
            return index
    raise ValueError(f"VTK section missing: {prefix}")


def parse_3dice_vtk(path: Path) -> dict[str, np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    points_header = _find_line(lines, "POINTS ")
    point_count = int(lines[points_header].split()[1])
    point_values = []
    cursor = points_header + 1
    while len(point_values) < point_count * 3:
        point_values.extend(float(value) for value in lines[cursor].split())
        cursor += 1
    points = np.asarray(point_values[: point_count * 3], dtype=np.float64).reshape(-1, 3)

    cells_header = _find_line(lines, "CELLS ")
    cell_count = int(lines[cells_header].split()[1])
    cells = []
    cursor = cells_header + 1
    for _ in range(cell_count):
        values = [int(value) for value in lines[cursor].split()]
        cursor += 1
        vertex_count = values[0]
        if len(values[1:]) != vertex_count:
            raise ValueError("malformed VTK cell connectivity")
        cells.append(values[1:])

    scalar_header = _find_line(lines, "SCALARS Temperature")
    lookup_line = scalar_header + 1
    if not lines[lookup_line].strip().startswith("LOOKUP_TABLE"):
        raise ValueError("temperature scalar lacks LOOKUP_TABLE")
    temperature_values = []
    cursor = lookup_line + 1
    while len(temperature_values) < cell_count:
        temperature_values.extend(float(value) for value in lines[cursor].split())
        cursor += 1
    temperature = np.asarray(temperature_values[:cell_count], dtype=np.float64)
    centers = np.asarray([np.mean(points[cell], axis=0) for cell in cells])
    if not np.isfinite(centers).all() or not np.isfinite(temperature).all():
        raise ValueError("VTK contains non-finite coordinates or temperatures")
    return {"centers_um": centers, "temperature_k": temperature}


def layer_temperature_summary(
    centers_um: np.ndarray, temperature_k: np.ndarray
) -> list[dict[str, float | int]]:
    centers = np.asarray(centers_um, dtype=np.float64)
    temperatures = np.asarray(temperature_k, dtype=np.float64).reshape(-1)
    if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] != temperatures.size:
        raise ValueError("centers must be N×3 and match N temperatures")
    layers = []
    rounded_z = np.round(centers[:, 2], decimals=9)
    for z_value in sorted(np.unique(rounded_z)):
        values = temperatures[rounded_z == z_value]
        layers.append({
            "z_center_um": float(z_value),
            "n_cells": int(values.size),
            "min_k": float(np.min(values)),
            "max_k": float(np.max(values)),
            "mean_k": float(np.mean(values)),
            "range_k": float(np.ptp(values)),
        })
    return layers


def validate_multilayer_truth(
    layers: list[dict[str, float | int]],
    min_layers: int,
    min_mean_span_k: float,
    max_k: float,
) -> dict[str, float | int]:
    """Gate 3D truth on layer count, vertical signal, and physical temperature."""
    if len(layers) < min_layers:
        raise ValueError(f"requires at least {min_layers} physical layers")
    means = np.asarray([float(layer["mean_k"]) for layer in layers])
    maxima = np.asarray([float(layer["max_k"]) for layer in layers])
    mean_span = float(np.ptp(means))
    if mean_span < min_mean_span_k:
        raise ValueError("layer-to-layer temperature signal is too small")
    observed_max = float(np.max(maxima))
    if observed_max > max_k:
        raise ValueError("maximum temperature exceeds the configured physical gate")
    return {
        "n_layers": len(layers),
        "layer_mean_span_k": mean_span,
        "maximum_temperature_k": observed_max,
    }
