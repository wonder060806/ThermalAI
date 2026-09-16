"""Thermal-field metrics with engineering-relevant normalization."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def thermal_metrics_3d(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    centers_um: np.ndarray,
    ambient_k: float,
) -> dict[str, float | None]:
    """Compute field metrics with physical 3D hotspot coordinates."""
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    centers = np.asarray(centers_um, dtype=np.float64)
    if centers.shape != (true.size, 3) or pred.size != true.size:
        raise ValueError("3D centers must be N×3 and match temperature arrays")
    if not np.isfinite(centers).all():
        raise ValueError("3D centers must be finite")
    result = thermal_metrics(true, pred, ambient_k)
    true_index = int(np.argmax(true))
    pred_index = int(np.argmax(pred))
    result["hotspot_location_error_um"] = float(
        np.linalg.norm(centers[true_index] - centers[pred_index])
    )
    result["true_hotspot_x_um"] = float(centers[true_index, 0])
    result["true_hotspot_y_um"] = float(centers[true_index, 1])
    result["true_hotspot_z_um"] = float(centers[true_index, 2])
    result["predicted_hotspot_x_um"] = float(centers[pred_index, 0])
    result["predicted_hotspot_y_um"] = float(centers[pred_index, 1])
    result["predicted_hotspot_z_um"] = float(centers[pred_index, 2])
    layer_mae = [
        float(np.mean(np.abs(pred[centers[:, 2] == z] - true[centers[:, 2] == z])))
        for z in np.unique(centers[:, 2])
    ]
    result["layer_mae_k_mean"] = float(np.mean(layer_mae))
    result["layer_mae_k_max"] = float(np.max(layer_mae))
    return result


def thermal_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ambient_k: float,
    spacing: Sequence[float] | None = None,
) -> dict[str, float | None]:
    """Compute absolute, hotspot, and temperature-rise-relative errors."""
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if true.size == 0:
        raise ValueError("temperature arrays must not be empty")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("temperature arrays must contain only finite values")

    error = np.abs(pred - true)
    true_hotspot = np.unravel_index(int(np.argmax(true)), true.shape)
    pred_hotspot = np.unravel_index(int(np.argmax(pred)), pred.shape)
    if spacing is None:
        axis_spacing = np.ones(true.ndim, dtype=np.float64)
    else:
        axis_spacing = np.asarray(spacing, dtype=np.float64)
        if axis_spacing.shape != (true.ndim,):
            raise ValueError("spacing must provide one value per array dimension")
    location_delta = (
        np.asarray(true_hotspot, dtype=np.float64)
        - np.asarray(pred_hotspot, dtype=np.float64)
    ) * axis_spacing

    mean_rise = float(np.mean(np.abs(true - float(ambient_k))))
    hotspot_rise = float(abs(np.max(true) - float(ambient_k)))
    field_range = float(np.ptp(true))
    nonzero_temperature = np.abs(true) > np.finfo(np.float64).eps

    return {
        "mae_k": float(np.mean(error)),
        "rmse_k": float(np.sqrt(np.mean(np.square(pred - true)))),
        "max_abs_error_k": float(np.max(error)),
        "hotspot_temperature_error_k": float(abs(np.max(pred) - np.max(true))),
        "hotspot_location_error": float(np.linalg.norm(location_delta)),
        "relative_temperature_rise_error": (
            float(np.mean(error) / mean_rise) if mean_rise > 0.0 else None
        ),
        "relative_hotspot_temperature_rise_error": (
            float(abs(np.max(pred) - np.max(true)) / hotspot_rise)
            if hotspot_rise > 0.0 else None
        ),
        "nmae_by_range": float(np.mean(error) / field_range) if field_range > 0.0 else None,
        "absolute_temperature_mape": (
            float(np.mean(error[nonzero_temperature] / np.abs(true[nonzero_temperature])))
            if np.any(nonzero_temperature)
            else None
        ),
    }
