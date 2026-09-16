"""Pure dataset-audit and trivial-baseline utilities."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np


def numeric_parameter_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, float | int]]:
    """Summarize scalar numeric parameter coverage, excluding case identifiers."""
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for name, value in row.items():
            if name == "id" or isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            number = float(value)
            if np.isfinite(number):
                values[name].append(number)
    return {
        name: {
            "count": len(numbers), "min": min(numbers), "max": max(numbers),
            "range": max(numbers) - min(numbers), "mean": float(np.mean(numbers)),
            "std": float(np.std(numbers)),
        }
        for name, numbers in sorted(values.items()) if numbers
    }


def cross_split_audit(
    arrays: Mapping[str, np.ndarray],
    train_ids: Sequence[str],
    test_ids: Sequence[str],
) -> dict[str, object]:
    """Measure exact leakage and nearest-map similarity strictly across a split."""
    train = [str(value) for value in train_ids]
    test = [str(value) for value in test_ids]
    if set(train) & set(test):
        raise ValueError("train and test IDs overlap")
    missing = (set(train) | set(test)) - set(arrays)
    if missing:
        raise ValueError(f"split IDs lack fields: {sorted(missing)}")
    duplicates = []
    nearest = {}
    train_hashes = {case_id: array_fingerprint(arrays[case_id]) for case_id in train}
    for test_id in test:
        test_array = _finite_array(arrays[test_id])
        matches = [case_id for case_id in train if train_hashes[case_id] == array_fingerprint(test_array)]
        duplicates.extend([[case_id, test_id] for case_id in matches])
        candidates = []
        for train_id in train:
            train_array = _finite_array(arrays[train_id])
            if train_array.shape != test_array.shape:
                raise ValueError("all cross-split arrays must share a shape")
            candidates.append((float(np.sqrt(np.mean((test_array - train_array) ** 2))), train_id))
        distance, neighbor = min(candidates)
        nearest[test_id] = {"train_case": neighbor, "rmse": distance}
    return {
        "train_count": len(train), "test_count": len(test),
        "exact_cross_split_duplicates": duplicates,
        "nearest_train_by_test": nearest,
    }


def _finite_array(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("arrays must be non-empty and finite")
    return array


def array_fingerprint(value: np.ndarray, decimals: int | None = None) -> str:
    """Hash shape and float64 contents, optionally after decimal rounding."""
    array = _finite_array(value)
    if decimals is not None:
        array = np.ascontiguousarray(np.round(array, decimals=decimals))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def duplicate_groups(
    arrays: Mapping[str, np.ndarray], decimals: int | None = None
) -> list[list[str]]:
    """Return sorted groups of case IDs sharing an exact or rounded hash."""
    by_hash: dict[str, list[str]] = defaultdict(list)
    for case_id, array in arrays.items():
        by_hash[array_fingerprint(array, decimals=decimals)].append(str(case_id))
    return sorted(
        [sorted(case_ids) for case_ids in by_hash.values() if len(case_ids) > 1],
        key=lambda group: group[0],
    )


def nearest_neighbor_distances(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float | str]]:
    """Find each case's nearest different case using elementwise RMSE."""
    prepared = {str(key): _finite_array(value) for key, value in arrays.items()}
    if len(prepared) < 2:
        return {}
    shapes = {array.shape for array in prepared.values()}
    if len(shapes) != 1:
        raise ValueError("all arrays must have the same shape")
    result: dict[str, dict[str, float | str]] = {}
    for case_id, array in prepared.items():
        candidates = []
        for other_id, other in prepared.items():
            if other_id == case_id:
                continue
            rmse = float(np.sqrt(np.mean(np.square(array - other))))
            candidates.append((rmse, other_id))
        distance, neighbor = min(candidates)
        result[case_id] = {"neighbor": neighbor, "rmse": distance}
    return result


def dataset_temperature_summary(
    fields: Mapping[str, np.ndarray], ambient_k: float
) -> dict[str, object]:
    """Summarize spatial signal and temperature rise for every case."""
    cases: dict[str, dict[str, float]] = {}
    for case_id, field in fields.items():
        array = _finite_array(field)
        cases[str(case_id)] = {
            "min_k": float(np.min(array)),
            "max_k": float(np.max(array)),
            "mean_k": float(np.mean(array)),
            "std_k": float(np.std(array)),
            "range_k": float(np.ptp(array)),
            "mean_rise_k": float(np.mean(array - ambient_k)),
            "max_rise_k": float(np.max(array) - ambient_k),
        }
    ranges = [case["range_k"] for case in cases.values()]
    return {
        "n_cases": len(cases),
        "ambient_k": float(ambient_k),
        "median_spatial_range_k": float(np.median(ranges)) if ranges else None,
        "cases": cases,
    }


def predict_global_mean(train_fields: np.ndarray, output_shape: Sequence[int]) -> np.ndarray:
    """Predict one scalar training-set mean at every output location."""
    train = _finite_array(train_fields)
    return np.full(tuple(output_shape), float(np.mean(train)), dtype=np.float64)


def total_power_regression_predictions(
    train_power: np.ndarray,
    train_temperature: np.ndarray,
    query_power: np.ndarray,
) -> np.ndarray:
    """Fit temperature = intercept + slope * total_power and predict queries."""
    x = _finite_array(train_power).reshape(-1)
    y = _finite_array(train_temperature).reshape(-1)
    query = _finite_array(query_power).reshape(-1)
    if x.size != y.size or x.size < 2:
        raise ValueError("power regression requires at least two paired samples")
    design = np.column_stack([np.ones_like(x), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coefficients[0] + coefficients[1] * query
