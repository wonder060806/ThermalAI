"""Aggregation of paired PINN-versus-random formal experiment records."""

from __future__ import annotations

import math
import statistics
from itertools import product
from typing import Any

from scipy.stats import t as student_t
from scipy.stats import ttest_1samp


def _summary(values: list[float]) -> dict[str, float | int]:
    mean = statistics.fmean(values)
    if len(values) > 1:
        standard_error = statistics.stdev(values) / math.sqrt(len(values))
        critical_value = float(student_t.ppf(0.975, df=len(values) - 1))
        half_width = critical_value * standard_error
    else:
        critical_value = 0.0
        half_width = 0.0
    return {"n": len(values), "mean": mean, "ci95_low": mean - half_width,
            "ci95_high": mean + half_width, "ci_method": "student_t",
            "ci_critical_value": critical_value}


def _aggregate_mae_k(record: dict[str, Any]) -> float:
    aggregate = record["test"]["aggregate"]
    if "mae_k" in aggregate:
        return float(aggregate["mae_k"])
    if "mae_k_mean" in aggregate:
        return float(aggregate["mae_k_mean"])
    raise KeyError("test.aggregate must contain mae_k or legacy mae_k_mean")


def _paired_t_p_value(differences: list[float]) -> float | None:
    if len(differences) < 2:
        return None
    if all(math.isclose(value, differences[0], rel_tol=0.0, abs_tol=1e-15)
           for value in differences):
        return 1.0 if math.isclose(differences[0], 0.0, abs_tol=1e-15) else 0.0
    return float(ttest_1samp(differences, popmean=0.0).pvalue)


def _exact_sign_flip_p_value(differences: list[float]) -> float | None:
    """Exact two-sided paired randomization test on the absolute mean difference."""
    if not differences:
        return None
    observed = abs(statistics.fmean(differences))
    if math.isclose(observed, 0.0, abs_tol=1e-15):
        return 1.0
    magnitudes = [abs(value) for value in differences]
    extreme = sum(
        abs(statistics.fmean(sign * value for sign, value in zip(signs, magnitudes)))
        >= observed - 1e-15
        for signs in product((-1.0, 1.0), repeat=len(magnitudes))
    )
    return extreme / (2 ** len(magnitudes))


def _add_holm_adjustment(groups: dict[str, dict[str, Any]]) -> None:
    available = [
        (name, float(group["exact_sign_flip_p_value_two_sided"]))
        for name, group in groups.items()
        if group.get("exact_sign_flip_p_value_two_sided") is not None
    ]
    ordered = sorted(available, key=lambda item: item[1])
    running_max = 0.0
    count = len(ordered)
    for rank, (name, p_value) in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * p_value)
        running_max = max(running_max, adjusted)
        groups[name]["holm_adjusted_p_value"] = running_max
        groups[name]["significant_after_holm_0p05"] = running_max < 0.05
    for group in groups.values():
        group.setdefault("holm_adjusted_p_value", None)
        group.setdefault("significant_after_holm_0p05", False)


def aggregate_paired_results(
    records: list[dict[str, Any]], expected_seeds: dict[int, list[int]]
) -> dict[str, Any]:
    indexed = {}
    for record in records:
        if record.get("status") != "completed":
            continue
        config = record["config"]
        init_seed = int(config["init_seed"])
        data_seed = int(config.get("data_seed", init_seed))
        key = (int(config["train_size"]), data_seed, init_seed, str(config["method"]))
        if key in indexed:
            raise ValueError(f"duplicate completed result: {key}")
        indexed[key] = _aggregate_mae_k(record)
    groups = {}
    complete = True
    for train_size, seeds in sorted(expected_seeds.items()):
        differences = []
        wins = 0
        missing = []
        pinn_values = []
        random_values = []
        for seed in seeds:
            pinn = indexed.get((train_size, seed, seed, "pinn"))
            random = indexed.get((train_size, seed, seed, "random"))
            if pinn is None or random is None:
                missing.append(seed)
                continue
            pinn_values.append(pinn)
            random_values.append(random)
            differences.append(pinn - random)
            wins += int(pinn < random)
        if missing:
            complete = False
        groups[str(train_size)] = {
            "paired_count": len(differences),
            "missing_seeds": missing,
            "pinn_mae_k": _summary(pinn_values) if pinn_values else None,
            "random_mae_k": _summary(random_values) if random_values else None,
            "pinn_minus_random_mae_k": _summary(differences) if differences else None,
            "pinn_win_rate": wins / len(differences) if differences else None,
            "paired_t_p_value_two_sided": _paired_t_p_value(differences),
            "exact_sign_flip_p_value_two_sided": _exact_sign_flip_p_value(differences),
            "uncertainty_unit": "paired_training_run_seed",
        }
    _add_holm_adjustment(groups)
    return {"status": "complete" if complete else "incomplete", "groups": groups}


def aggregate_physics_results(
    records: list[dict[str, Any]],
    variants=("surface_flux", "volumetric_source"),
    weights=(0.01, 0.05, 0.1, 0.25, 0.5),
    seeds=tuple(range(10)),
) -> dict[str, Any]:
    indexed = {}
    for record in records:
        if record.get("status") != "completed":
            continue
        config = record["config"]
        init_seed = int(config["init_seed"])
        data_seed = int(config.get("data_seed", init_seed))
        key = (
            str(config.get("physics_variant", "none")),
            float(config.get("physics_weight", 0.0)),
            data_seed,
            init_seed,
        )
        if key in indexed:
            raise ValueError(f"duplicate completed physics result: {key}")
        indexed[key] = _aggregate_mae_k(record)
    settings = {}
    complete = True
    for variant in variants:
        for weight in weights:
            differences = []
            values = []
            missing = []
            wins = 0
            for seed in seeds:
                baseline = indexed.get(("none", 0.0, int(seed), int(seed)))
                value = indexed.get((str(variant), float(weight), int(seed), int(seed)))
                if baseline is None or value is None:
                    missing.append(int(seed))
                    continue
                values.append(value)
                differences.append(value - baseline)
                wins += int(value < baseline)
            if missing:
                complete = False
            settings[f"{variant}:w={weight}"] = {
                "paired_count": len(differences), "missing_seeds": missing,
                "mae_k": _summary(values) if values else None,
                "mae_delta_vs_supervised_k": _summary(differences) if differences else None,
                "win_rate_vs_supervised": wins / len(differences) if differences else None,
                "paired_t_p_value_two_sided": _paired_t_p_value(differences),
                "exact_sign_flip_p_value_two_sided": _exact_sign_flip_p_value(differences),
                "uncertainty_unit": "paired_training_run_seed",
            }
    _add_holm_adjustment(settings)
    return {"status": "complete" if complete else "incomplete", "settings": settings}


def aggregate_seeded_metrics(
    records: list[dict[str, Any]], expected_seeds=(0, 1, 2, 3, 4)
) -> dict[str, Any]:
    by_seed = {}
    for record in records:
        if record.get("status") != "completed":
            continue
        seed = int(record["config"]["init_seed"])
        if seed in by_seed:
            raise ValueError(f"duplicate completed seed result: {seed}")
        by_seed[seed] = record["test"]["aggregate"]
    missing = [int(seed) for seed in expected_seeds if int(seed) not in by_seed]
    metric_names = sorted({name for metrics in by_seed.values() for name in metrics})
    summaries = {}
    for name in metric_names:
        values = [float(metrics[name]) for metrics in by_seed.values()
                  if metrics.get(name) is not None and math.isfinite(float(metrics[name]))]
        summaries[name] = _summary(values) if values else None
    return {"status": "incomplete" if missing else "complete",
            "completed_seeds": sorted(by_seed), "missing_seeds": missing,
            "metrics": summaries}
