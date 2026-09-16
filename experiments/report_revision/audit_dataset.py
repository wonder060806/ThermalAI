#!/usr/bin/env python3
"""Audit ThermalAI temperature maps without modifying source data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from revision.audit import cross_split_audit, dataset_temperature_summary, duplicate_groups
from revision.audit import nearest_neighbor_distances, total_power_regression_predictions
from revision.audit import numeric_parameter_summary
from revision.metrics import thermal_metrics
from revision.schema import canonical_json
from revision.splits import layout_group_id


def load_dataset(data_dir: Path, mode: str):
    if mode == "solid":
        prefix = "case_"
        params_path = data_dir / "cases_params.json"
    else:
        prefix = f"case_{mode}_"
        params_path = data_dir / f"cases_{mode}_params.json"
    params = json.loads(params_path.read_text(encoding="utf-8"))
    powers = {str(item["id"]): float(item["total_power_mw"]) for item in params}
    fields = {}
    for path in sorted(data_dir.glob(f"{prefix}*_temp.txt")):
        case_id = path.stem.removeprefix(prefix).removesuffix("_temp")
        fields[case_id] = np.loadtxt(path, comments="%", dtype=np.float64)
    extra_maps = set(fields) - set(powers)
    if extra_maps:
        raise ValueError(f"temperature maps lack parameter rows: {sorted(extra_maps)}")
    missing_maps = sorted(set(powers) - set(fields), key=lambda value: int(value))
    return fields, powers, params, missing_maps


def leave_one_out_baselines(fields, powers, ambient_k):
    case_ids = sorted(fields)
    output = {}
    for held_out in case_ids:
        train_ids = [case_id for case_id in case_ids if case_id != held_out]
        true = fields[held_out]
        entry = {
            "ambient_constant": thermal_metrics(
                true, np.full_like(true, ambient_k), ambient_k
            ),
            "train_global_mean": thermal_metrics(
                true,
                np.full_like(true, float(np.mean([fields[c] for c in train_ids]))),
                ambient_k,
            ),
        }
        if len(train_ids) >= 2:
            predicted_mean = float(total_power_regression_predictions(
                np.array([powers[c] for c in train_ids]),
                np.array([np.mean(fields[c]) for c in train_ids]),
                np.array([powers[held_out]]),
            )[0])
            entry["total_power_regression"] = thermal_metrics(
                true, np.full_like(true, predicted_mean), ambient_k
            )
        output[held_out] = entry
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--mode", default="microchannel")
    parser.add_argument("--ambient-k", type=float, default=293.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    fields, powers, params, missing_maps = load_dataset(args.data_dir, args.mode)
    if args.require_complete and missing_maps:
        raise ValueError(f"dataset is incomplete: {len(missing_maps)} temperature maps missing")
    if len(fields) < 2:
        raise ValueError("audit requires at least two temperature maps")
    report = {
        "schema_version": 1,
        "mode_label_in_source": args.mode,
        "scientific_label": "equivalent_strong_convection"
        if args.mode == "microchannel" else args.mode,
        "source_directory": str(args.data_dir.resolve()),
        "coverage": {
            "parameter_rows": len(params), "temperature_maps": len(fields),
            "missing_temperature_map_count": len(missing_maps),
            "missing_temperature_map_ids": missing_maps,
            "complete": not missing_maps,
        },
        "numeric_parameter_summary": numeric_parameter_summary(params),
        "temperature_summary": dataset_temperature_summary(fields, args.ambient_k),
        "exact_duplicate_groups": duplicate_groups(fields),
        "rounded_3dp_duplicate_groups": duplicate_groups(fields, decimals=3),
        "nearest_temperature_map": nearest_neighbor_distances(fields),
        "leave_one_out_trivial_baselines": leave_one_out_baselines(
            fields, powers, args.ambient_k
        ),
        "limitations": [
            "Only locally available maps were audited.",
            "The source mode uses an equivalent HTC boundary, not native channel flow.",
            "Formal leakage conclusions require the complete workstation dataset.",
        ],
    }
    geometry_groups = {}
    for item in params:
        group = layout_group_id(item.get("blocks", []))
        geometry_groups.setdefault(group, []).append(str(item["id"]))
    report["repeated_layout_groups"] = sorted(
        [sorted(ids) for ids in geometry_groups.values() if len(ids) > 1],
        key=lambda ids: ids[0],
    )
    if args.split:
        split_manifest = json.loads(args.split.read_text(encoding="utf-8"))
        split = split_manifest.get("split", split_manifest)
        train_ids = split.get("train_pool") or next(iter(split["train_subsets"].values()))
        report["cross_split_audit"] = cross_split_audit(
            fields, [str(value) for value in train_ids],
            [str(value) for value in split["test"]],
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({len(fields)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
