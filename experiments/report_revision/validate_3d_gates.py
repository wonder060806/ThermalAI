#!/usr/bin/env python3
"""Validate zero-power, multilayer signal, and grid convergence VTK controls."""

from __future__ import annotations

import argparse
from pathlib import Path

from revision.schema import canonical_json
from revision.vtk_io import (
    grid_convergence_summary, layer_temperature_summary, parse_3dice_vtk,
    validate_multilayer_truth, validate_zero_power_truth,
)


def layers(path: Path):
    field = parse_3dice_vtk(path)
    return field, layer_temperature_summary(field["centers_um"], field["temperature_k"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero", type=Path, required=True)
    parser.add_argument("--medium", type=Path, required=True)
    parser.add_argument("--fine", type=Path, required=True)
    parser.add_argument("--ambient-k", type=float, default=298.15)
    parser.add_argument("--zero-tolerance-k", type=float, default=0.05)
    parser.add_argument("--convergence-tolerance-k", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    zero_field, _ = layers(args.zero)
    _, medium_layers = layers(args.medium)
    _, fine_layers = layers(args.fine)
    report = {
        "status": "pass",
        "zero_power": validate_zero_power_truth(
            zero_field["temperature_k"], args.ambient_k, args.zero_tolerance_k
        ),
        "medium_multilayer": validate_multilayer_truth(medium_layers, 3, 0.1, 500.0),
        "fine_multilayer": validate_multilayer_truth(fine_layers, 3, 0.1, 500.0),
        "grid_convergence": grid_convergence_summary(
            medium_layers, fine_layers, args.convergence_tolerance_k
        ),
        "files": {"zero": str(args.zero.resolve()), "medium": str(args.medium.resolve()),
                  "fine": str(args.fine.resolve())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    print(canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
