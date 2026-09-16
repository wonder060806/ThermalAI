"""Validate a 3D-ICE VTK field before admitting it as multilayer truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from revision.vtk_io import (
    layer_temperature_summary,
    parse_3dice_vtk,
    validate_multilayer_truth,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-layers", type=int, default=3)
    parser.add_argument("--min-mean-span-k", type=float, default=0.1)
    parser.add_argument("--max-k", type=float, default=500.0)
    args = parser.parse_args()

    field = parse_3dice_vtk(args.input)
    layers = layer_temperature_summary(field["centers_um"], field["temperature_k"])
    gate = validate_multilayer_truth(
        layers,
        min_layers=args.min_layers,
        min_mean_span_k=args.min_mean_span_k,
        max_k=args.max_k,
    )
    report = {
        "status": "pass",
        "source_vtk": str(args.input.resolve()),
        "thresholds": {
            "min_layers": args.min_layers,
            "min_mean_span_k": args.min_mean_span_k,
            "max_k": args.max_k,
        },
        "gate": gate,
        "layers": layers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
