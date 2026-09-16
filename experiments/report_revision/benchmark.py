#!/usr/bin/env python3
"""Benchmark deployment inference and an optional external simulator command."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from revision.benchmarking import measure_callable, measure_command
from revision.schema import canonical_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--cooling", default="solid")
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--load-repeats", type=int, default=5)
    parser.add_argument("--external-command-json")
    parser.add_argument("--external-cwd", type=Path)
    parser.add_argument("--external-repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np
    import torch

    sys.path.insert(0, str(args.model_dir.resolve()))
    predict_module = importlib.import_module("predict")
    predict_module.DEVICE = args.device
    ThermalAI = predict_module.ThermalAI
    model_path = args.model_path or (args.model_dir / "model_final.pth")

    def load_once():
        instance = ThermalAI(model_path=str(model_path))
        del instance

    model_load = measure_callable(load_once, warmups=0, repeats=args.load_repeats)
    model = ThermalAI(model_path=str(model_path))
    power_map = np.ones((21, 21), dtype=np.float32)
    synchronize = (
        (lambda: torch.cuda.synchronize(torch.device(args.device)))
        if args.device.startswith("cuda")
        else None
    )
    inference = measure_callable(
        lambda: model.predict(power_map, cooling=args.cooling),
        warmups=args.warmups,
        repeats=args.repeats,
        synchronize=synchronize,
    )

    external = None
    speedup = None
    if args.external_command_json:
        command = json.loads(args.external_command_json)
        if not isinstance(command, list) or not all(isinstance(v, str) for v in command):
            raise ValueError("external command JSON must be a string list")
        external = measure_command(command, args.external_repeats, cwd=args.external_cwd)
        speedup = (
            external["summary"]["median_ms"]
            / inference["summary"]["median_ms"]
        )

    report = {
        "schema_version": 1,
        "device_requested": args.device,
        "device_actual": (
            torch.cuda.get_device_name(torch.device(args.device))
            if args.device.startswith("cuda") else platform.processor() or "CPU"
        ),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "model_path": str(model_path.resolve()),
        "cooling": args.cooling,
        "model_load": model_load,
        "warm_inference": inference,
        "external_end_to_end": external,
        "speedup_vs_external_median": speedup,
        "timing_scope": {
            "model_load": "model construction and checkpoint load",
            "warm_inference": "predict call including CPU-to-device input and device-to-CPU output",
            "external": "process start through process exit, including configured I/O",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
