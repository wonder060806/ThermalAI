#!/usr/bin/env python3
"""Benchmark a revised surface or full-volume checkpoint against 3D-ICE."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--kind", choices=("surface", "3d"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True,
                        help="Frozen split; the benchmark case must belong to test")
    parser.add_argument("--case-id", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--load-repeats", type=int, default=5)
    parser.add_argument("--external-command-json")
    parser.add_argument("--external-cwd", type=Path)
    parser.add_argument("--external-repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    split_payload = json.loads(args.split.read_text(encoding="utf-8"))
    split = split_payload.get("split", split_payload)
    test_ids = {int(value) for value in split.get("test", [])}
    if args.case_id not in test_ids:
        parser.error(f"case {args.case_id} is not in the frozen test split")

    import numpy as np
    import torch
    from revision.dataset import load_3d_case, load_solid_case

    checkpoint_data = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    saved_config = checkpoint_data.get("config", {})
    saved_study = str(saved_config.get("study", ""))
    if saved_study:
        checkpoint_kind = (
            "3d" if saved_study == "representative_high_htc_full_volume_3d"
            else "surface"
        )
        if checkpoint_kind != args.kind:
            raise ValueError(
                f"checkpoint study {saved_study!r} is {checkpoint_kind}, not {args.kind}"
            )
    grid = int(saved_config.get("model_grid_size", 21))
    power_scale = float(saved_config.get("power_per_unit_mw", 250.0 if args.kind == "3d" else 0.00625))
    temp_ref = float(saved_config.get("temperature_ref_k", 298.15 if args.kind == "3d" else 293.15))
    temp_scale = float(saved_config.get("temperature_scale_k", 50.0 if args.kind == "3d" else 25.0))
    params = json.loads((args.data_dir / "cases_params.json").read_text(encoding="utf-8"))
    meta = {int(row["id"]): row for row in params}[args.case_id]
    loader = load_3d_case if args.kind == "3d" else load_solid_case
    sample = loader(args.data_dir, args.case_id, float(meta["chip_length"]), grid,
                    power_scale, temp_ref, temp_scale)
    device = torch.device(args.device)
    workspace = ROOT.parents[2]
    sys.path.insert(0, str(workspace / "DeepOHeat"))
    from src import modules

    def load_model():
        model = modules.DeepONet(
            trunk_in_features=3, trunk_hidden_features=128,
            branch_in_features=grid**2, branch_hidden_features=256,
            inner_prod_features=128, num_trunk_hidden_layers=3,
            num_branch_hidden_layers=7, nonlinearity="silu", freq=2 * torch.pi,
            std=1, freq_trainable=True, device=str(device),
        ).to(device)
        model.load_state_dict(checkpoint_data.get("model", checkpoint_data))
        model.eval()
        return model

    model_load = measure_callable(lambda: load_model(), 0, args.load_repeats)
    model = load_model()
    if args.kind == "3d":
        coords = torch.as_tensor(sample["coords"], dtype=torch.float32, device=device)
    else:
        axis = torch.linspace(0, 1, grid, device=device)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        coords = torch.column_stack([xx.reshape(-1), yy.reshape(-1),
                                     torch.full((grid**2,), 0.5, device=device)])
    beta_map = torch.as_tensor(sample["power_dimless"], dtype=torch.float32, device=device)
    beta = beta_map.reshape(1, -1).repeat(coords.shape[0], 1)

    def infer():
        with torch.no_grad():
            return model({"coords": coords, "beta": beta})["model_out"]

    synchronize = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else None
    inference = measure_callable(infer, args.warmups, args.repeats, synchronize=synchronize)
    external = None
    speedup = None
    if args.external_command_json:
        command = json.loads(args.external_command_json)
        external = measure_command(command, args.external_repeats, args.external_cwd)
        speedup = external["summary"]["median_ms"] / inference["summary"]["median_ms"]
    report = {
        "schema_version": 1, "kind": args.kind,
        "checkpoint": str(args.checkpoint.resolve()), "case_id": args.case_id,
        "split": str(args.split.resolve()), "case_role": "test",
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "CPU",
        "python_version": platform.python_version(), "torch_version": torch.__version__,
        "model_load": model_load, "warm_inference": inference,
        "external_end_to_end": external, "speedup_vs_external_median": speedup,
        "prediction_points": int(coords.shape[0]),
        "timing_scope": {"inference": "prepared tensors through model output with device synchronization",
                         "external": "process start through exit including simulator I/O"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    print(canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
