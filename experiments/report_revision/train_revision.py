#!/usr/bin/env python3
"""Comparable PINN-initialized and random-initialized training entry point."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from revision.schema import canonical_json, config_hash
from revision.dataset import load_solid_case
from revision.training_config import load_training_selection, validate_training_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--method", choices=("random", "pinn"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--init-seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ambient-k", type=float, default=298.15)
    parser.add_argument("--temperature-ref-k", type=float, default=293.15)
    parser.add_argument("--temperature-scale-k", type=float, default=25.0)
    parser.add_argument("--power-per-unit-mw", type=float, default=0.00625)
    parser.add_argument("--model-grid-size", type=int, default=21)
    parser.add_argument("--physics-variant", choices=("none", "source_free", "surface_flux", "volumetric_source"), default="none")
    parser.add_argument("--physics-weight", type=float, default=0.0)
    parser.add_argument("--physics-points", type=int, default=200)
    parser.add_argument("--boundary-points", type=int, default=60)
    parser.add_argument("--source-scale-per-beta", type=float, default=200.0)
    parser.add_argument("--surface-flux-scale-per-beta", type=float, default=1.0)
    parser.add_argument("--biot", type=float, default=5.0)
    parser.add_argument("--source-top-z", type=float, default=0.5)
    parser.add_argument("--source-thickness-z", type=float, default=0.005)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    config = {
        "method": args.method,
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "init_seed": args.init_seed,
        "data_seed": args.data_seed,
        "train_size": args.train_size,
        "split": str(args.split.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "device": args.device,
        "ambient_k": args.ambient_k,
        "temperature_ref_k": args.temperature_ref_k,
        "temperature_scale_k": args.temperature_scale_k,
        "power_per_unit_mw": args.power_per_unit_mw,
        "model_grid_size": args.model_grid_size,
        "physics_variant": args.physics_variant,
        "physics_weight": args.physics_weight,
        "physics_points": args.physics_points,
        "boundary_points": args.boundary_points,
        "source_scale_per_beta": args.source_scale_per_beta,
        "surface_flux_scale_per_beta": args.surface_flux_scale_per_beta,
        "biot": args.biot,
        "source_top_z": args.source_top_z,
        "source_thickness_z": args.source_thickness_z,
        "deterministic_algorithms": True,
    }
    if not 0.0 <= args.physics_weight <= 1.0:
        raise ValueError("physics-weight must be between zero and one")
    if (args.physics_variant == "none") != (args.physics_weight == 0.0):
        raise ValueError("physics variant none must use weight 0, and non-none variants require positive weight")
    validate_training_config(config)
    train_ids, test_ids = load_training_selection(
        args.split, args.train_size, data_seed=args.data_seed
    )
    result = {
        "schema_version": 1,
        "status": "validated" if args.validate_only else "blocked_before_smoke_test",
        "config_hash": config_hash(config),
        "config": config,
        "train_ids": train_ids,
        "test_ids": test_ids,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(result) + "\n", encoding="utf-8")
    if args.validate_only:
        print(f"validated {args.method} train={len(train_ids)} test={len(test_ids)}")
        return 0

    import json
    import random
    import numpy as np
    import torch

    from revision.physics_training import sample_physics_coordinates
    from revision.training import (
        aggregate_physics_steps, calibrate_physics_reference_scales, evaluate_cases,
        mixed_physics_step, physics_raw_losses, supervised_epoch,
    )

    random.seed(args.init_seed)
    np.random.seed(args.init_seed)
    torch.manual_seed(args.init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.init_seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)

    params = json.loads((args.data_dir / "cases_params.json").read_text(encoding="utf-8"))
    meta_by_id = {int(item["id"]): item for item in params}
    all_ids = train_ids + test_ids
    missing_ids = [case_id for case_id in all_ids if case_id not in meta_by_id]
    if missing_ids:
        raise ValueError(f"case IDs missing from parameters: {missing_ids}")

    def load_selected(case_id):
        return load_solid_case(
            args.data_dir,
            case_id,
            chip_um=float(meta_by_id[case_id]["chip_length"]),
            model_grid_size=args.model_grid_size,
            power_per_unit_mw=args.power_per_unit_mw,
            temperature_ref_k=args.temperature_ref_k,
            temperature_scale_k=args.temperature_scale_k,
        )

    train_samples = [load_selected(case_id) for case_id in train_ids]
    test_samples = [load_selected(case_id) for case_id in test_ids]

    workspace_root = ROOT.parents[2]
    sys.path.insert(0, str(workspace_root / "DeepOHeat"))
    from src import modules

    inner_model = modules.DeepONet(
        trunk_in_features=3,
        trunk_hidden_features=128,
        branch_in_features=args.model_grid_size**2,
        branch_hidden_features=256,
        inner_prod_features=128,
        num_trunk_hidden_layers=3,
        num_branch_hidden_layers=7,
        nonlinearity="silu",
        freq=2 * torch.pi,
        std=1,
        freq_trainable=True,
        device=str(device),
    )
    if args.method == "pinn":
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        inner_model.load_state_dict(checkpoint.get("model", checkpoint))

    class DeepONetAdapter(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, coords, beta):
            return self.inner({"coords": coords, "beta": beta})["model_out"]

        def physics_forward(self, coords, beta):
            result = self.inner({"coords": coords, "beta": beta})
            return result["model_out"], result["model_in"]

    model = DeepONetAdapter(inner_model).to(device)
    axis = torch.linspace(0.0, 1.0, args.model_grid_size, device=device)
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    coords = torch.column_stack([
        grid_x.reshape(-1), grid_y.reshape(-1),
        torch.full((args.model_grid_size**2,), 0.5, device=device),
    ])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    started = time.perf_counter()
    train_mse = []
    physics_history = []
    reference_scales = None
    if args.physics_variant != "none":
        calibration_losses = []
        for sample in train_samples:
            groups = sample_physics_coordinates(
                args.physics_points, args.boundary_points,
                seed=args.init_seed * 1_000_000 + int(sample["case_id"]),
                source_top_z=args.source_top_z,
                source_thickness_z=args.source_thickness_z,
                device=device,
            )
            calibration_losses.append(physics_raw_losses(
                model, sample, coords, groups, args.physics_variant,
                args.source_scale_per_beta, args.biot,
                (args.ambient_k - args.temperature_ref_k) / args.temperature_scale_k,
                args.surface_flux_scale_per_beta,
            ))
        reference_scales = calibrate_physics_reference_scales(calibration_losses)
    for epoch in range(args.epochs):
        epoch_samples = list(train_samples)
        random.Random(args.init_seed * 1_000_003 + epoch).shuffle(epoch_samples)
        if args.physics_variant == "none":
            train_mse.append(supervised_epoch(model, optimizer, epoch_samples, coords))
        else:
            epoch_steps = []
            for sample in epoch_samples:
                groups = sample_physics_coordinates(
                    args.physics_points, args.boundary_points,
                    seed=args.init_seed * 1_000_000 + epoch * 10_000 + int(sample["case_id"]),
                    source_top_z=args.source_top_z,
                    source_thickness_z=args.source_thickness_z,
                    device=device,
                )
                step = mixed_physics_step(
                    model, optimizer, sample, coords, groups,
                    variant=args.physics_variant,
                    physics_weight=args.physics_weight,
                    reference_scales=reference_scales,
                    source_scale_per_beta=args.source_scale_per_beta,
                    biot=args.biot,
                    ambient_u=(args.ambient_k - args.temperature_ref_k) / args.temperature_scale_k,
                    surface_flux_scale_per_beta=args.surface_flux_scale_per_beta,
                )
                reference_scales = step["reference_scales"]
                epoch_steps.append(step)
            train_mse.append(float(np.mean([step["data_loss_raw"] for step in epoch_steps])))
            epoch_summary = aggregate_physics_steps(epoch_steps)
            physics_history.append({"epoch": epoch, **epoch_summary})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started
    test_result = evaluate_cases(
        model,
        test_samples,
        coords,
        temperature_ref_k=args.temperature_ref_k,
        temperature_scale_k=args.temperature_scale_k,
        ambient_k=args.ambient_k,
    )

    checkpoint_path = args.output.with_suffix(".pth").resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": inner_model.state_dict(),
        "config": config,
        "config_hash": result["config_hash"],
    }, checkpoint_path)
    result.update({
        "status": "completed",
        "history": {"train_mse": train_mse, "physics": physics_history,
                    "physics_reference_scales": reference_scales},
        "test": test_result,
        "training_elapsed_seconds": elapsed_seconds,
        "checkpoint_path": str(checkpoint_path),
    })
    args.output.write_text(canonical_json(result) + "\n", encoding="utf-8")
    print(f"completed {args.method} train={len(train_ids)} test={len(test_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
