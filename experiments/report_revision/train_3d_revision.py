#!/usr/bin/env python3
"""Train a DeepONet surrogate on full-volume 3D-ICE VTK temperatures."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from revision.schema import canonical_json, config_hash
from revision.training_config import load_training_selection


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--init-seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-grid-size", type=int, default=21)
    parser.add_argument("--power-per-unit-mw", type=float, default=250.0)
    parser.add_argument("--temperature-ref-k", type=float, default=298.15)
    parser.add_argument("--temperature-scale-k", type=float, default=50.0)
    parser.add_argument("--ambient-k", type=float, default=298.15)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    train_ids, test_ids = load_training_selection(
        args.split, args.train_size, data_seed=args.data_seed
    )
    config = {
        "study": "representative_high_htc_full_volume_3d", "train_size": args.train_size,
        "data_dir": str(args.data_dir.resolve()), "epochs": args.epochs,
        "learning_rate": args.learning_rate, "init_seed": args.init_seed,
        "data_seed": args.data_seed,
        "device": args.device, "model_grid_size": args.model_grid_size,
        "power_per_unit_mw": args.power_per_unit_mw,
        "temperature_ref_k": args.temperature_ref_k,
        "temperature_scale_k": args.temperature_scale_k,
        "ambient_k": args.ambient_k, "split": str(args.split.resolve()),
        "deterministic_algorithms": True,
    }
    result = {"schema_version": 1, "status": "validated" if args.validate_only else "running",
              "config": config, "config_hash": config_hash(config),
              "train_ids": train_ids, "test_ids": test_ids}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(result) + "\n", encoding="utf-8")
    if args.validate_only:
        return 0

    import numpy as np
    import torch

    from revision.dataset import load_3d_case
    from revision.training import evaluate_3d_cases, supervised_3d_epoch

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
    metadata = {int(case["id"]): case for case in params}

    def load(case_id):
        if case_id not in metadata:
            raise ValueError(f"case {case_id} missing from parameters")
        return load_3d_case(
            args.data_dir, case_id, float(metadata[case_id]["chip_length"]),
            args.model_grid_size, args.power_per_unit_mw,
            args.temperature_ref_k, args.temperature_scale_k,
        )

    train_samples = [load(case_id) for case_id in train_ids]
    test_samples = [load(case_id) for case_id in test_ids]
    workspace_root = ROOT.parents[2]
    sys.path.insert(0, str(workspace_root / "DeepOHeat"))
    from src import modules

    inner = modules.DeepONet(
        trunk_in_features=3, trunk_hidden_features=128,
        branch_in_features=args.model_grid_size**2, branch_hidden_features=256,
        inner_prod_features=128, num_trunk_hidden_layers=3,
        num_branch_hidden_layers=7, nonlinearity="silu", freq=2 * torch.pi,
        std=1, freq_trainable=True, device=str(device),
    )

    class Adapter(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, coords, beta):
            return self.wrapped({"coords": coords, "beta": beta})["model_out"]

    model = Adapter(inner).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        epoch_samples = list(train_samples)
        random.Random(args.init_seed * 1_000_003 + epoch).shuffle(epoch_samples)
        history.append(supervised_3d_epoch(model, optimizer, epoch_samples, device))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    evaluation = evaluate_3d_cases(
        model, test_samples, device, args.temperature_ref_k,
        args.temperature_scale_k, args.ambient_k,
    )
    checkpoint = args.output.with_suffix(".pth").resolve()
    torch.save({"model": inner.state_dict(), "config": config,
                "config_hash": result["config_hash"]}, checkpoint)
    result.update({"status": "completed", "history": {"train_mse": history},
                   "test": evaluation, "training_elapsed_seconds": elapsed,
                   "checkpoint_path": str(checkpoint)})
    args.output.write_text(canonical_json(result) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
