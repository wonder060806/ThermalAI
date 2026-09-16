#!/usr/bin/env python3
"""Fail-fast readiness gate for the formal four-GPU experiment run."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from revision.preflight import _flag_value, validate_3d_dataset_files, validate_dataset_files, validate_matrix_paths
from revision.schema import canonical_json
from revision.training_config import load_training_selection


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--train-sizes", default=None,
                        help="deprecated compatibility check; job commands are authoritative")
    parser.add_argument("--minimum-cases", type=int, default=250)
    parser.add_argument("--expected-gpus", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-3d", action="store_true")
    args = parser.parse_args()

    dataset = validate_dataset_files(
        args.data_dir, args.minimum_cases, allow_nonuniform_tmap=args.require_3d
    )
    matrix_data = json.loads(args.matrix.read_text(encoding="utf-8"))
    matrix = validate_matrix_paths(matrix_data)
    selections = {}
    dataset_case_ids = set(int(value) for value in dataset["case_ids"])
    requests = set()
    for job in matrix_data["jobs"]:
        command = job["command"]
        split_raw = _flag_value(command, "--split")
        size_raw = _flag_value(command, "--train-size")
        seed_raw = _flag_value(command, "--data-seed")
        if split_raw is None:
            raise ValueError(f"job {job['job_id']} lacks --split")
        if size_raw is None:
            if args.train_sizes:
                continue
            raise ValueError(f"job {job['job_id']} lacks --train-size")
        requests.add((split_raw, int(size_raw), int(seed_raw or 0)))
    if args.train_sizes:
        first_command = matrix_data["jobs"][0]["command"]
        split_raw = _flag_value(first_command, "--split")
        requests.update((split_raw, int(raw), 0) for raw in args.train_sizes.split(","))
    for split_raw, size, data_seed in sorted(requests):
        split_path = Path(split_raw)
        train_ids, test_ids = load_training_selection(split_path, size, data_seed=data_seed)
        missing_split_ids = sorted((set(train_ids) | set(test_ids)) - dataset_case_ids)
        if missing_split_ids:
            raise ValueError(
                f"split references case IDs missing from dataset: {missing_split_ids[:10]}"
            )
        selections[f"{split_path.resolve()}|n={size}|data_seed={data_seed}"] = {
            "train": len(train_ids), "test": len(test_ids)
        }

    cuda = {"required": args.expected_gpus, "available": 0, "devices": []}
    if args.expected_gpus:
        import torch

        cuda["available"] = torch.cuda.device_count()
        cuda["torch_version"] = torch.__version__
        cuda["cuda_version"] = torch.version.cuda
        cuda["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if not torch.cuda.is_available() or torch.cuda.device_count() < args.expected_gpus:
            raise RuntimeError(
                f"requires {args.expected_gpus} CUDA GPUs, found {torch.cuda.device_count()}"
            )

    report = {
        "status": "ready",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {"node": platform.node(), "platform": platform.platform(), "python": sys.version},
        "cuda": cuda,
        "dataset": dataset,
        "matrix": matrix,
        "split_selections": selections,
    }
    if args.require_3d:
        report["dataset_3d"] = validate_3d_dataset_files(
            args.data_dir, args.minimum_cases
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    print(canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
