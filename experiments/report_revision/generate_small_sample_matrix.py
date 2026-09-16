#!/usr/bin/env python3
"""Generate the approved paired 70-job sample-efficiency matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from revision.runner import build_small_sample_jobs
from revision.schema import canonical_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--train-script", default=str(ROOT / "train_revision.py"))
    parser.add_argument("--split", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--matrix-output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--power-per-unit-mw", type=float, default=0.00625)
    args = parser.parse_args()

    jobs = build_small_sample_jobs(
        python_executable=args.python,
        train_script=args.train_script,
        split_path=args.split,
        data_dir=args.data_dir,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        power_per_unit_mw=args.power_per_unit_mw,
    )
    matrix = {
        "schema_version": 2,
        "study": "small_sample_pinn_vs_random",
        "paired_by": ["train_size", "data_seed", "init_seed"],
        "jobs": jobs,
    }
    args.matrix_output.parent.mkdir(parents=True, exist_ok=True)
    args.matrix_output.write_text(canonical_json(matrix) + "\n", encoding="utf-8")
    print(f"wrote {args.matrix_output} ({len(jobs)} jobs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
