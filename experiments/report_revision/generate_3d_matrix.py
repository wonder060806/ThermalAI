#!/usr/bin/env python3
"""Generate five-seed full-volume realistic-chip training jobs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from revision.runner import build_realistic_3d_jobs
from revision.schema import canonical_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--train-script", default=str(ROOT / "train_3d_revision.py"))
    parser.add_argument("--split", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--matrix-output", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    jobs = build_realistic_3d_jobs(
        args.python, args.train_script, args.split, args.data_dir,
        args.output_dir, args.train_size, args.epochs, args.learning_rate,
    )
    matrix = {"schema_version": 2, "study": "representative_high_htc_full_volume_3d",
              "paired_by": ["data_seed", "init_seed"], "jobs": jobs}
    args.matrix_output.parent.mkdir(parents=True, exist_ok=True)
    args.matrix_output.write_text(canonical_json(matrix) + "\n", encoding="utf-8")
    print(f"wrote {args.matrix_output} ({len(jobs)} jobs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
