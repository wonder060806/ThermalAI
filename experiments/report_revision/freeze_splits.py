#!/usr/bin/env python3
"""Freeze a deterministic layout-group-isolated experiment split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from revision.schema import canonical_json
from revision.splits import layout_group_id, make_nested_group_split, make_replicated_train_subsets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--train-sizes", required=True)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = args.params.read_bytes()
    cases = json.loads(raw.decode("utf-8"))
    case_ids = [str(case["id"]) for case in cases]
    group_ids = [layout_group_id(case.get("blocks", [])) for case in cases]
    train_sizes = [int(value) for value in args.train_sizes.split(",")]
    split = make_nested_group_split(
        case_ids, group_ids, train_sizes, args.test_fraction, args.seed
    )
    manifest = {
        "schema_version": 2,
        "source_params": str(args.params.resolve()),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "layout_group_by_case": dict(zip(case_ids, group_ids)),
        "split": split,
        "replicates": make_replicated_train_subsets(
            split, train_sizes, args.replicates, args.seed + 10_000
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({len(case_ids)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
