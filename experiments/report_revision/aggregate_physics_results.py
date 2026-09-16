#!/usr/bin/env python3
"""Aggregate the formal physics-loss sweep against seed-paired baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from revision.aggregate import aggregate_physics_results
from revision.schema import canonical_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for path in sorted(args.input_dir.glob("*.json")):
        if path.resolve() != args.output.resolve():
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and "config" in value:
                records.append(value)
    report = aggregate_physics_results(records)
    report["input_dir"] = str(args.input_dir.resolve())
    report["result_files_read"] = len(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    print(canonical_json(report))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
