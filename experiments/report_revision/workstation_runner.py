#!/usr/bin/env python3
"""Validated, resumable launcher for formal multi-GPU experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from revision.runner import assign_jobs, execute_assigned_jobs, pending_jobs
from revision.schema import canonical_json, config_hash


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "workstation_artifacts")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    jobs = matrix.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("matrix must contain a non-empty jobs list")
    prepared = []
    for raw_job in jobs:
        job = dict(raw_job)
        if not job.get("job_id") or not isinstance(job.get("command"), list):
            raise ValueError("each job needs job_id and command list")
        job["config_hash"] = config_hash(job)
        prepared.append(job)
    gpu_ids = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    assigned = assign_jobs(prepared, gpu_ids)

    if args.dry_run:
        print(canonical_json({"job_count": len(assigned), "jobs": assigned}))
        return 0

    existing_markers = {}
    if args.artifacts.exists():
        for path in args.artifacts.glob("*.marker.json"):
            marker = json.loads(path.read_text(encoding="utf-8"))
            existing_markers[str(marker.get("job_id"))] = marker
    remaining = pending_jobs(assigned, existing_markers)
    markers = execute_assigned_jobs(remaining, args.artifacts) if remaining else []
    failed = sum(marker["status"] != "completed" for marker in markers)
    summary = {
        "requested": len(assigned),
        "skipped_completed": len(assigned) - len(remaining),
        "executed": len(markers),
        "completed": sum(marker["status"] == "completed" for marker in markers),
        "failed": failed,
        "markers": markers,
    }
    print(canonical_json(summary))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
