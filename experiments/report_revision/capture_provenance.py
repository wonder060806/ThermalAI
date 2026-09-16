#!/usr/bin/env python3
"""Capture exact upstream and binary/checkpoint provenance for formal runs."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from revision.schema import canonical_json


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args], text=True, capture_output=True
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or f"git failed in {repo}")
    return completed.stdout.strip()


def repository_metadata(repo: Path) -> dict[str, object]:
    status_lines = _git(repo, "status", "--short").splitlines()
    remotes = _git(repo, "remote", "-v").splitlines()
    return {
        "path": str(repo.resolve()),
        "commit": _git(repo, "rev-parse", "HEAD"),
        "describe": _git(repo, "describe", "--tags", "--always", "--dirty"),
        "dirty": bool(status_lines),
        "status_entry_count": len(status_lines),
        "remotes": remotes,
    }


def artifact_metadata(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"required provenance artifact missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.workspace_root.resolve()
    report = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "repositories": {
            "3d-ice": repository_metadata(root / "3d-ice"),
            "DeepOHeat": repository_metadata(root / "DeepOHeat"),
        },
        "artifacts": {
            "3d_ice_emulator": artifact_metadata(root / "3d-ice" / "bin" / "3D-ICE-Emulator"),
            "pinn_checkpoint": artifact_metadata(
                root / "DeepOHeat" / "DeepOHeat" / "2d_power_map" / "log"
                / "pinn_pretrain" / "checkpoints" / "model_epoch_2000.pth"
            ),
        },
        "upstream_urls": {
            "3d-ice": "https://github.com/esl-epfl/3d-ice",
            "DeepOHeat": "https://github.com/Cadence-Celsius/DeepOHeat",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    print(canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
