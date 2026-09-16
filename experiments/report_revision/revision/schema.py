"""Stable identifiers and manifests for report-revision experiments."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data deterministically."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def config_hash(config: Mapping[str, Any]) -> str:
    """Return the SHA-256 identity of an experiment configuration."""
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def build_manifest(
    config: Mapping[str, Any],
    splits: Mapping[str, list[str]],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a serializable manifest after validating split isolation."""
    train_cases = set(splits.get("train", []))
    test_cases = set(splits.get("test", []))
    if train_cases & test_cases:
        raise ValueError("train/test split overlap detected")

    config_copy = deepcopy(dict(config))
    manifest = {
        "schema_version": 1,
        "experiment_id": config_hash(config_copy),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config_copy,
        "splits": deepcopy(dict(splits)),
        "runtime": deepcopy(dict(runtime)),
    }
    canonical_json(manifest)
    return manifest

