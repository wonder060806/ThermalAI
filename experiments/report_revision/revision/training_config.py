"""Validation for comparable PINN-versus-random training runs."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path


def load_training_selection(
    split_path: Path, train_size: int, data_seed: int | None = None
) -> tuple[list[int], list[int]]:
    manifest = json.loads(split_path.read_text(encoding="utf-8"))
    try:
        split = manifest["split"]
        if data_seed is not None and "replicates" in manifest:
            if str(data_seed) not in manifest["replicates"]:
                raise ValueError(f"split manifest lacks data replicate {data_seed}")
            train_raw = manifest["replicates"][str(data_seed)]["train_subsets"][str(train_size)]
        else:
            train_raw = split["train_subsets"][str(train_size)]
        test_raw = split["test"]
    except KeyError as error:
        raise ValueError(f"split manifest lacks {error.args[0]}") from error
    train_ids = [int(case_id) for case_id in train_raw]
    test_ids = [int(case_id) for case_id in test_raw]
    if set(train_ids) & set(test_ids):
        raise ValueError("training and fixed test selections overlap")
    if len(train_ids) != train_size:
        raise ValueError("selected training subset does not match train_size")
    return train_ids, test_ids


def validate_training_config(config: Mapping[str, object]) -> None:
    method = config.get("method")
    checkpoint = config.get("checkpoint")
    if method not in {"random", "pinn"}:
        raise ValueError("method must be random or pinn")
    if method == "random" and checkpoint:
        raise ValueError("random initialization must not use a checkpoint")
    if method == "pinn" and not checkpoint:
        raise ValueError("pinn initialization requires a checkpoint")
    epochs = int(config.get("epochs", 0))
    learning_rate = float(config.get("learning_rate", 0.0))
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive and finite")
