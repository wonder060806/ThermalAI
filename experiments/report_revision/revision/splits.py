"""Deterministic, leakage-safe splits for sample-efficiency experiments."""

from __future__ import annotations

import math
import random
import hashlib
from collections import defaultdict
from collections.abc import Sequence

from .schema import canonical_json


def layout_group_id(blocks: Sequence[dict[str, object]]) -> str:
    """Identify a geometric layout independently of block order and power."""
    geometry = []
    for block in blocks:
        try:
            geometry.append({name: float(block[name]) for name in ("x", "y", "w", "h")})
        except KeyError as error:
            raise ValueError(f"layout block lacks {error.args[0]}") from error
    geometry.sort(key=lambda item: (item["x"], item["y"], item["w"], item["h"]))
    return hashlib.sha256(canonical_json(geometry).encode("utf-8")).hexdigest()


def make_nested_group_split(
    case_ids: Sequence[str],
    group_ids: Sequence[str],
    train_sizes: Sequence[int],
    test_fraction: float,
    seed: int,
) -> dict[str, object]:
    """Freeze group-isolated test cases and exact nested train subsets."""
    cases = list(case_ids)
    groups = list(group_ids)
    sizes = sorted(set(int(size) for size in train_sizes))
    if len(cases) != len(groups):
        raise ValueError("case_ids and group_ids must have the same length")
    if len(set(cases)) != len(cases):
        raise ValueError("case_ids must be unique")
    if not cases:
        raise ValueError("at least one case is required")
    if not sizes or sizes[0] <= 0:
        raise ValueError("train_sizes must contain positive integers")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between zero and one")

    cases_by_group: dict[str, list[str]] = defaultdict(list)
    for case_id, group_id in zip(cases, groups):
        cases_by_group[group_id].append(case_id)

    rng = random.Random(seed)
    shuffled_groups = list(cases_by_group)
    rng.shuffle(shuffled_groups)
    target_test_count = max(1, math.ceil(len(cases) * test_fraction))
    test_groups: list[str] = []
    test_cases: list[str] = []
    for group_id in shuffled_groups:
        if len(test_cases) >= target_test_count:
            break
        test_groups.append(group_id)
        test_cases.extend(cases_by_group[group_id])

    test_group_set = set(test_groups)
    train_pool = [
        case_id
        for case_id, group_id in zip(cases, groups)
        if group_id not in test_group_set
    ]
    rng.shuffle(train_pool)
    if sizes[-1] > len(train_pool):
        raise ValueError(
            f"largest train size {sizes[-1]} exceeds available pool {len(train_pool)}"
        )

    return {
        "seed": int(seed),
        "test_fraction_requested": float(test_fraction),
        "test_groups": test_groups,
        "test": sorted(test_cases),
        "train_pool": train_pool,
        "train_subsets": {str(size): train_pool[:size] for size in sizes},
    }


def make_replicated_train_subsets(
    fixed_split: dict[str, object], train_sizes: Sequence[int],
    replicate_count: int, seed: int,
) -> dict[str, dict[str, object]]:
    """Resample nested training subsets while preserving one frozen test set."""
    if replicate_count <= 0:
        raise ValueError("replicate_count must be positive")
    sizes = sorted(set(int(size) for size in train_sizes))
    if not sizes or sizes[0] <= 0:
        raise ValueError("train_sizes must contain positive integers")
    pool = list(fixed_split.get("train_pool", []))
    test = set(fixed_split.get("test", []))
    if not pool or set(pool) & test:
        raise ValueError("fixed split must contain a non-overlapping train pool")
    if sizes[-1] > len(pool):
        raise ValueError("largest train size exceeds fixed training pool")
    replicates = {}
    for replicate in range(replicate_count):
        shuffled = list(pool)
        shuffle_seed = seed + replicate * 1_000_003
        random.Random(shuffle_seed).shuffle(shuffled)
        replicates[str(replicate)] = {
            "data_seed": replicate, "shuffle_seed": shuffle_seed,
            "train_subsets": {str(size): shuffled[:size] for size in sizes},
        }
    return replicates
