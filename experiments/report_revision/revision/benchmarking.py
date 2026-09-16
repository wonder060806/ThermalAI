"""Reproducible latency measurement primitives."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from .runner import latency_summary


def measure_callable(
    operation: Callable[[], object],
    warmups: int,
    repeats: int,
    synchronize: Callable[[], object] | None = None,
) -> dict[str, object]:
    if warmups < 0 or repeats <= 0:
        raise ValueError("warmups must be non-negative and repeats positive")
    sync = synchronize or (lambda: None)
    for _ in range(warmups):
        operation()
        sync()
    samples_ms = []
    for _ in range(repeats):
        sync()
        started = time.perf_counter_ns()
        operation()
        sync()
        samples_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return {"samples_ms": samples_ms, "summary": latency_summary(samples_ms)}


def measure_command(
    command: Sequence[str],
    repeats: int,
    cwd: Path | None = None,
) -> dict[str, object]:
    if not command or repeats <= 0:
        raise ValueError("command must be non-empty and repeats positive")
    samples_ms = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        completed = subprocess.run(
            list(command), cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        if completed.returncode != 0:
            raise RuntimeError(f"benchmark command returned exit code {completed.returncode}")
        samples_ms.append(elapsed_ms)
    return {"samples_ms": samples_ms, "summary": latency_summary(samples_ms)}

