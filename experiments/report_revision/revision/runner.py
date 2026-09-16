"""Scheduling and timing helpers for the remote workstation."""

from __future__ import annotations

import math
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .schema import canonical_json


FAMILY_GPU_SLOT = {
    "small_sample_pinn": 0,
    "small_sample_random": 1,
    "physics_loss": 2,
    "realistic_3d": 3,
}


def build_physics_weight_jobs(
    python_executable: str,
    train_script: str,
    split_path: str,
    data_dir: str,
    checkpoint: str,
    output_dir: str,
    train_size: int,
    epochs: int,
    learning_rate: float,
    weights: Sequence[float] = (0.01, 0.05, 0.1, 0.25, 0.5),
    seeds: Sequence[int] = tuple(range(10)),
) -> list[dict[str, object]]:
    """Build paired historical/corrected physics-loss weight sweeps."""
    settings = [("none", 0.0)] + [
        (variant, float(weight))
        for variant in ("surface_flux", "volumetric_source")
        for weight in weights
    ]
    jobs = []
    for seed in seeds:
        for variant, weight in settings:
            weight_tag = str(weight).replace(".", "p")
            job_id = f"physics-{variant}-w{weight_tag}-n{train_size}-s{seed}"
            command = [
                python_executable, train_script, "--split", split_path,
                "--train-size", str(train_size), "--method", "pinn",
                "--checkpoint", checkpoint, "--epochs", str(epochs),
                "--learning-rate", str(learning_rate), "--init-seed", str(seed),
                "--data-seed", str(seed),
                "--data-dir", data_dir,
                "--output", str(Path(output_dir) / f"{job_id}.json"),
                "--device", "cuda:0",
            ]
            if variant != "none":
                command.extend([
                    "--physics-variant", variant,
                    "--physics-weight", str(weight),
                ])
            jobs.append({
                "job_id": job_id, "family": "physics_loss", "method": "pinn",
                "train_size": train_size, "init_seed": int(seed),
                "data_seed": int(seed),
                "physics_variant": variant, "physics_weight": weight,
                "epochs": epochs, "learning_rate": learning_rate,
                "command": command,
            })
    return jobs


def build_realistic_3d_jobs(
    python_executable: str, train_script: str, split_path: str,
    data_dir: str, output_dir: str, train_size: int, epochs: int,
    learning_rate: float, seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> list[dict[str, object]]:
    jobs = []
    for seed in seeds:
        job_id = f"realistic-3d-n{train_size}-s{seed}"
        jobs.append({
            "job_id": job_id, "family": "realistic_3d", "train_size": train_size,
            "init_seed": int(seed), "data_seed": int(seed),
            "epochs": epochs, "learning_rate": learning_rate,
            "command": [
                python_executable, train_script, "--split", split_path,
                "--train-size", str(train_size), "--data-dir", data_dir,
                "--epochs", str(epochs), "--learning-rate", str(learning_rate),
                "--init-seed", str(seed),
                "--data-seed", str(seed),
                "--output", str(Path(output_dir) / f"{job_id}.json"),
                "--device", "cuda:0", "--power-per-unit-mw", "250.0",
            ],
        })
    return jobs


def build_small_sample_jobs(
    python_executable: str,
    train_script: str,
    split_path: str,
    data_dir: str,
    checkpoint: str,
    output_dir: str,
    epochs: int,
    learning_rate: float,
    power_per_unit_mw: float = 0.00625,
) -> list[dict[str, object]]:
    """Build paired PINN/random jobs for the approved sample-size design."""
    jobs: list[dict[str, object]] = []
    for train_size in (10, 20, 50, 100, 200):
        seed_count = 10 if train_size in (10, 20) else 5
        for init_seed in range(seed_count):
            for method in ("pinn", "random"):
                job_id = f"small-{method}-n{train_size}-s{init_seed}"
                output = str(Path(output_dir) / f"{job_id}.json")
                command = [
                    python_executable,
                    train_script,
                    "--split", split_path,
                    "--train-size", str(train_size),
                    "--method", method,
                    "--epochs", str(epochs),
                    "--learning-rate", str(learning_rate),
                    "--init-seed", str(init_seed),
                    "--data-seed", str(init_seed),
                    "--data-dir", data_dir,
                    "--output", output,
                    "--device", "cuda:0",
                    "--power-per-unit-mw", str(power_per_unit_mw),
                ]
                if method == "pinn":
                    command.extend(["--checkpoint", checkpoint])
                jobs.append({
                    "job_id": job_id,
                    "family": f"small_sample_{method}",
                    "method": method,
                    "train_size": train_size,
                    "init_seed": init_seed,
                    "data_seed": init_seed,
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "power_per_unit_mw": power_per_unit_mw,
                    "command": command,
                })
    return jobs


def latency_summary(samples_ms: Sequence[float]) -> dict[str, float | int]:
    values = np.asarray(samples_ms, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("latency samples must be a non-empty finite sequence")
    if np.any(values < 0):
        raise ValueError("latency samples cannot be negative")
    ordered = np.sort(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": int(values.size),
        "median_ms": float(np.median(values)),
        "p95_ms": float(ordered[p95_index]),
        "min_ms": float(ordered[0]),
        "max_ms": float(ordered[-1]),
    }


def assign_jobs(
    jobs: Sequence[Mapping[str, object]], gpu_ids: Sequence[int]
) -> list[dict[str, object]]:
    if not gpu_ids:
        raise ValueError("at least one GPU ID is required")
    assigned = []
    for index, job in enumerate(jobs):
        item = dict(job)
        gpu_id = gpu_ids[index % len(gpu_ids)]
        item["gpu_id"] = int(gpu_id)
        assigned.append(item)
    return assigned


def pending_jobs(
    jobs: Sequence[Mapping[str, object]],
    markers: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    pending = []
    for raw_job in jobs:
        job = dict(raw_job)
        job_id = str(job["job_id"])
        marker = markers.get(job_id, {})
        completed = (
            marker.get("status") == "completed"
            and marker.get("config_hash") == job.get("config_hash")
        )
        if not completed:
            pending.append(job)
    return pending


def run_job(job: Mapping[str, object], artifact_dir: Path) -> dict[str, object]:
    """Run one configured process with an isolated GPU and durable artifacts."""
    job_id = str(job.get("job_id", ""))
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", job_id):
        raise ValueError("job_id may contain only letters, digits, dot, dash, underscore")
    command = job.get("command")
    if not isinstance(command, list) or not command or not all(
        isinstance(item, str) for item in command
    ):
        raise ValueError("job command must be a non-empty string list")
    if "config_hash" not in job or "gpu_id" not in job:
        raise ValueError("job requires config_hash and gpu_id")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_path = artifact_dir / f"{job_id}.log"
    marker_path = artifact_dir / f"{job_id}.marker.json"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(job["gpu_id"])
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=str(job["cwd"]) if job.get("cwd") else None,
            env=environment,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    marker = {
        "job_id": job_id,
        "config_hash": str(job["config_hash"]),
        "gpu_id": int(job["gpu_id"]),
        "status": "completed" if completed.returncode == 0 else "failed",
        "exit_code": int(completed.returncode),
        "elapsed_seconds": float(time.time() - started),
        "log_path": str(log_path.resolve()),
    }
    temporary = marker_path.with_suffix(marker_path.suffix + ".tmp")
    temporary.write_text(canonical_json(marker) + "\n", encoding="utf-8")
    temporary.replace(marker_path)
    return marker


def gpu_queues(
    assigned_jobs: Sequence[Mapping[str, object]],
) -> dict[int, list[dict[str, object]]]:
    """Group jobs into stable per-device serial queues."""
    queues: dict[int, list[dict[str, object]]] = defaultdict(list)
    for job in assigned_jobs:
        if "gpu_id" not in job:
            raise ValueError("assigned job lacks gpu_id")
        queues[int(job["gpu_id"])].append(dict(job))
    return dict(queues)


def execute_assigned_jobs(
    assigned_jobs: Sequence[Mapping[str, object]], artifact_dir: Path
) -> list[dict[str, object]]:
    """Run one serial queue per GPU while GPUs execute concurrently."""
    queues = gpu_queues(assigned_jobs)

    def run_queue(queue):
        return [run_job(job, artifact_dir) for job in queue]

    markers: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, len(queues))) as executor:
        for queue_markers in executor.map(run_queue, queues.values()):
            markers.extend(queue_markers)
    return markers
