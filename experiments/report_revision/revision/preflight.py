"""Read-only readiness checks for formal workstation experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .vtk_io import layer_temperature_summary, parse_3dice_vtk, validate_multilayer_truth
from .dataset import parse_temperature_map


def validate_3d_dataset_files(
    data_dir: Path, minimum_cases: int = 250, min_mean_span_k: float = 0.1,
    max_k: float = 500.0,
) -> dict[str, Any]:
    params_path = data_dir / "cases_params.json"
    if not params_path.is_file():
        raise ValueError(f"missing dataset manifest: {params_path}")
    rows = json.loads(params_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) < minimum_cases:
        raise ValueError(f"3D dataset requires at least {minimum_cases} cases")
    spans = []
    maxima = []
    for row in rows:
        case_id = int(row["id"])
        floorplan = data_dir / f"case_{case_id}.flp"
        vtk = data_dir / f"case_{case_id}.vtk"
        if not floorplan.is_file() or not vtk.is_file():
            raise ValueError(f"missing 3D case input/truth for case {case_id}")
        field = parse_3dice_vtk(vtk)
        layers = layer_temperature_summary(field["centers_um"], field["temperature_k"])
        gate = validate_multilayer_truth(layers, 3, min_mean_span_k, max_k)
        spans.append(float(gate["layer_mean_span_k"]))
        maxima.append(float(gate["maximum_temperature_k"]))
    return {
        "case_count": len(rows), "minimum_layer_mean_span_k": min(spans),
        "maximum_temperature_k": max(maxima), "all_cases_passed": True,
    }


def validate_dataset_files(
    data_dir: Path, minimum_cases: int = 250, allow_nonuniform_tmap: bool = False
) -> dict[str, Any]:
    params_path = data_dir / "cases_params.json"
    if not params_path.is_file():
        raise ValueError(f"missing dataset manifest: {params_path}")
    params = json.loads(params_path.read_text(encoding="utf-8"))
    if not isinstance(params, list) or len(params) < minimum_cases:
        raise ValueError(f"dataset has {len(params) if isinstance(params, list) else 0} cases; requires {minimum_cases}")
    ids = [int(item["id"]) for item in params]
    if len(ids) != len(set(ids)):
        raise ValueError("dataset manifest contains duplicate case IDs")
    missing = []
    for case_id in ids:
        for suffix in (".flp", "_temp.txt"):
            path = data_dir / f"case_{case_id}{suffix}"
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise ValueError("missing input case files: " + ", ".join(missing[:10]))
    minima = []
    maxima = []
    nonuniform_count = 0
    for case_id in ids:
        path = data_dir / f"case_{case_id}_temp.txt"
        try:
            if allow_nonuniform_tmap and "xyaxis_" in path.read_text(
                encoding="utf-8", errors="replace"
            ).split("\n", 1)[0]:
                field = np.loadtxt(path, comments="%", dtype=np.float64).reshape(-1)
                nonuniform_count += 1
            else:
                field = parse_temperature_map(path)
        except Exception as error:
            raise ValueError(f"invalid temperature map: {path}") from error
        expected_ndim = 1 if allow_nonuniform_tmap and "xyaxis_" in path.read_text(
            encoding="utf-8", errors="replace"
        ).split("\n", 1)[0] else 2
        if field.ndim != expected_ndim or field.size == 0 or not np.isfinite(field).all():
            raise ValueError(f"invalid temperature map: {path}")
        minima.append(float(np.min(field)))
        maxima.append(float(np.max(field)))
    observed_min = min(minima)
    observed_max = max(maxima)
    if observed_min < 250.0 or observed_max > 500.0:
        raise ValueError(f"temperature maps outside physical gate: {observed_min}..{observed_max} K")
    return {"case_count": len(ids), "case_ids": sorted(ids),
            "case_id_min": min(ids), "case_id_max": max(ids),
            "minimum_temperature_k": observed_min, "maximum_temperature_k": observed_max,
            "nonuniform_temperature_map_count": nonuniform_count,
            "spatial_temperature_maps_validated": nonuniform_count == 0}


def _flag_value(command: list[str], flag: str) -> str | None:
    if flag not in command:
        return None
    index = command.index(flag)
    if index + 1 >= len(command):
        raise ValueError(f"job command lacks value for {flag}")
    return command[index + 1]


def validate_matrix_paths(matrix: dict[str, Any]) -> dict[str, Any]:
    jobs = matrix.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("matrix must contain jobs")
    seen = set()
    output_paths = set()
    for job in jobs:
        job_id = str(job.get("job_id", ""))
        command = job.get("command")
        if not job_id or job_id in seen or not isinstance(command, list) or len(command) < 2:
            raise ValueError("jobs require unique job_id and command")
        seen.add(job_id)
        for candidate in command[:2]:
            if not Path(candidate).is_file():
                raise ValueError(f"missing input path for {job_id}: {candidate}")
        for flag in ("--split", "--data-dir", "--checkpoint"):
            value = _flag_value(command, flag)
            if value is not None and not Path(value).exists():
                raise ValueError(f"missing input path for {job_id}: {value}")
        output = _flag_value(command, "--output")
        if output is None:
            raise ValueError(f"job {job_id} lacks --output")
        resolved_output = str(Path(output).resolve())
        if resolved_output in output_paths:
            raise ValueError(f"duplicate output path in matrix: {resolved_output}")
        output_paths.add(resolved_output)
    return {"job_count": len(jobs), "unique_job_ids": len(seen)}
