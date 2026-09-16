"""Data loading shared by revised training and evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .vtk_io import parse_3dice_vtk


def parse_temperature_map(path: Path) -> np.ndarray:
    header = path.read_text(encoding="utf-8", errors="replace").splitlines()[:3]
    if any("xyaxis_" in line for line in header):
        raise ValueError(
            f"non-uniform temperature map requires its xyaxis coordinates: {path}"
        )
    field = np.loadtxt(path, comments="%", dtype=np.float64)
    if field.ndim == 1:
        side = int(round(np.sqrt(field.size)))
        if side * side == field.size:
            field = field.reshape(side, side)
    if field.ndim != 2 or field.size == 0 or not np.isfinite(field).all():
        raise ValueError(f"invalid temperature map: {path}")
    return field


def _resize_square(field: np.ndarray, output_size: int) -> np.ndarray:
    if field.shape[0] != field.shape[1]:
        raise ValueError("temperature map must be square")
    source_axis = np.linspace(0.0, 1.0, field.shape[0])
    target_axis = np.linspace(0.0, 1.0, output_size)
    yy, xx = np.meshgrid(target_axis, target_axis, indexing="ij")
    interpolator = RegularGridInterpolator(
        (source_axis, source_axis), field, method="linear"
    )
    return interpolator(np.column_stack([yy.ravel(), xx.ravel()])).reshape(
        output_size, output_size
    )


def _parse_floorplan(path: Path) -> list[dict[str, float]]:
    blocks: list[dict[str, float]] = []
    current: dict[str, float] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith(":"):
            if "power_w" in current:
                blocks.append(current)
            current = {}
        elif "position" in line:
            parts = line.replace(";", "").replace(",", " ").split()
            current["x_um"], current["y_um"] = float(parts[1]), float(parts[2])
        elif "dimension" in line:
            parts = line.replace(";", "").replace(",", " ").split()
            current["w_um"], current["h_um"] = float(parts[1]), float(parts[2])
        elif "power values" in line:
            parts = line.replace(";", "").replace(",", " ").split()
            current["power_w"] = float(parts[2])
    if "power_w" in current:
        blocks.append(current)
    required = {"x_um", "y_um", "w_um", "h_um", "power_w"}
    if any(set(block) != required for block in blocks):
        raise ValueError(f"incomplete floorplan block in {path}")
    return blocks


def load_solid_case(
    data_dir: Path,
    case_id: int,
    chip_um: float,
    model_grid_size: int,
    power_per_unit_mw: float,
    temperature_ref_k: float,
    temperature_scale_k: float,
) -> dict[str, np.ndarray | int]:
    """Load and normalize one solid 3D-ICE surface case."""
    if chip_um <= 0 or model_grid_size < 2 or power_per_unit_mw <= 0:
        raise ValueError("chip, grid, and power scale must be positive")
    raw_temperature = parse_temperature_map(data_dir / f"case_{case_id}_temp.txt")
    temperature_k = _resize_square(raw_temperature, model_grid_size)
    blocks = _parse_floorplan(data_dir / f"case_{case_id}.flp")
    power_mw = _power_map(blocks, chip_um, model_grid_size)
    return {
        "case_id": int(case_id),
        "power_dimless": power_mw / power_per_unit_mw,
        "floorplan_total_power_mw": float(sum(block["power_w"] for block in blocks) * 1000.0),
        "temperature_k": temperature_k,
        "temperature_u": (temperature_k - temperature_ref_k) / temperature_scale_k,
    }


def _power_map(blocks, chip_um: float, model_grid_size: int) -> np.ndarray:
    """Rasterize interval powers, then reproduce DeepOHeat's cell-to-node averaging."""
    interval_count = model_grid_size - 1
    cell_um = chip_um / interval_count
    cell_power_mw = np.zeros((interval_count, interval_count), dtype=np.float64)
    for block in blocks:
        block_area = block["w_um"] * block["h_um"]
        if block_area <= 0:
            raise ValueError("floorplan block does not cover a model cell")
        for x_index in range(interval_count):
            x0, x1 = x_index * cell_um, (x_index + 1) * cell_um
            overlap_x = max(0.0, min(x1, block["x_um"] + block["w_um"]) - max(x0, block["x_um"]))
            if overlap_x == 0.0:
                continue
            for y_index in range(interval_count):
                y0, y1 = y_index * cell_um, (y_index + 1) * cell_um
                overlap_y = max(0.0, min(y1, block["y_um"] + block["h_um"]) - max(y0, block["y_um"]))
                if overlap_y:
                    cell_power_mw[x_index, y_index] += (
                        block["power_w"] * 1000.0 * overlap_x * overlap_y / block_area
                    )
    nodal = np.zeros((model_grid_size, model_grid_size), dtype=np.float64)
    counts = np.zeros_like(nodal)
    for x_offset, y_offset in ((0, 0), (1, 0), (0, 1), (1, 1)):
        nodal[x_offset:x_offset + interval_count,
              y_offset:y_offset + interval_count] += cell_power_mw
        counts[x_offset:x_offset + interval_count,
               y_offset:y_offset + interval_count] += 1.0
    return nodal / counts


def load_3d_case(
    data_dir: Path,
    case_id: int,
    chip_um: float,
    model_grid_size: int,
    power_per_unit_mw: float,
    temperature_ref_k: float,
    temperature_scale_k: float,
) -> dict[str, np.ndarray | int]:
    """Load full T3d cell centers and temperatures plus the branch power map."""
    if chip_um <= 0 or model_grid_size < 2 or power_per_unit_mw <= 0:
        raise ValueError("chip, grid, and power scale must be positive")
    field = parse_3dice_vtk(data_dir / f"case_{case_id}.vtk")
    blocks = _parse_floorplan(data_dir / f"case_{case_id}.flp")
    power_mw = _power_map(blocks, chip_um, model_grid_size)
    temperature_k = np.asarray(field["temperature_k"], dtype=np.float64)
    return {
        "case_id": int(case_id),
        "power_dimless": power_mw / power_per_unit_mw,
        "floorplan_total_power_mw": float(sum(block["power_w"] for block in blocks) * 1000.0),
        "coords": np.asarray(field["centers_um"], dtype=np.float64) / chip_um,
        "centers_um": np.asarray(field["centers_um"], dtype=np.float64),
        "temperature_k": temperature_k,
        "temperature_u": (temperature_k - temperature_ref_k) / temperature_scale_k,
    }
