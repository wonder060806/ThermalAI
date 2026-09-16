"""Validated heterogeneous die floorplans for report-revision experiments."""

from __future__ import annotations

import math


def render_3dice_floorplan(
    blocks: list[dict[str, float | str]],
    discretization: tuple[int, int] = (5, 5),
) -> str:
    """Render validated block data in the 3D-ICE floorplan syntax."""
    rows, columns = discretization
    if rows <= 0 or columns <= 0:
        raise ValueError("discretization values must be positive")
    sections = []
    for block in blocks:
        sections.append(
            f"{block['name']} :\n"
            f"  position {float(block['x_um']):.6f}, {float(block['y_um']):.6f} ;\n"
            f"  dimension {float(block['width_um']):.6f}, {float(block['height_um']):.6f} ;\n"
            "  material SILICON ;\n"
            f"  discretization {rows}, {columns} ;\n"
            f"  power values {float(block['power_w']):.12f} ;\n"
        )
    return "\n".join(sections)


def build_heterogeneous_floorplan(
    chip_width_um: float,
    chip_height_um: float,
    total_power_w: float,
) -> list[dict[str, float | str]]:
    """Build an 8-region accelerator-like layout with nonuniform power density."""
    if chip_width_um <= 0 or chip_height_um <= 0 or total_power_w <= 0:
        raise ValueError("chip dimensions and total power must be positive")
    labels = ["compute_0", "compute_1", "cache_0", "io_0", "compute_2", "compute_3", "cache_1", "io_1"]
    weights = [1.45, 1.30, 0.65, 0.35, 1.25, 1.50, 0.70, 0.30]
    weight_sum = sum(weights)
    cell_width = chip_width_um / 4.0
    cell_height = chip_height_um / 2.0
    blocks: list[dict[str, float | str]] = []
    for index, (label, weight) in enumerate(zip(labels, weights)):
        column = index % 4
        row = index // 4
        blocks.append({
            "name": label,
            "x_um": column * cell_width,
            "y_um": row * cell_height,
            "width_um": cell_width,
            "height_um": cell_height,
            "power_w": total_power_w * weight / weight_sum,
        })
    return blocks


def validate_floorplan(
    blocks: list[dict[str, float | str]],
    chip_width_um: float,
    chip_height_um: float,
    expected_power_w: float,
    tolerance: float = 1e-9,
) -> dict[str, float | int]:
    if not blocks:
        raise ValueError("floorplan must contain blocks")
    total_area = 0.0
    total_power = 0.0
    densities = []
    rectangles = []
    for block in blocks:
        x = float(block["x_um"])
        y = float(block["y_um"])
        width = float(block["width_um"])
        height = float(block["height_um"])
        power = float(block["power_w"])
        if width <= 0 or height <= 0 or power < 0:
            raise ValueError("block dimensions must be positive and power nonnegative")
        if x < 0 or y < 0 or x + width > chip_width_um + tolerance or y + height > chip_height_um + tolerance:
            raise ValueError("floorplan block exceeds chip bounds")
        rectangle = (x, y, x + width, y + height)
        for previous in rectangles:
            overlap_x = min(rectangle[2], previous[2]) - max(rectangle[0], previous[0])
            overlap_y = min(rectangle[3], previous[3]) - max(rectangle[1], previous[1])
            if overlap_x > tolerance and overlap_y > tolerance:
                raise ValueError("floorplan blocks overlap")
        rectangles.append(rectangle)
        area = width * height
        total_area += area
        total_power += power
        densities.append(power / (area / 1_000_000.0))
    chip_area = chip_width_um * chip_height_um
    if not math.isclose(total_area, chip_area, rel_tol=0.0, abs_tol=tolerance * max(1.0, chip_area)):
        raise ValueError("floorplan does not cover the full chip area")
    if not math.isclose(total_power, expected_power_w, rel_tol=0.0, abs_tol=tolerance * max(1.0, expected_power_w)):
        raise ValueError("floorplan total power does not match expected power")
    return {
        "n_blocks": len(blocks),
        "covered_area_um2": total_area,
        "total_power_w": total_power,
        "min_power_density_w_per_mm2": min(densities),
        "max_power_density_w_per_mm2": max(densities),
    }
