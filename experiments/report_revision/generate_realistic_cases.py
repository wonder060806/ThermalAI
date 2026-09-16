#!/usr/bin/env python3
"""Generate deterministic engineering-scale heterogeneous 3D-ICE definitions."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from revision.geometry import render_3dice_floorplan, validate_floorplan
from revision.schema import canonical_json


STACK = """material SILICON :
   thermal conductivity     1.30e-4 ;
   volumetric heat capacity 1.628e-12 ;

material TIM :
   thermal conductivity     4.0e-6 ;
   volumetric heat capacity 1.50e-12 ;

top heat sink :
   heat transfer coefficient {htc_w_um2_k} ;
   temperature               298.15 ;

dimensions :
   chip length {chip_um}, width {chip_um} ;
   cell length {cell_um}, width {cell_um} ;
   non-uniform true;

die REALISTIC_DIE :
   layer {tim_thickness_um} TIM ;
   layer {silicon_thickness_um} SILICON ;
   source {source_thickness_um} SILICON ;

stack:
   die CHIP REALISTIC_DIE floorplan "../case_{case_id}.flp" ;

solver:
   steady ;
   initial temperature 298.15 ;
   numofcores 1 ;

output:
   Tmap ( CHIP, "../case_{case_id}_temp.txt", final ) ;
   T3d ( "../case_{case_id}.vtk", final ) ;
"""


def make_case(case_id: int, rng: random.Random, chip_um: int,
              power_min_w: float, power_max_w: float) -> tuple[list[dict], float]:
    step = chip_um // 100
    x1 = rng.randrange(18 * step, 32 * step + 1, step)
    x2 = rng.randrange(43 * step, 57 * step + 1, step)
    x3 = rng.randrange(68 * step, 82 * step + 1, step)
    y1 = rng.randrange(38 * step, 62 * step + 1, step)
    xs = [0, x1, x2, x3, chip_um]
    ys = [0, y1, chip_um]
    total_power = rng.uniform(power_min_w, power_max_w)
    weights = [rng.uniform(0.35, 1.65) for _ in range(8)]
    weighted_areas = []
    for row in range(2):
        for column in range(4):
            weighted_areas.append((xs[column + 1] - xs[column]) * (ys[row + 1] - ys[row]) * weights[row * 4 + column])
    normalizer = sum(weighted_areas)
    blocks = []
    for row in range(2):
        for column in range(4):
            index = row * 4 + column
            width = xs[column + 1] - xs[column]
            height = ys[row + 1] - ys[row]
            blocks.append({
                "name": f"region_{index}", "x_um": xs[column], "y_um": ys[row],
                "width_um": width, "height_um": height,
                "power_w": total_power * weighted_areas[index] / normalizer,
            })
    validate_floorplan(blocks, chip_um, chip_um, total_power)
    return blocks, total_power


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-cases", type=int, default=260)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--chip-size-mm", type=int, default=20)
    parser.add_argument("--power-min-w", type=float, default=150.0)
    parser.add_argument("--power-max-w", type=float, default=500.0)
    args = parser.parse_args()
    if args.num_cases < 1:
        raise ValueError("num-cases must be positive")
    if args.chip_size_mm <= 0 or args.power_min_w <= 0 or args.power_max_w <= args.power_min_w:
        raise ValueError("chip size and ordered positive power range are required")
    chip_um = args.chip_size_mm * 1000
    if chip_um % 20:
        raise ValueError("chip size must produce an integer 20-by-20 simulator grid")
    cell_um = chip_um // 20
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stacks = args.output_dir / "stacks"
    stacks.mkdir(exist_ok=True)
    rng = random.Random(args.seed)
    cases = []
    first_blocks = None
    for case_id in range(args.num_cases):
        blocks, total_power = make_case(
            case_id, rng, chip_um, args.power_min_w, args.power_max_w
        )
        if first_blocks is None:
            first_blocks = blocks
        (args.output_dir / f"case_{case_id}.flp").write_text(
            render_3dice_floorplan(blocks, discretization=(5, 10)), encoding="utf-8"
        )
        (stacks / f"case_{case_id}.stk").write_text(
            STACK.format(case_id=case_id, chip_um=chip_um, cell_um=cell_um,
                         htc_w_um2_k="1.0e-7", tim_thickness_um=100,
                         silicon_thickness_um=50, source_thickness_um=5), encoding="utf-8"
        )
        cases.append({
            "id": case_id, "chip_length": chip_um, "num_cells": 20,
            "cell_length": cell_um, "total_thickness": 155,
            "source_thickness": 5, "total_power_w": total_power,
            "blocks": [{
                "x": b["x_um"], "y": b["y_um"], "w": b["width_um"],
                "h": b["height_um"], "power_w": b["power_w"],
                "power_mw": b["power_w"] * 1000,
            } for b in blocks],
        })
    (args.output_dir / "cases_params.json").write_text(
        canonical_json(cases) + "\n", encoding="utf-8"
    )
    area_cm2 = (args.chip_size_mm / 10.0) ** 2
    manifest = {"schema_version": 2, "seed": args.seed, "case_count": len(cases),
                "chip_um": chip_um, "chip_area_mm2": args.chip_size_mm ** 2,
                "power_range_w": [args.power_min_w, args.power_max_w],
                "scope": "representative_high_htc_liquid_boundary",
                "claims_exact_commercial_package_geometry": False,
                "derived": {"power_density_range_w_cm2": [
                    args.power_min_w / area_cm2, args.power_max_w / area_cm2
                ]},
                "physical_parameters_si": {
                    "silicon_thermal_conductivity_w_m_k": 130.0,
                    "silicon_volumetric_heat_capacity_j_m3_k": 1.628e6,
                    "tim_thermal_conductivity_w_m_k": 4.0,
                    "top_htc_w_m2_k": 1.0e5,
                    "ambient_temperature_k": 298.15,
                    "layer_thickness_um": {"source_silicon": 5, "silicon": 50, "tim": 100},
                },
                "evidence": [
                    {"claim": "GH100 die area 814 mm2 and H100 TDP up to 700 W",
                     "url": "https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/"},
                    {"claim": "MI300X maximum TBP 750 W",
                     "url": "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf"},
                    {"claim": "silicon microchannel cooler apparent HTC about 104000 W/(m2 K)",
                     "url": "https://research.ibm.com/publications/fabrication-and-performance-of-300-mm-wafer-scale-silicon-microchannel-cooler"},
                    {"claim": "3D-ICE is an early-stage compact thermal model validated against liquid-cooled 3D IC measurements",
                     "url": "https://research.ibm.com/publications/3d-ice-a-compact-thermal-model-for-early-stage-design-of-liquid-cooled-ics"},
                ],
                "limitations": [
                    "20 mm square is a representative engineering domain, not an H100/MI300X package replica",
                    "top HTC is an equivalent boundary condition; coolant flow and channel geometry are not resolved",
                    "layer thicknesses are a simplified three-layer stack and require sensitivity analysis",
                    "the eight rectangular heat regions do not reproduce proprietary commercial floorplans",
                ],
                "sensitivity_cases": ["htc_low", "htc_high", "silicon_thin", "silicon_thick"],
                "status": "definitions_only_unvalidated_truth"}
    (args.output_dir / "generation_manifest.json").write_text(
        canonical_json(manifest) + "\n", encoding="utf-8"
    )
    validation = args.output_dir / "validation"
    validation_stacks = validation / "stacks"
    validation_stacks.mkdir(parents=True, exist_ok=True)
    zero_blocks = [dict(block, power_w=0.0) for block in first_blocks]
    variants = {
        "zero": (render_3dice_floorplan(zero_blocks, discretization=(5, 10)), {}),
        "medium": (render_3dice_floorplan(first_blocks, discretization=(5, 10)), {}),
        "fine": (render_3dice_floorplan(first_blocks, discretization=(10, 20)), {}),
        "htc_low": (render_3dice_floorplan(first_blocks, discretization=(5, 10)), {"htc_w_um2_k": "5.0e-8"}),
        "htc_high": (render_3dice_floorplan(first_blocks, discretization=(5, 10)), {"htc_w_um2_k": "2.0e-7"}),
        "silicon_thin": (render_3dice_floorplan(first_blocks, discretization=(5, 10)), {"silicon_thickness_um": 25}),
        "silicon_thick": (render_3dice_floorplan(first_blocks, discretization=(5, 10)), {"silicon_thickness_um": 100}),
    }
    for name, (floorplan, overrides) in variants.items():
        (validation / f"{name}.flp").write_text(floorplan, encoding="utf-8")
        stack_parameters = {"htc_w_um2_k": "1.0e-7", "tim_thickness_um": 100,
                            "silicon_thickness_um": 50, "source_thickness_um": 5}
        stack_parameters.update(overrides)
        (validation_stacks / f"{name}.stk").write_text(
            STACK.format(case_id=0, chip_um=chip_um, cell_um=cell_um,
                         **stack_parameters).replace("case_0", name), encoding="utf-8"
        )
    print(canonical_json(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
