#!/usr/bin/env python3
"""Generate a large dataset in the legacy PINN physical domain for sample studies."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from revision.schema import canonical_json
from revision.splits import layout_group_id


STACK = """material TOY :
   thermal conductivity     1.0e-7 ;
   volumetric heat capacity 1.628e-12 ;

top heat sink :
   heat transfer coefficient 0.0 ;
   temperature               298.15 ;

bottom heat sink :
   heat transfer coefficient 5.0e-10 ;
   temperature               298.15 ;

dimensions :
   chip length 1000, width 1000 ;
   cell length 50, width 50 ;
   non-uniform false;

die TOP_IC :
   layer 495 TOY ;
   source 5 TOY ;

stack:
   die CHIP TOP_IC floorplan "../case_{case_id}.flp" ;

solver:
   steady ;
   initial temperature 298.15 ;
   numofcores 1 ;

output:
   Tmap ( CHIP, "../case_{case_id}_temp.txt", final ) ;
"""


def _overlaps(candidate: dict, blocks: list[dict]) -> bool:
    return any(not (
        candidate["x"] + candidate["w"] <= block["x"]
        or block["x"] + block["w"] <= candidate["x"]
        or candidate["y"] + candidate["h"] <= block["y"]
        or block["y"] + block["h"] <= candidate["y"]
    ) for block in blocks)


def _make_blocks(rng: random.Random) -> list[dict]:
    target = rng.randint(4, 8)
    blocks = []
    for _ in range(target):
        placed = False
        for _ in range(500):
            w_cells = rng.randint(2, 6)
            h_cells = rng.randint(2, 6)
            candidate = {
                "x": rng.randint(0, 20 - w_cells) * 50,
                "y": rng.randint(0, 20 - h_cells) * 50,
                "w": w_cells * 50,
                "h": h_cells * 50,
            }
            if not _overlaps(candidate, blocks):
                blocks.append(candidate)
                placed = True
                break
        if not placed:
            raise RuntimeError("could not place non-overlapping legacy blocks")
    return blocks


def _render_floorplan(blocks: list[dict]) -> str:
    sections = []
    for index, block in enumerate(blocks):
        sections.append(
            f"Block{index} :\n"
            f"  position {block['x']}, {block['y']} ;\n"
            f"  dimension {block['w']}, {block['h']} ;\n"
            f"  power values {block['power_w']:.12f} ;\n"
        )
    return "\n".join(sections)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-cases", type=int, default=260)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stacks = args.output_dir / "stacks"
    stacks.mkdir(exist_ok=True)
    rng = random.Random(args.seed)
    used_groups = set()
    cases = []
    for case_id in range(args.num_cases):
        while True:
            blocks = _make_blocks(rng)
            group = layout_group_id(blocks)
            if group not in used_groups:
                used_groups.add(group)
                break
        total_power_mw = rng.uniform(0.5, 5.0)
        raw_weights = [block["w"] * block["h"] * rng.uniform(0.7, 1.3) for block in blocks]
        for block, weight in zip(blocks, raw_weights):
            block["power_mw"] = total_power_mw * weight / sum(raw_weights)
            block["power_w"] = block["power_mw"] / 1000.0
        (args.output_dir / f"case_{case_id}.flp").write_text(_render_floorplan(blocks), encoding="utf-8")
        (stacks / f"case_{case_id}.stk").write_text(STACK.format(case_id=case_id), encoding="utf-8")
        cases.append({
            "id": case_id, "chip_length": 1000, "num_cells": 20,
            "cell_length": 50, "total_thickness": 500, "source_thickness": 5,
            "total_power_mw": total_power_mw, "layout_group": group,
            "blocks": [{key: block[key] for key in ("x", "y", "w", "h", "power_mw")} for block in blocks],
        })
    (args.output_dir / "cases_params.json").write_text(canonical_json(cases) + "\n", encoding="utf-8")
    (args.output_dir / "generation_manifest.json").write_text(canonical_json({
        "schema_version": 2, "study": "sample_efficiency_matched_deepoheat_domain",
        "seed": args.seed, "case_count": len(cases), "status": "definitions_only_unvalidated_truth",
        "stack_alignment": {
            "source": "top 5 um volumetric approximation to DeepOHeat top surface power",
            "cooling": "bottom Robin boundary, h=500 W/(m2 K), ambient=298.15 K",
            "domain": "1 mm x 1 mm x 0.5 mm",
            "note": "toy k=0.1 W/(m K) retained solely to match the pretrained PINN domain",
        },
    }) + "\n", encoding="utf-8")
    print(f"generated {len(cases)} legacy-domain definitions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
