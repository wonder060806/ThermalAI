#!/usr/bin/env python3
"""Rebuild equivalent-convection 3D-ICE inputs from an immutable parameter list."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from revision.schema import canonical_json


STACK = """material SILICON :
   thermal conductivity     1.30e-4 ;
   volumetric heat capacity 1.628e-12 ;

top heat sink :
   heat transfer coefficient 5.0e-9 ;
   temperature               293.0 ;

bottom heat sink :
   heat transfer coefficient 5.0e-10 ;
   temperature               293.0 ;

dimensions :
   chip length 1000, width 1000 ;
   cell length 50, width 50 ;
   non-uniform false;

die TOP_IC :
   source 5 SILICON ;
   layer 495 SILICON ;

stack:
   die CHIP TOP_IC floorplan "../case_microchannel_{case_id}.flp" ;

solver:
   steady ;
   initial temperature 293.0 ;
   numofcores 1 ;

output:
   Tmap ( CHIP, "../case_microchannel_{case_id}_temp.txt", final ) ;
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = args.params.read_bytes()
    cases = json.loads(raw.decode("utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stacks = args.output_dir / "stacks"
    stacks.mkdir(exist_ok=True)
    for case in cases:
        case_id = int(case["id"])
        sections = []
        total_power_w = 0.0
        for index, block in enumerate(case["blocks"]):
            power_w = float(block["power_mw"]) / 1000.0
            total_power_w += power_w
            sections.append(
                f"Block{index} :\n  position {block['x']}, {block['y']} ;\n"
                f"  dimension {block['w']}, {block['h']} ;\n"
                f"  power values {power_w:.12f} ;\n"
            )
        expected_w = float(case["total_power_mw"]) / 1000.0
        if abs(total_power_w - expected_w) > 1e-5:
            raise ValueError(f"case {case_id} block power does not match total power")
        (args.output_dir / f"case_microchannel_{case_id}.flp").write_text(
            "\n".join(sections), encoding="utf-8"
        )
        (stacks / f"case_microchannel_{case_id}.stk").write_text(
            STACK.format(case_id=case_id), encoding="utf-8"
        )
    (args.output_dir / "cases_microchannel_params.json").write_bytes(raw)
    manifest = {
        "schema_version": 1, "case_count": len(cases),
        "source_params": str(args.params.resolve()),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "scientific_label": "equivalent_strong_convection",
        "status": "definitions_only_unvalidated_truth",
    }
    (args.output_dir / "generation_manifest.json").write_text(
        canonical_json(manifest) + "\n", encoding="utf-8"
    )
    print(canonical_json(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
