"""Prepare source-only runtime inputs or run a bounded zero-oracle probe."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.experiments.t4_objective_reset import zero_oracle_probe

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "configs/t4_objective_reset_runtime_v1.json"
CELLS = ("5ht1b_0", "braf_1", "jak2_1", "parp1_0", "fa7_0")


def prepare():
    source = ROOT / "configs/t4_frozen_program_benchmark_v2.json"
    old = unseal(source)
    units = []
    for cell in CELLS:
        unit = next(r for r in old["units"] if r["cell"] == cell and r["replicate"] == 0)
        values = {
            key: unit[key]
            for key in (
                "cell",
                "unit_id",
                "original_seed",
                "oracle_protocol",
                "target",
                "source_idx",
                "controller_seed",
                "docking_seed",
            )
        }
        values["source_state"] = old["cells"][cell]["source_state"]
        values["source_group"] = identity(
            {
                "target": unit["target"],
                "source_idx": unit["source_idx"],
                "seed": unit["original_seed"],
            }
        )
        units.append(values)
    # The authoritative oracle adapter owns boxes; extract literals offline,
    # without importing its Modal image or copying them into controller code.
    oracle = ROOT / "modal_apps/genmol_t4_opt_app.py"
    tree = ast.parse(oracle.read_text())
    box_node = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "BOXES" for t in n.targets)
    )
    boxes = ast.literal_eval(box_node.value)
    payload = {
        "schema_version": "t4_objective_reset_runtime_v1",
        "units": units,
        "delta": 0.4,
        "arms": ["support_control", "objective_search"],
        "pilot_cells": ["braf_1", "jak2_1", "fa7_0"],
        "calls_per_unit": 20,
        "new_call_ceiling": 120,
        "maximum_concurrent_workers": 6,
        "automatic_retries": 0,
        "initial_stored_routes": 0,
        "learned_prior": "not_fitted_not_in_this_revision",
        "runtime_input_sha256": old["runtime_input_sha256"],
        "docking_boxes": boxes,
        "source_registry": {"path": str(source.relative_to(ROOT)), "sha256": sha256_file(source)},
        "oracle_reference": {"path": str(oracle.relative_to(ROOT)), "sha256": sha256_file(oracle)},
    }
    seal(RUNTIME, payload)
    print(
        {
            "runtime": str(RUNTIME),
            "sha256": sha256_file(RUNTIME),
            "units": len(units),
            "scored_launch": False,
            "new_oracle_calls": 0,
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "probe"))
    parser.add_argument("--cell", choices=CELLS)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
        return
    if args.cell is None or args.output is None or not 1 <= args.rounds <= 8:
        parser.error("probe needs --cell, --output and 1–8 rounds")
    capsule = unseal(RUNTIME)
    unit = next(r for r in capsule["units"] if r["cell"] == args.cell)
    zero_oracle_probe(unit, args.output, rounds=args.rounds)


if __name__ == "__main__":
    main()
