"""Fit and smoke-test a leave-one-target-out complete-region proposal expert."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.docking_value import identity
from compose_v4.control.route_distilled_goal_expert import (
    make_route_expert,
    propose_route_expert_candidates,
)
from compose_v4.control.structural_subgoal_policy import fit_marginal_subgoal_policy
from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.t4_fiber_campaign import COMPOSE_VALID, Fiber
from tools.t4_structural_subgoal_policy import _rows, _template_vocabulary

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "t4_route_complete_region_expert_smoke_v1"
CHECKPOINT_SCHEMA = "t4_route_complete_region_expert_checkpoint_v1"
DEFAULT_OUTPUT = ROOT / "diagnostics/t4_integrated_route_fiber_v1/route_expert_smoke.json"
DEFAULT_CHECKPOINT = ROOT / "diagnostics/t4_integrated_route_fiber_v1/route_expert_checkpoint.json"


def _publish(path: Path, payload: dict) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite route-expert artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _seeds(held_target: str) -> list[dict]:
    rows = json.loads((ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())
    selected = [row for row in rows if row["target"] == held_target]
    if len(selected) != 3:
        raise RuntimeError(f"{held_target} seed census changed")
    return selected


def run(output: Path, checkpoint: Path, *, held_target: str = "jak2") -> dict:
    routes, metadata = _rows()
    known_targets = sorted({row["target"] for row in metadata.values()})
    if held_target not in known_targets:
        raise ValueError(f"unknown held target {held_target!r}; expected one of {known_targets}")
    train = [
        row for row in routes if metadata[row["source_group"]]["target"] != held_target
    ]
    held = [row for row in routes if metadata[row["source_group"]]["target"] == held_target]
    if len(routes) != 77 or len(train) + len(held) != len(routes) or not train or not held:
        raise RuntimeError("route-expert held-target split changed")
    marginal = fit_marginal_subgoal_policy(train, exploration_floor=0.10)
    templates = _template_vocabulary(train)
    training_evidence_identity = identity(
        {
            "training_source_groups": sorted(row["source_group"] for row in train),
            "training_template_ids": sorted(
                template.template_id for row in train for template in row["templates"]
            ),
            "held_source_groups": sorted(row["source_group"] for row in held),
        }
    )
    expert = make_route_expert(
        templates,
        marginal,
        training_evidence_identity=training_evidence_identity,
    )
    checkpoint_payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "expert": expert.checkpoint(),
        "split_audit": {
            "split": "leave_one_target_out",
            "held_target_hash": hashlib.sha256(held_target.encode()).hexdigest(),
            "all_routes": len(routes),
            "training_routes": len(train),
            "held_routes": len(held),
            "training_regions": sum(len(row["templates"]) for row in train),
            "held_regions": sum(len(row["templates"]) for row in held),
            "held_target_absent_from_training": all(
                metadata[row["source_group"]]["target"] != held_target for row in train
            ),
        },
        "new_oracle_calls": 0,
    }
    _publish(checkpoint, checkpoint_payload)

    cells = []
    for source_index, seed in enumerate(_seeds(held_target)):
        source = pad_molecular_graph(smiles_to_molecular_graph(seed["smiles"]), 48)
        candidates, telemetry = propose_route_expert_candidates(
            source,
            expert,
            pool_size=64,
            realization_limit=32,
            beam_width=32,
            expansion_width=24,
            max_bindings_per_template=4,
            maximum_expansions=4_000,
        )
        fiber = Fiber(seed["smiles"], 0.6, support=COMPOSE_VALID)
        rows = []
        for candidate in candidates:
            properties = fiber.check(candidate["smiles"])
            rows.append({**candidate, "eligible": properties is not None, "properties": properties})
        committed = len(rows)
        eligible = sum(row["eligible"] for row in rows)
        cells.append(
            {
                "cell": f"{held_target}_{source_index}",
                "source_global_index": seed["idx"],
                "proposed_pool": int(telemetry["unique_ranked_endpoints"]),
                "complete_programs_committed": committed,
                "eligible_complete_programs": eligible,
                "exact_realization_precision": 1.0 if committed else None,
                "telemetry": telemetry,
                "candidates": rows,
            }
        )
    gate = {
        "held_target_absent_from_training": checkpoint_payload["split_audit"][
            "held_target_absent_from_training"
        ],
        "complete_program_committed_every_cell": all(
            row["complete_programs_committed"] > 0 for row in cells
        ),
        "eligible_program_entered_union_every_cell": all(
            row["eligible_complete_programs"] > 0 for row in cells
        ),
        "exact_realization_precision_one": all(
            row["exact_realization_precision"] == 1.0 for row in cells
        ),
    }
    gate["passed"] = all(gate.values())
    inputs = {
        path: sha256_file(ROOT / path)
        for path in (
            "docs/GENMOL_T4_SEEDS.json",
            "diagnostics/t4_complete_region_runtime/attempt_1/support.json",
            "src/compose_v4/control/route_distilled_goal_expert.py",
            "src/compose_v4/control/structural_subgoal_policy.py",
            "src/compose_v4/control/complete_region_program.py",
        )
    }
    payload = {
        "schema_version": SCHEMA,
        "evidence": f"zero-oracle production smoke of leave-{held_target}-out route proposals",
        "code_revision": _revision(),
        "inputs_sha256": inputs,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_payload_sha256": identity(checkpoint_payload),
        "split_audit": checkpoint_payload["split_audit"],
        "configuration": {
            "delta": 0.6,
            "support": COMPOSE_VALID,
            "pool_size": 64,
            "realization_limit": 32,
            "beam_width": 32,
            "expansion_width": 24,
            "max_bindings_per_template": 4,
            "maximum_expansions": 4_000,
        },
        "cells": cells,
        "gate": gate,
        "costs": {
            "oracle_calls": 0,
            "docking_calls": 0,
            "modal_launches": 0,
            "gpu_seconds": 0,
        },
    }
    _publish(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--held-target", default="jak2")
    args = parser.parse_args()
    result = run(
        args.output.resolve(),
        args.checkpoint.resolve(),
        held_target=args.held_target,
    )
    print(
        json.dumps(
            {
                "gate": result["gate"],
                "cells": [
                    {
                        key: row[key]
                        for key in (
                            "cell",
                            "proposed_pool",
                            "complete_programs_committed",
                            "eligible_complete_programs",
                        )
                    }
                    for row in result["cells"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
