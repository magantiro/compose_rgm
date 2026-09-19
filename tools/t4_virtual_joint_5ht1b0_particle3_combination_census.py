"""Census every depth-three STOP combination selected by binding particle 3."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import rdkit

from compose_v4.control import virtual_joint_region_proposer
from compose_v4.control.docking_value import identity
from compose_v4.control.virtual_joint_region_proposer import (
    VirtualJointRegionBudgets,
    _bound_constituents,
    _PlanningPrefix,
    _valid_stop_target,
)
from compose_v4.rewrite.kernel import canonical_state_key
from tools.t4_virtual_joint_5ht1b0_particle_gate import (
    EXPECTED_BUDGETS,
    _compact_bound_census,
    _sealed,
    _teacher_constituents,
)
from tools.t4_virtual_joint_5ht1b0_support_gate import (
    CHECKPOINT,
    FORENSICS,
    RECEIPTS,
    ROOT,
    ROUTE_DIRECTORY,
    ROUTE_IDS,
    SHARED_LIBRARY,
    SOURCE_AUDIT,
    SOURCE_GROUP,
    SOURCE_REGISTRY,
    T4_CONTRACT,
    _load_expert,
    _route_inputs,
    _sha256,
)

SCHEMA = "t4_virtual_joint_5ht1b0_particle3_depth3_combination_census_v1"
PARTICLE_INDEX = 3
ALLOCATION = "template_binding_particle"
SEALED_GATE = (
    ROOT / "diagnostics/t4_virtual_joint_5ht1b0_support_gate/attempt_2/result.json"
)
IMPLEMENTATION_COMMIT = "5f9d08203016b7d978fe604f0aa30db09d5361b3"
IMPLEMENTATION_CODE_INPUTS = (
    "src/compose_v4/control/structural_subgoal.py",
    "src/compose_v4/control/structural_subgoal_policy.py",
    "src/compose_v4/control/virtual_joint_region_proposer.py",
)
DIAGNOSTIC_CODE_INPUTS = (
    "tools/t4_virtual_joint_5ht1b0_support_gate.py",
    "tools/t4_virtual_joint_5ht1b0_particle_gate.py",
    "tools/t4_virtual_joint_5ht1b0_particle3_combination_census.py",
)


def _revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _implementation_input_hashes() -> dict[str, str]:
    hashes = {}
    for path in IMPLEMENTATION_CODE_INPUTS:
        content = subprocess.check_output(
            ["git", "show", f"{IMPLEMENTATION_COMMIT}:{path}"], cwd=ROOT
        )
        hashes[path] = hashlib.sha256(content).hexdigest()
    return hashes


def _material_inputs() -> dict[str, str]:
    paths = (
        CHECKPOINT,
        SOURCE_AUDIT,
        FORENSICS,
        SHARED_LIBRARY,
        SOURCE_REGISTRY,
        T4_CONTRACT,
        SEALED_GATE,
        *(ROUTE_DIRECTORY / f"{route_id}.json" for route_id in ROUTE_IDS),
        *(RECEIPTS[route_id] for route_id in ROUTE_IDS),
    )
    return {_relative(path): _sha256(path) for path in paths}


def _teacher_reports(
    routes: list[dict[str, Any]],
    teachers: list[dict[str, Any]],
    ranked_rows: list[dict[str, Any]],
    unique_targets: list[dict[str, Any]],
    *,
    match_constituent_sets: bool = False,
) -> list[dict[str, Any]]:
    all_ranks = {
        row["constituent_keys"]: rank for rank, row in enumerate(ranked_rows, 1)
    }
    valid_ranks = {
        row["constituent_keys"]: rank
        for rank, row in enumerate((row for row in ranked_rows if row["valid"]), 1)
    }
    unique_ranks = {
        row["endpoint_key"]: rank for rank, row in enumerate(unique_targets, 1)
    }
    by_route: dict[str, list[str]] = {route_id: [] for route_id in ROUTE_IDS}
    for row in teachers:
        by_route[row["route_id"]].append(row["constituent_key"])
    reports = []
    for route in routes:
        teacher_keys = tuple(sorted(by_route[route["route_id"]]))
        if match_constituent_sets:
            row = next(
                item
                for item in ranked_rows
                if set(item["constituent_keys"]) == set(teacher_keys)
            )
        else:
            row = next(
                item for item in ranked_rows if item["constituent_keys"] == teacher_keys
            )
        keys = row["constituent_keys"]
        reports.append(
            {
                "route_id": route["route_id"],
                "constituent_keys": list(keys),
                "marginal_log_score": row["marginal_log_score"],
                "all_combination_marginal_rank": all_ranks[keys],
                "valid_combination_marginal_rank": valid_ranks.get(keys),
                "valid_complete_stop_target": row["valid"],
                "exact_teacher_endpoint": row.get("endpoint_key")
                == route["endpoint_key"],
                "unique_valid_target_marginal_rank": unique_ranks.get(
                    route["endpoint_key"]
                ),
                "endpoint_sha256": identity(route["endpoint_key"]),
            }
        )
    return reports


def _combination_census(
    constituents: tuple[Any, ...], source: Any, expert: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    for combined in itertools.combinations(constituents, 3):
        prefix = _PlanningPrefix(
            combined,
            expert.marginal.score(tuple(row.template for row in combined)),
        )
        valid = _valid_stop_target(source, prefix)
        row: dict[str, Any] = {
            "constituent_keys": prefix.keys,
            "marginal_log_score": prefix.log_probability,
            "valid": valid is not None,
        }
        if valid is not None:
            endpoint, _ = valid
            row["endpoint_key"] = canonical_state_key(endpoint)
        rows.append(row)
    ranked_rows = sorted(
        rows,
        key=lambda row: (-row["marginal_log_score"], row["constituent_keys"]),
    )
    valid_rows = [row for row in ranked_rows if row["valid"]]
    best_by_endpoint: dict[str, dict[str, Any]] = {}
    for row in valid_rows:
        best_by_endpoint.setdefault(row["endpoint_key"], row)
    unique_targets = sorted(
        best_by_endpoint.values(),
        key=lambda row: (
            -row["marginal_log_score"],
            row["endpoint_key"],
            row["constituent_keys"],
        ),
    )
    return ranked_rows, valid_rows, unique_targets


def run() -> dict[str, Any]:
    budgets = VirtualJointRegionBudgets()
    if asdict(budgets) != EXPECTED_BUDGETS:
        raise RuntimeError("default virtual-joint budgets changed")
    sealed_gate, sealed_gate_payload_sha256 = _sealed(SEALED_GATE)
    if sealed_gate["implementation_commit_under_test"] != IMPLEMENTATION_COMMIT:
        raise RuntimeError("sealed attempt_2 implementation identity changed")
    implementation_input_hashes = _implementation_input_hashes()
    loaded_proposer_path = Path(virtual_joint_region_proposer.__file__).resolve()
    if (
        _sha256(loaded_proposer_path)
        != implementation_input_hashes[
            "src/compose_v4/control/virtual_joint_region_proposer.py"
        ]
    ):
        raise RuntimeError(
            "loaded virtual-joint proposer does not match implementation commit"
        )
    particle_report = next(
        row
        for row in sealed_gate["particles"]
        if row["particle_index"] == PARTICLE_INDEX
    )

    routes, route_context = _route_inputs()
    expert, checkpoint_payload_sha256 = _load_expert()
    telemetry: Counter = Counter()
    constituents = _bound_constituents(
        route_context["source"],
        expert,
        budgets,
        telemetry,
        constituent_allocation=ALLOCATION,
        binding_particle_index=PARTICLE_INDEX,
    )
    compact_census = _compact_bound_census(telemetry)
    if (
        identity(compact_census) != particle_report["bound_constituent_census_sha256"]
        or len(constituents) != budgets.expansion_width
        or telemetry["bound_constituents"] != particle_report["bound_constituents"]
        or telemetry["binding_visits"]
        != particle_report["budgets"]["binding_visits"]["used"]
    ):
        raise RuntimeError("particle-3 binding census disagrees with sealed attempt_2")

    ordered = tuple(sorted(constituents, key=lambda row: row.constituent_key))
    ranked_rows, valid_rows, unique_targets = _combination_census(
        ordered, route_context["source"], expert
    )
    if len(ranked_rows) != math.comb(budgets.expansion_width, 3):
        raise RuntimeError("depth-three combination census is incomplete")
    selection_order_rows, selection_order_valid, selection_order_unique = (
        _combination_census(constituents, route_context["source"], expert)
    )

    # Teacher identities are joined only after the exhaustive source/expert census.
    teachers = _teacher_constituents(routes)
    teacher_reports = _teacher_reports(routes, teachers, ranked_rows, unique_targets)
    selection_order_teacher_reports = _teacher_reports(
        routes,
        teachers,
        selection_order_rows,
        selection_order_unique,
        match_constituent_sets=True,
    )
    if not all(
        row["valid_complete_stop_target"] and row["exact_teacher_endpoint"]
        for row in teacher_reports
    ):
        raise RuntimeError(
            "a strong-route teacher triple is not a valid exact STOP target"
        )

    first_valid_rank = next(
        rank for rank, row in enumerate(ranked_rows, 1) if row["valid"]
    )
    payload = {
        "schema_version": SCHEMA,
        "scientific_problem": (
            "separate beam-pruning from STOP-target validity for binding particle 3"
        ),
        "evidence": "computed zero-oracle exhaustive combination census",
        "claim_boundary": (
            "this is a post-hoc diagnostic over the sealed particle-3 binding selection, "
            "not a proposer pass, realization pass, budget revision or scored gate"
        ),
        "code_revision": _revision(),
        "implementation_commit_under_test": IMPLEMENTATION_COMMIT,
        "configuration": {
            "particle_index": PARTICLE_INDEX,
            "constituent_allocation": ALLOCATION,
            "selected_constituents": len(ordered),
            "combination_depth": 3,
            "combination_enumeration": "all unordered constituent-key combinations",
            "target_validation": "unchanged _valid_stop_target",
            "marginal_rank_key": "descending joint marginal log score, then constituent keys",
            "realization_enabled": False,
            "budget_widening": False,
            "joint_budgets": asdict(budgets),
            "seed": None,
        },
        "source": {
            "source_group": SOURCE_GROUP,
            "source_state_sha256": route_context["source_state_sha256"],
        },
        "census": {
            "combination_count": len(ranked_rows),
            "expected_combination_count": math.comb(len(ordered), 3),
            "valid_combination_occurrences": len(valid_rows),
            "invalid_combination_occurrences": len(ranked_rows) - len(valid_rows),
            "valid_unique_targets": len(unique_targets),
            "canonical_target_aliases": len(valid_rows) - len(unique_targets),
            "valid_combinations_in_marginal_top_64": sum(
                row["valid"] for row in ranked_rows[: budgets.beam_width]
            ),
            "first_valid_all_combination_marginal_rank": first_valid_rank,
            "valid_combination_rows_sha256": identity(
                [
                    {
                        "constituent_keys": list(row["constituent_keys"]),
                        "endpoint_key": row["endpoint_key"],
                        "marginal_log_score": row["marginal_log_score"],
                    }
                    for row in valid_rows
                ]
            ),
            "unique_target_rows_sha256": identity(
                [
                    {
                        "constituent_keys": list(row["constituent_keys"]),
                        "endpoint_key": row["endpoint_key"],
                        "marginal_log_score": row["marginal_log_score"],
                    }
                    for row in unique_targets
                ]
            ),
        },
        "teacher_targets_posthoc": teacher_reports,
        "nonproduction_selection_order_comparison": {
            "role": (
                "reproduce the prior ad hoc count by retaining particle selection "
                "order inside each combination; production planning instead enforces "
                "monotonically increasing constituent keys"
            ),
            "valid_combination_occurrences": len(selection_order_valid),
            "valid_unique_targets": len(selection_order_unique),
            "canonical_target_aliases": len(selection_order_valid)
            - len(selection_order_unique),
            "teacher_targets_posthoc": selection_order_teacher_reports,
        },
        "sealed_gate_comparison": {
            "path": _relative(SEALED_GATE),
            "payload_sha256": sealed_gate_payload_sha256,
            "particle_3_depth_3_valid_stops": particle_report[
                "valid_stop_targets_depth_2_to_4"
            ]["3"]["valid"],
            "particle_3_planning_beam_width": budgets.beam_width,
            "particle_3_bound_constituent_census_sha256": identity(compact_census),
        },
        "costs": {
            "oracle_calls": 0,
            "docking_calls": 0,
            "modal_launches": 0,
            "realization_attempts": 0,
            "primitive_teacher_actions": 0,
        },
        "material_inputs_sha256": _material_inputs(),
        "material_payload_sha256": {
            _relative(CHECKPOINT): checkpoint_payload_sha256,
            _relative(SEALED_GATE): sealed_gate_payload_sha256,
        },
        "implementation_inputs_sha256": implementation_input_hashes,
        "diagnostic_inputs_sha256": {
            path: _sha256(ROOT / path) for path in DIAGNOSTIC_CODE_INPUTS
        },
        "environment": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform_system": platform.system(),
            "machine": platform.machine(),
            "rdkit": rdkit.__version__,
            "loaded_proposer_source_sha256": _sha256(loaded_proposer_path),
            "precision": "native CPU graph validation; no learned numeric inference",
        },
        "determinism": {
            "proposal_rng": "none",
            "seed": None,
            "serialization": "sorted-key compact JSON with SHA-256 payload identity",
        },
        "limitations": [
            "The three teacher routes are retrospective Full-146 development evidence.",
            "No target was realized, compiled, docked or scored by an objective.",
            "Exhaustive combination validation is a diagnostic and does not change proposer budgets.",
        ],
    }
    return {"payload": payload, "payload_sha256": identity(payload)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    envelope = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(envelope, sort_keys=True, separators=(",", ":")))
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), **envelope}, sort_keys=True))


if __name__ == "__main__":
    main()
