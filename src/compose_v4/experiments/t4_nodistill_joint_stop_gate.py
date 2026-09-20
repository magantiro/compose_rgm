"""Lock-first evaluation of the route-free NoDistill joint STOP support gate."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.control.edit_program import EditProgram
from compose_v4.control.edit_program_graph import (
    compile_program_graph,
    execute_program_graph,
)
from compose_v4.control.nodistill_joint_stop import (
    ARMS,
    JointStopConfig,
    generate_candidate_lock,
    strata_census,
)
from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.t4_frozen_program_benchmark import strict_endpoint_scorer
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

SCHEMA = "t4_nodistill_joint_stop_gate_v1"
CONFIG = "configs/t4_nodistill_joint_stop_gate_v1.json"
SOURCE = "configs/t4_no_complete_routes_diagnostic_v1.json"
TEACHERS = "diagnostics/t4_strategy_reset/20260916/route_comparison.json"


def load_contract(root: Path) -> dict:
    payload = unseal(root / CONFIG)
    if payload.get("schema_version") != SCHEMA:
        raise ValueError("unexpected NoDistill joint STOP contract schema")
    for relative, expected in payload["inputs_sha256"].items():
        actual = sha256_file(root / relative)
        if actual != expected:
            raise ValueError(
                f"NoDistill joint STOP input changed: {relative}: {actual} != {expected}"
            )
    config = JointStopConfig(**payload["generation"])
    if asdict(config) != payload["generation"]:
        raise ValueError("NoDistill joint STOP generation policy changed")
    if tuple(payload["arms"]) != ARMS:
        raise ValueError("NoDistill joint STOP arms changed")
    if (
        len(payload["cases"]) < 2
        or sum(row["role"] == "primary" for row in payload["cases"]) != 1
    ):
        raise ValueError("joint STOP gate needs one primary and a contrasting source")
    acceptance = payload["acceptance"]
    required = {
        "exact_replay_precision": 1.0,
        "primary_nonzero_selection_ready_support": True,
        "primary_diverse_selection_ready_support": True,
        "proposed_strictly_better_valid_yield_or_teacher_support": True,
        "proposed_exact_precision_not_lower": True,
        "deterministic_locks": True,
    }
    if any(acceptance.get(key) != value for key, value in required.items()):
        raise ValueError("NoDistill joint STOP acceptance gate changed")
    return payload


def source_for_case(root: Path, case: dict):
    envelope = json.loads((root / SOURCE).read_text())
    cells = envelope["payload"]["cells"]
    row = cells[case["source_cell"]]
    if row["source_state_sha256"] != case["source_state_sha256"]:
        raise ValueError("NoDistill case source-state identity changed")
    return decode_state(row["source_state"])


def generate_case_lock(
    root: Path, case: dict, arm: str, config: JointStopConfig
) -> dict:
    """Wrapper metadata never crosses the graph-only runtime boundary."""
    source = source_for_case(root, case)
    return generate_candidate_lock(source, arm=arm, config=config)


def _heavy_band(delta: int) -> str:
    if delta <= -4:
        return "major_shrink"
    if delta < 0:
        return "shrink"
    if delta == 0:
        return "stable"
    if delta < 4:
        return "grow"
    return "major_grow"


def _cycle_band(delta: int) -> str:
    return "decrease" if delta < 0 else "increase" if delta > 0 else "stable"


def _scale(primitives: int, changed: int) -> str:
    extent = max(primitives, changed)
    return "small" if extent <= 4 else "medium" if extent <= 8 else "large"


def _teacher_signature(row: dict) -> tuple[str, str, str]:
    return (
        _scale(int(row["primitives"]), int(row["changed_original_atoms"])),
        _heavy_band(int(row["net_created"] - row["net_deleted"])),
        _cycle_band(int(row["ring_count_change"])),
    )


def _candidate_signature(row: dict) -> tuple[str, str, str]:
    strata = row["features"]["strata"]
    return strata["scale"], strata["delta_heavy"], strata["delta_cycle"]


def _teacher_support(candidates: list[dict], teachers: list[dict]) -> dict:
    signatures = {_candidate_signature(row) for row in candidates}
    semantic = {
        _candidate_signature(row)
        for row in candidates
        if row["features"]["strata"]["rewrite_mode"] == "mixed"
    }
    coarse_hits = sum(_teacher_signature(row) in signatures for row in teachers)
    semantic_hits = sum(_teacher_signature(row) in semantic for row in teachers)
    denominator = len(teachers)
    return {
        "teacher_rows": denominator,
        "coarse_supported": coarse_hits,
        "coarse_coverage": coarse_hits / denominator if denominator else None,
        "semantic_supported": semantic_hits,
        "semantic_coverage": semantic_hits / denominator if denominator else None,
        "semantic_definition": (
            "coarse scale/delta-heavy/delta-cycle match plus a completed mixed-mode "
            "candidate; no route or endpoint identity match"
        ),
        "retained_interface_teacher_label": "unavailable; not inferred",
    }


def _replay_and_admit(source, lock: dict, scorer) -> tuple[list[dict], dict]:
    rows = []
    exact = 0
    for candidate in lock["candidates"]:
        program = EditProgram.from_payload(candidate["program"])
        product, trace = execute_program_graph(
            source,
            compile_program_graph(program),
            tuple(candidate["assignment"]),
            max_primitives=32,
            max_blocks=8,
        )
        endpoint = canonical_state_key(product)
        replay_ok = (
            endpoint == candidate["endpoint"]
            and trace["endpoint"] == candidate["primitive_trace_endpoint"]
            and identity(trace) == candidate["trace_identity"]
        )
        exact += int(replay_ok)
        admitted = scorer({"smiles": endpoint})
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "endpoint": endpoint,
                "replay_exact": replay_ok,
                "oracle_eligible": bool(admitted["oracle_eligible"]),
                "qed": admitted["qed"],
                "sa": admitted["sa"],
                "similarity": admitted["sim"],
                "endpoint_exclusion_reasons": admitted["endpoint_exclusion_reasons"],
                "features": candidate["features"],
            }
        )
    denominator = len(rows)
    eligible = [row for row in rows if row["oracle_eligible"]]
    return rows, {
        "emitted": denominator,
        "exact_replay": exact,
        "exact_replay_precision": exact / denominator if denominator else None,
        "valid_complete_stop_yield": len(eligible),
        "valid_unique_endpoints": len({row["endpoint"] for row in eligible}),
        "valid_joint_strata": len(
            {tuple(sorted(row["features"]["strata"].items())) for row in eligible}
        ),
    }


def evaluate_published_locks(
    root: Path,
    contract: dict,
    lock_paths: dict[tuple[str, str], Path],
    determinism_checks: dict[tuple[str, str], bool],
) -> dict:
    """Load teacher metadata only after every candidate lock is published."""
    teachers_payload = json.loads((root / TEACHERS).read_text())
    teacher_rows = teachers_payload["routes"]
    case_results = []
    primary_comparison = None
    for case in contract["cases"]:
        source_envelope = json.loads((root / SOURCE).read_text())["payload"]
        source_row = source_envelope["cells"][case["source_cell"]]
        source = decode_state(source_row["source_state"])
        scorer = strict_endpoint_scorer(
            source_row["original_seed"], delta=case["delta"]
        )
        selected_teachers = [
            row for row in teacher_rows if row.get("cell") == case["teacher_cell"]
        ]
        arm_results = {}
        for arm in ARMS:
            path = lock_paths[(case["case_id"], arm)]
            lock = json.loads(path.read_text())
            rows, metrics = _replay_and_admit(source, lock, scorer)
            eligible_candidates = [
                candidate
                for candidate, evaluation in zip(lock["candidates"], rows, strict=True)
                if evaluation["oracle_eligible"]
            ]
            arm_results[arm] = {
                "lock_path": str(path.relative_to(root)),
                "lock_sha256": sha256_file(path),
                "lock_id": lock["lock_id"],
                "lock_strata": strata_census(lock),
                "metrics": metrics,
                "teacher_support_after_lock": _teacher_support(
                    eligible_candidates, selected_teachers
                ),
                "candidates": rows,
            }
        comparison = {
            "valid_yield_delta": (
                arm_results["coverage_joint_stop"]["metrics"][
                    "valid_complete_stop_yield"
                ]
                - arm_results["independent_marginal"]["metrics"][
                    "valid_complete_stop_yield"
                ]
            ),
            "teacher_coarse_delta": (
                arm_results["coverage_joint_stop"]["teacher_support_after_lock"][
                    "coarse_supported"
                ]
                - arm_results["independent_marginal"]["teacher_support_after_lock"][
                    "coarse_supported"
                ]
            ),
            "teacher_semantic_delta": (
                arm_results["coverage_joint_stop"]["teacher_support_after_lock"][
                    "semantic_supported"
                ]
                - arm_results["independent_marginal"]["teacher_support_after_lock"][
                    "semantic_supported"
                ]
            ),
        }
        case_result = {"case": case, "arms": arm_results, "comparison": comparison}
        case_results.append(case_result)
        if case["role"] == "primary":
            primary_comparison = case_result
    if primary_comparison is None:
        raise RuntimeError("NoDistill primary comparison disappeared")
    baseline = primary_comparison["arms"]["independent_marginal"]
    proposed = primary_comparison["arms"]["coverage_joint_stop"]
    deltas = primary_comparison["comparison"]
    exact_precision = all(
        arm_result["metrics"]["exact_replay_precision"] == 1.0
        for case in case_results
        for arm_result in case["arms"].values()
    )
    nonzero = proposed["metrics"]["valid_complete_stop_yield"] > 0
    diverse = proposed["metrics"]["valid_joint_strata"] >= 2
    improvement = any(
        deltas[field] > 0
        for field in (
            "valid_yield_delta",
            "teacher_coarse_delta",
            "teacher_semantic_delta",
        )
    )
    precision_not_lower = (
        proposed["metrics"]["exact_replay_precision"]
        >= baseline["metrics"]["exact_replay_precision"]
    )
    deterministic = all(determinism_checks.values()) and set(determinism_checks) == set(
        lock_paths
    )
    checks = {
        "exact_replay_precision_1": exact_precision,
        "primary_nonzero_selection_ready_support": nonzero,
        "primary_diverse_selection_ready_support": diverse,
        "proposed_strictly_better_valid_yield_or_teacher_support": improvement,
        "proposed_exact_precision_not_lower": precision_not_lower,
        "deterministic_locks": deterministic,
        "unchanged_support": JointStopConfig(**contract["generation"]).max_primitives
        == 32
        and JointStopConfig(**contract["generation"]).max_blocks == 8
        and JointStopConfig(**contract["generation"]).max_active_atoms == 40,
    }
    return {
        "schema_version": SCHEMA,
        "contract_identity": identity(contract),
        "new_oracle_calls": 0,
        "docking_calls": 0,
        "teacher_loaded_only_after_locks": True,
        "cases": case_results,
        "gate": {
            "checks": checks,
            "decision": "PASS" if all(checks.values()) else "FAIL",
            "negative_result_preserved": not all(checks.values()),
        },
    }
