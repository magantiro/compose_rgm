#!/usr/bin/env python3
"""Run one equal-coefficient unique-state canonical-successor diagnostic.

This is a bounded development diagnostic, not a training or P50 authority.  It
reuses the frozen V8 exact states and successor fibers but replaces the source
forensics importance coefficients with one equal coefficient per unique exact
source/time/canonical-successor supervision unit.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
import sys
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.run_editing_t1_successor_gate import (  # noqa: E402
    _execution_panel_from_frozen_authority,
    _load_successor_cache_receipt,
    _sha256,
    _write_if_absent,
)
from compose_v4.chem.persistent_state_identity import (  # noqa: E402
    persistent_slot_state_sha256,
)
from compose_v4.data.active8_trace_inventory import (  # noqa: E402
    load_active8_trace_admission,
)
from compose_v4.experiments.editing_gate_zero_runtime import (  # noqa: E402
    build_scratch_ringcore_model,
    load_frozen_validation_source,
    load_gate_zero_runtime_contract,
)
from compose_v4.experiments.editing_p50_gate import (  # noqa: E402
    state_dict_semantic_sha256,
)
from compose_v4.experiments.editing_t1_panel import (  # noqa: E402
    ACTIVE8_T1_IDENTITY_FIELDS,
    EDITING_T1_UNIQUE_PANEL_KIND,
    editing_t1_panel_capacity_strata_sha256,
    editing_t1_panel_census_sha256,
    editing_t1_panel_selection_sha256,
    validate_charge_policy_exclusions,
)
from compose_v4.experiments.editing_t1_successor_cache import (  # noqa: E402
    load_t1_successor_cache,
    t1_selected_trace_set_sha256,
)
from compose_v4.experiments.editing_t1_successor_runtime import (  # noqa: E402
    EDITING_T1_V4_CAPACITY_CENSUS_RELATIVE_PATH,
    EDITING_T1_V4_PANEL_RELATIVE_PATH,
    EDITING_T1_V8_CONTRACT_RELATIVE_PATH,
    EditingT1RuntimeError,
    build_t1_successor_cache_identity,
    load_editing_t1_launch_authority,
    materialize_t1_panel,
    resolve_t1_panel_view,
    validate_t1_active8_runtime_binding,
)
from compose_v4.experiments.ringcore_successor_leaderboard import (  # noqa: E402
    load_json_object,
)
from compose_v4.experiments.ringcore_validation_panel import (  # noqa: E402
    validate_validation_panel_artifact,
)
from compose_v4.experiments.successor_micro_overfit import (  # noqa: E402
    RINGCORE_EDITING_FAMILIES,
    PreparedSuccessorPanel,
    prepare_cached_successor_panel,
    train_successor_micro_panel,
)

UNIFORM_CONTRACT_SCHEMA = "compose.editing.t1_uniform_successor_capacity_contract"
UNIFORM_CONTRACT_VERSION = 1
UNIFORM_RESULT_SCHEMA = "compose.editing.t1_uniform_successor_capacity_result"
UNIFORM_RESULT_VERSION = 1
UNIFORM_RESULT_STATUS = "UNIFORM_CANONICAL_SUCCESSOR_CAPACITY_DIAGNOSTIC_COMPLETE_NO_GATE_DECISION"
EXPECTED_UNIFORM_CONTRACT_SHA256 = (
    "2b1c57325d24698f674c3e4b55d2eba881e2115ce9ab3db9570a740569c7221a"
)
EXPECTED_THRESHOLDS = {
    "minimum_unique_state_teacher_successor_top1": 0.95,
    "minimum_unique_state_teacher_successor_probability": 0.8,
    "maximum_unique_state_teacher_successor_nll": 0.22314355131420976,
    "require_every_example_teacher_successor_top1": True,
    "minimum_every_example_teacher_successor_probability": 0.8,
    "minimum_nonzero_gradient_optimizer_steps": 1,
    "require_every_declared_component_nonzero_gradient": True,
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _stable_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_uniform_capacity_contract(path: Path) -> Mapping[str, object]:
    """Load and self-validate the frozen equal-coefficient diagnostic contract."""

    source = Path(path)
    try:
        payload = json.loads(source.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EditingT1RuntimeError(f"uniform-capacity contract is unreadable: {source}") from error
    if not isinstance(payload, dict):
        raise EditingT1RuntimeError("uniform-capacity contract must be one JSON object")
    body = {key: value for key, value in payload.items() if key != "contract_sha256"}
    if (
        payload.get("schema") != UNIFORM_CONTRACT_SCHEMA
        or payload.get("schema_version") != UNIFORM_CONTRACT_VERSION
        or payload.get("status") != "FROZEN_BOUNDED_DEVELOPMENT_DIAGNOSTIC_NO_TRAINING_AUTHORITY"
        or payload.get("training_authorized") is not False
        or payload.get("bounded_p50_authorized") is not False
        or payload.get("contract_sha256") != _stable_sha256(body)
        or payload.get("contract_sha256") != EXPECTED_UNIFORM_CONTRACT_SHA256
    ):
        raise EditingT1RuntimeError(
            "uniform-capacity contract identity, status, or self-hash is invalid"
        )
    families = payload.get("families")
    if tuple(families) != RINGCORE_EDITING_FAMILIES:
        raise EditingT1RuntimeError("uniform-capacity contract family order is not Active8")
    panel_policy = payload.get("panel_policy")
    optimization = payload.get("optimization_law")
    evaluation = payload.get("evaluation")
    if not all(isinstance(value, Mapping) for value in (panel_policy, optimization, evaluation)):
        raise EditingT1RuntimeError("uniform-capacity contract policy sections are invalid")
    assert isinstance(panel_policy, Mapping)
    assert isinstance(optimization, Mapping)
    assert isinstance(evaluation, Mapping)
    if (
        panel_policy.get("panel_kind") != EDITING_T1_UNIQUE_PANEL_KIND
        or panel_policy.get("examples_per_family") != 64
        or panel_policy.get("duplicate_exact_source_time_policy") != "reject"
        or optimization.get("objective") != "productive_embedded_canonical_successor_nll"
        or optimization.get("scope") != "all"
        or optimization.get("steps") != 500
        or optimization.get("learning_rate") != 0.001
        or optimization.get("weight_decay") != 0.0
        or optimization.get("teacher_rate") != 1.0
        or optimization.get("importance_coefficient") != 1.0
        or optimization.get("hazard_included") is not False
    ):
        raise EditingT1RuntimeError("uniform-capacity contract changes the declared capacity law")
    thresholds = evaluation.get("thresholds")
    if not isinstance(thresholds, Mapping) or dict(thresholds) != EXPECTED_THRESHOLDS:
        raise EditingT1RuntimeError("uniform-capacity thresholds are absent or malformed")
    decision = payload.get("decision_policy")
    if (
        not isinstance(decision, Mapping)
        or decision.get("formal_gate_decision_emitted") is not False
        or decision.get("all_family_local_arms_must_pass_before_joint_shared_model_capacity")
        is not True
        or decision.get("joint_shared_model_capacity_required_before_p50_recipe_freeze") is not True
        or decision.get("family_local_arms_alone_are_insufficient_for_p50") is not True
        or decision.get("corpus_rebuild_default") is not False
    ):
        raise EditingT1RuntimeError("uniform-capacity decision boundary is invalid")
    return payload


def _original_coefficient_audit(weights: tuple[float, ...]) -> dict[str, object]:
    if not weights or any(not math.isfinite(weight) or weight <= 0.0 for weight in weights):
        raise EditingT1RuntimeError("source-panel importance coefficients are invalid")
    weight_sum = sum(weights)
    squared_sum = sum(weight * weight for weight in weights)
    ordered = sorted(weights)
    return {
        "count": len(weights),
        "sum": weight_sum,
        "squared_sum": squared_sum,
        "coefficient_effective_sample_size": weight_sum * weight_sum / squared_sum,
        "minimum": ordered[0],
        "median": float(statistics.median(ordered)),
        "maximum": ordered[-1],
        "count_below_1e_6": sum(weight < 1e-6 for weight in weights),
        "count_below_1e_3": sum(weight < 1e-3 for weight in weights),
    }


def prepare_uniform_unique_successor_panel(
    model: object,
    prepared: PreparedSuccessorPanel,
) -> tuple[PreparedSuccessorPanel, dict[str, object]]:
    """Replace inherited coefficients only after exact unique-unit validation."""

    source_time_keys: set[tuple[str, float]] = set()
    supervision_keys: set[tuple[str, float, str]] = set()
    original_weights: list[float] = []
    uniform_examples = []
    for index, (example, fiber) in enumerate(zip(prepared.examples, prepared.fibers, strict=True)):
        source_digest = persistent_slot_state_sha256(example.state)
        source_time = (source_digest, float(example.time))
        supervision = (*source_time, example.target_key)
        if source_time in source_time_keys:
            raise EditingT1RuntimeError(
                "uniform unique-state panel repeats an exact source/time prediction problem"
            )
        if supervision in supervision_keys:
            raise EditingT1RuntimeError("uniform unique-state panel repeats a supervision unit")
        if fiber.target_key != example.target_key or len(fiber.aliases) < 1:
            raise EditingT1RuntimeError(
                f"uniform canonical-successor fiber is invalid at row {index}"
            )
        source_time_keys.add(source_time)
        supervision_keys.add(supervision)
        original_weights.append(float(example.importance_weight))
        uniform_examples.append(
            replace(
                example,
                teacher_rate=1.0,
                importance_weight=1.0,
                data_lane="frozen_validation_forensics:unique_state:uniform_successor_capacity",
            )
        )
    uniform = prepare_cached_successor_panel(
        model,
        tuple(uniform_examples),
        prepared.fibers,
    )
    return uniform, _original_coefficient_audit(tuple(original_weights))


def _capacity_checks(
    report: Mapping[str, object],
    thresholds: Mapping[str, object],
) -> tuple[dict[str, bool], dict[str, float | int]]:
    final = report.get("final")
    if not isinstance(final, Mapping):
        raise EditingT1RuntimeError("uniform-capacity report lacks final metrics")
    per_example = final.get("per_example")
    if not isinstance(per_example, list) or not per_example:
        raise EditingT1RuntimeError("uniform-capacity report lacks per-example metrics")
    probabilities: list[float] = []
    nlls: list[float] = []
    ranks: list[int] = []
    for row in per_example:
        if not isinstance(row, Mapping):
            raise EditingT1RuntimeError("uniform-capacity per-example metric is malformed")
        probability = float(row["teacher_successor_probability"])
        nll = float(row["teacher_successor_nll"])
        rank = int(row["teacher_successor_rank"])
        if (
            not math.isfinite(probability)
            or not 0.0 <= probability <= 1.0
            or not math.isfinite(nll)
            or nll < 0.0
            or rank <= 0
        ):
            raise EditingT1RuntimeError("uniform-capacity per-example metric is invalid")
        probabilities.append(probability)
        nlls.append(nll)
        ranks.append(rank)
    component_updates = report.get("component_gradient_update_counts")
    missing_components = report.get("required_components_without_gradient")
    if not isinstance(component_updates, Mapping) or not isinstance(missing_components, list):
        raise EditingT1RuntimeError("uniform-capacity gradient evidence is malformed")
    every_component_updated = not missing_components and all(
        type(value) is int and value > 0 for value in component_updates.values()
    )
    checks = {
        "minimum_unique_state_teacher_successor_top1": (
            float(final["teacher_successor_top1_recall"])
            >= float(thresholds["minimum_unique_state_teacher_successor_top1"])
        ),
        "minimum_unique_state_teacher_successor_probability": (
            float(final["teacher_successor_probability"])
            >= float(thresholds["minimum_unique_state_teacher_successor_probability"])
        ),
        "maximum_unique_state_teacher_successor_nll": (
            float(final["canonical_successor_nll"])
            <= float(thresholds["maximum_unique_state_teacher_successor_nll"])
        ),
        "require_every_example_teacher_successor_top1": (
            not bool(thresholds["require_every_example_teacher_successor_top1"])
            or all(rank == 1 for rank in ranks)
        ),
        "minimum_every_example_teacher_successor_probability": (
            min(probabilities)
            >= float(thresholds["minimum_every_example_teacher_successor_probability"])
        ),
        "minimum_nonzero_gradient_optimizer_steps": (
            int(report["optimizer_steps_with_nonzero_gradient"])
            >= int(thresholds["minimum_nonzero_gradient_optimizer_steps"])
        ),
        "require_every_declared_component_nonzero_gradient": (
            not bool(thresholds["require_every_declared_component_nonzero_gradient"])
            or every_component_updated
        ),
    }
    floors: dict[str, float | int] = {
        "minimum_teacher_successor_probability": min(probabilities),
        "maximum_teacher_successor_nll": max(nlls),
        "maximum_teacher_successor_rank": max(ranks),
        "top1_example_count": sum(rank == 1 for rank in ranks),
        "example_count": len(ranks),
    }
    return checks, floors


def validate_uniform_capacity_result(
    payload: object,
    *,
    contract: Mapping[str, object],
    expected_active8_identity: Mapping[str, str] | None = None,
    expected_cache_manifest_receipt: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Validate the durable family-local diagnostic and recompute every check."""

    if not isinstance(payload, dict):
        raise EditingT1RuntimeError("uniform-capacity result must be one JSON object")
    body = {key: value for key, value in payload.items() if key != "result_sha256"}
    if (
        payload.get("schema") != UNIFORM_RESULT_SCHEMA
        or payload.get("schema_version") != UNIFORM_RESULT_VERSION
        or payload.get("status") != UNIFORM_RESULT_STATUS
        or payload.get("training_authorized") is not False
        or payload.get("bounded_p50_authorized") is not False
        or payload.get("gate_decision") is not None
        or payload.get("diagnostic_scope") != "INDEPENDENT_FAMILY_LOCAL_SCRATCH_MODEL"
        or payload.get("uniform_contract_sha256") != contract["contract_sha256"]
        or payload.get("result_sha256") != _stable_sha256(body)
    ):
        raise EditingT1RuntimeError("uniform-capacity result identity or self-hash is invalid")
    for field in (
        "uniform_contract_file_sha256",
        "parent_t1_runtime_contract_sha256",
        "parent_t1_runtime_contract_file_sha256",
        "gate_zero_runtime_contract_sha256",
        "panel_artifact_sha256",
        "panel_selection_sha256",
        "panel_census_sha256",
        "panel_capacity_strata_sha256",
        *ACTIVE8_T1_IDENTITY_FIELDS,
        "final_model_state_sha256",
        "runner_file_sha256",
        "result_sha256",
    ):
        if not _is_sha256(payload.get(field)):
            raise EditingT1RuntimeError(f"uniform-capacity result {field} is not a SHA-256")
    parents = contract["parents"]
    if not isinstance(parents, Mapping) or (
        payload["parent_t1_runtime_contract_sha256"] != parents["runtime_contract_sha256"]
        or payload["parent_t1_runtime_contract_file_sha256"]
        != parents["runtime_contract_file_sha256"]
        or payload["panel_artifact_sha256"] != parents["panel_artifact_sha256"]
        or payload["gate_zero_runtime_contract_sha256"]
        != payload["active8_support_contract_sha256"]
    ):
        raise EditingT1RuntimeError("uniform-capacity result parent bindings are invalid")
    family = payload.get("family")
    if (
        family not in RINGCORE_EDITING_FAMILIES
        or payload.get("panel_kind") != EDITING_T1_UNIQUE_PANEL_KIND
        or payload.get("scope") != "all"
        or payload.get("example_count") != 64
        or payload.get("unique_progress_address_count") != 64
    ):
        raise EditingT1RuntimeError("uniform-capacity result panel identity is invalid")
    source_rows = payload.get("source_row_sha256s")
    if (
        not isinstance(source_rows, list)
        or len(source_rows) != 64
        or len(set(source_rows)) != 64
        or any(not _is_sha256(value) for value in source_rows)
    ):
        raise EditingT1RuntimeError("uniform-capacity source-row identities are invalid")
    operator_identity = payload.get("operator_identity")
    if (
        not isinstance(operator_identity, Mapping)
        or operator_identity.get("enable_ring_system_delete") is not False
        or operator_identity.get("enable_ring_grow_macro") is not False
        or operator_identity.get("enable_cycle_ops") is not True
        or not isinstance(operator_identity.get("operator_capability_fingerprint"), str)
        or not operator_identity["operator_capability_fingerprint"]
    ):
        raise EditingT1RuntimeError("uniform-capacity operator identity is invalid")
    weight_policy = payload.get("weight_policy")
    if (
        not isinstance(weight_policy, Mapping)
        or weight_policy.get("teacher_rate") != 1.0
        or weight_policy.get("importance_coefficient") != 1.0
        or not isinstance(weight_policy.get("original_coefficient_audit"), Mapping)
        or weight_policy["original_coefficient_audit"].get("count") != 64
    ):
        raise EditingT1RuntimeError("uniform-capacity weight policy is invalid")
    report = payload.get("training_report")
    if (
        not isinstance(report, Mapping)
        or report.get("scope") != "all"
        or report.get("steps") != 500
        or report.get("families") != [family]
    ):
        raise EditingT1RuntimeError("uniform-capacity training report is invalid")
    if payload.get("thresholds") != EXPECTED_THRESHOLDS:
        raise EditingT1RuntimeError("uniform-capacity result thresholds changed after freeze")
    checks, floors = _capacity_checks(report, EXPECTED_THRESHOLDS)
    if (
        payload.get("threshold_checks") != checks
        or payload.get("per_example_floor_metrics") != floors
        or payload.get("all_threshold_checks_pass") != all(checks.values())
    ):
        raise EditingT1RuntimeError("uniform-capacity result checks were not recomputed exactly")
    if not isinstance(payload.get("cache_manifest_receipt"), Mapping) or not isinstance(
        payload.get("cache_receipts"), list
    ):
        raise EditingT1RuntimeError("uniform-capacity cache evidence is malformed")
    if not payload["cache_receipts"]:
        raise EditingT1RuntimeError("uniform-capacity cache evidence is empty")
    if expected_cache_manifest_receipt is not None and (
        payload["cache_manifest_receipt"] != dict(expected_cache_manifest_receipt)
    ):
        raise EditingT1RuntimeError("uniform-capacity cache manifest receipt changed")
    if expected_active8_identity is not None and any(
        payload.get(field) != expected_active8_identity[field]
        for field in ACTIVE8_T1_IDENTITY_FIELDS
    ):
        raise EditingT1RuntimeError("uniform-capacity Active8 identity changed")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer-root", type=Path, required=True)
    parser.add_argument(
        "--uniform-contract",
        type=Path,
        default=ROOT / "configs" / "editing_t1_uniform_successor_capacity_v1.json",
    )
    parser.add_argument(
        "--t1-contract",
        type=Path,
        default=ROOT / EDITING_T1_V8_CONTRACT_RELATIVE_PATH,
    )
    parser.add_argument(
        "--gate-zero-contract",
        type=Path,
        default=ROOT / "configs" / "editing_gate_zero_runtime_v2.json",
    )
    parser.add_argument("--active8-inventory", type=Path, required=True)
    parser.add_argument("--active8-inventory-file-sha256", required=True)
    parser.add_argument("--panel", type=Path, default=ROOT / EDITING_T1_V4_PANEL_RELATIVE_PATH)
    parser.add_argument(
        "--forensics",
        type=Path,
        default=(
            ROOT
            / "diagnostics"
            / "coherence"
            / "ringcore_v1_family_forensics_initial_2026-07-30.json.gz"
        ),
    )
    parser.add_argument(
        "--capacity-census",
        type=Path,
        default=ROOT / EDITING_T1_V4_CAPACITY_CENSUS_RELATIVE_PATH,
    )
    parser.add_argument(
        "--leaderboard-config",
        type=Path,
        default=ROOT / "configs" / "ringcore_v1_successor_leaderboard_v1.json",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=(
            ROOT
            / "diagnostics"
            / "coherence"
            / "ringcore_v1_scientific_a7546e2_frozen_inventory.json"
        ),
    )
    parser.add_argument(
        "--semantic-sidecar",
        type=Path,
        default=(
            ROOT
            / "diagnostics"
            / "coherence"
            / "ringcore_v1_validation_semantic_sidecar_2026-07-30.jsonl.gz"
        ),
    )
    parser.add_argument(
        "--semantic-sidecar-manifest",
        type=Path,
        default=(
            ROOT
            / "diagnostics"
            / "coherence"
            / "ringcore_v1_validation_semantic_sidecar_2026-07-30.manifest.json"
        ),
    )
    parser.add_argument(
        "--charge-policy-audit",
        type=Path,
        default=(
            ROOT / "diagnostics" / "coherence" / "packed_charge_policy_audit_v1_2026-07-30.json"
        ),
    )
    parser.add_argument(
        "--charge-policy-exclusions",
        type=Path,
        default=(
            ROOT
            / "diagnostics"
            / "coherence"
            / "packed_charge_policy_exclusions_v1_2026-07-30.json"
        ),
    )
    parser.add_argument("--family", choices=RINGCORE_EDITING_FAMILIES, required=True)
    parser.add_argument("--successor-cache-root", type=Path, required=True)
    parser.add_argument("--successor-cache-receipt", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    uniform_contract = load_uniform_capacity_contract(args.uniform_contract)
    try:
        launch_authority = load_editing_t1_launch_authority(
            contract_path=args.t1_contract,
            panel_path=args.panel,
            capacity_census_path=args.capacity_census,
            expected_active8_inventory_manifest_file_sha256=(args.active8_inventory_file_sha256),
        )
    except EditingT1RuntimeError as error:
        raise SystemExit(str(error)) from error
    t1_contract = launch_authority.contract
    parents = uniform_contract["parents"]
    assert isinstance(parents, Mapping)
    for field, observed in (
        ("runtime_contract_file_sha256", _sha256(args.t1_contract)),
        ("runtime_contract_sha256", t1_contract.sha256),
        ("panel_file_sha256", _sha256(args.panel)),
        ("panel_artifact_sha256", launch_authority.panel["artifact_sha256"]),
        ("forensics_file_sha256", _sha256(args.forensics)),
    ):
        if parents.get(field) != observed:
            raise SystemExit(f"uniform-capacity parent mismatch for {field}")

    gate_zero_contract = load_gate_zero_runtime_contract(args.gate_zero_contract)
    if gate_zero_contract.sha256 != t1_contract.payload["gate_zero_runtime_contract_sha256"]:
        raise SystemExit("uniform-capacity T1/Gate0 contract mismatch")
    active8_admission = load_active8_trace_admission(
        args.active8_inventory,
        expected_manifest_file_sha256=(
            t1_contract.payload["active8_inventory_manifest_file_sha256"]
        ),
        expected_inventory_sha256=t1_contract.payload["active8_inventory_sha256"],
        expected_effective_source_corpus_cache_sha256=(
            t1_contract.payload["active8_effective_source_corpus_cache_sha256"]
        ),
        expected_support_contract_sha256=(t1_contract.payload["active8_support_contract_sha256"]),
    )
    with gzip.open(args.forensics, "rt") as handle:
        forensics = json.load(handle)
    validate_validation_panel_artifact(
        forensics,
        config=load_json_object(args.leaderboard_config),
        inventory=load_json_object(args.inventory),
    )
    source = load_frozen_validation_source(
        contract=gate_zero_contract,
        transfer_root=args.transfer_root,
        sidecar_path=args.semantic_sidecar,
        sidecar_manifest_path=args.semantic_sidecar_manifest,
    )
    charge_policy_audit = load_json_object(args.charge_policy_audit)
    charge_policy_exclusions = load_json_object(args.charge_policy_exclusions)
    capacity_census_file_sha256 = _sha256(args.capacity_census)
    capacity_census = launch_authority.capacity_census
    for field, observed in (
        ("charge_policy_audit_file_sha256", _sha256(args.charge_policy_audit)),
        (
            "charge_policy_exclusions_file_sha256",
            _sha256(args.charge_policy_exclusions),
        ),
        (
            "charge_policy_exclusion_payload_sha256",
            charge_policy_exclusions["payload_sha256"],
        ),
        (
            "charge_policy_source_input_inventory_sha256",
            charge_policy_exclusions["source_input_inventory_sha256"],
        ),
        ("capacity_census_file_sha256", capacity_census_file_sha256),
        ("capacity_census_sha256", capacity_census["census_sha256"]),
    ):
        if t1_contract.payload[field] != observed:
            raise SystemExit(f"uniform-capacity frozen provenance mismatch for {field}")
    excluded_trace_ids = validate_charge_policy_exclusions(
        audit=charge_policy_audit,
        audit_file_sha256=_sha256(args.charge_policy_audit),
        exclusions=charge_policy_exclusions,
        exclusions_file_sha256=_sha256(args.charge_policy_exclusions),
    )
    panel = _execution_panel_from_frozen_authority(
        launch_authority,
        semantic_sidecar_file_sha256=_sha256(args.semantic_sidecar),
        semantic_sidecar_manifest_file_sha256=_sha256(args.semantic_sidecar_manifest),
    )
    active8_identity = validate_t1_active8_runtime_binding(
        contract=t1_contract,
        gate_zero_contract=gate_zero_contract,
        source=source,
        panel=panel,
        active8_admission=active8_admission,
    )

    model, parity = build_scratch_ringcore_model(gate_zero_contract)
    initial_model_state_sha256 = state_dict_semantic_sha256(model.state_dict())
    expected_cache_identity = build_t1_successor_cache_identity(
        contract=t1_contract,
        gate_zero_contract=gate_zero_contract,
        source=source,
        panel=panel,
        model=model,
        active8_admission=active8_admission,
        compiler_device="cpu",
    )
    resolved = resolve_t1_panel_view(
        source=source,
        panel=panel,
        forensics=forensics,
        family=args.family,
        panel_kind=EDITING_T1_UNIQUE_PANEL_KIND,
        max_atoms=int(gate_zero_contract.model["max_atoms"]),
        excluded_trace_ids=excluded_trace_ids,
        active8_admission=active8_admission,
    )
    cache_receipt = _load_successor_cache_receipt(args.successor_cache_receipt)
    expected_trace_set_sha256 = t1_selected_trace_set_sha256(resolved.selected_records)
    if cache_receipt.selected_trace_set_sha256 != expected_trace_set_sha256:
        raise SystemExit("uniform-capacity cache receipt names another exact trace set")
    successor_cache = load_t1_successor_cache(
        args.successor_cache_root,
        cache_receipt,
        expected_identity=expected_cache_identity,
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    model = model.to(device)
    materialized = materialize_t1_panel(
        model,
        source=source,
        panel=panel,
        forensics=forensics,
        family=args.family,
        panel_kind=EDITING_T1_UNIQUE_PANEL_KIND,
        max_atoms=int(gate_zero_contract.model["max_atoms"]),
        excluded_trace_ids=excluded_trace_ids,
        active8_admission=active8_admission,
        successor_cache=successor_cache,
    )
    uniform_panel, original_coefficient_audit = prepare_uniform_unique_successor_panel(
        model,
        materialized.prepared,
    )
    panel_policy = uniform_contract["panel_policy"]
    optimization = uniform_contract["optimization_law"]
    evaluation = uniform_contract["evaluation"]
    assert isinstance(panel_policy, Mapping)
    assert isinstance(optimization, Mapping)
    assert isinstance(evaluation, Mapping)
    if len(uniform_panel.examples) != int(panel_policy["examples_per_family"]):
        raise SystemExit("uniform-capacity panel has the wrong frozen family size")
    report = train_successor_micro_panel(
        model,
        uniform_panel,
        steps=int(optimization["steps"]),
        learning_rate=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
        scope=str(optimization["scope"]),
        seed=int(gate_zero_contract.model["seed"]),
        report_points=tuple(t1_contract.optimization["report_points"]),
    )
    thresholds = evaluation["thresholds"]
    assert isinstance(thresholds, Mapping)
    checks, per_example_floors = _capacity_checks(report, thresholds)
    result = {
        "schema": UNIFORM_RESULT_SCHEMA,
        "schema_version": UNIFORM_RESULT_VERSION,
        "status": UNIFORM_RESULT_STATUS,
        "training_authorized": False,
        "bounded_p50_authorized": False,
        "gate_decision": None,
        "diagnostic_scope": "INDEPENDENT_FAMILY_LOCAL_SCRATCH_MODEL",
        "uniform_contract_sha256": uniform_contract["contract_sha256"],
        "uniform_contract_file_sha256": _sha256(args.uniform_contract),
        "parent_t1_runtime_contract_sha256": t1_contract.sha256,
        "parent_t1_runtime_contract_file_sha256": _sha256(args.t1_contract),
        "gate_zero_runtime_contract_sha256": gate_zero_contract.sha256,
        "panel_artifact_sha256": panel["artifact_sha256"],
        "panel_selection_sha256": editing_t1_panel_selection_sha256(panel),
        "panel_census_sha256": editing_t1_panel_census_sha256(panel),
        "panel_capacity_strata_sha256": editing_t1_panel_capacity_strata_sha256(panel),
        **dict(active8_identity),
        "family": args.family,
        "panel_kind": EDITING_T1_UNIQUE_PANEL_KIND,
        "scope": optimization["scope"],
        "example_count": len(uniform_panel.examples),
        "unique_progress_address_count": materialized.unique_progress_address_count,
        "source_row_sha256s": list(materialized.source_row_sha256s),
        "operator_identity": {
            "operator_capability_fingerprint": model.operator_capabilities.fingerprint(),
            "enable_ring_system_delete": model.enable_ring_system_delete,
            "enable_ring_grow_macro": model.enable_ring_grow_macro,
            "enable_cycle_ops": model.enable_cycle_ops,
        },
        "weight_policy": {
            "unit": optimization["coefficient_unit"],
            "teacher_rate": optimization["teacher_rate"],
            "importance_coefficient": optimization["importance_coefficient"],
            "alias_policy": panel_policy["teacher_mark_alias_policy"],
            "original_coefficient_audit": original_coefficient_audit,
        },
        "scratch_initialization": {
            **asdict(parity),
            "initial_model_state_sha256": initial_model_state_sha256,
        },
        "cache_manifest_receipt": asdict(cache_receipt),
        "cache_receipts": [asdict(receipt) for receipt in materialized.cache_receipts],
        "training_report": report,
        "thresholds": dict(thresholds),
        "threshold_checks": checks,
        "per_example_floor_metrics": per_example_floors,
        "all_threshold_checks_pass": all(checks.values()),
        "final_model_state_sha256": state_dict_semantic_sha256(model.state_dict()),
        "runner_file_sha256": _sha256(Path(__file__)),
    }
    sealed = {**result, "result_sha256": _stable_sha256(result)}
    validate_uniform_capacity_result(
        sealed,
        contract=uniform_contract,
        expected_active8_identity=active8_identity,
        expected_cache_manifest_receipt=asdict(cache_receipt),
    )
    _write_if_absent(args.output, sealed)
    observed = json.loads(args.output.read_bytes())
    validate_uniform_capacity_result(
        observed,
        contract=uniform_contract,
        expected_active8_identity=active8_identity,
        expected_cache_manifest_receipt=asdict(cache_receipt),
    )
    if observed != sealed:
        raise SystemExit("immutable uniform-capacity output changed after publication")
    print(json.dumps(sealed, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
