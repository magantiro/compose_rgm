#!/usr/bin/env python3
"""Run one prospective worst-example-checkpointed uniform T1 V2 arm.

This bounded development diagnostic reuses the exact V1 64-row family panel
and immutable V8 successor cache.  It emits no P50 or training authority.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import compose_v4.experiments.editing_t1_uniform_successor_capacity_v2 as v2_optimization
from compose_v4.data.active8_trace_inventory import load_active8_trace_admission
from compose_v4.experiments.editing_gate_zero_runtime import (
    build_scratch_ringcore_model,
    load_frozen_validation_source,
    load_gate_zero_runtime_contract,
)
from compose_v4.experiments.editing_p50_gate import state_dict_semantic_sha256
from compose_v4.experiments.editing_t1_panel import (
    ACTIVE8_T1_IDENTITY_FIELDS,
    EDITING_T1_UNIQUE_PANEL_KIND,
    editing_t1_panel_capacity_strata_sha256,
    editing_t1_panel_census_sha256,
    editing_t1_panel_selection_sha256,
    validate_charge_policy_exclusions,
)
from compose_v4.experiments.editing_t1_successor_cache import (
    load_t1_successor_cache,
    t1_selected_trace_set_sha256,
)
from compose_v4.experiments.editing_t1_successor_runtime import (
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
from compose_v4.experiments.editing_t1_uniform_successor_capacity_v2 import (
    EXPECTED_V1_THRESHOLDS,
    UNIFORM_V2_RESULT_SCHEMA,
    UNIFORM_V2_RESULT_STATUS,
    UNIFORM_V2_RESULT_VERSION,
    is_sha256,
    load_uniform_capacity_v2_contract,
    serialized_source_tree_sha256,
    stable_sha256,
    train_uniform_successor_capacity_v2,
    validate_v2_training_report,
)
from compose_v4.experiments.ringcore_successor_leaderboard import load_json_object
from compose_v4.experiments.ringcore_validation_panel import (
    validate_validation_panel_artifact,
)
from compose_v4.experiments.successor_micro_overfit import (
    RINGCORE_EDITING_FAMILIES,
    SuccessorMicroOverfitError,
)
from scripts.run_editing_t1_successor_gate import (
    _execution_panel_from_frozen_authority,
    _load_successor_cache_receipt,
    _sha256,
    _write_if_absent,
)
from scripts.run_editing_t1_uniform_successor_capacity import (
    _json_normalized,
    prepare_uniform_unique_successor_panel,
)


def validate_uniform_capacity_v2_result(
    payload: object,
    *,
    contract: Mapping[str, object],
    expected_active8_identity: Mapping[str, str] | None = None,
    expected_cache_manifest_receipt: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Validate one immutable family-local V2 result and recompute its gates."""

    if not isinstance(payload, dict):
        raise EditingT1RuntimeError("uniform T1 V2 result must be one JSON object")
    source_commit = payload.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise EditingT1RuntimeError("uniform T1 V2 source commit is invalid")
    body = {key: value for key, value in payload.items() if key != "result_sha256"}
    if (
        payload.get("schema") != UNIFORM_V2_RESULT_SCHEMA
        or payload.get("schema_version") != UNIFORM_V2_RESULT_VERSION
        or payload.get("status") != UNIFORM_V2_RESULT_STATUS
        or payload.get("training_authorized") is not False
        or payload.get("bounded_p50_authorized") is not False
        or payload.get("gate_decision") is not None
        or payload.get("diagnostic_scope") != "INDEPENDENT_FAMILY_LOCAL_SCRATCH_MODEL"
        or payload.get("source_commit_binding")
        != "CLEAN_LAUNCHER_HEAD_ATTESTATION_PLUS_WORKER_VERIFIED_SERIALIZED_TREE"
        or payload.get("uniform_v2_contract_sha256") != contract["contract_sha256"]
        or payload.get("result_sha256") != stable_sha256(body)
    ):
        raise EditingT1RuntimeError("uniform T1 V2 result identity or self-hash is invalid")
    for field in (
        "uniform_v2_contract_file_sha256",
        "serialized_source_tree_sha256",
        "parent_v1_uniform_contract_sha256",
        "parent_v1_uniform_contract_file_sha256",
        "parent_t1_runtime_contract_sha256",
        "parent_t1_runtime_contract_file_sha256",
        "gate_zero_runtime_contract_sha256",
        "panel_artifact_sha256",
        "panel_selection_sha256",
        "panel_census_sha256",
        "panel_capacity_strata_sha256",
        *ACTIVE8_T1_IDENTITY_FIELDS,
        "optimization_law_sha256",
        "initial_model_state_sha256",
        "terminal_model_state_sha256",
        "selected_model_state_sha256",
        "returned_model_state_sha256",
        "runner_file_sha256",
        "optimizer_implementation_file_sha256",
        "result_sha256",
    ):
        if not is_sha256(payload.get(field)):
            raise EditingT1RuntimeError(f"uniform T1 V2 result {field} is not a SHA-256")
    parents = contract.get("parents")
    optimization = contract.get("optimization_law")
    if not isinstance(parents, Mapping) or not isinstance(optimization, Mapping):
        raise EditingT1RuntimeError("uniform T1 V2 contract parents or law are invalid")
    if (
        payload["parent_v1_uniform_contract_sha256"] != parents["v1_contract_sha256"]
        or payload["parent_v1_uniform_contract_file_sha256"] != parents["v1_contract_file_sha256"]
        or payload["parent_t1_runtime_contract_sha256"] != parents["runtime_contract_sha256"]
        or payload["parent_t1_runtime_contract_file_sha256"]
        != parents["runtime_contract_file_sha256"]
        or payload["panel_artifact_sha256"] != parents["panel_artifact_sha256"]
        or payload["optimization_law_sha256"] != stable_sha256(optimization)
        or payload["gate_zero_runtime_contract_sha256"]
        != payload["active8_support_contract_sha256"]
    ):
        raise EditingT1RuntimeError("uniform T1 V2 result parent bindings are invalid")
    family = payload.get("family")
    if (
        family not in RINGCORE_EDITING_FAMILIES
        or payload.get("panel_kind") != EDITING_T1_UNIQUE_PANEL_KIND
        or payload.get("scope") != "all"
        or payload.get("example_count") != 64
        or payload.get("unique_progress_address_count") != 64
    ):
        raise EditingT1RuntimeError("uniform T1 V2 result panel identity is invalid")
    source_rows = payload.get("source_row_sha256s")
    if (
        not isinstance(source_rows, list)
        or len(source_rows) != 64
        or len(set(source_rows)) != 64
        or any(not is_sha256(value) for value in source_rows)
    ):
        raise EditingT1RuntimeError("uniform T1 V2 source-row identities are invalid")
    operator_identity = payload.get("operator_identity")
    if (
        not isinstance(operator_identity, Mapping)
        or operator_identity.get("enable_ring_system_delete") is not False
        or operator_identity.get("enable_ring_grow_macro") is not False
        or operator_identity.get("enable_cycle_ops") is not True
        or not operator_identity.get("operator_capability_fingerprint")
    ):
        raise EditingT1RuntimeError("uniform T1 V2 operator identity is invalid")
    weight_policy = payload.get("weight_policy")
    if (
        not isinstance(weight_policy, Mapping)
        or weight_policy.get("teacher_rate") != 1.0
        or weight_policy.get("importance_coefficient") != 1.0
        or not isinstance(weight_policy.get("original_coefficient_audit"), Mapping)
        or weight_policy["original_coefficient_audit"].get("count") != 64
    ):
        raise EditingT1RuntimeError("uniform T1 V2 weight policy is invalid")
    report = payload.get("training_report")
    try:
        validated_report = validate_v2_training_report(
            report,
            family=str(family),
            maximum_updates=int(optimization["maximum_full_panel_updates"]),
            learning_rate=float(optimization["learning_rate"]),
            weight_decay=float(optimization["weight_decay"]),
            gradient_clip_norm=float(optimization["gradient_clip_norm"]),
            seed=int(optimization["seed"]),
        )
    except SuccessorMicroOverfitError as error:
        raise EditingT1RuntimeError("uniform T1 V2 training report is invalid") from error
    if (
        payload.get("thresholds") != EXPECTED_V1_THRESHOLDS
        or payload.get("threshold_checks") != validated_report["threshold_checks"]
        or payload.get("per_example_floor_metrics") != validated_report["per_example_floor_metrics"]
        or payload.get("all_threshold_checks_pass")
        is not validated_report["all_threshold_checks_pass"]
        or payload.get("selected_update") != validated_report["selected_update"]
        or payload.get("terminal_update") != validated_report["terminal_update"]
        or payload.get("initial_model_state_sha256")
        != validated_report["initial_model_state_sha256"]
        or payload.get("terminal_model_state_sha256")
        != validated_report["terminal_model_state_sha256"]
        or payload.get("selected_model_state_sha256")
        != validated_report["selected_model_state_sha256"]
        or payload.get("returned_model_state_sha256")
        != validated_report["returned_model_state_sha256"]
    ):
        raise EditingT1RuntimeError("uniform T1 V2 selected-state evidence changed")
    if not isinstance(payload.get("cache_manifest_receipt"), Mapping) or not isinstance(
        payload.get("cache_receipts"), list
    ):
        raise EditingT1RuntimeError("uniform T1 V2 cache evidence is malformed")
    if not payload["cache_receipts"]:
        raise EditingT1RuntimeError("uniform T1 V2 cache evidence is empty")
    if expected_cache_manifest_receipt is not None and (
        payload["cache_manifest_receipt"] != dict(expected_cache_manifest_receipt)
    ):
        raise EditingT1RuntimeError("uniform T1 V2 cache manifest receipt changed")
    if expected_active8_identity is not None and any(
        payload.get(field) != expected_active8_identity[field]
        for field in ACTIVE8_T1_IDENTITY_FIELDS
    ):
        raise EditingT1RuntimeError("uniform T1 V2 Active8 identity changed")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--transfer-root", type=Path, required=True)
    parser.add_argument(
        "--uniform-v2-contract",
        type=Path,
        default=ROOT / "configs" / "editing_t1_uniform_successor_capacity_v2.json",
    )
    parser.add_argument(
        "--uniform-v1-contract",
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
    if len(args.source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.source_commit
    ):
        raise SystemExit("--source-commit must be one lowercase 40-character Git commit")
    if not is_sha256(args.source_tree_sha256):
        raise SystemExit("--source-tree-sha256 must be one lowercase SHA-256")
    observed_source_tree_sha256 = serialized_source_tree_sha256(ROOT)
    if observed_source_tree_sha256 != args.source_tree_sha256:
        raise SystemExit(
            "uniform T1 V2 worker serialized source tree differs from its launcher attestation"
        )
    contract = load_uniform_capacity_v2_contract(args.uniform_v2_contract)
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
    parents = contract["parents"]
    assert isinstance(parents, Mapping)
    for field, observed in (
        ("v1_contract_file_sha256", _sha256(args.uniform_v1_contract)),
        ("runtime_contract_file_sha256", _sha256(args.t1_contract)),
        ("runtime_contract_sha256", t1_contract.sha256),
        ("panel_file_sha256", _sha256(args.panel)),
        ("panel_artifact_sha256", launch_authority.panel["artifact_sha256"]),
        ("forensics_file_sha256", _sha256(args.forensics)),
    ):
        if parents.get(field) != observed:
            raise SystemExit(f"uniform T1 V2 parent mismatch for {field}")

    gate_zero_contract = load_gate_zero_runtime_contract(args.gate_zero_contract)
    if gate_zero_contract.sha256 != t1_contract.payload["gate_zero_runtime_contract_sha256"]:
        raise SystemExit("uniform T1 V2 T1/Gate0 contract mismatch")
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
        ("charge_policy_exclusions_file_sha256", _sha256(args.charge_policy_exclusions)),
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
            raise SystemExit(f"uniform T1 V2 frozen provenance mismatch for {field}")
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
        raise SystemExit("uniform T1 V2 cache receipt names another exact trace set")
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
    panel_policy = contract["panel_policy"]
    optimization = contract["optimization_law"]
    assert isinstance(panel_policy, Mapping)
    assert isinstance(optimization, Mapping)
    if len(uniform_panel.examples) != int(panel_policy["examples_per_family"]):
        raise SystemExit("uniform T1 V2 panel has the wrong frozen family size")
    if int(gate_zero_contract.model["seed"]) != int(optimization["seed"]):
        raise SystemExit("uniform T1 V2 scratch seed changed from Gate0")
    report = train_uniform_successor_capacity_v2(
        model,
        uniform_panel,
        maximum_updates=int(optimization["maximum_full_panel_updates"]),
        learning_rate=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
        scope=str(optimization["scope"]),
        seed=int(optimization["seed"]),
        gradient_clip_norm=float(optimization["gradient_clip_norm"]),
        thresholds=EXPECTED_V1_THRESHOLDS,
    )
    if initial_model_state_sha256 != report["initial_model_state_sha256"]:
        raise SystemExit("uniform T1 V2 scratch initialization changed before optimization")
    if state_dict_semantic_sha256(model.state_dict()) != report["returned_model_state_sha256"]:
        raise SystemExit("uniform T1 V2 runner did not retain the selected checkpoint")
    raw_result = {
        "schema": UNIFORM_V2_RESULT_SCHEMA,
        "schema_version": UNIFORM_V2_RESULT_VERSION,
        "status": UNIFORM_V2_RESULT_STATUS,
        "training_authorized": False,
        "bounded_p50_authorized": False,
        "gate_decision": None,
        "diagnostic_scope": "INDEPENDENT_FAMILY_LOCAL_SCRATCH_MODEL",
        "source_commit": args.source_commit,
        "source_commit_binding": (
            "CLEAN_LAUNCHER_HEAD_ATTESTATION_PLUS_WORKER_VERIFIED_SERIALIZED_TREE"
        ),
        "serialized_source_tree_sha256": observed_source_tree_sha256,
        "uniform_v2_contract_sha256": contract["contract_sha256"],
        "uniform_v2_contract_file_sha256": _sha256(args.uniform_v2_contract),
        "parent_v1_uniform_contract_sha256": parents["v1_contract_sha256"],
        "parent_v1_uniform_contract_file_sha256": _sha256(args.uniform_v1_contract),
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
        "optimization_law_sha256": stable_sha256(optimization),
        "training_report": report,
        "selected_update": report["selected_update"],
        "terminal_update": report["terminal_update"],
        "thresholds": dict(EXPECTED_V1_THRESHOLDS),
        "threshold_checks": report["threshold_checks"],
        "per_example_floor_metrics": report["per_example_floor_metrics"],
        "all_threshold_checks_pass": report["all_threshold_checks_pass"],
        "initial_model_state_sha256": report["initial_model_state_sha256"],
        "terminal_model_state_sha256": report["terminal_model_state_sha256"],
        "selected_model_state_sha256": report["selected_model_state_sha256"],
        "returned_model_state_sha256": report["returned_model_state_sha256"],
        "runner_file_sha256": _sha256(Path(__file__)),
        "optimizer_implementation_file_sha256": _sha256(Path(v2_optimization.__file__).resolve()),
    }
    result = _json_normalized(raw_result)
    assert isinstance(result, dict)
    sealed = {**result, "result_sha256": stable_sha256(result)}
    validate_uniform_capacity_v2_result(
        sealed,
        contract=contract,
        expected_active8_identity=active8_identity,
        expected_cache_manifest_receipt=asdict(cache_receipt),
    )
    _write_if_absent(args.output, sealed)
    observed = json.loads(args.output.read_bytes())
    validate_uniform_capacity_v2_result(
        observed,
        contract=contract,
        expected_active8_identity=active8_identity,
        expected_cache_manifest_receipt=asdict(cache_receipt),
    )
    if observed != sealed:
        raise SystemExit("immutable uniform T1 V2 output changed after publication")
    print(json.dumps(sealed, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
