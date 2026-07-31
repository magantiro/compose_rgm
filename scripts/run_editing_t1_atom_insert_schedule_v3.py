#!/usr/bin/env python3
"""Run the prospective non-authorizing atom-insert V3 schedule diagnostic."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import compose_v4.experiments.editing_t1_atom_insert_schedule_v3 as v3_optimization
import scripts.run_editing_t1_uniform_successor_capacity_v2 as v2_runner
from compose_v4.experiments.editing_p50_gate import state_dict_semantic_sha256
from compose_v4.experiments.editing_t1_atom_insert_schedule_v3 import (
    EXPECTED_V1_THRESHOLDS,
    V3_RESULT_SCHEMA,
    V3_RESULT_STATUS,
    V3_RESULT_VERSION,
    load_atom_insert_schedule_v3_contract,
    train_atom_insert_schedule_v3,
    validate_atom_insert_v3_training_report,
)
from compose_v4.experiments.editing_t1_successor_runtime import EditingT1RuntimeError
from compose_v4.experiments.editing_t1_uniform_successor_capacity_v2 import (
    is_sha256,
    load_uniform_capacity_v2_contract,
    serialized_source_tree_sha256,
    stable_sha256,
)
from compose_v4.experiments.successor_micro_overfit import SuccessorMicroOverfitError
from scripts.run_editing_t1_successor_gate import _sha256, _write_if_absent
from scripts.run_editing_t1_uniform_successor_capacity import (
    _json_normalized,
    prepare_uniform_unique_successor_panel,
)


def _parser() -> argparse.ArgumentParser:
    parser = v2_runner._parser()
    parser.description = __doc__
    parser.add_argument(
        "--v3-contract",
        type=Path,
        default=ROOT / "configs" / "editing_t1_atom_insert_schedule_v3.json",
    )
    parser.add_argument("--v2-result", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    return parser


def validate_atom_insert_v3_result(
    payload: object,
    *,
    contract: Mapping[str, object],
) -> Mapping[str, object]:
    if not isinstance(payload, dict):
        raise EditingT1RuntimeError("atom-insert V3 result must be one object")
    body = {key: value for key, value in payload.items() if key != "result_sha256"}
    report = payload.get("training_report")
    try:
        validated_report = validate_atom_insert_v3_training_report(report, contract=contract)
    except SuccessorMicroOverfitError as error:
        raise EditingT1RuntimeError("atom-insert V3 training report is invalid") from error
    if (
        payload.get("schema") != V3_RESULT_SCHEMA
        or payload.get("schema_version") != V3_RESULT_VERSION
        or payload.get("status") != V3_RESULT_STATUS
        or payload.get("training_authorized") is not False
        or payload.get("bounded_p50_authorized") is not False
        or payload.get("gate_decision") is not None
        or payload.get("family") != "atom_insert"
        or payload.get("v3_contract_sha256") != contract["contract_sha256"]
        or payload.get("result_sha256") != stable_sha256(body)
        or validated_report.get("training_authorized") is not False
        or validated_report.get("bounded_p50_authorized") is not False
        or validated_report.get("thresholds") != EXPECTED_V1_THRESHOLDS
        or payload.get("selected_update") != validated_report.get("selected_update")
        or payload.get("terminal_update") != validated_report.get("terminal_update")
        or payload.get("all_threshold_checks_pass")
        is not validated_report.get("all_threshold_checks_pass")
        or payload.get("threshold_checks") != validated_report.get("threshold_checks")
        or payload.get("selected_model_state_sha256")
        != validated_report.get("selected_model_state_sha256")
        or payload.get("phase1_model_state_sha256")
        != validated_report.get("phase1_model_state_sha256")
        or payload.get("returned_model_state_sha256")
        != validated_report.get("returned_model_state_sha256")
    ):
        raise EditingT1RuntimeError("atom-insert V3 result identity or evidence is invalid")
    for field in (
        "source_tree_sha256",
        "v3_contract_file_sha256",
        "v3_contract_sha256",
        "v2_result_file_sha256",
        "v2_result_sha256",
        "optimization_law_sha256",
        "checkpoint_identity_sha256",
        "initial_model_state_sha256",
        "phase1_model_state_sha256",
        "selected_model_state_sha256",
        "terminal_model_state_sha256",
        "returned_model_state_sha256",
        "runner_file_sha256",
        "optimizer_implementation_file_sha256",
        "result_sha256",
    ):
        if not is_sha256(payload.get(field)):
            raise EditingT1RuntimeError(f"atom-insert V3 result {field} is not a SHA-256")
    receipts = validated_report.get("checkpoint_receipts")
    if not isinstance(receipts, list) or not receipts:
        raise EditingT1RuntimeError("atom-insert V3 emitted no durable checkpoint")
    if any(
        not isinstance(receipt, Mapping)
        or receipt.get("training_authorized") is not False
        or receipt.get("bounded_p50_authorized") is not False
        or not is_sha256(receipt.get("checkpoint_file_sha256"))
        or not is_sha256(receipt.get("checkpoint_semantic_sha256"))
        for receipt in receipts
    ):
        raise EditingT1RuntimeError("atom-insert V3 checkpoint receipt is invalid")
    return payload


def main(
    argv: Sequence[str] | None = None,
    *,
    checkpoint_publish_callback: Callable[[], None] | None = None,
) -> int:
    args = _parser().parse_args(argv)
    if args.family != "atom_insert":
        raise SystemExit("atom-insert V3 rejects every family except atom_insert")
    if len(args.source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.source_commit
    ):
        raise SystemExit("--source-commit must be one lowercase Git commit")
    if not is_sha256(args.source_tree_sha256):
        raise SystemExit("--source-tree-sha256 must be one lowercase SHA-256")
    observed_source_tree_sha256 = serialized_source_tree_sha256(ROOT)
    if observed_source_tree_sha256 != args.source_tree_sha256:
        raise SystemExit("atom-insert V3 serialized source tree differs from its attestation")

    contract = load_atom_insert_schedule_v3_contract(args.v3_contract)
    v2_contract = load_uniform_capacity_v2_contract(args.uniform_v2_contract)
    parents = contract["parents"]
    assert isinstance(parents, Mapping)
    if (
        parents["v2_contract_file_sha256"] != _sha256(args.uniform_v2_contract)
        or parents["v2_contract_sha256"] != v2_contract["contract_sha256"]
        or parents["v2_result_file_sha256"] != _sha256(args.v2_result)
    ):
        raise SystemExit("atom-insert V3 V2 parent bytes differ from the frozen contract")
    v2_payload = json.loads(args.v2_result.read_bytes())
    v2_runner.validate_uniform_capacity_v2_result(v2_payload, contract=v2_contract)
    if (
        v2_payload.get("result_sha256") != parents["v2_result_sha256"]
        or v2_payload.get("family") != "atom_insert"
        or v2_payload.get("all_threshold_checks_pass") is not False
        or v2_payload.get("selected_update") != 495
    ):
        raise SystemExit("atom-insert V3 parent is not the exact failed V2 result")

    try:
        launch_authority = v2_runner.load_editing_t1_launch_authority(
            contract_path=args.t1_contract,
            panel_path=args.panel,
            capacity_census_path=args.capacity_census,
            expected_active8_inventory_manifest_file_sha256=args.active8_inventory_file_sha256,
        )
    except EditingT1RuntimeError as error:
        raise SystemExit(str(error)) from error
    t1_contract = launch_authority.contract
    for field, observed in (
        ("runtime_contract_file_sha256", _sha256(args.t1_contract)),
        ("runtime_contract_sha256", t1_contract.sha256),
        ("panel_file_sha256", _sha256(args.panel)),
        ("panel_artifact_sha256", launch_authority.panel["artifact_sha256"]),
        ("forensics_file_sha256", _sha256(args.forensics)),
    ):
        if parents[field] != observed:
            raise SystemExit(f"atom-insert V3 parent mismatch for {field}")

    gate_zero_contract = v2_runner.load_gate_zero_runtime_contract(args.gate_zero_contract)
    if gate_zero_contract.sha256 != t1_contract.payload["gate_zero_runtime_contract_sha256"]:
        raise SystemExit("atom-insert V3 T1/Gate0 contract mismatch")
    active8_admission = v2_runner.load_active8_trace_admission(
        args.active8_inventory,
        expected_manifest_file_sha256=t1_contract.payload["active8_inventory_manifest_file_sha256"],
        expected_inventory_sha256=t1_contract.payload["active8_inventory_sha256"],
        expected_effective_source_corpus_cache_sha256=t1_contract.payload[
            "active8_effective_source_corpus_cache_sha256"
        ],
        expected_support_contract_sha256=t1_contract.payload["active8_support_contract_sha256"],
    )
    with gzip.open(args.forensics, "rt") as handle:
        forensics = json.load(handle)
    v2_runner.validate_validation_panel_artifact(
        forensics,
        config=v2_runner.load_json_object(args.leaderboard_config),
        inventory=v2_runner.load_json_object(args.inventory),
    )
    source = v2_runner.load_frozen_validation_source(
        contract=gate_zero_contract,
        transfer_root=args.transfer_root,
        sidecar_path=args.semantic_sidecar,
        sidecar_manifest_path=args.semantic_sidecar_manifest,
    )
    charge_policy_audit = v2_runner.load_json_object(args.charge_policy_audit)
    charge_policy_exclusions = v2_runner.load_json_object(args.charge_policy_exclusions)
    for field, observed in (
        ("charge_policy_audit_file_sha256", _sha256(args.charge_policy_audit)),
        ("charge_policy_exclusions_file_sha256", _sha256(args.charge_policy_exclusions)),
        ("charge_policy_exclusion_payload_sha256", charge_policy_exclusions["payload_sha256"]),
        (
            "charge_policy_source_input_inventory_sha256",
            charge_policy_exclusions["source_input_inventory_sha256"],
        ),
        ("capacity_census_file_sha256", _sha256(args.capacity_census)),
        ("capacity_census_sha256", launch_authority.capacity_census["census_sha256"]),
    ):
        if t1_contract.payload[field] != observed:
            raise SystemExit(f"atom-insert V3 provenance mismatch for {field}")
    excluded_trace_ids = v2_runner.validate_charge_policy_exclusions(
        audit=charge_policy_audit,
        audit_file_sha256=_sha256(args.charge_policy_audit),
        exclusions=charge_policy_exclusions,
        exclusions_file_sha256=_sha256(args.charge_policy_exclusions),
    )
    panel = v2_runner._execution_panel_from_frozen_authority(
        launch_authority,
        semantic_sidecar_file_sha256=_sha256(args.semantic_sidecar),
        semantic_sidecar_manifest_file_sha256=_sha256(args.semantic_sidecar_manifest),
    )
    active8_identity = v2_runner.validate_t1_active8_runtime_binding(
        contract=t1_contract,
        gate_zero_contract=gate_zero_contract,
        source=source,
        panel=panel,
        active8_admission=active8_admission,
    )
    model, parity = v2_runner.build_scratch_ringcore_model(gate_zero_contract)
    initial_model_state_sha256 = state_dict_semantic_sha256(model.state_dict())
    expected_cache_identity = v2_runner.build_t1_successor_cache_identity(
        contract=t1_contract,
        gate_zero_contract=gate_zero_contract,
        source=source,
        panel=panel,
        model=model,
        active8_admission=active8_admission,
        compiler_device="cpu",
    )
    resolved = v2_runner.resolve_t1_panel_view(
        source=source,
        panel=panel,
        forensics=forensics,
        family="atom_insert",
        panel_kind=v2_runner.EDITING_T1_UNIQUE_PANEL_KIND,
        max_atoms=int(gate_zero_contract.model["max_atoms"]),
        excluded_trace_ids=excluded_trace_ids,
        active8_admission=active8_admission,
    )
    cache_receipt = v2_runner._load_successor_cache_receipt(args.successor_cache_receipt)
    if cache_receipt.selected_trace_set_sha256 != v2_runner.t1_selected_trace_set_sha256(
        resolved.selected_records
    ):
        raise SystemExit("atom-insert V3 cache receipt names another trace set")
    successor_cache = v2_runner.load_t1_successor_cache(
        args.successor_cache_root,
        cache_receipt,
        expected_identity=expected_cache_identity,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    model = model.to(device)
    materialized = v2_runner.materialize_t1_panel(
        model,
        source=source,
        panel=panel,
        forensics=forensics,
        family="atom_insert",
        panel_kind=v2_runner.EDITING_T1_UNIQUE_PANEL_KIND,
        max_atoms=int(gate_zero_contract.model["max_atoms"]),
        excluded_trace_ids=excluded_trace_ids,
        active8_admission=active8_admission,
        successor_cache=successor_cache,
    )
    uniform_panel, original_coefficient_audit = prepare_uniform_unique_successor_panel(
        model, materialized.prepared
    )
    optimization = contract["optimization_law"]
    phase1 = optimization["phase_1"]
    phase2 = optimization["phase_2"]
    assert isinstance(optimization, Mapping)
    assert isinstance(phase1, Mapping)
    assert isinstance(phase2, Mapping)
    if len(uniform_panel.examples) != 64 or materialized.unique_progress_address_count != 64:
        raise SystemExit("atom-insert V3 exact panel cardinality changed")
    if int(gate_zero_contract.model["seed"]) != int(optimization["seed"]):
        raise SystemExit("atom-insert V3 scratch seed changed from Gate0")
    checkpoint_identity = {
        "v3_contract_sha256": contract["contract_sha256"],
        "source_commit": args.source_commit,
        "source_tree_sha256": observed_source_tree_sha256,
        "runtime_contract_sha256": t1_contract.sha256,
        "gate_zero_contract_sha256": gate_zero_contract.sha256,
        "panel_artifact_sha256": panel["artifact_sha256"],
        "active8_identity": dict(active8_identity),
        "cache_manifest_sha256": cache_receipt.manifest_sha256,
        "source_row_sha256s": list(materialized.source_row_sha256s),
        "operator_capability_fingerprint": model.operator_capabilities.fingerprint(),
        "initial_model_state_sha256": initial_model_state_sha256,
    }
    report = train_atom_insert_schedule_v3(
        model,
        uniform_panel,
        phase1_updates=int(phase1["completed_updates"]),
        phase1_learning_rate=float(phase1["learning_rate"]),
        phase2_updates=int(phase2["maximum_additional_updates"]),
        phase2_learning_rate=float(phase2["learning_rate"]),
        expected_initial_model_sha256=str(phase1["required_initial_model_state_sha256"]),
        expected_phase1_model_sha256=str(phase1["required_terminal_model_state_sha256"]),
        weight_decay=float(optimization["weight_decay"]),
        scope=str(optimization["scope"]),
        seed=int(optimization["seed"]),
        gradient_clip_norm=float(optimization["gradient_clip_norm"]),
        thresholds=EXPECTED_V1_THRESHOLDS,
        checkpoint_root=args.checkpoint_root,
        checkpoint_identity=checkpoint_identity,
        checkpoint_interval=50,
        resume_checkpoint=args.resume_checkpoint,
        checkpoint_publish_callback=checkpoint_publish_callback,
    )
    if state_dict_semantic_sha256(model.state_dict()) != report["returned_model_state_sha256"]:
        raise SystemExit("atom-insert V3 did not retain its selected checkpoint")
    raw_result = {
        "schema": V3_RESULT_SCHEMA,
        "schema_version": V3_RESULT_VERSION,
        "status": V3_RESULT_STATUS,
        "training_authorized": False,
        "bounded_p50_authorized": False,
        "gate_decision": None,
        "family": "atom_insert",
        "diagnostic_scope": "INDEPENDENT_FAMILY_LOCAL_SCRATCH_MODEL_SCHEDULE_REFINEMENT",
        "source_commit": args.source_commit,
        "source_tree_sha256": observed_source_tree_sha256,
        "v3_contract_file_sha256": _sha256(args.v3_contract),
        "v3_contract_sha256": contract["contract_sha256"],
        "v2_result_file_sha256": _sha256(args.v2_result),
        "v2_result_sha256": v2_payload["result_sha256"],
        "parent_t1_runtime_contract_sha256": t1_contract.sha256,
        "gate_zero_runtime_contract_sha256": gate_zero_contract.sha256,
        "panel_artifact_sha256": panel["artifact_sha256"],
        **dict(active8_identity),
        "operator_identity": {
            "operator_capability_fingerprint": model.operator_capabilities.fingerprint(),
            "enable_ring_system_delete": model.enable_ring_system_delete,
            "enable_ring_grow_macro": model.enable_ring_grow_macro,
            "enable_cycle_ops": model.enable_cycle_ops,
        },
        "weight_policy": {
            "teacher_rate": optimization["teacher_rate"],
            "importance_coefficient": optimization["importance_coefficient"],
            "original_coefficient_audit": original_coefficient_audit,
        },
        "scratch_initialization": {
            **asdict(parity),
            "initial_model_state_sha256": initial_model_state_sha256,
        },
        "cache_manifest_receipt": asdict(cache_receipt),
        "cache_receipts": [asdict(receipt) for receipt in materialized.cache_receipts],
        "optimization_law_sha256": stable_sha256(optimization),
        "checkpoint_identity_sha256": stable_sha256(checkpoint_identity),
        "training_report": report,
        "selected_update": report["selected_update"],
        "terminal_update": report["terminal_update"],
        "threshold_checks": report["threshold_checks"],
        "all_threshold_checks_pass": report["all_threshold_checks_pass"],
        "initial_model_state_sha256": report["initial_model_state_sha256"],
        "phase1_model_state_sha256": report["phase1_model_state_sha256"],
        "terminal_model_state_sha256": report["terminal_model_state_sha256"],
        "selected_model_state_sha256": report["selected_model_state_sha256"],
        "returned_model_state_sha256": report["returned_model_state_sha256"],
        "runner_file_sha256": _sha256(Path(__file__)),
        "optimizer_implementation_file_sha256": _sha256(Path(v3_optimization.__file__)),
    }
    result = _json_normalized(raw_result)
    assert isinstance(result, dict)
    sealed = {**result, "result_sha256": stable_sha256(result)}
    validate_atom_insert_v3_result(sealed, contract=contract)
    _write_if_absent(args.output, sealed)
    observed = json.loads(args.output.read_bytes())
    validate_atom_insert_v3_result(observed, contract=contract)
    if observed != sealed:
        raise SystemExit("immutable atom-insert V3 output changed after publication")
    print(json.dumps(sealed, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
