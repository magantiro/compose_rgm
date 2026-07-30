#!/usr/bin/env python3
"""Run one exact scratch-model T1 successor-capacity arm.

The output is a development diagnostic with null thresholds and no gate
decision.  It is not a checkpoint for downstream experiments.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compose_v4.experiments.editing_gate_zero_runtime import (  # noqa: E402
    build_scratch_ringcore_model,
    load_frozen_validation_source,
    load_gate_zero_runtime_contract,
)
from compose_v4.experiments.editing_t1_panel import (  # noqa: E402
    EDITING_T1_GLOBAL_REPEATED_PANEL_KIND,
    EDITING_T1_UNIQUE_PANEL_KIND,
    EDITING_T1_WITHIN_FAMILY_REPEATED_PANEL_KIND,
    load_editing_t1_panel,
    validate_charge_policy_exclusions,
)
from compose_v4.experiments.editing_t1_successor_runtime import (  # noqa: E402
    load_editing_t1_runtime_contract,
    load_editing_t1_result,
    materialize_t1_panel,
    run_editing_t1_arm,
)
from compose_v4.experiments.ringcore_successor_leaderboard import (  # noqa: E402
    load_json_object,
)
from compose_v4.experiments.ringcore_validation_panel import (  # noqa: E402
    validate_validation_panel_artifact,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _write_if_absent(path: Path, payload: object) -> None:
    content = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise FileExistsError(f"immutable T1 output already differs: {path}") from None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer-root", type=Path, required=True)
    parser.add_argument(
        "--t1-contract",
        type=Path,
        default=ROOT / "configs" / "editing_t1_successor_gate_v3.json",
    )
    parser.add_argument(
        "--gate-zero-contract",
        type=Path,
        default=ROOT / "configs" / "editing_gate_zero_runtime_v2.json",
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=(
            ROOT
            / "diagnostics"
            / "coherence"
            / "editing_t1_successor_panel_v3_active8_2026-07-30.json"
        ),
    )
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
    parser.add_argument("--family", required=True)
    parser.add_argument(
        "--panel-kind",
        choices=(
            EDITING_T1_UNIQUE_PANEL_KIND,
            EDITING_T1_GLOBAL_REPEATED_PANEL_KIND,
            EDITING_T1_WITHIN_FAMILY_REPEATED_PANEL_KIND,
        ),
        default=EDITING_T1_UNIQUE_PANEL_KIND,
    )
    parser.add_argument(
        "--scope",
        choices=("heads_only", "heads_plus_pair_projection", "all"),
        default="heads_only",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    t1_contract = load_editing_t1_runtime_contract(args.t1_contract)
    gate_zero_contract = load_gate_zero_runtime_contract(args.gate_zero_contract)
    if gate_zero_contract.sha256 != t1_contract.payload["gate_zero_runtime_contract_sha256"]:
        raise SystemExit("T1/Gate-0 runtime contract hash mismatch")
    with gzip.open(args.forensics, "rt") as handle:
        forensics = json.load(handle)
    forensics_file_sha256 = _sha256(args.forensics)
    if forensics_file_sha256 != t1_contract.payload["forensics_file_sha256"]:
        raise SystemExit("T1 forensics file hash mismatch")
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
    charge_policy_audit_file_sha256 = _sha256(args.charge_policy_audit)
    charge_policy_exclusions_file_sha256 = _sha256(args.charge_policy_exclusions)
    excluded_trace_ids = validate_charge_policy_exclusions(
        audit=charge_policy_audit,
        audit_file_sha256=charge_policy_audit_file_sha256,
        exclusions=charge_policy_exclusions,
        exclusions_file_sha256=charge_policy_exclusions_file_sha256,
    )
    for field, observed in (
        (
            "charge_policy_audit_file_sha256",
            charge_policy_audit_file_sha256,
        ),
        (
            "charge_policy_exclusions_file_sha256",
            charge_policy_exclusions_file_sha256,
        ),
        (
            "charge_policy_exclusion_payload_sha256",
            charge_policy_exclusions["payload_sha256"],
        ),
        (
            "charge_policy_source_input_inventory_sha256",
            charge_policy_exclusions["source_input_inventory_sha256"],
        ),
    ):
        if t1_contract.payload[field] != observed:
            raise SystemExit(f"T1 charge-policy provenance mismatch for {field}")
    panel = load_editing_t1_panel(
        args.panel,
        expected_artifact_sha256=t1_contract.payload["panel_artifact_sha256"],
        forensics=forensics,
        forensics_file_sha256=forensics_file_sha256,
        sidecar=source.sidecar,
        semantic_sidecar_file_sha256=_sha256(args.semantic_sidecar),
        semantic_sidecar_manifest_file_sha256=_sha256(args.semantic_sidecar_manifest),
        charge_policy_audit=charge_policy_audit,
        charge_policy_audit_file_sha256=charge_policy_audit_file_sha256,
        charge_policy_exclusions=charge_policy_exclusions,
        charge_policy_exclusions_file_sha256=(charge_policy_exclusions_file_sha256),
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    if args.audit_only:
        model, parity = build_scratch_ringcore_model(gate_zero_contract)
        model = model.to(device)
        materialized = materialize_t1_panel(
            model,
            source=source,
            panel=panel,
            forensics=forensics,
            family=args.family,
            panel_kind=args.panel_kind,
            max_atoms=int(gate_zero_contract.model["max_atoms"]),
            excluded_trace_ids=excluded_trace_ids,
        )
        result = {
            "status": "T1_PANEL_CACHE_AUDIT_COMPLETE_NO_OPTIMIZATION",
            "training_authorized": False,
            "gate_decision": None,
            "contract_sha256": t1_contract.sha256,
            "panel_artifact_sha256": panel["artifact_sha256"],
            "charge_policy_exclusion_payload_sha256": (charge_policy_exclusions["payload_sha256"]),
            "charge_policy_source_input_inventory_sha256": (
                charge_policy_exclusions["source_input_inventory_sha256"]
            ),
            "family": args.family,
            "panel_kind": args.panel_kind,
            "panel_role": (
                "PRIMARY_MIXED_FAMILY_EMPIRICAL_SUCCESSOR_LAW"
                if args.panel_kind == EDITING_T1_GLOBAL_REPEATED_PANEL_KIND
                else (
                    "SECONDARY_WITHIN_FAMILY_CONDITIONAL_DIAGNOSTIC"
                    if args.panel_kind == EDITING_T1_WITHIN_FAMILY_REPEATED_PANEL_KIND
                    else "DETERMINISTIC_FAMILY_CAPACITY_DIAGNOSTIC"
                )
            ),
            "example_count": len(materialized.examples),
            "unique_progress_address_count": (materialized.unique_progress_address_count),
            "repeated_progress_observation_count": (
                len(materialized.examples) - materialized.unique_progress_address_count
            ),
            "operator_identity": {
                "operator_capability_fingerprint": (model.operator_capabilities.fingerprint()),
                "enable_ring_system_delete": model.enable_ring_system_delete,
                "enable_ring_grow_macro": model.enable_ring_grow_macro,
                "enable_cycle_ops": model.enable_cycle_ops,
            },
            "initialization_parity": {
                "regime": parity.regime,
                "fresh_tensor_count": parity.fresh_tensor_count,
                "copied_row_count": parity.copied_row_count,
            },
            "cache_receipts": [receipt.__dict__ for receipt in materialized.cache_receipts],
        }
    else:
        result = run_editing_t1_arm(
            contract=t1_contract,
            gate_zero_contract=gate_zero_contract,
            source=source,
            panel=panel,
            forensics=forensics,
            family=args.family,
            panel_kind=args.panel_kind,
            scope=args.scope,
            device=device,
            excluded_trace_ids=excluded_trace_ids,
        )
    if args.output is not None:
        _write_if_absent(args.output, result)
        if not args.audit_only:
            load_editing_t1_result(
                args.output,
                expected_result_sha256=result["result_sha256"],
                expected_contract_sha256=t1_contract.sha256,
                expected_panel_artifact_sha256=panel["artifact_sha256"],
                expected_cache_receipts=result["cache_receipts"],
            )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
