#!/usr/bin/env python3
"""Run one exact scratch-model T1 successor-capacity arm.

The output is a development diagnostic bound to prospectively frozen numeric
thresholds, but it contains no gate decision. It is not a checkpoint for
downstream experiments.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compose_v4.data.active8_trace_inventory import (  # noqa: E402
    load_active8_trace_admission,
)
from compose_v4.data.immutable_artifact import (  # noqa: E402
    ImmutableArtifactError,
    write_bytes_if_absent,
)
from compose_v4.experiments.editing_gate_zero_runtime import (  # noqa: E402
    build_exact_cache_provenance,
    build_scratch_ringcore_model,
    load_frozen_validation_source,
    load_gate_zero_runtime_contract,
)
from compose_v4.experiments.editing_t1_panel import (  # noqa: E402
    EDITING_T1_GLOBAL_REPEATED_PANEL_KIND,
    EDITING_T1_UNIQUE_PANEL_KIND,
    EDITING_T1_WITHIN_FAMILY_REPEATED_PANEL_KIND,
    EditingT1PanelError,
    active8_t1_identity,
    editing_t1_panel_capacity_strata_sha256,
    editing_t1_panel_census_sha256,
    editing_t1_panel_selection_sha256,
    validate_charge_policy_exclusions,
    within_family_repeated_panel_families,
)
from compose_v4.experiments.editing_t1_successor_runtime import (  # noqa: E402
    EDITING_T1_V4_CAPACITY_CENSUS_RELATIVE_PATH,
    EDITING_T1_V4_PANEL_RELATIVE_PATH,
    EDITING_T1_V8_CONTRACT_RELATIVE_PATH,
    EditingT1LaunchAuthority,
    EditingT1RuntimeError,
    build_t1_successor_cache_identity,
    load_editing_t1_launch_authority,
    load_editing_t1_result,
    materialize_t1_panel,
    resolve_t1_panel_view,
    require_editing_t1_family_scope_applicable,
    run_editing_t1_arm,
    validate_t1_active8_runtime_binding,
)
from compose_v4.experiments.editing_t1_successor_cache import (  # noqa: E402
    T1SuccessorCacheManifest,
    T1SuccessorCacheManifestReceipt,
    build_t1_successor_cache,
    load_t1_successor_cache,
    t1_selected_trace_set_sha256,
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
    try:
        write_bytes_if_absent(path, content)
    except ImmutableArtifactError as error:
        raise FileExistsError(f"immutable T1 output already differs: {path}") from error


def _load_successor_cache_receipt(path: Path) -> T1SuccessorCacheManifestReceipt:
    try:
        payload = json.loads(Path(path).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"T1 successor-cache receipt is unreadable: {path}") from error
    if not isinstance(payload, dict):
        raise SystemExit("T1 successor-cache receipt must be an object")
    try:
        return T1SuccessorCacheManifestReceipt(**payload)
    except (TypeError, ValueError) as error:
        raise SystemExit("T1 successor-cache receipt is invalid") from error


def _successor_cache_completion_payload(
    *,
    family: str,
    panel_kind: str,
    manifest: T1SuccessorCacheManifest,
    receipt: T1SuccessorCacheManifestReceipt,
    initialization_parity: Mapping[str, object],
) -> dict[str, object]:
    """Build the validated CPU cache-completion status payload."""

    return {
        "status": "T1_SUCCESSOR_CACHE_COMPLETE_NO_OPTIMIZATION",
        "training_authorized": False,
        "gate_decision": None,
        "family": family,
        "panel_kind": panel_kind,
        "selected_trace_count": len(manifest.selected_traces),
        "cache_receipt": asdict(receipt),
        "initialization_parity": dict(initialization_parity),
    }


def _execution_panel_from_frozen_authority(
    launch_authority: EditingT1LaunchAuthority,
    *,
    semantic_sidecar_file_sha256: str,
    semantic_sidecar_manifest_file_sha256: str,
) -> Mapping[str, object]:
    """Bind the frozen panel to physical sidecars without replaying its census.

    Full production re-enumeration is a one-time panel-freeze invariant.  Each
    arm recompiles and round-trips the exact selected trace fibers before
    optimization, so replaying every eligible census row here would add no
    arm-specific evidence.
    """

    panel = launch_authority.panel
    source = panel["source"]
    for field, observed in (
        ("semantic_sidecar_file_sha256", semantic_sidecar_file_sha256),
        (
            "semantic_sidecar_manifest_file_sha256",
            semantic_sidecar_manifest_file_sha256,
        ),
    ):
        if source.get(field) != observed:
            raise SystemExit(f"T1 frozen provenance mismatch for {field}")
    return panel


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer-root", type=Path, required=True)
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
    parser.add_argument(
        "--active8-inventory-file-sha256",
        required=True,
        help="Expected SHA-256 of the physical Active8 inventory manifest.",
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=ROOT / EDITING_T1_V4_PANEL_RELATIVE_PATH,
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
        choices=("heads_only", "heads_plus_local_adapter", "all"),
        default="heads_only",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--successor-cache-root", type=Path)
    parser.add_argument("--successor-cache-receipt", type=Path)
    parser.add_argument("--build-successor-cache", action="store_true")
    parser.add_argument("--validate-successor-cache", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if sum(
        (
            bool(args.audit_only),
            bool(args.build_successor_cache),
            bool(args.validate_successor_cache),
        )
    ) > 1:
        raise SystemExit(
            "--audit-only, --build-successor-cache, and "
            "--validate-successor-cache are mutually exclusive"
        )
    try:
        require_editing_t1_family_scope_applicable(
            args.family,
            args.scope,
        )
    except EditingT1RuntimeError as error:
        raise SystemExit(str(error)) from error
    try:
        launch_authority = load_editing_t1_launch_authority(
            contract_path=args.t1_contract,
            panel_path=args.panel,
            capacity_census_path=args.capacity_census,
            expected_active8_inventory_manifest_file_sha256=(
                args.active8_inventory_file_sha256
            ),
        )
    except EditingT1RuntimeError as error:
        raise SystemExit(str(error)) from error
    t1_contract = launch_authority.contract
    gate_zero_contract = load_gate_zero_runtime_contract(args.gate_zero_contract)
    if gate_zero_contract.sha256 != t1_contract.payload["gate_zero_runtime_contract_sha256"]:
        raise SystemExit("T1/Gate-0 runtime contract hash mismatch")
    if gate_zero_contract.sha256 != t1_contract.payload["active8_support_contract_sha256"]:
        raise SystemExit("T1 Gate-0/Active8 support-contract hash mismatch")
    if (
        args.active8_inventory_file_sha256
        != t1_contract.payload["active8_inventory_manifest_file_sha256"]
    ):
        raise SystemExit(
            "T1 launch and runtime contract name different physical Active8 inventories"
        )
    active8_admission = load_active8_trace_admission(
        args.active8_inventory,
        expected_manifest_file_sha256=(
            t1_contract.payload["active8_inventory_manifest_file_sha256"]
        ),
        expected_inventory_sha256=(t1_contract.payload["active8_inventory_sha256"]),
        expected_effective_source_corpus_cache_sha256=(
            t1_contract.payload["active8_effective_source_corpus_cache_sha256"]
        ),
        expected_support_contract_sha256=(t1_contract.payload["active8_support_contract_sha256"]),
    )
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
    capacity_census = dict(launch_authority.capacity_census)
    charge_policy_audit_file_sha256 = _sha256(args.charge_policy_audit)
    charge_policy_exclusions_file_sha256 = _sha256(args.charge_policy_exclusions)
    capacity_census_file_sha256 = _sha256(args.capacity_census)
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
        (
            "capacity_census_file_sha256",
            capacity_census_file_sha256,
        ),
        (
            "capacity_census_sha256",
            capacity_census["census_sha256"],
        ),
    ):
        if t1_contract.payload[field] != observed:
            raise SystemExit(f"T1 frozen provenance mismatch for {field}")
    panel = _execution_panel_from_frozen_authority(
        launch_authority,
        semantic_sidecar_file_sha256=_sha256(args.semantic_sidecar),
        semantic_sidecar_manifest_file_sha256=_sha256(args.semantic_sidecar_manifest),
    )
    if args.panel_kind == EDITING_T1_WITHIN_FAMILY_REPEATED_PANEL_KIND:
        try:
            available_repeated_families = within_family_repeated_panel_families(panel)
        except EditingT1PanelError as error:
            raise SystemExit(str(error)) from error
        if args.family not in available_repeated_families:
            raise SystemExit(
                "no empirical within-family repeated multi-successor panel "
                f"exists for {args.family}"
            )
    active8_identity = validate_t1_active8_runtime_binding(
        contract=t1_contract,
        gate_zero_contract=gate_zero_contract,
        source=source,
        panel=panel,
        active8_admission=active8_admission,
    )
    if args.build_successor_cache:
        if args.device != "cpu":
            raise SystemExit("T1 successor caches must be compiled on canonical CPU/FP32")
        if args.successor_cache_root is None or args.output is None:
            raise SystemExit(
                "T1 cache build requires --successor-cache-root and --output receipt"
            )
        if args.successor_cache_receipt is not None:
            raise SystemExit("T1 cache build cannot consume another cache receipt")
        model, parity = build_scratch_ringcore_model(gate_zero_contract)
        cache_identity = build_t1_successor_cache_identity(
            contract=t1_contract,
            gate_zero_contract=gate_zero_contract,
            source=source,
            panel=panel,
            model=model,
            active8_admission=active8_admission,
            compiler_device="cpu",
        )
        resolved_panel = resolve_t1_panel_view(
            source=source,
            panel=panel,
            forensics=forensics,
            family=args.family,
            panel_kind=args.panel_kind,
            max_atoms=int(gate_zero_contract.model["max_atoms"]),
            excluded_trace_ids=excluded_trace_ids,
            active8_admission=active8_admission,
        )
        selected_shards = {
            record.corpus_address.packed_shard_content_sha256
            for record in resolved_panel.selected_records
            if record.corpus_address is not None
        }
        provenance = build_exact_cache_provenance(
            model,
            source,
            selected_shard_digests=selected_shards,
            max_atoms=int(gate_zero_contract.model["max_atoms"]),
        )
        manifest, receipt = build_t1_successor_cache(
            args.successor_cache_root,
            model,
            resolved_panel.selected_records,
            provenance_by_shard=provenance,
            identity=cache_identity,
        )
        _write_if_absent(args.output, asdict(receipt))
        print(
            json.dumps(
                _successor_cache_completion_payload(
                    family=args.family,
                    panel_kind=args.panel_kind,
                    manifest=manifest,
                    receipt=receipt,
                    initialization_parity=asdict(parity),
                ),
                sort_keys=True,
            )
        )
        return 0
    if args.validate_successor_cache and args.device != "cpu":
        raise SystemExit(
            "T1 successor caches must be validated on CPU before GPU dispatch"
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    successor_cache = None
    if not args.audit_only:
        if args.successor_cache_root is None or args.successor_cache_receipt is None:
            raise SystemExit(
                "T1 optimization requires a frozen --successor-cache-root and "
                "--successor-cache-receipt"
            )
        identity_model, _identity_parity = build_scratch_ringcore_model(
            gate_zero_contract
        )
        expected_cache_identity = build_t1_successor_cache_identity(
            contract=t1_contract,
            gate_zero_contract=gate_zero_contract,
            source=source,
            panel=panel,
            model=identity_model,
            active8_admission=active8_admission,
            compiler_device="cpu",
        )
        resolved_cache_panel = resolve_t1_panel_view(
            source=source,
            panel=panel,
            forensics=forensics,
            family=args.family,
            panel_kind=args.panel_kind,
            max_atoms=int(gate_zero_contract.model["max_atoms"]),
            excluded_trace_ids=excluded_trace_ids,
            active8_admission=active8_admission,
        )
        cache_receipt = _load_successor_cache_receipt(args.successor_cache_receipt)
        expected_trace_set_sha256 = t1_selected_trace_set_sha256(
            resolved_cache_panel.selected_records
        )
        if cache_receipt.selected_trace_set_sha256 != expected_trace_set_sha256:
            raise SystemExit(
                "T1 successor-cache receipt is bound to another exact panel trace set"
            )
        successor_cache = load_t1_successor_cache(
            args.successor_cache_root,
            cache_receipt,
            expected_identity=expected_cache_identity,
        )
        if args.validate_successor_cache:
            leaf_receipts = successor_cache.validate_all_leaves()
            print(
                json.dumps(
                    {
                        "status": "T1_SUCCESSOR_CACHE_VALIDATED_NO_OPTIMIZATION",
                        "training_authorized": False,
                        "gate_decision": None,
                        "family": args.family,
                        "panel_kind": args.panel_kind,
                        "selected_trace_set_sha256": expected_trace_set_sha256,
                        "validated_leaf_count": len(leaf_receipts),
                        "cache_receipts": [asdict(receipt) for receipt in leaf_receipts],
                    },
                    sort_keys=True,
                )
            )
            return 0
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
            active8_admission=active8_admission,
            successor_cache=successor_cache,
        )
        result = {
            "status": "T1_PANEL_CACHE_AUDIT_COMPLETE_NO_OPTIMIZATION",
            "training_authorized": False,
            "gate_decision": None,
            "contract_sha256": t1_contract.sha256,
            "numeric_thresholds_frozen": True,
            "numeric_thresholds_sha256": (t1_contract.numeric_thresholds_sha256),
            "panel_artifact_sha256": panel["artifact_sha256"],
            "charge_policy_exclusion_payload_sha256": (charge_policy_exclusions["payload_sha256"]),
            "charge_policy_source_input_inventory_sha256": (
                charge_policy_exclusions["source_input_inventory_sha256"]
            ),
            **active8_t1_identity(active8_admission),
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
            active8_admission=active8_admission,
            successor_cache=successor_cache,
        )
    if args.output is not None:
        _write_if_absent(args.output, result)
        if not args.audit_only:
            load_editing_t1_result(
                args.output,
                expected_result_sha256=result["result_sha256"],
                expected_contract_sha256=t1_contract.sha256,
                expected_numeric_thresholds_sha256=(t1_contract.numeric_thresholds_sha256),
                expected_panel_artifact_sha256=panel["artifact_sha256"],
                expected_panel_selection_sha256=(editing_t1_panel_selection_sha256(panel)),
                expected_panel_census_sha256=(editing_t1_panel_census_sha256(panel)),
                expected_panel_capacity_strata_sha256=(
                    editing_t1_panel_capacity_strata_sha256(panel)
                ),
                expected_active8_identity=active8_identity,
                expected_cache_receipts=result["cache_receipts"],
            )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
