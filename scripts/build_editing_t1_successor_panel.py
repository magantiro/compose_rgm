#!/usr/bin/env python3
"""Freeze the exact-address T1 successor capacity panels.

This command performs no optimization and cannot authorize training.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compose_v4.data.active8_trace_inventory import (  # noqa: E402
    load_active8_trace_admission,
)
from compose_v4.experiments.editing_gate_zero_runtime import (  # noqa: E402
    build_scratch_ringcore_model,
    load_frozen_validation_source,
    load_gate_zero_runtime_contract,
)
from compose_v4.experiments.editing_t1_panel import (  # noqa: E402
    EDITING_T1_MAX_CAPACITY_WORKERS,
    build_editing_t1_capacity_census,
    build_editing_t1_panel,
    write_editing_t1_capacity_census,
    write_editing_t1_panel,
)
from compose_v4.experiments.ringcore_successor_leaderboard import (  # noqa: E402
    load_json_object,
)
from compose_v4.experiments.ringcore_validation_panel import (  # noqa: E402
    validate_validation_panel_artifact,
)


def _capacity_workers(value: str) -> int:
    try:
        workers = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("workers must be an integer") from error
    if not 1 <= workers <= EDITING_T1_MAX_CAPACITY_WORKERS:
        raise argparse.ArgumentTypeError(
            f"workers must be between 1 and {EDITING_T1_MAX_CAPACITY_WORKERS}"
        )
    return workers


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gate-zero-contract",
        type=Path,
        default=ROOT / "configs" / "editing_gate_zero_runtime_v2.json",
    )
    parser.add_argument("--transfer-root", type=Path, required=True)
    parser.add_argument("--active8-inventory", type=Path, required=True)
    parser.add_argument(
        "--active8-inventory-file-sha256",
        required=True,
        help="Expected SHA-256 of the physical Active8 inventory manifest.",
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
    parser.add_argument(
        "--workers",
        type=_capacity_workers,
        default=1,
        help=(
            "Local worker processes for independent exact source-state "
            f"compilations (1-{EDITING_T1_MAX_CAPACITY_WORKERS}; default: 1)."
        ),
    )
    parser.add_argument("--capacity-census-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    gate_zero_contract = load_gate_zero_runtime_contract(args.gate_zero_contract)
    active8_admission = load_active8_trace_admission(
        args.active8_inventory,
        expected_manifest_file_sha256=(args.active8_inventory_file_sha256),
        expected_support_contract_sha256=gate_zero_contract.sha256,
    )
    gate_zero_unified_packed_manifest_sha256 = str(
        gate_zero_contract.sidecar["unified_packed_manifest_sha256"]
    )
    if active8_admission.unified_packed_manifest_sha256 != gate_zero_unified_packed_manifest_sha256:
        raise SystemExit(
            "Gate0 validation source and Active8 inventory name different unified packed manifests"
        )
    with gzip.open(args.forensics, "rt") as handle:
        forensics = json.load(handle)
    config = load_json_object(args.leaderboard_config)
    inventory = load_json_object(args.inventory)
    validate_validation_panel_artifact(
        forensics,
        config=config,
        inventory=inventory,
    )
    source = load_frozen_validation_source(
        contract=gate_zero_contract,
        transfer_root=args.transfer_root,
        sidecar_path=args.semantic_sidecar,
        sidecar_manifest_path=args.semantic_sidecar_manifest,
    )
    model, _parity = build_scratch_ringcore_model(gate_zero_contract)
    capacity_census = build_editing_t1_capacity_census(
        model,
        source=source,
        forensics=forensics,
        forensics_file_sha256=_sha256(args.forensics),
        active8_admission=active8_admission,
        gate_zero_runtime_contract_sha256=gate_zero_contract.sha256,
        workers=args.workers,
    )
    write_editing_t1_capacity_census(
        capacity_census,
        args.capacity_census_output,
    )
    kwargs = {
        "model": model,
        "source": source,
        "forensics": forensics,
        "forensics_file_sha256": _sha256(args.forensics),
        "sidecar": source.sidecar,
        "semantic_sidecar_file_sha256": _sha256(args.semantic_sidecar),
        "semantic_sidecar_manifest_file_sha256": _sha256(args.semantic_sidecar_manifest),
        "charge_policy_audit": load_json_object(args.charge_policy_audit),
        "charge_policy_audit_file_sha256": _sha256(args.charge_policy_audit),
        "charge_policy_exclusions": load_json_object(args.charge_policy_exclusions),
        "charge_policy_exclusions_file_sha256": _sha256(args.charge_policy_exclusions),
        "capacity_census": capacity_census,
        "capacity_census_file_sha256": _sha256(args.capacity_census_output),
        "active8_admission": active8_admission,
        "gate_zero_runtime_contract_sha256": gate_zero_contract.sha256,
        "gate_zero_unified_packed_manifest_sha256": (gate_zero_unified_packed_manifest_sha256),
        "capacity_workers": args.workers,
    }
    panel = build_editing_t1_panel(**kwargs)
    write_editing_t1_panel(panel, args.output)
    print(
        json.dumps(
            {
                "artifact_sha256": panel["artifact_sha256"],
                "capacity_census_file_sha256": _sha256(args.capacity_census_output),
                "capacity_census_sha256": capacity_census["census_sha256"],
                "capacity_workers": args.workers,
                "census": panel["census"],
                "output": str(args.output),
                "status": panel["status"],
                "training_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
