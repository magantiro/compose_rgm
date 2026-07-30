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
    load_gate_zero_runtime_contract,
)
from compose_v4.experiments.editing_t1_panel import (  # noqa: E402
    build_editing_t1_panel,
    validate_editing_t1_panel,
    write_editing_t1_panel,
)
from compose_v4.experiments.ringcore_semantic_sidecar import (  # noqa: E402
    read_semantic_cell_sidecar,
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    gate_zero_contract = load_gate_zero_runtime_contract(
        args.gate_zero_contract
    )
    active8_admission = load_active8_trace_admission(
        args.active8_inventory,
        expected_manifest_file_sha256=(
            args.active8_inventory_file_sha256
        ),
        expected_support_contract_sha256=gate_zero_contract.sha256,
    )
    gate_zero_unified_packed_manifest_sha256 = str(
        gate_zero_contract.sidecar["unified_packed_manifest_sha256"]
    )
    if (
        active8_admission.unified_packed_manifest_sha256
        != gate_zero_unified_packed_manifest_sha256
    ):
        raise SystemExit(
            "Gate0 validation source and Active8 inventory name different "
            "unified packed manifests"
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
    sidecar_manifest = load_json_object(args.semantic_sidecar_manifest)
    sidecar = read_semantic_cell_sidecar(
        args.semantic_sidecar,
        args.semantic_sidecar_manifest,
        expected_manifest_sha256=sidecar_manifest["manifest_sha256"],
        expected_config_sha256=sidecar_manifest["config_sha256"],
        expected_provenance_sha256=sidecar_manifest["provenance_sha256"],
    )
    kwargs = {
        "forensics": forensics,
        "forensics_file_sha256": _sha256(args.forensics),
        "sidecar": sidecar,
        "semantic_sidecar_file_sha256": _sha256(args.semantic_sidecar),
        "semantic_sidecar_manifest_file_sha256": _sha256(args.semantic_sidecar_manifest),
        "charge_policy_audit": load_json_object(args.charge_policy_audit),
        "charge_policy_audit_file_sha256": _sha256(args.charge_policy_audit),
        "charge_policy_exclusions": load_json_object(args.charge_policy_exclusions),
        "charge_policy_exclusions_file_sha256": _sha256(args.charge_policy_exclusions),
        "active8_admission": active8_admission,
        "gate_zero_unified_packed_manifest_sha256": (
            gate_zero_unified_packed_manifest_sha256
        ),
    }
    panel = build_editing_t1_panel(**kwargs)
    validate_editing_t1_panel(panel, **kwargs)
    write_editing_t1_panel(panel, args.output)
    print(
        json.dumps(
            {
                "artifact_sha256": panel["artifact_sha256"],
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
