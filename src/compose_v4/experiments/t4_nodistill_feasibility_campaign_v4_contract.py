"""Self-hashed contract for the three-cell route-free NoDistill v4 campaign."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from compose_v4.experiments.t4_shared_controller_completion_contract import (
    build_exact_commit_capsule,
    payload_identity,
    sha256_file,
    verify_exact_commit_capsule,
)
from compose_v4.experiments.t4_shared_controller_scored_contract import (
    AUTHORIZATION_SCHEMA_VERSION,
)

SCHEMA_VERSION = "t4_nodistill_feasibility_campaign_scored_contract_v4"
PREPARATION_RELATIVE_PATH = "configs/t4_nodistill_feasibility_campaign_v4.json"
SUPPORT_GATE_RELATIVE_PATH = (
    "diagnostics/t4_nodistill_feasibility_headroom_gate_v4/attempt_1/result.json"
)
SUPPORT_GATE_PAYLOAD_SHA256 = (
    "f49cb2fd734350261e247006a28f626d44843bce1978fffce7a2b0631669c87f"
)
FINAL_CONTRACT_RELATIVE_PATH = (
    "diagnostics/t4_nodistill_feasibility_campaign_v4/attempt_1/" "scored_contract.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "diagnostics/t4_nodistill_feasibility_campaign_v4/attempt_1/"
    "scored_authorization.json"
)
CAPSULE_ROOT_RELATIVE_PATH = (
    "diagnostics/t4_nodistill_feasibility_campaign_v4/attempt_1/source_capsule"
)
CAPSULE_MANIFEST_RELATIVE_PATH = (
    "diagnostics/t4_nodistill_feasibility_campaign_v4/attempt_1/"
    "source_capsule_manifest.json"
)
CAPSULE_INCLUDE = (
    PREPARATION_RELATIVE_PATH,
    SUPPORT_GATE_RELATIVE_PATH,
    "docs/GENMOL_T4_SEEDS.json",
    "modal_apps/t4_nodistill_feasibility_v4_app.py",
    "modal_apps/t4_shared_controller_completion_v1_app.py",
    "src/compose_v4",
    "tools/launch_t4_nodistill_feasibility_campaign_v4.py",
)
OPERATIONAL_SETTINGS = {
    "proposal_wait_seconds": 7_800,
    "query_wait_seconds": 1_800,
    "driver_poll_seconds": 10,
    "retained_core": {
        "maximum_fragment_atoms": 16,
        "maximum_stages": 2,
        "maximum_primitives": 32,
        "maximum_prefixes": 4_096,
        "program_family": "retained_core_prune",
        "proposal_expert": "route_complete_region",
    },
    "complete_combination_particles": {
        "enabled": False,
        "semantics": "disabled_with_explicit_trajectory_distillation",
        "jobs_per_parent": 0,
    },
}


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected one JSON object: {path}")
    return value


def _load_envelope(path: Path) -> tuple[dict[str, Any], str]:
    envelope = _load_object(path)
    payload = envelope.get("payload")
    identity = envelope.get("payload_sha256")
    if not isinstance(payload, dict) or payload_identity(payload) != identity:
        raise ValueError(f"invalid self-hashed envelope: {path}")
    return payload, str(identity)


def _publish_once(path: Path, payload: dict[str, Any]) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite sealed artifact: {path}")
    identity = payload_identity(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {"payload": payload, "payload_sha256": identity},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    temporary.replace(path)
    return identity


def required_authorization_statement(contract_payload_sha256: str) -> str:
    return (
        "I authorize the 147 scored calls in the route-free NoDistill v4 "
        f"contract payload {contract_payload_sha256}."
    )


def _validate_preparation(path: Path) -> dict[str, Any]:
    payload = _load_object(path)
    if payload.get("schema_version") != (
        "t4_nodistill_feasibility_campaign_preparation_v4"
    ):
        raise ValueError("unexpected NoDistill v4 campaign preparation schema")
    if payload.get("status") != "PREPARED_FOR_PAYLOAD_BOUND_SCORED_LAUNCH":
        raise ValueError("NoDistill v4 campaign preparation status drift")
    expected_cells = [
        ("jak2_0_d06", "jak2_0", "jak2", 12, 0.6, 2026091912),
        ("parp1_0_d04", "parp1_0", "parp1", 0, 0.4, 2026091900),
        ("5ht1b_1_d04", "5ht1b_1", "5ht1b", 7, 0.4, 2026091907),
    ]
    cells = payload.get("cells")
    if not isinstance(cells, list):
        raise TypeError("NoDistill v4 campaign cells must be a list")
    observed_cells = [
        (
            row.get("cell_key"),
            row.get("source_cell"),
            row.get("target"),
            row.get("source_global_index"),
            row.get("delta"),
            row.get("controller_seed"),
        )
        for row in cells
    ]
    if observed_cells != expected_cells:
        raise ValueError("NoDistill v4 campaign cell matrix drift")
    if (
        payload.get("campaign_cell_count") != 3
        or payload.get("charged_calls_per_cell") != 49
        or payload.get("total_charged_call_ceiling") != 147
    ):
        raise ValueError("NoDistill v4 campaign budget drift")
    if payload.get("trajectory_distillation") != {
        "enabled": False,
        "stored_structural_template_library": False,
        "route_marginal_or_scale_weights": False,
        "direct_template_rebinding": False,
        "route_template_particles": False,
        "route_template_quota": False,
        "target_independent_scale_floor": True,
        "route_checkpoint_allowed": False,
    }:
        raise ValueError("NoDistill v4 trajectory distillation was not disabled")
    controller = payload.get("controller")
    if not isinstance(controller, dict):
        raise TypeError("NoDistill v4 controller is missing")
    if (
        controller.get("batch") != 8
        or controller.get("parents") != 4
        or controller.get("expert_floor_rounds") != 2
        or controller.get("route_scale_floor_rounds") != 2
        or controller.get("scale_floor_scope") != "all_generic"
    ):
        raise ValueError("NoDistill v4 controller allocation drift")
    expected_experts = [
        "shallow",
        "anchored_replacement",
        "route_complete_region",
        "generic_feasibility_headroom_v4",
    ]
    if controller.get("proposal_experts") != expected_experts:
        raise ValueError("NoDistill v4 expert vocabulary is not append-only")
    route = controller.get("proposal", {}).get("route_complete_region")
    if route != {
        "mode": "generic_retained_core_only",
        "distilled_templates_enabled": False,
    }:
        raise ValueError("NoDistill v4 route lane contains distilled templates")
    v4 = controller.get("proposal", {}).get("generic_feasibility_headroom_v4")
    if not isinstance(v4, dict):
        raise TypeError("NoDistill v4 proposal settings are missing")
    expected_support = {
        "attempts_per_local_plan": 128,
        "candidate_quota_per_local_plan": 8,
        "maximum_primitives": 32,
        "maximum_blocks": 8,
        "maximum_heavy_atoms": 40,
        "qed_minimum": 0.6,
        "sa_maximum": 4.0,
        "similarity_minimum_source": "numeric_cell_fiber_threshold",
    }
    if any(v4.get(key) != value for key, value in expected_support.items()):
        raise ValueError("NoDistill v4 proposal support or work allocation drift")
    if v4.get("joint_stop") != {
        "min_depth": 2,
        "max_depth": 4,
        "beam_width": 24,
        "lock_width": 32,
        "candidates_per_family": 6,
        "expansions_per_parent": 24,
        "max_primitives": 32,
        "max_blocks": 8,
        "max_active_atoms": 40,
    }:
        raise ValueError("NoDistill v4 joint STOP settings drift")
    if "shared_route_checkpoint" in payload:
        raise ValueError("NoDistill v4 may not bind a route checkpoint")
    if payload.get("oracle_policy") != {
        "automatic_retries": 0,
        "replacement": False,
        "backfill": False,
        "prior_scored_outcomes_read": False,
        "runtime_comparator": False,
        "teacher_or_endpoint_injection": False,
    }:
        raise ValueError("NoDistill v4 oracle policy drift")
    return payload


def _validate_support_gate(path: Path) -> tuple[str, dict[str, dict[str, int]]]:
    payload, identity = _load_envelope(path)
    if identity != SUPPORT_GATE_PAYLOAD_SHA256:
        raise ValueError("NoDistill v4 support-gate payload identity drift")
    if (
        payload.get("schema_version")
        != "t4_nodistill_feasibility_headroom_gate_result_v4"
        or payload.get("decision") != "PASS_FEASIBILITY_HEADROOM_SUPPORT_GATE"
        or payload.get("costs")
        != {
            "docking_calls": 0,
            "gpu_seconds": 0,
            "modal_launches": 0,
            "oracle_calls": 0,
        }
        or payload.get("gates", {}).get("passed") is not True
        or not all(payload.get("gates", {}).get("values", {}).values())
    ):
        raise ValueError("NoDistill v4 support gate did not pass unchanged")
    expected_cells = {"jak2_0_d06", "parp1_0_d04", "5ht1b_1_d04"}
    support: dict[str, dict[str, int]] = {}
    for row in payload.get("cells", ()):  # frozen result contains two contrasts too
        key = row.get("cell_key")
        if key not in expected_cells:
            continue
        preteacher = row.get("preteacher_support", {})
        counts = {
            "archive_ready_unique": int(
                preteacher.get("archive_ready_unique_candidates", 0)
            ),
            "macro_plan_identities": int(
                preteacher.get("archive_ready_macro_plan_identity_count", 0)
            ),
            "macro_modes": int(preteacher.get("archive_ready_mode_count", 0)),
            "structural_signatures": int(
                preteacher.get("archive_ready_structural_signature_count", 0)
            ),
        }
        if (
            counts["archive_ready_unique"] < 8
            or counts["macro_plan_identities"] < 3
            or counts["macro_modes"] < 3
            or counts["structural_signatures"] < 3
        ):
            raise ValueError(f"NoDistill v4 support shortfall for {key}")
        support[str(key)] = counts
    if set(support) != expected_cells:
        raise ValueError("NoDistill v4 motivating support census drift")
    return identity, support


def seal_scored_contract(
    *,
    repository_root: Path,
    capsule_root: Path,
    capsule_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    root = repository_root.resolve()
    preparation_path = root / PREPARATION_RELATIVE_PATH
    preparation = _validate_preparation(preparation_path)
    support_path = root / SUPPORT_GATE_RELATIVE_PATH
    support_identity, support_counts = _validate_support_gate(support_path)
    capsule = verify_exact_commit_capsule(capsule_root, capsule_manifest_path)
    capsule_manifest, _ = _load_envelope(capsule_manifest_path)
    if capsule_manifest.get("include") != sorted(CAPSULE_INCLUDE):
        raise ValueError("NoDistill v4 source-capsule include set drift")
    if any(
        path.startswith("diagnostics/") and "checkpoint" in path
        for path in capsule_manifest["files"]
    ):
        raise ValueError("NoDistill v4 source capsule contains a checkpoint")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "SEALED_PENDING_EXACT_USER_AUTHORIZATION",
        "scored_calls_requested": 147,
        "charged_calls_per_cell": 49,
        "total_charged_call_ceiling": 147,
        "campaign_cell_count": 3,
        "automatic_retries": 0,
        "replacement": False,
        "backfill": False,
        "support": preparation["support"],
        "controller": preparation["controller"],
        "cells": preparation["cells"],
        "receipt_policy": preparation["receipt_policy"],
        "trajectory_distillation": preparation["trajectory_distillation"],
        "generic_capabilities_retained": preparation["generic_capabilities_retained"],
        "oracle_policy": preparation["oracle_policy"],
        "operational_settings": OPERATIONAL_SETTINGS,
        "preparation_contract": {
            "path": PREPARATION_RELATIVE_PATH,
            "sha256": sha256_file(preparation_path),
        },
        "zero_oracle_support_gate": {
            "path": SUPPORT_GATE_RELATIVE_PATH,
            "sha256": sha256_file(support_path),
            "payload_sha256": support_identity,
            "archive_ready_support": support_counts,
            "oracle_calls": 0,
            "docking_calls": 0,
        },
        "source_capsule": {
            **capsule,
            "root_relative_path": CAPSULE_ROOT_RELATIVE_PATH,
            "manifest_relative_path": CAPSULE_MANIFEST_RELATIVE_PATH,
            "manifest_sha256": sha256_file(capsule_manifest_path),
        },
        "authorization": {
            "state": "PENDING_EXACT_PAYLOAD_BOUND_USER_AUTHORIZATION",
            "receipt_path": AUTHORIZATION_RELATIVE_PATH,
            "required_format": (
                "I authorize the 147 scored calls in the route-free NoDistill "
                "v4 contract payload <payload_sha256>."
            ),
        },
        "claim_boundary": (
            "Prospective three-cell test of the frozen route-free v4 proposal "
            "mechanism; not a full T4 benchmark or evidence of docking utility "
            "before the separately authorized calls complete"
        ),
    }
    identity = _publish_once(output_path, payload)
    return {
        "contract_path": str(output_path),
        "contract_sha256": sha256_file(output_path),
        "contract_payload_sha256": identity,
        "authorized_scored_calls": 147,
        "required_user_statement": required_authorization_statement(identity),
    }


def publish_authorization_receipt(
    *, contract_path: Path, authorization_path: Path, user_statement: str
) -> dict[str, Any]:
    contract, contract_identity = _load_envelope(contract_path)
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected NoDistill v4 scored contract schema")
    required = required_authorization_statement(contract_identity)
    if user_statement != required:
        raise ValueError(f"authorization statement must exactly equal: {required}")
    payload = {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "contract_payload_sha256": contract_identity,
        "authorized_scored_calls": 147,
        "user_statement": user_statement,
    }
    identity = _publish_once(authorization_path, payload)
    return {
        "authorization_path": str(authorization_path),
        "authorization_sha256": sha256_file(authorization_path),
        "authorization_payload_sha256": identity,
    }


def validate_scored_contract(
    *,
    repository_root: Path,
    contract_path: Path,
    authorization_path: Path,
    capsule_root: Path,
    capsule_manifest_path: Path,
) -> dict[str, Any]:
    root = repository_root.resolve()
    contract, contract_identity = _load_envelope(contract_path)
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected NoDistill v4 scored contract schema")
    if contract.get("status") != "SEALED_PENDING_EXACT_USER_AUTHORIZATION":
        raise ValueError("NoDistill v4 scored contract status drift")
    preparation_path = root / PREPARATION_RELATIVE_PATH
    preparation = _validate_preparation(preparation_path)
    expected = {
        "campaign_cell_count": 3,
        "scored_calls_requested": 147,
        "charged_calls_per_cell": 49,
        "total_charged_call_ceiling": 147,
        "automatic_retries": 0,
        "replacement": False,
        "backfill": False,
        "support": "compose_valid",
        "controller": preparation["controller"],
        "cells": preparation["cells"],
        "receipt_policy": preparation["receipt_policy"],
        "trajectory_distillation": preparation["trajectory_distillation"],
        "generic_capabilities_retained": preparation["generic_capabilities_retained"],
        "oracle_policy": preparation["oracle_policy"],
        "operational_settings": OPERATIONAL_SETTINGS,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"NoDistill v4 scored contract drift: {key}")
    if "shared_route_checkpoint" in contract:
        raise ValueError("NoDistill v4 may not bind a route checkpoint")
    if contract.get("preparation_contract") != {
        "path": PREPARATION_RELATIVE_PATH,
        "sha256": sha256_file(preparation_path),
    }:
        raise ValueError("NoDistill v4 preparation binding drift")
    support_path = root / SUPPORT_GATE_RELATIVE_PATH
    support_identity, support_counts = _validate_support_gate(support_path)
    if contract.get("zero_oracle_support_gate") != {
        "path": SUPPORT_GATE_RELATIVE_PATH,
        "sha256": sha256_file(support_path),
        "payload_sha256": support_identity,
        "archive_ready_support": support_counts,
        "oracle_calls": 0,
        "docking_calls": 0,
    }:
        raise ValueError("NoDistill v4 support binding drift")
    capsule = verify_exact_commit_capsule(capsule_root, capsule_manifest_path)
    capsule_manifest, _ = _load_envelope(capsule_manifest_path)
    if capsule_manifest.get("include") != sorted(CAPSULE_INCLUDE):
        raise ValueError("NoDistill v4 source-capsule include set drift")
    for key, value in capsule.items():
        if contract["source_capsule"].get(key) != value:
            raise ValueError(f"NoDistill v4 source-capsule drift: {key}")
    if (
        contract["source_capsule"].get("root_relative_path")
        != CAPSULE_ROOT_RELATIVE_PATH
        or contract["source_capsule"].get("manifest_relative_path")
        != CAPSULE_MANIFEST_RELATIVE_PATH
        or contract["source_capsule"].get("manifest_sha256")
        != sha256_file(capsule_manifest_path)
    ):
        raise ValueError("NoDistill v4 source-capsule path drift")
    authorization, _ = _load_envelope(authorization_path)
    required = required_authorization_statement(contract_identity)
    if authorization != {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "contract_payload_sha256": contract_identity,
        "authorized_scored_calls": 147,
        "user_statement": required,
    }:
        raise ValueError("NoDistill v4 authorization does not match contract")
    return contract


__all__ = [
    "AUTHORIZATION_RELATIVE_PATH",
    "CAPSULE_INCLUDE",
    "CAPSULE_MANIFEST_RELATIVE_PATH",
    "CAPSULE_ROOT_RELATIVE_PATH",
    "FINAL_CONTRACT_RELATIVE_PATH",
    "OPERATIONAL_SETTINGS",
    "PREPARATION_RELATIVE_PATH",
    "SCHEMA_VERSION",
    "SUPPORT_GATE_PAYLOAD_SHA256",
    "SUPPORT_GATE_RELATIVE_PATH",
    "build_exact_commit_capsule",
    "publish_authorization_receipt",
    "required_authorization_statement",
    "seal_scored_contract",
    "validate_scored_contract",
]
