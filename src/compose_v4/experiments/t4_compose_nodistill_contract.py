"""Self-hashed contract for the matched PARP1 COMPOSE-NoDistill campaign."""

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

SCHEMA_VERSION = "t4_compose_nodistill_parp1_scored_contract_v1"
PREPARATION_RELATIVE_PATH = "configs/t4_compose_nodistill_parp1_v1.json"
FINAL_CONTRACT_RELATIVE_PATH = (
    "diagnostics/t4_compose_nodistill_parp1_v1/attempt_2/scored_contract.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "diagnostics/t4_compose_nodistill_parp1_v1/attempt_2/scored_authorization.json"
)
CAPSULE_ROOT_RELATIVE_PATH = (
    "diagnostics/t4_compose_nodistill_parp1_v1/attempt_2/source_capsule"
)
CAPSULE_MANIFEST_RELATIVE_PATH = (
    "diagnostics/t4_compose_nodistill_parp1_v1/attempt_2/source_capsule_manifest.json"
)
SUPPORT_RELATIVE_PATH = (
    "diagnostics/t4_retained_core_shared_support_v1/attempt_1/scientific_result.json"
)
CAPSULE_INCLUDE = (
    PREPARATION_RELATIVE_PATH,
    SUPPORT_RELATIVE_PATH,
    "docs/GENMOL_T4_SEEDS.json",
    "modal_apps/t4_compose_nodistill_parp1_v1_app.py",
    "modal_apps/t4_shared_controller_completion_v1_app.py",
    "src/compose_v4",
    "tools/launch_t4_shared_controller_deployed.py",
)
USER_AUTHORIZATION_STATEMENT = (
    "i explicitly authorize you start whatever run this yeidls btw"
)
OPERATIONAL_SETTINGS = {
    "proposal_wait_seconds": 2_400,
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


def _validate_preparation(path: Path) -> dict[str, Any]:
    payload = _load_object(path)
    if payload.get("schema_version") != ("t4_compose_nodistill_parp1_preparation_v1"):
        raise ValueError("unexpected NoDistill preparation schema")
    if payload.get("status") != "PREPARED_FOR_PAYLOAD_BOUND_SCORED_LAUNCH":
        raise ValueError("NoDistill preparation status drift")
    cells = payload.get("cells")
    if (
        not isinstance(cells, list)
        or len(cells) != 3
        or [row.get("cell_key") for row in cells]
        != ["parp1_0_d04", "parp1_1_d04", "parp1_2_d04"]
        or any(row.get("target") != "parp1" or row.get("delta") != 0.4 for row in cells)
    ):
        raise ValueError("NoDistill PARP1 cell matrix drift")
    if (
        payload.get("campaign_cell_count") != 3
        or payload.get("charged_calls_per_cell") != 49
        or payload.get("total_charged_call_ceiling") != 147
    ):
        raise ValueError("NoDistill budget drift")
    distillation = payload.get("trajectory_distillation")
    if distillation != {
        "enabled": False,
        "stored_structural_template_library": False,
        "route_marginal_or_scale_weights": False,
        "direct_template_rebinding": False,
        "route_template_particles": False,
        "route_template_quota": False,
        "target_independent_scale_floor": True,
        "route_checkpoint_allowed": False,
    }:
        raise ValueError("explicit trajectory distillation was not fully disabled")
    controller = payload.get("controller")
    if not isinstance(controller, dict):
        raise TypeError("NoDistill controller is missing")
    if (
        controller.get("route_scale_floor_rounds") != 2
        or controller.get("scale_floor_scope") != "all_generic"
    ):
        raise ValueError("NoDistill generic transformation-scale floor drift")
    route = controller.get("proposal", {}).get("route_complete_region")
    if route != {
        "mode": "generic_retained_core_only",
        "distilled_templates_enabled": False,
    }:
        raise ValueError("NoDistill route lane contains a distilled template source")
    if "shared_route_checkpoint" in payload:
        raise ValueError("NoDistill preparation may not bind a route checkpoint")
    oracle = payload.get("oracle_policy")
    if oracle != {
        "automatic_retries": 0,
        "replacement": False,
        "backfill": False,
        "prior_scored_outcomes_read": False,
        "runtime_comparator": False,
        "teacher_or_endpoint_injection": False,
    }:
        raise ValueError("NoDistill oracle policy drift")
    return payload


def _validate_support(path: Path, cell_keys: list[str]) -> tuple[str, dict[str, int]]:
    payload, identity = _load_envelope(path)
    if payload.get("schema_version") != (
        "t4_retained_core_shared_support_scientific_v1"
    ):
        raise ValueError("unexpected retained-core support schema")
    if payload.get("costs") != {
        "oracle_calls": 0,
        "docking_calls": 0,
        "modal_launches": 0,
    }:
        raise ValueError("NoDistill support evidence is not zero-oracle")
    census = {
        row["cell_key"]: int(row["eligible_unique"]) for row in payload.get("cells", ())
    }
    selected = {key: census.get(key, 0) for key in cell_keys}
    if any(count < 1 for count in selected.values()):
        raise ValueError("NoDistill generic retained-core support gate failed")
    return identity, selected


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
    cell_keys = [row["cell_key"] for row in preparation["cells"]]
    support_path = root / SUPPORT_RELATIVE_PATH
    support_identity, selected_support = _validate_support(support_path, cell_keys)
    capsule = verify_exact_commit_capsule(capsule_root, capsule_manifest_path)
    capsule_manifest, _ = _load_envelope(capsule_manifest_path)
    if capsule_manifest.get("include") != sorted(CAPSULE_INCLUDE):
        raise ValueError("NoDistill source capsule include set drift")
    if any(
        path.startswith("diagnostics/") and "checkpoint" in path
        for path in capsule_manifest["files"]
    ):
        raise ValueError("NoDistill capsule contains a checkpoint")
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
        "operational_settings": OPERATIONAL_SETTINGS,
        "preparation_contract": {
            "path": PREPARATION_RELATIVE_PATH,
            "sha256": sha256_file(preparation_path),
        },
        "generic_support_gate": {
            "path": SUPPORT_RELATIVE_PATH,
            "sha256": sha256_file(support_path),
            "payload_sha256": support_identity,
            "eligible_unique_by_cell": selected_support,
            "oracle_calls": 0,
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
            "authorized_statement": USER_AUTHORIZATION_STATEMENT,
        },
        "claim_boundary": (
            "Matched three-cell PARP1 delta-0.4 causal ablation of explicit "
            "trajectory-distilled proposal templates; not a new controller"
        ),
    }
    identity = _publish_once(output_path, payload)
    return {
        "contract_path": str(output_path),
        "contract_sha256": sha256_file(output_path),
        "contract_payload_sha256": identity,
        "authorized_scored_calls": 147,
    }


def publish_authorization_receipt(
    *, contract_path: Path, authorization_path: Path, user_statement: str
) -> dict[str, Any]:
    contract, contract_identity = _load_envelope(contract_path)
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected NoDistill scored contract schema")
    if user_statement != USER_AUTHORIZATION_STATEMENT:
        raise ValueError("NoDistill authorization statement drift")
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
        raise ValueError("unexpected NoDistill scored contract schema")
    if contract.get("status") != "SEALED_PENDING_EXACT_USER_AUTHORIZATION":
        raise ValueError("NoDistill scored contract status drift")
    preparation_path = root / PREPARATION_RELATIVE_PATH
    preparation = _validate_preparation(preparation_path)
    exact = {
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
        "operational_settings": OPERATIONAL_SETTINGS,
    }
    for key, value in exact.items():
        if contract.get(key) != value:
            raise ValueError(f"NoDistill scored contract drift: {key}")
    if "shared_route_checkpoint" in contract:
        raise ValueError("NoDistill scored contract may not bind a route checkpoint")
    if contract.get("preparation_contract") != {
        "path": PREPARATION_RELATIVE_PATH,
        "sha256": sha256_file(preparation_path),
    }:
        raise ValueError("NoDistill preparation binding drift")
    support_path = root / SUPPORT_RELATIVE_PATH
    support_identity, selected_support = _validate_support(
        support_path, [row["cell_key"] for row in preparation["cells"]]
    )
    if contract.get("generic_support_gate") != {
        "path": SUPPORT_RELATIVE_PATH,
        "sha256": sha256_file(support_path),
        "payload_sha256": support_identity,
        "eligible_unique_by_cell": selected_support,
        "oracle_calls": 0,
    }:
        raise ValueError("NoDistill support binding drift")
    capsule = verify_exact_commit_capsule(capsule_root, capsule_manifest_path)
    capsule_manifest, _ = _load_envelope(capsule_manifest_path)
    if capsule_manifest.get("include") != sorted(CAPSULE_INCLUDE):
        raise ValueError("NoDistill source capsule include set drift")
    for key, value in capsule.items():
        if contract["source_capsule"].get(key) != value:
            raise ValueError(f"NoDistill source capsule drift: {key}")
    if (
        contract["source_capsule"].get("root_relative_path")
        != CAPSULE_ROOT_RELATIVE_PATH
        or contract["source_capsule"].get("manifest_relative_path")
        != CAPSULE_MANIFEST_RELATIVE_PATH
        or contract["source_capsule"].get("manifest_sha256")
        != sha256_file(capsule_manifest_path)
    ):
        raise ValueError("NoDistill source capsule path drift")
    authorization, _ = _load_envelope(authorization_path)
    if authorization != {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "contract_payload_sha256": contract_identity,
        "authorized_scored_calls": 147,
        "user_statement": USER_AUTHORIZATION_STATEMENT,
    }:
        raise ValueError("NoDistill scored authorization does not match contract")
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
    "USER_AUTHORIZATION_STATEMENT",
    "build_exact_commit_capsule",
    "publish_authorization_receipt",
    "seal_scored_contract",
    "validate_scored_contract",
]
