"""Seal the two-cell generic shared-controller repaired-exhaustion campaign."""

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
    RECONCILIATION_SCHEMA_VERSION,
)

SCHEMA_VERSION = "t4_shared_controller_repaired_exhaustion_scored_contract_v1"
PREPARATION_SCHEMA_VERSION = "t4_shared_controller_repaired_exhaustion_preparation_v1"
PREPARATION_RELATIVE_PATH = "configs/t4_shared_controller_repaired_exhaustion_v1.json"
RESULT_ROOT_RELATIVE_PATH = (
    "diagnostics/t4_shared_controller_repaired_exhaustion_v1/attempt_1"
)
FINAL_CONTRACT_RELATIVE_PATH = f"{RESULT_ROOT_RELATIVE_PATH}/scored_contract.json"
AUTHORIZATION_RELATIVE_PATH = f"{RESULT_ROOT_RELATIVE_PATH}/scored_authorization.json"
CAPSULE_ROOT_RELATIVE_PATH = f"{RESULT_ROOT_RELATIVE_PATH}/source_capsule"
CAPSULE_MANIFEST_RELATIVE_PATH = (
    f"{RESULT_ROOT_RELATIVE_PATH}/source_capsule_manifest.json"
)
SUPPORT_GATE_RELATIVE_PATH = (
    "diagnostics/t4_shared_controller_production_support_gate_v1/attempt_1/result.json"
)
ROUTE_CHECKPOINT_RELATIVE_PATH = (
    "diagnostics/t4_shared_retained_rewrite_v1/checkpoint.json"
)
STALE_RECONCILIATION_RELATIVE_PATH = (
    "diagnostics/t4_shared_controller_completion_v1/stale_braf_v4_reconciliation.json"
)
CONTROLLER_IMPLEMENTATION_REVISION = "4bc298281640a8f42da027c026d68311c18e87d9"
SUPPORT_REPAIR_CODE_REVISION = "7c3d18f77dc71d294c229d9de72b2ae56b89b94a"
SUPPORT_GATE_EVIDENCE_REVISION = CONTROLLER_IMPLEMENTATION_REVISION
CONTROLLER_IMPLEMENTATION_FILES_SHA256 = {
    "src/compose_v4/control/fiber_control.py": (
        "8aa214ec25a12ea0b5806b64d22f92edba4617514886a2f2d3023322788e01db"
    ),
    "src/compose_v4/control/protonation_aware_proposal.py": (
        "97842262c5c7a0aa0f330045f82d107bbd441b63a01b1c4e82f8f1f7134ca413"
    ),
    "src/compose_v4/experiments/t4_integrated_route_fiber.py": (
        "37ca6dab53305f415229f772c1527b6d1c50fadea11be123ac0907acef6ff7c0"
    ),
    "src/compose_v4/experiments/t4_shared_controller_checkpoint.py": (
        "cfe328c36870a84f1cdd60d0b937443ea79e4789cd6465ece8b3a2b3c3a10b7a"
    ),
    "src/compose_v4/experiments/t4_shared_controller_scored_runtime.py": (
        "80a2aa96a36d14531e72801b6e6135fb09d961bcfff11bedbd80b7e75fca3c00"
    ),
}
CAPSULE_INCLUDE = (
    PREPARATION_RELATIVE_PATH,
    SUPPORT_GATE_RELATIVE_PATH,
    ROUTE_CHECKPOINT_RELATIVE_PATH,
    STALE_RECONCILIATION_RELATIVE_PATH,
    "docs/GENMOL_T4_SEEDS.json",
    "modal_apps/t4_shared_controller_completion_v1_app.py",
    "modal_apps/t4_shared_controller_repaired_exhaustion_v1_app.py",
    "src/compose_v4",
    "tools/launch_t4_shared_controller_repaired_exhaustion.py",
)
EXPERTS = (
    "shallow",
    "anchored_replacement",
    "route_complete_region",
    "protonation_aware_retained_subgraph",
)
PROTONATION_SETTINGS = {
    "shallow_draws": 256,
    "retained_maximum_fragment_atoms": 16,
    "retained_maximum_stages": 2,
    "retained_maximum_prefixes": 4096,
    "route_pool_size": 192,
    "route_realization_limit": 96,
    "route_beam_width": 48,
    "route_expansion_width": 48,
    "route_max_bindings_per_template": 4,
    "route_maximum_expansions": 4000,
    "route_candidate_timeout_seconds": 10.0,
    "maximum_primitives": 32,
    "maximum_active_atoms": 40,
    "persistent_slots": 48,
    "task_or_target_input_used": False,
    "teacher_or_endpoint_input_used": False,
}
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
        "enabled": True,
        "semantics": "additive_route_complete_region_subproducer",
        "binding_particles": [0, 1, 2, 3],
        "supported_complete_depths": [1, 2, 3],
        "scheduled_shard_counts": {"1": 1, "2": 1, "3": 5},
        "jobs_per_parent": 28,
        "maximum_combinations_per_job": 4_096,
        "missing_or_failed_receipt": "abstain_to_legacy_route_pool",
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


def required_authorization_sentence(contract_payload_sha256: str) -> str:
    return (
        "I authorize the two-cell repaired-exhaustion shared COMPOSE campaign "
        "under scored-contract payload SHA-256 "
        f"{contract_payload_sha256}, capped at 49 charged docking calls per cell "
        "and 98 total, with zero retry, replacement, backfill, prior-outcome "
        "reuse, winner injection, endpoint injection, teacher injection, or "
        "runtime comparator."
    )


def _validate_controller(controller: Any) -> dict[str, Any]:
    if not isinstance(controller, dict):
        raise TypeError("repaired-exhaustion controller must be one mapping")
    expected = {
        "batch": 8,
        "parents": 4,
        "parent_explore": 0.3,
        "exploration": 2,
        "expert_floor_rounds": 2,
        "route_scale_floor_rounds": 2,
        "value_penalty": 1.0,
        "docking_seed": 20260919,
        "proposal_experts": list(EXPERTS),
        "proposal": {
            "shallow": {"draws": 480, "horizon": 3},
            "anchored_replacement": {"draws": 512, "horizon": 3},
            "route_complete_region": {
                "pool_size": 192,
                "realization_limit": 96,
                "beam_width": 48,
                "expansion_width": 48,
                "max_bindings_per_template": 4,
                "maximum_expansions": 4000,
                "scale_balanced": True,
                "training_scope": (
                    "one shared task-independent expert fit on all 77 locked routes"
                ),
            },
            "protonation_aware_retained_subgraph": PROTONATION_SETTINGS,
        },
    }
    if controller != expected:
        raise ValueError("shared four-expert controller settings drift")
    return controller


def validate_preparation(path: Path) -> dict[str, Any]:
    preparation = _load_object(path)
    if preparation.get("schema_version") != PREPARATION_SCHEMA_VERSION:
        raise ValueError("unexpected repaired-exhaustion preparation schema")
    if preparation.get("status") != ("PREPARED_FOR_PAYLOAD_BOUND_SCORED_AUTHORIZATION"):
        raise ValueError("repaired-exhaustion preparation status drift")
    if (
        preparation.get("campaign_cell_count") != 2
        or preparation.get("charged_calls_per_cell") != 49
        or preparation.get("total_charged_call_ceiling") != 98
    ):
        raise ValueError("repaired-exhaustion budget drift")
    if preparation.get("controller_source_snapshot_revision") != (
        CONTROLLER_IMPLEMENTATION_REVISION
    ):
        raise ValueError("shared-controller implementation revision drift")
    if (
        preparation.get("support_repair_code_revision") != SUPPORT_REPAIR_CODE_REVISION
        or preparation.get("support_gate_evidence_revision")
        != SUPPORT_GATE_EVIDENCE_REVISION
    ):
        raise ValueError("support-repair code or evidence revision drift")
    _validate_controller(preparation.get("controller"))
    cells = preparation.get("cells")
    if not isinstance(cells, list) or [row.get("cell_key") for row in cells] != [
        "braf_0_d06",
        "5ht1b_2_d06",
    ]:
        raise ValueError("repaired-exhaustion cell matrix drift")
    expected_domains = [
        (
            "braf_0",
            "braf",
            9,
            ("CCN(CC)CCNC(=O)c3cnn4c(c2cccc(NC(=O)Nc1ccc(Cl)c(C(F)(F)F)c1)c2)ccnc34"),
            0.6,
            2026091909,
            "compose-t4-repaired-exhaustion-v1-braf-0-d06",
            "f8ac045235025e98b15fd90aae6617edfdcc125081f72a5a5315db22be1f46e0",
            "707b21bfb654321cb14d33ea07e9accab4cb9a884378285cd562c2324a31213a",
            [[84.194, 6.949, -7.081], [22.032, 19.211, 14.106]],
        ),
        (
            "5ht1b_2",
            "5ht1b",
            8,
            "C1=CC2=NC=C(CCCN3CC[NH+](CCc4ccccc4)CC3)[C@H]2C=C1n1cnnc1",
            0.6,
            2026091908,
            "compose-t4-repaired-exhaustion-v1-5ht1b-2-d06",
            "f8ac045235025e98b15fd90aae6617edfdcc125081f72a5a5315db22be1f46e0",
            "6ab5ac2d63be05c4d7992f96175258c2b253ccc41539833af4fb12465ee3a59f",
            [[-26.602, 5.277, 17.898], [22.5, 22.5, 22.5]],
        ),
    ]
    observed_domains = [
        (
            row.get("source_cell"),
            row.get("target"),
            row.get("source_global_index"),
            row.get("source_smiles"),
            row.get("delta"),
            row.get("controller_seed"),
            row.get("volume"),
            row.get("evaluator", {}).get("qvina02_sha256"),
            row.get("evaluator", {}).get("receptor_sha256"),
            row.get("evaluator", {}).get("docking_box"),
        )
        for row in cells
    ]
    if observed_domains != expected_domains:
        raise ValueError("repaired-exhaustion source, target, delta, or seed drift")
    volumes = [row.get("volume") for row in cells]
    if (
        any(not isinstance(value, str) or not value for value in volumes)
        or len(set(volumes)) != 2
    ):
        raise ValueError("repaired-exhaustion volumes are not private and unique")
    if any("controller" in row or "proposal" in row for row in cells):
        raise ValueError("cell-specific controller settings are forbidden")
    if preparation.get("trajectory_distillation") != {
        "enabled": True,
        "shared_task_independent_route_expert": True,
        "route_template_particles": True,
    }:
        raise ValueError("shared route configuration drift")
    if preparation.get("receipt_policy") != {
        "publish_query_lock_before_docking": True,
        "one_volume_per_cell": True,
        "one_driver_per_cell": True,
        "missing_locked_query": "charge_once_after_deadline_without_resubmission",
        "continuation": "reserved_running_terminal_generation",
        "candidate_exhaustion": "terminal_without_replacement_or_backfill",
    }:
        raise ValueError("repaired-exhaustion receipt policy drift")
    if preparation.get("oracle_policy") != {
        "automatic_retries": 0,
        "replacement": False,
        "backfill": False,
        "prior_scored_outcomes_read": False,
        "runtime_comparator": False,
        "teacher_or_endpoint_injection": False,
    }:
        raise ValueError("repaired-exhaustion oracle policy drift")
    if preparation.get("support") != "compose_valid":
        raise ValueError("endpoint support drift")
    return preparation


def _validate_controller_implementation(root: Path) -> dict[str, Any]:
    observed = {
        relative: sha256_file(root / relative)
        for relative in CONTROLLER_IMPLEMENTATION_FILES_SHA256
    }
    if observed != CONTROLLER_IMPLEMENTATION_FILES_SHA256:
        raise ValueError("shared-controller implementation differs from 4bc29828")
    return {
        "source_snapshot_revision": CONTROLLER_IMPLEMENTATION_REVISION,
        "support_repair_code_revision": SUPPORT_REPAIR_CODE_REVISION,
        "support_gate_evidence_revision": SUPPORT_GATE_EVIDENCE_REVISION,
        "source_files_sha256": dict(sorted(observed.items())),
    }


def _validate_support_gate(path: Path) -> tuple[dict[str, Any], str]:
    payload, identity = _load_envelope(path)
    if payload.get("schema_version") != (
        "t4_shared_controller_production_support_gate_result_v1"
    ):
        raise ValueError("unexpected repaired support-gate schema")
    if payload.get("gate") != {
        "braf_unseen_route_survives": True,
        "passed": True,
        "protonation_unseen_candidates_survive": True,
    }:
        raise ValueError("repaired support gate did not pass")
    cells = payload.get("cells", {})
    if (
        cells.get("braf_0_d06", {}).get("candidate_union_unique") != 1
        or cells.get("5ht1b_2_d06", {}).get("candidate_union_unique") != 14
        or payload.get("costs")
        != {
            "docking_calls": 0,
            "live_run_reads": 0,
            "modal_launches": 0,
            "oracle_calls": 0,
        }
    ):
        raise ValueError("repaired support evidence census or zero-cost gate drift")
    return payload, identity


def _validate_reconciliation(path: Path) -> tuple[dict[str, Any], str]:
    payload, identity = _load_envelope(path)
    if payload.get("schema_version") != RECONCILIATION_SCHEMA_VERSION:
        raise ValueError("unexpected stale-query reconciliation schema")
    exclusions = payload.get("excluded_canonical_smiles_sha256")
    if (
        payload.get("unresolved_locked_queries") != 0
        or not isinstance(exclusions, dict)
        or set(exclusions) != {"braf_0", "braf_1"}
        or len(exclusions["braf_0"]) != 20
        or len(exclusions["braf_1"]) != 23
    ):
        raise ValueError("stale-query reconciliation census drift")
    return payload, identity


def _bound_inputs(root: Path, preparation: dict[str, Any]) -> dict[str, Any]:
    checkpoint_path = root / ROUTE_CHECKPOINT_RELATIVE_PATH
    checkpoint, checkpoint_identity = _load_envelope(checkpoint_path)
    del checkpoint
    if preparation.get("shared_route_checkpoint") != {
        "path": ROUTE_CHECKPOINT_RELATIVE_PATH,
        "sha256": sha256_file(checkpoint_path),
        "payload_sha256": checkpoint_identity,
    }:
        raise ValueError("shared route checkpoint binding drift")
    support_path = root / SUPPORT_GATE_RELATIVE_PATH
    _, support_identity = _validate_support_gate(support_path)
    reconciliation_path = root / STALE_RECONCILIATION_RELATIVE_PATH
    reconciliation, reconciliation_identity = _validate_reconciliation(
        reconciliation_path
    )
    return {
        "shared_route_checkpoint": preparation["shared_route_checkpoint"],
        "zero_oracle_support_gate": {
            "path": SUPPORT_GATE_RELATIVE_PATH,
            "sha256": sha256_file(support_path),
            "payload_sha256": support_identity,
        },
        "stale_braf_v4_reconciliation": {
            "path": STALE_RECONCILIATION_RELATIVE_PATH,
            "sha256": sha256_file(reconciliation_path),
            "payload_sha256": reconciliation_identity,
            "excluded_canonical_smiles_sha256": reconciliation[
                "excluded_canonical_smiles_sha256"
            ],
        },
    }


def seal_scored_contract(
    *,
    repository_root: Path,
    capsule_root: Path,
    capsule_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    root = repository_root.resolve()
    preparation_path = root / PREPARATION_RELATIVE_PATH
    preparation = validate_preparation(preparation_path)
    controller_implementation = _validate_controller_implementation(root)
    inputs = _bound_inputs(root, preparation)
    capsule = verify_exact_commit_capsule(capsule_root, capsule_manifest_path)
    manifest, _ = _load_envelope(capsule_manifest_path)
    if manifest.get("include") != sorted(CAPSULE_INCLUDE):
        raise ValueError("repaired-exhaustion capsule include set drift")
    for relative, expected in CONTROLLER_IMPLEMENTATION_FILES_SHA256.items():
        if manifest["files"].get(relative, {}).get("sha256") != expected:
            raise ValueError("capsule controller implementation differs from 4bc29828")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "SEALED_PENDING_EXACT_USER_AUTHORIZATION",
        "campaign_cell_count": 2,
        "scored_calls_requested": 98,
        "charged_calls_per_cell": 49,
        "total_charged_call_ceiling": 98,
        "automatic_retries": 0,
        "replacement": False,
        "backfill": False,
        "support": "compose_valid",
        "controller": preparation["controller"],
        "controller_implementation": controller_implementation,
        "cells": preparation["cells"],
        "trajectory_distillation": preparation["trajectory_distillation"],
        "receipt_policy": preparation["receipt_policy"],
        "operational_settings": OPERATIONAL_SETTINGS,
        **inputs,
        "preparation_contract": {
            "path": PREPARATION_RELATIVE_PATH,
            "sha256": sha256_file(preparation_path),
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
        },
        "claim_boundary": preparation["claim_boundary"],
    }
    identity = _publish_once(output_path, payload)
    return {
        "contract_path": str(output_path),
        "contract_sha256": sha256_file(output_path),
        "contract_payload_sha256": identity,
        "authorized_scored_calls": 0,
        "required_authorization_sentence": required_authorization_sentence(identity),
        "modal_calls_created": 0,
    }


def validate_sealed_contract(
    *,
    repository_root: Path,
    contract_path: Path,
    capsule_root: Path,
    capsule_manifest_path: Path,
) -> tuple[dict[str, Any], str]:
    root = repository_root.resolve()
    contract, identity = _load_envelope(contract_path)
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected repaired-exhaustion scored contract schema")
    if contract.get("status") != "SEALED_PENDING_EXACT_USER_AUTHORIZATION":
        raise ValueError("repaired-exhaustion scored contract status drift")
    preparation_path = root / PREPARATION_RELATIVE_PATH
    preparation = validate_preparation(preparation_path)
    exact = {
        "campaign_cell_count": 2,
        "scored_calls_requested": 98,
        "charged_calls_per_cell": 49,
        "total_charged_call_ceiling": 98,
        "automatic_retries": 0,
        "replacement": False,
        "backfill": False,
        "support": "compose_valid",
        "controller": preparation["controller"],
        "controller_implementation": _validate_controller_implementation(root),
        "cells": preparation["cells"],
        "trajectory_distillation": preparation["trajectory_distillation"],
        "receipt_policy": preparation["receipt_policy"],
        "operational_settings": OPERATIONAL_SETTINGS,
    }
    for key, value in exact.items():
        if contract.get(key) != value:
            raise ValueError(f"repaired-exhaustion scored contract drift: {key}")
    if contract.get("preparation_contract") != {
        "path": PREPARATION_RELATIVE_PATH,
        "sha256": sha256_file(preparation_path),
    }:
        raise ValueError("repaired-exhaustion preparation binding drift")
    for key, value in _bound_inputs(root, preparation).items():
        if contract.get(key) != value:
            raise ValueError(f"repaired-exhaustion input binding drift: {key}")
    capsule = verify_exact_commit_capsule(capsule_root, capsule_manifest_path)
    manifest, _ = _load_envelope(capsule_manifest_path)
    if manifest.get("include") != sorted(CAPSULE_INCLUDE):
        raise ValueError("repaired-exhaustion capsule include set drift")
    for key, value in capsule.items():
        if contract.get("source_capsule", {}).get(key) != value:
            raise ValueError(f"repaired-exhaustion source capsule drift: {key}")
    if (
        contract["source_capsule"].get("root_relative_path")
        != CAPSULE_ROOT_RELATIVE_PATH
        or contract["source_capsule"].get("manifest_relative_path")
        != CAPSULE_MANIFEST_RELATIVE_PATH
        or contract["source_capsule"].get("manifest_sha256")
        != sha256_file(capsule_manifest_path)
    ):
        raise ValueError("repaired-exhaustion capsule path binding drift")
    if contract.get("authorization") != {
        "state": "PENDING_EXACT_PAYLOAD_BOUND_USER_AUTHORIZATION",
        "receipt_path": AUTHORIZATION_RELATIVE_PATH,
    }:
        raise ValueError("repaired-exhaustion authorization policy drift")
    return contract, identity


def publish_authorization_receipt(
    *, contract_path: Path, authorization_path: Path, user_statement: str
) -> dict[str, Any]:
    contract, identity = _load_envelope(contract_path)
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected repaired-exhaustion scored contract schema")
    required = required_authorization_sentence(identity)
    if user_statement != required:
        raise ValueError(f"authorization statement must exactly equal: {required}")
    payload = {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "contract_payload_sha256": identity,
        "authorized_scored_calls": 98,
        "user_statement": user_statement,
    }
    authorization_identity = _publish_once(authorization_path, payload)
    return {
        "authorization_path": str(authorization_path),
        "authorization_sha256": sha256_file(authorization_path),
        "authorization_payload_sha256": authorization_identity,
    }


def validate_scored_contract(
    *,
    repository_root: Path,
    contract_path: Path,
    authorization_path: Path,
    capsule_root: Path,
    capsule_manifest_path: Path,
) -> dict[str, Any]:
    contract, identity = validate_sealed_contract(
        repository_root=repository_root,
        contract_path=contract_path,
        capsule_root=capsule_root,
        capsule_manifest_path=capsule_manifest_path,
    )
    authorization, _ = _load_envelope(authorization_path)
    required = required_authorization_sentence(identity)
    if authorization != {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "contract_payload_sha256": identity,
        "authorized_scored_calls": 98,
        "user_statement": required,
    }:
        raise ValueError("repaired-exhaustion authorization does not match contract")
    return contract


__all__ = [
    "AUTHORIZATION_RELATIVE_PATH",
    "CAPSULE_INCLUDE",
    "CAPSULE_MANIFEST_RELATIVE_PATH",
    "CAPSULE_ROOT_RELATIVE_PATH",
    "CONTROLLER_IMPLEMENTATION_REVISION",
    "FINAL_CONTRACT_RELATIVE_PATH",
    "OPERATIONAL_SETTINGS",
    "PREPARATION_RELATIVE_PATH",
    "SCHEMA_VERSION",
    "build_exact_commit_capsule",
    "publish_authorization_receipt",
    "required_authorization_sentence",
    "seal_scored_contract",
    "validate_preparation",
    "validate_scored_contract",
    "validate_sealed_contract",
]
