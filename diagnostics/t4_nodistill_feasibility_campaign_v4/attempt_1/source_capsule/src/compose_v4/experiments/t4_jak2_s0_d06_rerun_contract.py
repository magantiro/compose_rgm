"""Fail-closed preflight for the prepared JAK2 seed-0 delta-0.6 rerun."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.t4_matched_pilot import unseal

SCHEMA_VERSION = "t4_jak2_s0_d06_shared_controller_rerun_contract_v1"
PREPARED_STATUS = "PREPARED_NO_SCORED_AUTHORIZATION"
EXPECTED_CELL = {
    "cell": "jak2_0",
    "source_global_index": 12,
    "smiles": "OCCCCc2nc1ccccc1c4ncnc3[nH]cc2c34",
    "controller_seed": 2026091912,
}
EXPECTED_EXPERTS = (
    "shallow",
    "anchored_replacement",
    "route_complete_region",
)
EXPECTED_CHECKPOINT = "diagnostics/t4_shared_retained_rewrite_v1/checkpoint.json"


def _validate_checkpoint(root: Path, contract: dict) -> dict:
    binding = contract.get("shared_route_checkpoint", {})
    if binding.get("path") != EXPECTED_CHECKPOINT:
        raise ValueError("shared all-route checkpoint path drift")
    path = root / EXPECTED_CHECKPOINT
    if sha256_file(path) != binding.get("sha256"):
        raise ValueError("shared all-route checkpoint physical hash mismatch")
    envelope = json.loads(path.read_text())
    if identity(envelope.get("payload")) != envelope.get("payload_sha256"):
        raise ValueError("shared all-route checkpoint envelope is invalid")
    if envelope["payload_sha256"] != binding.get("payload_sha256"):
        raise ValueError("shared all-route checkpoint payload hash mismatch")
    payload = envelope["payload"]
    expected = {
        "schema_version": "t4_shared_retained_rewrite_checkpoint_v1",
        "training_routes": 77,
        "training_regions": 147,
        "training_scope": "all_locked_t4_routes_shared_task_independent",
        "runtime_target_conditioning": False,
        "new_oracle_calls": 0,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"shared all-route checkpoint {key} drift")
    if len(payload.get("expert", {}).get("templates", ())) != 137:
        raise ValueError("shared all-route checkpoint template census drift")
    return payload


def validate_rerun_preflight(
    root: Path,
    contract_path: Path,
    *,
    authorization_payload_sha256: str | None = None,
    require_clean_runtime: bool = True,
) -> dict:
    """Validate the frozen preparation and optional later hash-bound authority."""

    root = root.resolve()
    contract_path = contract_path.resolve()
    contract = unseal(contract_path)
    contract_identity = identity(contract)
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected JAK2 rerun contract schema")
    if contract.get("status") != PREPARED_STATUS:
        raise ValueError("JAK2 rerun contract is not in prepared state")
    if authorization_payload_sha256 is not None and (
        authorization_payload_sha256 != contract_identity
    ):
        raise ValueError("launch authorization does not bind the contract payload")

    if contract.get("cells") != [EXPECTED_CELL]:
        raise ValueError("rerun must contain exactly the frozen JAK2 seed-0 cell")
    if contract.get("delta") != 0.6 or contract.get("support") != "compose_valid":
        raise ValueError("JAK2 rerun fiber or support drift")
    if contract.get("charged_calls_per_cell") != 49:
        raise ValueError("rerun must retain the 49-call cell horizon")
    if contract.get("total_charged_call_ceiling") != 49:
        raise ValueError("rerun total charged-call ceiling drift")
    if contract.get("automatic_retries") != 0:
        raise ValueError("JAK2 rerun permits no automatic retries")
    if tuple(contract.get("proposal", {})) != EXPECTED_EXPERTS:
        raise ValueError("JAK2 rerun proposal mixture drift")
    if contract.get("deferred_joint_planning") != {
        "enabled": False,
        "reason": "irrelevant_to_one_region_jak_routes",
    }:
        raise ValueError("deferred-joint exclusion drift")
    if "legacy_resume" in contract:
        raise ValueError("fresh JAK2 rerun may not resume a prior scored run")

    expected_settings = {
        "batch": 8,
        "parents": 4,
        "parent_explore": 0.3,
        "exploration": 2,
        "expert_floor_rounds": 2,
        "route_scale_floor_rounds": 2,
        "value_penalty": 1.0,
        "docking_seed": 20260919,
    }
    for key, value in expected_settings.items():
        if contract.get(key) != value:
            raise ValueError(f"current FiberControl setting drift: {key}")
    if contract["proposal"]["shallow"] != {"draws": 480, "horizon": 3}:
        raise ValueError("shallow proposal law drift")
    if contract["proposal"]["anchored_replacement"] != {
        "draws": 512,
        "horizon": 3,
    }:
        raise ValueError("anchored proposal law drift")
    expected_route = {
        "pool_size": 192,
        "realization_limit": 96,
        "beam_width": 48,
        "expansion_width": 48,
        "max_bindings_per_template": 4,
        "maximum_expansions": 4000,
        "scale_balanced": True,
        "training_scope": "one shared task-independent expert fit on all 77 locked routes",
    }
    if contract["proposal"]["route_complete_region"] != expected_route:
        raise ValueError("shared route proposal law drift")

    registry = json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())
    if registry[12] != {
        "idx": 12,
        "target": "jak2",
        "smiles": EXPECTED_CELL["smiles"],
        "published_ds": 7.7,
        "seed_qed": 0.725,
        "seed_sa": 2.889,
        "heavy": 22,
    }:
        raise ValueError("JAK2 seed registry identity drift")
    checkpoint = _validate_checkpoint(root, contract)

    forbidden_top_level = {
        "reported_ivg_delta_0_6",
        "reported_ivg_source",
        "winner_smiles",
        "known_endpoint",
        "target_to_program",
        "cell_to_winner",
        "legacy_resume",
    }
    present = sorted(forbidden_top_level.intersection(contract))
    if present:
        raise ValueError(f"forbidden runtime contract inputs: {present}")

    material_hashes: dict[str, str] = {}
    for relative, expected in sorted(contract.get("runtime_inputs_sha256", {}).items()):
        actual = sha256_file(root / relative)
        if actual != expected:
            raise ValueError(f"runtime input mismatch for {relative}: {actual}")
        material_hashes[relative] = actual
    app_relative = contract.get("application")
    app_source = (root / str(app_relative)).read_text()
    forbidden_app_tokens = (
        "virtual_joint_region_proposer",
        "deferred_joint",
        "reported_ivg",
        "winner_smiles",
        "target_to_program",
    )
    found = [token for token in forbidden_app_tokens if token in app_source.lower()]
    if found:
        raise ValueError(f"forbidden runtime app tokens: {found}")

    runtime_paths = sorted({*material_hashes, str(contract_path.relative_to(root))})
    if require_clean_runtime:
        subprocess.run(
            ["git", "diff", "--exit-code", "HEAD", "--", *runtime_paths],
            cwd=root,
            check=True,
        )
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "--", *runtime_paths],
            cwd=root,
            text=True,
        )
        if untracked.strip():
            raise ValueError(
                f"untracked JAK2 rerun runtime inputs: {untracked.strip()}"
            )

    sentence = str(contract.get("required_scored_authorization_template", "")).format(
        contract_payload_sha256=contract_identity
    )
    return {
        "schema_version": "t4_jak2_s0_d06_shared_controller_rerun_preflight_v1",
        "status": (
            "LAUNCH_AUTHORIZATION_BOUND"
            if authorization_payload_sha256 is not None
            else PREPARED_STATUS
        ),
        "contract_payload_sha256": contract_identity,
        "contract_file_sha256": sha256_file(contract_path),
        "required_scored_authorization": sentence,
        "charged_call_ceiling": 49,
        "automatic_retries": 0,
        "cells": ["jak2_0"],
        "delta": 0.6,
        "proposal_experts": list(EXPECTED_EXPERTS),
        "route_scale_floor_rounds": 2,
        "shared_route_checkpoint_sha256": contract["shared_route_checkpoint"]["sha256"],
        "shared_route_checkpoint_payload_sha256": contract["shared_route_checkpoint"][
            "payload_sha256"
        ],
        "shared_route_training_routes": checkpoint["training_routes"],
        "shared_route_training_regions": checkpoint["training_regions"],
        "deferred_joint_planning": False,
        "runtime_inputs_sha256": material_hashes,
    }


__all__ = [
    "EXPECTED_CELL",
    "EXPECTED_EXPERTS",
    "PREPARED_STATUS",
    "SCHEMA_VERSION",
    "validate_rerun_preflight",
]
