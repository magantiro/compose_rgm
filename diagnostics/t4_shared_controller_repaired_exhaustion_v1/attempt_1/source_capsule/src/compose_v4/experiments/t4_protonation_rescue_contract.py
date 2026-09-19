"""Fail-closed preflight for the prospective 5HT1B-2 protonation rescue."""

from __future__ import annotations

import subprocess
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.control.t4_cold_start_allocation import (
    PROTONATION_EXPERT,
    cold_start_allocation_v1,
)
from compose_v4.control.t4_cold_start_allocation import (
    SCHEMA_VERSION as ALLOCATION_SCHEMA_VERSION,
)
from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.t4_matched_pilot import unseal

SCHEMA_VERSION = "t4_5ht1b2_protonation_rescue_contract_v1"
PREPARED_STATUS = "PREPARED_AWAITING_SEPARATE_SCORED_AUTHORIZATION"
EXPECTED_CELL = "5ht1b_2"
EXPECTED_SOURCE_INDEX = 8
EXPECTED_EXPERT = PROTONATION_EXPERT


def validate_rescue_preflight(
    root: Path,
    contract_path: Path,
    *,
    authorization_payload_sha256: str | None = None,
    require_clean_runtime: bool = True,
) -> dict:
    """Validate every frozen input and optional hash-bound launch authority.

    Passing without ``authorization_payload_sha256`` prepares the experiment but
    explicitly does not authorize scoring.  A launch must provide the exact
    payload identity after the user separately authorizes that contract.
    """

    root = root.resolve()
    contract_path = contract_path.resolve()
    contract = unseal(contract_path)
    contract_identity = identity(contract)
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected protonation rescue contract schema")
    if contract.get("status") != PREPARED_STATUS:
        raise ValueError("protonation rescue contract is not in prepared state")
    if authorization_payload_sha256 is not None and (
        authorization_payload_sha256 != contract_identity
    ):
        raise ValueError("launch authorization does not bind the contract payload")

    cells = contract.get("cells")
    if not isinstance(cells, list) or len(cells) != 1:
        raise ValueError("rescue must contain exactly one prospectively frozen cell")
    cell = cells[0]
    if cell.get("cell") != EXPECTED_CELL or cell.get("source_global_index") != 8:
        raise ValueError("rescue cell identity drift")
    if contract.get("delta") != 0.6:
        raise ValueError("rescue delta drift")
    if contract.get("charged_calls_per_cell") != 49:
        raise ValueError("rescue must retain the 49-call horizon")
    if contract.get("total_charged_call_ceiling") != 49:
        raise ValueError("rescue total call ceiling drift")
    if contract.get("automatic_retries") != 0:
        raise ValueError("scored rescue permits no automatic retries")
    if EXPECTED_EXPERT not in contract.get("proposal", {}):
        raise ValueError(
            "protonation-aware expert is missing from the proposal mixture"
        )
    allocation = contract.get("allocation_policy", {})
    if allocation.get("schema_version") != ALLOCATION_SCHEMA_VERSION:
        raise ValueError("missing versioned shared expert-allocation policy")
    if allocation.get("status") != "FROZEN_PRE_SCORE":
        raise ValueError(
            "expert-allocation policy is not frozen for prospective scoring"
        )
    allocation_evidence = allocation.get("selection_evidence", {})
    allocation_evidence_path = root / str(allocation_evidence.get("path", ""))
    if sha256_file(allocation_evidence_path) != allocation_evidence.get("sha256"):
        raise ValueError("cold-start allocation replay physical hash mismatch")
    allocation_result = unseal(allocation_evidence_path)
    if identity(allocation_result) != allocation_evidence.get("payload_sha256"):
        raise ValueError("cold-start allocation replay payload hash mismatch")
    if allocation_result.get("status") != "selected_zero_oracle_policy":
        raise ValueError("cold-start allocation replay did not select a policy")
    if allocation.get("implementation") != (
        "src/compose_v4/control/t4_cold_start_allocation.py"
    ):
        raise ValueError("cold-start allocation implementation identity drift")
    if contract.get("cold_start_floor_rounds") != 1:
        raise ValueError("cold-start allocation must apply only to round one")
    first_plan = cold_start_allocation_v1(
        round_index=1,
        batch_size=contract["batch"],
        available_experts=contract["proposal"],
    )
    expected_first = allocation.get("expected_all_pools_nonempty", {})
    if expected_first != {
        "expert_floor_counts": first_plan.expert_floor_counts,
        "route_scale_floor_counts": first_plan.route_scale_floor_counts,
        "exploration_slots": first_plan.exploration_slots,
    }:
        raise ValueError("recorded cold-start allocation does not match implementation")
    without_optional = cold_start_allocation_v1(
        round_index=1,
        batch_size=contract["batch"],
        available_experts=(
            "shallow",
            "anchored_replacement",
            "route_complete_region",
        ),
    )
    if without_optional.exploration_slots != allocation.get(
        "empty_optional_protonation_pool", {}
    ).get("exploration_slots"):
        raise ValueError("optional-expert abstention fallback allocation drift")
    later_plan = cold_start_allocation_v1(
        round_index=2,
        batch_size=contract["batch"],
        available_experts=contract["proposal"],
    )
    if later_plan.active or allocation.get("later_rounds", {}).get(
        "ordinary_exploration_slots"
    ) != contract.get("exploration"):
        raise ValueError("later-round allocation drift")
    if contract.get("editing_support") != "Editing-V3 atom_protonation_restate":
        raise ValueError("rescue support version drift")

    gate_spec = contract.get("zero_oracle_proposal_gate", {})
    gate_path = root / str(gate_spec.get("path", ""))
    if sha256_file(gate_path) != gate_spec.get("sha256"):
        raise ValueError("zero-oracle proposal gate physical hash mismatch")
    gate = unseal(gate_path)
    if identity(gate) != gate_spec.get("payload_sha256"):
        raise ValueError("zero-oracle proposal gate payload hash mismatch")
    primary = gate.get("roots", [None])[0]
    if not isinstance(primary, dict):
        raise TypeError("zero-oracle proposal gate has no primary root")
    if primary.get("summary", {}).get("fully_eligible_unique_endpoints", 0) < 2:
        raise ValueError("zero-oracle autonomous proposal gate did not pass")
    exact = primary.get("telemetry", {})
    if exact.get("exact_execution_precision_numerator") != exact.get(
        "exact_execution_precision_denominator"
    ):
        raise ValueError("zero-oracle proposal gate exactness failure")

    material_hashes: dict[str, str] = {}
    for relative, expected in sorted(contract.get("runtime_inputs_sha256", {}).items()):
        path = root / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"runtime input mismatch for {relative}: {actual}")
        material_hashes[relative] = actual

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
            raise ValueError(f"untracked rescue runtime inputs: {untracked.strip()}")

    return {
        "schema_version": "t4_5ht1b2_protonation_rescue_preflight_v1",
        "status": (
            "LAUNCH_AUTHORIZATION_BOUND"
            if authorization_payload_sha256 is not None
            else "PREPARED_NO_SCORED_AUTHORIZATION"
        ),
        "contract_payload_sha256": contract_identity,
        "contract_file_sha256": sha256_file(contract_path),
        "charged_call_ceiling": 49,
        "automatic_retries": 0,
        "cell": EXPECTED_CELL,
        "delta": 0.6,
        "proposal_experts": sorted(contract["proposal"]),
        "allocation_evidence_payload_sha256": identity(allocation_result),
        "zero_oracle_primary_eligible": int(
            primary["summary"]["fully_eligible_unique_endpoints"]
        ),
        "runtime_inputs_sha256": material_hashes,
    }


__all__ = [
    "EXPECTED_EXPERT",
    "PREPARED_STATUS",
    "SCHEMA_VERSION",
    "validate_rescue_preflight",
]
