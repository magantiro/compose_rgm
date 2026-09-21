"""Fail-closed preflight for the versioned 5HT1B-2 protonation rescue (schema v2).

This is a SEPARATE module from ``t4_protonation_rescue_contract`` on purpose.  That
module validates the superseded 49-call, delta-0.6-only experiment and is pinned by
that experiment's contract; it is left byte-identical here.  This module validates the
two-arm v2 rescue: one cell (``5ht1b_2``) at TWO deltas, 0.6 and 0.4, with per-arm
charged-call ceilings 248 and 249.

Data structure and invariants
-----------------------------
A rescue contract is a sealed envelope ``{"payload": ..., "payload_sha256": ...}``.
The payload is the SCIENTIFIC contract and nothing else.  Three invariants are
maintained here and exercised by ``tests/test_t4_protonation_rescue_contract_v2.py``:

1. **A scientific payload is not authorization metadata.**  Consent must not change the
   bytes consented to, so a payload carrying an ``authorization`` block, an authorized
   status, or any owner-consent field is REFUSED (``assert_payload_carries_no_authorization``).
   Authorization lives externally in the authorization receipt, names the payload by
   hash, and reaches this module through ``authorization_payload_sha256``.
2. **The arm decides delta and ceiling, and the contract must agree.**  The caller
   declares which arm it is (``d06``/``d04``); ``ARMS`` fixes that arm's delta, ceiling
   and contract path.  Nothing here hardcodes a single delta or a single ceiling, which
   is what made the v1 validator wrong for this experiment.
3. **Every input that decides whether a number is produced is hash-verified**, and the
   working tree holding them must be clean, before any launch authority is returned.

Passing WITHOUT ``authorization_payload_sha256`` prepares the experiment and explicitly
confers no scored-call authority.  A launch must supply the exact payload identity, and
that identity must also be named by the external authorization receipt.
"""

from __future__ import annotations

import json
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

# ---- Frozen experiment identity ----

SCHEMA_VERSION = "t4_5ht1b2_protonation_rescue_contract_v2"
PREPARED_STATUS = "PREPARED_AWAITING_OWNER_AUTHORIZATION"
EXPECTED_CELL = "5ht1b_2"
EXPECTED_SOURCE_INDEX = 8
EXPECTED_SOURCE_SMILES = "C1=CC2=NC=C(CCCN3CC[NH+](CCc4ccccc4)CC3)[C@H]2C=C1n1cnnc1"
EXPECTED_EXPERT = PROTONATION_EXPERT
EXPECTED_EDITING_SUPPORT = "Editing-V3 atom_protonation_restate"
EXPECTED_COLD_START_FLOOR_ROUNDS = 1

AUTHORIZATION_RECEIPT = (
    "diagnostics/t4_5ht1b2_protonation_rescue_authorization_v1.json"
)
PINNED_GATE = (
    "diagnostics/t4_5ht1b2_protonation_rescue_feasibility_pinned_v1.json"
)

# Per-arm expectations.  delta and the charged-call ceiling are ARM properties: the v1
# validator hardcoded 0.6 and 49, which is exactly why it cannot validate this pair.
ARMS: dict[str, dict] = {
    "d06": {
        "delta": 0.6,
        "ceiling": 248,
        "contract": "configs/t4_5ht1b2_protonation_rescue_d06_v1.json",
        "app": "modal_apps/t4_5ht1b2_protonation_rescue_d06_app.py",
        "gate_eligible_unique": 12,
    },
    "d04": {
        "delta": 0.4,
        "ceiling": 249,
        "contract": "configs/t4_5ht1b2_protonation_rescue_d04_v1.json",
        "app": "modal_apps/t4_5ht1b2_protonation_rescue_d04_app.py",
        "gate_eligible_unique": 44,
    },
}

# Keys that record OWNER CONSENT.  None of them may appear anywhere in a scientific
# payload; if one does, the payload's hash has been moved by the act of recording
# consent, which destroys the binding the consent was supposed to create.
AUTHORIZATION_METADATA_KEYS = frozenset(
    {
        "authorization",
        "authorization_form",
        "authorization_sentence",
        "authorized_at_utc",
        "authorized_payload_sha256",
        "authorized_payload_sha256_at_authorization",
        "owner_authorization",
        "owner_response_verbatim",
        "payload_sha256_after_authorization",
    }
)

# Statuses that assert in-payload authorization.  ``PREPARED_STATUS`` is the only
# admissible status: a contract must be authorized from OUTSIDE.
FORBIDDEN_IN_PAYLOAD_STATUSES = frozenset(
    {
        "AUTHORIZED_BY_OWNER_FOR_PROTONATION_RESCUE",
        "AUTHORIZED",
        "AUTHORIZED_FOR_SCORED_LAUNCH",
    }
)


# ---- Design-rule guard ----


def assert_payload_carries_no_authorization(payload: dict) -> None:
    """Refuse a scientific payload that has absorbed owner-consent metadata.

    Recording consent must not change the bytes consented to.  This walks the whole
    payload, not just its top level, because a nested ``authorization`` block moves the
    hash exactly as much as a top-level one does.
    """

    def walk(node, trail: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in AUTHORIZATION_METADATA_KEYS:
                    location = f"{trail}.{key}" if trail else key
                    raise ValueError(
                        "authorization metadata is present inside the scientific "
                        f"payload at {location!r}; authorization must point AT an "
                        "immutable payload by hash and never become part of it"
                    )
                walk(value, f"{trail}.{key}" if trail else key)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{trail}[{index}]")

    walk(payload, "")
    status = payload.get("status")
    if status in FORBIDDEN_IN_PAYLOAD_STATUSES:
        raise ValueError(
            f"contract status {status!r} asserts in-payload authorization; the "
            f"payload must stay at {PREPARED_STATUS!r} and be authorized externally"
        )


# ---- Authorization receipt ----


def _authorized_payload_hashes(root: Path) -> dict[str, str]:
    """Return {arm: payload sha256} that the EXTERNAL receipt authorizes.

    The receipt is a plain (unsealed) diagnostics record, so it is read with ``json``
    rather than ``unseal``.  Only the hashes named as authorized at authorization time
    are returned: a payload whose bytes were altered after consent is not authorized,
    which is the whole point of keeping consent out of the payload.
    """

    receipt_path = root / AUTHORIZATION_RECEIPT
    if not receipt_path.exists():
        raise ValueError(f"authorization receipt is absent: {AUTHORIZATION_RECEIPT}")
    receipt = json.loads(receipt_path.read_text())
    named = receipt.get("authorized_payload_sha256_at_authorization")
    if not isinstance(named, dict) or not named:
        raise ValueError("authorization receipt names no authorized payload hashes")
    ceilings = receipt.get("per_cell_ceilings", {})
    for arm, spec in ARMS.items():
        key = f"{EXPECTED_CELL}_{arm}"
        if ceilings.get(key) != spec["ceiling"]:
            raise ValueError(
                f"authorization receipt ceiling for {key} is "
                f"{ceilings.get(key)!r}, not the contracted {spec['ceiling']}"
            )
    total = sum(spec["ceiling"] for spec in ARMS.values())
    if receipt.get("total_charged_call_ceiling") != total:
        raise ValueError(
            "authorization receipt total ceiling is "
            f"{receipt.get('total_charged_call_ceiling')!r}, not {total}"
        )
    return {arm: str(value) for arm, value in named.items()}


# ---- Launch-path revision continuity ----


def reconstruct_superseded_payload(payload: dict) -> dict:
    """Rebuild the payload this one superseded, from its own revision record.

    The app file is pinned in ``runtime_inputs_sha256`` because the remote worker must
    prove it is running the protonation-wired code.  But the SAME file also carries the
    local launch plumbing, so repairing the launcher necessarily moves a pin that the
    zero-oracle gate and the authorization receipt both address by hash.

    Rather than assert that nothing scientific changed, this reconstructs the previous
    payload byte-for-byte: undo the recorded pin moves, drop the recorded pin additions,
    remove the revision record itself.  ``validate_rescue_preflight_v2`` then requires
    the result to hash to the superseded identity.  Any OTHER mutation -- a delta, a
    ceiling, a cell, a proposal setting, an extra field -- survives the reconstruction
    and breaks the hash, so continuity is proved rather than claimed.
    """

    revision = payload.get("launch_path_revision")
    if not isinstance(revision, dict):
        raise TypeError("contract carries no launch_path_revision record")
    restored = json.loads(json.dumps(payload))
    restored.pop("launch_path_revision", None)
    pins = restored.get("runtime_inputs_sha256")
    if not isinstance(pins, dict):
        raise TypeError("contract carries no runtime_inputs_sha256 block")
    for relative, move in (revision.get("changed_runtime_inputs") or {}).items():
        if pins.get(relative) != move.get("to"):
            raise ValueError(
                f"launch_path_revision claims {relative} moved to {move.get('to')!r} "
                f"but the contract pins {pins.get(relative)!r}"
            )
        pins[relative] = move["from"]
    for relative, digest in (revision.get("added_runtime_inputs") or {}).items():
        if pins.get(relative) != digest:
            raise ValueError(
                f"launch_path_revision claims {relative} was added at {digest!r} "
                f"but the contract pins {pins.get(relative)!r}"
            )
        del pins[relative]
    return restored


# ---- Preflight ----


def validate_rescue_preflight_v2(
    root: Path,
    contract_path: Path,
    *,
    arm: str,
    authorization_payload_sha256: str | None = None,
    require_clean_runtime: bool = True,
) -> dict:
    """Validate every frozen input for one rescue arm, and optionally bind launch authority.

    ``arm`` is declared by the CALLER (the Modal wrapper knows which arm it is) and is
    cross-checked against the contract: the wrapper's identity and the contract's delta,
    ceiling and pinned app path must all agree.  Returning without
    ``authorization_payload_sha256`` confers no scored-call authority.
    """

    if arm not in ARMS:
        raise ValueError(f"unknown protonation rescue arm {arm!r}")
    expected = ARMS[arm]

    root = root.resolve()
    contract_path = contract_path.resolve()
    relative_contract = str(contract_path.relative_to(root))
    if relative_contract != expected["contract"]:
        raise ValueError(
            f"arm {arm!r} expects {expected['contract']}, got {relative_contract}"
        )

    contract = unseal(contract_path)
    contract_identity = identity(contract)

    # -- design rule: the payload is science, never consent --
    assert_payload_carries_no_authorization(contract)
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "unexpected protonation rescue contract schema: "
            f"{contract.get('schema_version')!r}"
        )
    if contract.get("status") != PREPARED_STATUS:
        raise ValueError(
            f"contract status {contract.get('status')!r} is not {PREPARED_STATUS!r}"
        )

    # -- cell identity --
    cells = contract.get("cells")
    if not isinstance(cells, list) or len(cells) != 1:
        raise ValueError("rescue must contain exactly one prospectively frozen cell")
    cell = cells[0]
    if cell.get("cell") != EXPECTED_CELL:
        raise ValueError(f"rescue cell identity drift: {cell.get('cell')!r}")
    if cell.get("source_global_index") != EXPECTED_SOURCE_INDEX:
        raise ValueError(
            f"rescue source index drift: {cell.get('source_global_index')!r}"
        )
    if cell.get("smiles") != EXPECTED_SOURCE_SMILES:
        raise ValueError("rescue frozen source SMILES drift")

    # -- per-arm budget and threshold (the v1 validator hardcoded 0.6 and 49) --
    if contract.get("delta") != expected["delta"]:
        raise ValueError(
            f"arm {arm!r} requires delta {expected['delta']}, "
            f"contract declares {contract.get('delta')!r}"
        )
    if contract.get("charged_calls_per_cell") != expected["ceiling"]:
        raise ValueError(
            f"arm {arm!r} requires a {expected['ceiling']}-call per-cell horizon, "
            f"contract declares {contract.get('charged_calls_per_cell')!r}"
        )
    if contract.get("total_charged_call_ceiling") != expected["ceiling"]:
        raise ValueError(
            f"arm {arm!r} requires total ceiling {expected['ceiling']}, "
            f"contract declares {contract.get('total_charged_call_ceiling')!r}"
        )
    if contract.get("automatic_retries") != 0:
        raise ValueError("scored rescue permits no automatic retries")

    # -- mechanism --
    proposal = contract.get("proposal", {})
    if EXPECTED_EXPERT not in proposal:
        raise ValueError(
            "protonation-aware expert is missing from the proposal mixture"
        )
    if contract.get("editing_support") != EXPECTED_EDITING_SUPPORT:
        raise ValueError("rescue support version drift")
    if contract.get("cold_start_floor_rounds") != EXPECTED_COLD_START_FLOOR_ROUNDS:
        raise ValueError("cold-start allocation must apply only to round one")

    # -- shared cold-start expert-allocation policy (adapted from the v1 validator) --
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
    first_plan = cold_start_allocation_v1(
        round_index=1,
        batch_size=contract["batch"],
        available_experts=proposal,
    )
    expected_first = allocation.get("expected_all_pools_nonempty", {})
    if expected_first != {
        "expert_floor_counts": first_plan.expert_floor_counts,
        "route_scale_floor_counts": first_plan.route_scale_floor_counts,
        "exploration_slots": first_plan.exploration_slots,
    }:
        raise ValueError("recorded cold-start allocation does not match implementation")
    if int(first_plan.expert_floor_counts.get(EXPECTED_EXPERT, 0)) < 1:
        raise ValueError(
            "the protonation expert is not guaranteed a cold-start slot, so this arm "
            "would reproduce the candidate exhaustion it exists to fix"
        )
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
        available_experts=proposal,
    )
    if later_plan.active or allocation.get("later_rounds", {}).get(
        "ordinary_exploration_slots"
    ) != contract.get("exploration"):
        raise ValueError("later-round allocation drift")

    # -- launch-path revision continuity (see reconstruct_superseded_payload) --
    bound_identities = {contract_identity}
    superseded_identity: str | None = None
    if "launch_path_revision" in contract:
        superseded_identity = identity(reconstruct_superseded_payload(contract))
        declared = contract["launch_path_revision"].get("supersedes_payload_sha256")
        if superseded_identity != declared:
            raise ValueError(
                "launch_path_revision does not reconstruct the payload it claims to "
                f"supersede: rebuilt {superseded_identity}, declared {declared!r}"
            )
        bound_identities.add(superseded_identity)

    # -- zero-oracle proposal gate, production-pinned kernel, binds this payload --
    gate_path = root / PINNED_GATE
    if not gate_path.exists():
        raise ValueError(f"pinned zero-oracle feasibility gate is absent: {PINNED_GATE}")
    gate = unseal(gate_path)
    if gate.get("oracle_calls") != 0 or gate.get("docking_calls") != 0:
        raise ValueError("the feasibility gate is not a zero-oracle measurement")
    if gate.get("modal_launches") != 0:
        raise ValueError("the feasibility gate recorded a Modal launch")
    gate_arms = [
        entry
        for entry in gate.get("arms", [])
        if entry.get("contract") == expected["contract"]
    ]
    if len(gate_arms) != 1:
        raise ValueError(
            f"pinned gate has {len(gate_arms)} entries for {expected['contract']}"
        )
    gate_arm = gate_arms[0]
    if gate_arm.get("contract_payload_sha256") not in bound_identities:
        raise ValueError(
            "the pinned zero-oracle gate measured a different contract payload "
            f"({gate_arm.get('contract_payload_sha256')}); it does not cover this one"
        )
    if gate_arm.get("delta") != expected["delta"] or gate_arm.get("cell") != EXPECTED_CELL:
        raise ValueError("pinned gate arm identity drift")
    aggregate = gate.get("aggregate", {})
    delta_key = str(expected["delta"])
    eligible = (aggregate.get("primary_eligible_unique_by_delta") or {}).get(delta_key)
    if eligible != expected["gate_eligible_unique"]:
        raise ValueError(
            f"pinned gate eligible-endpoint drift at delta {delta_key}: "
            f"{eligible!r}, expected {expected['gate_eligible_unique']}"
        )
    if eligible is None or int(eligible) < 2:
        raise ValueError("zero-oracle autonomous proposal gate did not pass")
    if (aggregate.get("abstention_contrast_endpoints_by_delta") or {}).get(delta_key) != 0:
        raise ValueError("pinned gate abstention-contrast drift")
    if (aggregate.get("reverse_contrast_sites_by_delta") or {}).get(delta_key) != 2:
        raise ValueError("pinned gate reverse-contrast site drift")
    if aggregate.get("settings_consumed_every_arm") is not True:
        raise ValueError(
            "the pinned gate did not prove every arm consumes its contract settings"
        )

    # -- relayed proposal gate: verified by its recorded hashes only.  The v1 validator
    # read its eligibility counts as the pass criterion; here that role belongs to the
    # PINNED gate above, because the relayed one was measured on a non-production kernel
    # and transcribed the thresholds rather than calling the production Fiber.check.
    relayed = (contract.get("supporting_evidence", {}) or {}).get(
        "zero_oracle_proposal_gate", {}
    )
    relayed_path = root / str(relayed.get("path", ""))
    if sha256_file(relayed_path) != relayed.get("sha256"):
        raise ValueError("relayed zero-oracle proposal gate physical hash mismatch")
    if identity(unseal(relayed_path)) != relayed.get("payload_sha256"):
        raise ValueError("relayed zero-oracle proposal gate payload hash mismatch")

    # -- every input that decides whether a number is produced --
    pins = contract.get("runtime_inputs_sha256", {})
    if expected["app"] not in pins:
        raise ValueError(
            f"arm {arm!r} does not pin its own wrapper {expected['app']}; a contract "
            "that does not pin its launcher cannot prove which code spent its budget"
        )
    material_hashes: dict[str, str] = {}
    for relative, expected_digest in sorted(pins.items()):
        path = root / relative
        actual = sha256_file(path)
        if actual != expected_digest:
            raise ValueError(f"runtime input mismatch for {relative}: {actual}")
        material_hashes[relative] = actual

    runtime_paths = sorted({*material_hashes, relative_contract, AUTHORIZATION_RECEIPT})
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

    # -- external authorization: points AT the payload, never part of it --
    authorized = _authorized_payload_hashes(root)
    receipt_hash = authorized.get(arm)
    if receipt_hash is None:
        raise ValueError(f"authorization receipt names no payload for arm {arm!r}")
    if receipt_hash not in bound_identities:
        raise ValueError(
            f"the authorization receipt authorizes {receipt_hash} for arm {arm!r}, "
            f"which is neither this payload ({contract_identity}) nor the launch-path "
            "revision it supersedes"
        )
    if (
        authorization_payload_sha256 is not None
        and authorization_payload_sha256 != contract_identity
    ):
        raise ValueError(
            "launch authorization does not bind the contract payload: supplied "
            f"{authorization_payload_sha256}, contract is {contract_identity}"
        )

    return {
        "schema_version": "t4_5ht1b2_protonation_rescue_preflight_v2",
        "status": (
            "LAUNCH_AUTHORIZATION_BOUND"
            if authorization_payload_sha256 is not None
            else "PREPARED_NO_SCORED_AUTHORIZATION"
        ),
        "arm": arm,
        "contract": relative_contract,
        "contract_payload_sha256": contract_identity,
        "contract_file_sha256": sha256_file(contract_path),
        "superseded_payload_sha256": superseded_identity,
        "authorization_receipt": AUTHORIZATION_RECEIPT,
        "authorization_receipt_payload_sha256": receipt_hash,
        "charged_call_ceiling": int(contract["total_charged_call_ceiling"]),
        "charged_calls_per_cell": int(contract["charged_calls_per_cell"]),
        "automatic_retries": 0,
        "cell": EXPECTED_CELL,
        "source_global_index": EXPECTED_SOURCE_INDEX,
        "delta": contract["delta"],
        "proposal_experts": sorted(proposal),
        "allocation_evidence_payload_sha256": identity(allocation_result),
        "pinned_gate": PINNED_GATE,
        "pinned_gate_eligible_unique": int(eligible),
        "runtime_inputs_sha256": material_hashes,
    }


__all__ = [
    "ARMS",
    "AUTHORIZATION_METADATA_KEYS",
    "AUTHORIZATION_RECEIPT",
    "EXPECTED_EXPERT",
    "EXPECTED_SOURCE_SMILES",
    "PINNED_GATE",
    "PREPARED_STATUS",
    "SCHEMA_VERSION",
    "assert_payload_carries_no_authorization",
    "reconstruct_superseded_payload",
    "validate_rescue_preflight_v2",
]
