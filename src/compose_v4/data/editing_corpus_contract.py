"""Fail-closed validation for the versioned COMPOSE editing-corpus contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


EXPECTED_SCHEMA = "compose.editing_corpus_contract"
EXPECTED_SCHEMA_VERSION = 1
REQUIRED_EXPERIMENTS = frozenset({"E2", "E3", "E4", "E7"})
REQUIRED_CORE_FAMILIES = frozenset(
    {
        "atom_insert",
        "atom_delete",
        "atom_restate",
        "bond_reorder",
        "bond_reroute",
        "cycle_insert",
        "cycle_attach",
    }
)


class EditingCorpusContractError(ValueError):
    """The proposed corpus contract is malformed or not launch-ready."""


def load_editing_corpus_contract(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    validate_editing_corpus_contract(payload)
    return payload


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicate: set[str] = set()
    for value in values:
        if value in seen:
            duplicate.add(value)
        seen.add(value)
    return sorted(duplicate)


def validate_editing_corpus_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != EXPECTED_SCHEMA:
        raise EditingCorpusContractError("unexpected editing-corpus schema")
    if contract.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise EditingCorpusContractError("unexpected editing-corpus schema version")

    basis = contract.get("operator_basis") or {}
    required_families = set(basis.get("required") or ())
    if required_families != REQUIRED_CORE_FAMILIES:
        raise EditingCorpusContractError(
            "required operator basis must equal the seven-family core"
        )
    disabled = set(basis.get("disabled_by_default") or ())
    overlap = required_families & disabled
    if overlap:
        raise EditingCorpusContractError(
            f"operator families are both required and disabled: {sorted(overlap)}"
        )

    lanes = contract.get("data_lanes") or []
    lane_ids = [str(lane.get("id")) for lane in lanes]
    duplicate_lanes = _duplicates(lane_ids)
    if duplicate_lanes:
        raise EditingCorpusContractError(
            f"duplicate data-lane identifiers: {duplicate_lanes}"
        )
    known_lanes = set(lane_ids)

    capabilities = contract.get("capabilities") or []
    capability_ids = [str(capability.get("id")) for capability in capabilities]
    duplicate_capabilities = _duplicates(capability_ids)
    if duplicate_capabilities:
        raise EditingCorpusContractError(
            f"duplicate capability identifiers: {duplicate_capabilities}"
        )
    covered_experiments: set[str] = set()
    for capability in capabilities:
        capability_id = str(capability.get("id"))
        evidence_lanes = set(capability.get("evidence_lanes") or ())
        unknown_lanes = evidence_lanes - known_lanes
        if unknown_lanes:
            raise EditingCorpusContractError(
                f"{capability_id} references unknown lanes: {sorted(unknown_lanes)}"
            )
        family_names = set(capability.get("required_families_all") or ())
        family_names.update(capability.get("required_families_any") or ())
        unknown_families = family_names - (
            required_families
            | set(basis.get("conditional_on_reachability_audit") or ())
        )
        if unknown_families:
            raise EditingCorpusContractError(
                f"{capability_id} references undeclared families: "
                f"{sorted(unknown_families)}"
            )
        covered_experiments.update(capability.get("experiments") or ())

    missing_experiments = REQUIRED_EXPERIMENTS - covered_experiments
    if missing_experiments:
        raise EditingCorpusContractError(
            f"capability matrix does not cover experiments: {sorted(missing_experiments)}"
        )

    required_fields = [str(field) for field in contract.get("required_record_fields") or ()]
    duplicate_fields = _duplicates(required_fields)
    if duplicate_fields:
        raise EditingCorpusContractError(
            f"duplicate required record fields: {duplicate_fields}"
        )
    for field in (
        "source_state_exact",
        "target_state_exact",
        "executable_actions",
        "operator_sequence",
        "source_provenance",
        "operator_contract_hash",
    ):
        if field not in required_fields:
            raise EditingCorpusContractError(f"required record field is absent: {field}")


def training_launch_blockers(contract: dict[str, Any]) -> list[str]:
    """Return every unresolved item that must block a new training launch."""

    validate_editing_corpus_contract(contract)
    blockers: list[str] = []
    if contract.get("training_authorized") is not True:
        blockers.append("training_authorized is not true")
    if contract.get("status") != "FROZEN_TRAINING_AUTHORIZED":
        blockers.append("status is not FROZEN_TRAINING_AUTHORIZED")
    thresholds = (
        contract.get("pretraining_gates", {})
        .get("thresholds_to_freeze_after_development_census", {})
    )
    for name, value in sorted(thresholds.items()):
        if value is None:
            blockers.append(f"unfrozen threshold: {name}")
    return blockers


def assert_training_launch_authorized(contract: dict[str, Any]) -> None:
    blockers = training_launch_blockers(contract)
    if blockers:
        raise EditingCorpusContractError(
            "editing-corpus contract blocks training: " + "; ".join(blockers)
        )
