"""Zero-oracle strict-delta-0.6 lock for frozen structural-subgoal endpoints."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import subprocess
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import QED, rdFingerprintGenerator
from rdkit.Contrib.SA_Score import sascorer

from compose_v4.chem.molecular_graph import molecular_graph_to_smiles
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.control.docking_value import identity
from compose_v4.rewrite.trace_shard import decode_state

CONTRACT_SCHEMA = "t4_delta06_structural_subgoal_utility_lock_contract_v1"
LEDGER_SCHEMA = "t4_delta06_structural_subgoal_eligibility_ledger_v1"
CANDIDATE_SCHEMA = "t4_delta06_structural_subgoal_candidate_lock_v1"
REQUEST_SCHEMA = "t4_delta06_structural_subgoal_request_lock_v1"
ABSTENTION_SCHEMA = "t4_delta06_structural_subgoal_abstention_ledger_v1"
RESULT_SCHEMA = "t4_delta06_structural_subgoal_utility_lock_result_v1"
POLICIES = (
    "graph_conditioned_subgoal",
    "source_balanced_marginal_subgoal",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _read_json(path: Path) -> dict:
    raw = (
        gzip.decompress(path.read_bytes())
        if path.suffix == ".gz"
        else path.read_bytes()
    )
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _sealed_payload(path: Path, *, expected_payload_hash: str | None = None) -> dict:
    envelope = _read_json(path)
    payload = envelope.get("payload")
    claimed = envelope.get("payload_sha256", envelope.get("contract_sha256"))
    if not isinstance(payload, dict) or claimed != identity(payload):
        raise ValueError(f"artifact is not self-hashed: {path}")
    if expected_payload_hash is not None and claimed != expected_payload_hash:
        raise ValueError(f"artifact payload identity changed: {path}")
    return payload


def load_contract(path: Path) -> dict:
    payload = _sealed_payload(path)
    if payload.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError(f"unexpected utility-lock contract schema: {path}")
    oracle = payload.get("oracle", {})
    if (
        oracle.get("calls_authorized") != 0
        or oracle.get("docking_calls_authorized") != 0
        or oracle.get("modal_launch_authorized") is not False
        or oracle.get("live_run_access_authorized") is not False
        or oracle.get("scored_launch_authorized") is not False
    ):
        raise ValueError("utility-lock contract must remain zero-oracle and unlaunched")
    return payload


def _walk_input_entries(value: Any) -> Iterable[dict]:
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            yield value
        else:
            for child in value.values():
                yield from _walk_input_entries(child)


def verify_inputs(root: Path, contract: dict) -> None:
    for entry in _walk_input_entries(contract["inputs"]):
        path = root / entry["path"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"frozen utility-lock input changed: {path}")


def verify_environment(contract: dict) -> None:
    expected = contract["environment"]
    if rdBase.rdkitVersion != expected["rdkit"]:
        raise ValueError(
            f"utility lock requires RDKit {expected['rdkit']}; got {rdBase.rdkitVersion}"
        )
    scorer = Path(sascorer.__file__)
    if sha256_file(scorer) != expected["sa_scorer_sha256"]:
        raise ValueError("SA scorer implementation identity changed")
    if (
        sha256_file(scorer.with_name("fpscores.pkl.gz"))
        != expected["sa_fragment_scores_sha256"]
    ):
        raise ValueError("SA fragment-score identity changed")


@dataclass(frozen=True)
class EligibilityThresholds:
    similarity_gt: float
    qed_gt: float
    sa_lt: float
    active_atoms_lte: int

    @classmethod
    def from_contract(cls, payload: dict) -> EligibilityThresholds:
        return cls(
            similarity_gt=float(payload["similarity_strictly_greater_than"]),
            qed_gt=float(payload["qed_strictly_greater_than"]),
            sa_lt=float(payload["sa_strictly_less_than"]),
            active_atoms_lte=int(payload["active_atoms_less_than_or_equal_to"]),
        )


def endpoint_exclusion_reasons(
    *,
    valid: bool,
    connected: bool,
    is_null: bool,
    canonical_available: bool,
    active_heavy_match: bool | None,
    non_self: bool | None,
    similarity: float | None,
    qed: float | None,
    sa: float | None,
    active_atoms: int,
    thresholds: EligibilityThresholds,
) -> list[str]:
    """Return every failed eligibility invariant in a stable policy order."""
    reasons = []
    if not valid:
        reasons.append("invalid_molecular_graph")
    if not connected:
        reasons.append("disconnected_molecular_graph")
    if is_null:
        reasons.append("null_endpoint")
    if not canonical_available:
        reasons.append("canonical_molecule_unavailable")
    if active_heavy_match is False:
        reasons.append("active_heavy_atom_count_mismatch")
    if non_self is False:
        reasons.append("self_endpoint")
    if similarity is None or not similarity > thresholds.similarity_gt:
        reasons.append("similarity_not_strictly_greater_than_0.6")
    if qed is None or not qed > thresholds.qed_gt:
        reasons.append("qed_not_strictly_greater_than_0.6")
    if sa is None or not sa < thresholds.sa_lt:
        reasons.append("sa_not_strictly_less_than_4.0")
    if active_atoms > thresholds.active_atoms_lte:
        reasons.append("active_atoms_greater_than_40")
    return reasons


def _source_descriptor(
    source_state_payload: dict,
    generator,
) -> dict:
    state = decode_state(source_state_payload)
    if (
        not is_valid_state(state)
        or not is_connected_or_null(state)
        or state.n_real_atoms == 0
    ):
        raise ValueError("frozen T4 source state is not a valid connected molecule")
    smiles = molecular_graph_to_smiles(state)
    molecule = Chem.MolFromSmiles(smiles) if smiles is not None else None
    if molecule is None:
        raise ValueError("frozen T4 source state has no canonical molecule")
    canonical = Chem.MolToSmiles(molecule)
    if molecule.GetNumHeavyAtoms() != state.n_real_atoms:
        raise ValueError("frozen T4 source active/heavy atom counts disagree")
    return {
        "state": state,
        "canonical_smiles": canonical,
        "fingerprint": generator.GetFingerprint(molecule),
        "active_atoms": int(state.n_real_atoms),
    }


def describe_attempt(
    attempt: dict,
    *,
    fold: int,
    cell: str,
    target: str,
    source_case_id: str,
    source_canonical_smiles: str,
    source_fingerprint,
    policy_id: str,
    generator,
    thresholds: EligibilityThresholds,
) -> dict:
    status = attempt.get("status")
    row = {
        "attempt_uid": identity(
            {
                "fold": fold,
                "source_case_id": source_case_id,
                "policy_id": policy_id,
                "attempt_id": attempt.get("attempt_id"),
            }
        ),
        "fold": fold,
        "cell": cell,
        "target": target,
        "source_case_id": source_case_id,
        "source_canonical_smiles": source_canonical_smiles,
        "policy_id": policy_id,
        "rank": attempt.get("rank"),
        "attempt_id": attempt.get("attempt_id"),
        "generation_status": status,
        "compiler_strategy": attempt.get("compiler_strategy"),
        "component_count": attempt.get("component_count"),
        "primitive_count": attempt.get("primitive_count"),
        "delta_template_ids": attempt.get("delta_template_ids"),
        "candidate_trace_available": any(
            key in attempt for key in ("trace", "actions", "states")
        ),
        "endpoint_state_available": attempt.get("endpoint_state") is not None,
        "valid": None,
        "connected": None,
        "canonical_smiles": None,
        "qed": None,
        "sa": None,
        "morgan_radius2_2048_similarity_to_source": None,
        "active_atoms": None,
        "heavy_atoms": None,
        "non_self": None,
        "eligible": False,
        "exclusion_reasons": [],
    }
    if row["candidate_trace_available"]:
        raise ValueError(f"candidate trace unexpectedly present: {row['attempt_uid']}")
    if status != "complete":
        if row["endpoint_state_available"]:
            raise ValueError(
                "incomplete proposal unexpectedly carries an endpoint state"
            )
        row["exclusion_reasons"] = [
            f"generation_status:{status}",
            "endpoint_state_unavailable",
        ]
        return row
    if not row["endpoint_state_available"]:
        raise ValueError("complete proposal is missing its endpoint state")

    state = decode_state(attempt["endpoint_state"])
    active_atoms = int(state.n_real_atoms)
    valid = bool(is_valid_state(state))
    connected = bool(is_connected_or_null(state))
    state_smiles = molecular_graph_to_smiles(state) if active_atoms else None
    molecule = Chem.MolFromSmiles(state_smiles) if state_smiles is not None else None
    canonical = Chem.MolToSmiles(molecule) if molecule is not None else None
    heavy_atoms = int(molecule.GetNumHeavyAtoms()) if molecule is not None else None
    similarity = (
        float(
            DataStructs.TanimotoSimilarity(
                source_fingerprint, generator.GetFingerprint(molecule)
            )
        )
        if molecule is not None
        else None
    )
    qed = float(QED.qed(molecule)) if molecule is not None else None
    sa = float(sascorer.calculateScore(molecule)) if molecule is not None else None
    non_self = canonical != source_canonical_smiles if canonical is not None else None
    active_heavy_match = (
        active_atoms == heavy_atoms if heavy_atoms is not None else None
    )
    reasons = endpoint_exclusion_reasons(
        valid=valid,
        connected=connected,
        is_null=active_atoms == 0,
        canonical_available=canonical is not None,
        active_heavy_match=active_heavy_match,
        non_self=non_self,
        similarity=similarity,
        qed=qed,
        sa=sa,
        active_atoms=active_atoms,
        thresholds=thresholds,
    )
    row.update(
        {
            "valid": valid,
            "connected": connected,
            "canonical_smiles": canonical,
            "qed": qed,
            "sa": sa,
            "morgan_radius2_2048_similarity_to_source": similarity,
            "active_atoms": active_atoms,
            "heavy_atoms": heavy_atoms,
            "non_self": non_self,
            "eligible": not reasons,
            "exclusion_reasons": reasons,
        }
    )
    return row


def choose_unit_candidates(
    eligible: list[dict],
    *,
    distance: Callable[[dict, dict], float],
) -> list[tuple[str, dict, float | None]]:
    """Choose lowest rank, then a distinct endpoint farthest from it."""
    canonical_best: dict[str, dict] = {}
    for row in eligible:
        canonical = row["canonical_smiles"]
        current = canonical_best.get(canonical)
        key = (row["rank"], canonical, row["attempt_id"])
        if current is None or key < (
            current["rank"],
            current["canonical_smiles"],
            current["attempt_id"],
        ):
            canonical_best[canonical] = row
    ordered = sorted(
        canonical_best.values(),
        key=lambda row: (row["rank"], row["canonical_smiles"], row["attempt_id"]),
    )
    if not ordered:
        return []
    first = ordered[0]
    selected: list[tuple[str, dict, float | None]] = [
        ("lowest_rank_eligible", first, None)
    ]
    if len(ordered) > 1:
        candidates = [
            (float(distance(first, row)), row)
            for row in ordered[1:]
            if row["canonical_smiles"] != first["canonical_smiles"]
        ]
        farthest_distance, farthest = min(
            candidates,
            key=lambda item: (
                -item[0],
                item[1]["rank"],
                item[1]["canonical_smiles"],
                item[1]["attempt_id"],
            ),
        )
        selected.append(("maximally_morgan_distant", farthest, farthest_distance))
    return selected


def _evaluator(cell: dict, unit: dict, scoring_sha256: str) -> dict:
    domain = unit["oracle_domain"]
    if (
        unit["cell"] != cell["cell"]
        or unit["target"] != cell["target"]
        or unit["original_seed"] != cell["original_seed"]
        or unit["source_idx"] != cell["source_idx"]
    ):
        raise ValueError(f"replicate-0 unit disagrees with frozen cell: {cell['cell']}")
    return {
        "schema_version": "t4_quickvina_endpoint_evaluator_v1",
        "target": cell["target"],
        "box_definition": domain["box_definition"],
        "docking": domain["docking"],
        "qvina02_sha256": domain["qvina02_sha256"],
        "receptor_sha256": domain["receptor_sha256"],
        "scoring_implementation_sha256": scoring_sha256,
        "score_direction": "minimize",
        "score_unit": "kcal/mol as emitted by the bound QuickVina wrapper",
    }


def build_request_records(memberships: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in memberships:
        key = (
            row["target"],
            row["canonical_smiles"],
            row["docking_seed"],
            row["evaluator"],
        )
        grouped[key].append(row)
    requests = []
    for key, rows in sorted(grouped.items()):
        target, canonical, docking_seed, evaluator = key
        request_identity = {
            "schema_version": "t4_docking_request_identity_v1",
            "target": target,
            "canonical_smiles": canonical,
            "docking_seed": docking_seed,
            "evaluator": evaluator,
        }
        requests.append(
            {
                "request_id": identity(request_identity),
                **request_identity,
                "evaluator_payload": rows[0]["evaluator_payload"],
                "membership_ids": sorted(row["membership_id"] for row in rows),
                "memberships": sorted(
                    [
                        {
                            field: row[field]
                            for field in (
                                "membership_id",
                                "fold",
                                "cell",
                                "source_case_id",
                                "policy_id",
                                "selection_role",
                                "rank",
                                "attempt_id",
                            )
                        }
                        for row in rows
                    ],
                    key=lambda row: (
                        row["cell"],
                        row["policy_id"],
                        row["selection_role"],
                        row["rank"],
                        row["attempt_id"],
                    ),
                ),
            }
        )
    return requests


def _code_revision(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def _source_dirty(root: Path) -> bool:
    return bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).strip()
    )


def build_payloads(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = load_contract(contract_path)
    verify_inputs(root, contract)
    verify_environment(contract)
    thresholds = EligibilityThresholds.from_contract(contract["eligibility"])
    fingerprint_spec = contract["eligibility"]["fingerprint"]
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(fingerprint_spec["radius"]),
        fpSize=int(fingerprint_spec["bits"]),
        includeChirality=bool(fingerprint_spec["include_chirality"]),
    )

    t4_input = contract["inputs"]["frozen_t4_source_cell_contract"]
    t4 = _sealed_payload(
        root / t4_input["path"], expected_payload_hash=t4_input["payload_sha256"]
    )
    cells = t4["cells"]
    if sorted(cells) != sorted(
        f"{target}_{source}"
        for target in ("5ht1b", "braf", "fa7", "jak2", "parp1")
        for source in range(3)
    ):
        raise ValueError("frozen T4 contract does not contain the exact 15-cell panel")
    cell_by_source_identity = {}
    for cell_name, cell in sorted(cells.items()):
        if (
            cell_name != cell["cell"]
            or identity(cell["source_state"]) != cell["source_state_sha256"]
        ):
            raise ValueError(f"frozen source-state identity mismatch: {cell_name}")
        source_identity = identity(cell["source_state"])
        if source_identity in cell_by_source_identity:
            raise ValueError("frozen T4 source-state collision")
        cell_by_source_identity[source_identity] = cell
    replicate_zero = {
        row["cell"]: row for row in t4["units"] if row.get("replicate") == 0
    }
    if sorted(replicate_zero) != sorted(cells):
        raise ValueError("frozen T4 contract lacks one replicate-0 unit per cell")
    scoring_sha = contract["inputs"]["docking_scoring_implementation"]["sha256"]

    rows = []
    observed_source_cases: dict[str, dict] = {}
    for fold_text, fold_inputs in sorted(contract["inputs"]["folds"].items()):
        fold = int(fold_text)
        source_manifest = _sealed_payload(
            root / fold_inputs["source_manifest"]["path"],
            expected_payload_hash=fold_inputs["source_manifest"]["payload_sha256"],
        )
        candidate_lock = _sealed_payload(
            root / fold_inputs["candidate_lock"]["path"],
            expected_payload_hash=fold_inputs["candidate_lock"]["payload_sha256"],
        )
        if (
            source_manifest.get("fold") != fold
            or candidate_lock.get("fold") != fold
            or source_manifest.get("new_oracle_calls") != 0
            or candidate_lock.get("new_oracle_calls") != 0
            or candidate_lock.get("teacher_fields_present") is not False
            or candidate_lock.get("task_identity_present") is not False
        ):
            raise ValueError(
                f"fold {fold} source/candidate lock violates zero-oracle schema"
            )
        source_cases = {
            row["source_case_id"]: row for row in source_manifest["source_cases"]
        }
        locked_cases = {
            row["source_case_id"]: row for row in candidate_lock["source_cases"]
        }
        if len(source_cases) != 5 or set(source_cases) != set(locked_cases):
            raise ValueError(f"fold {fold} source-case manifests disagree")
        for source_case_id, source_row in sorted(source_cases.items()):
            locked = locked_cases[source_case_id]
            if source_row["source_state"] != locked["source_state"]:
                raise ValueError(f"source state drift in fold {fold}: {source_case_id}")
            source_identity = identity(source_row["source_state"])
            cell = cell_by_source_identity.get(source_identity)
            if cell is None:
                raise ValueError(
                    f"source case is outside frozen T4 cells: {source_case_id}"
                )
            if cell["cell"] in observed_source_cases:
                raise ValueError(f"T4 cell occurs in multiple folds: {cell['cell']}")
            observed_source_cases[cell["cell"]] = {
                "fold": fold,
                "source_case_id": source_case_id,
                "source_state_sha256": source_identity,
            }
            source = _source_descriptor(source_row["source_state"], generator)
            original = Chem.MolFromSmiles(cell["original_seed"])
            if original is None or Chem.MolToSmiles(
                original, isomericSmiles=False
            ) != Chem.MolToSmiles(
                Chem.MolFromSmiles(source["canonical_smiles"]), isomericSmiles=False
            ):
                raise ValueError(
                    f"source graph and original seed disagree: {cell['cell']}"
                )
            policies = {row["policy_id"]: row for row in locked["policies"]}
            if set(policies) != set(POLICIES):
                raise ValueError(
                    f"source case has unexpected policy set: {source_case_id}"
                )
            for policy_id in POLICIES:
                attempts = policies[policy_id]["attempts"]
                ranks = [row.get("rank") for row in attempts]
                if len(attempts) != 128 or sorted(ranks) != list(range(1, 129)):
                    raise ValueError(
                        f"source-policy attempt ranks are incomplete: {source_case_id}/{policy_id}"
                    )
                rows.extend(
                    describe_attempt(
                        attempt,
                        fold=fold,
                        cell=cell["cell"],
                        target=cell["target"],
                        source_case_id=source_case_id,
                        source_canonical_smiles=source["canonical_smiles"],
                        source_fingerprint=source["fingerprint"],
                        policy_id=policy_id,
                        generator=generator,
                        thresholds=thresholds,
                    )
                    for attempt in attempts
                )

    if sorted(observed_source_cases) != sorted(cells):
        raise ValueError("candidate locks do not cover the exact frozen 15-cell panel")
    rows.sort(
        key=lambda row: (
            row["fold"],
            row["cell"],
            row["policy_id"],
            row["rank"],
            row["attempt_id"],
        )
    )
    expected = contract["preregistered_preview"]
    if len(rows) != expected["attempts"]:
        raise RuntimeError(f"authoritative attempt count differs: {len(rows)}")
    complete = sum(row["generation_status"] == "complete" for row in rows)
    if complete != expected["completed_endpoints"]:
        raise RuntimeError(
            f"authoritative completed-endpoint count differs: {complete}"
        )

    by_unit: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["eligible"]:
            by_unit[(row["cell"], row["policy_id"])].append(row)

    def distance(first: dict, second: dict) -> float:
        a = Chem.MolFromSmiles(first["canonical_smiles"])
        b = Chem.MolFromSmiles(second["canonical_smiles"])
        if a is None or b is None:
            raise ValueError(
                "eligible candidate lost canonical molecule during selection"
            )
        return 1.0 - float(
            DataStructs.TanimotoSimilarity(
                generator.GetFingerprint(a), generator.GetFingerprint(b)
            )
        )

    memberships = []
    for (cell_name, policy_id), eligible in sorted(by_unit.items()):
        cell = cells[cell_name]
        unit = replicate_zero[cell_name]
        evaluator_payload = _evaluator(cell, unit, scoring_sha)
        evaluator = identity(evaluator_payload)
        for selection_index, (role, row, distance_to_first) in enumerate(
            choose_unit_candidates(eligible, distance=distance), start=1
        ):
            membership_body = {
                "fold": row["fold"],
                "cell": cell_name,
                "target": cell["target"],
                "source_case_id": row["source_case_id"],
                "source_state_sha256": observed_source_cases[cell_name][
                    "source_state_sha256"
                ],
                "policy_id": policy_id,
                "selection_index": selection_index,
                "selection_role": role,
                "rank": row["rank"],
                "attempt_id": row["attempt_id"],
                "attempt_uid": row["attempt_uid"],
                "canonical_smiles": row["canonical_smiles"],
                "qed": row["qed"],
                "sa": row["sa"],
                "morgan_radius2_2048_similarity_to_source": row[
                    "morgan_radius2_2048_similarity_to_source"
                ],
                "morgan_radius2_2048_distance_to_first": distance_to_first,
                "active_atoms": row["active_atoms"],
                "heavy_atoms": row["heavy_atoms"],
                "docking_seed": int(contract["selection"]["docking_seed"]),
                "evaluator": evaluator,
                "evaluator_payload": evaluator_payload,
                "candidate_trace_available": False,
            }
            memberships.append(
                {"membership_id": identity(membership_body), **membership_body}
            )
    memberships.sort(
        key=lambda row: (
            row["fold"],
            row["cell"],
            row["policy_id"],
            row["selection_index"],
            row["rank"],
            row["attempt_id"],
        )
    )
    requests = build_request_records(memberships)
    request_by_membership = {
        membership_id: request["request_id"]
        for request in requests
        for membership_id in request["membership_ids"]
    }
    memberships = [
        {**row, "request_id": request_by_membership[row["membership_id"]]}
        for row in memberships
    ]
    selected_cells = {row["cell"] for row in memberships}
    observed_counts = {
        "selected_memberships": len(memberships),
        "unique_requests": len(requests),
        "cells_with_requests": len(selected_cells),
        "cell_abstentions": len(cells) - len(selected_cells),
    }
    for field, value in observed_counts.items():
        if value != expected[field]:
            raise RuntimeError(
                f"authoritative {field} differs from preregistered preview: {value}"
            )

    contract_sha = _read_json(contract_path)["contract_sha256"]
    common = {
        "contract": {
            "path": str(contract_path.relative_to(root)),
            "contract_sha256": contract_sha,
            "physical_sha256": sha256_file(contract_path),
        },
        "code_revision": _code_revision(root),
        "new_oracle_calls": 0,
        "new_docking_calls": 0,
        "modal_launches": 0,
    }
    ledger_payload = {
        "schema_version": LEDGER_SCHEMA,
        **common,
        "selection_inputs": {
            "teacher_endpoint_accessed": False,
            "teacher_transformation_accessed": False,
            "teacher_score_accessed": False,
            "docking_score_accessed": False,
            "candidate_regeneration": False,
            "semantic_evaluation_manifest_access": False,
            "note": "Evaluation manifests and reports were verified only by contract-bound physical SHA-256; their teacher-bearing payloads were not loaded for selection.",
        },
        "eligibility": contract["eligibility"],
        "attempts": rows,
    }
    candidate_payload = {
        "schema_version": CANDIDATE_SCHEMA,
        **common,
        "selection": contract["selection"],
        "candidate_trace_availability": {
            "available": False,
            "observed_trace_fields": 0,
            "consequence": "Endpoint utility cannot establish route recovery or executable candidate-route precision.",
        },
        "memberships": memberships,
    }
    candidate_payload_sha = identity(candidate_payload)
    request_payload = {
        "schema_version": REQUEST_SCHEMA,
        **common,
        "candidate_lock_payload_sha256": candidate_payload_sha,
        "query_identity": contract["selection"]["deduplicate_requests_by"],
        "scoring_status": "UNAUTHORIZED_NOT_LAUNCHED",
        "call_ceiling_if_later_authorized": len(requests),
        "automatic_retries": 0,
        "replacement_or_backfill": False,
        "candidate_trace_available": False,
        "interpretation_limit": contract["interpretation_limit"],
        "requests": requests,
    }

    memberships_by_cell: dict[str, list[dict]] = defaultdict(list)
    for row in memberships:
        memberships_by_cell[row["cell"]].append(row)
    abstention_rows = []
    for cell_name, cell in sorted(cells.items()):
        policy_rows = []
        for policy_id in POLICIES:
            eligible = by_unit.get((cell_name, policy_id), [])
            selected = [
                row
                for row in memberships_by_cell[cell_name]
                if row["policy_id"] == policy_id
            ]
            policy_rows.append(
                {
                    "policy_id": policy_id,
                    "eligible_attempts": len(eligible),
                    "eligible_unique_canonical_endpoints": len(
                        {row["canonical_smiles"] for row in eligible}
                    ),
                    "selected_memberships": len(selected),
                    "status": (
                        "candidate_locked"
                        if selected
                        else "abstain_no_strict_eligible_candidate"
                    ),
                }
            )
        abstention_rows.append(
            {
                "cell": cell_name,
                "target": cell["target"],
                "fold": observed_source_cases[cell_name]["fold"],
                "source_case_id": observed_source_cases[cell_name]["source_case_id"],
                "status": (
                    "candidate_locked"
                    if memberships_by_cell[cell_name]
                    else "abstain_no_strict_eligible_candidate_in_either_policy"
                ),
                "policy_statuses": policy_rows,
            }
        )
    abstention_payload = {
        "schema_version": ABSTENTION_SCHEMA,
        **common,
        "candidate_lock_payload_sha256": candidate_payload_sha,
        "cells": abstention_rows,
    }

    exclusion_counts = Counter(
        reason for row in rows for reason in row["exclusion_reasons"]
    )
    per_policy = []
    for policy_id in POLICIES:
        subset = [row for row in rows if row["policy_id"] == policy_id]
        selected = [row for row in memberships if row["policy_id"] == policy_id]
        per_policy.append(
            {
                "policy_id": policy_id,
                "attempts": len(subset),
                "completed": sum(
                    row["generation_status"] == "complete" for row in subset
                ),
                "valid_completed": sum(row["valid"] is True for row in subset),
                "connected_completed": sum(row["connected"] is True for row in subset),
                "eligible_attempts": sum(row["eligible"] for row in subset),
                "eligible_source_conditioned_unique_endpoints": len(
                    {
                        (row["cell"], row["canonical_smiles"])
                        for row in subset
                        if row["eligible"]
                    }
                ),
                "source_policy_units_with_eligibility": len(
                    {row["cell"] for row in subset if row["eligible"]}
                ),
                "selected_memberships": len(selected),
            }
        )
    per_cell = []
    for cell_name in sorted(cells):
        subset = [row for row in rows if row["cell"] == cell_name]
        selected = memberships_by_cell[cell_name]
        per_cell.append(
            {
                "cell": cell_name,
                "target": cells[cell_name]["target"],
                "attempts": len(subset),
                "completed": sum(
                    row["generation_status"] == "complete" for row in subset
                ),
                "eligible_attempts": sum(row["eligible"] for row in subset),
                "eligible_unique_canonical_endpoints": len(
                    {row["canonical_smiles"] for row in subset if row["eligible"]}
                ),
                "selected_memberships": len(selected),
                "status": "candidate_locked" if selected else "cell_abstention",
            }
        )
    result_payload = {
        "schema_version": RESULT_SCHEMA,
        **common,
        "evidence": "computed zero-oracle endpoint eligibility and score-blind deterministic selection",
        "environment": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "precision": contract["environment"]["numeric_precision"],
            "randomness": contract["environment"]["randomness"],
        },
        "inputs": contract["inputs"],
        "summary": {
            "attempts": len(rows),
            "completed_endpoints": complete,
            "valid_completed_endpoints": sum(row["valid"] is True for row in rows),
            "connected_completed_endpoints": sum(
                row["connected"] is True for row in rows
            ),
            "eligible_attempts": sum(row["eligible"] for row in rows),
            "eligible_source_conditioned_unique_endpoints": len(
                {
                    (row["cell"], row["canonical_smiles"])
                    for row in rows
                    if row["eligible"]
                }
            ),
            "source_policy_units": 30,
            "source_policy_units_with_eligibility": len(by_unit),
            **observed_counts,
            "policy_unit_abstentions": 30 - len(by_unit),
            "candidate_trace_fields_observed": 0,
            "new_oracle_calls": 0,
            "new_docking_calls": 0,
            "modal_launches": 0,
        },
        "exclusion_reason_counts": dict(sorted(exclusion_counts.items())),
        "per_policy": per_policy,
        "per_cell": per_cell,
        "preregistered_preview": expected,
        "preregistered_preview_match": True,
        "negative_findings_and_abstentions": {
            "cell_abstentions": sorted(set(cells) - selected_cells),
            "policy_unit_abstentions": [
                {"cell": cell_name, "policy_id": policy_id}
                for cell_name in sorted(cells)
                for policy_id in POLICIES
                if (cell_name, policy_id) not in by_unit
            ],
            "candidate_traces_available": False,
        },
        "interpretation": {
            "measured": "The frozen candidate set contains a small strict-delta-0.6 eligible panel selected without task scores.",
            "not_measured": "No docking utility, teacher recovery, executable candidate route, route precision or optimization improvement was evaluated.",
            "later_launch_authority": contract["oracle"][
                "later_authorization_required"
            ],
        },
    }
    return {
        "eligibility_ledger.json.gz": ledger_payload,
        "candidate_lock.json": candidate_payload,
        "request_lock.json": request_payload,
        "abstention_ledger.json": abstention_payload,
        "result.json": result_payload,
    }


def _sealed_bytes(payload: dict, *, compressed: bool) -> bytes:
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    raw = canonical_json(envelope) + b"\n"
    return gzip.compress(raw, mtime=0) if compressed else raw


def _readme(result: dict, physical: dict[str, str], request_payload_hash: str) -> str:
    summary = result["summary"]
    abstentions = result["negative_findings_and_abstentions"]["cell_abstentions"]
    return f"""# T4 strict-delta-0.6 structural-subgoal utility lock

## Outcome

The zero-oracle lock passed its preregistered count checks. All {summary['attempts']:,}
frozen attempts were audited under RDKit 2024.03.5. The {summary['completed_endpoints']:,}
completed endpoints yielded {summary['eligible_source_conditioned_unique_endpoints']} unique
strictly eligible source-conditioned endpoints. Deterministic within-source/policy
selection produced {summary['selected_memberships']} memberships and
{summary['unique_requests']} deduplicated prospective docking requests across
{summary['cells_with_requests']} of 15 cells.

The seven explicit cell abstentions are: {', '.join(abstentions)}.

No oracle, docking function, Modal job or live run was called. The request lock is
not launch authority. Candidate traces are unavailable in the frozen input locks,
so endpoint utility cannot establish route recovery or executable candidate-route
precision.

## Artifacts

- `eligibility_ledger.json.gz`: all 3,840 attempts, descriptors and exclusions
- `candidate_lock.json`: all 22 source/policy selection memberships
- `request_lock.json`: 19 deduplicated, currently unauthorized physical requests
- `abstention_ledger.json`: all 15 cells and all 30 source/policy statuses
- `result.json`: aggregate, per-policy and per-cell findings

Request-lock payload SHA-256: `{request_payload_hash}`

Physical SHA-256 values at publication:

{os.linesep.join(f'- `{name}`: `{digest}`' for name, digest in sorted(physical.items()))}

## Required next authorization

To launch, the user must explicitly authorize the exact physical SHA-256 of
`request_lock.json`, a ceiling of 19 first-score docking calls, no retry, no
replacement and no backfill. Until then, the panel remains unscored.
"""


def publish_bundle(output: Path, payloads: dict[str, dict]) -> dict[str, str]:
    if output.exists():
        raise ValueError(f"refusing to overwrite utility-lock output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise ValueError(f"stale utility-lock temporary directory exists: {temporary}")
    temporary.mkdir()
    primary = [
        "eligibility_ledger.json.gz",
        "candidate_lock.json",
        "request_lock.json",
        "abstention_ledger.json",
    ]
    for name in primary:
        (temporary / name).write_bytes(
            _sealed_bytes(payloads[name], compressed=name.endswith(".gz"))
        )
    physical = {name: sha256_file(temporary / name) for name in primary}
    result = {
        **payloads["result.json"],
        "output_artifacts": {
            name: {
                "sha256": physical[name],
                "payload_sha256": identity(payloads[name]),
            }
            for name in primary
        },
    }
    (temporary / "result.json").write_bytes(_sealed_bytes(result, compressed=False))
    physical["result.json"] = sha256_file(temporary / "result.json")
    readme = _readme(result, physical, identity(payloads["request_lock.json"]))
    (temporary / "README.md").write_text(readme)
    physical["README.md"] = sha256_file(temporary / "README.md")
    temporary.replace(output)
    return dict(sorted(physical.items()))


def run(root: Path, contract_path: Path, output: Path) -> dict[str, str]:
    if _source_dirty(root):
        raise ValueError("utility lock requires a clean committed source worktree")
    payloads = build_payloads(root, contract_path)
    return publish_bundle(output, payloads)
