"""Zero-oracle boundary from structural macro candidates to Dynamic-v0.

This module does not generate, rank, score, or refine molecules.  It validates
immutable candidate references, merges endpoint aliases without losing lane
provenance, and converts an admitted exact macro into the existing
``ProgramOptimizer`` record shape.  The optimizer itself remains unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from compose_v4.control.compositional_structural_subgoal_generator import PATCH_SCHEMA
from compose_v4.control.docking_value import identity
from compose_v4.control.edit_program import extract_program
from compose_v4.experiments.whole_ring_plan import execute_program
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state

ARCHIVE_SCHEMA = "t4_macro_shared_archive_v1"
RECEIPT_SCHEMA = "t4_macro_archive_integration_receipt_v1"
SELECTOR_SCHEMA = "t4_complete_macro_selector_candidate_lock_v1"


class MacroLane(StrEnum):
    LEARNED = "learned"
    UNIFORM = "uniform"


@dataclass(frozen=True)
class MacroCandidateReference:
    source_case_id: str
    candidate_identity: str
    generator_lock_sha256: str
    lane: MacroLane
    original_rank: int
    selector_rank: int | None = None
    selector_score: float | None = None

    def __post_init__(self) -> None:
        if (
            not self.source_case_id
            or len(self.candidate_identity) != 64
            or len(self.generator_lock_sha256) != 64
            or self.original_rank < 0
        ):
            raise ValueError("malformed immutable macro candidate reference")
        if self.lane is MacroLane.LEARNED:
            if self.selector_rank is None or self.selector_rank < 0:
                raise ValueError(
                    "learned references require a nonnegative selector rank"
                )
        elif self.selector_rank is not None or self.selector_score is not None:
            raise ValueError("uniform references cannot carry learned-selector fields")


@dataclass(frozen=True)
class EligibilityReceipt:
    candidate_identity: str
    filter_contract_sha256: str
    eligible: bool
    properties: Mapping[str, Any]

    def __post_init__(self) -> None:
        if len(self.candidate_identity) != 64 or len(self.filter_contract_sha256) != 64:
            raise ValueError("eligibility receipt requires physical identities")
        if type(self.eligible) is not bool:
            raise ValueError("eligibility status must be boolean")


@dataclass(frozen=True)
class AdmissionDecision:
    endpoint_key: str
    admitted: bool
    admission_rank: int | None
    policy: str

    def __post_init__(self) -> None:
        if not self.endpoint_key or not self.policy:
            raise ValueError("admission decision is incomplete")
        if self.admitted != (self.admission_rank is not None):
            raise ValueError("admitted decisions require exactly one rank")
        if self.admission_rank is not None and self.admission_rank < 0:
            raise ValueError("admission rank must be nonnegative")


CandidateResolver = Callable[[MacroCandidateReference], tuple[dict, dict]]
AdmissionPolicy = Callable[[tuple[dict, ...]], Sequence[AdmissionDecision]]


def selector_references(
    selector_envelope: Mapping[str, Any], *, generator_lock_sha256: str
) -> tuple[MacroCandidateReference, ...]:
    """Validate the selector lock boundary and return learned-lane references."""

    payload = selector_envelope.get("payload")
    if (
        not isinstance(payload, dict)
        or selector_envelope.get("payload_sha256") != identity(payload)
        or payload.get("schema_version") != SELECTOR_SCHEMA
        or payload.get("teacher_fields_present") is not False
        or payload.get("task_identity_present") is not False
        or payload.get("candidate_generation_calls") != 0
        or len(generator_lock_sha256) != 64
    ):
        raise ValueError(
            "selector lock violates the frozen candidate-reference boundary"
        )
    references = []
    for fold in payload.get("folds", []):
        for case in fold.get("cases", []):
            source_case_id = case.get("source_case_id")
            rows = case.get("ranked_candidates", [])
            if case.get("candidate_count") != len(rows):
                raise ValueError(
                    "selector candidate count disagrees with ranked references"
                )
            for row in rows:
                references.append(
                    MacroCandidateReference(
                        source_case_id=source_case_id,
                        candidate_identity=row["candidate_identity"],
                        generator_lock_sha256=generator_lock_sha256,
                        lane=MacroLane.LEARNED,
                        original_rank=int(row["original_rank"]),
                        selector_rank=int(row["selector_rank"]),
                        selector_score=float(row["selector_score"]),
                    )
                )
    if not references:
        raise ValueError("selector lock contains no candidate references")
    return tuple(references)


def candidate_resolver(generator_envelope: Mapping[str, Any]) -> CandidateResolver:
    """Index an immutable generator lock without copying candidate chemistry."""

    payload = generator_envelope.get("payload")
    if (
        not isinstance(payload, dict)
        or generator_envelope.get("payload_sha256") != identity(payload)
        or payload.get("teacher_fields_present") is not False
        or payload.get("task_identity_present") is not False
        or payload.get("new_oracle_calls", 0) != 0
    ):
        raise ValueError("generator lock violates the frozen zero-oracle boundary")
    index = {}
    for fold in payload.get("folds", []):
        for case in fold.get("cases", []):
            source_case_id, source_state = case.get("source_case_id"), case.get(
                "source_state"
            )
            for policy in case.get("policies", []):
                for raw in policy.get("candidates", []):
                    candidate_id = identity(raw)
                    key = (source_case_id, candidate_id)
                    if key in index:
                        raise ValueError(
                            "generator lock repeats a source/candidate identity"
                        )
                    index[key] = (source_state, raw)
    if not index:
        raise ValueError("generator lock contains no candidate payloads")

    def resolve(reference: MacroCandidateReference) -> tuple[dict, dict]:
        try:
            source_state, raw = index[
                (reference.source_case_id, reference.candidate_identity)
            ]
        except KeyError as error:
            raise ValueError(
                "selector candidate is absent from the declared generator lock"
            ) from error
        return deepcopy_json(source_state), deepcopy_json(raw)

    return resolve


def deepcopy_json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _validate_candidate(
    reference: MacroCandidateReference, source_state: dict, raw: dict
) -> str:
    if (
        raw.get("schema_version") != PATCH_SCHEMA
        or identity(raw) != reference.candidate_identity
    ):
        raise ValueError(
            "candidate reference does not resolve to its immutable payload"
        )
    realization = raw.get("realization", {})
    if (
        realization.get("status") != "realized"
        or realization.get("endpoint_matches_bound_target") is not True
        or realization.get("primitive_teacher_actions_used") != 0
    ):
        raise ValueError(
            "candidate lacks a passing teacher-free exact-realization receipt"
        )
    source = decode_state(source_state)
    endpoint, replay = execute_program(source, raw.get("actions", []))
    if encode_state(endpoint) != raw.get("endpoint_state"):
        raise ValueError("candidate action replay disagrees with its endpoint state")
    return replay["endpoint"]


def build_shared_archive(
    references: Sequence[MacroCandidateReference],
    *,
    resolver: CandidateResolver,
    eligibility: Mapping[str, EligibilityReceipt],
    admission_policy: AdmissionPolicy,
) -> dict:
    """Build one deterministic archive while retaining endpoint aliases.

    ``resolver`` is the only component allowed to read candidate chemistry.
    ``admission_policy`` sees deduplicated, eligible summaries and cannot alter
    candidate payloads.  Input order is deliberately erased.
    """

    if not references:
        raise ValueError("shared archive requires at least one candidate reference")
    if len({row.source_case_id for row in references}) != 1:
        raise ValueError(
            "one run-local shared archive must bind exactly one source case"
        )
    resolved = []
    seen_identity = set()
    for ref in sorted(
        references, key=lambda row: (row.source_case_id, row.candidate_identity)
    ):
        if ref.candidate_identity in seen_identity:
            raise ValueError("candidate identity was referenced more than once")
        seen_identity.add(ref.candidate_identity)
        receipt = eligibility.get(ref.candidate_identity)
        if receipt is None or receipt.candidate_identity != ref.candidate_identity:
            raise ValueError("candidate lacks a bound eligibility receipt")
        source_state, raw = resolver(ref)
        endpoint_key = _validate_candidate(ref, source_state, raw)
        resolved.append((ref, receipt, source_state, raw, endpoint_key))

    groups: dict[str, list[tuple]] = {}
    for row in resolved:
        groups.setdefault(row[-1], []).append(row)
    summaries, payload_by_endpoint = [], {}
    for endpoint_key, rows in sorted(groups.items()):
        source_ids = {row[0].source_case_id for row in rows}
        if len(source_ids) != 1:
            raise ValueError("one endpoint collision cannot cross source cases")
        rows.sort(
            key=lambda row: (
                0 if row[1].eligible else 1,
                0 if row[0].lane is MacroLane.LEARNED else 1,
                row[0].selector_rank if row[0].selector_rank is not None else 2**31,
                row[0].original_rank,
                row[0].candidate_identity,
            )
        )
        representative = rows[0]
        contributions = [asdict(row[0]) for row in rows]
        summary = {
            "endpoint_key": endpoint_key,
            "source_case_id": representative[0].source_case_id,
            "representative_candidate_identity": representative[0].candidate_identity,
            "eligible": any(row[1].eligible for row in rows),
            "contributions": contributions,
        }
        summaries.append(summary)
        payload_by_endpoint[endpoint_key] = representative

    eligible_summaries = tuple(row for row in summaries if row["eligible"])
    decisions = tuple(admission_policy(eligible_summaries))
    decision_by_endpoint = {row.endpoint_key: row for row in decisions}
    if len(decision_by_endpoint) != len(decisions) or set(decision_by_endpoint) != {
        row["endpoint_key"] for row in eligible_summaries
    }:
        raise ValueError(
            "admission policy must decide every eligible endpoint exactly once"
        )
    ranks = [row.admission_rank for row in decisions if row.admitted]
    if sorted(ranks) != list(range(len(ranks))):
        raise ValueError("admission ranks must be a contiguous zero-based ordering")

    entries = []
    for summary in summaries:
        decision = decision_by_endpoint.get(summary["endpoint_key"])
        if decision is None or not decision.admitted:
            continue
        ref, receipt, source_state, raw, endpoint_key = payload_by_endpoint[
            summary["endpoint_key"]
        ]
        entries.append(
            {
                "entry_id": identity(
                    {
                        "source_case_id": ref.source_case_id,
                        "endpoint_key": endpoint_key,
                        "candidate_identity": ref.candidate_identity,
                    }
                ),
                "source_case_id": ref.source_case_id,
                "endpoint_key": endpoint_key,
                "candidate_identity": ref.candidate_identity,
                "source_state": source_state,
                "candidate": raw,
                "eligibility": asdict(receipt),
                "admission": asdict(decision),
                "contributions": summary["contributions"],
            }
        )
    entries.sort(key=lambda row: row["admission"]["admission_rank"])
    body = {
        "schema_version": ARCHIVE_SCHEMA,
        "entries": entries,
        "endpoint_collision_ledger": summaries,
        "candidate_count": len(resolved),
        "unique_endpoint_count": len(groups),
        "eligible_endpoint_count": len(eligible_summaries),
        "admitted_endpoint_count": len(entries),
        "oracle_calls": 0,
    }
    return {**body, "archive_id": identity(body)}


def dynamic_v0_record(
    entry: Mapping[str, Any], *, source_group: str, oracle_protocol: str
) -> dict:
    """Convert one archive entry to the existing measured-program record shape.

    The returned record remains unmeasured.  A caller may pass it to unchanged
    ``DynamicProgramOptimizer.add_measured_program`` only after binding a real
    charged receipt and score.
    """

    if (
        not source_group
        or not oracle_protocol
        or entry.get("source_case_id") != source_group
    ):
        raise ValueError("Dynamic-v0 handoff requires the bound source/oracle domain")
    raw, source_state = entry["candidate"], entry["source_state"]
    source = decode_state(source_state)
    endpoint, trace = execute_program(source, raw["actions"])
    if canonical_state_key(endpoint) != entry["endpoint_key"]:
        raise ValueError("archive endpoint changed before Dynamic-v0 handoff")
    program, assignment = extract_program(source, [trace])
    return json.loads(
        json.dumps(
            {
                "source_group": source_group,
                "oracle_protocol": oracle_protocol,
                "source_state": source_state,
                "program": program.payload(),
                "assignment": list(assignment),
                "trace": trace,
                "endpoint": trace["endpoint"],
                "provenance": {
                    "schema_version": RECEIPT_SCHEMA,
                    "archive_id": entry["entry_id"],
                    "candidate_identity": entry["candidate_identity"],
                    "lane_contributions": entry["contributions"],
                    "oracle_calls": 0,
                },
                "score": None,
            }
        )
    )


def seed_sequences(seed: int) -> dict[str, list[int]]:
    """Return independent serializable SeedSequence states for four namespaces."""

    if type(seed) is not int or seed < 0:
        raise ValueError("integration seed must be a nonnegative integer")
    import numpy as np

    names = (
        "learned_macro_generation",
        "uniform_macro_generation",
        "archive_admission",
        "dynamic_v0_refinement",
    )
    children = np.random.SeedSequence(seed).spawn(len(names))
    return {
        name: child.generate_state(4).tolist()
        for name, child in zip(names, children, strict=True)
    }


__all__ = [
    "ARCHIVE_SCHEMA",
    "RECEIPT_SCHEMA",
    "SELECTOR_SCHEMA",
    "AdmissionDecision",
    "EligibilityReceipt",
    "MacroCandidateReference",
    "MacroLane",
    "build_shared_archive",
    "candidate_resolver",
    "dynamic_v0_record",
    "seed_sequences",
    "selector_references",
]
