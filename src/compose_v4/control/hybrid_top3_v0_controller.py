"""Score-blind initial scheduling for the hybrid top-three plus Dynamic-v0 controller.

The module consumes already exact-realized, strictly eligible candidate rows and
one cross-fitted target-blind utility checkpoint.  It does not generate, score,
or dock molecules.  Unsupported cells fall back to unchanged Dynamic-v0 at
call one; supported cells charge three frozen macro endpoints before Dynamic-v0
starts at call four.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from compose_v4.control.target_conditioned_utility_selector import (
    UtilityRanker,
    fixed_structural_graph_pair_score,
)

SCHEDULE_SCHEMA = "t4_hybrid_top3_v0_cell_schedule_v1"


@dataclass(frozen=True)
class RankedInitialCandidate:
    candidate_id: str
    canonical_smiles: str
    source_case_id: str
    original_rank: int
    structural_score: float
    target_blind_utility_score: float
    row: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            len(self.candidate_id) != 64
            or not self.canonical_smiles
            or len(self.source_case_id) != 64
            or self.original_rank <= 0
        ):
            raise ValueError("malformed ranked initial candidate")


def _validate_candidate_row(row: Mapping[str, Any], *, cell: str) -> None:
    realization = row.get("realization", {})
    if (
        row.get("cell") != cell
        or row.get("generator_revision") != "baseline"
        or row.get("arm_id") != "baseline_learned"
        or row.get("policy_id") != "balanced_joint_autoregressive"
        or row.get("eligible") is not True
        or row.get("eligibility_exclusion_reasons") != []
        or row.get("exact_action_replay") is not True
        or realization.get("status") != "realized"
        or realization.get("endpoint_matches_bound_target") is not True
        or realization.get("primitive_teacher_actions_used") != 0
    ):
        raise ValueError(f"candidate does not pass the frozen exact eligibility gate: {cell}")


def freeze_cell_schedule(
    *,
    cell: str,
    candidate_rows: Sequence[Mapping[str, Any]],
    ranker: UtilityRanker,
    checkpoint_fold: int,
    report_calls: Sequence[int],
) -> dict[str, Any]:
    """Freeze one cell's support-conditioned initial schedule.

    The structural control is selected first.  Utility candidates are then
    selected by descending target-blind utility score after canonical endpoint
    deduplication and exclusion of the structural endpoint.  No alternative is
    filled after a future score is observed.
    """

    target, source_index_text = cell.rsplit("_", 1)
    source_index = int(source_index_text)
    if checkpoint_fold != source_index or ranker.conditioned:
        raise ValueError("cell requires its matching target-blind held-source checkpoint")
    if tuple(report_calls) != (1, 5, 10, 20, 50, 100):
        raise ValueError("hybrid controller requires the qualification-v2 checkpoints")

    eligible: list[RankedInitialCandidate] = []
    seen_ids: set[str] = set()
    for raw in candidate_rows:
        _validate_candidate_row(raw, cell=cell)
        candidate_id = str(raw["candidate_id"])
        if candidate_id in seen_ids:
            raise ValueError(f"candidate identity repeated in cell pool: {cell}")
        seen_ids.add(candidate_id)
        if int(raw["fold"]) != checkpoint_fold or raw["target"] != target:
            raise ValueError("candidate fold or target differs from its cell")
        source_smiles = str(raw["source_canonical_smiles"])
        endpoint_smiles = str(raw["canonical_smiles"])
        eligible.append(
            RankedInitialCandidate(
                candidate_id=candidate_id,
                canonical_smiles=endpoint_smiles,
                source_case_id=str(raw["source_case_id"]),
                original_rank=int(raw["rank"]),
                structural_score=fixed_structural_graph_pair_score(source_smiles, endpoint_smiles),
                target_blind_utility_score=ranker.score_graph_pair(
                    source_smiles, endpoint_smiles, target=target
                ),
                row=raw,
            )
        )

    by_endpoint: dict[str, RankedInitialCandidate] = {}
    for candidate in sorted(
        eligible,
        key=lambda row: (
            row.original_rank,
            row.candidate_id,
        ),
    ):
        by_endpoint.setdefault(candidate.canonical_smiles, candidate)
    distinct = list(by_endpoint.values())
    base = {
        "schema_version": SCHEDULE_SCHEMA,
        "cell": cell,
        "target": target,
        "source_index": source_index,
        "checkpoint_fold": checkpoint_fold,
        "eligible_candidate_rows": len(candidate_rows),
        "distinct_eligible_endpoints": len(distinct),
        "report_calls": list(report_calls),
        "target_conditioned_arm_used": False,
        "score_aware_fill": False,
        "automatic_retry": False,
        "replacement_or_backfill": False,
        "plateau_stopping": False,
    }
    if len(distinct) < 3:
        return {
            **base,
            "status": "macro_support_abstention_dynamic_v0_from_call_1",
            "abstention_reason": "fewer_than_three_distinct_exact_eligible_baseline_learned_endpoints",
            "initial_macro_calls": [],
            "dynamic_v0_start_call": 1,
            "dynamic_v0_calls_through_100": 100,
        }

    structural = min(
        distinct,
        key=lambda row: (
            -row.structural_score,
            row.original_rank,
            row.candidate_id,
        ),
    )
    remaining = [row for row in distinct if row.canonical_smiles != structural.canonical_smiles]
    utility = sorted(
        remaining,
        key=lambda row: (
            -row.target_blind_utility_score,
            row.original_rank,
            row.candidate_id,
        ),
    )[:2]
    if len(utility) != 2:
        return {
            **base,
            "status": "macro_support_abstention_dynamic_v0_from_call_1",
            "abstention_reason": "utility_order_has_fewer_than_two_endpoints_after_structural_dedup",
            "initial_macro_calls": [],
            "dynamic_v0_start_call": 1,
            "dynamic_v0_calls_through_100": 100,
        }

    chosen = [("fixed_target_blind_structural", structural)] + [
        ("target_blind_utility", row) for row in utility
    ]
    calls = []
    for call_index, (selection_role, candidate) in enumerate(chosen, start=1):
        calls.append(
            {
                "call": call_index,
                "selection_role": selection_role,
                "candidate_id": candidate.candidate_id,
                "canonical_smiles": candidate.canonical_smiles,
                "source_case_id": candidate.source_case_id,
                "original_generator_rank": candidate.original_rank,
                "structural_control_score": candidate.structural_score,
                "target_blind_utility_score": candidate.target_blind_utility_score,
                "candidate": dict(candidate.row),
                "future_admission": "admit_to_one_run_local_archive_only_after_a_finite_charged_score; charged_failure_is_not_replaced",
            }
        )
    if len({row["canonical_smiles"] for row in calls}) != 3:
        raise RuntimeError("frozen initial macro calls are not endpoint-distinct")
    return {
        **base,
        "status": "three_macro_calls_then_dynamic_v0",
        "abstention_reason": None,
        "initial_macro_calls": calls,
        "dynamic_v0_start_call": 4,
        "dynamic_v0_calls_through_100": 97,
    }


__all__ = [
    "SCHEDULE_SCHEMA",
    "RankedInitialCandidate",
    "freeze_cell_schedule",
]
