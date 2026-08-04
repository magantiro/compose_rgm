"""Bounded Process-V2 T1 selection from authenticated Active8 outputs.

This module owns only the seam between the completed structural stages and the
T1 panel selector.  It opens the train decision shards named by the already
authenticated Gate-0 source index, checks their declared bytes and row counts,
and delegates the scientific selection law to
``editing_v2_process_v2_t1_panel``.  It never opens molecular cache chunks or
enumerates a successor fiber.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from compose_v4.data.editing_v2_process_v2_active8_map import read_task_transitions
from compose_v4.data.editing_v2_process_v2_gate_zero import GateZeroSourceIndex
from compose_v4.data.editing_v2_process_v2_pipeline_schema import (
    ACTIVE8_DECISION_SHARD_FILENAME,
    ACTIVE8_TASKS_DIRNAME,
)
from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
    ProcessV2T1CandidateSelection,
    ProcessV2T1PanelError,
    ProcessV2T1TargetCensus,
    candidates_from_sentinel_plan,
    census_candidate_targets,
    select_process_v2_t1_candidates,
)


def iter_authenticated_train_transitions(
    active8_run_root: Path,
    *,
    source_index: GateZeroSourceIndex,
) -> Iterator[dict[str, Any]]:
    """Yield the exact eligible train stream in deterministic task order."""

    root = Path(active8_run_root)
    if str(root) != source_index.active8_run_root:
        raise ProcessV2T1PanelError(
            "the Gate-0 source index addresses another Active8 run root"
        )
    shards = sorted(
        source_index.eligible(), key=lambda shard: str(shard["task_identity_sha256"])
    )
    if not shards:
        raise ProcessV2T1PanelError("the Gate-0 source index has no eligible train shard")
    for shard in shards:
        identity = str(shard["task_identity_sha256"])
        role = shard.get("partition_role")
        if role != "train" or shard.get("decision_eligible_role") is not True:
            raise ProcessV2T1PanelError(
                f"T1 refuses non-train or ineligible Active8 shard {identity}"
            )
        task_root = root / ACTIVE8_TASKS_DIRNAME / identity
        shard_path = task_root / ACTIVE8_DECISION_SHARD_FILENAME
        try:
            compressed = shard_path.read_bytes()
            rows = read_task_transitions(task_root)
        except (OSError, ValueError, RuntimeError) as error:
            raise ProcessV2T1PanelError(
                f"T1 cannot authenticate Active8 decision shard {shard_path}"
            ) from error
        if hashlib.sha256(compressed).hexdigest() != shard.get("decision_shard_sha256"):
            raise ProcessV2T1PanelError(
                f"T1 decision shard bytes disagree with receipt {identity}"
            )
        if len(rows) != shard.get("transition_count"):
            raise ProcessV2T1PanelError(
                f"T1 decision shard length disagrees with receipt {identity}"
            )
        for row in rows:
            if (
                row.get("task_identity_sha256") != identity
                or row.get("partition_role") != role
                or row.get("data_lane") != shard.get("data_lane")
            ):
                raise ProcessV2T1PanelError(
                    f"T1 transition belongs to another task, role, or lane: {identity}"
                )
            yield row


def select_authenticated_process_v2_t1_panel(
    *,
    active8_run_root: Path,
    source_index: GateZeroSourceIndex,
    sentinel_plan: Mapping[str, Any],
    panel_policy: Mapping[str, Any],
    required_cell_ids: Sequence[str],
) -> tuple[ProcessV2T1TargetCensus, ProcessV2T1CandidateSelection]:
    """Apply the frozen bounded selection law to authenticated train rows."""

    if panel_policy.get("panel_kind") != (
        "unique_state_single_target_canonical_successor_capacity"
    ):
        raise ProcessV2T1PanelError("the Process-V2 T1 panel kind disagrees")
    if panel_policy.get("empirical_multiplicity_receipts_included") is not False:
        raise ProcessV2T1PanelError(
            "the unique-state T1 selector refuses empirical multiplicity"
        )
    families = panel_policy.get("active_families")
    minimums = panel_policy.get("minimum_entries_by_family")
    maximums = panel_policy.get("maximum_entries_by_family")
    if (
        not isinstance(families, list)
        or not isinstance(minimums, Mapping)
        or not isinstance(maximums, Mapping)
    ):
        raise ProcessV2T1PanelError("the Process-V2 T1 panel policy is incomplete")

    candidates = candidates_from_sentinel_plan(
        sentinel_plan,
        required_cell_ids=required_cell_ids,
        active_families=families,
    )
    census = census_candidate_targets(
        iter_authenticated_train_transitions(
            active8_run_root,
            source_index=source_index,
        ),
        candidate_source_state_sha256s=(
            candidate.source_state_sha256 for candidate in candidates
        ),
    )
    selection = select_process_v2_t1_candidates(
        candidates,
        target_census=census,
        required_cell_ids=required_cell_ids,
        minimum_entries_by_family=minimums,
        maximum_entries_by_family=maximums,
    )
    return census, selection


__all__ = [
    "iter_authenticated_train_transitions",
    "select_authenticated_process_v2_t1_panel",
]
