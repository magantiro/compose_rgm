"""Authentication tests for the Process-V2 T1 train transition stream."""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import pytest

from compose_v4.data.editing_v2_process_v2_active8_map import TRANSITIONS_FILENAME
from compose_v4.data.editing_v2_process_v2_gate_zero import GateZeroSourceIndex
from compose_v4.data.editing_v2_process_v2_pipeline_schema import (
    ACCEPTED_TRANSITION_FIELDS,
    ACTIVE8_TASKS_DIRNAME,
    CANDIDATE_EVIDENCE_FIELDS,
)
from compose_v4.data.editing_v2_process_v2_schema import canonical_bytes, canonical_sha256
from compose_v4.experiments.editing_v2_process_v2_t1_panel import ProcessV2T1PanelError
from compose_v4.experiments.editing_v2_process_v2_t1_source import (
    iter_authenticated_train_transitions,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _transition(task: str, *, entry: int) -> dict[str, object]:
    evidence = {
        "supported": True,
        "exclusion_reason": None,
        "action_sha256": _sha(f"action:{entry}"),
        "source_state_sha256": _sha(f"source:{entry}"),
        "target_state_sha256": _sha(f"target:{entry}"),
        "source_canonical_key": f"source-{entry}",
        "canonical_successor_key": f"target-{entry}",
        "raw_mark_count": 3,
        "matching_mark_count": 1,
        "exact_successor_mark_count": 1,
    }
    assert set(evidence) == set(CANDIDATE_EVIDENCE_FIELDS)
    body = {
        "v1_task_identity_sha256": _sha(f"v1:{task}"),
        "entry_index": entry,
        "trace_id": f"trace-{entry}",
        "step_index": 0,
        "task_identity_sha256": task,
        "data_lane": "observed_local_analogue",
        "partition_role": "train",
        "executor_rule": "atom_insert",
        "progress_index": 0,
        "terminal": False,
        "capability_cell_id": "editing_v2_active8_v1:atom_insert:one_neighbor_birth",
        "model_family": "atom_insert",
        "family_context": "one_neighbor_birth",
        "audit_axes": {"degree": 1},
        "candidate_evidence": evidence,
    }
    row = {**body, "assignment_sha256": canonical_sha256(body)}
    assert set(row) == set(ACCEPTED_TRANSITION_FIELDS)
    return row


def _publish_task(root: Path, label: str, entries: tuple[int, ...]) -> dict[str, object]:
    task = _sha(label)
    rows = [_transition(task, entry=entry) for entry in entries]
    content = b"".join(canonical_bytes(row) + b"\n" for row in rows)
    compressed = gzip.compress(content, mtime=0)
    task_root = root / ACTIVE8_TASKS_DIRNAME / task
    task_root.mkdir(parents=True)
    (task_root / TRANSITIONS_FILENAME).write_bytes(compressed)
    return {
        "task_identity_sha256": task,
        "partition_role": "train",
        "data_lane": "observed_local_analogue",
        "decision_shard_sha256": hashlib.sha256(compressed).hexdigest(),
        "transition_count": len(rows),
        "decision_eligible_role": True,
    }


def _index(root: Path, shards: list[dict[str, object]]) -> GateZeroSourceIndex:
    identities = tuple(str(shard["task_identity_sha256"]) for shard in shards)
    return GateZeroSourceIndex(
        active8_run_root=str(root),
        active8_completion_sha256=_sha("completion"),
        active8_sentinel_sha256=_sha("sentinel"),
        contracts_binding_sha256=_sha("contracts"),
        shards=tuple(shards),
        eligible_task_identities=identities,
        role_census={},
        sealed_role_metadata={},
        index_sha256=_sha("index"),
    )


def test_authenticated_stream_uses_deterministic_task_order(tmp_path: Path) -> None:
    later = _publish_task(tmp_path, "z", (2,))
    earlier = _publish_task(tmp_path, "a", (1, 3))
    index = _index(tmp_path, [later, earlier])

    rows = list(iter_authenticated_train_transitions(tmp_path, source_index=index))

    expected = sorted(
        (
            (str(earlier["task_identity_sha256"]), 2),
            (str(later["task_identity_sha256"]), 1),
        )
    )
    assert [row["task_identity_sha256"] for row in rows] == [
        identity for identity, count in expected for _ in range(count)
    ]


def test_authenticated_stream_refuses_changed_shard_bytes(tmp_path: Path) -> None:
    shard = _publish_task(tmp_path, "task", (1,))
    index = _index(tmp_path, [shard])
    shard["decision_shard_sha256"] = _sha("wrong")

    with pytest.raises(ProcessV2T1PanelError, match="bytes disagree"):
        list(iter_authenticated_train_transitions(tmp_path, source_index=index))


def test_authenticated_stream_refuses_a_sealed_role(tmp_path: Path) -> None:
    shard = _publish_task(tmp_path, "task", (1,))
    shard["partition_role"] = "validation"
    index = _index(tmp_path, [shard])

    with pytest.raises(ProcessV2T1PanelError, match="non-train or ineligible"):
        list(iter_authenticated_train_transitions(tmp_path, source_index=index))
