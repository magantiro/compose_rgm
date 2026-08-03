"""The join between the Process-V2 overlay and the V1 lane/role inventory.

The two halves each have their own authority and their own tests. What is new,
and therefore what is tested here, is the JOIN: that the overlay and the
inventory describe the same payload, that the per-task census reconciles, and
that the two process identities stay distinguishable.

The failure this guards is specific and silent. An admitted overlay proved
against one migration, paired with a different migration's packed shards, yields
a corpus whose chemistry and whose admission decisions came from different runs.
Every count would reconcile run-wide, because the reduction only ever sees its
own overlay; only a cross-check against the shard identities catches it.

The two resolvers are substituted here rather than driven end to end. That is
deliberate: driving them needs a full V1 migration payload, which
`tests/test_process_v2_rebind_end_to_end.py` covers, and it would test the
resolvers rather than the join. Substituting them isolates the code this module
actually adds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from compose_v4.data import editing_v2_process_v2_active8_source as joiner
from compose_v4.data.editing_v2_process_v2_active8_source import (
    SOURCE_SCHEMA,
    SOURCE_STATUS,
    ProcessV2Active8SourceError,
    resolve_process_v2_active8_source_inventory,
)

# The SUPERSEDED V1 payload identity. Named for its lineage: the chain
# verifier refuses a historical value carried under a live-sounding name,
# and it is right to -- that is how a dead identity gets read as current.
_SUPERSEDED_V1_IDENTITY = "6b98ee21ef8b853deda9fa56a2963178208ecc893a397fb4aa412629fc2414d7"
_V2_IDENTITY = "0c938177a34819e6e828920c1f66e240c6eb251fe7c9ea6cfe6757829dceb2dd"

_TASKS = (
    {"task": "a" * 64, "shard": "1" * 64, "lane": "observed_local_analogue", "split": "train"},
    {"task": "b" * 64, "shard": "2" * 64, "lane": "reversible_synthetic_walk", "split": "train"},
)


# ---- Stand-ins for the two authorities ----


class _Source:
    def __init__(self, task: str, shard: str, lane: str, split: str) -> None:
        self.task_identity_sha256 = task
        self.semantic_shard_sha256 = shard
        self.semantic_manifest_sha256 = "3" * 64
        self.data_lane = lane
        self.partition_role = split


class _V1Inventory:
    def __init__(self, sources) -> None:
        self.sources = tuple(sources)
        self.migration_completion_sha256 = "4" * 64


class _Admitted:
    def __init__(self, *, results, counts, plan) -> None:
        self.completion = {"result_inventory": results, "completion_sha256": "5" * 64}
        self.counts = counts
        self.rejected_traces_by_code = {"atom_delete_outside_process_v2_mask": 1}
        self.plan = plan

    def identity(self) -> dict[str, Any]:
        return {"admitted_source_sha256": "6" * 64}


def _plan(tasks=_TASKS) -> dict[str, Any]:
    return {
        "v1_payload_binding": {
            "v1_tasks": [
                {
                    "v1_task_identity_sha256": task["task"],
                    "v1_semantic_shard_sha256": task["shard"],
                    "data_lane": task["lane"],
                    "split": task["split"],
                    "v1_entries": 4,
                }
                for task in tasks
            ]
        },
        "pinned_process_identity": {"process_identity_sha256": _SUPERSEDED_V1_IDENTITY},
        "process_v2_identity": {"process_identity_sha256": _V2_IDENTITY},
    }


def _install(monkeypatch, *, admitted, inventory) -> None:
    monkeypatch.setattr(
        joiner, "resolve_process_v2_admitted_source", lambda *a, **k: admitted
    )
    monkeypatch.setattr(
        joiner, "resolve_editing_v2_semantic_active8_sources", lambda *a, **k: inventory
    )


def _resolve(plan):
    return resolve_process_v2_active8_source_inventory(
        plan,
        v1_migration_completion_path=Path("/artifacts/x/COMPLETE.json"),
        artifact_root=Path("/artifacts"),
        repo_root=Path("/repo"),
    )


def _admitted_for(tasks=_TASKS, *, admitted_each=3, rejected_each=1, plan=None):
    results = [
        {
            "v1_task_identity_sha256": task["task"],
            "counts": {"admitted_entries": admitted_each, "rejected_entries": rejected_each},
        }
        for task in tasks
    ]
    total = len(tasks)
    counts = {
        "source_entries": total * (admitted_each + rejected_each),
        "admitted_entries": total * admitted_each,
        "rejected_entries": total * rejected_each,
        "admitted_states": total * admitted_each * 2,
        "admitted_transitions": total * admitted_each,
    }
    return _Admitted(results=results, counts=counts, plan=plan or _plan(tasks))


def _inventory_for(tasks=_TASKS):
    return _V1Inventory(
        _Source(task["task"], task["shard"], task["lane"], task["split"]) for task in tasks
    )


# ---- The join succeeds and keeps both identities distinguishable ----


def test_the_join_reconciles_and_keeps_both_identities_separate(monkeypatch) -> None:
    _install(monkeypatch, admitted=_admitted_for(), inventory=_inventory_for())
    resolved = _resolve(_plan())

    assert resolved.v1_payload_process_identity_sha256 == _SUPERSEDED_V1_IDENTITY
    assert resolved.process_v2_identity_sha256 == _V2_IDENTITY
    # The two must never be collapsed into one field: that is how a V1 artifact
    # comes to be read as a V2 one.
    assert resolved.v1_payload_process_identity_sha256 != resolved.process_v2_identity_sha256

    assert len(resolved.sources) == len(_TASKS)
    for source in resolved.sources:
        assert source.admitted_entry_count + source.rejected_entry_count == (
            source.v1_entry_count
        )

    identity = resolved.identity()
    assert identity["schema"] == SOURCE_SCHEMA
    assert identity["status"] == SOURCE_STATUS
    assert identity["v1_payload_process_identity_sha256"] == _SUPERSEDED_V1_IDENTITY
    assert identity["process_v2_identity_sha256"] == _V2_IDENTITY
    assert len(identity["process_v2_active8_source_sha256"]) == 64


def test_the_source_identity_grants_nothing(monkeypatch) -> None:
    _install(monkeypatch, admitted=_admitted_for(), inventory=_inventory_for())
    identity = _resolve(_plan()).identity()
    for flag in (
        "training_authorized",
        "gate_zero_authorized",
        "t1_authorized",
        "bounded_p50_authorized",
    ):
        assert identity[flag] is False, flag


# ---- The cross-checks this module exists for ----


def test_a_different_task_set_is_refused(monkeypatch) -> None:
    """An overlay proved against one migration, paired with another's shards."""

    other = (
        {"task": "c" * 64, "shard": "1" * 64, "lane": "observed_local_analogue", "split": "train"},
        _TASKS[1],
    )
    _install(monkeypatch, admitted=_admitted_for(), inventory=_inventory_for(other))
    with pytest.raises(ProcessV2Active8SourceError, match="different\n?\\s*task sets|different task sets"):
        _resolve(_plan())


def test_a_different_semantic_shard_for_the_same_task_is_refused(monkeypatch) -> None:
    """Same task identity, different chemistry: the silent case."""

    drifted = (
        {**_TASKS[0], "shard": "9" * 64},
        _TASKS[1],
    )
    _install(monkeypatch, admitted=_admitted_for(), inventory=_inventory_for(drifted))
    with pytest.raises(ProcessV2Active8SourceError, match="different semantic shard"):
        _resolve(_plan())


def test_a_lane_or_split_disagreement_is_refused(monkeypatch) -> None:
    relabelled = (
        {**_TASKS[0], "split": "validation"},
        _TASKS[1],
    )
    _install(monkeypatch, admitted=_admitted_for(), inventory=_inventory_for(relabelled))
    with pytest.raises(ProcessV2Active8SourceError, match="lane or split disagrees"):
        _resolve(_plan())


def test_a_per_task_census_that_does_not_reconcile_is_refused(monkeypatch) -> None:
    """Run-wide totals can reconcile while a single task does not."""

    admitted = _admitted_for(admitted_each=2, rejected_each=1)  # 3 != v1_entries 4
    _install(monkeypatch, admitted=admitted, inventory=_inventory_for())
    with pytest.raises(ProcessV2Active8SourceError, match="admits 2 and rejects 1 of 4"):
        _resolve(_plan())


def test_the_joined_total_must_account_for_every_source_entry(monkeypatch) -> None:
    """The per-task sum and the published run census must agree."""

    admitted = _admitted_for()
    admitted.counts = {**admitted.counts, "source_entries": 99}
    _install(monkeypatch, admitted=admitted, inventory=_inventory_for())
    with pytest.raises(ProcessV2Active8SourceError, match="does not account for every source entry"):
        _resolve(_plan())
