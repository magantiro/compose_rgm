"""Focused invariants for the bounded Active8 reduction derivation."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import test_editing_v2_process_v2_active8_pipeline as pipeline_fixture
from compose_v4.data import editing_v2_process_v2_active8_reduce as reduce_module
from compose_v4.data import editing_v2_process_v2_active8_sentinel as sentinel_module
from compose_v4.data.editing_v2_process_v2_active8_reduce import (
    COMPLETION_FILENAME,
    PREPARATION_FILENAME,
    ProcessV2Active8Incomplete,
    ProcessV2Active8ReduceError,
    _create_reduction_index,
    _index_source_row,
    finalize_process_v2_active8_reduction,
    prepare_process_v2_active8_reduction,
)
from compose_v4.data.editing_v2_process_v2_active8_sentinel import (
    create_sentinel_pair_index,
    index_sentinel_transition,
    prepare_release_sentinel_plan,
    prepare_release_sentinel_plan_from_selection,
    select_sentinel_pairs,
    select_sentinel_pairs_from_index,
    sentinel_rank,
)
from compose_v4.data.editing_v2_process_v2_pipeline_schema import (
    SENTINEL_EXHAUSTIVE_THRESHOLD,
)
from compose_v4.data.editing_v2_process_v2_schema import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


def _transition(
    index: int,
    *,
    task_identity: str,
    required_cells: list[str],
    pair_index: int | None = None,
) -> dict[str, object]:
    pair_index = index if pair_index is None else pair_index
    return {
        "task_identity_sha256": task_identity,
        "v1_task_identity_sha256": canonical_sha256(["legacy", index]),
        "entry_index": index,
        "step_index": 0,
        "candidate_evidence": {
            "source_state_sha256": canonical_sha256(["source", pair_index]),
            "action_sha256": canonical_sha256(["action", pair_index]),
        },
        "capability_cell_id": required_cells[pair_index % len(required_cells)],
        "family_context": "bounded-reduction-probe",
        "audit_axes": {"probe": pair_index % 3},
        "model_family": "atom_delete",
    }


def _pair(transition: dict[str, object]) -> tuple[str, str]:
    evidence = transition["candidate_evidence"]
    assert isinstance(evidence, dict)
    return str(evidence["source_state_sha256"]), str(evidence["action_sha256"])


def test_disk_selection_and_two_pass_plan_are_exactly_eager_equivalent(
    tmp_path: Path,
) -> None:
    required_cells = ["cell_a", "cell_b", "cell_c"]
    task_identity = "a" * 64
    unique_count = SENTINEL_EXHAUSTIVE_THRESHOLD + 1
    transitions = [
        _transition(
            index,
            task_identity=task_identity,
            required_cells=required_cells,
        )
        for index in range(unique_count)
    ]
    ranked = sorted(
        (_pair(item) for item in transitions),
        key=lambda pair: (
            sentinel_rank(*pair),
            pair[0],
            pair[1],
        ),
    )
    duplicated = set(ranked[:5])
    for pair_index, transition in enumerate(list(transitions)):
        if _pair(transition) in duplicated:
            transitions.append(
                _transition(
                    unique_count + pair_index,
                    task_identity=task_identity,
                    required_cells=required_cells,
                    pair_index=pair_index,
                )
            )

    eager = select_sentinel_pairs(transitions, required_cell_ids=required_cells)
    database = tmp_path / "selection.sqlite3"
    with sqlite3.connect(database) as connection:
        create_sentinel_pair_index(connection)
        for transition in reversed(transitions):
            index_sentinel_transition(connection, transition)
        disk = select_sentinel_pairs_from_index(connection, required_cell_ids=required_cells)

    for field in (
        "selection_mode",
        "unique_accepted_pairs",
        "per_cell_examples",
        "global_examples",
        "selected",
    ):
        assert disk[field] == eager[field]

    plan = {
        "binding": {"required_cell_ids": required_cells},
        "binding_sha256": "1" * 64,
        "plan_sha256": "2" * 64,
        "run_identity_sha256": "3" * 64,
        "task_inventory_sha256": "4" * 64,
        "tasks": [{"task_identity_sha256": task_identity}],
    }
    result_inventory_sha256 = canonical_sha256([[task_identity, "f" * 64]])
    eager_plan = prepare_release_sentinel_plan(
        plan,
        transitions,
        result_inventory_sha256=result_inventory_sha256,
        max_pairs_per_partition=73,
    )
    selected = set(disk["selected"])
    disk_plan = prepare_release_sentinel_plan_from_selection(
        plan,
        disk,
        (item for item in reversed(transitions) if _pair(item) in selected),
        result_inventory_sha256=result_inventory_sha256,
        max_pairs_per_partition=73,
    )
    assert disk_plan == eager_plan
    planned_occurrences = sum(
        len(pair["occurrences"])
        for partition in disk_plan["partitions"]
        for pair in partition["pairs"]
    )
    assert planned_occurrences == sum(_pair(item) in selected for item in transitions)


def test_eager_and_disk_ranking_use_the_same_total_tie_break(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sentinel_module, "sentinel_rank", lambda *_: "f" * 64)
    required_cells = ["cell"]
    transitions = [
        _transition(index, task_identity="a" * 64, required_cells=required_cells)
        for index in reversed(range(8))
    ]
    eager = select_sentinel_pairs(transitions, required_cell_ids=required_cells)
    with sqlite3.connect(tmp_path / "ties.sqlite3") as connection:
        create_sentinel_pair_index(connection)
        for transition in transitions:
            index_sentinel_transition(connection, transition)
        disk = select_sentinel_pairs_from_index(connection, required_cell_ids=required_cells)
    expected = sorted(_pair(item) for item in transitions)
    assert eager["selected"] == disk["selected"] == expected


def test_disk_index_refuses_a_duplicate_source_row(tmp_path: Path) -> None:
    with sqlite3.connect(tmp_path / "rows.sqlite3") as connection:
        _create_reduction_index(connection)
        row = {"v1_task_identity_sha256": "a" * 64, "entry_index": 7}
        _index_source_row(connection, row)
        with pytest.raises(ProcessV2Active8ReduceError, match="same source entry"):
            _index_source_row(connection, dict(row))


@pytest.fixture(scope="module")
def bounded_stage(tmp_path_factory: pytest.TempPathFactory):
    stage = pipeline_fixture._Stage(
        tmp_path_factory.mktemp("active8-bounded") / "artifacts",
        prefix="/artifacts/active8_bounded_reduction",
    )
    stage.run()
    return stage


def _run_root(stage) -> Path:
    return stage.artifact_root / str(stage.plan["run_artifact_root"]).removeprefix("/artifacts/")


def test_interrupted_derivation_publishes_no_preparation(
    bounded_stage, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(reduce_module, "index_sentinel_transition", interrupt)
    with pytest.raises(KeyboardInterrupt):
        prepare_process_v2_active8_reduction(
            bounded_stage.plan,
            runtime=bounded_stage.runtime,
            artifact_root=bounded_stage.artifact_root,
            repo_root=ROOT,
            max_sentinel_pairs_per_partition=3,
            publish=True,
        )
    assert not (_run_root(bounded_stage) / PREPARATION_FILENAME).exists()
    assert not (_run_root(bounded_stage) / COMPLETION_FILENAME).exists()


def test_finalization_uses_a_fresh_shared_bounded_derivation(
    bounded_stage, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = prepare_process_v2_active8_reduction(
        bounded_stage.plan,
        runtime=bounded_stage.runtime,
        artifact_root=bounded_stage.artifact_root,
        repo_root=ROOT,
        max_sentinel_pairs_per_partition=3,
        publish=False,
    )
    original = reduce_module._derive_reduction_inputs

    def changed(*args, **kwargs):
        derived = original(*args, **kwargs)
        derived["accepted_transitions"] += 1
        return derived

    monkeypatch.setattr(reduce_module, "_derive_reduction_inputs", changed)
    with pytest.raises(ProcessV2Active8Incomplete, match="accepted_transitions changed"):
        finalize_process_v2_active8_reduction(
            bounded_stage.plan,
            prepared,
            [],
            artifact_root=bounded_stage.artifact_root,
            repo_root=ROOT,
            publish=False,
        )
    assert not (_run_root(bounded_stage) / COMPLETION_FILENAME).exists()
