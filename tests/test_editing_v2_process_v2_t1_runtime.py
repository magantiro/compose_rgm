"""Exact Process-V2 successor preparation for the bounded T1 panel."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from compose_v4.data.editing_v2_process_v2_schema import canonical_sha256
from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
    build_process_v2_t1_panel,
    iter_process_v2_t1_candidates,
    resolve_process_v2_t1_entries,
)
from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
    ProcessV2T1RuntimeError,
    _entry_from_exact_transition,
    authenticate_process_v2_t1_prepared_plan_for_worker,
    build_process_v2_t1_prepared_inputs,
    build_process_v2_t1_prepared_plan,
    build_process_v2_t1_scratch_runtime,
    validate_process_v2_t1_prepared_leaf,
    write_process_v2_t1_prepared_leaf,
)

from test_editing_v2_process_v2_t1_panel import _pass_source, _stage_evidence

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def genuine_stage(tmp_path_factory):
    return _stage_evidence(tmp_path_factory.mktemp("process-v2-t1-runtime"))


def _fixture_cardinality_source(source):
    required_by_family: dict[str, int] = {}
    for cell in source.contracts.required_cell_ids:
        _namespace, family, _context = cell.split(":", 2)
        required_by_family[family] = required_by_family.get(family, 0) + 1
    policy = dict(source.policy)
    policy["minimum_entries_by_family"] = required_by_family
    policy["maximum_entries_by_family"] = required_by_family
    policy["contract_sha256"] = canonical_sha256(
        {key: value for key, value in policy.items() if key != "contract_sha256"}
    )
    return replace(
        source,
        policy=policy,
        policy_file_sha256=canonical_sha256("runtime-fixture-policy"),
    )


def _standin_planned_panel(genuine_stage, tmp_path: Path):
    _stage, genuine_source, _decision = genuine_stage
    source = _pass_source(tmp_path)
    task_ids = sorted(
        {str(item["task_identity_sha256"]) for item in iter_process_v2_t1_candidates(source)}
    )
    active8_plan = {
        **source.plan,
        "binding": genuine_source.plan["binding"],
        "tasks": [
            {
                "task_identity_sha256": identity,
                "chunk_file_sha256": canonical_sha256(f"chunk:{identity}"),
            }
            for identity in task_ids
        ],
    }
    source = _fixture_cardinality_source(replace(source, plan=active8_plan))
    panel = build_process_v2_t1_panel(source, scratch_dir=tmp_path)
    revision = {
        "commit": "a" * 40,
        "source_revision_sha256": canonical_sha256({"commit": "a" * 40}),
    }
    plan = build_process_v2_t1_prepared_plan(
        panel, source=source, source_revision=revision
    )
    return source, panel, revision, plan


def test_one_genuine_cached_transition_compiles_the_complete_successor_partition(
    genuine_stage,
) -> None:
    _stage, source, _decision = genuine_stage
    candidate = next(iter(iter_process_v2_t1_candidates(source)))
    addressed = resolve_process_v2_t1_entries(source, (candidate,))[0].addressed_trace
    scratch, binding = build_process_v2_t1_scratch_runtime(source)

    entry = _entry_from_exact_transition(
        {**candidate, "panel_entry_sha256": candidate["assignment_sha256"]},
        addressed=addressed,
        scratch_runtime=scratch,
        support_time=0.5,
    )

    assert binding["model_runtime"]["initial_model_state_sha256"] == (
        source.plan["binding"]["model_runtime"]["initial_model_state_sha256"]
    )
    assert entry["raw_mark_count"] == (
        entry["productive_alias_count"] + entry["virtual_alias_count"]
    )
    assert entry["canonical_successor_count"] > 0
    assert entry["production_successor_alias_multiplicity"] > 0
    assert entry["model_scores_or_probabilities_stored"] is False
    assert entry["hazard_included"] is False


def test_prepared_plan_groups_panel_entries_by_existing_active8_chunk(
    genuine_stage, tmp_path: Path
) -> None:
    source, panel, revision, first = _standin_planned_panel(genuine_stage, tmp_path)
    second = build_process_v2_t1_prepared_plan(
        panel, source=source, source_revision=revision
    )
    authenticated = authenticate_process_v2_t1_prepared_plan_for_worker(
        first, panel=panel, source=source
    )

    assert second == first
    assert authenticated == first
    assert first["task_count"] == len(first["tasks"])
    assert first["selected_entry_count"] == len(panel["entries"])
    assert sorted(
        identifier
        for task in first["tasks"]
        for identifier in task["panel_entry_sha256s"]
    ) == sorted(entry["panel_entry_sha256"] for entry in panel["entries"])


def test_reduction_refuses_an_incomplete_leaf_inventory(
    genuine_stage, tmp_path: Path
) -> None:
    source, panel, _revision, plan = _standin_planned_panel(genuine_stage, tmp_path)

    with pytest.raises(ProcessV2T1RuntimeError, match="prepared leaf is absent"):
        build_process_v2_t1_prepared_inputs(
            plan,
            panel=panel,
            source=source,
            run_root=tmp_path / "empty-run",
        )


def test_worker_authentication_refuses_changed_upstream_provenance(
    genuine_stage, tmp_path: Path
) -> None:
    source, panel, _revision, plan = _standin_planned_panel(genuine_stage, tmp_path)
    changed_decision = {**source.decision, "decision_sha256": "0" * 64}
    changed_source = replace(source, decision=changed_decision)

    with pytest.raises(ProcessV2T1RuntimeError, match="source binding disagrees"):
        authenticate_process_v2_t1_prepared_plan_for_worker(
            plan, panel=panel, source=changed_source
        )


def test_a_resealed_wrong_teacher_key_is_refused(
    genuine_stage, tmp_path: Path
) -> None:
    _stage, source, _decision = genuine_stage
    candidate = next(iter(iter_process_v2_t1_candidates(source)))
    addressed = resolve_process_v2_t1_entries(source, (candidate,))[0].addressed_trace
    scratch, binding = build_process_v2_t1_scratch_runtime(source)
    entry = _entry_from_exact_transition(
        {**candidate, "panel_entry_sha256": candidate["assignment_sha256"]},
        addressed=addressed,
        scratch_runtime=scratch,
        support_time=0.5,
    )
    task = {
        "task_identity_sha256": "b" * 64,
        "active8_task_identity_sha256": candidate["task_identity_sha256"],
        "panel_entry_sha256s": [candidate["assignment_sha256"]],
    }
    plan = {
        "plan_sha256": "c" * 64,
        "run_identity_sha256": "d" * 64,
        "build_identity_sha256": "e" * 64,
        "panel_sha256": "f" * 64,
        "model_binding": binding,
        "tasks": [task],
    }
    leaf_body = {
        "schema": "compose.editing_v2.process_v2_t1_prepared_leaf",
        "schema_version": 1,
        "status": "PROCESS_V2_T1_PREPARED_LEAF_NO_DOWNSTREAM_AUTHORITY",
        "training_authorized": False,
        "gate_zero_authorized": False,
        "t1_authorized": False,
        "bounded_p50_authorized": False,
        "long_training_authorized": False,
        "checkpoint_selection_authorized": False,
        "final_test_selection_authorized": False,
        "plan_sha256": plan["plan_sha256"],
        "run_identity_sha256": plan["run_identity_sha256"],
        "build_identity_sha256": plan["build_identity_sha256"],
        "task_identity_sha256": task["task_identity_sha256"],
        "active8_task_identity_sha256": task["active8_task_identity_sha256"],
        "panel_sha256": plan["panel_sha256"],
        "initial_model_state_sha256": binding["model_runtime"][
            "initial_model_state_sha256"
        ],
        "entry_count": 1,
        "entry_inventory_sha256": canonical_sha256([entry]),
        "entries": [entry],
        "model_scores_or_probabilities_stored": False,
        "hazard_included": False,
    }
    leaf = {**leaf_body, "leaf_sha256": canonical_sha256(leaf_body)}

    first_path = write_process_v2_t1_prepared_leaf(
        leaf, plan=plan, run_root=tmp_path
    )
    first_bytes = first_path.read_bytes()
    second_path = write_process_v2_t1_prepared_leaf(
        leaf, plan=plan, run_root=tmp_path
    )
    assert second_path == first_path
    assert second_path.read_bytes() == first_bytes

    changed_body = {
        **{key: value for key, value in entry.items() if key != "entry_sha256"},
        "successor_canonical_key": "NOT-A-REAL-SUCCESSOR",
    }
    changed = {**changed_body, "entry_sha256": canonical_sha256(changed_body)}
    invalid_leaf_body = {
        **{key: value for key, value in leaf.items() if key != "leaf_sha256"},
        "entry_inventory_sha256": canonical_sha256([changed]),
        "entries": [changed],
    }
    invalid_leaf = {
        **invalid_leaf_body,
        "leaf_sha256": canonical_sha256(invalid_leaf_body),
    }

    with pytest.raises(
        ProcessV2T1RuntimeError, match="state or successor census disagrees"
    ):
        validate_process_v2_t1_prepared_leaf(invalid_leaf, plan=plan)
