"""Exact Process-V2 successor preparation for the bounded T1 panel."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from compose_v4.data.editing_v2_process_v2_schema import canonical_bytes, canonical_sha256
from compose_v4.experiments import editing_v2_process_v2_t1_runtime as t1_runtime
from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
    build_process_v2_t1_panel,
    iter_process_v2_t1_candidates,
    resolve_process_v2_t1_entries,
)
from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
    ProcessV2T1RuntimeError,
    _build_prepared_completion,
    _entry_from_exact_transition,
    _load_prepared_completion,
    _validate_source_revision,
    authenticate_process_v2_t1_prepared_plan_for_worker,
    build_process_v2_t1_prepared_inputs,
    build_process_v2_t1_prepared_plan,
    build_process_v2_t1_scratch_runtime,
    publish_process_v2_t1_prepared_inputs,
    validate_process_v2_t1_prepared_inputs,
    validate_process_v2_t1_prepared_leaf,
    write_process_v2_t1_prepared_leaf,
)

from test_editing_v2_process_v2_t1_panel import _pass_source, _stage_evidence

ROOT = Path(__file__).resolve().parents[1]


def _test_source_revision() -> dict[str, object]:
    body = {
        "schema": "compose.editing_v2.process_v2_t1_authenticated_image_revision",
        "schema_version": 1,
        "commit": "a" * 40,
        "tree": "b" * 40,
        "image_revision_sha256": "c" * 64,
        "serialized_source_inventory_sha256": "d" * 64,
    }
    return {**body, "source_revision_sha256": canonical_sha256(body)}


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
    revision = _test_source_revision()
    plan = build_process_v2_t1_prepared_plan(panel, source=source, source_revision=revision)
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

    assert (
        binding["model_runtime"]["initial_model_state_sha256"]
        == (source.plan["binding"]["model_runtime"]["initial_model_state_sha256"])
    )
    assert entry["raw_mark_count"] == (
        entry["productive_alias_count"] + entry["virtual_alias_count"]
    )
    assert entry["canonical_successor_count"] > 0
    assert entry["production_successor_alias_multiplicity"] > 0
    assert entry["model_scores_or_probabilities_stored"] is False
    assert entry["hazard_included"] is False


def test_score_only_model_revision_reuses_active8_support_but_not_its_state(
    genuine_stage,
) -> None:
    _stage, source, _decision = genuine_stage
    predecessor = dict(source.plan["binding"]["model_runtime"])
    predecessor["model_identity_sha256"] = "1" * 64
    predecessor["initial_model_state_sha256"] = "2" * 64
    predecessor["semantic_model_process_contract_sha256"] = "3" * 64
    revised = replace(
        source,
        plan={
            **source.plan,
            "binding": {**source.plan["binding"], "model_runtime": predecessor},
        },
    )

    runtime, binding = build_process_v2_t1_scratch_runtime(revised)

    assert binding["model_runtime"]["model_identity_sha256"] != "1" * 64
    assert binding["model_runtime"]["initial_model_state_sha256"] == (
        runtime.initial_model_state_sha256
    )
    assert binding["model_runtime"]["semantic_model_process_contract_sha256"] != "3" * 64


def test_score_revision_bridge_adds_only_contextual_ring_restate_capacity(
    genuine_stage,
    monkeypatch,
) -> None:
    stage, source, _decision = genuine_stage
    predecessor = dict(source.plan["binding"]["model_runtime"])
    predecessor_state = stage.runtime.model.state_dict()
    predecessor["initial_model_state_sha256"] = t1_runtime.state_dict_semantic_sha256(
        {
            name: value
            for name, value in predecessor_state.items()
            if name != "graft_relation_head.weight"
        }
    )
    frozen = {
        "prepared_completion_sha256": "a" * 64,
        "prepared_completion_file_sha256": "b" * 64,
        "prepared_artifact_sha256": "c" * 64,
        "prepared_file_sha256": "d" * 64,
        "prepared_implementation_sha256": "e" * 64,
        "initial_model_state_sha256": predecessor["initial_model_state_sha256"],
        "model_identity_sha256": predecessor["model_identity_sha256"],
        "semantic_model_process_contract_sha256": predecessor[
            "semantic_model_process_contract_sha256"
        ],
    }
    monkeypatch.setattr(
        t1_runtime,
        "_SCORE_REVISION_PREDECESSOR",
        MappingProxyType(frozen),
    )

    runtime, binding, bridge = t1_runtime._scratch_runtime_for_score_revision(
        predecessor,
        repo_root=ROOT,
    )

    assert isinstance(
        runtime.model,
        t1_runtime.ContextualRingRestateFactorizedTraceletRateModel,
    )
    assert runtime.process_identity_sha256 == source.contracts.process_identity_sha256
    assert binding["model_runtime"]["operator_capability_fingerprint"] == (
        predecessor["operator_capability_fingerprint"]
    )
    assert binding["model_runtime"]["initial_model_state_sha256"] != (
        predecessor["initial_model_state_sha256"]
    )
    assert bridge["score_revision"]["ring_restate_context_scorer_mode"] == (
        t1_runtime.CONTEXTUAL_RING_RESTATE_SCORER_MODE
    )
    assert bridge["score_revision"]["added_parameters"] == [
        "ring_restate_context_head.weight"
    ]


def test_active8_support_geometry_change_blocks_t1_model_reconstruction(
    genuine_stage,
) -> None:
    _stage, source, _decision = genuine_stage
    changed = dict(source.plan["binding"]["model_runtime"])
    changed["operator_capability_fingerprint"] = "changed-support"
    revised = replace(
        source,
        plan={
            **source.plan,
            "binding": {**source.plan["binding"], "model_runtime": changed},
        },
    )

    with pytest.raises(ProcessV2T1RuntimeError, match="architecture or legal support"):
        build_process_v2_t1_scratch_runtime(revised)


def test_prepared_plan_groups_panel_entries_by_existing_active8_chunk(
    genuine_stage, tmp_path: Path
) -> None:
    source, panel, revision, first = _standin_planned_panel(genuine_stage, tmp_path)
    second = build_process_v2_t1_prepared_plan(panel, source=source, source_revision=revision)
    authenticated = authenticate_process_v2_t1_prepared_plan_for_worker(
        first, panel=panel, source=source
    )

    assert second == first
    assert authenticated == first
    assert first["task_count"] == len(first["tasks"])
    assert first["selected_entry_count"] == len(panel["entries"])
    assert sorted(
        identifier for task in first["tasks"] for identifier in task["panel_entry_sha256s"]
    ) == sorted(entry["panel_entry_sha256"] for entry in panel["entries"])


def test_prepared_leaf_publication_is_restart_safe(
    genuine_stage, tmp_path: Path, monkeypatch
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
        "schema": t1_runtime.LEAF_SCHEMA,
        "schema_version": t1_runtime.LEAF_SCHEMA_VERSION,
        "status": t1_runtime.LEAF_STATUS,
        **t1_runtime.authority_false_block(),
        "plan_sha256": plan["plan_sha256"],
        "run_identity_sha256": plan["run_identity_sha256"],
        "build_identity_sha256": plan["build_identity_sha256"],
        "task_identity_sha256": task["task_identity_sha256"],
        "active8_task_identity_sha256": task["active8_task_identity_sha256"],
        "panel_sha256": plan["panel_sha256"],
        "initial_model_state_sha256": binding["model_runtime"]["initial_model_state_sha256"],
        "entry_count": 1,
        "entry_inventory_sha256": canonical_sha256([entry]),
        "entries": [entry],
        "model_scores_or_probabilities_stored": False,
        "hazard_included": False,
    }
    leaf = {**leaf_body, "leaf_sha256": canonical_sha256(leaf_body)}
    first = write_process_v2_t1_prepared_leaf(leaf, plan=plan, run_root=tmp_path / "run")
    first_bytes = first.read_bytes()
    second = write_process_v2_t1_prepared_leaf(leaf, plan=plan, run_root=tmp_path / "run")
    assert second == first
    assert second.read_bytes() == first_bytes

    def forbidden_decode(*_args, **_kwargs):
        raise AssertionError("authenticated leaf reuse must not decode chemistry")

    monkeypatch.setattr(t1_runtime, "decode_state", forbidden_decode)
    assert t1_runtime._load_reusable_prepared_leaf(second, plan=plan) == leaf


def test_reduction_refuses_an_incomplete_leaf_inventory(genuine_stage, tmp_path: Path) -> None:
    source, panel, _revision, plan = _standin_planned_panel(genuine_stage, tmp_path)

    with pytest.raises(ProcessV2T1RuntimeError, match="prepared leaf is absent"):
        build_process_v2_t1_prepared_inputs(
            plan,
            panel=panel,
            source=source,
            run_root=tmp_path / "empty-run",
        )


def test_reduction_inventory_ignores_runtime_family_order() -> None:
    expected = ["0" * 64, "f" * 64]
    observed_in_family_order = ["f" * 64, "0" * 64]

    t1_runtime._require_complete_panel_entry_inventory(
        observed_in_family_order,
        expected,
    )


def test_runtime_reads_required_cells_from_the_process_v2_role_schema() -> None:
    roles = t1_runtime.load_process_v2_chain_artifact(
        t1_runtime.DEVELOPMENT_CELL_ROLES,
        repo_root=ROOT,
    )
    required = t1_runtime._required_editing_cell_ids(roles)

    assert required == set(roles["required_cell_ids"])
    assert len(required) == 17


def test_leaf_reuse_requires_the_exact_authorized_plan(genuine_stage, tmp_path: Path) -> None:
    source, panel, _revision, plan = _standin_planned_panel(genuine_stage, tmp_path)

    assert (
        t1_runtime.validate_process_v2_t1_leaf_reuse_plan(
            plan,
            panel=panel,
            source=source,
            expected_plan_sha256=plan["plan_sha256"],
        )
        == plan
    )
    with pytest.raises(ProcessV2T1RuntimeError, match="explicitly authorized plan"):
        t1_runtime.validate_process_v2_t1_leaf_reuse_plan(
            plan,
            panel=panel,
            source=source,
            expected_plan_sha256="0" * 64,
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


def test_a_resealed_wrong_teacher_key_is_refused(genuine_stage, tmp_path: Path) -> None:
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
        "initial_model_state_sha256": binding["model_runtime"]["initial_model_state_sha256"],
        "entry_count": 1,
        "entry_inventory_sha256": canonical_sha256([entry]),
        "entries": [entry],
        "model_scores_or_probabilities_stored": False,
        "hazard_included": False,
    }
    leaf = {**leaf_body, "leaf_sha256": canonical_sha256(leaf_body)}

    first_path = write_process_v2_t1_prepared_leaf(leaf, plan=plan, run_root=tmp_path)
    first_bytes = first_path.read_bytes()
    second_path = write_process_v2_t1_prepared_leaf(leaf, plan=plan, run_root=tmp_path)
    assert second_path == first_path
    assert second_path.read_bytes() == first_bytes

    post_seal_mutation = {**leaf, "entry_count": 2}
    with pytest.raises(ProcessV2T1RuntimeError, match="self-hash"):
        validate_process_v2_t1_prepared_leaf(post_seal_mutation, plan=plan)

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

    with pytest.raises(ProcessV2T1RuntimeError, match="state or successor census disagrees"):
        validate_process_v2_t1_prepared_leaf(invalid_leaf, plan=plan)


def test_prepared_plan_refuses_a_forged_commit() -> None:
    forged_body = {
        "commit": "b" * 40,
    }
    forged = {
        **forged_body,
        "source_revision_sha256": canonical_sha256(forged_body),
    }

    with pytest.raises(ProcessV2T1RuntimeError, match="fields disagree"):
        _validate_source_revision(forged)


def test_publish_writes_nothing_when_a_planned_leaf_is_missing(
    genuine_stage, tmp_path: Path
) -> None:
    source, panel, _revision, plan = _standin_planned_panel(genuine_stage, tmp_path)
    output = tmp_path / "failed-release"
    with pytest.raises(ProcessV2T1RuntimeError, match="prepared leaf is absent"):
        publish_process_v2_t1_prepared_inputs(
            plan,
            panel=panel,
            source=source,
            run_root=output,
        )
    assert not (output / t1_runtime.PREPARED_FILENAME).exists()
    assert not (output / t1_runtime.COMPLETION_FILENAME).exists()


def _minimal_published_release(
    root: Path,
    *,
    source,
    binding,
    entry,
) -> Path:
    revision = _test_source_revision()
    body = {
        "schema": t1_runtime.PREPARED_SCHEMA,
        "schema_version": t1_runtime.PREPARED_SCHEMA_VERSION,
        "status": t1_runtime.PREPARED_STATUS,
        **t1_runtime.authority_false_block(),
        "source_revision": revision,
        "implementation_sha256": t1_runtime._implementation_sha256(ROOT),
        "leaf_source_revision": revision,
        "leaf_implementation_sha256": t1_runtime._implementation_sha256(ROOT),
        "leaf_reuse": {
            "reused_precomputed_leaves": False,
            "leaf_plan_sha256": canonical_sha256("minimal-plan"),
            "leaf_run_identity_sha256": canonical_sha256("minimal-run"),
            "reduction_rule": "complete_panel_entry_identity_multiset_v1",
        },
        "process_identity_sha256": source.contracts.process_identity_sha256,
        "active8_completion_sha256": source.index.active8_completion_sha256,
        "active8_sentinel_sha256": source.index.active8_sentinel_sha256,
        "gate_zero_decision_sha256": source.decision["decision_sha256"],
        "panel_sha256": canonical_sha256("minimal-panel"),
        "panel_entry_inventory_sha256": canonical_sha256([entry["panel_entry_sha256"]]),
        "plan_sha256": canonical_sha256("minimal-plan"),
        "run_identity_sha256": canonical_sha256("minimal-run"),
        "build_identity_sha256": canonical_sha256("minimal-build"),
        "manifest_sha256": canonical_sha256("minimal-manifest"),
        "model_binding": binding,
        "support_time_hex": float(0.5).hex(),
        "state_encoding": "compose.rewrite.trace.encoded_state_v2_exact_slots",
        "successor_partition_contract": {
            "productive_groups_complete": True,
            "productive_aliases_disjoint": True,
            "virtual_aliases_disjoint": True,
            "self_events_excluded_from_embedded_jump_chain": True,
            "model_scores_or_probabilities_stored": False,
            "hazard_included": False,
        },
        "leaf_count": 1,
        "leaf_inventory_sha256": canonical_sha256("minimal-leaf"),
        "entry_count": 1,
        "entry_inventory_sha256": canonical_sha256((entry,)),
        "entries": [entry],
    }
    artifact = {**body, "artifact_sha256": canonical_sha256(body)}
    artifact = validate_process_v2_t1_prepared_inputs(artifact, repo_root=ROOT)
    artifact_raw = canonical_bytes(artifact) + b"\n"
    completion = _build_prepared_completion(
        artifact,
        artifact_raw=artifact_raw,
    )
    root.mkdir(parents=True)
    (root / t1_runtime.PREPARED_FILENAME).write_bytes(artifact_raw)
    completion_path = root / t1_runtime.COMPLETION_FILENAME
    completion_path.write_bytes(canonical_bytes(completion) + b"\n")
    return completion_path


def test_completion_reopens_without_chemistry(genuine_stage, tmp_path: Path, monkeypatch) -> None:
    _stage, source, _decision = genuine_stage
    candidate = next(iter(iter_process_v2_t1_candidates(source)))
    panel_entry = {
        **candidate,
        "panel_entry_sha256": candidate["assignment_sha256"],
    }
    resolved = resolve_process_v2_t1_entries(source, (panel_entry,))
    scratch, binding = build_process_v2_t1_scratch_runtime(source)
    entry = _entry_from_exact_transition(
        panel_entry,
        addressed=resolved[0].addressed_trace,
        scratch_runtime=scratch,
        support_time=0.5,
    )
    completion_path = _minimal_published_release(
        tmp_path / "valid-release",
        source=source,
        binding=binding,
        entry=entry,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("normal reopen must not execute or enumerate chemistry")

    monkeypatch.setattr(t1_runtime, "compile_state_successor_map", forbidden)
    monkeypatch.setattr(t1_runtime, "resolve_process_v2_t1_entries", forbidden)
    completion, artifact, _completion_file, _artifact_file = _load_prepared_completion(
        completion_path,
        repo_root=ROOT,
    )
    assert completion["entry_count"] == 1
    assert artifact["entry_count"] == 1
