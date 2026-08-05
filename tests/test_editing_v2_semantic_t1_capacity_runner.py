"""Prepared-input and cache-only semantic T1 runner invariants."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
import torch

from compose_v4.chem.molecular_graph import (
    ORGANIC_VOCABULARY,
    smiles_to_molecular_graph,
)
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.successor_fiber_cache import (
    SuccessorFiberCacheAddress,
    SuccessorFiberCacheRecord,
)
from compose_v4.experiments.editing_v2_semantic_development_cell_roles import (
    load_semantic_development_cell_roles,
)
from compose_v4.experiments.editing_v2_semantic_t1_capacity_policy import (
    LEGACY_DENSE_TRAJECTORY_EVALUATION,
    PROCESS_V2_SPARSE_TRAJECTORY_EVALUATION,
    SPARSE_REPORT_POINT_TRAJECTORY_EVALUATION,
    load_semantic_t1_capacity_policy,
)
from compose_v4.experiments import editing_v2_semantic_t1_capacity_runner as capacity_runner
from compose_v4.experiments.editing_v2_semantic_t1_capacity_runner import (
    RESULT_FILENAME,
    SemanticT1CapacityRunnerError,
    SemanticT1RuntimeInputs,
    _build_and_publish_capacity_result,
    build_semantic_t1_failure_diagnostics,
    optimizer_state_semantic_sha256,
    semantic_t1_address_stream,
    semantic_t1_evaluation_schedule,
    semantic_t1_threshold_stop_allowed,
    semantic_t1_threshold_checks,
    summarize_semantic_t1_metrics,
)
from compose_v4.experiments.editing_v2_semantic_t1_prepared_inputs import (
    LoadedSemanticT1PreparedInputs,
    SemanticT1PreparedInputError,
    build_semantic_t1_prepared_inputs,
    load_semantic_t1_prepared_inputs,
    write_semantic_t1_prepared_inputs,
)
from compose_v4.experiments.editing_v2_semantic_t1_successor_cache import (
    SemanticT1SuccessorCache,
)
from compose_v4.experiments.factorized_successor_training import (
    CanonicalSuccessorAliasGroup,
    CompiledStateSuccessorMap,
    CompiledSuccessorMark,
    StateProductiveSupport,
    TeacherSuccessorAlias,
    TeacherSuccessorFiber,
    compile_state_successor_map,
    forward_compiled_successor_partitions,
    resolve_successor_process_runtime,
)
from compose_v4.experiments.production_successor_kernel import (
    canonical_successor_result,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedTraceletRateModel,
    prepare_factorized_mark_batch,
)
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "configs/editing_v2_semantic_t1_capacity_policy_v1.json"


def test_sparse_policy_calls_full_panel_exactly_at_initial_and_report_points() -> None:
    optimization = {
        "maximum_optimizer_steps": 500,
        "report_points": [1, 10, 50, 100, 250, 500],
        "trajectory_evaluation": SPARSE_REPORT_POINT_TRAJECTORY_EVALUATION,
    }
    schedule = semantic_t1_evaluation_schedule(optimization)
    assert schedule == (0, 1, 10, 50, 100, 250, 500)
    assert len(schedule) == 7


def test_process_v2_sparse_schedule_spelling_resolves_the_same_frozen_steps() -> None:
    optimization = {
        "maximum_optimizer_steps": 500,
        "report_points": [1, 10, 50, 100, 250, 500],
        "trajectory_evaluation": PROCESS_V2_SPARSE_TRAJECTORY_EVALUATION,
    }
    assert semantic_t1_evaluation_schedule(optimization) == (
        0,
        1,
        10,
        50,
        100,
        250,
        500,
    )


def test_legacy_dense_policy_retains_every_state_evaluation() -> None:
    optimization = {
        "maximum_optimizer_steps": 3,
        "report_points": [1, 3],
        "trajectory_evaluation": LEGACY_DENSE_TRAJECTORY_EVALUATION,
    }
    assert semantic_t1_evaluation_schedule(optimization) == (0, 1, 2, 3)


def test_threshold_pass_cannot_stop_before_ten_optimizer_steps() -> None:
    passed = {"capacity": True, "gradient": True}
    failed = {"capacity": True, "gradient": False}
    assert not semantic_t1_threshold_stop_allowed(
        optimizer_step=1,
        threshold_checks=passed,
    )
    assert semantic_t1_threshold_stop_allowed(
        optimizer_step=10,
        threshold_checks=passed,
    )
    assert not semantic_t1_threshold_stop_allowed(
        optimizer_step=10,
        threshold_checks=failed,
    )


def test_default_result_publication_preserves_the_legacy_builder_and_filename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    expected = {"schema": "legacy-result", "result_sha256": "a" * 64}

    def legacy_builder(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(capacity_runner, "build_semantic_t1_capacity_result", legacy_builder)
    projections = {
        "provenance": {"legacy": "provenance"},
        "run_integrity": {"steps": 10},
        "evaluation_trajectory": [{"step": 0}, {"step": 10}],
        "entry_metrics": [{"entry": "legacy"}],
        "gradient_evidence": [{"family": "atom_insert"}],
    }
    result, result_path = _build_and_publish_capacity_result(
        output_root=tmp_path,
        **projections,
    )

    assert calls == [projections]
    assert result == expected
    assert result_path == tmp_path / RESULT_FILENAME
    assert result_path.read_bytes() == (
        json.dumps(expected, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )


def test_injected_result_builder_publishes_without_legacy_v1_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_legacy_builder(**_kwargs):
        raise AssertionError("legacy result builder must not run")

    observed: list[dict[str, object]] = []

    def process_v2_builder(**kwargs):
        observed.append(kwargs)
        return {
            "schema": "compose.editing_v2.process_v2_t1_capacity_result",
            "result_sha256": "b" * 64,
        }

    monkeypatch.setattr(
        capacity_runner,
        "build_semantic_t1_capacity_result",
        reject_legacy_builder,
    )
    process_v2_provenance = {
        "process_identity_sha256": "c" * 64,
        "prepared_completion_sha256": "d" * 64,
    }
    result, result_path = _build_and_publish_capacity_result(
        output_root=tmp_path,
        provenance=process_v2_provenance,
        run_integrity={"steps": 10},
        evaluation_trajectory=[{"step": 0}, {"step": 10}],
        entry_metrics=[{"entry": "process-v2"}],
        gradient_evidence=[{"family": "atom_insert"}],
        result_builder=process_v2_builder,
        result_filename="PROCESS_V2_T1_CAPACITY_RESULT.json",
    )

    assert observed[0]["provenance"] == process_v2_provenance
    assert set(observed[0]) == {
        "provenance",
        "run_integrity",
        "evaluation_trajectory",
        "entry_metrics",
        "gradient_evidence",
    }
    assert result["schema"] == "compose.editing_v2.process_v2_t1_capacity_result"
    assert result_path == tmp_path / "PROCESS_V2_T1_CAPACITY_RESULT.json"
    assert result_path.is_file()


def _source_revision() -> dict[str, object]:
    body: dict[str, object] = {"worktree_clean": True, "commit": "4" * 40}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return {**body, "source_revision_sha256": hashlib.sha256(encoded).hexdigest()}


def _state(smiles: str, *, slots: int = 12):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), slots)


def _synthetic_prepared_fixture():
    source = _state("C")
    # The strict state digest must be real, unlike unrelated lineage fixture IDs.
    from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256

    source_sha = persistent_slot_state_sha256(source)
    target_sha = hashlib.sha256(b"target").hexdigest()
    alias = TeacherSuccessorAlias(
        family_name="atom_insert",
        table_name="grow_connected",
        coordinate=(0, 0, 0),
    )
    support = StateProductiveSupport("C", source_sha, ())
    teacher = TeacherSuccessorFiber("C", "CC", target_sha, (alias,), support)
    record = SuccessorFiberCacheRecord(
        address=SuccessorFiberCacheAddress(
            packed_shard_content_sha256="a" * 64,
            packed_shard_name="train.jsonl.gz",
            entry_index=0,
            layer="observed_local_analogue",
            partition="train",
            trace_id="trace-0",
            trace_source_key="C",
            trace_target_key="CC",
            progress_index=0,
            path_length=1,
        ),
        state_support=support,
        teacher_fiber=teacher,
    )
    partition = CompiledStateSuccessorMap(
        state_support=support,
        successor_groups=(
            CanonicalSuccessorAliasGroup(
                target_key="CC",
                marks=(
                    CompiledSuccessorMark(
                        alias=alias,
                        successor_state_sha256=target_sha,
                        action_sha256="b" * 64,
                    ),
                ),
            ),
        ),
        virtual_marks=(),
    )
    panel_id = "c" * 64
    cache = SemanticT1SuccessorCache(
        completion=MappingProxyType(
            {
                "completion_sha256": "d" * 64,
                "decision_source_inventory_sha256": "e" * 64,
                "initial_model_state_sha256": "f" * 64,
            }
        ),
        manifest=MappingProxyType(
            {
                "manifest_sha256": "1" * 64,
                "panel_binding": {"panel_artifact_sha256": "2" * 64},
                "semantic_model_process_contract": {"contract_sha256": "3" * 64},
                "source_revision": {"source_revision_sha256": "a" * 64},
            }
        ),
        records=(record,),
        records_by_address_sha256=MappingProxyType({}),
        records_by_panel_entry_sha256=MappingProxyType({panel_id: record}),
    )
    entry = SimpleNamespace(
        panel_entry_sha256=panel_id,
        model_family="atom_insert",
        capability_cell_id="cell:atom_insert:root",
        support_time_hex=(0.5).hex(),
        source_state_sha256=source_sha,
        successor_canonical_key="CC",
        representative_target_state_sha256=target_sha,
        objective_coefficient=1,
        raw_mark_count=1,
        canonical_successor_count=1,
        production_successor_alias_multiplicity=1,
    )
    panel = SimpleNamespace(
        artifact_sha256="2" * 64,
        request=SimpleNamespace(support_time_hex=(0.5).hex()),
        entries=(entry,),
    )
    return source, partition, cache, panel, panel_id


def test_prepared_input_round_trip_binds_exact_state_and_full_partition(
    tmp_path: Path,
) -> None:
    source, partition, cache, panel, panel_id = _synthetic_prepared_fixture()
    policy = load_semantic_t1_capacity_policy(POLICY_PATH)
    artifact = build_semantic_t1_prepared_inputs(
        panel=panel,
        cache=cache,
        source_states_by_panel_entry_sha256={panel_id: source},
        successor_partitions_by_panel_entry_sha256={panel_id: partition},
        capacity_policy=policy,
        capacity_policy_file_sha256=hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest(),
        source_revision=_source_revision(),
        repo_root=ROOT,
    )
    path = tmp_path / "SEMANTIC_T1_PREPARED_INPUTS.json"
    assert write_semantic_t1_prepared_inputs(path, artifact)
    loaded = load_semantic_t1_prepared_inputs(
        path,
        expected_capacity_policy_sha256=policy["policy_sha256"],
        expected_panel_artifact_sha256=panel.artifact_sha256,
        expected_cache_completion_sha256=cache.completion["completion_sha256"],
        expected_initial_model_state_sha256=cache.completion["initial_model_state_sha256"],
        repo_root=ROOT,
    )
    assert loaded.partitions_by_panel_entry_sha256[panel_id] == partition

    tampered = copy.deepcopy(artifact)
    tampered["entries"][0]["successor_partition"]["successor_groups"] = []
    body = dict(tampered["entries"][0])
    body.pop("entry_sha256")
    from compose_v4.experiments import (
        editing_v2_semantic_t1_prepared_inputs as prepared,
    )

    tampered["entries"][0]["entry_sha256"] = prepared._sha(body)
    tampered["entry_inventory_sha256"] = prepared._sha(tampered["entries"])
    root_body = dict(tampered)
    root_body.pop("artifact_sha256")
    tampered["artifact_sha256"] = prepared._sha(root_body)
    with pytest.raises(SemanticT1PreparedInputError, match="successor"):
        prepared.validate_semantic_t1_prepared_inputs(
            tampered,
            expected_capacity_policy_sha256=policy["policy_sha256"],
            expected_panel_artifact_sha256=panel.artifact_sha256,
            expected_cache_completion_sha256=cache.completion["completion_sha256"],
            expected_initial_model_state_sha256=cache.completion["initial_model_state_sha256"],
            repo_root=ROOT,
        )

    unknown = copy.deepcopy(artifact)
    unknown["entries"][0]["undeclared_field"] = "not allowed"
    unknown_entry_body = dict(unknown["entries"][0])
    unknown_entry_body.pop("entry_sha256")
    unknown["entries"][0]["entry_sha256"] = prepared._sha(unknown_entry_body)
    unknown["entry_inventory_sha256"] = prepared._sha(unknown["entries"])
    unknown_body = dict(unknown)
    unknown_body.pop("artifact_sha256")
    unknown["artifact_sha256"] = prepared._sha(unknown_body)
    with pytest.raises(SemanticT1PreparedInputError, match="prepared entry"):
        prepared.validate_semantic_t1_prepared_inputs(
            unknown,
            expected_capacity_policy_sha256=policy["policy_sha256"],
            expected_panel_artifact_sha256=panel.artifact_sha256,
            expected_cache_completion_sha256=cache.completion["completion_sha256"],
            expected_initial_model_state_sha256=cache.completion["initial_model_state_sha256"],
            repo_root=ROOT,
        )


def test_optimizer_state_semantic_hash_is_order_stable_and_value_sensitive() -> None:
    first = {
        "state": {0: {"step": torch.tensor(1), "exp_avg": torch.tensor([1.0, 2.0])}},
        "param_groups": [{"lr": 1e-3, "params": [0]}],
    }
    reordered = {
        "param_groups": [{"params": [0], "lr": 1e-3}],
        "state": {0: {"exp_avg": torch.tensor([1.0, 2.0]), "step": torch.tensor(1)}},
    }
    changed = copy.deepcopy(first)
    changed["state"][0]["exp_avg"][1] = 3.0
    assert optimizer_state_semantic_sha256(first) == (optimizer_state_semantic_sha256(reordered))
    assert optimizer_state_semantic_sha256(first) != (optimizer_state_semantic_sha256(changed))


def test_recovery_checkpoint_requires_trusted_physical_hash(
    tmp_path: Path,
    partition_model,
) -> None:
    from compose_v4.experiments import editing_v2_semantic_t1_capacity_runner as runner

    model = copy.deepcopy(partition_model).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    runtime = SimpleNamespace(
        capacity_policy={"policy_sha256": "1" * 64},
        prepared=SimpleNamespace(artifact={"artifact_sha256": "2" * 64}),
        cache=SimpleNamespace(
            completion={
                "completion_sha256": "3" * 64,
                "initial_model_state_sha256": "4" * 64,
            },
            manifest={"manifest_sha256": "5" * 64},
        ),
    )
    path = tmp_path / "step_0010.pt"
    checkpoint_identity = {"test_identity_sha256": "7" * 64}
    receipt = runner._write_checkpoint(
        path,
        runtime=runtime,
        model=model,
        optimizer=optimizer,
        completed_steps=10,
        selected_step=4,
        selected_criterion=(0.5, -0.7, -4),
        selected_state=runner._clone_state_dict(model),
        trajectory=[],
        gradient_evidence={},
        stream_sha256="6" * 64,
        resume_count=2,
        checkpoint_identity=checkpoint_identity,
    )
    recovered_model = copy.deepcopy(partition_model).train()
    recovered_optimizer = torch.optim.AdamW(recovered_model.parameters(), lr=1e-3)
    with pytest.raises(SemanticT1CapacityRunnerError, match="physical SHA-256"):
        runner._load_checkpoint(
            path,
            runtime=runtime,
            model=recovered_model,
            optimizer=recovered_optimizer,
            expected_stream_sha256="6" * 64,
            expected_file_sha256="0" * 64,
            expected_checkpoint_identity=checkpoint_identity,
        )
    recovered = runner._load_checkpoint(
        path,
        runtime=runtime,
        model=recovered_model,
        optimizer=recovered_optimizer,
        expected_stream_sha256="6" * 64,
        expected_file_sha256=receipt["file_sha256"],
        expected_checkpoint_identity=checkpoint_identity,
    )
    assert recovered["completed_steps"] == 10
    assert recovered["resume_count"] == 2

    mismatched_identity = {**checkpoint_identity, "environment": "changed"}
    with pytest.raises(SemanticT1CapacityRunnerError, match="identity disagrees"):
        runner._load_checkpoint(
            path,
            runtime=runtime,
            model=copy.deepcopy(partition_model).train(),
            optimizer=torch.optim.AdamW(copy.deepcopy(partition_model).parameters(), lr=1e-3),
            expected_stream_sha256="6" * 64,
            expected_file_sha256=receipt["file_sha256"],
            expected_checkpoint_identity=mismatched_identity,
        )


def test_checkpoint_identity_binds_implementation_environment_and_optimizer(
    partition_model,
) -> None:
    from compose_v4.experiments import editing_v2_semantic_t1_capacity_runner as runner

    torch.use_deterministic_algorithms(True)
    model = copy.deepcopy(partition_model).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    source_revision_sha256 = "a" * 64
    optimization = {
        "optimizer": "adamw",
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "deterministic_algorithms_required": True,
    }
    runtime = SimpleNamespace(
        capacity_policy={"policy_sha256": "1" * 64, "optimization": optimization},
        prepared=SimpleNamespace(
            artifact={
                "artifact_sha256": "2" * 64,
                "source_revision": {"source_revision_sha256": source_revision_sha256},
            }
        ),
        cache=SimpleNamespace(
            completion={
                "completion_sha256": "3" * 64,
                "initial_model_state_sha256": "4" * 64,
            },
            manifest={"manifest_sha256": "5" * 64},
        ),
    )
    environment_body = {
        "accelerator_class": "cpu",
        "dtype": "float32",
        "device_name": "unit-test-cpu",
    }
    provenance = {
        "runner_implementation_sha256": (
            runner.semantic_t1_runner_implementation_sha256(repo_root=ROOT)
        ),
        "runner_source_revision_sha256": source_revision_sha256,
        "execution_environment": {
            **environment_body,
            "environment_sha256": runner._sha(environment_body),
        },
    }
    identity = runner._checkpoint_identity(
        runtime,
        provenance=provenance,
        model=model,
        optimizer=optimizer,
    )
    assert identity["runner_source_revision_sha256"] == source_revision_sha256
    assert identity["runner_implementation_sha256"] == provenance["runner_implementation_sha256"]
    assert identity["execution_environment"] == provenance["execution_environment"]
    assert identity["model_device_type"] == "cpu"
    assert identity["model_dtype"] == "torch.float32"
    assert identity["deterministic_algorithms_enabled"] is True
    assert identity["optimizer_configuration"]["class"].endswith(".AdamW")
    assert identity["optimization_policy"] == optimization


def test_checkpoint_resume_is_bit_identical_to_uninterrupted_cpu_execution(
    tmp_path: Path,
    partition_model,
) -> None:
    from compose_v4.experiments import editing_v2_semantic_t1_capacity_runner as runner
    from compose_v4.experiments.editing_p50_gate import (
        state_dict_semantic_sha256,
    )

    def seed_all() -> None:
        random.seed(77)
        np.random.seed(77)
        torch.manual_seed(77)

    def advance(model, optimizer, count: int) -> None:
        parameters = tuple(model.parameters())
        for _ in range(count):
            scale = random.random() + float(np.random.random()) + float(torch.rand(()))
            optimizer.zero_grad(set_to_none=True)
            loss = sum((parameter.reshape(-1)[0] * scale).square() for parameter in parameters)
            loss.backward()
            optimizer.step()

    initial = copy.deepcopy(partition_model).train()
    baseline = copy.deepcopy(initial).train()
    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=1e-3)
    seed_all()
    advance(baseline, baseline_optimizer, 6)
    baseline_future = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
    )

    interrupted = copy.deepcopy(initial).train()
    interrupted_optimizer = torch.optim.AdamW(interrupted.parameters(), lr=1e-3)
    seed_all()
    advance(interrupted, interrupted_optimizer, 3)
    checkpoint_identity = {"resume_equivalence_sha256": "8" * 64}
    path = tmp_path / "step_0003.pt"
    receipt = runner._write_checkpoint(
        path,
        runtime=SimpleNamespace(),
        model=interrupted,
        optimizer=interrupted_optimizer,
        completed_steps=3,
        selected_step=3,
        selected_criterion=(0.1, -1.0, -3),
        selected_state=runner._clone_state_dict(interrupted),
        trajectory=[],
        gradient_evidence={},
        stream_sha256="9" * 64,
        resume_count=0,
        checkpoint_identity=checkpoint_identity,
    )
    random.random()
    np.random.random()
    torch.rand(())
    resumed = copy.deepcopy(initial).train()
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    runner._load_checkpoint(
        path,
        runtime=SimpleNamespace(),
        model=resumed,
        optimizer=resumed_optimizer,
        expected_stream_sha256="9" * 64,
        expected_file_sha256=receipt["file_sha256"],
        expected_checkpoint_identity=checkpoint_identity,
    )
    advance(resumed, resumed_optimizer, 3)
    resumed_future = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
    )
    assert state_dict_semantic_sha256(resumed.state_dict()) == (
        state_dict_semantic_sha256(baseline.state_dict())
    )
    assert optimizer_state_semantic_sha256(resumed_optimizer.state_dict()) == (
        optimizer_state_semantic_sha256(baseline_optimizer.state_dict())
    )
    assert resumed_future == baseline_future


def test_hierarchical_stream_is_deterministic_and_bound_to_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from compose_v4.experiments import editing_v2_semantic_t1_capacity_runner as runner

    policy = load_semantic_t1_capacity_policy(POLICY_PATH)
    entries = []
    required_cells = load_semantic_development_cell_roles().required_cell_ids
    for family in policy["required_families"]:
        family_cells = tuple(cell for cell in required_cells if cell.rsplit(":", 2)[1] == family)
        for index in range(64):
            cell_id = family_cells[index % len(family_cells)]
            entries.append(
                {
                    "panel_entry_sha256": hashlib.sha256(f"{cell_id}:{index}".encode()).hexdigest(),
                    "model_family": family,
                    "capability_cell_id": cell_id,
                }
            )
    prepared = LoadedSemanticT1PreparedInputs(
        artifact=MappingProxyType(
            {
                "artifact_sha256": "6" * 64,
                "capacity_policy_sha256": policy["policy_sha256"],
                "cache_completion_sha256": "7" * 64,
                "cache_manifest_sha256": "8" * 64,
                "panel_artifact_sha256": "9" * 64,
                "entries": entries,
            }
        ),
        states_by_panel_entry_sha256=MappingProxyType({}),
        partitions_by_panel_entry_sha256=MappingProxyType({}),
    )
    cache = SemanticT1SuccessorCache(
        completion=MappingProxyType({"completion_sha256": "7" * 64}),
        manifest=MappingProxyType(
            {
                "manifest_sha256": "8" * 64,
                "panel_binding": {"panel_artifact_sha256": "9" * 64},
            }
        ),
        records=(),
        records_by_address_sha256=MappingProxyType({}),
        records_by_panel_entry_sha256=MappingProxyType({}),
    )
    monkeypatch.setattr(
        runner,
        "authenticate_semantic_t1_prepared_inputs",
        lambda candidate, *, panel, cache: candidate,
    )
    runtime = SemanticT1RuntimeInputs(
        prepared,
        SimpleNamespace(artifact_sha256="9" * 64),
        cache,
        policy,
    )
    assert semantic_t1_address_stream(runtime, draw_count=512) == (
        semantic_t1_address_stream(runtime, draw_count=512)
    )
    assert len(set(semantic_t1_address_stream(runtime, draw_count=512))) > 16

    missing = next(cell for cell in required_cells if cell.endswith(":connected_nonleaf_death"))
    replacement = next(cell for cell in required_cells if cell.endswith(":leaf_death"))
    missing_cell_entries = [
        {
            **entry,
            "capability_cell_id": (
                replacement
                if entry["capability_cell_id"] == missing
                else entry["capability_cell_id"]
            ),
        }
        for entry in entries
    ]
    bad_prepared = LoadedSemanticT1PreparedInputs(
        artifact=MappingProxyType({**dict(prepared.artifact), "entries": missing_cell_entries}),
        states_by_panel_entry_sha256=MappingProxyType({}),
        partitions_by_panel_entry_sha256=MappingProxyType({}),
    )
    with pytest.raises(SemanticT1CapacityRunnerError, match="exact frozen required"):
        SemanticT1RuntimeInputs(
            bad_prepared,
            SimpleNamespace(artifact_sha256="9" * 64),
            cache,
            policy,
        )


def test_metric_checks_apply_family_cell_entry_and_gradient_floors() -> None:
    policy = load_semantic_t1_capacity_policy(POLICY_PATH)
    rows = [
        {
            "panel_entry_sha256": "a" * 64,
            "model_family": "atom_insert",
            "capability_cell_id": "cell:atom_insert:root",
            "teacher_successor_probability": 0.9,
            "teacher_successor_nll": -np.log(0.9),
            "teacher_successor_rank": 1,
            "teacher_successor_top1": True,
            "canonical_successor_count": 2,
        }
    ]
    metrics = summarize_semantic_t1_metrics(rows)
    gradients = {
        "atom_insert": {
            "family_route_finite_nonzero_seen": True,
            "action_route_finite_nonzero_seen": True,
        }
    }
    assert all(
        semantic_t1_threshold_checks(
            metrics,
            thresholds=policy["thresholds"],
            gradient_evidence=gradients,
        ).values()
    )
    gradients["atom_insert"]["action_route_finite_nonzero_seen"] = False
    assert not semantic_t1_threshold_checks(
        metrics,
        thresholds=policy["thresholds"],
        gradient_evidence=gradients,
    )["action_route_gradient"]

    diagnostics = build_semantic_t1_failure_diagnostics(
        capacity_policy=policy,
        selected_metrics=metrics,
        gradient_evidence=gradients,
        result_sha256="f" * 64,
    )
    assert diagnostics["status"].startswith("BLOCKED_")
    assert diagnostics["failure_diagnostic_scope_order"] == [
        "heads_only",
        "heads_plus_local_adapter_if_distinct",
        "all",
    ]
    assert diagnostics["failing_families"] == [
        {
            "family": "atom_insert",
            "failed_gates": ["action_route_gradient"],
        }
    ]
    assert diagnostics["diagnostic_training_executed"] is False
    assert diagnostics["bounded_p50_authorized"] is False


@pytest.fixture(scope="module")
def partition_model():
    target = _state("c1ccccc1")
    source = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,)).sample(
        np.random.default_rng(2), n_slots=12
    )
    trace = compile_carbon_tree_to_target(
        source,
        target,
        use_bond_reroute=True,
        align_source=True,
    )
    catalog = build_typed_ring_catalog((trace,))
    torch.manual_seed(4)
    return FactorizedTraceletRateModel(
        catalog,
        hidden_dim=12,
        message_passing_steps=1,
        enable_ring_restates=True,
        enable_cyclic_graft=True,
        enable_heteroatom_scan=True,
        enable_ring_opening=True,
        enable_cycle_ops=True,
        enable_ring_grow_macro=False,
        enable_ring_system_delete=False,
        atom_vocabulary=ORGANIC_VOCABULARY,
    ).eval()


@pytest.fixture(scope="module")
def process_v2_partition_model():
    """The same architecture as `partition_model`, under Process V2.

    Kept separate rather than parameterised: `partition_model` is the legacy
    control that must stay byte-identical, and every V2-only field is `None`
    there.
    """

    from compose_v4.model.factorized_tracelet_rate_model import (
        PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
        PROCESS_V2_EDITING_PROCESS_SEMANTICS,
        SEMANTIC_ATOM_RESTATE_ACTION_SEMANTICS,
        SEMANTIC_CYCLE_CLOSE_ACTION_SEMANTICS,
        SEMANTIC_CYCLE_OPEN_ACTION_SEMANTICS,
        SEMANTIC_RING_RESTATE_SCORER_MODE,
    )

    target = _state("c1ccccc1")
    source = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,)).sample(
        np.random.default_rng(2), n_slots=12
    )
    trace = compile_carbon_tree_to_target(
        source,
        target,
        use_bond_reroute=True,
        align_source=True,
    )
    catalog = build_typed_ring_catalog((trace,))
    torch.manual_seed(4)
    return FactorizedTraceletRateModel(
        catalog,
        hidden_dim=12,
        message_passing_steps=1,
        enable_ring_restates=True,
        enable_cyclic_graft=True,
        enable_heteroatom_scan=True,
        enable_ring_opening=True,
        enable_cycle_ops=True,
        enable_ring_grow_macro=False,
        enable_ring_system_delete=False,
        editing_process_semantics=PROCESS_V2_EDITING_PROCESS_SEMANTICS,
        atom_restate_action_semantics=SEMANTIC_ATOM_RESTATE_ACTION_SEMANTICS,
        ring_restate_scorer_mode=SEMANTIC_RING_RESTATE_SCORER_MODE,
        cycle_close_action_semantics=SEMANTIC_CYCLE_CLOSE_ACTION_SEMANTICS,
        cycle_open_action_semantics=SEMANTIC_CYCLE_OPEN_ACTION_SEMANTICS,
        atom_delete_action_semantics=PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
        atom_vocabulary=ORGANIC_VOCABULARY,
    ).eval()


def test_process_v2_successor_batch_attaches_family_after_mark_free_collation(
    process_v2_partition_model,
) -> None:
    """Successor training names a family without inventing a mark teacher."""

    from compose_v4.experiments.factorized_mark_conditional import (
        FactorizedMarkExample,
    )

    model = process_v2_partition_model
    example = FactorizedMarkExample(
        state=_state("CCO"),
        time=0.5,
        teacher_action=None,
        teacher_rule_name=None,
        teacher_rate=1.0,
        importance_weight=1.0,
    )
    collated = capacity_runner._collator(model)([example])
    batch = capacity_runner._attach_successor_family_coordinates(
        collated,
        ({"model_family": "atom_restate"},),
    )
    assert batch.teacher_actions == (None,)
    assert batch.teacher_rule_names == ("atom_restate",)


def test_full_partition_scorer_matches_production_segmented_kernel_and_oracle(
    partition_model,
) -> None:
    model = partition_model
    source = _state("c1ccccc1")
    time = 0.41
    partition = compile_state_successor_map(model, source, time=time)
    capabilities = model.operator_capabilities
    batch = prepare_factorized_mark_batch(
        (source,),
        (time,),
        (None,),
        (None,),
        (1.0,),
        use_aromatic_bond_view=True,
        ring_catalog=model.ring_catalog,
        compute_ring_grow_support=capabilities.compute_ring_grow_support,
        compute_ring_restates=capabilities.compute_ring_restates,
        compute_cyclic_graft=capabilities.compute_cyclic_graft,
        compute_ring_opening=capabilities.compute_ring_opening,
        compute_ring_system_delete=capabilities.compute_ring_system_delete,
        editing_process_semantics=capabilities.editing_process_semantics,
        atom_restate_action_semantics=capabilities.atom_restate_action_semantics,
        ring_restate_scorer_mode=capabilities.ring_restate_scorer_mode,
        cycle_close_action_semantics=capabilities.cycle_close_action_semantics,
        cycle_open_action_semantics=capabilities.cycle_open_action_semantics,
    )
    prediction = forward_compiled_successor_partitions(model, batch, (partition,))
    observed = {
        key: float(value.detach().exp())
        for key, value in zip(
            prediction.successor_keys[0],
            prediction.successor_log_probabilities[0],
            strict=True,
        )
    }
    production = canonical_successor_result(model, source, time)
    expected = {successor.key: successor.probability for successor in production.batch.successors}
    assert observed == pytest.approx(expected, abs=2e-6)

    # The independent dictionary oracle remains bounded to this test.
    from compose_v4.experiments.reference_successor_kernel import (
        reference_successor_batch,
    )
    from compose_v4.rewrite.kernel import de_novo_rewrite_system

    reference = reference_successor_batch(
        source,
        [
            (mark.executor_rule_name, mark.action, mark.probability)
            for mark in production.marked_law.marks
        ],
        system=de_novo_rewrite_system(),
        identity=production.batch.identity,
    )
    assert observed == pytest.approx(
        {successor.key: successor.probability for successor in reference.successors},
        abs=2e-6,
    )


def test_full_partition_scorer_covers_exact_semantic_editing_v2_active8() -> None:
    from compose_v4.experiments.editing_gate_zero_semantic_contract import (
        load_gate_zero_semantic_contract,
    )
    from compose_v4.experiments.editing_v2_semantic_runtime import (
        SemanticScratchModelConfig,
        build_semantic_scratch_runtime,
    )
    from compose_v4.experiments.reference_successor_kernel import (
        reference_successor_batch,
    )

    contract = load_gate_zero_semantic_contract(
        ROOT / "configs/editing_gate_zero_semantic_model_process_v1.json"
    )
    model = build_semantic_scratch_runtime(
        SemanticScratchModelConfig(
            initialization_seed=4,
            max_atoms=40,
            hidden_dim=12,
            message_passing_steps=1,
            mark_dim=32,
            dtype="torch.float32",
            atom_vocabulary_class_count=15,
            catalog_fingerprint="639ff6078c32d43c",
        ),
        contract,
    ).model
    source = _state("CCOc1ccccc1", slots=40)
    time = 0.5
    partition = compile_state_successor_map(model, source, time=time)
    observed_families = {
        mark.alias.family_name for group in partition.successor_groups for mark in group.marks
    } | {mark.family_name for mark in partition.virtual_marks}
    policy = load_semantic_t1_capacity_policy(POLICY_PATH)
    assert observed_families == set(policy["required_families"])
    assert model.editing_process_semantics == "semantic_editing_v2_v1"

    capabilities = model.operator_capabilities
    batch = prepare_factorized_mark_batch(
        (source,),
        (time,),
        (None,),
        (None,),
        (1.0,),
        use_aromatic_bond_view=True,
        ring_catalog=model.ring_catalog,
        compute_ring_grow_support=capabilities.compute_ring_grow_support,
        compute_ring_restates=capabilities.compute_ring_restates,
        compute_cyclic_graft=capabilities.compute_cyclic_graft,
        compute_ring_opening=capabilities.compute_ring_opening,
        compute_ring_system_delete=capabilities.compute_ring_system_delete,
        editing_process_semantics=capabilities.editing_process_semantics,
        atom_restate_action_semantics=capabilities.atom_restate_action_semantics,
        ring_restate_scorer_mode=capabilities.ring_restate_scorer_mode,
        cycle_close_action_semantics=capabilities.cycle_close_action_semantics,
        cycle_open_action_semantics=capabilities.cycle_open_action_semantics,
    )
    prediction = forward_compiled_successor_partitions(model, batch, (partition,))
    observed = {
        key: float(value.detach().exp())
        for key, value in zip(
            prediction.successor_keys[0],
            prediction.successor_log_probabilities[0],
            strict=True,
        )
    }
    production = canonical_successor_result(model, source, time)
    expected = {successor.key: successor.probability for successor in production.batch.successors}
    assert observed == pytest.approx(expected, abs=2e-6)
    semantic_system = resolve_successor_process_runtime(model).system
    reference = reference_successor_batch(
        source,
        [
            (mark.executor_rule_name, mark.action, mark.probability)
            for mark in production.marked_law.marks
        ],
        system=semantic_system,
        identity=production.batch.identity,
    )
    assert observed == pytest.approx(
        {successor.key: successor.probability for successor in reference.successors},
        abs=2e-6,
    )


def test_materialized_panel_row_selection_matches_fresh_collation(
    partition_model,
) -> None:
    from compose_v4.experiments import editing_v2_semantic_t1_capacity_runner as runner
    from compose_v4.experiments.factorized_mark_conditional import FactorizedMarkExample

    model = partition_model
    states = (_state("C"), _state("CC"), _state("c1ccccc1"))

    def example(state):
        return FactorizedMarkExample(
            state=state,
            time=0.5,
            teacher_action=None,
            teacher_rule_name=None,
            teacher_rate=1.0,
            importance_weight=1.0,
        )

    collator = runner._collator(model)
    full = collator([example(state) for state in states])
    selected = runner._index_factorized_batch(full, (2, 0, 2))
    direct = collator([example(states[index]) for index in (2, 0, 2)])
    _assert_row_selection_matches(selected, direct)
    with torch.no_grad():
        selected_node, selected_global, selected_pair = model._encode_batch(selected)
        direct_node, direct_global, direct_pair = model._encode_batch(direct)
    assert torch.equal(selected_node, direct_node)
    assert torch.equal(selected_global, direct_global)
    assert torch.equal(selected_pair, direct_pair)


def _assert_row_selection_matches(selected, direct) -> None:
    """Compare EVERY batch field, derived from the dataclass.

    This comparison used to enumerate the fields by hand, and the enumeration is
    what let `atom_delete_admission_mask` go unre-indexed: a hand-written list
    silently omits any field added after it was written, so the batch member
    that most needed checking was the one field not checked. Deriving the names
    from `dataclasses.fields` means a future field is covered on the day it is
    added rather than on the day someone remembers.
    """

    import dataclasses

    names = [item.name for item in dataclasses.fields(type(selected))]
    assert "atom_delete_admission_mask" in names
    for name in names:
        selected_value = getattr(selected, name)
        direct_value = getattr(direct, name)
        if selected_value is None or direct_value is None:
            assert selected_value is direct_value, name
        elif isinstance(selected_value, torch.Tensor):
            if isinstance(direct_value, bool):
                assert selected_value.dtype == torch.bool, name
                assert selected_value.shape[0] == selected.batch_size, name
                assert bool(torch.all(selected_value == direct_value)), name
            else:
                assert torch.equal(selected_value, direct_value), name
        else:
            assert selected_value == direct_value, name


def test_parallel_materialization_preserves_exact_panel_order_and_batch(
    partition_model,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from compose_v4.experiments import editing_v2_semantic_t1_capacity_runner as runner
    from compose_v4.experiments.factorized_mark_conditional import FactorizedMarkExample

    model = partition_model
    states = tuple(_state(smiles) for smiles in ("C", "CC", "CCC", "CO", "CCO", "CN"))
    entries = tuple(
        {
            "panel_entry_sha256": f"{index:064x}",
            "model_family": "atom_insert",
        }
        for index in range(len(states))
    )

    def example(state):
        return FactorizedMarkExample(
            state=state,
            time=0.5,
            teacher_action=None,
            teacher_rule_name=None,
            teacher_rate=1.0,
            importance_weight=1.0,
        )

    collator = runner._collator(model)
    direct = collator([example(state) for state in states])
    retained_collators = []

    def batch_for_ids(_runtime, _model, shard_collator, panel_ids):
        retained_collators.append(shard_collator)
        indices = tuple(int(identifier, 16) for identifier in panel_ids)
        selected_entries = tuple(entries[index] for index in indices)
        return (
            runner._index_factorized_batch(direct, indices),
            tuple(f"fiber-{index}" for index in indices),
            tuple(f"partition-{index}" for index in indices),
            selected_entries,
        )

    monkeypatch.setattr(runner, "MATERIALIZATION_WORKERS", 3)
    monkeypatch.setattr(runner, "_batch_for_ids", batch_for_ids)
    materialized = runner._materialize_panel(
        SimpleNamespace(entries=entries),
        model,
        collator,
    )

    assert materialized.panel_ids == tuple(entry["panel_entry_sha256"] for entry in entries)
    assert materialized.fibers == tuple(f"fiber-{index}" for index in range(len(entries)))
    assert materialized.partitions == tuple(f"partition-{index}" for index in range(len(entries)))
    assert len({id(item) for item in retained_collators}) == 3
    _assert_row_selection_matches(materialized.batch, direct)


def test_process_v2_row_selection_reindexes_the_admission_mask(
    process_v2_partition_model,
) -> None:
    """A non-identity selection of a Process-V2 batch must stay self-consistent.

    `_index_factorized_batch` rebuilds the batch with `dataclasses.replace`, so
    a field it does not name keeps the FULL-batch tensor. For the Process-V2
    admission mask that produced a batch whose delete mask had the selected rows
    and whose admission mask had all of them, and the forward guard then refused
    the batch outright -- the whole T1 capacity path was unable to run under
    Process V2 while every other test passed, because the V1 model that the
    other selection test uses carries `None` here and compares vacuously.
    """

    from compose_v4.experiments import editing_v2_semantic_t1_capacity_runner as runner
    from compose_v4.experiments.factorized_mark_conditional import FactorizedMarkExample

    model = process_v2_partition_model
    states = (_state("C1CCCCC1"), _state("CC1CCCCC1"), _state("CCO"))

    def example(state):
        return FactorizedMarkExample(
            state=state,
            time=0.5,
            teacher_action=None,
            teacher_rule_name=None,
            teacher_rate=1.0,
            importance_weight=1.0,
        )

    collator = runner._collator(model)
    full = collator([example(state) for state in states])
    # The premise: under Process V2 this field is a real tensor, not None.
    assert full.atom_delete_admission_mask is not None
    assert tuple(full.atom_delete_admission_mask.shape) == (3, full.atom_types.shape[1])

    for selection in ((2, 0), (0,), (2, 0, 2)):
        selected = runner._index_factorized_batch(full, selection)
        direct = collator([example(states[index]) for index in selection])
        assert selected.atom_delete_admission_mask is not None
        assert selected.atom_delete_admission_mask.shape[0] == len(selection), selection
        _assert_row_selection_matches(selected, direct)
        # The forward guard is what caught this; it must now pass.
        with torch.no_grad():
            model.forward_mark_batch(selected)


def test_gradient_route_diagnostics_retire_after_both_routes_are_observed() -> None:
    from compose_v4.experiments import editing_v2_semantic_t1_capacity_runner as runner

    evidence = {
        "atom_insert": {
            "family_route_finite_nonzero_seen": True,
            "action_route_finite_nonzero_seen": True,
        },
        "cycle_attach": {
            "family_route_finite_nonzero_seen": True,
            "action_route_finite_nonzero_seen": False,
        },
    }

    assert runner._pending_gradient_families(evidence) == frozenset({"cycle_attach"})
