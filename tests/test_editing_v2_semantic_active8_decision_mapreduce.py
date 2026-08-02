"""Focused end-to-end tests for semantic Active8 decision map/reduce."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from compose_v4.data import semantic_active8_chunk_cache as chunk_cache_core
from compose_v4.data import semantic_active8_chunk_cache_mapreduce as chunk_cache_mr

from compose_v4.chem.molecular_graph import (
    ORGANIC_VOCABULARY,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.editing_corpus_contract import (
    REQUIRED_DATA_LANES,
    REQUIRED_PARTITION_ROLES,
)
from compose_v4.data.editing_v2_semantic_active8_admission import (
    ProductionSemanticExactCandidateChecker,
)
from compose_v4.data.editing_v2_semantic_active8_decision_mapreduce import (
    SemanticActive8DecisionIncomplete,
    SemanticActive8DecisionMapReduceError,
    completed_semantic_active8_decision_task_ids,
    execute_semantic_active8_decision_task,
    plan_semantic_active8_decisions,
    reduce_semantic_active8_decisions,
    write_semantic_active8_decision_plan,
)
from compose_v4.data.editing_v2_semantic_active8_source_adapter import (
    ACTIVE8_ADMISSION_STATUS,
    SOURCE_INVENTORY_STATUS,
    EditingV2SemanticActive8SourceInventory,
    SemanticActive8CountFlow,
    SemanticActive8Source,
    SemanticActive8SourceBinding,
)
from compose_v4.data.semantic_active8_chunk_cache import (
    semantic_active8_chunk_cache_builder_identity,
)
from compose_v4.data.semantic_active8_chunk_cache_mapreduce import (
    execute_semantic_active8_chunk_cache_task,
    plan_semantic_active8_chunk_cache,
    reduce_semantic_active8_chunk_caches_with_witness,
    write_semantic_active8_chunk_cache_plan,
)
from compose_v4.data.semantic_packed_trace_store import (
    COMPLETION_FILENAME as SEMANTIC_COMPLETION_FILENAME,
    MANIFEST_FILENAME,
    SHARD_FILENAME,
    write_semantic_packed_artifact,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    SEMANTIC_ATOM_RESTATE_ACTION_SEMANTICS,
    SEMANTIC_CYCLE_CLOSE_ACTION_SEMANTICS,
    SEMANTIC_CYCLE_OPEN_ACTION_SEMANTICS,
    SEMANTIC_EDITING_V2_PROCESS_SEMANTICS,
    SEMANTIC_RING_RESTATE_SCORER_MODE,
    FactorizedTraceletRateModel,
)
from compose_v4.rewrite.kernel import editing_v2_semantic_rewrite_system
from compose_v4.rewrite.operators import CycleCloseEdge
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace
from compose_v4.rewrite.trace_shard_v3 import encode_semantic_trace_record
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _value_sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(trace_id: str, lane: str, role: str) -> dict[str, object]:
    source = pad_molecular_graph(smiles_to_molecular_graph("CCCCCC"), 16)
    step = RewriteStep("cycle_close", CycleCloseEdge(0, 5, 1))
    target = editing_v2_semantic_rewrite_system().apply(
        source, step.rule_name, step.action
    )
    return encode_semantic_trace_record(
        RewriteTrace(source, target, (step,), {"fixture": trace_id}),
        trace_id=trace_id,
        data_lane=lane,
        split=role,
        source_address={
            "source_address_sha256": hashlib.sha256(trace_id.encode()).hexdigest()
        },
        lineage={"semantic_migration": "fixture"},
    )


def _inventory(root: Path) -> EditingV2SemanticActive8SourceInventory:
    sources = []
    for index, (lane, role) in enumerate(
        (lane, role)
        for lane in REQUIRED_DATA_LANES
        for role in REQUIRED_PARTITION_ROLES
    ):
        output = root / "sources" / lane / role
        semantic = output / "semantic"
        binding = SemanticActive8SourceBinding(
            source_shard_name="legacy.jsonl.gz",
            source_shard_sha256=hashlib.sha256(f"source-{index}".encode()).hexdigest(),
            source_manifest_sha256=hashlib.sha256(
                f"manifest-{index}".encode()
            ).hexdigest(),
            source_overlay_sha256=None,
            source_unified_manifest_sha256="d" * 64,
            source_entry_count=1,
        )
        write_semantic_packed_artifact(
            semantic,
            [_record(f"trace-{index}", lane, role)],
            data_lane=lane,
            split=role,
            source_binding=binding.as_mapping(),
            decision_binding={
                "decision_ledger_sha256": hashlib.sha256(
                    f"decision-{index}".encode()
                ).hexdigest(),
                "source_count": 1,
                "admitted_count": 1,
                "rejected_count": 0,
            },
        )
        manifest = json.loads((semantic / MANIFEST_FILENAME).read_text())
        completion = json.loads((semantic / SEMANTIC_COMPLETION_FILENAME).read_text())
        receipt = output / "RECEIPT.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text('{"fixture":true}\n')
        sources.append(
            SemanticActive8Source(
                data_lane=lane,
                partition_role=role,
                task_identity_sha256=hashlib.sha256(
                    f"task-{index}".encode()
                ).hexdigest(),
                task_output_artifact_path=f"/artifacts/sources/{lane}/{role}",
                task_output_directory=output,
                task_receipt_path=receipt,
                task_receipt_file_sha256=_file_sha(receipt),
                task_receipt_sha256="e" * 64,
                semantic_artifact_directory=semantic,
                semantic_shard_path=semantic / SHARD_FILENAME,
                semantic_shard_sha256=_file_sha(semantic / SHARD_FILENAME),
                semantic_manifest_path=semantic / MANIFEST_FILENAME,
                semantic_manifest_file_sha256=_file_sha(semantic / MANIFEST_FILENAME),
                semantic_manifest_sha256=manifest["manifest_sha256"],
                semantic_completion_path=semantic / SEMANTIC_COMPLETION_FILENAME,
                semantic_completion_file_sha256=_file_sha(
                    semantic / SEMANTIC_COMPLETION_FILENAME
                ),
                semantic_completion_sha256=completion["completion_sha256"],
                semantic_record_stream_sha256=manifest["record_stream_sha256"],
                process_identity_sha256=manifest["process_identity_sha256"],
                action_codec_schema_version=manifest["action_codec_schema_version"],
                action_codec_implementation_hash=manifest[
                    "action_codec_implementation_hash"
                ],
                source_binding=binding,
                family_histogram=tuple(manifest["family_histogram"].items()),
                counts=SemanticActive8CountFlow(1, 1, 1, 0, 1, 2, 1),
            )
        )
    return EditingV2SemanticActive8SourceInventory(
        status=SOURCE_INVENTORY_STATUS,
        training_authorized=False,
        active8_admission_status=ACTIVE8_ADMISSION_STATUS,
        migration_completion_path=root / "migration.json",
        migration_completion_file_sha256="0" * 64,
        migration_completion_sha256="1" * 64,
        migration_plan_path=root / "migration-plan.json",
        migration_plan_file_sha256="2" * 64,
        migration_plan_sha256="3" * 64,
        run_identity_sha256="4" * 64,
        source_revision_sha256="5" * 64,
        source_inventory_sha256="6" * 64,
        task_inventory_sha256="7" * 64,
        result_inventory_sha256="8" * 64,
        process_identity_sha256=sources[0].process_identity_sha256,
        builder_identity_sha256="9" * 64,
        candidate_materialization_manifest_sha256="a" * 64,
        candidate_provenance_source_stream_sha256="b" * 64,
        split_assignment_sha256="c" * 64,
        lane_registry_sha256="d" * 64,
        membership_receipt_file_sha256="e" * 64,
        membership_receipt_sha256="f" * 64,
        family_histogram=(("cycle_insert", 20),),
        rejection_histogram=(),
        counts=SemanticActive8CountFlow(20, 20, 20, 0, 20, 40, 20),
        sources=tuple(sources),
    )


def _revision() -> dict[str, object]:
    body = {
        "schema": "compose.data.semantic_active8_chunk_cache_modal_revision",
        "schema_version": 1,
        "commit": "a" * 40,
        "tree": "b" * 40,
        "worktree_clean": True,
        "serialized_sources": {"fixture.py": "c" * 64},
        "chunk_builder_identity": semantic_active8_chunk_cache_builder_identity(),
    }
    return {**body, "source_revision_sha256": _value_sha(body)}


def _model() -> FactorizedTraceletRateModel:
    torch.manual_seed(19)
    return FactorizedTraceletRateModel(
        build_typed_ring_catalog(()),
        hidden_dim=8,
        message_passing_steps=1,
        enable_ring_restates=True,
        enable_cyclic_graft=True,
        enable_heteroatom_scan=True,
        enable_ring_opening=True,
        enable_cycle_ops=True,
        editing_process_semantics=SEMANTIC_EDITING_V2_PROCESS_SEMANTICS,
        atom_restate_action_semantics=SEMANTIC_ATOM_RESTATE_ACTION_SEMANTICS,
        ring_restate_scorer_mode=SEMANTIC_RING_RESTATE_SCORER_MODE,
        cycle_close_action_semantics=SEMANTIC_CYCLE_CLOSE_ACTION_SEMANTICS,
        cycle_open_action_semantics=SEMANTIC_CYCLE_OPEN_ACTION_SEMANTICS,
        enable_ring_grow_macro=False,
        enable_ring_system_delete=False,
        atom_vocabulary=ORGANIC_VOCABULARY,
    ).eval()


def _runtime_identity(_model: FactorizedTraceletRateModel) -> dict[str, object]:
    body = {
        "schema": "fixture.model_runtime",
        "schema_version": 1,
        "fixture": "semantic-active8",
    }
    return {**body, "identity_sha256": _value_sha(body)}


@pytest.fixture()
def planned(tmp_path: Path):
    root = tmp_path / "artifacts"
    root.mkdir()
    inventory = _inventory(root)
    cache_plan = plan_semantic_active8_chunk_cache(
        inventory,
        source_revision=_revision(),
        output_artifact_root="/artifacts/cache",
        target_rows_per_chunk=1,
    )
    write_semantic_active8_chunk_cache_plan(cache_plan, artifact_root=root)
    for task in cache_plan["tasks"]:
        execute_semantic_active8_chunk_cache_task(
            cache_plan, task["task_identity_sha256"], artifact_root=root
        )
    cache_witness = reduce_semantic_active8_chunk_caches_with_witness(
        cache_plan,
        artifact_root=root,
    )
    plan = plan_semantic_active8_decisions(
        inventory,
        chunk_cache_plan=cache_plan,
        chunk_cache_witness=cache_witness,
        model_runtime_identity=_runtime_identity(_model()),
        output_artifact_root="/artifacts/decisions",
        artifact_root=root,
    )
    write_semantic_active8_decision_plan(plan, artifact_root=root)
    return root, inventory, cache_plan, cache_witness, plan


def test_planner_consumes_one_strict_witness_without_reducing_or_scanning_again(
    planned,
    monkeypatch,
) -> None:
    root, inventory, cache_plan, cache_witness, expected = planned

    def unexpected(*_args, **_kwargs):
        raise AssertionError("planner repeated a full chunk-cache validation")

    monkeypatch.setattr(
        chunk_cache_mr, "reduce_semantic_active8_chunk_caches", unexpected
    )
    monkeypatch.setattr(
        chunk_cache_core, "load_semantic_active8_chunk_cache", unexpected
    )
    observed = plan_semantic_active8_decisions(
        inventory,
        chunk_cache_plan=cache_plan,
        chunk_cache_witness=cache_witness,
        model_runtime_identity=_runtime_identity(_model()),
        output_artifact_root="/artifacts/decisions",
        artifact_root=root,
    )
    assert observed == expected


def test_planner_rejects_missing_tampered_or_misordered_witness(planned) -> None:
    root, inventory, cache_plan, cache_witness, _ = planned
    common = {
        "chunk_cache_plan": cache_plan,
        "model_runtime_identity": _runtime_identity(_model()),
        "output_artifact_root": "/artifacts/decisions",
        "artifact_root": root,
    }
    with pytest.raises(
        SemanticActive8DecisionMapReduceError,
        match="requires a strict chunk-cache reduction witness",
    ):
        plan_semantic_active8_decisions(
            inventory,
            chunk_cache_witness=None,  # type: ignore[arg-type]
            **common,
        )

    tampered_caches = deepcopy(cache_witness.validated_caches)
    tampered_caches[0]["chunk_inventory"][0]["row_count"] += 1
    tampered = replace(cache_witness, validated_caches=tuple(tampered_caches))
    with pytest.raises(
        SemanticActive8DecisionMapReduceError,
        match="witness cache 0 disagrees",
    ):
        plan_semantic_active8_decisions(
            inventory,
            chunk_cache_witness=tampered,
            **common,
        )

    misordered = replace(
        cache_witness,
        validated_caches=(
            cache_witness.validated_caches[1],
            cache_witness.validated_caches[0],
            *cache_witness.validated_caches[2:],
        ),
    )
    with pytest.raises(
        SemanticActive8DecisionMapReduceError,
        match="witness cache 0 disagrees",
    ):
        plan_semantic_active8_decisions(
            inventory,
            chunk_cache_witness=misordered,
            **common,
        )


def test_exact_chunk_tasks_reduce_to_non_authorizing_twenty_source_census(
    planned,
) -> None:
    root, inventory, _, _, plan = planned
    model = _model()
    checker = ProductionSemanticExactCandidateChecker(model, cache_size=32)
    for task in plan["tasks"]:
        execute_semantic_active8_decision_task(
            plan,
            task["task_identity_sha256"],
            artifact_root=root,
            model=model,
            exact_candidate_checker=checker,
            model_runtime_identity_resolver=_runtime_identity,
        )
    completion = reduce_semantic_active8_decisions(
        plan, inventory=inventory, artifact_root=root
    )
    assert completed_semantic_active8_decision_task_ids(
        plan, artifact_root=root
    ) == frozenset(task["task_identity_sha256"] for task in plan["tasks"])
    assert completion["active8_counts"]["traces"] == 20
    assert completion["active8_counts"]["actions"] == 20
    assert completion["active8_counts"]["accepted_traces"] == 20
    assert completion["active8_counts"]["progress_rows"] == 40
    assert completion["active8_action_family_histogram"] == {"cycle_insert": 20}
    assert completion["migration_rejection_census"] == {
        "rejected_traces": 0,
        "rejections_by_code": {},
        "rejected_traces_reevaluated_by_active8": 0,
    }
    assert completion["training_authorized"] is False
    assert completion["bounded_p50_authorized"] is False


def test_reducer_refuses_missing_task_and_worker_refuses_runtime_substitution(
    planned,
) -> None:
    root, inventory, _, _, plan = planned
    with pytest.raises(SemanticActive8DecisionIncomplete, match="is absent"):
        reduce_semantic_active8_decisions(plan, inventory=inventory, artifact_root=root)
    model = _model()
    checker = ProductionSemanticExactCandidateChecker(model)

    def wrong_runtime(_model: FactorizedTraceletRateModel) -> dict[str, object]:
        body = {
            "schema": "fixture.model_runtime",
            "schema_version": 1,
            "fixture": "wrong",
        }
        return {**body, "identity_sha256": _value_sha(body)}

    with pytest.raises(SemanticActive8DecisionMapReduceError, match="runtime identity"):
        execute_semantic_active8_decision_task(
            plan,
            plan["tasks"][0]["task_identity_sha256"],
            artifact_root=root,
            model=model,
            exact_candidate_checker=checker,
            model_runtime_identity_resolver=wrong_runtime,
        )


@pytest.mark.parametrize("missing_name", ["RECEIPT.json", "decisions.jsonl.gz"])
def test_worker_recovers_exact_partial_task_publication(
    planned, missing_name: str
) -> None:
    root, _, _, _, plan = planned
    task = plan["tasks"][0]
    model = _model()
    checker = ProductionSemanticExactCandidateChecker(model, cache_size=32)
    original = execute_semantic_active8_decision_task(
        plan,
        task["task_identity_sha256"],
        artifact_root=root,
        model=model,
        exact_candidate_checker=checker,
        model_runtime_identity_resolver=_runtime_identity,
    )
    task_root = (
        root
        / "decisions"
        / "runs"
        / plan["run_identity_sha256"]
        / "tasks"
        / task["task_identity_sha256"]
    )
    (task_root / missing_name).unlink()

    assert task[
        "task_identity_sha256"
    ] not in completed_semantic_active8_decision_task_ids(plan, artifact_root=root)
    recovered = execute_semantic_active8_decision_task(
        plan,
        task["task_identity_sha256"],
        artifact_root=root,
        model=model,
        exact_candidate_checker=checker,
        model_runtime_identity_resolver=_runtime_identity,
    )

    assert recovered == original
    assert task["task_identity_sha256"] in completed_semantic_active8_decision_task_ids(
        plan, artifact_root=root
    )


def test_abandoned_private_temporary_does_not_poison_completed_task(planned) -> None:
    root, _, _, _, plan = planned
    task = plan["tasks"][0]
    model = _model()
    checker = ProductionSemanticExactCandidateChecker(model, cache_size=32)
    execute_semantic_active8_decision_task(
        plan,
        task["task_identity_sha256"],
        artifact_root=root,
        model=model,
        exact_candidate_checker=checker,
        model_runtime_identity_resolver=_runtime_identity,
    )
    task_root = (
        root
        / "decisions"
        / "runs"
        / plan["run_identity_sha256"]
        / "tasks"
        / task["task_identity_sha256"]
    )
    (task_root / ".RECEIPT.json.abandoned.tmp").write_bytes(b"partial")

    assert task["task_identity_sha256"] in completed_semantic_active8_decision_task_ids(
        plan, artifact_root=root
    )
