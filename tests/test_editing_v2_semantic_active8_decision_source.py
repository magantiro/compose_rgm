"""Behavior and tamper tests for the semantic Active8 decision source index."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from compose_v4.chem.molecular_graph import (
    ELEMENT_TO_IDX,
    ORGANIC_VOCABULARY,
    smiles_to_molecular_graph,
)
from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data import (
    editing_v2_semantic_active8_decision_mapreduce as decision_mr,
)
from compose_v4.data import editing_v2_semantic_active8_decision_source as source_index
from compose_v4.data.editing_corpus_contract import (
    ACTIVE8_FAMILIES,
    REQUIRED_DATA_LANES,
    REQUIRED_PARTITION_ROLES,
)
from compose_v4.data.editing_v2_semantic_active8_admission import (
    ProductionSemanticExactCandidateChecker,
    SemanticActive8ActionDecision,
    SemanticActive8Exclusion,
    SemanticActive8TraceDecision,
)
from compose_v4.data.editing_v2_semantic_active8_source_adapter import (
    ACTIVE8_ADMISSION_STATUS,
    SOURCE_INVENTORY_STATUS,
    EditingV2SemanticActive8SourceInventory,
    SemanticActive8CountFlow,
    SemanticActive8Source,
    SemanticActive8SourceBinding,
)
from compose_v4.data.editing_v2_semantic_capability_cells import (
    SemanticCapabilityCellError,
    classify_verified_structural_transition,
    load_semantic_capability_cell_registry,
)
from compose_v4.data.semantic_active8_chunk_cache import (
    semantic_active8_chunk_cache_builder_identity,
)
from compose_v4.data.semantic_active8_chunk_cache_mapreduce import (
    GLOBAL_COMPLETION_FILENAME,
    execute_semantic_active8_chunk_cache_task,
    plan_semantic_active8_chunk_cache,
    reduce_semantic_active8_chunk_caches_with_witness,
    write_semantic_active8_chunk_cache_plan,
)
from compose_v4.data.semantic_packed_trace_store import (
    COMPLETION_FILENAME as SEMANTIC_COMPLETION_FILENAME,
)
from compose_v4.data.semantic_packed_trace_store import (
    MANIFEST_FILENAME,
    SHARD_FILENAME,
    write_semantic_packed_artifact,
)
from compose_v4.experiments import editing_v2_semantic_gate_zero as semantic_gate_zero
from compose_v4.experiments import (
    editing_v2_semantic_t1_panel_cache as semantic_t1_panel,
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
from compose_v4.rewrite.operators import (
    AtomDelete,
    AtomInsert,
    BondReorder,
    BondReroute,
    CycleCloseEdge,
    CycleOpenEdge,
    SemanticAtomRestate,
)
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace
from compose_v4.rewrite.trace_shard_v3 import encode_semantic_trace_record
from compose_v4.rewrite.tracelets import BondOrderChange, RingSystemRestate
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog


def _canonical(value: object, *, newline: bool = False) -> bytes:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return payload + (b"\n" if newline else b"")


def _value_sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state(smiles: str):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), 40)


def _record_from_step(
    trace_id: str,
    lane: str,
    role: str,
    source,
    step: RewriteStep,
) -> dict[str, object]:
    target = editing_v2_semantic_rewrite_system().apply(
        source,
        step.rule_name,
        step.action,
    )
    return encode_semantic_trace_record(
        RewriteTrace(source, target, (step,), {"fixture": trace_id}),
        trace_id=trace_id,
        data_lane=lane,
        split=role,
        source_address={"source_address_sha256": hashlib.sha256(trace_id.encode()).hexdigest()},
        lineage={"semantic_migration": "fixture"},
    )


def _record(trace_id: str, lane: str, role: str) -> dict[str, object]:
    return _record_from_step(
        trace_id,
        lane,
        role,
        _state("CCCCCC"),
        RewriteStep("cycle_close", CycleCloseEdge(0, 5, 1)),
    )


def _active8_train_records(
    lane: str,
    *,
    distinct_exact_sources: bool = False,
) -> tuple[dict[str, object], ...]:
    system = editing_v2_semantic_rewrite_system()
    carbon = _state("C")
    null = system.apply(carbon, "atom_delete", AtomDelete(0))
    root_birth = AtomInsert(
        slot=0,
        atom_type=int(carbon.atom_types[0]),
        formal_charge=int(carbon.formal_charges[0]),
        implicit_h_count=int(carbon.implicit_h_counts[0]),
        neighbors=(),
    )
    nitrogen_class = next(
        index
        for index, (element, _valence) in enumerate(ORGANIC_VOCABULARY.classes)
        if element == ELEMENT_TO_IDX["N"]
    )
    graft_source = _state("c1ccccc1CC")
    saturated_ring = _state("C1CCCCC1")
    atom_delete_source = _state("CCC") if distinct_exact_sources else _state("CC")
    atom_delete_slot = 2 if distinct_exact_sources else 1
    cycle_open_source = _state("C1CCCC1") if distinct_exact_sources else saturated_ring
    cycle_open_last_slot = 4 if distinct_exact_sources else 5
    cases = (
        ("atom_insert", null, RewriteStep("atom_insert", root_birth)),
        (
            "atom_delete",
            atom_delete_source,
            RewriteStep("atom_delete", AtomDelete(atom_delete_slot)),
        ),
        (
            "atom_restate",
            carbon,
            RewriteStep("atom_restate_semantic", SemanticAtomRestate(0, nitrogen_class)),
        ),
        (
            "bond_reorder",
            _state("CC"),
            RewriteStep("bond_reorder", BondReorder(0, 1, 2)),
        ),
        (
            "bond_reroute",
            graft_source,
            RewriteStep(
                "bond_reroute",
                BondReroute(a=7, b=6, u=7, v=0, new_order=1),
            ),
        ),
        (
            "cycle_insert",
            _state("CCCCCC"),
            RewriteStep("cycle_close", CycleCloseEdge(0, 5, 1)),
        ),
        (
            "cycle_attach",
            cycle_open_source,
            RewriteStep("cycle_open", CycleOpenEdge(0, cycle_open_last_slot)),
        ),
        (
            "ring_system_restate",
            saturated_ring,
            RewriteStep(
                "ring_system_restate",
                RingSystemRestate(
                    (
                        BondOrderChange(0, 1, 2),
                        BondOrderChange(2, 3, 2),
                        BondOrderChange(4, 5, 2),
                    )
                ),
            ),
        ),
    )
    return tuple(
        _record_from_step(f"active8-{family}", lane, "train", source, step)
        for family, source, step in cases
    )


def _inventory(
    root: Path,
    *,
    full_train_active8: bool = False,
    distinct_exact_train_sources: bool = False,
) -> EditingV2SemanticActive8SourceInventory:
    migration = root / "migration" / "SEMANTIC_MIGRATION_COMPLETE.json"
    migration.parent.mkdir(parents=True)
    migration.write_bytes(_canonical({"fixture": "migration"}, newline=True))
    migration_file_sha = _file_sha(migration)
    sources = []
    total_entries = 0
    total_states = 0
    total_actions = 0
    total_family_histogram: Counter[str] = Counter()
    for index, (lane, role) in enumerate(
        (lane, role) for lane in REQUIRED_DATA_LANES for role in REQUIRED_PARTITION_ROLES
    ):
        output = root / "sources" / lane / role
        semantic = output / "semantic"
        records = (
            _active8_train_records(
                lane,
                distinct_exact_sources=distinct_exact_train_sources,
            )
            if full_train_active8 and role == "train" and lane == REQUIRED_DATA_LANES[0]
            else (_record(f"trace-{index}", lane, role),)
        )
        record_count = len(records)
        binding = SemanticActive8SourceBinding(
            source_shard_name="legacy.jsonl.gz",
            source_shard_sha256=hashlib.sha256(f"source-{index}".encode()).hexdigest(),
            source_manifest_sha256=hashlib.sha256(f"manifest-{index}".encode()).hexdigest(),
            source_overlay_sha256=None,
            source_unified_manifest_sha256="d" * 64,
            source_entry_count=record_count,
        )
        write_semantic_packed_artifact(
            semantic,
            records,
            data_lane=lane,
            split=role,
            source_binding=binding.as_mapping(),
            decision_binding={
                "decision_ledger_sha256": hashlib.sha256(f"decision-{index}".encode()).hexdigest(),
                "source_count": record_count,
                "admitted_count": record_count,
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
                task_identity_sha256=hashlib.sha256(f"task-{index}".encode()).hexdigest(),
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
                semantic_completion_file_sha256=_file_sha(semantic / SEMANTIC_COMPLETION_FILENAME),
                semantic_completion_sha256=completion["completion_sha256"],
                semantic_record_stream_sha256=manifest["record_stream_sha256"],
                process_identity_sha256=manifest["process_identity_sha256"],
                action_codec_schema_version=manifest["action_codec_schema_version"],
                action_codec_implementation_hash=manifest["action_codec_implementation_hash"],
                source_binding=binding,
                family_histogram=tuple(manifest["family_histogram"].items()),
                counts=SemanticActive8CountFlow(
                    record_count,
                    record_count,
                    record_count,
                    0,
                    record_count,
                    record_count * 2,
                    record_count,
                ),
            )
        )
        total_entries += record_count
        total_states += record_count * 2
        total_actions += record_count
        total_family_histogram.update(manifest["family_histogram"])
    return EditingV2SemanticActive8SourceInventory(
        status=SOURCE_INVENTORY_STATUS,
        training_authorized=False,
        active8_admission_status=ACTIVE8_ADMISSION_STATUS,
        migration_completion_path=migration,
        migration_completion_file_sha256=migration_file_sha,
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
        family_histogram=tuple(sorted(total_family_histogram.items())),
        rejection_histogram=(),
        counts=SemanticActive8CountFlow(
            total_entries,
            total_entries,
            total_entries,
            0,
            total_entries,
            total_states,
            total_actions,
        ),
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
        hidden_dim=256,
        message_passing_steps=6,
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


def _runtime_identity(model: FactorizedTraceletRateModel) -> dict[str, object]:
    contract = semantic_gate_zero.load_semantic_gate_zero_structural_contract()
    root = Path(__file__).resolve().parents[1]
    decision_runtime = json.loads(
        (root / contract.payload["parents"]["decision_runtime"]["path"]).read_text()
    )
    semantic_model = json.loads(
        (root / contract.payload["parents"]["semantic_model_process"]["path"]).read_text()
    )
    parameter_digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        parameter_digest.update(name.encode("utf-8"))
        parameter_digest.update(str(array.dtype).encode("ascii"))
        parameter_digest.update(str(tuple(array.shape)).encode("ascii"))
        parameter_digest.update(array.tobytes())
    required = contract.payload["required_architecture"]
    body = {
        "schema": "compose.data.semantic_active8_exact_model_runtime",
        "schema_version": 2,
        "runtime_contract_sha256": decision_runtime["runtime_contract_sha256"],
        "semantic_model_process_contract_sha256": semantic_model["contract_sha256"],
        "semantic_model_identity": semantic_model["model_identity"],
        "process_identity_sha256": semantic_model["process_identity_sha256"],
        "architecture": {
            "max_atoms": required["max_atoms"],
            "hidden_dim": required["hidden_dim"],
            "message_passing_steps": required["message_passing_steps"],
            "mark_dim": required["mark_dim"],
            "dtype": required["dtype"],
            "parameter_dtypes": [required["dtype"]],
            "atom_vocabulary_class_count": required["atom_vocabulary_class_count"],
            "catalog_fingerprint": required["catalog_fingerprint"],
            "operator_capability_fingerprint": semantic_model["model_identity"][
                "operator_capability_fingerprint"
            ],
        },
        "initialization_seed": required["initialization_seed"],
        "initial_model_state_sha256": parameter_digest.hexdigest(),
        "software": decision_runtime["software"],
        "producer_source_revision_sha256": _value_sha(
            {"fixture": "semantic-active8-producer-source"}
        ),
        "execution_source_revision_sha256": _value_sha(
            {"fixture": "semantic-active8-execution-source"}
        ),
    }
    return {**body, "identity_sha256": _value_sha(body)}


def _build_completed_source(
    root: Path,
    *,
    full_train_active8: bool,
    inject_exclusion: bool,
    distinct_exact_train_sources: bool = False,
):
    root.mkdir()
    inventory = _inventory(
        root,
        full_train_active8=full_train_active8,
        distinct_exact_train_sources=distinct_exact_train_sources,
    )
    cache_plan = plan_semantic_active8_chunk_cache(
        inventory,
        source_revision=_revision(),
        output_artifact_root="/artifacts/cache",
        target_rows_per_chunk=1,
    )
    cache_plan_path = write_semantic_active8_chunk_cache_plan(
        cache_plan,
        artifact_root=root,
    )
    for task in cache_plan["tasks"]:
        execute_semantic_active8_chunk_cache_task(
            cache_plan,
            task["task_identity_sha256"],
            artifact_root=root,
        )
    cache_witness = reduce_semantic_active8_chunk_caches_with_witness(
        cache_plan,
        artifact_root=root,
    )
    cache_completion_path = (
        root
        / "cache"
        / "global_runs"
        / cache_plan["run_identity_sha256"]
        / GLOBAL_COMPLETION_FILENAME
    )
    model = _model()
    plan = decision_mr.plan_semantic_active8_decisions(
        inventory,
        chunk_cache_plan=cache_plan,
        chunk_cache_witness=cache_witness,
        model_runtime_identity=_runtime_identity(model),
        output_artifact_root="/artifacts/decisions",
        artifact_root=root,
    )
    decision_plan_path = decision_mr.write_semantic_active8_decision_plan(
        plan,
        artifact_root=root,
    )
    checker = ProductionSemanticExactCandidateChecker(model, cache_size=32)
    original = decision_mr.evaluate_semantic_active8_trace

    def one_exclusion(addressed, *, exact_candidate_checker, policy):
        decision = original(
            addressed,
            exact_candidate_checker=exact_candidate_checker,
            policy=policy,
        )
        if addressed.address.trace_id != "trace-0":
            return decision
        item = decision.action_decisions[0]
        assert item.candidate_evidence is not None
        evidence = replace(
            item.candidate_evidence,
            supported=False,
            matching_mark_count=1,
            successor_alias_count=0,
            exclusion_reason="teacher_action_does_not_reproduce_exact_successor",
        )
        exclusion = SemanticActive8Exclusion(
            stage="production_candidate_support",
            step_index=item.classification.step_index,
            executor_rule=item.classification.executor_rule,
            model_family=item.classification.model_family,
            reason="teacher_action_does_not_reproduce_exact_successor",
        )
        return SemanticActive8TraceDecision(
            trace_id=decision.trace_id,
            policy_sha256=decision.policy_sha256,
            semantic_migration_status="admitted",
            semantic_migration_rejection=None,
            active8_status="excluded",
            emits_progress_rows=False,
            action_decisions=(
                SemanticActive8ActionDecision(
                    classification=item.classification,
                    candidate_evidence=evidence,
                ),
            ),
            active8_exclusions=(exclusion,),
        )

    if inject_exclusion:
        decision_mr.evaluate_semantic_active8_trace = one_exclusion
    try:
        for task in plan["tasks"]:
            decision_mr.execute_semantic_active8_decision_task(
                plan,
                task["task_identity_sha256"],
                artifact_root=root,
                model=model,
                exact_candidate_checker=checker,
                model_runtime_identity_resolver=_runtime_identity,
            )
    finally:
        decision_mr.evaluate_semantic_active8_trace = original
    decision_mr.reduce_semantic_active8_decisions(
        plan,
        inventory=inventory,
        artifact_root=root,
    )
    decision_completion_path = (
        root / "decisions" / "runs" / plan["run_identity_sha256"] / decision_mr.COMPLETION_FILENAME
    )
    return {
        "root": root,
        "inventory": inventory,
        "cache_plan_path": cache_plan_path,
        "cache_completion_path": cache_completion_path,
        "decision_plan_path": decision_plan_path,
        "decision_completion_path": decision_completion_path,
    }


@pytest.fixture(scope="module")
def completed_source(tmp_path_factory: pytest.TempPathFactory):
    return _build_completed_source(
        tmp_path_factory.mktemp("semantic-active8-source") / "artifacts",
        full_train_active8=False,
        inject_exclusion=True,
    )


@pytest.fixture(scope="module")
def completed_active8_source(tmp_path_factory: pytest.TempPathFactory):
    return _build_completed_source(
        tmp_path_factory.mktemp("semantic-active8-full-source") / "artifacts",
        full_train_active8=True,
        inject_exclusion=False,
    )


@pytest.fixture(scope="module")
def completed_single_target_active8_source(
    tmp_path_factory: pytest.TempPathFactory,
):
    return _build_completed_source(
        tmp_path_factory.mktemp("semantic-active8-single-target-source") / "artifacts",
        full_train_active8=True,
        inject_exclusion=False,
        distinct_exact_train_sources=True,
    )


def _patch_migration_resolver(monkeypatch, inventory):
    def resolve(path, *, artifact_root, repo_root):
        assert Path(path).resolve() == inventory.migration_completion_path.resolve()
        assert Path(artifact_root).resolve() == inventory.migration_completion_path.parents[1]
        assert Path(repo_root).is_dir()
        return inventory

    monkeypatch.setattr(
        source_index,
        "resolve_editing_v2_semantic_active8_sources",
        resolve,
    )


def _resolve(completed, *, root=None, inventory=None):
    root = completed["root"] if root is None else Path(root)
    inventory = completed["inventory"] if inventory is None else inventory
    return source_index.resolve_editing_v2_semantic_active8_decision_source(
        migration_completion_path=inventory.migration_completion_path,
        chunk_cache_plan_path=root / completed["cache_plan_path"].relative_to(completed["root"]),
        chunk_cache_global_completion_path=root
        / completed["cache_completion_path"].relative_to(completed["root"]),
        decision_plan_path=root / completed["decision_plan_path"].relative_to(completed["root"]),
        decision_completion_path=root
        / completed["decision_completion_path"].relative_to(completed["root"]),
        artifact_root=root,
        repo_root=Path.cwd(),
    )


def test_index_streams_exact_accepted_progress_and_preserves_exclusions(
    completed_source,
    monkeypatch,
) -> None:
    inventory = completed_source["inventory"]
    _patch_migration_resolver(monkeypatch, inventory)
    index = _resolve(completed_source)
    assert index.count_map["traces"] == 20
    assert index.count_map["accepted_traces"] == 19
    assert index.count_map["excluded_traces"] == 1
    assert index.count_map["progress_rows"] == 38
    assert index.identity_payload()["training_authorized"] is False
    assert index.identity_payload()["gate_zero_authorized"] is False
    assert index.identity_payload()["t1_authorized"] is False
    assert index.identity_payload()["bounded_p50_authorized"] is False

    resolved = tuple(index.iter_resolved_traces())
    assert len(resolved) == 20
    assert sum(accepted for accepted, _ in resolved) == 19
    projected = tuple(
        transition
        for accepted_status, trace in resolved
        if accepted_status
        for transition in index.accepted_transitions_for(trace)
    )
    assert len(projected) == 19

    excluded = tuple(index.iter_excluded_traces())
    assert len(excluded) == 1
    assert excluded[0].addressed_trace.address.trace_id == "trace-0"
    assert tuple(item.reason for item in excluded[0].exclusions) == (
        "teacher_action_does_not_reproduce_exact_successor",
    )
    assert excluded[0].action_decisions[0].candidate_evidence is not None
    assert excluded[0].action_decisions[0].candidate_evidence.matching_mark_count == 1
    assert excluded[0].action_decisions[0].candidate_evidence.successor_alias_count == 0
    assert not index.is_accepted(excluded[0].addressed_trace.address)

    accepted = tuple(index.iter_accepted_traces())
    assert len(accepted) == 19
    assert all(index.is_accepted(item.addressed_trace.address) for item in accepted)
    assert all(item.decision.active8_status == "accepted" for item in accepted)

    original_loader = decision_mr._load_task_result

    def train_only_loader(plan, task, *, artifact_root):
        assert task["partition_role"] == "train"
        return original_loader(plan, task, artifact_root=artifact_root)

    monkeypatch.setattr(decision_mr, "_load_task_result", train_only_loader)
    train_accepted = tuple(index.iter_accepted_traces_for_partition("train"))
    assert len(train_accepted) == 4
    assert all(item.addressed_trace.address.partition == "train" for item in train_accepted)
    with pytest.raises(ValueError, match="partition role must be one of"):
        tuple(index.iter_accepted_traces_for_partition("outer_test"))
    monkeypatch.setattr(decision_mr, "_load_task_result", original_loader)

    assert all(
        index.decision_sha256_for(item.addressed_trace.address) == item.decision_sha256
        for item in accepted
    )
    changed_address = replace(
        accepted[0].addressed_trace.address,
        source_key="substituted-source-key",
    )
    with pytest.raises(
        source_index.SemanticActive8DecisionSourceError,
        match="trace identity differs",
    ):
        index.is_accepted(changed_address)
    progress = tuple(index.iter_accepted_progress_addresses())
    assert len(progress) == 38
    assert sum(item.terminal for item in progress) == 19
    assert len({item.key for item in progress}) == len(progress)
    transitions = tuple(index.iter_accepted_nonterminal_transitions())
    assert len(transitions) == 19
    assert tuple(
        (
            item.addressed_trace.address.trace_id,
            item.step_index,
            item.decision_sha256,
        )
        for item in projected
    ) == tuple(
        (
            item.addressed_trace.address.trace_id,
            item.step_index,
            item.decision_sha256,
        )
        for item in transitions
    )
    assert all(item.step_index == 0 for item in transitions)
    assert all(not item.source_progress_address.terminal for item in transitions)
    assert all(item.successor_progress_address.terminal for item in transitions)
    assert all(
        item.source_progress_address.state_sha256
        == persistent_slot_state_sha256(item.addressed_trace.path.state_at(0))
        for item in transitions
    )
    assert all(
        item.successor_progress_address.state_sha256
        == persistent_slot_state_sha256(item.addressed_trace.path.state_at(1))
        for item in transitions
    )
    assert all(
        item.decision_source_inventory_sha256 == index.inventory_sha256 for item in transitions
    )
    assignments = tuple(
        classify_verified_structural_transition(index, item) for item in transitions
    )
    assert len(assignments) == sum(len(item.action_decisions) for item in accepted)
    assert len(assignments) == (
        index.count_map["progress_rows"] - index.count_map["accepted_traces"]
    )
    assert all(item.matching_mark_count == 1 for item in assignments)
    assert all(item.progress_index == 0 for item in assignments)
    assert len({item.assignment_sha256 for item in assignments}) == len(assignments)
    structural_payload = assignments[0].as_payload()
    assert structural_payload["training_authorized"] is False
    assert structural_payload["gate_zero_authorized"] is False
    assert structural_payload["t1_authorized"] is False
    assert structural_payload["bounded_p50_authorized"] is False
    assert structural_payload["long_training_authorized"] is False
    assert structural_payload["checkpoint_selection_authorized"] is False
    assert structural_payload["final_test_selection_authorized"] is False
    assert "sampling_coefficients" not in structural_payload
    assert "mappings" not in structural_payload
    assert "relationship_group_ids" not in structural_payload
    wrong_process_registry = replace(
        load_semantic_capability_cell_registry(),
        process_identity_sha256="0" * 64,
    )
    with pytest.raises(
        SemanticCapabilityCellError,
        match="process or source identity",
    ):
        classify_verified_structural_transition(
            index,
            transitions[0],
            registry=wrong_process_registry,
        )
    for transition in transitions:
        index.validate_accepted_transition(transition)
    first_transition = transitions[0]
    forged_evidence = replace(
        first_transition.action_decision.candidate_evidence,
        raw_mark_count=(first_transition.action_decision.candidate_evidence.raw_mark_count + 1),
    )
    forged_action_decision = replace(
        first_transition.action_decision,
        candidate_evidence=forged_evidence,
    )
    forged_decision = replace(
        first_transition.decision,
        action_decisions=(forged_action_decision,),
    )
    with pytest.raises(
        source_index.SemanticActive8DecisionSourceError,
        match="indexed decision receipt",
    ):
        index.validate_accepted_transition(
            replace(
                first_transition,
                decision=forged_decision,
                action_decision=forged_action_decision,
            )
        )
    with pytest.raises(
        source_index.SemanticActive8DecisionSourceError,
        match="exact progress states",
    ):
        index.validate_accepted_transition(
            replace(
                first_transition,
                source_progress_address=replace(
                    first_transition.source_progress_address,
                    terminal=True,
                ),
            )
        )
    identity = index.identity_payload()
    assert identity["decision_source_implementation_sha256"] == _file_sha(
        Path(source_index.__file__)
    )
    assert identity["decision_lookup_inventory_sha256"] == index.decision_lookup_inventory_sha256
    assert identity["exclusion_lookup_inventory_sha256"] == index.exclusion_lookup_inventory_sha256
    with pytest.raises(ValueError, match="self-hash disagrees"):
        replace(index, inventory_sha256="f" * 64)

    decisions = dict(index._decisions_by_shard)
    shard_sha, entries = next(iter(decisions.items()))
    changed_entries = list(entries)
    trace_id, accepted_status, address_sha, decision_sha = changed_entries[0]
    changed_entries[0] = (
        trace_id,
        accepted_status,
        "0" * 64 if address_sha != "0" * 64 else "1" * 64,
        decision_sha,
    )
    decisions[shard_sha] = tuple(changed_entries)
    with pytest.raises(ValueError, match="decision lookup hash disagrees"):
        replace(index, _decisions_by_shard=decisions)

    decisions = dict(index._decisions_by_shard)
    shard_sha, entries = next(iter(decisions.items()))
    changed_entries = list(entries)
    trace_id, accepted_status, address_sha, decision_sha = changed_entries[0]
    changed_entries[0] = (
        trace_id,
        accepted_status,
        address_sha,
        "0" * 64 if decision_sha != "0" * 64 else "1" * 64,
    )
    decisions[shard_sha] = tuple(changed_entries)
    with pytest.raises(ValueError, match="decision lookup hash disagrees"):
        replace(index, _decisions_by_shard=decisions)

    exclusions = dict(index._exclusions_by_key)
    exclusion_key, exclusion_values = next(iter(exclusions.items()))
    exclusions[exclusion_key] = (replace(exclusion_values[0], reason="tampered_exclusion_reason"),)
    with pytest.raises(ValueError, match="exclusion lookup hash disagrees"):
        replace(index, _exclusions_by_key=exclusions)

    with pytest.raises(ValueError, match="cached decision plan"):
        replace(index, _decision_plan_bytes=b'{"tampered":true}\n')


def test_semantic_gate_zero_consumes_real_index_without_held_out_rescue(
    completed_source,
    monkeypatch,
) -> None:
    inventory = completed_source["inventory"]
    _patch_migration_resolver(monkeypatch, inventory)
    index = _resolve(completed_source)
    contract = semantic_gate_zero.load_semantic_gate_zero_structural_contract(repo_root=Path.cwd())
    evidence = semantic_gate_zero.build_semantic_gate_zero_structural_evidence(
        index,
        decision_plan_path=completed_source["decision_plan_path"],
        contract=contract,
        repo_root=Path.cwd(),
    )
    assert evidence["structural_result"] == "FAIL"
    assert evidence["counts"]["accepted_traces"] == 19
    assert evidence["counts"]["excluded_traces"] == 1
    assert evidence["counts"]["decision_eligible_accepted_traces"] == 4
    assert evidence["counts"]["decision_eligible_actions"] == 4
    assert evidence["counts"]["structural_assignments"] == 4
    assert evidence["decision_eligible_teacher_counts_by_family"]["cycle_insert"] == 4
    assert all(
        evidence["decision_eligible_teacher_counts_by_family"][family] == 0
        for family in ACTIVE8_FAMILIES
        if family != "cycle_insert"
    )
    assert set(evidence["sealed_nondecision_role_inventory_sha256"]) == {
        "controller_validation",
        "final_test",
        "validation",
    }
    decision = semantic_gate_zero.structural_decision_from_evidence(
        evidence,
        index=index,
        contract=contract,
        decision_plan_path=completed_source["decision_plan_path"],
        repo_root=Path.cwd(),
    )
    assert decision["structural_result"] == "FAIL"
    assert decision["next_authorized_stage"] is None


def test_real_gate_zero_pass_rejects_fixture_without_single_target_active8_coverage(
    completed_active8_source,
    monkeypatch,
) -> None:
    inventory = completed_active8_source["inventory"]
    _patch_migration_resolver(monkeypatch, inventory)
    index = _resolve(completed_active8_source)
    contract = semantic_gate_zero.load_semantic_gate_zero_structural_contract(repo_root=Path.cwd())
    output = completed_active8_source["root"] / "gate-zero-t1-integration"
    artifacts = semantic_gate_zero.run_semantic_gate_zero_structural_evidence(
        migration_completion_path=inventory.migration_completion_path,
        chunk_cache_plan_path=completed_active8_source["cache_plan_path"],
        chunk_cache_global_completion_path=completed_active8_source["cache_completion_path"],
        decision_plan_path=completed_active8_source["decision_plan_path"],
        decision_completion_path=completed_active8_source["decision_completion_path"],
        artifact_root=completed_active8_source["root"],
        repo_root=Path.cwd(),
        output_directory=output,
        contract_path=contract.source,
    )
    assert artifacts["evidence"]["structural_result"] == "PASS"
    runtime = artifacts["evidence"]["model_runtime_identity"]
    request = semantic_t1_panel.SemanticT1PanelRequest.create(
        request_id="real_typed_index_integration",
        source_revision_sha256=runtime["execution_source_revision_sha256"],
        support_time=0.5,
        maximum_entries_by_family={family: 2 for family in ACTIVE8_FAMILIES},
    )
    with pytest.raises(
        semantic_t1_panel.SemanticT1PanelError,
        match="no train stratum",
    ):
        semantic_t1_panel.prepare_editing_v2_semantic_t1_panel(
            index,
            gate_zero_artifacts=artifacts,
            decision_plan_path=completed_active8_source["decision_plan_path"],
            request=request,
            gate_zero_contract=contract,
            repo_root=Path.cwd(),
        )

    mismatched_plan = output / "MISMATCHED_PLAN.json"
    mismatched = json.loads(completed_active8_source["decision_plan_path"].read_text())
    mismatched["model_runtime_identity"]["execution_source_revision_sha256"] = "f" * 64
    mismatched_plan.write_bytes(_canonical(mismatched, newline=True))
    with pytest.raises(
        semantic_gate_zero.SemanticGateZeroStructuralError,
        match="decision plan differs from decision-source index|model runtime",
    ):
        semantic_t1_panel.prepare_editing_v2_semantic_t1_panel(
            index,
            gate_zero_artifacts=artifacts,
            decision_plan_path=mismatched_plan,
            request=request,
            gate_zero_contract=contract,
            repo_root=Path.cwd(),
        )


def test_real_gate_zero_pass_prepares_unique_panel_and_resolves_cache_traces(
    completed_single_target_active8_source,
    monkeypatch,
) -> None:
    completed = completed_single_target_active8_source
    inventory = completed["inventory"]
    _patch_migration_resolver(monkeypatch, inventory)
    index = _resolve(completed)
    contract = semantic_gate_zero.load_semantic_gate_zero_structural_contract(repo_root=Path.cwd())
    artifacts = semantic_gate_zero.run_semantic_gate_zero_structural_evidence(
        migration_completion_path=inventory.migration_completion_path,
        chunk_cache_plan_path=completed["cache_plan_path"],
        chunk_cache_global_completion_path=completed["cache_completion_path"],
        decision_plan_path=completed["decision_plan_path"],
        decision_completion_path=completed["decision_completion_path"],
        artifact_root=completed["root"],
        repo_root=Path.cwd(),
        output_directory=completed["root"] / "gate-zero-t1-single-target-integration",
        contract_path=contract.source,
    )
    assert artifacts["evidence"]["structural_result"] == "PASS"

    runtime = artifacts["evidence"]["model_runtime_identity"]
    request = semantic_t1_panel.SemanticT1PanelRequest.create(
        request_id="real_single_target_typed_index_integration",
        source_revision_sha256=runtime["execution_source_revision_sha256"],
        support_time=0.5,
        maximum_entries_by_family={family: 2 for family in ACTIVE8_FAMILIES},
    )
    panel = semantic_t1_panel.prepare_editing_v2_semantic_t1_panel(
        index,
        gate_zero_artifacts=artifacts,
        decision_plan_path=completed["decision_plan_path"],
        request=request,
        gate_zero_contract=contract,
        repo_root=Path.cwd(),
    )

    assert {entry.model_family for entry in panel.entries} == set(ACTIVE8_FAMILIES)
    assert panel.single_target_source_state_count == len(ACTIVE8_FAMILIES)
    assert panel.repeated_source_state_count == 0
    assert {
        panel_entry_sha256
        for cache_input in panel.cache_trace_inputs
        for panel_entry_sha256 in cache_input.panel_entry_sha256s
    } == {entry.panel_entry_sha256 for entry in panel.entries}

    resolved = semantic_t1_panel.resolve_semantic_t1_cache_trace_inputs(
        index,
        panel,
        repo_root=Path.cwd(),
    )
    expected_trace_keys = {
        (
            cache_input.packed_shard_content_sha256,
            cache_input.packed_entry_index,
            cache_input.trace_id,
        )
        for cache_input in panel.cache_trace_inputs
    }
    observed_trace_keys = {
        (
            trace.addressed_trace.address.packed_shard_content_sha256,
            trace.addressed_trace.address.entry_index,
            trace.addressed_trace.address.trace_id,
        )
        for trace in resolved
    }
    assert observed_trace_keys == expected_trace_keys
    assert len(resolved) == len(panel.cache_trace_inputs)
    assert all(index.is_accepted(trace.addressed_trace.address) for trace in resolved)


def test_candidate_evidence_rejects_wrong_successor_and_impossible_counts(
    completed_source,
    monkeypatch,
) -> None:
    inventory = completed_source["inventory"]
    _patch_migration_resolver(monkeypatch, inventory)
    plan = json.loads(completed_source["decision_plan_path"].read_text())
    addressed, original, _ = next(
        item
        for item in source_index._iter_plan_rows(
            plan,
            artifact_root=completed_source["root"],
        )
        if item[1]["active8_status"] == "accepted"
    )

    wrong_successor = copy.deepcopy(original)
    wrong_successor["actions"][0]["candidate_evidence"]["canonical_successor_key"] = (
        "definitely-not-the-exact-successor"
    )
    wrong_successor["decision_sha256"] = source_index._sha(
        {key: value for key, value in wrong_successor.items() if key != "decision_sha256"}
    )
    with pytest.raises(
        source_index.SemanticActive8DecisionSourceError,
        match="exact action or successor",
    ):
        source_index._typed_decision(wrong_successor, addressed)

    impossible_counts = copy.deepcopy(original)
    evidence = impossible_counts["actions"][0]["candidate_evidence"]
    evidence["raw_mark_count"] = 0
    evidence["canonical_successor_count"] = 999
    impossible_counts["decision_sha256"] = source_index._sha(
        {key: value for key, value in impossible_counts.items() if key != "decision_sha256"}
    )
    with pytest.raises(
        source_index.SemanticActive8DecisionSourceError,
        match="counts are internally inconsistent",
    ):
        source_index._typed_decision(impossible_counts, addressed)

    typed_original, _ = source_index._typed_decision(
        original,
        addressed,
    )
    unsupported_evidence = copy.deepcopy(original["actions"][0]["candidate_evidence"])
    unsupported_evidence.update(
        {
            "supported": False,
            "matching_mark_count": 1,
            "successor_alias_count": 0,
            "exclusion_reason": "teacher_action_does_not_reproduce_exact_successor",
        }
    )
    preserved = source_index._candidate_evidence(
        unsupported_evidence,
        classification=typed_original.action_decisions[0].classification,
        addressed=addressed,
    )
    assert preserved is not None
    assert preserved.supported is False
    assert preserved.matching_mark_count == 1
    assert preserved.successor_alias_count == 0


def test_decision_source_rejects_unbounded_chunk_plan(completed_source) -> None:
    plan = json.loads(completed_source["decision_plan_path"].read_text())
    plan["tasks"][0]["row_count"] = source_index.MAX_DECISION_SOURCE_CHUNK_ROWS + 1
    with pytest.raises(
        source_index.SemanticActive8DecisionSourceError,
        match="bounded source chunk size",
    ):
        next(
            iter(
                source_index._iter_plan_rows(
                    plan,
                    artifact_root=completed_source["root"],
                )
            )
        )


@pytest.mark.parametrize(
    ("limit_name", "expected_error"),
    (
        ("MAX_DECISION_SOURCE_CHUNK_ROWS", "bounded source chunk size"),
        (
            "MAX_DECISION_SOURCE_CHUNK_COMPRESSED_BYTES",
            "bounded compressed byte size",
        ),
        (
            "MAX_DECISION_SOURCE_CHUNK_UNCOMPRESSED_BYTES",
            "bounded uncompressed byte size",
        ),
    ),
)
def test_full_resolution_checks_chunk_bounds_before_reducer(
    completed_source,
    monkeypatch,
    limit_name: str,
    expected_error: str,
) -> None:
    inventory = completed_source["inventory"]
    _patch_migration_resolver(monkeypatch, inventory)
    reducer_called = False

    def forbidden_reducer(*args, **kwargs):
        nonlocal reducer_called
        reducer_called = True
        raise AssertionError("decision reducer ran before source bounds were checked")

    monkeypatch.setattr(
        decision_mr,
        "reduce_semantic_active8_decisions",
        forbidden_reducer,
    )
    monkeypatch.setattr(source_index, limit_name, 0)
    with pytest.raises(
        source_index.SemanticActive8DecisionSourceError,
        match=expected_error,
    ):
        _resolve(completed_source)
    assert reducer_called is False


def test_index_rejects_tampered_decision_bytes(
    completed_source,
    monkeypatch,
    tmp_path: Path,
) -> None:
    copied = tmp_path / "artifacts"
    shutil.copytree(completed_source["root"], copied)
    original_inventory = completed_source["inventory"]
    copied_migration = copied / original_inventory.migration_completion_path.relative_to(
        completed_source["root"]
    )
    inventory = replace(
        original_inventory,
        migration_completion_path=copied_migration,
    )
    _patch_migration_resolver(monkeypatch, inventory)
    decision_plan = json.loads(
        (
            copied / completed_source["decision_plan_path"].relative_to(completed_source["root"])
        ).read_text()
    )
    first = decision_plan["tasks"][0]
    decision_file = (
        copied
        / "decisions"
        / "runs"
        / decision_plan["run_identity_sha256"]
        / "tasks"
        / first["task_identity_sha256"]
        / decision_mr.DECISION_FILENAME
    )
    content = bytearray(decision_file.read_bytes())
    content[-1] ^= 1
    decision_file.write_bytes(bytes(content))
    with pytest.raises(
        source_index.SemanticActive8DecisionSourceError,
        match="failed revalidation",
    ):
        _resolve(completed_source, root=copied, inventory=inventory)


def test_missing_chunk_completion_object_is_not_repaired(
    completed_source,
    monkeypatch,
    tmp_path: Path,
) -> None:
    copied = tmp_path / "artifacts"
    shutil.copytree(completed_source["root"], copied)
    original_inventory = completed_source["inventory"]
    copied_migration = copied / original_inventory.migration_completion_path.relative_to(
        completed_source["root"]
    )
    inventory = replace(
        original_inventory,
        migration_completion_path=copied_migration,
    )
    _patch_migration_resolver(monkeypatch, inventory)
    pointer_path = copied / completed_source["cache_completion_path"].relative_to(
        completed_source["root"]
    )
    pointer = json.loads(pointer_path.read_text())
    completion_object = copied / "cache" / pointer["global_completion_object_path"]
    completion_object.unlink()
    with pytest.raises(
        source_index.SemanticActive8DecisionSourceError,
        match="global completion object is unreadable",
    ):
        _resolve(completed_source, root=copied, inventory=inventory)
    assert not completion_object.exists()


def test_repeated_resolution_is_identity_stable_and_does_not_republish(
    completed_source,
    monkeypatch,
) -> None:
    inventory = completed_source["inventory"]
    _patch_migration_resolver(monkeypatch, inventory)

    def snapshot() -> dict[str, tuple[int, int]]:
        return {
            str(path.relative_to(completed_source["root"])): (
                path.stat().st_mtime_ns,
                path.stat().st_size,
            )
            for path in completed_source["root"].rglob("*")
            if path.is_file()
        }

    before = snapshot()
    first = _resolve(completed_source)
    middle = snapshot()
    second = _resolve(completed_source)
    after = snapshot()
    assert before == middle == after
    assert first.inventory_sha256 == second.inventory_sha256
    assert first.identity_payload() == second.identity_payload()
