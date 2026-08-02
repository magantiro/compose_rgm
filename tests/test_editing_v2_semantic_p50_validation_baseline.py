"""Physical source/cache/per-address validation baseline invariants."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import compose_v4.experiments.editing_v2_semantic_p50_validation_baseline as baseline
from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.editing_corpus_contract import ACTIVE8_FAMILIES
from compose_v4.data.editing_v2_semantic_capability_cells import (
    load_semantic_capability_cell_registry,
)
from compose_v4.data.packed_trace_store import PackedTraceAddress
from compose_v4.data.successor_fiber_cache import (
    SuccessorFiberCacheAddress,
    SuccessorFiberCacheRecord,
    successor_fiber_cache_record_payload,
)
from compose_v4.experiments.editing_p50_gate import state_dict_semantic_sha256
from compose_v4.experiments.editing_v2_semantic_p50_recipe_stream import (
    SemanticP50Candidate,
    SemanticP50CandidateInventory,
    SemanticP50Prerequisites,
    compile_semantic_p50_prepared_recipe,
    write_semantic_p50_prepared_recipe,
)
from compose_v4.experiments.editing_v2_semantic_development_cell_roles import (
    load_semantic_development_cell_roles,
)
from compose_v4.experiments.editing_v2_semantic_p50_source_inventory import (
    SemanticP50SourceInventoryBinding,
    VerifiedSemanticP50SourceInventory,
)
from compose_v4.experiments.editing_v2_semantic_p50_successor_cache import (
    SemanticP50SuccessorCache,
)
from compose_v4.experiments.editing_v2_semantic_p50_validation_baseline import (
    BASELINE_RESULT_FILENAME,
    INVENTORY_ROWS_FILENAME,
    SemanticP50ValidationBaselineError,
    SemanticP50ValidationEvaluation,
    build_semantic_p50_validation_evaluation_environment_receipt,
    materialize_semantic_p50_validation_baseline,
    materialize_semantic_p50_validation_inventory,
    open_semantic_p50_validation_baseline,
    open_semantic_p50_validation_baseline_from_paths,
    open_semantic_p50_validation_evaluation_environment_receipt,
    open_semantic_p50_validation_inventory,
)
from compose_v4.experiments.editing_v2_semantic_runtime import SemanticScratchRuntime
from compose_v4.experiments.factorized_successor_training import (
    StateProductiveSupport,
    TeacherSuccessorAlias,
    TeacherSuccessorFiber,
)
from compose_v4.rewrite.editing_v2_process_identity import editing_v2_process_identity
from compose_v4.rewrite.kernel import canonical_state_key


def _bytes(value: object, *, newline: bool = False) -> bytes:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return encoded + (b"\n" if newline else b"")


def _sha(value: object) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _state(smiles: str):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), 40)


class _Model:
    def __init__(self) -> None:
        self._state = {"weight": torch.arange(4, dtype=torch.float32)}

    def state_dict(self):
        return self._state


class _Path:
    def __init__(self, source, target) -> None:
        self._states = (source, target)

    def state_at(self, index: int):
        return self._states[index]


class _Index:
    def __init__(self, *, identity: dict[str, object], transitions: tuple[object, ...]) -> None:
        self._identity = identity
        self._traces = tuple(SimpleNamespace(transition=item) for item in transitions)

    def identity_payload(self):
        return self._identity

    def iter_accepted_traces_for_partition(self, partition: str):
        assert partition == "validation"
        return iter(self._traces)

    def accepted_transitions_for(self, trace):
        return (trace.transition,)


@pytest.fixture
def evidence(tmp_path: Path, monkeypatch):
    root = tmp_path / "artifacts"
    root.mkdir()
    registry = load_semantic_capability_cell_registry()
    process_sha = str(editing_v2_process_identity()["process_identity_sha256"])
    source_state = _state("C")
    target_state = _state("CC")
    source_key = canonical_state_key(source_state)
    target_key = canonical_state_key(target_state)
    transitions = []
    assignments: dict[int, object] = {}
    roles = load_semantic_development_cell_roles()
    for index, cell in enumerate(roles.required_cell_ids):
        family = cell.rsplit(":", 2)[1]
        address = PackedTraceAddress(
            packed_shard_content_sha256=_digest(f"validation-shard-{index}"),
            packed_shard_name=f"validation-{index}.jsonl.gz",
            entry_index=index,
            trace_id=f"validation-trace-{index}",
            layer="reversible_synthetic_walk",
            partition="validation",
            source_key=source_key,
            target_key=target_key,
            path_length=1,
        )
        transition = SimpleNamespace(
            step_index=0,
            addressed_trace=SimpleNamespace(
                address=address,
                path=_Path(source_state, target_state),
            ),
        )
        transitions.append(transition)
        assignments[id(transition)] = SimpleNamespace(
            model_family=family,
            capability_cell_id=cell,
            data_lane="reversible_synthetic_walk",
            assignment_sha256=_digest(f"assignment-{cell}"),
        )

    def classify(_index, transition, *, registry):
        assert registry is evidence_registry
        return assignments[id(transition)]

    evidence_registry = registry
    monkeypatch.setattr(baseline, "classify_verified_structural_transition", classify)

    model = _Model()
    initial_sha = state_dict_semantic_sha256(model.state_dict())
    fingerprint = "0123456789abcdef"
    scratch = SemanticScratchRuntime(
        model=model,
        config=SimpleNamespace(),
        architecture=SimpleNamespace(operator_capability_fingerprint=fingerprint),
        semantic_model_identity={},
        semantic_model_process_contract_sha256=_digest("semantic-contract"),
        process_identity_sha256=process_sha,
        initial_model_state_sha256=initial_sha,
    )
    source_identity = {
        "training_authorized": False,
        "gate_zero_authorized": False,
        "t1_authorized": False,
        "bounded_p50_authorized": False,
        "long_training_authorized": False,
        "checkpoint_selection_authorized": False,
        "final_test_selection_authorized": False,
        "inventory_sha256": _digest("source-inventory"),
    }
    source_path = root / "inputs" / "SEMANTIC_P50_SOURCE_INVENTORY.json"
    source_path.parent.mkdir()
    source_path.write_bytes(_bytes(source_identity, newline=True))
    source_binding = SemanticP50SourceInventoryBinding(
        source_inventory_file_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        source_inventory_sha256=source_identity["inventory_sha256"],
        process_identity_sha256=process_sha,
        model_runtime_identity_sha256=_digest("model-runtime"),
        active8_policy_sha256=_digest("active8-policy"),
        operator_capability_fingerprint=fingerprint,
        decision_source_implementation_sha256=_digest("decision-source"),
    )
    source = VerifiedSemanticP50SourceInventory(
        source_path,
        source_binding,
        _Index(identity=source_identity, transitions=tuple(transitions)),
    )
    values = {
        name: _digest(name)
        for name in SemanticP50Prerequisites.__dataclass_fields__
        if name != "operator_capability_fingerprint"
    }
    values.update(
        {
            "source_inventory_file_sha256": source_binding.source_inventory_file_sha256,
            "source_inventory_sha256": source_binding.source_inventory_sha256,
            "process_identity_sha256": process_sha,
            "model_runtime_identity_sha256": source_binding.model_runtime_identity_sha256,
            "active8_policy_sha256": source_binding.active8_policy_sha256,
            "operator_capability_fingerprint": fingerprint,
            "decision_source_implementation_sha256": (
                source_binding.decision_source_implementation_sha256
            ),
            "scratch_initial_model_state_sha256": initial_sha,
            "capability_registry_sha256": registry.registry_sha256,
            "classifier_implementation_sha256": (registry.classifier_implementation_sha256),
            "cell_role_policy_sha256": roles.policy_sha256,
        }
    )
    prerequisites = SemanticP50Prerequisites(**values)
    train = tuple(
        SemanticP50Candidate(
            address=SuccessorFiberCacheAddress(
                packed_shard_content_sha256=_digest(f"train-shard-{index}"),
                packed_shard_name=f"train-{index}.jsonl.gz",
                entry_index=index,
                layer="reversible_synthetic_walk",
                partition="train",
                trace_id=f"train-trace-{index}",
                trace_source_key=source_key,
                trace_target_key=target_key,
                progress_index=0,
                path_length=1,
            ),
            family=cell.rsplit(":", 2)[1],
            semantic_cell_id=cell,
            data_lane="reversible_synthetic_walk",
            assignment_sha256=_digest(f"train-assignment-{cell}"),
        )
        for index, cell in enumerate(roles.required_cell_ids)
    )
    prepared = compile_semantic_p50_prepared_recipe(
        SemanticP50CandidateInventory(prerequisites=prerequisites, candidates=train)
    )
    prepared_path = root / "inputs" / "prepared.json"
    write_semantic_p50_prepared_recipe(prepared_path, prepared)
    validation_items = []
    for transition in transitions:
        assignment = assignments[id(transition)]
        address = SuccessorFiberCacheAddress.from_packed_trace(
            transition.addressed_trace.address, progress_index=transition.step_index
        )
        support = StateProductiveSupport(
            source_key=source_key,
            source_state_sha256=persistent_slot_state_sha256(source_state),
            virtual_aliases=(),
        )
        teacher = TeacherSuccessorFiber(
            source_key=source_key,
            target_key=target_key,
            target_state_sha256=persistent_slot_state_sha256(target_state),
            aliases=(
                TeacherSuccessorAlias(
                    family_name=assignment.model_family,
                    table_name="fixture_table",
                    coordinate=(address.entry_index,),
                ),
            ),
            state_support=support,
        )
        record = SuccessorFiberCacheRecord(
            address=address,
            state_support=support,
            teacher_fiber=teacher,
        )
        candidate_body = {
            "address": baseline._address_payload(address),
            "family": assignment.model_family,
            "semantic_cell_id": assignment.capability_cell_id,
            "data_lane": assignment.data_lane,
            "assignment_sha256": assignment.assignment_sha256,
        }
        validation_items.append(
            (
                address,
                record,
                {**candidate_body, "candidate_sha256": _sha(candidate_body)},
            )
        )
    validation_items.sort(key=lambda item: item[0])
    records = tuple(item[1] for item in validation_items)
    candidate_rows = [item[2] for item in validation_items]
    contract_body = {
        "partition_role": "validation",
        "candidate_rows": candidate_rows,
        "candidate_inventory_sha256": _sha(candidate_rows),
        "candidate_count": len(candidate_rows),
        "required_families": list(ACTIVE8_FAMILIES),
        "complete_required_coverage": True,
    }
    validation_contract = {
        **contract_body,
        "validation_contract_sha256": _sha(contract_body),
    }
    validation_address_set_sha256 = _sha(
        sorted(_sha(baseline._address_payload(record.address)) for record in records)
    )
    manifest_body = {
        "validation_contract": validation_contract,
        "validation_requested_address_set_sha256": validation_address_set_sha256,
    }
    manifest = {**manifest_body, "manifest_sha256": _sha(manifest_body)}
    completion_body = {
        "manifest_sha256": manifest["manifest_sha256"],
        "validation_contract_sha256": validation_contract["validation_contract_sha256"],
        "validation_requested_address_set_sha256": validation_address_set_sha256,
    }
    cache_completion = {
        **completion_body,
        "completion_sha256": _sha(completion_body),
    }
    cache = SemanticP50SuccessorCache(
        completion=cache_completion,
        manifest=manifest,
        records=records,
        requested_records=records,
        train_requested_records=(),
        validation_requested_records=records,
        records_by_address_sha256={
            _sha(baseline._address_payload(record.address)): record for record in records
        },
    )
    inventory_path = materialize_semantic_p50_validation_inventory(
        prepared_recipe_path=prepared_path,
        source=source,
        scratch_runtime=scratch,
        successor_cache=cache,
        artifact_root=root,
        output_root=root / "validation-inventory",
        registry=registry,
    )
    inventory = open_semantic_p50_validation_inventory(
        inventory_path,
        prepared_recipe_path=prepared_path,
        source=source,
        scratch_runtime=scratch,
        successor_cache=cache,
        artifact_root=root,
        registry=registry,
    )
    evaluations = tuple(
        SemanticP50ValidationEvaluation(
            address=candidate.address,
            family=candidate.family,
            semantic_cell_id=candidate.semantic_cell_id,
            cache_record_sha256=_sha(successor_fiber_cache_record_payload(records[index])),
            time_hex=candidate.time_hex,
            canonical_successor_nll_nats=1.0 + index / 10.0,
        )
        for index, candidate in enumerate(inventory.candidates)
    )
    environment_body = {
        "hardware_class": "fixture-cpu",
        "device_name": "fixture-cpu",
        "device_capability": None,
        "accelerator_class": "cpu",
        "dtype": "float32",
        "mixed_precision": False,
        "batch_size": 1,
        "python_version": "fixture-python",
        "torch_version": "fixture-torch",
        "cuda_version": None,
        "cudnn_version": None,
        "rdkit_version": "fixture-rdkit",
        "numpy_version": "fixture-numpy",
    }
    environment = {
        **environment_body,
        "environment_sha256": _sha(environment_body),
    }
    receipt_payload = build_semantic_p50_validation_evaluation_environment_receipt(
        repo_root=Path.cwd(),
        execution_environment=environment,
        scratch_runtime=scratch,
        successor_cache=cache,
        evaluations=evaluations,
    )
    receipt_path = root / "inputs" / "evaluation-environment.json"
    receipt_path.write_bytes(_bytes(receipt_payload, newline=True))
    evaluation_environment_receipt = open_semantic_p50_validation_evaluation_environment_receipt(
        receipt_path,
        artifact_root=root,
        repo_root=Path.cwd(),
        scratch_runtime=scratch,
        successor_cache=cache,
        evaluations=evaluations,
    )
    return SimpleNamespace(
        root=root,
        registry=registry,
        source=source,
        scratch=scratch,
        prepared_path=prepared_path,
        inventory_path=inventory_path,
        inventory=inventory,
        cache=cache,
        evaluations=evaluations,
        evaluation_environment_receipt=evaluation_environment_receipt,
    )


def _baseline(evidence, *, evaluations=None):
    return materialize_semantic_p50_validation_baseline(
        inventory_completion_path=evidence.inventory_path,
        prepared_recipe_path=evidence.prepared_path,
        source=evidence.source,
        scratch_runtime=evidence.scratch,
        artifact_root=evidence.root,
        evaluations=evidence.evaluations if evaluations is None else evaluations,
        successor_cache=evidence.cache,
        evaluation_environment_receipt=evidence.evaluation_environment_receipt,
        repo_root=Path.cwd(),
        output_root=evidence.root / "validation-baseline",
        registry=evidence.registry,
    )


def _open_baseline(evidence, completion_path):
    return open_semantic_p50_validation_baseline(
        completion_path,
        inventory_completion_path=evidence.inventory_path,
        prepared_recipe_path=evidence.prepared_path,
        source=evidence.source,
        scratch_runtime=evidence.scratch,
        artifact_root=evidence.root,
        successor_cache=evidence.cache,
        evaluation_environment_receipt=evidence.evaluation_environment_receipt,
        repo_root=Path.cwd(),
        registry=evidence.registry,
    )


def test_inventory_is_exact_source_projection_and_nonauthorizing(evidence) -> None:
    verified = evidence.inventory
    assert len(verified.candidates) == len(load_semantic_development_cell_roles().required_cell_ids)
    assert {item.family for item in verified.candidates} == set(ACTIVE8_FAMILIES)
    assert all(item.address.partition == "validation" for item in verified.candidates)
    assert verified.completion["bounded_p50_authorized"] is False
    rows = (verified.completion_path.parent / INVENTORY_ROWS_FILENAME).read_text().splitlines()
    parsed = [json.loads(row) for row in rows]
    assert [row["stream_index"] for row in parsed] == list(range(len(rows)))
    assert [row["time_hex"] for row in parsed] == [
        baseline.semantic_p50_time_hex(
            stream_index=index,
            address=evidence.inventory.candidates[index].address,
        )
        for index in range(len(parsed))
    ]
    assert (
        evidence.inventory.completion["successor_cache_completion_sha256"]
        == evidence.cache.completion["completion_sha256"]
    )
    assert evidence.inventory.completion["ordered_validation_time_stream_sha256"]


def test_inventory_and_baseline_metrics_allow_extra_cells_but_keep_required_subset(
    evidence,
) -> None:
    first = evidence.inventory.candidates[0]
    extra_address = replace(
        first.address,
        packed_shard_content_sha256="f" * 64,
        packed_shard_name="zz-extra-validation.jsonl.gz",
        entry_index=999,
        trace_id="zz-extra-validation-trace",
        trace_source_key="zz-extra-source",
        trace_target_key="zz-extra-target",
    )
    extra = replace(
        first,
        address=extra_address,
        semantic_cell_id=f"extra:{first.family}:observed_only",
        time_hex=baseline._validation_time_hex(
            stream_index=len(evidence.inventory.candidates),
            address=extra_address,
        ),
    )
    candidates, _, _, cell_counts = baseline._inventory_rows(
        (*evidence.inventory.candidates, extra),
        required_cells=evidence.inventory.completion["required_semantic_cells"],
    )
    assert candidates[-1] == extra
    assert cell_counts[extra.semantic_cell_id] == 1

    first_record = evidence.cache.validation_requested_records[0]
    extra_record = SuccessorFiberCacheRecord(
        address=extra_address,
        state_support=first_record.state_support,
        teacher_fiber=first_record.teacher_fiber,
    )
    extra_evaluation = SemanticP50ValidationEvaluation(
        address=extra_address,
        family=extra.family,
        semantic_cell_id=extra.semantic_cell_id,
        cache_record_sha256=_sha(successor_fiber_cache_record_payload(extra_record)),
        time_hex=extra.time_hex,
        canonical_successor_nll_nats=0.75,
    )
    expanded_inventory = replace(
        evidence.inventory,
        completion={
            **evidence.inventory.completion,
            "semantic_cell_counts": cell_counts,
        },
        candidates=candidates,
    )
    cache_by_address = {
        record.address: record for record in evidence.cache.validation_requested_records
    }
    cache_by_address[extra_address] = extra_record
    _, _, cell_rows, _ = baseline._evaluation_rows_and_metrics(
        (*evidence.evaluations, extra_evaluation),
        inventory=expanded_inventory,
        cache_by_address=cache_by_address,
    )
    assert {row["semantic_cell_id"] for row in cell_rows} == set(cell_counts)


def test_inventory_opener_rejects_relabelled_source_candidate(evidence) -> None:
    rows_path = evidence.inventory_path.parent / INVENTORY_ROWS_FILENAME
    rows = rows_path.read_text().splitlines()
    row = json.loads(rows[0])
    row["target_key"] = "fabricated"
    body = dict(row)
    body.pop("row_sha256")
    row["row_sha256"] = _sha(body)
    rows[0] = _bytes(row).decode()
    rows_path.write_text("\n".join(rows) + "\n")
    with pytest.raises(SemanticP50ValidationBaselineError):
        open_semantic_p50_validation_inventory(
            evidence.inventory_path,
            prepared_recipe_path=evidence.prepared_path,
            source=evidence.source,
            scratch_runtime=evidence.scratch,
            successor_cache=evidence.cache,
            artifact_root=evidence.root,
            registry=evidence.registry,
        )


def test_inventory_rejects_source_outside_artifact_root(evidence, tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    outside.write_bytes(evidence.source.path.read_bytes())
    forged = VerifiedSemanticP50SourceInventory(
        outside, evidence.source.binding, evidence.source.index
    )
    with pytest.raises(SemanticP50ValidationBaselineError, match="outside artifact_root"):
        materialize_semantic_p50_validation_inventory(
            prepared_recipe_path=evidence.prepared_path,
            source=forged,
            scratch_runtime=evidence.scratch,
            successor_cache=evidence.cache,
            artifact_root=evidence.root,
            output_root=evidence.root / "other",
            registry=evidence.registry,
        )


def test_baseline_recomputes_metrics_from_per_address_receipts(evidence) -> None:
    completion = _baseline(evidence)
    verified = _open_baseline(evidence, completion)
    assert verified.result["example_count"] == len(
        load_semantic_development_cell_roles().required_cell_ids
    )
    first_family = ACTIVE8_FAMILIES[0]
    expected = next(
        item.canonical_successor_nll_nats
        for item in evidence.evaluations
        if item.family == first_family
    )
    assert verified.result["family_metrics"][0]["canonical_successor_nll_nats"] == expected
    assert completion.parent.name == verified.result["result_sha256"]
    assert verified.result["bounded_p50_authorized"] is False
    assert (
        verified.result["successor_cache_completion_sha256"]
        == (evidence.cache.completion["completion_sha256"])
    )
    assert verified.result["evaluator_implementation_sha256"] == (
        baseline.semantic_p50_validation_evaluator_implementation_sha256(repo_root=Path.cwd())
    )


def test_baseline_rejects_missing_or_duplicate_address_receipts(evidence) -> None:
    with pytest.raises(SemanticP50ValidationBaselineError, match="complete validation inventory"):
        _baseline(evidence, evaluations=evidence.evaluations[:-1])
    with pytest.raises(SemanticP50ValidationBaselineError, match="complete validation inventory"):
        _baseline(evidence, evaluations=(*evidence.evaluations, evidence.evaluations[0]))


def test_baseline_rejects_decorative_cache_record_hash(evidence) -> None:
    bad = list(evidence.evaluations)
    first = bad[0]
    bad[0] = SemanticP50ValidationEvaluation(
        address=first.address,
        family=first.family,
        semantic_cell_id=first.semantic_cell_id,
        cache_record_sha256=_digest("decorative-cache-hash"),
        time_hex=first.time_hex,
        canonical_successor_nll_nats=first.canonical_successor_nll_nats,
    )
    with pytest.raises(SemanticP50ValidationBaselineError, match="cache record"):
        _baseline(evidence, evaluations=tuple(bad))


def test_baseline_rejects_wrong_deterministic_time(evidence) -> None:
    bad = list(evidence.evaluations)
    first = bad[0]
    bad[0] = SemanticP50ValidationEvaluation(
        address=first.address,
        family=first.family,
        semantic_cell_id=first.semantic_cell_id,
        cache_record_sha256=first.cache_record_sha256,
        time_hex=evidence.evaluations[1].time_hex,
        canonical_successor_nll_nats=first.canonical_successor_nll_nats,
    )
    with pytest.raises(SemanticP50ValidationBaselineError, match="classification or cache"):
        _baseline(evidence, evaluations=tuple(bad))


def test_inventory_rejects_self_rehashed_cache_candidate_relabel(evidence) -> None:
    contract = dict(evidence.cache.manifest["validation_contract"])
    rows = [dict(row) for row in contract["candidate_rows"]]
    rows[0]["family"] = ACTIVE8_FAMILIES[1]
    row_body = dict(rows[0])
    row_body.pop("candidate_sha256")
    rows[0]["candidate_sha256"] = _sha(row_body)
    contract_body = dict(contract)
    contract_body.pop("validation_contract_sha256")
    contract_body["candidate_rows"] = rows
    contract_body["candidate_inventory_sha256"] = _sha(rows)
    forged_contract = {
        **contract_body,
        "validation_contract_sha256": _sha(contract_body),
    }
    manifest_body = dict(evidence.cache.manifest)
    manifest_body.pop("manifest_sha256")
    manifest_body["validation_contract"] = forged_contract
    forged_manifest = {**manifest_body, "manifest_sha256": _sha(manifest_body)}
    completion_body = dict(evidence.cache.completion)
    completion_body.pop("completion_sha256")
    completion_body["manifest_sha256"] = forged_manifest["manifest_sha256"]
    completion_body["validation_contract_sha256"] = forged_contract["validation_contract_sha256"]
    forged_cache = replace(
        evidence.cache,
        completion={**completion_body, "completion_sha256": _sha(completion_body)},
        manifest=forged_manifest,
    )
    with pytest.raises(
        SemanticP50ValidationBaselineError,
        match="authenticated validation contract",
    ):
        open_semantic_p50_validation_inventory(
            evidence.inventory_path,
            prepared_recipe_path=evidence.prepared_path,
            source=evidence.source,
            scratch_runtime=evidence.scratch,
            successor_cache=forged_cache,
            artifact_root=evidence.root,
            registry=evidence.registry,
        )


def test_evaluation_receipt_binds_scores_and_tracked_evaluator(evidence) -> None:
    first = evidence.evaluations[0]
    changed = (
        SemanticP50ValidationEvaluation(
            address=first.address,
            family=first.family,
            semantic_cell_id=first.semantic_cell_id,
            cache_record_sha256=first.cache_record_sha256,
            time_hex=first.time_hex,
            canonical_successor_nll_nats=first.canonical_successor_nll_nats + 0.25,
        ),
        *evidence.evaluations[1:],
    )
    with pytest.raises(
        SemanticP50ValidationBaselineError,
        match="producer, runtime, model, cache, or score identities",
    ):
        _baseline(evidence, evaluations=changed)

    receipt_path = evidence.evaluation_environment_receipt.path
    receipt = json.loads(receipt_path.read_bytes())
    receipt["producer_implementation_sha256"] = _digest("caller-selected-evaluator")
    receipt_body = dict(receipt)
    receipt_body.pop("receipt_sha256")
    receipt["receipt_sha256"] = _sha(receipt_body)
    receipt_path.write_bytes(_bytes(receipt, newline=True))
    with pytest.raises(
        SemanticP50ValidationBaselineError,
        match="producer, runtime, model, cache, or score identities",
    ):
        open_semantic_p50_validation_evaluation_environment_receipt(
            receipt_path,
            artifact_root=evidence.root,
            repo_root=Path.cwd(),
            scratch_runtime=evidence.scratch,
            successor_cache=evidence.cache,
            evaluations=evidence.evaluations,
        )


def test_baseline_rejects_incomplete_physical_cache_coverage(evidence) -> None:
    incomplete_cache = replace(
        evidence.cache,
        validation_requested_records=evidence.cache.validation_requested_records[:-1],
    )
    with pytest.raises(SemanticP50ValidationBaselineError, match="address set"):
        materialize_semantic_p50_validation_baseline(
            inventory_completion_path=evidence.inventory_path,
            prepared_recipe_path=evidence.prepared_path,
            source=evidence.source,
            scratch_runtime=evidence.scratch,
            artifact_root=evidence.root,
            evaluations=evidence.evaluations,
            successor_cache=incomplete_cache,
            evaluation_environment_receipt=evidence.evaluation_environment_receipt,
            repo_root=Path.cwd(),
            output_root=evidence.root / "validation-baseline-incomplete",
            registry=evidence.registry,
        )


def test_baseline_opener_rejects_rehashed_aggregate_tampering(evidence) -> None:
    completion = _baseline(evidence)
    result_path = completion.parent / BASELINE_RESULT_FILENAME
    result = json.loads(result_path.read_bytes())
    result["family_metrics"][0]["canonical_successor_nll_nats"] += 0.5
    body = dict(result)
    body.pop("result_sha256")
    result["result_sha256"] = _sha(body)
    forged_result_bytes = _bytes(result, newline=True)
    forged_root = completion.parent.parent / result["result_sha256"]
    forged_root.mkdir()
    (forged_root / BASELINE_RESULT_FILENAME).write_bytes(forged_result_bytes)
    forged_completion = json.loads(completion.read_bytes())
    forged_completion["result_file_sha256"] = hashlib.sha256(forged_result_bytes).hexdigest()
    forged_completion["result_sha256"] = result["result_sha256"]
    completion_body = dict(forged_completion)
    completion_body.pop("completion_sha256")
    forged_completion["completion_sha256"] = _sha(completion_body)
    forged_completion_path = forged_root / completion.name
    forged_completion_path.write_bytes(_bytes(forged_completion, newline=True))
    with pytest.raises(SemanticP50ValidationBaselineError, match="recomputed aggregates"):
        _open_baseline(evidence, forged_completion_path)


def test_baseline_path_opener_authenticates_environment_before_return(evidence) -> None:
    completion = _baseline(evidence)

    reopened = open_semantic_p50_validation_baseline_from_paths(
        completion,
        inventory_completion_path=evidence.inventory_path,
        evaluation_environment_receipt_path=(evidence.evaluation_environment_receipt.path),
        prepared_recipe_path=evidence.prepared_path,
        source=evidence.source,
        scratch_runtime=evidence.scratch,
        artifact_root=evidence.root,
        successor_cache=evidence.cache,
        repo_root=Path.cwd(),
        registry=evidence.registry,
    )

    assert reopened.completion_path == completion.resolve()


def test_baseline_rejects_nonfinite_nll_and_mutated_scratch(evidence) -> None:
    first = evidence.evaluations[0]
    with pytest.raises(SemanticP50ValidationBaselineError, match="finite nonnegative NLL"):
        SemanticP50ValidationEvaluation(
            address=first.address,
            family=first.family,
            semantic_cell_id=first.semantic_cell_id,
            cache_record_sha256=first.cache_record_sha256,
            time_hex=first.time_hex,
            canonical_successor_nll_nats=float("nan"),
        )
    evidence.scratch.model._state["weight"][0] = 99.0
    with pytest.raises(SemanticP50ValidationBaselineError, match="physical scratch runtime"):
        _baseline(evidence)
