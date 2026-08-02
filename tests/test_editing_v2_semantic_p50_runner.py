from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import pytest
import torch
from torch import nn

import compose_v4.experiments.editing_v2_semantic_p50_runner as runner_module
from compose_v4.data.editing_v2_semantic_capability_cells import ACTIVE8_FAMILIES
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
)
from compose_v4.experiments.editing_v2_semantic_development_cell_roles import (
    load_semantic_development_cell_roles,
)
from compose_v4.experiments.editing_v2_semantic_p50_runner import (
    SemanticP50RunnerError,
    SemanticP50RuntimeInputs,
    open_semantic_p50_run_artifacts,
    publish_semantic_p50_run_artifacts,
    run_semantic_p50,
    semantic_p50_validation_time_hex,
    validate_semantic_p50_checkpoint_payload,
)
from compose_v4.experiments.editing_v2_semantic_p50_successor_cache import (
    COMPLETION_SCHEMA,
    COMPLETION_STATUS,
    MANIFEST_SCHEMA,
    MANIFEST_STATUS,
    SCHEMA_VERSION as SUCCESSOR_CACHE_SCHEMA_VERSION,
    SemanticP50SuccessorCache,
)
from compose_v4.experiments.editing_v2_semantic_p50_validation_baseline import (
    SemanticP50ValidationBinding,
    SemanticP50ValidationEvaluation,
    VerifiedSemanticP50ValidationBaseline,
    VerifiedSemanticP50ValidationInventory,
)
from compose_v4.experiments.editing_v2_semantic_runtime import (
    SemanticScratchArchitecture,
    SemanticScratchModelConfig,
    SemanticScratchRuntime,
)
from compose_v4.experiments.factorized_successor_training import (
    StateProductiveSupport,
    TeacherSuccessorAlias,
    TeacherSuccessorFiber,
)


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


@dataclass(frozen=True)
class _State:
    digest: str


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.family_head = nn.Linear(1, 1, bias=False)
        for name in (
            "grow_root_head",
            "delete_head",
            "restate_head",
            "reorder_head",
            "graft_head",
            "cycle_close_head",
            "cycle_open_head",
            "ring_restate_head",
            "total_hazard_head",
        ):
            setattr(self, name, nn.Linear(1, 1, bias=False))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


@dataclass(frozen=True)
class _Batch:
    states: tuple[_State, ...]
    times: torch.Tensor
    teacher_rule_names: tuple[str, ...]
    teacher_rates: torch.Tensor
    importance_weights: torch.Tensor

    @property
    def batch_size(self) -> int:
        return len(self.states)

    def to(self, device: torch.device) -> _Batch:
        return _Batch(
            states=self.states,
            times=self.times.to(device),
            teacher_rule_names=self.teacher_rule_names,
            teacher_rates=self.teacher_rates.to(device),
            importance_weights=self.importance_weights.to(device),
        )


@dataclass(frozen=True)
class _Prediction:
    selected_productive_successor_log_probability: torch.Tensor
    total_hazard: torch.Tensor


_ACTION_MODULE = {
    "atom_insert": "grow_root_head",
    "atom_delete": "delete_head",
    "atom_restate": "restate_head",
    "bond_reorder": "reorder_head",
    "bond_reroute": "graft_head",
    "cycle_insert": "cycle_close_head",
    "cycle_attach": "cycle_open_head",
    "ring_system_restate": "ring_restate_head",
}


def _install_tiny_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    def collator(_model: nn.Module):
        def collate(examples):
            return _Batch(
                states=tuple(example.state for example in examples),
                times=torch.tensor([example.time for example in examples], dtype=torch.float32),
                teacher_rule_names=tuple(example.teacher_rule_name for example in examples),
                teacher_rates=torch.tensor(
                    [example.teacher_rate for example in examples], dtype=torch.float32
                ),
                importance_weights=torch.tensor(
                    [example.importance_weight for example in examples],
                    dtype=torch.float32,
                ),
            )

        return collate

    def forward(model: _TinyModel, batch: _Batch, fibers):
        assert len(fibers) == batch.batch_size
        values = []
        for family in batch.teacher_rule_names:
            score = model.family_head.weight.reshape(()) + getattr(
                model, _ACTION_MODULE[family]
            ).weight.reshape(())
            values.append(-torch.nn.functional.softplus(score))
        log_probability = torch.stack(values)
        return _Prediction(log_probability, torch.ones_like(log_probability))

    monkeypatch.setattr(runner_module, "_collator", collator)
    monkeypatch.setattr(runner_module, "forward_teacher_successor_batch", forward)
    monkeypatch.setattr(runner_module, "persistent_slot_state_sha256", lambda state: state.digest)


def _address(*, family_index: int, partition: str) -> SuccessorFiberCacheAddress:
    return SuccessorFiberCacheAddress(
        packed_shard_content_sha256=_digest(f"shard:{partition}"),
        packed_shard_name=f"{partition}.jsonl",
        entry_index=family_index,
        layer="reversible_synthetic_walk",
        partition=partition,
        trace_id=f"{partition}-trace-{family_index}",
        trace_source_key=f"{partition}-source-{family_index}",
        trace_target_key=f"{partition}-target-{family_index}",
        progress_index=0,
        path_length=1,
    )


def _record(
    address: SuccessorFiberCacheAddress, family: str
) -> tuple[SuccessorFiberCacheRecord, _State]:
    state = _State(_digest(f"state:{address.partition}:{family}"))
    support = StateProductiveSupport(
        source_key=address.trace_source_key,
        source_state_sha256=state.digest,
    )
    fiber = TeacherSuccessorFiber(
        source_key=address.trace_source_key,
        target_key=address.trace_target_key,
        target_state_sha256=_digest(f"target:{address.partition}:{family}"),
        aliases=(TeacherSuccessorAlias(family, "test_table", (0,)),),
        state_support=support,
    )
    return SuccessorFiberCacheRecord(address, support, fiber), state


def _scratch() -> SemanticScratchRuntime:
    torch.manual_seed(911)
    model = _TinyModel().to(dtype=torch.float32)
    initial = state_dict_semantic_sha256(model.state_dict())
    fingerprint = "f" * 16
    return SemanticScratchRuntime(
        model=model,
        config=SemanticScratchModelConfig(
            initialization_seed=1,
            max_atoms=40,
            hidden_dim=1,
            message_passing_steps=1,
            mark_dim=1,
            dtype="torch.float32",
            atom_vocabulary_class_count=15,
            catalog_fingerprint=fingerprint,
        ),
        architecture=SemanticScratchArchitecture(
            max_atoms=40,
            hidden_dim=1,
            message_passing_steps=1,
            mark_dim=1,
            dtype="torch.float32",
            parameter_dtypes=("torch.float32",),
            atom_vocabulary_class_count=15,
            catalog_fingerprint=fingerprint,
            operator_capability_fingerprint=fingerprint,
        ),
        semantic_model_identity={},
        semantic_model_process_contract_sha256=_digest("contract"),
        process_identity_sha256=_digest("process"),
        initial_model_state_sha256=initial,
    )


def _completion(*, prepared: dict, manifest: dict, train_count: int, validation_count: int) -> dict:
    prerequisite = prepared["prerequisites"]
    body = {
        "schema": COMPLETION_SCHEMA,
        "schema_version": SUCCESSOR_CACHE_SCHEMA_VERSION,
        "status": COMPLETION_STATUS,
        "training_authorized": False,
        "bounded_p50_authorized": False,
        "p500_authorized": False,
        "long_training_authorized": False,
        "checkpoint_selection_authorized": False,
        "final_test_selection_authorized": False,
        "run_identity_sha256": _digest("cache-run"),
        "build_identity_sha256": _digest("cache-build"),
        "source_revision_sha256": _digest("revision"),
        "execution_source_revision_sha256": _digest("execution-revision"),
        "implementation_sha256": _digest("cache-code"),
        "prepared_recipe_sha256": prepared["prepared_recipe_sha256"],
        "validation_contract_sha256": manifest["validation_contract"]["validation_contract_sha256"],
        "source_inventory_sha256": prerequisite["source_inventory_sha256"],
        "process_identity_sha256": prerequisite["process_identity_sha256"],
        "model_runtime_identity_sha256": prerequisite["model_runtime_identity_sha256"],
        "active8_policy_sha256": prerequisite["active8_policy_sha256"],
        "gate_zero_evidence_sha256": prerequisite["gate_zero_evidence_sha256"],
        "t1_decision_sha256": prerequisite["t1_decision_sha256"],
        "scratch_initial_model_state_sha256": prerequisite["scratch_initial_model_state_sha256"],
        "plan_artifact_path": "cache/plan.json",
        "plan_file_sha256": _digest("plan-file"),
        "plan_sha256": _digest("plan"),
        "manifest_artifact_path": "cache/manifest.json",
        "manifest_file_sha256": _digest("manifest-file"),
        "manifest_sha256": manifest["manifest_sha256"],
        "compiler_runtime_sha256": _digest("compiler-runtime"),
        "leaf_count": 1,
        "requested_address_count": train_count + validation_count,
        "requested_address_set_sha256": _digest("requested"),
        "closure_record_count": train_count + validation_count + 2,
        "closure_address_set_sha256": _digest("closure"),
        "train_requested_address_count": train_count,
        "train_closure_record_count": train_count + 1,
        "validation_requested_address_count": validation_count,
        "validation_closure_record_count": validation_count + 1,
        "train_requested_address_set_sha256": _digest("train-requested"),
        "train_closure_address_set_sha256": _digest("train-closure"),
        "validation_requested_address_set_sha256": _digest("validation-requested"),
        "validation_closure_address_set_sha256": _digest("validation-closure"),
        "stream_union_successor_cache_compiled": True,
        "training_launched": False,
        "next_stage_authorized": None,
    }
    return {**body, "completion_sha256": _sha(body)}


def _validation_baseline(
    *,
    prepared: dict,
    cache: SemanticP50SuccessorCache,
    scratch: SemanticScratchRuntime,
) -> VerifiedSemanticP50ValidationBaseline:
    prerequisites = prepared["prerequisites"]
    binding = SemanticP50ValidationBinding(
        process_identity_sha256=prerequisites["process_identity_sha256"],
        source_inventory_file_sha256=prerequisites["source_inventory_file_sha256"],
        source_inventory_sha256=prerequisites["source_inventory_sha256"],
        model_runtime_identity_sha256=prerequisites["model_runtime_identity_sha256"],
        active8_policy_sha256=prerequisites["active8_policy_sha256"],
        operator_capability_fingerprint=prerequisites["operator_capability_fingerprint"],
        decision_source_implementation_sha256=prerequisites[
            "decision_source_implementation_sha256"
        ],
        prepared_recipe_file_sha256=_digest("prepared-file"),
        prepared_recipe_sha256=prepared["prepared_recipe_sha256"],
        recipe_policy_file_sha256=_digest("policy-file"),
        recipe_policy_sha256=prepared["recipe_policy_sha256"],
        scratch_initial_model_state_sha256=scratch.initial_model_state_sha256,
        capability_registry_sha256=prerequisites["capability_registry_sha256"],
        classifier_implementation_sha256=prerequisites["classifier_implementation_sha256"],
    )
    records = {_sha(vars(record.address)): record for record in cache.validation_requested_records}
    evaluations: list[SemanticP50ValidationEvaluation] = []
    for index, candidate in enumerate(cache.manifest["validation_contract"]["candidate_rows"]):
        address = SuccessorFiberCacheAddress(**candidate["address"])
        record = records[_sha(vars(address))]
        family = candidate["family"]
        score = scratch.model.family_head.weight.reshape(()) + getattr(
            scratch.model, _ACTION_MODULE[family]
        ).weight.reshape(())
        evaluations.append(
            SemanticP50ValidationEvaluation(
                address=address,
                family=family,
                semantic_cell_id=candidate["semantic_cell_id"],
                cache_record_sha256=_sha(successor_fiber_cache_record_payload(record)),
                time_hex=semantic_p50_validation_time_hex(stream_index=index, address=address),
                canonical_successor_nll_nats=float(torch.nn.functional.softplus(score).detach()),
            )
        )
    result_body = {
        "evaluated_model_state_sha256": scratch.initial_model_state_sha256,
        "successor_cache_completion_sha256": cache.completion["completion_sha256"],
        "successor_cache_manifest_sha256": cache.manifest["manifest_sha256"],
        "successor_cache_validation_contract_sha256": cache.manifest["validation_contract"][
            "validation_contract_sha256"
        ],
        "evaluation_records": [item.as_payload() for item in evaluations],
    }
    result = {**result_body, "result_sha256": _sha(result_body)}
    completion_body = {
        "result_sha256": result["result_sha256"],
        "successor_cache_completion_sha256": cache.completion["completion_sha256"],
    }
    completion = {
        **completion_body,
        "completion_sha256": _sha(completion_body),
    }
    inventory = VerifiedSemanticP50ValidationInventory(
        completion_path=Path("validation-inventory"),
        completion={},
        binding=binding,
        candidates=(),
    )
    return VerifiedSemanticP50ValidationBaseline(
        completion_path=Path("validation-baseline"),
        completion=completion,
        result=result,
        inventory=inventory,
    )


def _runtime_inputs() -> SemanticP50RuntimeInputs:
    scratch = _scratch()
    shas = {
        name: _digest(name)
        for name in (
            "source_inventory_file",
            "source_inventory",
            "model_runtime",
            "active8_policy",
            "decision_source",
            "gate_file",
            "gate",
            "t1_file",
            "t1",
            "t1_selected",
            "registry",
            "classifier",
        )
    }
    prerequisites = SemanticP50Prerequisites(
        source_inventory_file_sha256=shas["source_inventory_file"],
        source_inventory_sha256=shas["source_inventory"],
        process_identity_sha256=scratch.process_identity_sha256,
        model_runtime_identity_sha256=shas["model_runtime"],
        active8_policy_sha256=shas["active8_policy"],
        operator_capability_fingerprint=scratch.architecture.operator_capability_fingerprint,
        decision_source_implementation_sha256=shas["decision_source"],
        gate_zero_evidence_file_sha256=shas["gate_file"],
        gate_zero_evidence_sha256=shas["gate"],
        t1_decision_file_sha256=shas["t1_file"],
        t1_decision_sha256=shas["t1"],
        t1_selected_model_state_sha256=shas["t1_selected"],
        scratch_initial_model_state_sha256=scratch.initial_model_state_sha256,
        capability_registry_sha256=shas["registry"],
        classifier_implementation_sha256=shas["classifier"],
        cell_role_policy_sha256=load_semantic_development_cell_roles().policy_sha256,
    )
    required_cells = load_semantic_development_cell_roles().required_cell_ids
    train_candidates = tuple(
        SemanticP50Candidate(
            address=_address(family_index=index, partition="train"),
            family=cell.rsplit(":", 2)[1],
            semantic_cell_id=cell,
            data_lane="reversible_synthetic_walk",
            assignment_sha256=_digest(f"assignment:train:{cell}"),
        )
        for index, cell in enumerate(required_cells)
    )
    prepared = compile_semantic_p50_prepared_recipe(
        SemanticP50CandidateInventory(prerequisites, train_candidates)
    )

    train_records: list[SuccessorFiberCacheRecord] = []
    validation_records: list[SuccessorFiberCacheRecord] = []
    states: dict[str, _State] = {}
    validation_candidates: list[dict] = []
    for index, cell in enumerate(required_cells):
        family = cell.rsplit(":", 2)[1]
        train_record, train_state = _record(train_candidates[index].address, family)
        validation_address = _address(family_index=index, partition="validation")
        validation_record, validation_state = _record(validation_address, family)
        train_records.append(train_record)
        validation_records.append(validation_record)
        states[_sha(vars(train_record.address))] = train_state
        states[_sha(vars(validation_record.address))] = validation_state
        candidate_body = {
            "address": vars(validation_address),
            "family": family,
            "semantic_cell_id": cell,
            "data_lane": "reversible_synthetic_walk",
            "assignment_sha256": _digest(f"assignment:validation:{cell}"),
        }
        validation_candidates.append({**candidate_body, "candidate_sha256": _sha(candidate_body)})
    validation_body = {"candidate_rows": validation_candidates}
    validation_contract = {
        **validation_body,
        "validation_contract_sha256": _sha(validation_body),
    }
    manifest_body = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": 1,
        "status": MANIFEST_STATUS,
        "prepared_recipe_binding": {"prepared_recipe_sha256": prepared["prepared_recipe_sha256"]},
        "validation_contract": validation_contract,
    }
    manifest = {**manifest_body, "manifest_sha256": _sha(manifest_body)}
    completion = _completion(
        prepared=prepared,
        manifest=manifest,
        train_count=len(train_records),
        validation_count=len(validation_records),
    )
    all_records = (*train_records, *validation_records)
    by_address = {_sha(vars(record.address)): record for record in all_records}
    cache = SemanticP50SuccessorCache(
        completion=MappingProxyType(completion),
        manifest=MappingProxyType(manifest),
        records=all_records,
        requested_records=all_records,
        train_requested_records=tuple(train_records),
        validation_requested_records=tuple(validation_records),
        records_by_address_sha256=MappingProxyType(by_address),
    )
    return SemanticP50RuntimeInputs(
        prepared_recipe=prepared,
        cache=cache,
        scratch_runtime=scratch,
        validation_baseline=_validation_baseline(prepared=prepared, cache=cache, scratch=scratch),
        states_by_address_sha256=MappingProxyType(states),
    )


def _rehashed_prepared(prepared: dict, mutate) -> dict:
    changed = copy.deepcopy(prepared)
    mutate(changed)
    for row in changed["ordered_stream_rows"]:
        row.pop("row_sha256", None)
        row["row_sha256"] = _sha(row)
    changed["ordered_training_stream_sha256"] = _sha(changed["ordered_stream_rows"])
    changed.pop("prepared_recipe_sha256")
    changed["prepared_recipe_sha256"] = _sha(changed)
    return changed


def test_exact_runner_executes_50_updates_and_returns_reducible_nlls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_tiny_backend(monkeypatch)
    inputs = _runtime_inputs()

    artifacts = run_semantic_p50(inputs)

    assert artifacts.result["completed_optimizer_steps"] == 50
    assert artifacts.result["scheduled_example_count"] == 3200
    assert len(artifacts.result["trajectory"]) == 50
    required_count = len(load_semantic_development_cell_roles().required_cell_ids)
    assert len(artifacts.scratch_validation_evaluations) == required_count
    assert len(artifacts.final_validation_evaluations) == required_count
    assert all(
        item.canonical_successor_nll_nats >= 0.0
        for item in artifacts.scratch_validation_evaluations
    )
    assert all(
        evidence["finite_nonzero_gradient_update_count"]
        >= evidence["minimum_required_finite_nonzero_gradient_update_count"]
        for evidence in artifacts.result["semantic_cell_exposure_and_gradient_evidence"].values()
    )
    assert (
        artifacts.checkpoint["hazard_initial_state_sha256"]
        == artifacts.checkpoint["hazard_final_state_sha256"]
    )
    assert validate_semantic_p50_checkpoint_payload(artifacts.checkpoint) == artifacts.checkpoint


@pytest.mark.parametrize(
    "mutate",
    [
        lambda prepared: prepared["recipe"].__setitem__("optimizer_steps", 49),
        lambda prepared: prepared["ordered_stream_rows"][0].__setitem__("time_hex", (0.5).hex()),
        lambda prepared: prepared["ordered_stream_rows"][0].__setitem__(
            "identity_coefficient", 0.5
        ),
    ],
)
def test_runner_rejects_rehashed_recipe_semantic_drift(
    monkeypatch: pytest.MonkeyPatch, mutate
) -> None:
    _install_tiny_backend(monkeypatch)
    inputs = _runtime_inputs()
    prepared = _rehashed_prepared(dict(inputs.prepared_recipe), mutate)
    changed = SemanticP50RuntimeInputs(
        prepared_recipe=prepared,
        cache=inputs.cache,
        scratch_runtime=inputs.scratch_runtime,
        validation_baseline=inputs.validation_baseline,
        states_by_address_sha256=inputs.states_by_address_sha256,
    )

    with pytest.raises(SemanticP50RunnerError):
        run_semantic_p50(changed)


def test_runner_rejects_missing_combined_cache_fiber(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_tiny_backend(monkeypatch)
    inputs = _runtime_inputs()
    missing = tuple(inputs.cache.validation_requested_records[1:])
    cache = SemanticP50SuccessorCache(
        completion=inputs.cache.completion,
        manifest=inputs.cache.manifest,
        records=inputs.cache.records,
        requested_records=inputs.cache.requested_records,
        train_requested_records=inputs.cache.train_requested_records,
        validation_requested_records=missing,
        records_by_address_sha256=inputs.cache.records_by_address_sha256,
    )

    with pytest.raises(SemanticP50RunnerError, match="requested union"):
        run_semantic_p50(
            SemanticP50RuntimeInputs(
                prepared_recipe=inputs.prepared_recipe,
                cache=cache,
                scratch_runtime=inputs.scratch_runtime,
                validation_baseline=inputs.validation_baseline,
                states_by_address_sha256=inputs.states_by_address_sha256,
            )
        )


def test_runner_rolls_back_model_on_zero_route_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_tiny_backend(monkeypatch)
    inputs = _runtime_inputs()
    initial = state_dict_semantic_sha256(inputs.scratch_runtime.model.state_dict())
    original = runner_module._route_gradient_evidence

    def zero_one_cell(model, prediction, rows, *, group_field):
        observed = original(model, prediction, rows, group_field=group_field)
        if group_field == "semantic_cell_id":
            first = next(iter(observed))
            observed[first]["action_route_finite_nonzero"] = False
        return observed

    monkeypatch.setattr(runner_module, "_route_gradient_evidence", zero_one_cell)

    with pytest.raises(SemanticP50RunnerError, match="zero or nonfinite"):
        run_semantic_p50(inputs)
    assert state_dict_semantic_sha256(inputs.scratch_runtime.model.state_dict()) == initial


def test_runner_fails_closed_on_required_cell_validation_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_tiny_backend(monkeypatch)
    inputs = _runtime_inputs()
    initial = state_dict_semantic_sha256(inputs.scratch_runtime.model.state_dict())
    scoring = runner_module.forward_teacher_successor_batch
    call_count = 0

    def regress_final(model, batch, fibers):
        nonlocal call_count
        call_count += 1
        prediction = scoring(model, batch, fibers)
        if call_count == 52:
            return _Prediction(
                prediction.selected_productive_successor_log_probability - 1.0,
                prediction.total_hazard,
            )
        return prediction

    monkeypatch.setattr(runner_module, "forward_teacher_successor_batch", regress_final)

    with pytest.raises(SemanticP50RunnerError, match="validation NLL non-increase"):
        run_semantic_p50(inputs)
    assert call_count == 52
    assert state_dict_semantic_sha256(inputs.scratch_runtime.model.state_dict()) == initial


def test_runner_evaluates_but_does_not_gate_extra_validation_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_tiny_backend(monkeypatch)
    inputs = _runtime_inputs()
    family = ACTIVE8_FAMILIES[0]
    address = _address(family_index=99, partition="validation")
    record, state = _record(address, family)
    candidate_body = {
        "address": vars(address),
        "family": family,
        "semantic_cell_id": f"extra:{family}:observed_only",
        "data_lane": address.layer,
        "assignment_sha256": _digest("extra-validation-assignment"),
    }
    candidate = {**candidate_body, "candidate_sha256": _sha(candidate_body)}

    manifest = copy.deepcopy(dict(inputs.cache.manifest))
    validation = dict(manifest["validation_contract"])
    validation["candidate_rows"] = [*validation["candidate_rows"], candidate]
    validation.pop("validation_contract_sha256")
    validation["validation_contract_sha256"] = _sha(
        {"candidate_rows": validation["candidate_rows"]}
    )
    manifest["validation_contract"] = validation
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = _sha(manifest)

    completion = copy.deepcopy(dict(inputs.cache.completion))
    completion["manifest_sha256"] = manifest["manifest_sha256"]
    completion["validation_contract_sha256"] = validation["validation_contract_sha256"]
    completion["requested_address_count"] += 1
    completion["closure_record_count"] += 1
    completion["validation_requested_address_count"] += 1
    completion["validation_closure_record_count"] += 1
    completion.pop("completion_sha256")
    completion["completion_sha256"] = _sha(completion)
    by_address = dict(inputs.cache.records_by_address_sha256)
    by_address[_sha(vars(address))] = record
    states = dict(inputs.states_by_address_sha256)
    states[_sha(vars(address))] = state
    cache = SemanticP50SuccessorCache(
        completion=MappingProxyType(completion),
        manifest=MappingProxyType(manifest),
        records=(*inputs.cache.records, record),
        requested_records=(*inputs.cache.requested_records, record),
        train_requested_records=inputs.cache.train_requested_records,
        validation_requested_records=(
            *inputs.cache.validation_requested_records,
            record,
        ),
        records_by_address_sha256=MappingProxyType(by_address),
    )

    artifacts = run_semantic_p50(
        SemanticP50RuntimeInputs(
            prepared_recipe=inputs.prepared_recipe,
            cache=cache,
            scratch_runtime=inputs.scratch_runtime,
            validation_baseline=_validation_baseline(
                prepared=dict(inputs.prepared_recipe),
                cache=cache,
                scratch=inputs.scratch_runtime,
            ),
            states_by_address_sha256=MappingProxyType(states),
        )
    )

    required_count = len(load_semantic_development_cell_roles().required_cell_ids)
    assert len(artifacts.final_validation_evaluations) == required_count + 1
    assert artifacts.result["validation_cell_coverage"][
        "extra_observed_semantic_cells_not_gated"
    ] == [f"extra:{family}:observed_only"]
    assert len(artifacts.result["semantic_cell_validation_nll_checks"]) == required_count


def test_checkpoint_rejects_tensor_tampering(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_tiny_backend(monkeypatch)
    artifacts = run_semantic_p50(_runtime_inputs())
    checkpoint = dict(artifacts.checkpoint)
    checkpoint["model_state"] = dict(checkpoint["model_state"])
    first = next(iter(checkpoint["model_state"]))
    checkpoint["model_state"][first] = checkpoint["model_state"][first] + 1.0

    with pytest.raises(SemanticP50RunnerError, match="identity disagrees"):
        validate_semantic_p50_checkpoint_payload(checkpoint)


def test_runner_recomputes_and_rejects_arbitrary_frozen_baseline_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_tiny_backend(monkeypatch)
    inputs = _runtime_inputs()
    initial = state_dict_semantic_sha256(inputs.scratch_runtime.model.state_dict())
    result_body = copy.deepcopy(dict(inputs.validation_baseline.result))
    result_body.pop("result_sha256")
    row = result_body["evaluation_records"][0]
    row["canonical_successor_nll_nats"] += 0.125
    row_body = dict(row)
    row_body.pop("evaluation_sha256")
    row["evaluation_sha256"] = _sha(row_body)
    result = {**result_body, "result_sha256": _sha(result_body)}
    completion_body = copy.deepcopy(dict(inputs.validation_baseline.completion))
    completion_body.pop("completion_sha256")
    completion_body["result_sha256"] = result["result_sha256"]
    completion = {
        **completion_body,
        "completion_sha256": _sha(completion_body),
    }
    baseline = VerifiedSemanticP50ValidationBaseline(
        completion_path=inputs.validation_baseline.completion_path,
        completion=completion,
        result=result,
        inventory=inputs.validation_baseline.inventory,
    )
    changed = SemanticP50RuntimeInputs(
        prepared_recipe=inputs.prepared_recipe,
        cache=inputs.cache,
        scratch_runtime=inputs.scratch_runtime,
        validation_baseline=baseline,
        states_by_address_sha256=inputs.states_by_address_sha256,
    )

    with pytest.raises(SemanticP50RunnerError, match="recomputed scratch validation"):
        run_semantic_p50(changed)
    assert state_dict_semantic_sha256(inputs.scratch_runtime.model.state_dict()) == initial


def test_atomic_publication_strictly_reopens_and_rejects_tampering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_tiny_backend(monkeypatch)
    inputs = _runtime_inputs()
    artifacts = run_semantic_p50(inputs)

    completion_path = publish_semantic_p50_run_artifacts(
        tmp_path,
        artifacts=artifacts,
        inputs=inputs,
    )
    reopened = open_semantic_p50_run_artifacts(
        completion_path,
        artifact_root=tmp_path,
        inputs=inputs,
    )

    assert reopened.result["learning_demonstrated"] is False
    assert reopened.result["next_stage_authorized"] is None
    assert reopened.checkpoint["checkpoint_sha256"] == artifacts.checkpoint["checkpoint_sha256"]
    with pytest.raises(SemanticP50RunnerError, match="write-once"):
        publish_semantic_p50_run_artifacts(
            tmp_path,
            artifacts=artifacts,
            inputs=inputs,
        )

    result = json.loads(reopened.result_path.read_text())
    result["learning_demonstrated"] = True
    reopened.result_path.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    )
    with pytest.raises(SemanticP50RunnerError, match="physical result"):
        open_semantic_p50_run_artifacts(
            completion_path,
            artifact_root=tmp_path,
            inputs=inputs,
        )
