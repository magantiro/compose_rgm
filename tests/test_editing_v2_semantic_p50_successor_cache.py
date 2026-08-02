"""Physical and exact-set boundaries for the semantic P50 successor cache."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from compose_v4.data.editing_corpus_contract import ACTIVE8_FAMILIES
from compose_v4.data.editing_v2_semantic_capability_cells import (
    load_semantic_capability_cell_registry,
)
from compose_v4.data.packed_trace_store import PackedTraceAddress
from compose_v4.data.successor_fiber_cache import (
    SuccessorFiberCacheAddress,
    SuccessorFiberCacheRecord,
)
from compose_v4.experiments import editing_v2_semantic_p50_successor_cache as cache
from compose_v4.experiments.editing_v2_semantic_p50_recipe_stream import (
    SemanticP50Candidate,
    SemanticP50CandidateInventory,
    SemanticP50Prerequisites,
    compile_semantic_p50_prepared_recipe,
    write_semantic_p50_prepared_recipe,
)
from compose_v4.experiments.editing_v2_semantic_p50_source_inventory import (
    SemanticP50SourceInventoryBinding,
    VerifiedSemanticP50SourceInventory,
)
from compose_v4.experiments.editing_p50_gate import state_dict_semantic_sha256
from compose_v4.experiments.editing_v2_semantic_runtime import (
    SemanticScratchRuntime,
)
from compose_v4.experiments.factorized_successor_training import (
    StateProductiveSupport,
    TeacherSuccessorAlias,
    TeacherSuccessorFiber,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_REVISION = hashlib.sha256(b"p50-cache-source-revision").hexdigest()


class _FakeScratchModel:
    def __init__(self) -> None:
        self._state = {"weight": torch.arange(4, dtype=torch.float32)}

    def state_dict(self):
        return self._state

    def eval(self):
        return self


FAKE_SCRATCH_STATE_SHA256 = state_dict_semantic_sha256(_FakeScratchModel().state_dict())


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _prerequisites(*, run: str = "a") -> SemanticP50Prerequisites:
    registry = load_semantic_capability_cell_registry()
    values = {
        field: hashlib.sha256(f"{run}:{field}".encode()).hexdigest()
        for field in SemanticP50Prerequisites.__dataclass_fields__
        if field != "operator_capability_fingerprint"
    }
    values["operator_capability_fingerprint"] = hashlib.sha256(
        f"{run}:operator".encode()
    ).hexdigest()[:16]
    values["capability_registry_sha256"] = registry.registry_sha256
    values["classifier_implementation_sha256"] = registry.classifier_implementation_sha256
    values["scratch_initial_model_state_sha256"] = FAKE_SCRATCH_STATE_SHA256
    return SemanticP50Prerequisites(**values)


def _candidate_address(index: int) -> SuccessorFiberCacheAddress:
    return SuccessorFiberCacheAddress(
        packed_shard_content_sha256=hashlib.sha256(
            f"semantic-p50-shard-{index}".encode()
        ).hexdigest(),
        packed_shard_name=f"train-{index}.jsonl.gz",
        entry_index=index,
        layer="reversible_synthetic_walk",
        partition="train",
        trace_id=f"trace-{index}",
        trace_source_key="C",
        trace_target_key="CCC",
        progress_index=0,
        path_length=2,
    )


def _prepared(*, run: str = "a") -> dict[str, object]:
    candidates = tuple(
        SemanticP50Candidate(
            address=_candidate_address(index),
            family=family,
            semantic_cell_id=f"editing_v2_active8_v1:{family}:fixture_context",
            data_lane="reversible_synthetic_walk",
            assignment_sha256=hashlib.sha256(f"{run}:assignment:{family}".encode()).hexdigest(),
        )
        for index, family in enumerate(ACTIVE8_FAMILIES)
    )
    inventory = SemanticP50CandidateInventory(
        prerequisites=_prerequisites(run=run),
        candidates=candidates,
    )
    return compile_semantic_p50_prepared_recipe(inventory)


class _Path:
    def __init__(self, states: tuple[str, ...]) -> None:
        self._states = states

    def state_at(self, progress_index: int) -> str:
        return self._states[progress_index]


class _ValidationIndex:
    def __init__(self, traces: tuple[object, ...]) -> None:
        self._traces = traces

    def iter_accepted_traces_for_partition(self, partition: str):
        return (
            trace for trace in self._traces if trace.addressed_trace.address.partition == partition
        )

    def accepted_transitions_for(self, trace):
        return trace.transitions

    def validate_accepted_transition(self, transition) -> None:
        if transition not in tuple(item for trace in self._traces for item in trace.transitions):
            raise AssertionError("fixture transition is outside the source index")


def _verified_validation_source(prepared: dict[str, object], monkeypatch):
    registry = load_semantic_capability_cell_registry()
    prerequisites = prepared["prerequisites"]
    traces: list[object] = []
    assignments: dict[int, object] = {}
    required_cells = prepared["validation_contract"]["required_nonempty_semantic_cells"]
    assert len(required_cells) == len(ACTIVE8_FAMILIES)
    for index, (family, cell) in enumerate(zip(ACTIVE8_FAMILIES, required_cells, strict=True)):
        address = PackedTraceAddress(
            packed_shard_content_sha256=hashlib.sha256(
                f"validation-shard-{index}".encode()
            ).hexdigest(),
            packed_shard_name=f"validation-{index}.jsonl.gz",
            entry_index=index,
            trace_id=f"validation-trace-{index}",
            layer="reversible_synthetic_walk",
            partition="validation",
            source_key="C",
            target_key="CC",
            path_length=1,
        )
        transition = SimpleNamespace(
            step_index=0,
            addressed_trace=SimpleNamespace(
                address=address,
                path=_Path(("C", "CC")),
            ),
        )
        traces.append(
            SimpleNamespace(
                addressed_trace=transition.addressed_trace,
                transitions=(transition,),
            )
        )
        assignments[id(transition)] = SimpleNamespace(
            model_family=family,
            capability_cell_id=cell,
            data_lane="reversible_synthetic_walk",
            assignment_sha256=hashlib.sha256(
                f"validation-assignment-{family}".encode()
            ).hexdigest(),
        )
    for candidate in prepared["requested_address_union"]:
        address = SuccessorFiberCacheAddress(**candidate)
        packed = PackedTraceAddress(
            packed_shard_content_sha256=address.packed_shard_content_sha256,
            packed_shard_name=address.packed_shard_name,
            entry_index=address.entry_index,
            trace_id=address.trace_id,
            layer=address.layer,
            partition=address.partition,
            source_key=address.trace_source_key,
            target_key=address.trace_target_key,
            path_length=address.path_length,
        )
        addressed = SimpleNamespace(
            address=packed,
            path=_Path(("C", "CC", "CCC")),
        )
        traces.append(
            SimpleNamespace(
                addressed_trace=addressed,
                transitions=tuple(
                    SimpleNamespace(step_index=index, addressed_trace=addressed)
                    for index in range(address.path_length)
                ),
            )
        )

    def classify(_index, transition, *, registry):
        assert registry is selected_registry
        return assignments[id(transition)]

    selected_registry = registry
    monkeypatch.setattr(cache, "classify_verified_structural_transition", classify)
    monkeypatch.setattr(cache, "canonical_state_key", lambda state: state)
    monkeypatch.setattr(
        cache,
        "persistent_slot_state_sha256",
        lambda state: hashlib.sha256(state.encode()).hexdigest(),
    )

    def compile_trace_union(_model, path_records, *, time):
        assert time == cache.SUPPORT_COMPILATION_TIME
        compiled: list[SuccessorFiberCacheRecord] = []
        for record in path_records:
            packed = record.corpus_address
            compiled.extend(
                _records_for_task(
                    {
                        "closure_rows": [
                            {
                                "address": {
                                    "packed_shard_content_sha256": (
                                        packed.packed_shard_content_sha256
                                    ),
                                    "packed_shard_name": packed.packed_shard_name,
                                    "entry_index": packed.entry_index,
                                    "layer": packed.layer,
                                    "partition": packed.partition,
                                    "trace_id": packed.trace_id,
                                    "trace_source_key": packed.source_key,
                                    "trace_target_key": packed.target_key,
                                    "progress_index": progress_index,
                                    "path_length": packed.path_length,
                                }
                            }
                            for progress_index in range(packed.path_length + 1)
                        ]
                    }
                )
            )
        return tuple(sorted(compiled, key=lambda item: item.address))

    monkeypatch.setattr(cache, "compile_successor_fiber_trace_union", compile_trace_union)
    binding = SemanticP50SourceInventoryBinding(
        source_inventory_file_sha256=prerequisites["source_inventory_file_sha256"],
        source_inventory_sha256=prerequisites["source_inventory_sha256"],
        process_identity_sha256=prerequisites["process_identity_sha256"],
        model_runtime_identity_sha256=prerequisites["model_runtime_identity_sha256"],
        active8_policy_sha256=prerequisites["active8_policy_sha256"],
        operator_capability_fingerprint=prerequisites["operator_capability_fingerprint"],
        decision_source_implementation_sha256=prerequisites[
            "decision_source_implementation_sha256"
        ],
    )
    return (
        VerifiedSemanticP50SourceInventory(
            path=Path("fixture-source-inventory.json"),
            binding=binding,
            index=_ValidationIndex(tuple(traces)),
        ),
        registry,
    )


def _source_reopen_paths() -> cache.SemanticP50SourceReopenPaths:
    return cache.SemanticP50SourceReopenPaths(
        source_inventory_path=Path("source.json"),
        migration_completion_path=Path("migration.json"),
        chunk_cache_plan_path=Path("chunk-plan.json"),
        chunk_cache_global_completion_path=Path("chunk-completion.json"),
        decision_plan_path=Path("decision-plan.json"),
        decision_completion_path=Path("decision-completion.json"),
    )


def _expected_identity(
    prepared: dict[str, object],
) -> cache.SemanticP50SuccessorCacheExpectedIdentity:
    prerequisites = prepared["prerequisites"]
    return cache.SemanticP50SuccessorCacheExpectedIdentity(
        prepared_recipe_sha256=prepared["prepared_recipe_sha256"],
        source_inventory_sha256=prerequisites["source_inventory_sha256"],
        process_identity_sha256=prerequisites["process_identity_sha256"],
        model_runtime_identity_sha256=prerequisites["model_runtime_identity_sha256"],
        active8_policy_sha256=prerequisites["active8_policy_sha256"],
        gate_zero_evidence_sha256=prerequisites["gate_zero_evidence_sha256"],
        t1_decision_sha256=prerequisites["t1_decision_sha256"],
        scratch_initial_model_state_sha256=prerequisites["scratch_initial_model_state_sha256"],
    )


def _scratch_runtime(prepared: dict[str, object]) -> SemanticScratchRuntime:
    prerequisites = prepared["prerequisites"]
    return SemanticScratchRuntime(
        model=_FakeScratchModel(),
        config=SimpleNamespace(),
        architecture=SimpleNamespace(
            operator_capability_fingerprint=prerequisites["operator_capability_fingerprint"]
        ),
        semantic_model_identity={},
        semantic_model_process_contract_sha256=hashlib.sha256(
            b"fixture-semantic-contract"
        ).hexdigest(),
        process_identity_sha256=prerequisites["process_identity_sha256"],
        initial_model_state_sha256=FAKE_SCRATCH_STATE_SHA256,
    )


def _records_for_task(task: dict[str, object]) -> tuple[SuccessorFiberCacheRecord, ...]:
    closure = task["closure_rows"]
    assert isinstance(closure, list)
    addresses = [SuccessorFiberCacheAddress(**row["address"]) for row in closure]
    state_keys = ("C", "CC", "CCC")
    state_hashes = tuple(hashlib.sha256(key.encode()).hexdigest() for key in state_keys)
    records: list[SuccessorFiberCacheRecord] = []
    for progress_index, address in enumerate(addresses):
        support = StateProductiveSupport(
            source_key=state_keys[progress_index],
            source_state_sha256=state_hashes[progress_index],
            virtual_aliases=(),
        )
        if address.is_terminal:
            teacher = None
        else:
            teacher = TeacherSuccessorFiber(
                source_key=state_keys[progress_index],
                target_key=state_keys[progress_index + 1],
                target_state_sha256=state_hashes[progress_index + 1],
                aliases=(
                    TeacherSuccessorAlias(
                        family_name="atom_insert",
                        table_name="atom_insert_bond_order",
                        coordinate=(progress_index, 0, 0),
                    ),
                ),
                state_support=support,
            )
        records.append(
            SuccessorFiberCacheRecord(
                address=address,
                state_support=support,
                teacher_fiber=teacher,
            )
        )
    return tuple(records)


def _runtime(plan: dict[str, object]) -> dict[str, object]:
    body = {
        "device": "cpu",
        "dtype": "torch.float32",
        "python_version": "fixture-python",
        "torch_version": "fixture-torch",
        "rdkit_version": "fixture-rdkit",
        "implementation_sha256": plan["implementation_sha256"],
        "source_revision_sha256": plan["source_revision_sha256"],
    }
    return {**body, "runtime_sha256": _sha(body)}


def _write_prepared(artifact_root: Path, prepared: dict[str, object]) -> Path:
    path = artifact_root / "inputs" / "semantic_p50_prepared_recipe.json"
    write_semantic_p50_prepared_recipe(path, prepared)
    return path


def _build_plan(
    artifact_root: Path,
    prepared: dict[str, object],
    monkeypatch,
) -> tuple[
    dict[str, object],
    Path,
    VerifiedSemanticP50SourceInventory,
    object,
]:
    prepared_path = _write_prepared(artifact_root, prepared)
    source, registry = _verified_validation_source(prepared, monkeypatch)
    plan = cache.build_semantic_p50_successor_cache_plan(
        prepared_recipe_path=prepared_path,
        verified_source=source,
        source_revision_sha256=SOURCE_REVISION,
        artifact_root=artifact_root,
        repo_root=REPO_ROOT,
        registry=registry,
        output_prefix="/artifacts/p50_cache",
    )
    return plan, prepared_path, source, registry


def _leaf_path(artifact_root: Path, plan: dict[str, object], task: dict[str, object]) -> Path:
    run_root = artifact_root / "p50_cache" / plan["build_identity_sha256"]
    return run_root / "tasks" / task["task_identity_sha256"] / cache.LEAF_FILENAME


def _materialize(
    artifact_root: Path, prepared: dict[str, object], monkeypatch
) -> tuple[
    dict[str, object],
    Path,
    dict[str, object],
    Path,
    Path,
    VerifiedSemanticP50SourceInventory,
    object,
]:
    plan, prepared_path, source, registry = _build_plan(artifact_root, prepared, monkeypatch)
    run_root = artifact_root / "p50_cache" / plan["build_identity_sha256"]
    plan_path = run_root / cache.PLAN_FILENAME
    cache.write_semantic_p50_successor_cache_artifact(plan_path, plan)
    runtime = _runtime(plan)
    for task in plan["tasks"]:
        leaf = cache.build_semantic_p50_successor_cache_leaf(
            plan,
            task_identity_sha256=task["task_identity_sha256"],
            records=_records_for_task(task),
            compiler_runtime=runtime,
            artifact_root=artifact_root,
            repo_root=REPO_ROOT,
        )
        cache.write_semantic_p50_successor_cache_artifact(
            _leaf_path(artifact_root, plan, task), leaf
        )
    manifest = cache.build_semantic_p50_successor_cache_manifest(
        plan,
        plan_path=plan_path,
        artifact_root=artifact_root,
        repo_root=REPO_ROOT,
    )
    manifest_path = run_root / cache.MANIFEST_FILENAME
    cache.write_semantic_p50_successor_cache_artifact(manifest_path, manifest)
    completion = cache.build_semantic_p50_successor_cache_completion(
        plan,
        manifest,
        plan_path=plan_path,
        manifest_path=manifest_path,
        artifact_root=artifact_root,
    )
    completion_path = run_root / cache.COMPLETION_FILENAME
    cache.write_semantic_p50_successor_cache_artifact(completion_path, completion)
    monkeypatch.setattr(cache, "_open_verified_source", lambda *args, **kwargs: source)
    return (
        plan,
        prepared_path,
        manifest,
        manifest_path,
        completion_path,
        source,
        registry,
    )


def _rehash_leaf(leaf: dict[str, object]) -> None:
    leaf["record_inventory_sha256"] = _sha(leaf["records"])
    body = dict(leaf)
    body.pop("leaf_sha256", None)
    leaf["leaf_sha256"] = _sha(body)


def _canonical_file_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _republish_tampered_leaf_chain(
    *,
    artifact_root: Path,
    plan: dict[str, object],
    manifest: dict[str, object],
    manifest_path: Path,
    completion_path: Path,
    task_index: int,
    mutate,
) -> None:
    task = plan["tasks"][task_index]
    leaf_path = _leaf_path(artifact_root, plan, task)
    leaf = json.loads(leaf_path.read_text())
    mutate(leaf)
    _rehash_leaf(leaf)
    leaf_bytes = _canonical_file_bytes(leaf)
    leaf_path.write_bytes(leaf_bytes)

    leaf_spec = manifest["leaves"][task_index]
    leaf_spec["leaf_file_sha256"] = hashlib.sha256(leaf_bytes).hexdigest()
    leaf_spec["leaf_file_bytes"] = len(leaf_bytes)
    leaf_spec["leaf_sha256"] = leaf["leaf_sha256"]
    leaf_spec["record_inventory_sha256"] = leaf["record_inventory_sha256"]
    manifest["leaf_inventory_sha256"] = _sha(manifest["leaves"])
    manifest_body = dict(manifest)
    manifest_body.pop("manifest_sha256")
    manifest["manifest_sha256"] = _sha(manifest_body)
    manifest_bytes = _canonical_file_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)

    completion = json.loads(completion_path.read_text())
    completion["manifest_file_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    completion["manifest_sha256"] = manifest["manifest_sha256"]
    completion_body = dict(completion)
    completion_body.pop("completion_sha256")
    completion["completion_sha256"] = _sha(completion_body)
    completion_path.write_bytes(_canonical_file_bytes(completion))


def test_complete_cache_opens_exact_train_and_validation_unions(
    tmp_path: Path, monkeypatch
) -> None:
    artifact_root = tmp_path / "artifacts"
    prepared = _prepared()
    plan, _, _, _, completion_path, _, registry = _materialize(artifact_root, prepared, monkeypatch)

    loaded = cache.open_semantic_p50_successor_cache(
        completion_path,
        artifact_root=artifact_root,
        repo_root=REPO_ROOT,
        expected_identity=_expected_identity(prepared),
        source_reopen_paths=_source_reopen_paths(),
        scratch_runtime=_scratch_runtime(prepared),
        registry=registry,
    )

    assert len(loaded.train_requested_records) == len(prepared["requested_address_union"])
    assert len(loaded.validation_requested_records) == len(ACTIVE8_FAMILIES)
    assert len(loaded.requested_records) == 2 * len(ACTIVE8_FAMILIES)
    assert len(loaded.records) == (
        len(prepared["complete_trace_closure_inventory"]) + 2 * len(ACTIVE8_FAMILIES)
    )
    assert all(record.teacher_fiber is not None for record in loaded.requested_records)
    assert sum(record.address.is_terminal for record in loaded.records) == len(plan["tasks"])
    assert loaded.completion["training_authorized"] is False
    assert loaded.completion["bounded_p50_authorized"] is False
    assert loaded.completion["training_launched"] is False


def test_prepared_recipe_rejects_terminal_or_closure_only_scheduling() -> None:
    prepared = copy.deepcopy(_prepared())
    terminal = next(row for row in prepared["complete_trace_closure_inventory"] if row["terminal"])
    terminal["schedulable"] = True
    prepared["complete_trace_closure_inventory_sha256"] = _sha(
        prepared["complete_trace_closure_inventory"]
    )
    body = dict(prepared)
    body.pop("prepared_recipe_sha256")
    prepared["prepared_recipe_sha256"] = _sha(body)

    with pytest.raises(
        cache.SemanticP50SuccessorCacheError,
        match="closure scheduling flags",
    ):
        cache.validate_semantic_p50_prepared_recipe(prepared)


def test_leaf_rejects_missing_extra_duplicate_and_same_count_substitution(
    tmp_path: Path, monkeypatch
) -> None:
    artifact_root = tmp_path / "artifacts"
    plan, _, _, _ = _build_plan(artifact_root, _prepared(), monkeypatch)
    task = plan["tasks"][0]
    leaf = cache.build_semantic_p50_successor_cache_leaf(
        plan,
        task_identity_sha256=task["task_identity_sha256"],
        records=_records_for_task(task),
        compiler_runtime=_runtime(plan),
        artifact_root=artifact_root,
        repo_root=REPO_ROOT,
    )

    missing = copy.deepcopy(leaf)
    missing["records"].pop()
    missing["record_count"] -= 1
    _rehash_leaf(missing)
    with pytest.raises(cache.SemanticP50SuccessorCacheError):
        cache.validate_semantic_p50_successor_cache_leaf_for_plan(missing, plan=plan)

    extra = copy.deepcopy(leaf)
    extra["records"].append(copy.deepcopy(extra["records"][-1]))
    extra["records"][-1]["progress_index"] = 3
    extra["records"][-1]["path_length"] = 3
    extra["record_count"] += 1
    _rehash_leaf(extra)
    with pytest.raises(cache.SemanticP50SuccessorCacheError):
        cache.validate_semantic_p50_successor_cache_leaf_for_plan(extra, plan=plan)

    duplicate = copy.deepcopy(leaf)
    duplicate["records"].append(copy.deepcopy(duplicate["records"][0]))
    duplicate["record_count"] += 1
    _rehash_leaf(duplicate)
    with pytest.raises(cache.SemanticP50SuccessorCacheError):
        cache.validate_semantic_p50_successor_cache_leaf_for_plan(duplicate, plan=plan)

    substituted = copy.deepcopy(leaf)
    for record in substituted["records"]:
        record["trace_id"] = "same-count-substituted-trace"
    _rehash_leaf(substituted)
    assert cache.validate_semantic_p50_successor_cache_leaf(substituted) == substituted
    with pytest.raises(
        cache.SemanticP50SuccessorCacheError,
        match="address set differs",
    ):
        cache.validate_semantic_p50_successor_cache_leaf_for_plan(substituted, plan=plan)


def test_open_rejects_cross_run_identity_and_physical_leaf_tamper(
    tmp_path: Path, monkeypatch
) -> None:
    artifact_root = tmp_path / "artifacts"
    prepared = _prepared()
    plan, _, _, _, completion_path, _, registry = _materialize(artifact_root, prepared, monkeypatch)
    expected = _expected_identity(prepared)
    wrong = cache.SemanticP50SuccessorCacheExpectedIdentity(
        **{
            **expected.as_mapping(),
            "process_identity_sha256": hashlib.sha256(b"other-run").hexdigest(),
        }
    )
    with pytest.raises(
        cache.SemanticP50SuccessorCacheError,
        match="cross-run identity",
    ):
        cache.open_semantic_p50_successor_cache(
            completion_path,
            artifact_root=artifact_root,
            repo_root=REPO_ROOT,
            expected_identity=wrong,
            source_reopen_paths=_source_reopen_paths(),
            scratch_runtime=_scratch_runtime(prepared),
            registry=registry,
        )

    first_task = plan["tasks"][0]
    leaf_path = _leaf_path(artifact_root, plan, first_task)
    leaf_path.write_bytes(leaf_path.read_bytes() + b" ")
    with pytest.raises(cache.SemanticP50SuccessorCacheError):
        cache.open_semantic_p50_successor_cache(
            completion_path,
            artifact_root=artifact_root,
            repo_root=REPO_ROOT,
            expected_identity=expected,
            source_reopen_paths=_source_reopen_paths(),
            scratch_runtime=_scratch_runtime(prepared),
            registry=registry,
        )


def test_open_physically_reopens_prepared_recipe(tmp_path: Path, monkeypatch) -> None:
    artifact_root = tmp_path / "artifacts"
    prepared = _prepared()
    _, prepared_path, _, _, completion_path, _, registry = _materialize(
        artifact_root, prepared, monkeypatch
    )
    expected = _expected_identity(prepared)

    prepared_path.write_bytes(prepared_path.read_bytes() + b"\n")
    with pytest.raises(
        cache.SemanticP50SuccessorCacheError,
        match="canonical JSON",
    ):
        cache.open_semantic_p50_successor_cache(
            completion_path,
            artifact_root=artifact_root,
            repo_root=REPO_ROOT,
            expected_identity=expected,
            source_reopen_paths=_source_reopen_paths(),
            scratch_runtime=_scratch_runtime(prepared),
            registry=registry,
        )


def test_manifest_rejects_same_count_leaf_swap(tmp_path: Path, monkeypatch) -> None:
    artifact_root = tmp_path / "artifacts"
    prepared = _prepared()
    (
        plan,
        _,
        manifest,
        manifest_path,
        completion_path,
        _,
        registry,
    ) = _materialize(artifact_root, prepared, monkeypatch)
    swapped = copy.deepcopy(manifest)
    first, second = swapped["leaves"][:2]
    first["leaf_artifact_path"], second["leaf_artifact_path"] = (
        second["leaf_artifact_path"],
        first["leaf_artifact_path"],
    )
    swapped["leaf_inventory_sha256"] = _sha(swapped["leaves"])
    body = dict(swapped)
    body.pop("manifest_sha256")
    swapped["manifest_sha256"] = _sha(body)

    # The metadata is self-consistent and has identical counts, but the strict
    # opener requires each leaf at the content-addressed task path.
    assert cache.validate_semantic_p50_successor_cache_manifest(swapped, plan=plan) == swapped
    manifest_path.write_bytes(_canonical_file_bytes(swapped))
    completion = json.loads(completion_path.read_text())
    completion["manifest_file_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    completion["manifest_sha256"] = swapped["manifest_sha256"]
    completion_body = dict(completion)
    completion_body.pop("completion_sha256")
    completion["completion_sha256"] = _sha(completion_body)
    completion_path.write_bytes(_canonical_file_bytes(completion))
    first_path = cache._artifact_path(
        first["leaf_artifact_path"],
        artifact_root=artifact_root,
        field_name="fixture leaf",
    )
    expected_first = _leaf_path(artifact_root, plan, plan["tasks"][0])
    assert first_path != expected_first
    with pytest.raises(
        cache.SemanticP50SuccessorCacheError,
        match="content-addressed path",
    ):
        cache.open_semantic_p50_successor_cache(
            completion_path,
            artifact_root=artifact_root,
            repo_root=REPO_ROOT,
            expected_identity=_expected_identity(prepared),
            source_reopen_paths=_source_reopen_paths(),
            scratch_runtime=_scratch_runtime(prepared),
            registry=registry,
        )


def test_validation_contract_rejects_role_relabel_even_when_rehashed(
    tmp_path: Path, monkeypatch
) -> None:
    artifact_root = tmp_path / "artifacts"
    prepared = _prepared()
    plan, _, _, _ = _build_plan(artifact_root, prepared, monkeypatch)
    contract = copy.deepcopy(plan["validation_contract"])
    candidate = contract["candidate_rows"][0]
    candidate["address"]["partition"] = "train"
    candidate_body = dict(candidate)
    candidate_body.pop("candidate_sha256")
    candidate["candidate_sha256"] = _sha(candidate_body)
    contract["candidate_inventory_sha256"] = _sha(contract["candidate_rows"])
    contract_body = dict(contract)
    contract_body.pop("validation_contract_sha256")
    contract["validation_contract_sha256"] = _sha(contract_body)

    with pytest.raises(
        cache.SemanticP50SuccessorCacheError,
        match="candidate role or support",
    ):
        cache._validate_authenticated_validation_contract(contract, prepared=prepared)


def test_open_rejects_physically_rederived_validation_set_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    artifact_root = tmp_path / "artifacts"
    prepared = _prepared()
    (
        _,
        _,
        _,
        _,
        completion_path,
        source,
        registry,
    ) = _materialize(artifact_root, prepared, monkeypatch)
    shortened_source = VerifiedSemanticP50SourceInventory(
        path=source.path,
        binding=source.binding,
        index=_ValidationIndex(source.index._traces[1:]),
    )
    monkeypatch.setattr(
        cache,
        "_open_verified_source",
        lambda *args, **kwargs: shortened_source,
    )

    with pytest.raises(
        cache.SemanticP50SuccessorCacheError,
        match="physical validation source differs",
    ):
        cache.open_semantic_p50_successor_cache(
            completion_path,
            artifact_root=artifact_root,
            repo_root=REPO_ROOT,
            expected_identity=_expected_identity(prepared),
            source_reopen_paths=_source_reopen_paths(),
            scratch_runtime=_scratch_runtime(prepared),
            registry=registry,
        )


def test_open_rejects_same_address_wrong_states_and_teacher_fibers_after_full_rehash(
    tmp_path: Path, monkeypatch
) -> None:
    artifact_root = tmp_path / "artifacts"
    prepared = _prepared()
    (
        plan,
        _,
        manifest,
        manifest_path,
        completion_path,
        _,
        registry,
    ) = _materialize(artifact_root, prepared, monkeypatch)
    task_index = next(
        index
        for index, task in enumerate(plan["tasks"])
        if task["population_role"] == "train_optimization"
    )
    task = plan["tasks"][task_index]
    leaf_path = _leaf_path(artifact_root, plan, task)
    leaf = json.loads(leaf_path.read_text())
    wrong_hashes = tuple(
        hashlib.sha256(f"wrong-physical-state-{index}".encode()).hexdigest()
        for index in range(len(leaf["records"]))
    )
    for index, record in enumerate(leaf["records"]):
        record["source_state_sha256"] = wrong_hashes[index]
        if record["target_state_sha256"] is not None:
            record["target_state_sha256"] = wrong_hashes[index + 1]
    _rehash_leaf(leaf)
    leaf_bytes = _canonical_file_bytes(leaf)
    leaf_path.write_bytes(leaf_bytes)

    leaf_spec = manifest["leaves"][task_index]
    leaf_spec["leaf_file_sha256"] = hashlib.sha256(leaf_bytes).hexdigest()
    leaf_spec["leaf_file_bytes"] = len(leaf_bytes)
    leaf_spec["leaf_sha256"] = leaf["leaf_sha256"]
    leaf_spec["record_inventory_sha256"] = leaf["record_inventory_sha256"]
    manifest["leaf_inventory_sha256"] = _sha(manifest["leaves"])
    manifest_body = dict(manifest)
    manifest_body.pop("manifest_sha256")
    manifest["manifest_sha256"] = _sha(manifest_body)
    manifest_bytes = _canonical_file_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)

    completion = json.loads(completion_path.read_text())
    completion["manifest_file_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    completion["manifest_sha256"] = manifest["manifest_sha256"]
    completion_body = dict(completion)
    completion_body.pop("completion_sha256")
    completion["completion_sha256"] = _sha(completion_body)
    completion_path.write_bytes(_canonical_file_bytes(completion))

    with pytest.raises(
        cache.SemanticP50SuccessorCacheError,
        match="cached source state differs",
    ):
        cache.open_semantic_p50_successor_cache(
            completion_path,
            artifact_root=artifact_root,
            repo_root=REPO_ROOT,
            expected_identity=_expected_identity(prepared),
            source_reopen_paths=_source_reopen_paths(),
            scratch_runtime=_scratch_runtime(prepared),
            registry=registry,
        )


def test_open_rejects_same_state_fabricated_candidate_coordinates_after_full_rehash(
    tmp_path: Path, monkeypatch
) -> None:
    artifact_root = tmp_path / "artifacts"
    prepared = _prepared()
    (
        plan,
        _,
        manifest,
        manifest_path,
        completion_path,
        _,
        registry,
    ) = _materialize(artifact_root, prepared, monkeypatch)
    task_index = next(
        index
        for index, task in enumerate(plan["tasks"])
        if task["population_role"] == "validation_baseline"
    )

    def fabricate_candidate(leaf: dict[str, object]) -> None:
        nonterminal = next(
            record for record in leaf["records"] if record["target_state_sha256"] is not None
        )
        nonterminal["aliases"][0]["coordinate"] = [17, 0, 0]

    _republish_tampered_leaf_chain(
        artifact_root=artifact_root,
        plan=plan,
        manifest=manifest,
        manifest_path=manifest_path,
        completion_path=completion_path,
        task_index=task_index,
        mutate=fabricate_candidate,
    )

    with pytest.raises(
        cache.SemanticP50SuccessorCacheError,
        match="coordinates differ from fresh production compilation",
    ):
        cache.open_semantic_p50_successor_cache(
            completion_path,
            artifact_root=artifact_root,
            repo_root=REPO_ROOT,
            expected_identity=_expected_identity(prepared),
            source_reopen_paths=_source_reopen_paths(),
            scratch_runtime=_scratch_runtime(prepared),
            registry=registry,
        )


def test_expected_identity_cannot_be_derived_from_prepared_receipt() -> None:
    assert not hasattr(cache.SemanticP50SuccessorCacheExpectedIdentity, "from_prepared")
