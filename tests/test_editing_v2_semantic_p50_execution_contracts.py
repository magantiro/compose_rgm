"""Adversarial tests for fail-closed semantic P50 execution contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from compose_v4.data.editing_corpus_contract import ACTIVE8_FAMILIES
from compose_v4.experiments import (
    editing_v2_semantic_p50_execution_contracts as contracts,
)
from compose_v4.experiments.editing_p50_gate import state_dict_semantic_sha256
from compose_v4.experiments.editing_v2_semantic_p50_recipe_stream import (
    SemanticP50Prerequisites,
    authorize_semantic_p50_recipe,
)
from compose_v4.experiments.editing_v2_semantic_p50_source_inventory import (
    SemanticP50SourceInventoryBinding,
)
from compose_v4.experiments.editing_v2_semantic_runtime import (
    SemanticScratchArchitecture,
    SemanticScratchModelConfig,
    SemanticScratchRuntime,
)


def _bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        + b"\n"
    )


def _sha(value: object) -> str:
    return hashlib.sha256(_bytes(value)[:-1]).hexdigest()


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_bytes(payload))
    return path


def _self_hashed(
    *,
    schema: str,
    status: str,
    semantic_field: str,
    **extra: object,
) -> dict[str, object]:
    body = {
        "schema": schema,
        "schema_version": 1,
        "status": status,
        **extra,
    }
    return {**body, semantic_field: _sha(body)}


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repo), *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _source_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "clean-source"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Semantic P50 Test")
    trainer = repo / "src" / "compose_v4" / "experiments" / "semantic_p50_trainer.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text("# frozen fixture trainer\n", encoding="utf-8")
    image = repo / "modal" / "semantic_p50_image.py"
    image.parent.mkdir(parents=True)
    image.write_text("# frozen fixture image definition\n", encoding="utf-8")
    gate = repo / "configs" / "editing_training_v2_gate.json"
    gate.parent.mkdir(parents=True)
    gate.write_bytes(
        (
            Path(__file__).parents[1] / "configs/editing_training_v2_gate.json"
        ).read_bytes()
    )
    _git(repo, "add", "src/compose_v4/experiments/semantic_p50_trainer.py")
    _git(repo, "add", "modal/semantic_p50_image.py")
    _git(repo, "add", "configs/editing_training_v2_gate.json")
    _git(repo, "commit", "-qm", "test: freeze fixture source")
    return repo, trainer, image


def _scratch(
    *,
    process_sha256: str,
    process_contract_sha256: str,
    capability_fingerprint: str,
) -> SemanticScratchRuntime:
    model = torch.nn.Linear(2, 2)
    config = SemanticScratchModelConfig(
        initialization_seed=104729,
        max_atoms=40,
        hidden_dim=128,
        message_passing_steps=4,
        mark_dim=32,
        dtype="torch.float32",
        atom_vocabulary_class_count=15,
        catalog_fingerprint="fedcba9876543210",
    )
    architecture = SemanticScratchArchitecture(
        max_atoms=config.max_atoms,
        hidden_dim=config.hidden_dim,
        message_passing_steps=config.message_passing_steps,
        mark_dim=config.mark_dim,
        dtype=config.dtype,
        parameter_dtypes=(config.dtype,),
        atom_vocabulary_class_count=config.atom_vocabulary_class_count,
        catalog_fingerprint=config.catalog_fingerprint,
        operator_capability_fingerprint=capability_fingerprint,
    )
    return SemanticScratchRuntime(
        model=model,
        config=config,
        architecture=architecture,
        semantic_model_identity={
            "operator_capability_fingerprint": capability_fingerprint,
            "editing_process_semantics": "editing_v2_semantic",
        },
        semantic_model_process_contract_sha256=process_contract_sha256,
        process_identity_sha256=process_sha256,
        initial_model_state_sha256=state_dict_semantic_sha256(model.state_dict()),
    )


@pytest.fixture
def physical_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    repo, trainer, image_definition = _source_repo(tmp_path)
    source_revision = contracts.observe_clean_source_revision(repo)
    process_sha = _digest("process")
    process_contract_sha = _digest("process-contract")
    capability_fingerprint = "0123456789abcdef"
    scratch = _scratch(
        process_sha256=process_sha,
        process_contract_sha256=process_contract_sha,
        capability_fingerprint=capability_fingerprint,
    )

    runtime_body = {
        "schema": "compose.data.semantic_active8_exact_model_runtime",
        "schema_version": 2,
        "runtime_contract_sha256": _digest("model-runtime-contract"),
        "semantic_model_process_contract_sha256": process_contract_sha,
        "semantic_model_identity": scratch.semantic_model_identity,
        "process_identity_sha256": process_sha,
        "architecture": scratch.architecture.as_payload(),
        "initialization_seed": scratch.config.initialization_seed,
        "initial_model_state_sha256": scratch.initial_model_state_sha256,
        "software": {"torch": str(torch.__version__)},
        "producer_source_revision_sha256": _digest("active8-producer-source"),
        "execution_source_revision_sha256": source_revision["source_revision_sha256"],
    }
    model_runtime = {**runtime_body, "identity_sha256": _sha(runtime_body)}

    source_payload = _self_hashed(
        schema="compose.test.physically_reopened_source",
        status="COMPLETE_NO_AUTHORITY",
        semantic_field="inventory_sha256",
    )
    source_path = _write(
        artifact_root / "source" / "SEMANTIC_P50_SOURCE_INVENTORY.json",
        source_payload,
    )
    source_binding = SemanticP50SourceInventoryBinding(
        source_inventory_file_sha256=hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest(),
        source_inventory_sha256=source_payload["inventory_sha256"],
        process_identity_sha256=process_sha,
        model_runtime_identity_sha256=model_runtime["identity_sha256"],
        active8_policy_sha256=_digest("active8-policy"),
        operator_capability_fingerprint=capability_fingerprint,
        decision_source_implementation_sha256=_digest("decision-source"),
    )

    gate = _self_hashed(
        schema="compose.test.gate-zero",
        status="PASS_NO_AUTHORITY",
        semantic_field="evidence_sha256",
        structural_result="PASS",
    )
    gate_path = _write(artifact_root / "gate" / "STRUCTURAL_EVIDENCE.json", gate)
    t1 = _self_hashed(
        schema="compose.test.t1",
        status="GO_BOUNDED_P50",
        semantic_field="decision_sha256",
        bounded_p50_authorized=True,
        p500_authorized=False,
        checkpoint_selection_authorized=False,
        final_test_selection_authorized=False,
    )
    t1_path = _write(
        artifact_root / "t1" / "SEMANTIC_T1_CAPACITY_DECISION.json",
        t1,
    )

    prerequisite_values = {
        name: _digest(name)
        for name in SemanticP50Prerequisites.__dataclass_fields__
        if name != "operator_capability_fingerprint"
    }
    prerequisite_values.update(
        {
            "source_inventory_file_sha256": source_binding.source_inventory_file_sha256,
            "source_inventory_sha256": source_binding.source_inventory_sha256,
            "process_identity_sha256": process_sha,
            "model_runtime_identity_sha256": model_runtime["identity_sha256"],
            "active8_policy_sha256": source_binding.active8_policy_sha256,
            "operator_capability_fingerprint": capability_fingerprint,
            "decision_source_implementation_sha256": (
                source_binding.decision_source_implementation_sha256
            ),
            "gate_zero_evidence_file_sha256": hashlib.sha256(
                gate_path.read_bytes()
            ).hexdigest(),
            "gate_zero_evidence_sha256": gate["evidence_sha256"],
            "t1_decision_file_sha256": hashlib.sha256(t1_path.read_bytes()).hexdigest(),
            "t1_decision_sha256": t1["decision_sha256"],
            "scratch_initial_model_state_sha256": scratch.initial_model_state_sha256,
        }
    )
    prerequisites = SemanticP50Prerequisites(**prerequisite_values)
    prepared = _self_hashed(
        schema="compose.test.prepared",
        status="PREPARED_NO_AUTHORITY",
        semantic_field="prepared_recipe_sha256",
        prerequisites=prerequisites.as_payload(),
        bounded_p50_authorized=False,
        unresolved_physical_bindings=list(contracts.REQUIRED_BINDING_PURPOSES),
        required_families=list(ACTIVE8_FAMILIES),
        minimum_nonzero_gradient_updates_by_family={
            family: 40 for family in ACTIVE8_FAMILIES
        },
        validation_contract={
            "maximum_family_final_minus_baseline_for_p50_nonincrease_nats": 1e-7
        },
    )
    prepared_path = _write(artifact_root / "recipe" / "prepared.json", prepared)
    cache = _self_hashed(
        schema="compose.test.cache",
        status="COMPLETE_NO_AUTHORITY",
        semantic_field="completion_sha256",
        source_revision_sha256=source_revision["source_revision_sha256"],
        execution_source_revision_sha256=source_revision["source_revision_sha256"],
    )
    cache_path = _write(
        artifact_root / "cache" / "semantic_p50_successor_cache_completion.json",
        cache,
    )
    validation_inventory_path = _write(
        artifact_root
        / "validation"
        / "SEMANTIC_P50_VALIDATION_INVENTORY_COMPLETE.json",
        _self_hashed(
            schema="compose.test.validation-inventory",
            status="COMPLETE_NO_AUTHORITY",
            semantic_field="completion_sha256",
        ),
    )
    baseline_binding = {
        "source_inventory_file_sha256": source_binding.source_inventory_file_sha256,
        "source_inventory_sha256": source_binding.source_inventory_sha256,
        "prepared_recipe_file_sha256": hashlib.sha256(
            prepared_path.read_bytes()
        ).hexdigest(),
        "prepared_recipe_sha256": prepared["prepared_recipe_sha256"],
        "scratch_initial_model_state_sha256": scratch.initial_model_state_sha256,
    }
    baseline_result = {
        "binding": baseline_binding,
        "successor_cache_completion_file_sha256": hashlib.sha256(
            cache_path.read_bytes()
        ).hexdigest(),
        "successor_cache_completion_sha256": cache["completion_sha256"],
        "evaluated_model_state_sha256": scratch.initial_model_state_sha256,
    }
    baseline_completion = _self_hashed(
        schema="compose.test.validation-baseline",
        status="COMPLETE_NO_AUTHORITY",
        semantic_field="completion_sha256",
    )
    baseline_path = _write(
        artifact_root / "validation" / "SEMANTIC_P50_VALIDATION_BASELINE_COMPLETE.json",
        baseline_completion,
    )
    validation_environment_path = _write(
        artifact_root / "validation" / "evaluation-environment.json",
        _self_hashed(
            schema="compose.test.validation-environment",
            status="COMPLETE_NO_AUTHORITY",
            semantic_field="receipt_sha256",
        ),
    )
    paths = contracts.SemanticP50ExecutionPrerequisitePaths(
        artifact_root=artifact_root,
        repo_root=repo,
        source_inventory_path=source_path,
        migration_completion_path=artifact_root / "source" / "migration.json",
        chunk_cache_plan_path=artifact_root / "source" / "chunk-plan.json",
        chunk_cache_global_completion_path=artifact_root
        / "source"
        / "chunk-complete.json",
        decision_plan_path=artifact_root / "source" / "decision-plan.json",
        decision_completion_path=artifact_root / "source" / "decision-complete.json",
        gate_zero_evidence_path=gate_path,
        t1_decision_path=t1_path,
        prepared_recipe_path=prepared_path,
        successor_cache_completion_path=cache_path,
        validation_inventory_completion_path=validation_inventory_path,
        validation_baseline_completion_path=baseline_path,
        validation_evaluation_environment_receipt_path=validation_environment_path,
    )

    calls: dict[str, int] = {
        "source": 0,
        "gate": 0,
        "t1": 0,
        "relationships": 0,
        "prepared": 0,
        "cache": 0,
        "baseline": 0,
    }
    source = SimpleNamespace(
        binding=source_binding,
        index=SimpleNamespace(identity_payload=lambda: source_payload),
        path=source_path,
    )

    def load_source(*args, **kwargs):
        calls["source"] += 1
        assert kwargs["expected_binding"] == source_binding
        return source

    def open_gate(*args, **kwargs):
        calls["gate"] += 1
        assert (
            kwargs["expected_file_sha256"]
            == hashlib.sha256(gate_path.read_bytes()).hexdigest()
        )
        return gate

    def open_t1(value, **kwargs):
        calls["t1"] += 1
        assert value == t1
        assert kwargs["require_p50_go"] is True
        return t1

    def relationships(**kwargs):
        calls["relationships"] += 1
        assert kwargs["source"] is source
        return prerequisites

    def load_prepared(path):
        calls["prepared"] += 1
        assert Path(path).resolve() == prepared_path.resolve()
        return prepared, hashlib.sha256(prepared_path.read_bytes()).hexdigest()

    class ExpectedCache:
        pass

    def expected_cache_identity(**kwargs):
        assert kwargs["prepared_recipe_sha256"] == prepared["prepared_recipe_sha256"]
        assert kwargs["source"] is source
        assert Path(kwargs["gate_zero_evidence_path"]).resolve() == gate_path.resolve()
        assert Path(kwargs["t1_decision_path"]).resolve() == t1_path.resolve()
        assert Path(kwargs["repo_root"]).resolve() == repo.resolve()
        return ExpectedCache()

    def open_cache(*args, **kwargs):
        calls["cache"] += 1
        assert isinstance(kwargs["expected_identity"], ExpectedCache)
        assert isinstance(
            kwargs["source_reopen_paths"], contracts.SemanticP50SourceReopenPaths
        )
        assert kwargs["source_reopen_paths"].source_inventory_path == source_path
        return SimpleNamespace(completion=cache)

    def open_baseline(*args, **kwargs):
        calls["baseline"] += 1
        assert Path(args[0]).resolve() == baseline_path.resolve()
        assert Path(kwargs["inventory_completion_path"]).resolve() == (
            validation_inventory_path.resolve()
        )
        assert kwargs["source"] is source
        assert kwargs["scratch_runtime"] is scratch
        assert kwargs["successor_cache"] is not None
        assert Path(kwargs["evaluation_environment_receipt_path"]).resolve() == (
            validation_environment_path.resolve()
        )
        return SimpleNamespace(
            result=baseline_result,
            completion=baseline_completion,
        )

    monkeypatch.setattr(contracts, "load_semantic_p50_source_inventory", load_source)
    monkeypatch.setattr(
        contracts, "validate_semantic_gate_zero_evidence_receipt", open_gate
    )
    monkeypatch.setattr(contracts, "validate_semantic_t1_capacity_decision", open_t1)
    monkeypatch.setattr(
        contracts, "load_semantic_capability_cell_registry", lambda: object()
    )
    monkeypatch.setattr(
        contracts,
        "validate_semantic_p50_prerequisite_relationships",
        relationships,
    )
    monkeypatch.setattr(contracts, "load_semantic_p50_prepared_recipe", load_prepared)
    monkeypatch.setattr(
        contracts,
        "expected_semantic_p50_successor_cache_identity_from_reopened_prerequisites",
        expected_cache_identity,
    )
    monkeypatch.setattr(contracts, "open_semantic_p50_successor_cache", open_cache)
    monkeypatch.setattr(
        contracts, "open_semantic_p50_validation_baseline_from_paths", open_baseline
    )
    monkeypatch.setattr(
        contracts, "_exact_requested_source_states", lambda **kwargs: {}
    )
    monkeypatch.setattr(
        contracts,
        "SemanticP50RuntimeInputs",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    return SimpleNamespace(
        artifact_root=artifact_root,
        repo=repo,
        trainer=trainer,
        image_definition=image_definition,
        scratch=scratch,
        model_runtime=model_runtime,
        source_binding=source_binding,
        source_revision=source_revision,
        paths=paths,
        prepared=prepared,
        prepared_path=prepared_path,
        cache=cache,
        cache_path=cache_path,
        baseline_result=baseline_result,
        calls=calls,
    )


def _runtime(fixture) -> tuple[dict[str, object], Path]:
    runtime = contracts.build_semantic_p50_runtime_contract(
        scratch_runtime=fixture.scratch,
        model_runtime_identity=fixture.model_runtime,
        prerequisite_paths=fixture.paths,
        expected_source_binding=fixture.source_binding,
        trainer_source_path=fixture.trainer,
    )
    path = _write(
        fixture.artifact_root / "contracts" / contracts.RUNTIME_FILENAME,
        runtime,
    )
    return runtime, path


def _hardware() -> dict[str, object]:
    return {
        "accelerator_class": "gpu",
        "modal_gpu_type": "A10G",
        "device_type": "cuda",
        "device_name": "NVIDIA A10G",
        "device_capability": "8.6",
        "cuda_device_count": 1,
        "cpu_count": 8,
        "memory_mb": 65536,
    }


def _software() -> dict[str, str]:
    return {
        "python_version": "3.11.9",
        "torch_version": "2.4.0",
        "cuda_version": "12.1",
        "cudnn_version": "90100",
        "rdkit_version": "2024.03.5",
        "numpy_version": "1.26.4",
    }


def _environment(fixture, runtime_path: Path) -> tuple[dict[str, object], Path]:
    image_sha = _digest("image-content")
    environment = contracts.build_semantic_p50_environment_contract(
        runtime_contract_path=runtime_path,
        scratch_runtime=fixture.scratch,
        prerequisite_paths=fixture.paths,
        expected_source_binding=fixture.source_binding,
        trainer_source_path=fixture.trainer,
        image_reference=f"registry.invalid/compose@sha256:{image_sha}",
        image_definition_path=fixture.image_definition,
        expected_hardware=_hardware(),
        expected_software=_software(),
    )
    path = _write(
        fixture.artifact_root / "contracts" / contracts.ENVIRONMENT_FILENAME,
        environment,
    )
    return environment, path


def _physical_kwargs(fixture, runtime_path: Path, environment_path: Path):
    return {
        "runtime_contract_path": runtime_path,
        "environment_contract_path": environment_path,
        "scratch_runtime": fixture.scratch,
        "prerequisite_paths": fixture.paths,
        "expected_source_binding": fixture.source_binding,
        "trainer_source_path": fixture.trainer,
        "image_definition_path": fixture.image_definition,
    }


def _launch(fixture):
    _runtime_value, runtime_path = _runtime(fixture)
    environment, environment_path = _environment(fixture, runtime_path)
    kwargs = _physical_kwargs(fixture, runtime_path, environment_path)
    projection = contracts.build_semantic_p50_launch_projection(
        **kwargs,
        output_prefix_relative="editing_v2/semantic_p50",
    )
    projection_path = _write(
        fixture.artifact_root / "contracts" / contracts.LAUNCH_FILENAME,
        projection,
    )
    return environment, projection, projection_path, kwargs


def _reseal(value: dict[str, object], field: str) -> None:
    body = dict(value)
    body.pop(field)
    value[field] = _sha(body)


def test_contract_publication_rejects_same_byte_symlink(tmp_path: Path) -> None:
    payload = {"schema": "compose.test.contract", "schema_version": 1}
    target = tmp_path / "target.json"
    target.write_bytes(contracts.canonical_semantic_p50_contract_bytes(payload))
    destination = tmp_path / "contract.json"
    destination.symlink_to(target)

    with pytest.raises(
        contracts.SemanticP50ExecutionContractError,
        match="immutable artifact collision",
    ):
        contracts._publish_once(destination, payload)


def test_generic_self_hash_binding_is_retired_and_all_domain_openers_run(
    physical_fixture,
) -> None:
    with pytest.raises(
        contracts.SemanticP50ExecutionContractError,
        match="generic self-hash prerequisite binding is forbidden",
    ):
        contracts.bind_canonical_semantic_p50_artifact()
    opened = contracts.open_semantic_p50_execution_prerequisites(
        physical_fixture.paths,
        scratch_runtime=physical_fixture.scratch,
        expected_source_binding=physical_fixture.source_binding,
        expected_source_revision_sha256=physical_fixture.source_revision[
            "source_revision_sha256"
        ],
    )
    assert {item.role for item in opened.bindings} == set(
        contracts.UPSTREAM_RUNTIME_ROLES
    )
    assert all(physical_fixture.calls[name] == 1 for name in physical_fixture.calls)


def test_runtime_reopens_lineage_and_rehashes_scratch_state(physical_fixture) -> None:
    runtime, _path = _runtime(physical_fixture)
    assert runtime["optimizer"] == contracts.OPTIMIZER_CONTRACT
    assert runtime["training"] == contracts.TRAINING_CONTRACT
    assert runtime["source_revision"] == contracts.observe_clean_source_revision(
        physical_fixture.repo
    )
    assert physical_fixture.calls["cache"] >= 2

    with torch.no_grad():
        next(physical_fixture.scratch.model.parameters()).add_(1.0)
    with pytest.raises(
        contracts.SemanticP50ExecutionContractError,
        match="scratch model",
    ):
        contracts.validate_semantic_p50_runtime_contract(
            runtime,
            scratch_runtime=physical_fixture.scratch,
            prerequisite_paths=physical_fixture.paths,
            expected_source_binding=physical_fixture.source_binding,
            trainer_source_path=physical_fixture.trainer,
        )


def test_rehashed_prepared_cross_lineage_forgery_is_rejected(
    physical_fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forged = copy.deepcopy(physical_fixture.prepared)
    forged["prerequisites"]["scratch_initial_model_state_sha256"] = _digest(
        "forged-scratch"
    )
    _reseal(forged, "prepared_recipe_sha256")
    physical_fixture.prepared_path.write_bytes(_bytes(forged))
    monkeypatch.setattr(
        contracts,
        "load_semantic_p50_prepared_recipe",
        lambda path: (forged, hashlib.sha256(Path(path).read_bytes()).hexdigest()),
    )
    with pytest.raises(
        contracts.SemanticP50ExecutionContractError,
        match="physically reconstructed",
    ):
        contracts.open_semantic_p50_execution_prerequisites(
            physical_fixture.paths,
            scratch_runtime=physical_fixture.scratch,
            expected_source_binding=physical_fixture.source_binding,
            expected_source_revision_sha256=physical_fixture.source_revision[
                "source_revision_sha256"
            ],
        )


def test_missing_purpose_specific_cache_opener_fails_closed(
    physical_fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*args, **kwargs):
        raise contracts.SemanticP50ExecutionContractError(
            "purpose-specific successor cache opener unavailable"
        )

    monkeypatch.setattr(contracts, "open_semantic_p50_successor_cache", unavailable)
    with pytest.raises(
        contracts.SemanticP50ExecutionContractError,
        match="opener unavailable",
    ):
        contracts.open_semantic_p50_execution_prerequisites(
            physical_fixture.paths,
            scratch_runtime=physical_fixture.scratch,
            expected_source_binding=physical_fixture.source_binding,
            expected_source_revision_sha256=physical_fixture.source_revision[
                "source_revision_sha256"
            ],
        )


def test_validation_baseline_must_name_reopened_cache_and_scratch(
    physical_fixture,
) -> None:
    physical_fixture.baseline_result["evaluated_model_state_sha256"] = _digest(
        "another-scratch"
    )
    with pytest.raises(
        contracts.SemanticP50ExecutionContractError,
        match="validation baseline differs",
    ):
        contracts.open_semantic_p50_execution_prerequisites(
            physical_fixture.paths,
            scratch_runtime=physical_fixture.scratch,
            expected_source_binding=physical_fixture.source_binding,
            expected_source_revision_sha256=physical_fixture.source_revision[
                "source_revision_sha256"
            ],
        )


def test_launch_reopens_physical_artifacts_and_requires_exact_argv(
    physical_fixture,
) -> None:
    _environment_value, projection, _projection_path, kwargs = _launch(physical_fixture)
    assert projection["argv_schema"] == contracts.ARGV_SCHEMA
    assert projection["argv"][-2:] == ["--hazard-weight", "0.0"]

    forged = copy.deepcopy(projection)
    forged["argv"][forged["argv"].index("50")] = "500"
    forged["argv_sha256"] = _sha(forged["argv"])
    _reseal(forged, "projection_sha256")
    with pytest.raises(contracts.SemanticP50ExecutionContractError, match="argv"):
        contracts.validate_semantic_p50_launch_projection(forged, **kwargs)

    physical_fixture.cache_path.write_bytes(
        physical_fixture.cache_path.read_bytes() + b"\n"
    )
    with pytest.raises(contracts.SemanticP50ExecutionContractError):
        contracts.validate_semantic_p50_launch_projection(projection, **kwargs)


def test_execution_permit_is_content_addressed_and_p50_only(
    physical_fixture,
) -> None:
    _environment, projection, projection_path, kwargs = _launch(physical_fixture)
    permit_path = contracts.materialize_semantic_p50_execution_permit(
        output_root=physical_fixture.artifact_root / "permits",
        prepared=physical_fixture.prepared,
        launch_projection_path=projection_path,
        **kwargs,
    )

    verified = contracts.open_semantic_p50_execution_permit(
        permit_path,
        prepared=physical_fixture.prepared,
        launch_projection_path=projection_path,
        **kwargs,
    )
    authorized = authorize_semantic_p50_recipe(
        prepared=physical_fixture.prepared,
        permit_path=permit_path,
        launch_projection_path=projection_path,
        **kwargs,
    )

    assert permit_path.parent.name == verified.permit["permit_sha256"]
    assert authorized.path == permit_path.resolve()
    assert verified.permit["training_authorized"] is True
    assert verified.permit["bounded_p50_authorized"] is True
    assert verified.permit["p500_authorized"] is False
    assert verified.permit["checkpoint_selection_authorized"] is False
    assert verified.permit["unresolved_physical_bindings"] == []
    refinement = verified.permit["editing_training_gate_refinement"]
    assert refinement["base_status"] == "DESIGN_NOT_TRAINING_AUTHORIZED"
    assert refinement["base_bounded_p50_authorized"] is False
    assert refinement["resolved_status"] == "FROZEN_BOUNDED_P50_AUTHORIZED"
    assert refinement["resolved_bounded_p50_authorized"] is True
    assert refinement["p500_authorized"] is False
    assert set(
        refinement["numeric_thresholds"][
            "maximum_required_slice_successor_nll_regression"
        ].values()
    ) == {1e-7}
    assert verified.permit["resolved_physical_binding_purposes"] == list(
        contracts.REQUIRED_BINDING_PURPOSES
    )
    assert verified.permit["run_identity_sha256"] == projection["run_identity_sha256"]

    physical_fixture.baseline_result["evaluated_model_state_sha256"] = _digest(
        "another-scratch"
    )
    with pytest.raises(
        contracts.SemanticP50ExecutionContractError,
        match="validation baseline differs",
    ):
        contracts.open_semantic_p50_execution_permit(
            permit_path,
            prepared=physical_fixture.prepared,
            launch_projection_path=projection_path,
            **kwargs,
        )


def test_remote_reopen_uses_bound_revision_and_exact_bytes_without_git(
    physical_fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _environment, projection, projection_path, kwargs = _launch(physical_fixture)
    permit_path = contracts.materialize_semantic_p50_execution_permit(
        output_root=physical_fixture.artifact_root / "remote-permits",
        prepared=physical_fixture.prepared,
        launch_projection_path=projection_path,
        **kwargs,
    )

    def forbid_git(*args, **kwargs):
        raise AssertionError("remote physical reopen must not invoke Git")

    monkeypatch.setattr(contracts, "_git", forbid_git)
    remote_kwargs = {
        **kwargs,
        "expected_source_revision": physical_fixture.source_revision,
    }
    reopened_projection = contracts.open_semantic_p50_launch_projection(
        projection_path,
        **remote_kwargs,
    )
    reopened_permit = contracts.open_semantic_p50_execution_permit(
        permit_path,
        prepared=physical_fixture.prepared,
        launch_projection_path=projection_path,
        **remote_kwargs,
    )

    assert reopened_projection == projection
    assert (
        reopened_permit.permit["run_identity_sha256"]
        == projection["run_identity_sha256"]
    )

    gate_path = physical_fixture.repo / "configs/editing_training_v2_gate.json"
    gate = json.loads(gate_path.read_bytes())
    gate["bounded_p50_authorized"] = True
    gate["status"] = "FROZEN_BOUNDED_P50_AUTHORIZED"
    gate_path.write_bytes(_bytes(gate))
    with pytest.raises(
        contracts.SemanticP50ExecutionContractError,
        match="base Editing-V2 training gate",
    ):
        contracts.open_semantic_p50_execution_permit(
            permit_path,
            prepared=physical_fixture.prepared,
            launch_projection_path=projection_path,
            **remote_kwargs,
        )


def test_remote_builders_use_bound_revision_and_physical_bytes_without_git(
    physical_fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision = physical_fixture.source_revision

    def forbid_git(*args, **kwargs):
        raise AssertionError("remote contract construction must not invoke Git")

    monkeypatch.setattr(contracts, "_git", forbid_git)
    runtime = contracts.build_semantic_p50_runtime_contract(
        scratch_runtime=physical_fixture.scratch,
        model_runtime_identity=physical_fixture.model_runtime,
        prerequisite_paths=physical_fixture.paths,
        expected_source_binding=physical_fixture.source_binding,
        trainer_source_path=physical_fixture.trainer,
        expected_source_revision=revision,
    )
    runtime_path = _write(
        physical_fixture.artifact_root
        / "remote-contracts"
        / contracts.RUNTIME_FILENAME,
        runtime,
    )
    environment = contracts.build_semantic_p50_environment_contract(
        runtime_contract_path=runtime_path,
        scratch_runtime=physical_fixture.scratch,
        prerequisite_paths=physical_fixture.paths,
        expected_source_binding=physical_fixture.source_binding,
        trainer_source_path=physical_fixture.trainer,
        image_reference=f"registry.invalid/compose@sha256:{_digest('remote-image')}",
        image_definition_path=physical_fixture.image_definition,
        expected_hardware=_hardware(),
        expected_software=_software(),
        expected_source_revision=revision,
    )
    environment_path = _write(
        physical_fixture.artifact_root
        / "remote-contracts"
        / contracts.ENVIRONMENT_FILENAME,
        environment,
    )
    projection = contracts.build_semantic_p50_launch_projection(
        runtime_contract_path=runtime_path,
        environment_contract_path=environment_path,
        scratch_runtime=physical_fixture.scratch,
        prerequisite_paths=physical_fixture.paths,
        expected_source_binding=physical_fixture.source_binding,
        trainer_source_path=physical_fixture.trainer,
        image_definition_path=physical_fixture.image_definition,
        output_prefix_relative="editing_v2/semantic_p50",
        expected_source_revision=revision,
    )

    assert runtime["source_revision"] == revision
    assert environment["source_revision_sha256"] == revision["source_revision_sha256"]
    assert projection["source_revision"] == revision


def test_remote_observed_receipt_reopens_with_bound_revision_without_git(
    physical_fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment, _projection, projection_path, kwargs = _launch(physical_fixture)
    observed = {
        "image_content_sha256": environment["image"]["content_sha256"],
        "hardware": environment["hardware"],
        "software": environment["software"],
    }
    monkeypatch.setattr(contracts, "_physical_observed_environment", lambda: observed)
    remote_kwargs = {
        **kwargs,
        "expected_source_revision": physical_fixture.source_revision,
    }
    receipt = contracts.build_semantic_p50_observed_environment_receipt(
        launch_projection_path=projection_path,
        observed_at_utc="2026-08-01T12:00:00Z",
        **remote_kwargs,
    )
    receipt_path = contracts.publish_semantic_p50_observed_environment_receipt(
        physical_fixture.artifact_root
        / "remote-contracts"
        / contracts.ENVIRONMENT_RECEIPT_FILENAME,
        receipt,
    )

    def forbid_git(*args, **kwargs):
        raise AssertionError("remote observed-environment reopen must not invoke Git")

    monkeypatch.setattr(contracts, "_git", forbid_git)
    assert (
        contracts.open_semantic_p50_observed_environment_receipt(
            receipt_path,
            launch_projection_path=projection_path,
            **remote_kwargs,
        )
        == receipt
    )


def test_source_tree_and_image_definition_are_observed_not_asserted(
    physical_fixture,
) -> None:
    runtime, runtime_path = _runtime(physical_fixture)
    assert runtime["source_revision"]["commit"] == _git(
        physical_fixture.repo, "rev-parse", "HEAD"
    )
    physical_fixture.image_definition.write_text("# dirty image\n", encoding="utf-8")
    with pytest.raises(
        contracts.SemanticP50ExecutionContractError,
        match="clean committed source tree",
    ):
        contracts.build_semantic_p50_environment_contract(
            runtime_contract_path=runtime_path,
            scratch_runtime=physical_fixture.scratch,
            prerequisite_paths=physical_fixture.paths,
            expected_source_binding=physical_fixture.source_binding,
            trainer_source_path=physical_fixture.trainer,
            image_reference=f"registry.invalid/compose@sha256:{_digest('image')}",
            image_definition_path=physical_fixture.image_definition,
            expected_hardware=_hardware(),
            expected_software=_software(),
        )


def test_observed_environment_is_collected_published_and_physically_bound(
    physical_fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment, _projection, projection_path, kwargs = _launch(physical_fixture)
    observed = {
        "image_content_sha256": environment["image"]["content_sha256"],
        "hardware": environment["hardware"],
        "software": environment["software"],
    }
    monkeypatch.setattr(contracts, "_physical_observed_environment", lambda: observed)
    receipt = contracts.build_semantic_p50_observed_environment_receipt(
        launch_projection_path=projection_path,
        observed_at_utc="2026-08-01T12:00:00Z",
        **kwargs,
    )
    receipt_path = contracts.publish_semantic_p50_observed_environment_receipt(
        physical_fixture.artifact_root
        / "contracts"
        / contracts.ENVIRONMENT_RECEIPT_FILENAME,
        receipt,
    )
    reopened = contracts.open_semantic_p50_observed_environment_receipt(
        receipt_path,
        launch_projection_path=projection_path,
        **kwargs,
    )
    assert reopened == receipt

    mutated = copy.deepcopy(receipt)
    mutated["observed_software"]["torch_version"] = "forged"
    _reseal(mutated, "receipt_sha256")
    receipt_path.write_bytes(_bytes(mutated))
    with pytest.raises(
        contracts.SemanticP50ExecutionContractError,
        match="physical binding disagrees",
    ):
        contracts.open_semantic_p50_observed_environment_receipt(
            receipt_path,
            launch_projection_path=projection_path,
            reobserve=False,
            **kwargs,
        )


def test_output_is_atomically_reserved_after_observed_preflight(
    physical_fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment, projection, projection_path, kwargs = _launch(physical_fixture)
    observed = {
        "image_content_sha256": environment["image"]["content_sha256"],
        "hardware": environment["hardware"],
        "software": environment["software"],
    }
    monkeypatch.setattr(contracts, "_physical_observed_environment", lambda: observed)
    receipt = contracts.build_semantic_p50_observed_environment_receipt(
        launch_projection_path=projection_path,
        observed_at_utc="2026-08-01T12:00:00Z",
        **kwargs,
    )
    receipt_path = contracts.publish_semantic_p50_observed_environment_receipt(
        physical_fixture.artifact_root
        / "contracts"
        / contracts.ENVIRONMENT_RECEIPT_FILENAME,
        receipt,
    )
    permit_path = contracts.materialize_semantic_p50_execution_permit(
        output_root=physical_fixture.artifact_root / "reservation-permits",
        prepared=physical_fixture.prepared,
        launch_projection_path=projection_path,
        **kwargs,
    )
    destination = contracts.require_semantic_p50_output_available(
        projection_path,
        execution_permit_path=permit_path,
        environment_receipt_path=receipt_path,
        artifact_root=physical_fixture.artifact_root,
        **kwargs,
    )
    assert destination.name == projection["run_identity_sha256"]
    assert (destination / contracts.RESERVATION_FILENAME).is_file()
    with pytest.raises(contracts.SemanticP50ExecutionContractError, match="collision"):
        contracts.require_semantic_p50_output_available(
            projection_path,
            execution_permit_path=permit_path,
            environment_receipt_path=receipt_path,
            artifact_root=physical_fixture.artifact_root,
            **kwargs,
        )
