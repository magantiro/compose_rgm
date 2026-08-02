"""Focused fail-closed tests for the one-shot semantic P50 Modal launcher."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from compose_v4.data.immutable_artifact import (
    ImmutableArtifactError,
    write_bytes_if_absent,
)
from modal_apps import run_editing_v2_semantic_p50_app as launcher


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _clean_source_fixture(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "source"
    (root / "modal_apps").mkdir(parents=True)
    (root / "src" / "compose_v4").mkdir(parents=True)
    (root / "configs").mkdir(parents=True)
    (root / launcher.LAUNCHER_SOURCE).write_text(
        "# exact launcher fixture\n", encoding="utf-8"
    )
    (root / "src" / "compose_v4" / "runtime.py").write_text(
        "VALUE = 1\n", encoding="utf-8"
    )
    (root / "configs" / "editing_training_v2_gate.json").write_text(
        "{}\n", encoding="utf-8"
    )
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "p50@example.invalid")
    _git(root, "config", "user.name", "Semantic P50 Test")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "test: freeze P50 launcher source")
    return root, _git(root, "rev-parse", "HEAD")


def test_local_attestation_hashes_every_serialized_tracked_byte(
    tmp_path: Path,
) -> None:
    root, commit = _clean_source_fixture(tmp_path)
    revision = launcher.local_source_revision(
        expected_commit=commit,
        repo_root=root,
    )
    hashes = revision["serialized_source_hashes"]
    assert set(hashes) == {
        launcher.LAUNCHER_SOURCE,
        "src/compose_v4/runtime.py",
        "configs/editing_training_v2_gate.json",
    }
    assert revision["execution_source_revision"]["commit"] == commit
    assert revision["execution_source_revision"]["tree"] == _git(
        root, "rev-parse", "HEAD^{tree}"
    )
    assert revision["image_content_sha256"] == launcher._image_content_sha256(hashes)

    (root / "src" / "compose_v4" / "runtime.py").write_text(
        "VALUE = 2\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="clean committed worktree"):
        launcher.local_source_revision(expected_commit=commit, repo_root=root)


def test_remote_byte_validation_needs_no_git_and_rejects_mutation(
    tmp_path: Path,
) -> None:
    root, commit = _clean_source_fixture(tmp_path)
    revision = launcher.local_source_revision(
        expected_commit=commit,
        repo_root=root,
    )
    assert launcher._validate_source_revision(revision, remote_root=root) == revision

    (root / "configs" / "editing_training_v2_gate.json").write_text(
        '{"changed":true}\n', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="serialized source revision disagrees"):
        launcher._validate_source_revision(revision, remote_root=root)


def test_remote_source_validation_rejects_forged_nested_execution_identity(
    tmp_path: Path,
) -> None:
    root, commit = _clean_source_fixture(tmp_path)
    revision = launcher.local_source_revision(
        expected_commit=commit,
        repo_root=root,
    )
    forged = dict(revision)
    forged_execution = dict(revision["execution_source_revision"])
    forged_execution["commit"] = "f" * 40
    forged["execution_source_revision"] = forged_execution
    body = dict(forged)
    body.pop("source_revision_sha256")
    forged["source_revision_sha256"] = launcher._sha(body)

    with pytest.raises(RuntimeError, match="execution-source revision disagrees"):
        launcher._validate_source_revision(forged, remote_root=root)


def test_source_validation_rejects_v1_and_mixed_producer_namespaces(
    tmp_path: Path,
) -> None:
    root, commit = _clean_source_fixture(tmp_path)
    revision = launcher.local_source_revision(expected_commit=commit, repo_root=root)
    legacy = dict(revision)
    legacy["schema_version"] = 1
    legacy_body = dict(legacy)
    legacy_body.pop("source_revision_sha256")
    legacy["source_revision_sha256"] = launcher._sha(legacy_body)
    with pytest.raises(RuntimeError, match="serialized source revision disagrees"):
        launcher._validate_source_revision(legacy, remote_root=root)

    mixed = dict(revision)
    mixed["execution_source_revision"] = dict(revision)
    mixed_body = dict(mixed)
    mixed_body.pop("source_revision_sha256")
    mixed["source_revision_sha256"] = launcher._sha(mixed_body)
    with pytest.raises(RuntimeError, match="execution-source revision disagrees"):
        launcher._validate_source_revision(mixed, remote_root=root)


def test_serialized_inventory_exactly_matches_copied_sources() -> None:
    expected = {launcher.LAUNCHER_SOURCE}
    for directory in launcher.IMAGE_SOURCE_DIRECTORIES:
        expected.update(
            path.relative_to(launcher.ROOT).as_posix()
            for path in sorted((launcher.ROOT / directory).rglob("*"))
            if path.is_file()
            and path.suffix != ".pyc"
            and "__pycache__" not in path.parts
        )
    assert set(launcher._serialized_source_paths(launcher.ROOT)) == expected


@pytest.mark.parametrize(
    ("action", "paths", "prefix", "match"),
    (
        ("preflight", ("runtime", "", "", ""), "editing_v2/semantic_p50", "forbids"),
        (
            "run",
            ("runtime", "environment", "", "permit"),
            "editing_v2/semantic_p50",
            "requires",
        ),
        (
            "run",
            ("runtime", "environment", "launch", "permit"),
            "changed",
            "output prefix",
        ),
        ("both", ("", "", "", ""), "editing_v2/semantic_p50", "exactly"),
    ),
)
def test_action_modes_fail_closed(
    action: str,
    paths: tuple[str, str, str, str],
    prefix: str,
    match: str,
) -> None:
    with pytest.raises(RuntimeError, match=match):
        launcher._validate_action_inputs(
            action=action,
            output_prefix_relative=prefix,
            runtime_contract=paths[0],
            environment_contract=paths[1],
            launch_projection=paths[2],
            execution_permit=paths[3],
        )

    assert (
        launcher._validate_action_inputs(
            action="preflight",
            output_prefix_relative="editing_v2/semantic_p50",
            runtime_contract="",
            environment_contract="",
            launch_projection="",
            execution_permit="",
        )
        == "preflight"
    )
    assert (
        launcher._validate_action_inputs(
            action="run",
            output_prefix_relative="editing_v2/semantic_p50",
            runtime_contract="runtime",
            environment_contract="environment",
            launch_projection="launch",
            execution_permit="permit",
        )
        == "run"
    )


def test_result_environment_rejects_amp_tf32_or_downstream_authority() -> None:
    observed = {
        "observed_hardware": {"device_name": "NVIDIA A10G"},
        "observed_software": {
            "python_version": "3.11.9",
            "torch_version": "2.4.0+cu121",
            "cuda_version": "12.1",
        },
    }
    result = {
        "execution_environment": {
            "device_type": "cuda",
            "device_name": "NVIDIA A10G",
            "dtype": "torch.float32",
            "mixed_precision": False,
            "cuda_matmul_tf32_allowed": False,
            "cudnn_tf32_allowed": False,
            "deterministic_algorithms_enabled": True,
            "python_version": "3.11.9",
            "torch_version": "2.4.0+cu121",
            "cuda_version": "12.1",
        },
        "learning_demonstrated": False,
        "next_stage_authorized": None,
        "p500_authorized": False,
        "checkpoint_selection_authorized": False,
        "final_test_selection_authorized": False,
    }
    launcher._assert_result_environment(result, observed_receipt=observed)
    result["execution_environment"]["mixed_precision"] = True
    with pytest.raises(RuntimeError, match="environment or downstream authority"):
        launcher._assert_result_environment(result, observed_receipt=observed)


def test_contract_publication_is_content_addressed_and_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    monkeypatch.setattr(launcher, "ARTIFACT_ROOT", artifact_root)
    identity = "a" * 64
    payload = {"contract_sha256": identity, "value": 1}
    loaded = {
        "canonical_contract_bytes": lambda value: launcher._canonical_bytes(
            value, newline=True
        ),
        "ImmutableArtifactError": ImmutableArtifactError,
        "write_bytes_if_absent": write_bytes_if_absent,
    }
    path = launcher._publish_content_addressed_contract(
        kind="runtime",
        identity_sha256=identity,
        filename="SEMANTIC_P50_RUNTIME_CONTRACT.json",
        payload=payload,
        loaded=loaded,
    )
    assert path == launcher._content_addressed_contract_path(
        kind="runtime",
        identity_sha256=identity,
        filename="SEMANTIC_P50_RUNTIME_CONTRACT.json",
    )
    assert path.read_bytes() == launcher._canonical_bytes(payload, newline=True)
    assert (
        launcher._publish_content_addressed_contract(
            kind="runtime",
            identity_sha256=identity,
            filename="SEMANTIC_P50_RUNTIME_CONTRACT.json",
            payload=payload,
            loaded=loaded,
        )
        == path
    )
    with pytest.raises(RuntimeError, match="immutable semantic P50 contract collision"):
        launcher._publish_content_addressed_contract(
            kind="runtime",
            identity_sha256=identity,
            filename="SEMANTIC_P50_RUNTIME_CONTRACT.json",
            payload={**payload, "value": 2},
            loaded=loaded,
        )

    symlink_identity = "b" * 64
    symlink_path = launcher._content_addressed_contract_path(
        kind="runtime",
        identity_sha256=symlink_identity,
        filename="SEMANTIC_P50_RUNTIME_CONTRACT.json",
    )
    symlink_path.parent.mkdir(parents=True)
    target = tmp_path / "same-bytes.json"
    target.write_bytes(launcher._canonical_bytes(payload, newline=True))
    symlink_path.symlink_to(target)
    with pytest.raises(RuntimeError, match="symbolic link"):
        launcher._publish_content_addressed_contract(
            kind="runtime",
            identity_sha256=symlink_identity,
            filename="SEMANTIC_P50_RUNTIME_CONTRACT.json",
            payload=payload,
            loaded=loaded,
        )


def test_content_address_namespace_rejects_symlinked_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts"
    namespace = artifact_root / launcher.PREFLIGHT_CONTRACT_ROOT_RELATIVE
    namespace.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (namespace / "runtime").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(launcher, "ARTIFACT_ROOT", artifact_root)
    with pytest.raises(RuntimeError, match="symbolic link"):
        launcher._content_addressed_contract_path(
            kind="runtime",
            identity_sha256="c" * 64,
            filename="SEMANTIC_P50_RUNTIME_CONTRACT.json",
        )


def test_thirteen_file_snapshot_detects_preflight_input_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    monkeypatch.setattr(launcher, "ARTIFACT_ROOT", artifact_root)
    mounted = {}
    for index, field in enumerate(launcher._PREREQUISITE_ADDRESS_FIELDS.values()):
        path = artifact_root / f"input-{index}.json"
        path.write_bytes(f"{field}\n".encode())
        mounted[field] = path
    before = launcher._snapshot_prerequisite_files(mounted)
    assert len(before) == 13
    changed_field = next(iter(mounted))
    mounted[changed_field].write_bytes(b"mutated\n")
    after = launcher._snapshot_prerequisite_files(mounted)
    with pytest.raises(RuntimeError, match="changed during preflight"):
        launcher._require_unchanged_prerequisite_snapshot(before, after)


def test_a10g_preflight_publishes_and_reopens_in_order_without_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "artifacts"
    remote_root = tmp_path / "remote"
    artifact_root.mkdir()
    remote_root.mkdir()
    monkeypatch.setattr(launcher, "ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(launcher, "REMOTE_ROOT", remote_root)

    events: list[str] = []

    class Volume:
        def reload(self) -> None:
            events.append("volume.reload")

        def commit(self) -> None:
            events.append("volume.commit")

    monkeypatch.setattr(launcher, "artifact_volume", Volume())
    snapshot_calls = 0

    def snapshot(_mounted):
        nonlocal snapshot_calls
        snapshot_calls += 1
        events.append("inputs.snapshot")
        return {"all_13_inputs": {"file_sha256": "1" * 64}}

    monkeypatch.setattr(launcher, "_snapshot_prerequisite_files", snapshot)
    execution_revision = {"source_revision_sha256": "e" * 64}
    source_revision = {
        "source_revision_sha256": "s" * 64,
        "execution_source_revision": execution_revision,
        "image_content_sha256": launcher._LOCAL_IMAGE_CONTENT_SHA256,
    }
    monkeypatch.setattr(
        launcher, "_validate_source_revision", lambda value: source_revision
    )

    class Parameter:
        device = SimpleNamespace(type="cpu")

    class Model:
        def parameters(self):
            return (Parameter(),)

        def to(self, *args, **kwargs):
            raise AssertionError("validation-only preflight must not move the model")

    scratch = SimpleNamespace(model=Model())
    model_runtime = {"identity_sha256": "m" * 64}
    monkeypatch.setattr(
        launcher,
        "_scratch_runtime",
        lambda *args, **kwargs: (scratch, model_runtime),
    )
    monkeypatch.setattr(launcher, "_source_binding", lambda *args, **kwargs: object())

    image = {
        "reference": launcher.IMAGE_REFERENCE,
        "content_sha256": launcher._LOCAL_IMAGE_CONTENT_SHA256,
        "definition_relative_path": launcher.LAUNCHER_SOURCE,
    }
    runtime = {"contract_sha256": "a" * 64}
    environment = {"contract_sha256": "b" * 64, "image": image}
    launch = {"projection_sha256": "c" * 64, "image": image}
    permit = {
        "permit_sha256": "d" * 64,
        "optimizer_steps": 50,
        "batch_size": 64,
        "resume": False,
        "training_authorized": True,
        "bounded_p50_authorized": True,
        "p500_authorized": False,
        "checkpoint_selection_authorized": False,
        "final_test_selection_authorized": False,
    }
    permit_path = (
        artifact_root
        / launcher.PREFLIGHT_CONTRACT_ROOT_RELATIVE
        / "permit"
        / permit["permit_sha256"]
        / "SEMANTIC_P50_EXECUTION_PERMIT.json"
    ).resolve()
    prerequisites = SimpleNamespace(
        t1_decision={
            "bounded_p50_authorized": True,
            "p500_authorized": False,
            "checkpoint_selection_authorized": False,
            "final_test_selection_authorized": False,
        }
    )

    def build(name: str, payload: dict[str, object]):
        def invoke(**kwargs):
            events.append(f"{name}.build")
            return payload

        return invoke

    def opened(name: str, payload: dict[str, object]):
        def invoke(*args, **kwargs):
            events.append(f"{name}.open")
            return payload

        return invoke

    def publish_contract(**kwargs):
        kind = kwargs["kind"]
        events.append(f"{kind}.publish")
        return launcher._content_addressed_contract_path(
            kind=kind,
            identity_sha256=kwargs["identity_sha256"],
            filename=kwargs["filename"],
        )

    monkeypatch.setattr(
        launcher, "_publish_content_addressed_contract", publish_contract
    )

    def open_prerequisites(*args, **kwargs):
        events.append("prerequisites.open")
        return prerequisites

    def observe_environment():
        events.append("environment.observe")
        return {
            "image_content_sha256": launcher._LOCAL_IMAGE_CONTENT_SHA256,
            "hardware": {
                "accelerator_class": "gpu",
                "modal_gpu_type": "A10G",
                "device_type": "cuda",
                "cuda_device_count": 1,
            },
            "software": {"torch_version": "observed"},
        }

    def materialize_permit(**kwargs):
        events.append("permit.publish")
        return permit_path

    def open_permit(*args, **kwargs):
        events.append("permit.open")
        return SimpleNamespace(
            path=permit_path,
            permit=permit,
            launch_projection=launch,
        )

    loaded = {
        "SemanticP50ExecutionPrerequisitePaths": lambda **kwargs: SimpleNamespace(
            **kwargs
        ),
        "RUNTIME_FILENAME": "SEMANTIC_P50_RUNTIME_CONTRACT.json",
        "ENVIRONMENT_FILENAME": "SEMANTIC_P50_ENVIRONMENT_CONTRACT.json",
        "LAUNCH_FILENAME": "SEMANTIC_P50_LAUNCH_PROJECTION.json",
        "PERMIT_FILENAME": "SEMANTIC_P50_EXECUTION_PERMIT.json",
        "load_prepared_recipe": lambda path: ({"prerequisites": {}}, "f" * 64),
        "open_execution_prerequisites": open_prerequisites,
        "observe_physical_environment": observe_environment,
        "build_runtime_contract": build("runtime", runtime),
        "open_runtime_contract": opened("runtime", runtime),
        "build_environment_contract": build("environment", environment),
        "open_environment_contract": opened("environment", environment),
        "build_launch_projection": build("launch", launch),
        "open_launch_projection": opened("launch", launch),
        "materialize_execution_permit": materialize_permit,
        "open_execution_permit": open_permit,
    }
    monkeypatch.setattr(launcher, "_preflight_imports", lambda: loaded)
    addresses = {
        "source_inventory": "/artifacts/source.json",
        "migration_completion": "/artifacts/migration.json",
        "chunk_cache_plan": "/artifacts/chunk-plan.json",
        "chunk_cache_global_completion": "/artifacts/chunk-complete.json",
        "decision_plan": "/artifacts/decision-plan.json",
        "decision_completion": "/artifacts/decision-complete.json",
        "gate_zero_evidence": "/artifacts/gate.json",
        "t1_decision": "/artifacts/t1.json",
        "prepared_recipe": "/artifacts/prepared.json",
        "successor_cache_completion": "/artifacts/cache.json",
        "validation_inventory_completion": "/artifacts/validation-inventory.json",
        "validation_baseline_completion": "/artifacts/baseline.json",
        "validation_evaluation_environment_receipt": "/artifacts/baseline-env.json",
    }
    raw_preflight = launcher.materialize_bounded_semantic_p50_preflight.get_raw_f()
    receipt = raw_preflight(
        source_revision=source_revision,
        output_prefix_relative="editing_v2/semantic_p50",
        **addresses,
    )

    assert receipt["status"] == "PREFLIGHT_COMPLETE_EXACT_ONE_SHOT_PERMIT_NO_EXECUTION"
    assert receipt["training_authorized"] is False
    assert receipt["bounded_p50_authorized"] is False
    assert receipt["p500_authorized"] is False
    assert receipt["checkpoint_selection_authorized"] is False
    assert receipt["final_test_selection_authorized"] is False
    assert receipt["optimizer_constructed"] is False
    assert receipt["optimizer_steps_completed"] == 0
    assert receipt["model_device"] == "cpu"
    assert snapshot_calls == 2
    assert "run_p50" not in loaded
    expected_order = (
        "prerequisites.open",
        "environment.observe",
        "runtime.build",
        "runtime.publish",
        "runtime.open",
        "environment.build",
        "environment.publish",
        "environment.open",
        "launch.build",
        "launch.publish",
        "launch.open",
        "permit.publish",
        "permit.open",
    )
    positions = [events.index(item) for item in expected_order]
    assert positions == sorted(positions)
    assert events.index("inputs.snapshot") < events.index("prerequisites.open")
    assert len(events) - 1 - events[::-1].index("inputs.snapshot") > events.index(
        "permit.open"
    )
    for opened_event in (
        "runtime.open",
        "environment.open",
        "launch.open",
        "permit.open",
    ):
        index = events.index(opened_event)
        assert events[index - 2 : index] == ["volume.commit", "volume.reload"]


def test_preflight_stops_before_observation_when_physical_t1_reopen_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    monkeypatch.setattr(launcher, "ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(launcher, "REMOTE_ROOT", tmp_path / "remote")
    monkeypatch.setattr(
        launcher,
        "artifact_volume",
        SimpleNamespace(reload=lambda: None, commit=lambda: None),
    )
    monkeypatch.setattr(
        launcher,
        "_snapshot_prerequisite_files",
        lambda mounted: {"all_13_inputs": {"file_sha256": "1" * 64}},
    )
    source_revision = {
        "execution_source_revision": {"source_revision_sha256": "e" * 64},
        "source_revision_sha256": "s" * 64,
        "image_content_sha256": launcher._LOCAL_IMAGE_CONTENT_SHA256,
    }
    monkeypatch.setattr(
        launcher, "_validate_source_revision", lambda value: source_revision
    )
    scratch = SimpleNamespace(model=SimpleNamespace(parameters=lambda: ()))
    monkeypatch.setattr(
        launcher,
        "_scratch_runtime",
        lambda *args, **kwargs: (scratch, {"identity_sha256": "m" * 64}),
    )
    monkeypatch.setattr(launcher, "_source_binding", lambda *args, **kwargs: object())
    observed = False

    def observe():
        nonlocal observed
        observed = True
        raise AssertionError("mutated prerequisites must block before GPU observation")

    def reject(*args, **kwargs):
        raise RuntimeError("physical T1 input mutation")

    loaded = {
        "SemanticP50ExecutionPrerequisitePaths": lambda **kwargs: SimpleNamespace(
            **kwargs
        ),
        "load_prepared_recipe": lambda path: ({"prerequisites": {}}, "f" * 64),
        "open_execution_prerequisites": reject,
        "observe_physical_environment": observe,
    }
    monkeypatch.setattr(launcher, "_preflight_imports", lambda: loaded)
    raw_preflight = launcher.materialize_bounded_semantic_p50_preflight.get_raw_f()
    with pytest.raises(RuntimeError, match="physical T1 input mutation"):
        raw_preflight(
            source_revision=source_revision,
            source_inventory="/artifacts/source.json",
            migration_completion="/artifacts/migration.json",
            chunk_cache_plan="/artifacts/chunk-plan.json",
            chunk_cache_global_completion="/artifacts/chunk-complete.json",
            decision_plan="/artifacts/decision-plan.json",
            decision_completion="/artifacts/decision-complete.json",
            gate_zero_evidence="/artifacts/gate.json",
            t1_decision="/artifacts/t1.json",
            prepared_recipe="/artifacts/prepared.json",
            successor_cache_completion="/artifacts/cache.json",
            validation_inventory_completion="/artifacts/validation-inventory.json",
            validation_baseline_completion="/artifacts/baseline.json",
            validation_evaluation_environment_receipt="/artifacts/baseline-env.json",
            output_prefix_relative="editing_v2/semantic_p50",
        )
    assert observed is False


def test_gpu_worker_requires_existing_permit_reserves_then_strictly_reopens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "artifacts"
    remote_root = tmp_path / "remote"
    artifact_root.mkdir()
    remote_root.mkdir()
    monkeypatch.setattr(launcher, "ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(launcher, "REMOTE_ROOT", remote_root)

    events: list[str] = []

    class Volume:
        def reload(self) -> None:
            events.append("volume.reload")

        def commit(self) -> None:
            events.append("volume.commit")

    monkeypatch.setattr(launcher, "artifact_volume", Volume())
    execution_revision = {"source_revision_sha256": "e" * 64}
    source_revision = {
        "source_revision_sha256": "s" * 64,
        "execution_source_revision": execution_revision,
        "image_content_sha256": launcher._LOCAL_IMAGE_CONTENT_SHA256,
    }
    monkeypatch.setattr(
        launcher, "_validate_source_revision", lambda value: source_revision
    )
    monkeypatch.setattr(
        launcher,
        "_scratch_runtime",
        lambda *args, **kwargs: (scratch, {"identity_sha256": "m" * 64}),
    )
    monkeypatch.setattr(launcher, "_source_binding", lambda *args, **kwargs: object())
    monkeypatch.setattr(launcher, "_utc_now", lambda: "2026-08-01T12:00:00Z")
    monkeypatch.setattr(
        launcher, "_assert_result_environment", lambda *args, **kwargs: None
    )

    class Parameter:
        def __init__(self) -> None:
            self.device = SimpleNamespace(type="cpu")
            self.dtype = "float32"

    class Model:
        def __init__(self) -> None:
            self.parameter = Parameter()

        def parameters(self):
            return (self.parameter,)

        def to(self, *, device: str, dtype: str):
            assert device == "cuda"
            assert dtype == "float32"
            self.parameter.device = SimpleNamespace(type="cuda")
            self.parameter.dtype = dtype
            events.append("model.cuda_fp32")
            return self

    model = Model()
    scratch = SimpleNamespace(model=model)
    runtime_inputs = SimpleNamespace(scratch_runtime=scratch)
    permit_path = (
        artifact_root / "permits" / ("p" * 64) / "SEMANTIC_P50_EXECUTION_PERMIT.json"
    )
    output_relative = "editing_v2/semantic_p50/r" + "0" * 63
    permit = {
        "permit_sha256": "p" * 64,
        "source_revision_sha256": "e" * 64,
        "output_relative_path": output_relative,
        "optimizer_steps": 50,
        "batch_size": 64,
        "resume": False,
        "training_authorized": True,
        "bounded_p50_authorized": True,
        "p500_authorized": False,
        "checkpoint_selection_authorized": False,
        "final_test_selection_authorized": False,
    }
    verified = SimpleNamespace(
        path=permit_path.resolve(),
        permit=permit,
        prerequisites=SimpleNamespace(runtime_inputs=runtime_inputs),
        launch_projection={
            "image": {
                "reference": launcher.IMAGE_REFERENCE,
                "content_sha256": launcher._LOCAL_IMAGE_CONTENT_SHA256,
                "definition_relative_path": launcher.LAUNCHER_SOURCE,
            }
        },
    )
    destination = artifact_root / output_relative
    completion_path = destination / "result" / "SEMANTIC_P50_RUN_COMPLETE.json"
    reopened = SimpleNamespace(
        completion_path=completion_path,
        completion={"completion_sha256": "c" * 64},
        result_path=completion_path.parent / "semantic_p50_result.json",
        result={
            "result_sha256": "r" * 64,
            "completed_optimizer_steps": 50,
        },
        checkpoint_path=completion_path.parent / "semantic_p50_checkpoint.pt",
        checkpoint={"checkpoint_sha256": "k" * 64},
    )
    authorize_kwargs: dict[str, object] = {}
    reserve_kwargs: dict[str, object] = {}

    def authorize_recipe(**kwargs):
        events.append("permit.authorize_existing")
        authorize_kwargs.update(kwargs)
        return verified

    def open_permit(*args, **kwargs):
        events.append("permit.open")
        return verified

    def build_receipt(**kwargs):
        events.append("environment.observe")
        return {"receipt_sha256": "o" * 64}

    def publish_receipt(*args, **kwargs):
        events.append("environment.publish")

    def open_receipt(*args, **kwargs):
        events.append("environment.open")
        return {"receipt_sha256": "o" * 64}

    def reserve(*args, **kwargs):
        events.append("output.reserve")
        reserve_kwargs.update(kwargs)
        return destination

    artifacts = SimpleNamespace(result={})

    def run_p50(inputs):
        assert inputs is runtime_inputs
        events.append("p50.run")
        return artifacts

    def publish_run(*args, **kwargs):
        events.append("run.publish")
        return completion_path

    def open_run(*args, **kwargs):
        events.append("run.open")
        return reopened

    loaded = {
        "torch": SimpleNamespace(
            float32="float32",
            cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
        ),
        "SemanticP50ExecutionPrerequisitePaths": lambda **kwargs: SimpleNamespace(
            **kwargs
        ),
        "ENVIRONMENT_RECEIPT_FILENAME": "SEMANTIC_P50_OBSERVED_ENVIRONMENT.json",
        "load_prepared_recipe": lambda path: ({"prerequisites": {}}, "f" * 64),
        "authorize_recipe": authorize_recipe,
        "open_execution_permit": open_permit,
        "build_observed_environment_receipt": build_receipt,
        "publish_observed_environment_receipt": publish_receipt,
        "open_observed_environment_receipt": open_receipt,
        "require_output_available": reserve,
        "run_p50": run_p50,
        "publish_run_artifacts": publish_run,
        "open_run_artifacts": open_run,
    }
    monkeypatch.setattr(launcher, "_imports", lambda: loaded)
    addresses = {
        "source_inventory": "/artifacts/source.json",
        "migration_completion": "/artifacts/migration.json",
        "chunk_cache_plan": "/artifacts/chunk-plan.json",
        "chunk_cache_global_completion": "/artifacts/chunk-complete.json",
        "decision_plan": "/artifacts/decision-plan.json",
        "decision_completion": "/artifacts/decision-complete.json",
        "gate_zero_evidence": "/artifacts/gate.json",
        "t1_decision": "/artifacts/t1.json",
        "prepared_recipe": "/artifacts/prepared.json",
        "successor_cache_completion": "/artifacts/cache.json",
        "validation_inventory_completion": "/artifacts/validation-inventory.json",
        "validation_baseline_completion": "/artifacts/baseline.json",
        "validation_evaluation_environment_receipt": "/artifacts/baseline-env.json",
        "runtime_contract": "/artifacts/runtime.json",
        "environment_contract": "/artifacts/environment.json",
        "launch_projection": "/artifacts/launch.json",
        "execution_permit": launcher._artifact_address(permit_path),
    }
    raw_worker = launcher.run_bounded_semantic_p50.get_raw_f()
    result = raw_worker(source_revision=source_revision, **addresses)
    assert authorize_kwargs["permit_path"] == permit_path.resolve()
    assert "output_root" not in authorize_kwargs
    assert reserve_kwargs["execution_permit_path"] == permit_path.resolve()
    assert result["completed_optimizer_steps"] == 50
    assert result["p500_authorized"] is False
    assert result["checkpoint_selection_authorized"] is False
    assert result["final_test_selection_authorized"] is False
    assert events.index("permit.authorize_existing") < events.index("permit.open")
    assert events.index("permit.open") < events.index("environment.observe")
    assert events.index("output.reserve") < events.index("model.cuda_fp32")
    assert events.index("model.cuda_fp32") < events.index("p50.run")
    assert events.count("run.open") == 2
    assert events[-1] == "run.open"


def test_surface_is_one_gpu_no_amp_and_mounts_only_hashed_sources() -> None:
    source = (Path(__file__).parents[1] / launcher.LAUNCHER_SOURCE).read_text(
        encoding="utf-8"
    )
    preflight_surface = source[
        source.index("def materialize_bounded_semantic_p50_preflight")
        - 180 : source.index("def run_bounded_semantic_p50")
    ]
    run_surface = source[
        source.index("def run_bounded_semantic_p50")
        - 180 : source.index("@app.local_entrypoint()")
    ]
    assert 'MODAL_GPU_TYPE = "A10G"' in source
    for function_surface in (preflight_surface, run_surface):
        assert "image=image" in function_surface
        assert "gpu=MODAL_GPU_TYPE" in function_surface
        assert "cpu=8.0" in function_surface
        assert "memory=65536" in function_surface
        assert "timeout=8 * 3600" in function_surface
        assert "max_containers=1" in function_surface
        assert "volumes={str(ARTIFACT_ROOT): artifact_volume}" in function_surface
    assert "authorize_recipe" in source
    assert "open_execution_permit" in source
    assert "observe_physical_environment" in preflight_surface
    assert "build_runtime_contract" in preflight_surface
    assert "build_environment_contract" in preflight_surface
    assert "build_launch_projection" in preflight_surface
    assert "materialize_execution_permit" in preflight_surface
    assert "run_p50" not in preflight_surface
    assert "execution_permit_path=permit_path" in source
    assert "publish_run_artifacts" in source
    assert "open_run_artifacts" in source
    assert 'mixed_precision": False' in source
    assert "autocast" not in source
    assert "GradScaler" not in source
    assert ".git" not in source
    assert source.count("run_bounded_semantic_p50.remote(") == 1
    assert source.count("materialize_bounded_semantic_p50_preflight.remote(") == 1
