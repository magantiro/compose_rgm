from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("modal", reason="Modal runtime tests require the cloud extra")

modal_entrypoint = importlib.import_module("modal_apps.train_tracelet_gm")
_early_rollout_decision = modal_entrypoint._early_rollout_decision
_final_rollout_spec = modal_entrypoint._final_rollout_spec
_rollout_manifest_signature = modal_entrypoint._rollout_manifest_signature
_rollout_evaluation_profile = modal_entrypoint._rollout_evaluation_profile
_rollout_remote_summary = modal_entrypoint._rollout_remote_summary
_run_pipeline = modal_entrypoint._run_pipeline
_stable_json_sha256 = modal_entrypoint._stable_json_sha256
_remote_stage_kind = modal_entrypoint._remote_stage_kind
_source_fingerprint = modal_entrypoint._source_fingerprint
_validate_run_label = modal_entrypoint._validate_run_label


def test_early_rollout_waits_for_warmup_and_material_improvement() -> None:
    payload = {
        "initial_validation": {"factorized_gm_loss": 10.0},
        "selected_validation": {
            "factorized_gm_loss": 3.0,
            "family_accuracy": 0.8,
            "selected_step": 1000.0,
        },
    }

    assert (
        _early_rollout_decision(
            payload,
            completed_steps=999,
            warmup_steps=500,
        )
        is None
    )
    decision = _early_rollout_decision(
        payload,
        completed_steps=1000,
        warmup_steps=500,
    )

    assert decision is not None
    assert decision["selected_step"] == 1000.0
    assert decision["relative_improvement"] == pytest.approx(0.7)
    assert decision["selected_family_accuracy"] == pytest.approx(0.8)
    assert decision["minimum_relative_improvement"] == pytest.approx(0.65)
    assert decision["minimum_family_accuracy"] == pytest.approx(0.75)
    assert decision["minimum_selected_step"] == 1000.0


def test_early_rollout_rejects_marginal_or_malformed_checkpoints() -> None:
    marginal = {
        "initial_validation": {"factorized_gm_loss": 10.0},
        "selected_validation": {
            "factorized_gm_loss": 3.6,
            "family_accuracy": 0.9,
            "selected_step": 1000.0,
        },
    }
    assert (
        _early_rollout_decision(
            marginal,
            completed_steps=1000,
            warmup_steps=500,
        )
        is None
    )
    assert (
        _early_rollout_decision(
            {},
            completed_steps=1000,
            warmup_steps=500,
        )
        is None
    )


def test_early_rollout_launch_is_deduplicated_by_persistent_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "source-run"
    run_dir.mkdir()
    torch.save(
        {
            "initial_validation": {"factorized_gm_loss": 10.0},
            "selected_validation": {
                "factorized_gm_loss": 3.0,
                "family_accuracy": 0.8,
                "selected_step": 1000.0,
            },
        },
        run_dir / "checkpoint.best_so_far.pt",
    )

    class FakeVolume:
        def __init__(self) -> None:
            self.commits = 0

        def commit(self) -> None:
            self.commits += 1

    class FakeCall:
        object_id = "fc-test"

    class FakeStage:
        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []

        def spawn(self, *args: object) -> FakeCall:
            self.calls.append(args)
            return FakeCall()

    volume = FakeVolume()
    stage = FakeStage()
    monkeypatch.setattr(modal_entrypoint, "artifact_volume", volume)
    monkeypatch.setattr(modal_entrypoint, "rollout_evaluate_stage", stage)
    recipe: dict[str, object] = {
        "early_preview": {
            "minimum_relative_improvement": 0.65,
            "minimum_family_accuracy": 0.75,
            "minimum_selected_step": 1000,
        },
        "arguments": {"warmup_steps": 500},
    }

    for _ in range(2):
        modal_entrypoint._maybe_launch_early_rollout(
            run_label="source-run",
            run_dir=run_dir,
            recipe=recipe,
            completed_steps=1000,
        )

    assert len(stage.calls) == 1
    assert stage.calls[0][-1] == 100
    assert stage.calls[0][2] == "checkpoint.early_step1000.pt"
    marker = json.loads((run_dir / "early_eval_launch.json").read_text())
    assert marker["status"] == "spawned"
    assert marker["function_call_id"] == "fc-test"
    assert len(marker["source_checkpoint_sha256"]) == 64
    assert (run_dir / marker["source_checkpoint"]).is_file()
    assert volume.commits == 2


def test_early_rollout_marker_retries_failed_and_stale_reservations(
    tmp_path: Path,
) -> None:
    marker_path = tmp_path / "early_eval_launch.json"
    marker_path.write_text(
        json.dumps(
            {
                "status": "reserved",
                "updated_at_unix": modal_entrypoint.time.time(),
            }
        )
    )
    assert modal_entrypoint._early_launch_marker_blocks_retry(marker_path)

    marker_path.write_text(
        json.dumps(
            {
                "status": "reserved",
                "updated_at_unix": modal_entrypoint.time.time() - 901.0,
            }
        )
    )
    assert not modal_entrypoint._early_launch_marker_blocks_retry(marker_path)

    marker_path.write_text(json.dumps({"status": "failed"}))
    assert not modal_entrypoint._early_launch_marker_blocks_retry(marker_path)

    marker_path.write_text(
        json.dumps(
            {
                "status": "running",
                "updated_at_unix": modal_entrypoint.time.time() - 901.0,
            }
        )
    )
    assert not modal_entrypoint._early_launch_marker_blocks_retry(marker_path)


def test_final_rollout_spec_uses_selected_checkpoint_and_step() -> None:
    assert _final_rollout_spec(
        "production-v1",
        {"selected_validation": {"selected_step": 2750.0}},
    ) == (
        "production-v1-final-step2750-eval2000",
        "checkpoint.pt",
        2000,
    )

    assert _final_rollout_spec(
        "smoke-v1",
        {"selected_validation": {"selected_step": 2.0}},
        rollout_samples=16,
    ) == (
        "smoke-v1-final-step2-eval16",
        "checkpoint.pt",
        16,
    )


@pytest.mark.parametrize(
    "train_result",
    (
        {},
        {"selected_validation": {}},
        {"selected_validation": {"selected_step": float("nan")}},
        {"selected_validation": {"selected_step": 1.5}},
    ),
)
def test_final_rollout_spec_rejects_missing_or_ambiguous_step(
    train_result: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _final_rollout_spec("production-v1", train_result)


def test_rollout_manifest_signature_is_sensitive_to_checkpoint_and_rules() -> None:
    common = {
        "source_run_label": "source",
        "checkpoint_name": "checkpoint.pt",
        "rollout_samples": 2000,
        "train_sha256": "train",
        "reference_sha256": "reference",
        "source_sha256": "source-code",
    }
    first = _rollout_manifest_signature(
        **common,
        checkpoint_sha256="checkpoint-a",
        disabled_rule_names=(),
    )
    second = _rollout_manifest_signature(
        **common,
        checkpoint_sha256="checkpoint-b",
        disabled_rule_names=("ring_ear_insert",),
    )

    assert first != second
    assert first["rollout_samples"] == 2000
    assert first["sampling_seed"] == 20260721


def test_integration_profile_is_small_and_part_of_rollout_identity() -> None:
    production = _rollout_evaluation_profile("production_v1")
    smoke = _rollout_evaluation_profile("integration_smoke_v1")
    common = {
        "source_run_label": "source",
        "checkpoint_name": "checkpoint.pt",
        "checkpoint_sha256": "checkpoint",
        "rollout_samples": 16,
        "disabled_rule_names": (),
        "train_sha256": "train",
        "reference_sha256": "reference",
        "source_sha256": "source-code",
    }

    assert smoke["fast_split"] is True
    assert smoke["train_size"] == 32
    assert smoke["quality_reference_limit"] == 128
    assert _rollout_manifest_signature(
        **common,
        evaluation_profile=production,
    ) != _rollout_manifest_signature(
        **common,
        evaluation_profile=smoke,
    )


def test_rollout_remote_summary_exposes_trajectory_diagnostic_availability(
    tmp_path: Path,
) -> None:
    summary = _rollout_remote_summary(
        run_label="evaluation",
        run_dir=tmp_path,
        report={
            "generated_nonnull_smiles": 16,
            "rollout": {"valid_fraction": 0.75},
            "trajectory_diagnostics_available": True,
            "molecular_quality": {"frechet_chemnet_distance": 42.0},
        },
    )

    assert summary["valid_fraction"] == 0.75
    assert summary["trajectory_diagnostics_available"] is True


def test_pipeline_smoke_uses_real_stage_order_and_separate_cpu_eval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[object, ...]]] = []

    class FakeStage:
        def __init__(self, name: str, result: dict[str, object]) -> None:
            self.name = name
            self.result = result

        def remote(self, *args: object) -> dict[str, object]:
            calls.append((self.name, args))
            return self.result

    monkeypatch.setattr(
        modal_entrypoint,
        "compile_stage",
        FakeStage("compile", {"path_cache": "compiled"}),
    )
    monkeypatch.setattr(
        modal_entrypoint,
        "audit_teacher_stage",
        FakeStage("audit", {"family_failures": {}}),
    )
    monkeypatch.setattr(
        modal_entrypoint,
        "train_stage",
        FakeStage(
            "train",
            {
                "rollouts_skipped": True,
                "selected_validation": {"selected_step": 2.0},
            },
        ),
    )
    monkeypatch.setattr(
        modal_entrypoint,
        "rollout_evaluate_stage",
        FakeStage(
            "rollout",
            {
                "generated_nonnull_smiles": 16,
                "valid_fraction": 1.0,
                "trajectory_diagnostics_available": True,
                "fcd": 42.0,
            },
        ),
    )

    result = _run_pipeline(
        "integration-v1",
        "tree_fcd_transfer_stage3_integration_smoke.json",
        final_rollout_samples=16,
        audit_batch_size=16,
        audit_workers=1,
        evaluation_profile_name="integration_smoke_v1",
        require_fcd=True,
    )

    assert [name for name, _ in calls] == ["compile", "audit", "train", "rollout"]
    assert calls[-1][1] == (
        "integration-v1-final-step2-eval16",
        "integration-v1",
        "checkpoint.pt",
        16,
        False,
        "integration_smoke_v1",
    )
    assert result["final_rollout"]["fcd"] == 42.0


def test_pipeline_refuses_gpu_training_that_did_not_skip_rollouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStage:
        def __init__(self, result: dict[str, object]) -> None:
            self.result = result

        def remote(self, *args: object) -> dict[str, object]:
            return self.result

    monkeypatch.setattr(modal_entrypoint, "compile_stage", FakeStage({}))
    monkeypatch.setattr(modal_entrypoint, "audit_teacher_stage", FakeStage({}))
    monkeypatch.setattr(
        modal_entrypoint,
        "train_stage",
        FakeStage({"rollouts_skipped": False}),
    )

    with pytest.raises(RuntimeError, match="in-container rollouts"):
        _run_pipeline(
            "integration-v1",
            "tree_fcd_transfer_stage3_integration_smoke.json",
            final_rollout_samples=16,
            audit_batch_size=16,
            audit_workers=1,
            evaluation_profile_name="integration_smoke_v1",
        )


@pytest.mark.parametrize(
    ("rollout_result", "message"),
    (
        (
            {
                "generated_nonnull_smiles": 16,
                "trajectory_diagnostics_available": True,
                "fcd": 42.0,
            },
            "numeric valid fraction",
        ),
        (
            {
                "generated_nonnull_smiles": 16,
                "valid_fraction": 1.0,
                "trajectory_diagnostics_available": False,
                "fcd": 42.0,
            },
            "trajectory diagnostics",
        ),
    ),
)
def test_pipeline_smoke_requires_complete_cpu_evaluation_evidence(
    monkeypatch: pytest.MonkeyPatch,
    rollout_result: dict[str, object],
    message: str,
) -> None:
    class FakeStage:
        def __init__(self, result: dict[str, object]) -> None:
            self.result = result

        def remote(self, *args: object) -> dict[str, object]:
            return self.result

    monkeypatch.setattr(modal_entrypoint, "compile_stage", FakeStage({}))
    monkeypatch.setattr(modal_entrypoint, "audit_teacher_stage", FakeStage({}))
    monkeypatch.setattr(
        modal_entrypoint,
        "train_stage",
        FakeStage(
            {
                "rollouts_skipped": True,
                "selected_validation": {"selected_step": 2.0},
            }
        ),
    )
    monkeypatch.setattr(
        modal_entrypoint,
        "rollout_evaluate_stage",
        FakeStage(rollout_result),
    )

    with pytest.raises(RuntimeError, match=message):
        _run_pipeline(
            "integration-v1",
            "tree_fcd_transfer_stage3_integration_smoke.json",
            final_rollout_samples=16,
            audit_batch_size=16,
            audit_workers=1,
            evaluation_profile_name="integration_smoke_v1",
            require_fcd=True,
        )


def test_stage_identity_hash_is_order_independent_and_stage_specific() -> None:
    assert _stable_json_sha256({"b": 2, "a": 1}) == _stable_json_sha256(
        {"a": 1, "b": 2}
    )
    assert _remote_stage_kind(
        smoke=False,
        preflight=False,
        compile_paths_only=True,
        evaluation_source_run=None,
    ) == "compile"
    assert _remote_stage_kind(
        smoke=False,
        preflight=False,
        compile_paths_only=False,
        evaluation_source_run=None,
    ) == "training"


@pytest.mark.parametrize(
    "label",
    (
        "compose-v4-stage3-production-736dae4-v1",
        "A100_preflight.01",
        "run-1",
    ),
)
def test_run_label_validation_accepts_safe_ascii_basenames(label: str) -> None:
    _validate_run_label(label)


@pytest.mark.parametrize(
    "label",
    (
        "",
        ".",
        "..",
        "../escape",
        "nested/run",
        "/tmp/run",
        "bad label",
        "-leading-dash",
        "nonascii-μ",
        "a" * 129,
    ),
)
def test_run_label_validation_rejects_unsafe_or_ambiguous_labels(label: str) -> None:
    with pytest.raises(ValueError, match="ASCII basename"):
        _validate_run_label(label)


def test_source_fingerprint_ignores_bytecode_and_tracks_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("src", "scripts", "recipes"):
        (tmp_path / name).mkdir()
    source = tmp_path / "src" / "model.py"
    source.write_text("VALUE = 1\n")
    (tmp_path / "recipes" / "run.json").write_text('{"name": "run"}\n')
    monkeypatch.setattr(modal_entrypoint, "REMOTE_ROOT", tmp_path)

    original = _source_fingerprint()
    cache = tmp_path / "src" / "__pycache__"
    cache.mkdir()
    (cache / "model.cpython-311.pyc").write_bytes(b"untracked bytecode")
    (tmp_path / "scripts" / "worker.pyc").write_bytes(b"more bytecode")

    assert _source_fingerprint() == original
    source.write_text("VALUE = 2\n")
    assert _source_fingerprint() != original
