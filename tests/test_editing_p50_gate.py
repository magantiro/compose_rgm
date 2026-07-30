"""P50 evidence is strict, self-hashed, and bound to current step-50 state."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import compose_v4.experiments.editing_p50_gate as p50_gate
from compose_v4.experiments.editing_p50_gate import (
    P50_BALANCED_METRIC,
    P50_SELECTION_METRIC,
    P50GateError,
    P50RetentionEvidence,
    P50RunIdentity,
    p50_evidence_self_hash,
    p50_pass_authorizes_p500,
    resolved_p50_thresholds_sha256,
    seal_p50_evidence,
    state_dict_semantic_sha256,
    validate_p50_evidence,
)
from compose_v4.experiments.editing_training_gate import (
    REQUIRED_P50_FAMILIES,
    ResolvedP50Thresholds,
    resolve_p50_thresholds,
)


def _sha(character: str) -> str:
    return character * 64


def _thresholds(
    *,
    warm: bool = False,
    minimum_updates: int = 1,
) -> ResolvedP50Thresholds:
    contract_path = Path(__file__).parents[1] / "configs" / "editing_training_v2_gate.json"
    contract = json.loads(contract_path.read_text())
    p50 = next(
        gate for gate in contract["gates"] if gate["id"] == "P50_gradient_and_collapse_sentinel"
    )
    families = tuple(REQUIRED_P50_FAMILIES)
    p50["numeric_thresholds"] = {
        "minimum_gradient_updates_per_required_slice": {
            family: minimum_updates for family in families
        },
        "maximum_required_slice_successor_nll_regression": {family: 0.1 for family in families},
        "maximum_inherited_probe_nll_regression": {
            "scratch": "NOT_APPLICABLE",
            "compatible_warm_start": 0.2,
            "compatible_warm_start_with_retention": 0.2,
        },
    }
    return resolve_p50_thresholds(
        contract,
        initialization_regime="compatible_warm_start" if warm else "scratch",
    )


def _identity(
    thresholds: ResolvedP50Thresholds,
    *,
    warm: bool = False,
    planned: str | None = None,
    resume: bool = False,
) -> P50RunIdentity:
    initial_state_sha256 = state_dict_semantic_sha256({"body.weight": torch.tensor([[0.25, -0.5]])})
    return P50RunIdentity(
        training_gate_contract_sha256=_sha("1"),
        resolved_thresholds_sha256=resolved_p50_thresholds_sha256(thresholds),
        s0_evidence_sha256=_sha("2"),
        t1_evidence_sha256=_sha("3"),
        training_recipe_sha256=_sha("4"),
        initial_state_semantic_sha256=initial_state_sha256,
        objective_name="canonical_successor_productive_identity_v1",
        selection_metric=P50_SELECTION_METRIC,
        optimizer_kind="adamw_decoupled_v1",
        schedule_sha256=_sha("5"),
        seed=31,
        initialization_regime=("compatible_warm_start" if warm else "scratch"),
        corpus_manifest_sha256=_sha("6"),
        representability_overlay_sha256=_sha("7"),
        semantic_cell_sidecar_sha256=_sha("8"),
        validation_panel_sha256=_sha("9"),
        successor_cache_inventory_sha256=_sha("a"),
        planned_ordered_training_stream_sha256=planned or _sha("b"),
        resume=resume,
    )


def _metrics(*, regression: float = 0.0) -> tuple[dict, dict]:
    families = tuple(REQUIRED_P50_FAMILIES)
    baseline = {
        P50_SELECTION_METRIC: 2.5,
        P50_BALANCED_METRIC: 2.75,
    }
    final = {
        P50_SELECTION_METRIC: 2.4,
        P50_BALANCED_METRIC: 2.65,
    }
    for family in families:
        baseline[f"teacher_examples_{family}"] = 4.0
        baseline[f"canonical_successor_nll_{family}"] = 2.0
        final[f"teacher_examples_{family}"] = 4.0
        final[f"canonical_successor_nll_{family}"] = 2.0 + regression
    return baseline, final


def _checkpoint(
    path: Path,
    *,
    completed_steps: int = 50,
    current_value: float = 1.0,
    best_value: float = 9.0,
) -> Path:
    torch.save(
        {
            "optimizer_kind": "adamw_decoupled_v1",
            "training_objective_name": "canonical_successor_productive_identity_v1",
            "training_selection_metric": P50_SELECTION_METRIC,
            "completed_steps": completed_steps,
            "current_state_dict": {
                "body.weight": torch.tensor([[current_value, 2.0]]),
                "counter": torch.tensor(3, dtype=torch.int64),
            },
            "optimizer_state_dict": {
                "state": {
                    0: {
                        "step": torch.tensor(50),
                        "exp_avg": torch.tensor([0.0]),
                    }
                },
                "param_groups": [{"lr": 1e-3, "params": [0]}],
            },
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": None,
            "best_state_dict": {
                "body.weight": torch.tensor([[best_value, 2.0]]),
                "counter": torch.tensor(3, dtype=torch.int64),
            },
            "best_metrics": {P50_SELECTION_METRIC: 2.4},
            "history": [{"step": float(completed_steps), "train_loss": 2.3}],
            "evaluations_without_improvement": 0,
        },
        path,
    )
    return path


def _updates(value: int = 2) -> dict[str, int]:
    return {family: value for family in REQUIRED_P50_FAMILIES}


def _pass(
    tmp_path: Path,
    *,
    warm: bool = False,
    regression: float = 0.0,
) -> tuple[dict, P50RunIdentity, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    thresholds = _thresholds(warm=warm)
    identity = _identity(thresholds, warm=warm)
    baseline, final = _metrics(regression=regression)
    checkpoint = _checkpoint(tmp_path / "recovery.pt")
    retention = (
        P50RetentionEvidence(
            applicability="REQUIRED_WARM_START",
            probe_panel_sha256=_sha("c"),
            baseline_checkpoint_sha256=_sha("d"),
            baseline_nll=1.0,
            final_nll=1.05,
            observed_regression=0.05,
        )
        if warm
        else P50RetentionEvidence.scratch()
    )
    evidence = seal_p50_evidence(
        status="PASS",
        run_identity=identity,
        resolved_thresholds=thresholds,
        completed_optimizer_steps=50,
        checkpoint_path=checkpoint,
        observed_ordered_training_stream_sha256=(identity.planned_ordered_training_stream_sha256),
        family_gate_updates=_updates(),
        action_route_updates=_updates(),
        baseline_validation=baseline,
        final_validation=final,
        retention=retention,
    )
    return evidence, identity, checkpoint


def test_state_dict_semantic_hash_is_order_independent_and_sensitive() -> None:
    left = {
        "z": torch.tensor([1.0, 2.0]),
        "a": torch.tensor([[3]], dtype=torch.int64),
    }
    reordered = {
        "a": left["a"].clone(),
        "z": left["z"].clone(),
    }
    changed = {
        **reordered,
        "z": torch.tensor([1.0, 2.5]),
    }
    assert state_dict_semantic_sha256(left) == state_dict_semantic_sha256(reordered)
    assert state_dict_semantic_sha256(left) != state_dict_semantic_sha256(changed)
    with pytest.raises(P50GateError, match="nonfinite"):
        state_dict_semantic_sha256({"bad": torch.tensor([float("nan")])})


def test_gate_resolved_threshold_adapter_binds_regime_and_retention_status(
    tmp_path: Path,
) -> None:
    scratch = _thresholds()
    warm = _thresholds(warm=True)
    assert resolved_p50_thresholds_sha256(scratch) != resolved_p50_thresholds_sha256(warm)

    forged = replace(scratch, inherited_retention_status="REQUIRED")
    with pytest.raises(P50GateError, match="retention status"):
        resolved_p50_thresholds_sha256(forged)

    evidence, _, _ = _pass(tmp_path)
    tampered = copy.deepcopy(evidence)
    tampered["resolved_thresholds"]["inherited_retention_status"] = "REQUIRED"
    tampered["artifact_sha256"] = p50_evidence_self_hash(tampered)
    with pytest.raises(P50GateError, match="retention status"):
        validate_p50_evidence(tampered)


def test_initial_state_identity_is_required_and_checked_for_authorization(
    tmp_path: Path,
) -> None:
    evidence, identity, checkpoint = _pass(tmp_path)
    assert (
        evidence["run_identity"]["initial_state_semantic_sha256"]
        == identity.initial_state_semantic_sha256
    )

    wrong_initial_state = replace(identity, initial_state_semantic_sha256=_sha("e"))
    with pytest.raises(P50GateError, match="run identity mismatch"):
        p50_pass_authorizes_p500(
            evidence,
            expected_run_identity=wrong_initial_state,
            checkpoint_path=checkpoint,
        )

    missing = copy.deepcopy(evidence)
    del missing["run_identity"]["initial_state_semantic_sha256"]
    missing["artifact_sha256"] = p50_evidence_self_hash(missing)
    with pytest.raises(P50GateError, match="missing"):
        validate_p50_evidence(missing)


def test_checkpoint_inspection_uses_safe_weights_only_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_weights_only: list[object] = []
    original_load = p50_gate.torch.load

    def recording_load(*args, **kwargs):
        observed_weights_only.append(kwargs.get("weights_only"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(p50_gate.torch, "load", recording_load)
    _pass(tmp_path)
    assert observed_weights_only
    assert set(observed_weights_only) == {True}


def test_checkpoint_identity_change_during_inspection_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_identity = p50_gate._file_identity
    calls = 0

    def changing_identity(metadata):
        nonlocal calls
        calls += 1
        identity = original_identity(metadata)
        if calls == 3:
            return replace(identity, ctime_ns=identity.ctime_ns + 1)
        return identity

    monkeypatch.setattr(p50_gate, "_file_identity", changing_identity)
    with pytest.raises(P50GateError, match="changed while loading"):
        _pass(tmp_path)


def test_pass_seals_validates_and_authorizes_exact_current_state(
    tmp_path: Path,
) -> None:
    evidence, identity, checkpoint = _pass(tmp_path)
    validated = validate_p50_evidence(
        evidence,
        expected_run_identity=identity,
        checkpoint_path=checkpoint,
    )
    assert validated.status == "PASS"
    assert validated.checkpoint["state_source"] == "current_state_dict"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert validated.checkpoint["current_state_semantic_sha256"] == state_dict_semantic_sha256(
        payload["current_state_dict"]
    )
    assert validated.checkpoint["current_state_semantic_sha256"] != state_dict_semantic_sha256(
        payload["best_state_dict"]
    )
    assert p50_pass_authorizes_p500(
        evidence,
        expected_run_identity=identity,
        checkpoint_path=checkpoint,
    )


def test_unknown_missing_and_tampered_fields_are_rejected(
    tmp_path: Path,
) -> None:
    evidence, _identity_unused, checkpoint = _pass(tmp_path)

    unknown = copy.deepcopy(evidence)
    unknown["invented"] = True
    unknown["artifact_sha256"] = p50_evidence_self_hash(unknown)
    with pytest.raises(P50GateError, match="unknown"):
        validate_p50_evidence(unknown)

    missing = copy.deepcopy(evidence)
    del missing["retention"]
    missing["artifact_sha256"] = p50_evidence_self_hash(missing)
    with pytest.raises(P50GateError, match="missing"):
        validate_p50_evidence(missing)

    tampered = copy.deepcopy(evidence)
    tampered["completed_optimizer_steps"] = 49
    with pytest.raises(P50GateError, match="self-hash"):
        validate_p50_evidence(tampered)

    nested_unknown = copy.deepcopy(evidence)
    nested_unknown["checkpoint"]["best_state_semantic_sha256"] = _sha("f")
    nested_unknown["artifact_sha256"] = p50_evidence_self_hash(nested_unknown)
    with pytest.raises(P50GateError, match="unknown"):
        validate_p50_evidence(nested_unknown)

    wrong_identity = _identity(
        _thresholds(),
        planned=_sha("e"),
    )
    with pytest.raises(P50GateError, match="run identity mismatch"):
        validate_p50_evidence(
            evidence,
            expected_run_identity=wrong_identity,
            checkpoint_path=checkpoint,
        )


def test_checkpoint_bytes_are_reverified_before_authorization(
    tmp_path: Path,
) -> None:
    evidence, identity, checkpoint = _pass(tmp_path)
    _checkpoint(checkpoint, current_value=1.5)
    with pytest.raises(P50GateError, match="no longer matches"):
        p50_pass_authorizes_p500(
            evidence,
            expected_run_identity=identity,
            checkpoint_path=checkpoint,
        )


def test_pass_rejects_threshold_or_ordered_training_stream_mismatch(
    tmp_path: Path,
) -> None:
    thresholds = _thresholds()
    identity = _identity(thresholds)
    baseline, final = _metrics(regression=0.0)
    checkpoint = _checkpoint(tmp_path / "recovery.pt")

    with pytest.raises(P50GateError, match="ordered training stream"):
        seal_p50_evidence(
            status="PASS",
            run_identity=identity,
            resolved_thresholds=thresholds,
            completed_optimizer_steps=50,
            checkpoint_path=checkpoint,
            observed_ordered_training_stream_sha256=_sha("f"),
            family_gate_updates=_updates(),
            action_route_updates=_updates(),
            baseline_validation=baseline,
            final_validation=final,
            retention=P50RetentionEvidence.scratch(),
        )

    stricter = _thresholds(minimum_updates=3)
    with pytest.raises(P50GateError, match="different hashes"):
        seal_p50_evidence(
            status="PASS",
            run_identity=identity,
            resolved_thresholds=stricter,
            completed_optimizer_steps=50,
            checkpoint_path=checkpoint,
            observed_ordered_training_stream_sha256=(
                identity.planned_ordered_training_stream_sha256
            ),
            family_gate_updates=_updates(),
            action_route_updates=_updates(),
            baseline_validation=baseline,
            final_validation=final,
            retention=P50RetentionEvidence.scratch(),
        )


def test_family_gate_and_action_route_thresholds_are_independent(
    tmp_path: Path,
) -> None:
    thresholds = _thresholds()
    identity = _identity(thresholds)
    baseline, final = _metrics()
    checkpoint = _checkpoint(tmp_path / "recovery.pt")
    dead_route = _updates()
    dead_route["cycle_attach"] = 0
    with pytest.raises(P50GateError, match="action-route"):
        seal_p50_evidence(
            status="PASS",
            run_identity=identity,
            resolved_thresholds=thresholds,
            completed_optimizer_steps=50,
            checkpoint_path=checkpoint,
            observed_ordered_training_stream_sha256=(
                identity.planned_ordered_training_stream_sha256
            ),
            family_gate_updates=_updates(),
            action_route_updates=dead_route,
            baseline_validation=baseline,
            final_validation=final,
            retention=P50RetentionEvidence.scratch(),
        )

    dead_gate = _updates()
    dead_gate["bond_reroute"] = 0
    with pytest.raises(P50GateError, match="family-gate"):
        seal_p50_evidence(
            status="PASS",
            run_identity=identity,
            resolved_thresholds=thresholds,
            completed_optimizer_steps=50,
            checkpoint_path=checkpoint,
            observed_ordered_training_stream_sha256=(
                identity.planned_ordered_training_stream_sha256
            ),
            family_gate_updates=dead_gate,
            action_route_updates=_updates(),
            baseline_validation=baseline,
            final_validation=final,
            retention=P50RetentionEvidence.scratch(),
        )


def test_scratch_and_warm_start_retention_are_not_interchangeable(
    tmp_path: Path,
) -> None:
    scratch_thresholds = _thresholds()
    scratch_identity = _identity(scratch_thresholds)
    baseline, final = _metrics()
    checkpoint = _checkpoint(tmp_path / "scratch.pt")
    fabricated = P50RetentionEvidence(
        applicability="REQUIRED_WARM_START",
        probe_panel_sha256=_sha("c"),
        baseline_checkpoint_sha256=_sha("d"),
        baseline_nll=1.0,
        final_nll=1.0,
        observed_regression=0.0,
    )
    with pytest.raises(P50GateError, match="scratch"):
        seal_p50_evidence(
            status="PASS",
            run_identity=scratch_identity,
            resolved_thresholds=scratch_thresholds,
            completed_optimizer_steps=50,
            checkpoint_path=checkpoint,
            observed_ordered_training_stream_sha256=(
                scratch_identity.planned_ordered_training_stream_sha256
            ),
            family_gate_updates=_updates(),
            action_route_updates=_updates(),
            baseline_validation=baseline,
            final_validation=final,
            retention=fabricated,
        )

    warm_thresholds = _thresholds(warm=True)
    warm_identity = _identity(warm_thresholds, warm=True)
    with pytest.raises(P50GateError, match="warm-start"):
        seal_p50_evidence(
            status="PASS",
            run_identity=warm_identity,
            resolved_thresholds=warm_thresholds,
            completed_optimizer_steps=50,
            checkpoint_path=checkpoint,
            observed_ordered_training_stream_sha256=(
                warm_identity.planned_ordered_training_stream_sha256
            ),
            family_gate_updates=_updates(),
            action_route_updates=_updates(),
            baseline_validation=baseline,
            final_validation=final,
            retention=P50RetentionEvidence.scratch(),
        )

    missing_warm_baseline = P50RetentionEvidence(
        applicability="REQUIRED_WARM_START",
        probe_panel_sha256=_sha("c"),
        baseline_checkpoint_sha256=_sha("d"),
    )
    with pytest.raises(P50GateError, match="baseline_nll"):
        seal_p50_evidence(
            status="PASS",
            run_identity=warm_identity,
            resolved_thresholds=warm_thresholds,
            completed_optimizer_steps=50,
            checkpoint_path=checkpoint,
            observed_ordered_training_stream_sha256=(
                warm_identity.planned_ordered_training_stream_sha256
            ),
            family_gate_updates=_updates(),
            action_route_updates=_updates(),
            baseline_validation=baseline,
            final_validation=final,
            retention=missing_warm_baseline,
        )

    evidence, identity, warm_checkpoint = _pass(
        tmp_path / "warm",
        warm=True,
    )
    assert p50_pass_authorizes_p500(
        evidence,
        expected_run_identity=identity,
        checkpoint_path=warm_checkpoint,
    )


def test_fail_is_sealed_but_never_authorizes_p500(tmp_path: Path) -> None:
    thresholds = _thresholds()
    identity = _identity(thresholds)
    baseline, _ = _metrics()
    evidence = seal_p50_evidence(
        status="FAIL",
        run_identity=identity,
        resolved_thresholds=thresholds,
        completed_optimizer_steps=7,
        checkpoint_path=None,
        observed_ordered_training_stream_sha256=None,
        family_gate_updates=_updates(0),
        action_route_updates=_updates(0),
        baseline_validation=baseline,
        final_validation=None,
        retention=P50RetentionEvidence.scratch(),
        failure_reason="nonfinite gradient at step 8",
    )
    assert validate_p50_evidence(evidence).status == "FAIL"
    checkpoint = _checkpoint(tmp_path / "unrelated.pt")
    assert not p50_pass_authorizes_p500(
        evidence,
        expected_run_identity=identity,
        checkpoint_path=checkpoint,
    )


def test_resume_nonfinite_metrics_and_wrong_checkpoint_are_rejected(
    tmp_path: Path,
) -> None:
    thresholds = _thresholds()
    with pytest.raises(P50GateError, match="resume"):
        _identity(thresholds, resume=True)

    identity = _identity(thresholds)
    baseline, final = _metrics()
    final[P50_SELECTION_METRIC] = float("nan")
    checkpoint = _checkpoint(tmp_path / "wrong_step.pt", completed_steps=49)
    with pytest.raises(P50GateError, match="finite"):
        seal_p50_evidence(
            status="PASS",
            run_identity=identity,
            resolved_thresholds=thresholds,
            completed_optimizer_steps=50,
            checkpoint_path=checkpoint,
            observed_ordered_training_stream_sha256=(
                identity.planned_ordered_training_stream_sha256
            ),
            family_gate_updates=_updates(),
            action_route_updates=_updates(),
            baseline_validation=baseline,
            final_validation=final,
            retention=P50RetentionEvidence.scratch(),
        )

    _, final = _metrics()
    with pytest.raises(P50GateError, match="step-50"):
        seal_p50_evidence(
            status="PASS",
            run_identity=identity,
            resolved_thresholds=thresholds,
            completed_optimizer_steps=50,
            checkpoint_path=checkpoint,
            observed_ordered_training_stream_sha256=(
                identity.planned_ordered_training_stream_sha256
            ),
            family_gate_updates=_updates(),
            action_route_updates=_updates(),
            baseline_validation=baseline,
            final_validation=final,
            retention=P50RetentionEvidence.scratch(),
        )
