"""Semantic T1 decisions authorize P50 only from exact physical evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from compose_v4.experiments.editing_v2_semantic_gate_zero import (
    EVIDENCE_SCHEMA,
    EVIDENCE_SCHEMA_VERSION,
    EVIDENCE_STATUS,
)
from compose_v4.experiments.editing_v2_semantic_t1_artifact_contracts import (
    CACHE_COMPLETION_SCHEMA,
    CACHE_COMPLETION_SCHEMA_VERSION,
    CACHE_COMPLETION_STATUS,
    NO_DOWNSTREAM_AUTHORITY,
)
from compose_v4.experiments.editing_v2_semantic_t1_decision import (
    CAPACITY_POLICY_SHA256,
    COMPLETION_FILENAME,
    DECISION_FILENAME,
    DECISION_GO_STATUS,
    DECISION_NO_GO_STATUS,
    RESULT_FILENAME,
    SemanticT1DecisionError,
    build_semantic_t1_capacity_completion,
    build_semantic_t1_capacity_decision,
    build_semantic_t1_capacity_result,
    build_semantic_t1_execution_environment,
    build_semantic_t1_result_provenance,
    validate_semantic_t1_capacity_decision,
    validate_semantic_t1_capacity_result,
)
from compose_v4.experiments.editing_v2_semantic_t1_prepared_inputs import (
    NO_AUTHORITY as PREPARED_NO_AUTHORITY,
)
from compose_v4.experiments.editing_v2_semantic_t1_prepared_inputs import (
    SCHEMA as PREPARED_SCHEMA,
)
from compose_v4.experiments.editing_v2_semantic_t1_prepared_inputs import (
    SCHEMA_VERSION as PREPARED_SCHEMA_VERSION,
)
from compose_v4.experiments.editing_v2_semantic_t1_prepared_inputs import (
    STATUS as PREPARED_STATUS,
)
from compose_v4.experiments.editing_v2_semantic_t1_prepared_inputs import (
    SemanticT1PreparedInputError,
    semantic_t1_prepared_input_implementation_sha256,
)
from compose_v4.experiments.successor_micro_overfit import (
    RINGCORE_EDITING_FAMILIES,
)

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_BYTES = b"semantic-t1-selected-checkpoint-fixture"
_CONTEXT = {
    "atom_insert": "one_neighbor_birth",
    "atom_delete": "leaf_death",
    "atom_restate": "element_identity_change",
    "bond_reorder": "bond_order_increase",
    "bond_reroute": "single_atom_pendant_acyclic_source",
    "cycle_insert": "close_to_monocyclic_ring_system",
    "cycle_attach": "open_from_monocyclic_ring_system",
    "ring_system_restate": "aromatization",
}


def _bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _sha(value: object) -> str:
    return hashlib.sha256(_bytes(value).rstrip(b"\n")).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_bytes(value))


def _gate_zero(initial_model: str, inventory: str) -> dict[str, object]:
    body = {
        "schema": EVIDENCE_SCHEMA,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": EVIDENCE_STATUS,
        **NO_DOWNSTREAM_AUTHORITY,
        "structural_result": "PASS",
        "decision_source_inventory_sha256": inventory,
        "model_runtime_identity": {
            "initial_model_state_sha256": initial_model,
        },
    }
    return {**body, "evidence_sha256": _sha(body)}


def _cache(initial_model: str, inventory: str, *, entries: int = 512) -> dict[str, object]:
    body = {
        "schema": CACHE_COMPLETION_SCHEMA,
        "schema_version": CACHE_COMPLETION_SCHEMA_VERSION,
        "status": CACHE_COMPLETION_STATUS,
        **NO_DOWNSTREAM_AUTHORITY,
        "run_identity_sha256": "1" * 64,
        "build_identity_sha256": "2" * 64,
        "source_revision_sha256": "3" * 64,
        "panel_completion_sha256": "4" * 64,
        "panel_artifact_sha256": "5" * 64,
        "decision_source_inventory_sha256": inventory,
        "initial_model_state_sha256": initial_model,
        "plan_artifact_path": "semantic/t1/plan.json",
        "plan_file_sha256": "6" * 64,
        "plan_sha256": "7" * 64,
        "manifest_artifact_path": "semantic/t1/manifest.json",
        "manifest_file_sha256": "8" * 64,
        "manifest_sha256": "9" * 64,
        "leaf_count": 5,
        "record_count": entries * 2,
        "panel_entry_binding_count": entries,
        "unique_state_successor_cache_compiled": True,
        "repeated_state_empirical_law_compiled": False,
        "training_launched": False,
        "next_stage_authorized": None,
    }
    return {**body, "completion_sha256": _sha(body)}


def _prepared(cache: dict[str, object]) -> dict[str, object]:
    body = {
        "schema": PREPARED_SCHEMA,
        "schema_version": PREPARED_SCHEMA_VERSION,
        "status": PREPARED_STATUS,
        **PREPARED_NO_AUTHORITY,
        "capacity_policy_sha256": CAPACITY_POLICY_SHA256,
        "cache_completion_sha256": cache["completion_sha256"],
        "cache_manifest_sha256": cache["manifest_sha256"],
        "panel_artifact_sha256": cache["panel_artifact_sha256"],
        "decision_source_inventory_sha256": cache["decision_source_inventory_sha256"],
        "initial_model_state_sha256": cache["initial_model_state_sha256"],
        "cache_source_revision_sha256": cache["source_revision_sha256"],
        "implementation_sha256": semantic_t1_prepared_input_implementation_sha256(repo_root=ROOT),
    }
    return {**body, "artifact_sha256": _sha(body)}


def _environment() -> dict[str, object]:
    return build_semantic_t1_execution_environment(
        hardware_class="nvidia_test_gpu",
        device_name="fixture-gpu",
        device_capability="8.0",
        accelerator_class="gpu",
        dtype="float32",
        mixed_precision=False,
        batch_size=64,
        python_version="3.test",
        torch_version="2.test",
        cuda_version="12.test",
        cudnn_version="9.test",
        rdkit_version="test",
        numpy_version="2.test",
    )


def _entries(probability: float = 0.9, *, entries_per_family: int = 64) -> list[dict[str, object]]:
    return [
        {
            "panel_entry_sha256": format(
                family_index * entries_per_family + entry_index + 1, "064x"
            ),
            "family": family,
            "semantic_cell_id": f"editing_v2_active8_v1:{family}:{_CONTEXT[family]}",
            "teacher_successor_probability": probability,
            "canonical_successor_nll": -math.log(probability),
            "teacher_successor_rank": 1,
            "teacher_successor_top1": True,
        }
        for family_index, family in enumerate(RINGCORE_EDITING_FAMILIES)
        for entry_index in range(entries_per_family)
    ]


def _gradients(*, zero_family: str | None = None) -> list[dict[str, object]]:
    return [
        {
            "family": family,
            "family_gate_gradient_finite": True,
            "family_gate_cumulative_l2": 0.0 if family == zero_family else 1.0,
            "family_gate_nonzero_update_steps": 0 if family == zero_family else 1,
            "action_route_gradient_finite": True,
            "action_route_cumulative_l2": 1.0,
            "action_route_nonzero_update_steps": 1,
        }
        for family in RINGCORE_EDITING_FAMILIES
    ]


def _integrity(*, resumed: bool = False) -> dict[str, object]:
    return {
        "optimizer_steps_completed": 1,
        "evaluation_steps": [0, 1],
        "selected_step": 1,
        "termination_reason": "all_thresholds_passed_early",
        "resume_requested": resumed,
        "resume_count": 1 if resumed else 0,
        "abort_triggered": False,
        "abort_reasons": [],
        "nonfinite_event_count": 0,
        "unsupported_teacher_count": 0,
        "missing_candidate_count": 0,
        "provenance_drift_detected": False,
        "stream_identity_drift_detected": False,
        "deterministic_algorithms_enabled": True,
        "dtype": "float32",
        "mixed_precision": False,
        "hazard_included": False,
        "address_stream_sha256": "a" * 64,
        "selected_model_state_sha256": "b" * 64,
        "selected_checkpoint_file_sha256": hashlib.sha256(CHECKPOINT_BYTES).hexdigest(),
        "optimizer_state_sha256": "c" * 64,
    }


def _result(
    provenance: dict[str, object],
    *,
    probability: float = 0.9,
    resumed: bool = False,
    zero_gradient_family: str | None = None,
    entries_per_family: int = 64,
) -> dict[str, object]:
    entries = _entries(probability, entries_per_family=entries_per_family)
    selected_nll = -math.log(probability)
    return build_semantic_t1_capacity_result(
        provenance=provenance,
        run_integrity=_integrity(resumed=resumed),
        evaluation_trajectory=[
            {
                "step": 0,
                "minimum_entry_teacher_successor_probability": 0.2,
                "mean_entry_canonical_successor_nll": -math.log(0.2),
                "model_state_sha256": "d" * 64,
            },
            {
                "step": 1,
                "minimum_entry_teacher_successor_probability": probability,
                "mean_entry_canonical_successor_nll": selected_nll,
                "model_state_sha256": "b" * 64,
            },
        ],
        entry_metrics=entries,
        gradient_evidence=_gradients(zero_family=zero_gradient_family),
    )


def _physical_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    probability: float = 0.9,
    resumed: bool = False,
    zero_gradient_family: str | None = None,
    entries_per_family: int = 64,
) -> tuple[Path, dict[str, object]]:
    monkeypatch.setattr(
        "compose_v4.experiments.editing_v2_semantic_t1_decision.validate_semantic_t1_prepared_inputs",
        lambda value, **_kwargs: dict(value),
    )
    run = tmp_path / "run"
    run.mkdir()
    initial_model = "e" * 64
    inventory = "f" * 64
    gate_zero = _gate_zero(initial_model, inventory)
    cache = _cache(
        initial_model,
        inventory,
        entries=len(RINGCORE_EDITING_FAMILIES) * entries_per_family,
    )
    gate_zero_path = run / "GATE_ZERO_EVIDENCE.json"
    cache_path = run / "SEMANTIC_T1_SUCCESSOR_CACHE_COMPLETE.json"
    prepared_path = run / "SEMANTIC_T1_PREPARED_INPUTS.json"
    selected_checkpoint_path = run / "SEMANTIC_T1_SELECTED_CHECKPOINT.pt"
    _write(gate_zero_path, gate_zero)
    _write(cache_path, cache)
    _write(prepared_path, _prepared(cache))
    selected_checkpoint_path.write_bytes(CHECKPOINT_BYTES)
    provenance = build_semantic_t1_result_provenance(
        cache_completion_path=cache_path,
        gate_zero_evidence_path=gate_zero_path,
        prepared_input_path=prepared_path,
        runner_source_revision_sha256="a" * 64,
        execution_environment=_environment(),
        repo_root=ROOT,
    )
    result = _result(
        provenance,
        probability=probability,
        resumed=resumed,
        zero_gradient_family=zero_gradient_family,
        entries_per_family=entries_per_family,
    )
    result_path = run / RESULT_FILENAME
    _write(result_path, result)
    completion = build_semantic_t1_capacity_completion(
        artifact_directory=run,
        result_path=result_path,
        cache_completion_path=cache_path,
        gate_zero_evidence_path=gate_zero_path,
        prepared_input_path=prepared_path,
        selected_checkpoint_path=selected_checkpoint_path,
        repo_root=ROOT,
    )
    completion_path = run / COMPLETION_FILENAME
    _write(completion_path, completion)
    decision = build_semantic_t1_capacity_decision(
        completion_path=completion_path,
        repo_root=ROOT,
    )
    decision_path = run / DECISION_FILENAME
    _write(decision_path, decision)
    return decision_path, decision


def test_exact_semantic_unique_state_chain_authorizes_only_p50(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decision_path, decision = _physical_chain(tmp_path, monkeypatch)
    observed = validate_semantic_t1_capacity_decision(
        decision,
        decision_path=decision_path,
        repo_root=ROOT,
        expected_gate_zero_evidence_file_sha256=decision["gate_zero_evidence_file_sha256"],
        require_p50_go=True,
    )
    assert observed["status"] == DECISION_GO_STATUS
    assert observed["bounded_p50_authorized"] is True
    assert observed["long_training_authorized"] is False
    assert observed["repeated_state_empirical_law_required_for_p50"] is False


@pytest.mark.parametrize(
    ("kwargs", "failure_fragment"),
    [
        ({"probability": 0.7}, "entry:"),
        ({"resumed": True}, "run_integrity:resume"),
        ({"zero_gradient_family": "cycle_attach"}, "gradient:cycle_attach"),
    ],
)
def test_threshold_resume_and_gradient_defects_are_durable_no_go(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, object],
    failure_fragment: str,
) -> None:
    decision_path, decision = _physical_chain(tmp_path, monkeypatch, **kwargs)
    observed = validate_semantic_t1_capacity_decision(
        decision,
        decision_path=decision_path,
        repo_root=ROOT,
    )
    assert observed["status"] == DECISION_NO_GO_STATUS
    assert observed["bounded_p50_authorized"] is False
    assert any(failure_fragment in item for item in observed["failed_checks"])
    with pytest.raises(SemanticT1DecisionError, match="not an exact bounded-P50 GO"):
        validate_semantic_t1_capacity_decision(
            decision,
            decision_path=decision_path,
            repo_root=ROOT,
            require_p50_go=True,
        )


def test_sixty_three_entries_per_family_is_durable_no_go(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decision_path, decision = _physical_chain(
        tmp_path,
        monkeypatch,
        entries_per_family=63,
    )
    observed = validate_semantic_t1_capacity_decision(
        decision,
        decision_path=decision_path,
        repo_root=ROOT,
    )
    assert observed["status"] == DECISION_NO_GO_STATUS
    assert observed["bounded_p50_authorized"] is False
    assert all(
        f"family:{family}:entry_count_minimum" in observed["failed_checks"]
        for family in RINGCORE_EDITING_FAMILIES
    )


def test_result_aggregate_cannot_disagree_with_exact_entries() -> None:
    provenance = {
        name: "1" * 64
        for name in (
            "capacity_policy_file_sha256",
            "capacity_policy_sha256",
            "cell_role_policy_sha256",
            "cache_completion_file_sha256",
            "cache_completion_sha256",
            "cache_manifest_file_sha256",
            "cache_manifest_sha256",
            "cache_run_identity_sha256",
            "cache_build_identity_sha256",
            "cache_source_revision_sha256",
            "panel_completion_sha256",
            "panel_artifact_sha256",
            "decision_source_inventory_sha256",
            "gate_zero_evidence_file_sha256",
            "gate_zero_evidence_sha256",
            "initial_model_state_sha256",
            "prepared_input_file_sha256",
            "prepared_input_artifact_sha256",
            "prepared_input_implementation_sha256",
            "runner_implementation_sha256",
            "runner_source_revision_sha256",
        )
    }
    provenance["panel_entry_binding_count"] = 512
    provenance["execution_environment"] = _environment()
    result = _result(provenance)
    broken = copy.deepcopy(result)
    broken["family_metrics"][0]["teacher_successor_probability"] = 1.0
    body = {key: value for key, value in broken.items() if key != "result_sha256"}
    broken["result_sha256"] = _sha(body)
    with pytest.raises(SemanticT1DecisionError, match="recompute from entries"):
        validate_semantic_t1_capacity_result(broken)


def test_tampered_physical_result_invalidates_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decision_path, decision = _physical_chain(tmp_path, monkeypatch)
    result_path = decision_path.parent / RESULT_FILENAME
    result_path.write_bytes(result_path.read_bytes() + b" ")
    with pytest.raises(SemanticT1DecisionError, match="canonical"):
        validate_semantic_t1_capacity_decision(
            decision,
            decision_path=decision_path,
            repo_root=ROOT,
            require_p50_go=True,
        )


@pytest.mark.parametrize(
    ("relative_path", "failure_fragment"),
    [
        ("SEMANTIC_T1_PREPARED_INPUTS.json", "not readable JSON"),
        ("SEMANTIC_T1_SELECTED_CHECKPOINT.pt", "physical hash"),
    ],
)
def test_tampered_prepared_input_or_checkpoint_invalidates_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
    failure_fragment: str,
) -> None:
    decision_path, decision = _physical_chain(tmp_path, monkeypatch)
    target = decision_path.parent / relative_path
    target.write_bytes(target.read_bytes() + b"tamper")
    with pytest.raises(SemanticT1DecisionError, match=failure_fragment):
        validate_semantic_t1_capacity_decision(
            decision,
            decision_path=decision_path,
            repo_root=ROOT,
            require_p50_go=True,
        )


def test_provenance_requires_full_prepared_input_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial_model = "e" * 64
    inventory = "f" * 64
    gate_zero_path = tmp_path / "gate.json"
    cache_path = tmp_path / "cache.json"
    prepared_path = tmp_path / "prepared.json"
    cache = _cache(initial_model, inventory)
    _write(gate_zero_path, _gate_zero(initial_model, inventory))
    _write(cache_path, cache)
    _write(prepared_path, _prepared(cache))

    def reject(*args, **kwargs):
        raise SemanticT1PreparedInputError("deep partition defect")

    monkeypatch.setattr(
        "compose_v4.experiments.editing_v2_semantic_t1_decision.validate_semantic_t1_prepared_inputs",
        reject,
    )
    with pytest.raises(SemanticT1DecisionError, match="full validation"):
        build_semantic_t1_result_provenance(
            cache_completion_path=cache_path,
            gate_zero_evidence_path=gate_zero_path,
            prepared_input_path=prepared_path,
            runner_source_revision_sha256="a" * 64,
            execution_environment=_environment(),
            repo_root=ROOT,
        )


def test_legacy_or_bare_t1_decision_cannot_satisfy_semantic_gate(
    tmp_path: Path,
) -> None:
    decision_path = tmp_path / DECISION_FILENAME
    legacy = {
        "schema": "compose.editing.t1_successor_gate_decision",
        "schema_version": 3,
        "status": "PASS_BOUNDED_P50_PREREQUISITE",
        "bounded_p50_authorized": True,
    }
    _write(decision_path, legacy)
    with pytest.raises(SemanticT1DecisionError, match="missing or unknown"):
        validate_semantic_t1_capacity_decision(
            legacy,
            decision_path=decision_path,
            repo_root=ROOT,
            require_p50_go=True,
        )


def test_cache_and_gate_zero_provenance_must_match(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    gate_zero = _gate_zero("a" * 64, "b" * 64)
    cache = _cache("c" * 64, "b" * 64)
    gate_zero_path = run / "gate.json"
    cache_path = run / "cache.json"
    _write(gate_zero_path, gate_zero)
    _write(cache_path, cache)
    result_path = run / RESULT_FILENAME
    result_path.write_bytes(b"{}\n")
    with pytest.raises(SemanticT1DecisionError):
        build_semantic_t1_capacity_completion(
            artifact_directory=run,
            result_path=result_path,
            cache_completion_path=cache_path,
            gate_zero_evidence_path=gate_zero_path,
            prepared_input_path=run / "prepared.json",
            selected_checkpoint_path=run / "selected.pt",
            repo_root=ROOT,
        )
