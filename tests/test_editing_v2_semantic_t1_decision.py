"""Semantic T1 decisions authorize P50 only from exact physical evidence."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
from pathlib import Path

import pytest
import torch

from compose_v4.experiments.editing_p50_gate import state_dict_semantic_sha256
from compose_v4.experiments.editing_v2_semantic_development_cell_roles import (
    load_semantic_development_cell_roles,
)
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
from compose_v4.experiments.editing_v2_semantic_t1_capacity_policy import (
    load_semantic_t1_capacity_policy,
)
from compose_v4.experiments.editing_v2_semantic_t1_checkpoint import (
    CHECKPOINT_SCHEMA,
    CHECKPOINT_SCHEMA_VERSION,
    CHECKPOINT_STATUS,
)
from compose_v4.experiments.editing_v2_semantic_t1_checkpoint import (
    NO_AUTHORITY as CHECKPOINT_NO_AUTHORITY,
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
CHECKPOINT_MODEL_STATE = {"fixture.weight": torch.tensor([1.0], dtype=torch.float32)}
CHECKPOINT_MODEL_STATE_SHA256 = state_dict_semantic_sha256(CHECKPOINT_MODEL_STATE)


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


def _selected_checkpoint_bytes(provenance: dict[str, object]) -> bytes:
    policy = load_semantic_t1_capacity_policy(
        ROOT / "configs/editing_v2_semantic_t1_capacity_policy_v1.json"
    )
    optimization = policy["optimization"]
    optimizer_configuration = {
        "class": "fixture.Optimizer",
        "defaults": {},
        "parameter_groups": [],
    }
    environment = provenance["execution_environment"]
    identity = {
        "capacity_policy_sha256": policy["policy_sha256"],
        "optimization_policy": optimization,
        "optimization_policy_sha256": _sha(optimization),
        "prepared_input_artifact_sha256": provenance["prepared_input_artifact_sha256"],
        "cache_completion_sha256": provenance["cache_completion_sha256"],
        "cache_manifest_sha256": provenance["cache_manifest_sha256"],
        "initial_model_state_sha256": provenance["initial_model_state_sha256"],
        "runner_implementation_sha256": provenance["runner_implementation_sha256"],
        "runner_source_revision_sha256": provenance["runner_source_revision_sha256"],
        "execution_environment": environment,
        "execution_environment_sha256": environment["environment_sha256"],
        "model_device": "cuda:0",
        "model_device_type": "cuda",
        "model_device_index": 0,
        "model_dtype": "torch.float32",
        "deterministic_algorithms_enabled": True,
        "optimizer_configuration": optimizer_configuration,
        "optimizer_configuration_sha256": _sha(optimizer_configuration),
    }
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "status": CHECKPOINT_STATUS,
        **CHECKPOINT_NO_AUTHORITY,
        "identity": identity,
        "selected_step": 1,
        "model_state": CHECKPOINT_MODEL_STATE,
        "model_state_sha256": CHECKPOINT_MODEL_STATE_SHA256,
        "stream_sha256": "a" * 64,
        "torch_version": str(torch.__version__),
    }
    output = io.BytesIO()
    torch.save(payload, output)
    return output.getvalue()


def _entry_lineage_hashes(
    entries: list[dict[str, object]],
) -> tuple[str, str]:
    metadata = sorted(
        (
            {
                "panel_entry_sha256": entry["panel_entry_sha256"],
                "family": entry["family"],
                "semantic_cell_id": entry["semantic_cell_id"],
            }
            for entry in entries
        ),
        key=lambda item: item["panel_entry_sha256"],
    )
    return (
        _sha(sorted(item["panel_entry_sha256"] for item in metadata)),
        _sha(metadata),
    )


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


def _prepared(
    cache: dict[str, object],
    entry_metrics: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    result_entries = _entries() if entry_metrics is None else entry_metrics
    entries = [
        {
            "panel_entry_sha256": entry["panel_entry_sha256"],
            "model_family": entry["family"],
            "capability_cell_id": entry["semantic_cell_id"],
        }
        for entry in result_entries
    ]
    panel_entry_inventory_sha256, _ = _entry_lineage_hashes(result_entries)
    body = {
        "schema": PREPARED_SCHEMA,
        "schema_version": PREPARED_SCHEMA_VERSION,
        "status": PREPARED_STATUS,
        **PREPARED_NO_AUTHORITY,
        "capacity_policy_sha256": CAPACITY_POLICY_SHA256,
        "cache_completion_sha256": cache["completion_sha256"],
        "cache_manifest_sha256": cache["manifest_sha256"],
        "panel_artifact_sha256": cache["panel_artifact_sha256"],
        "panel_entry_inventory_sha256": panel_entry_inventory_sha256,
        "decision_source_inventory_sha256": cache["decision_source_inventory_sha256"],
        "initial_model_state_sha256": cache["initial_model_state_sha256"],
        "cache_source_revision_sha256": cache["source_revision_sha256"],
        "implementation_sha256": semantic_t1_prepared_input_implementation_sha256(repo_root=ROOT),
        "entries": entries,
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


def _result_provenance(entries: list[dict[str, object]]) -> dict[str, object]:
    provenance: dict[str, object] = {
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
            "panel_entry_inventory_sha256",
            "panel_entry_metadata_sha256",
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
    (
        provenance["panel_entry_inventory_sha256"],
        provenance["panel_entry_metadata_sha256"],
    ) = _entry_lineage_hashes(entries)
    provenance["panel_entry_binding_count"] = len(entries)
    provenance["execution_environment"] = _environment()
    return provenance


def _entries(probability: float = 0.9, *, entries_per_family: int = 64) -> list[dict[str, object]]:
    required_cells = load_semantic_development_cell_roles().required_cell_ids
    cells_by_family = {
        family: tuple(cell for cell in required_cells if cell.rsplit(":", 2)[1] == family)
        for family in RINGCORE_EDITING_FAMILIES
    }
    entries = [
        {
            "panel_entry_sha256": format(
                family_index * entries_per_family + entry_index + 1, "064x"
            ),
            "family": family,
            "semantic_cell_id": cells_by_family[family][entry_index % len(cells_by_family[family])],
            "teacher_successor_probability": probability,
            "canonical_successor_nll": -math.log(probability),
            "teacher_successor_rank": 1,
            "teacher_successor_top1": True,
        }
        for family_index, family in enumerate(RINGCORE_EDITING_FAMILIES)
        for entry_index in range(entries_per_family)
    ]
    return sorted(
        entries,
        key=lambda item: (
            RINGCORE_EDITING_FAMILIES.index(item["family"]),
            item["semantic_cell_id"],
            item["panel_entry_sha256"],
        ),
    )


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


def _integrity(
    *,
    resumed: bool = False,
    selected_checkpoint_file_sha256: str | None = None,
) -> dict[str, object]:
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
        "selected_model_state_sha256": CHECKPOINT_MODEL_STATE_SHA256,
        "selected_checkpoint_file_sha256": (
            "b" * 64 if selected_checkpoint_file_sha256 is None else selected_checkpoint_file_sha256
        ),
        "optimizer_state_sha256": "c" * 64,
    }


def _result(
    provenance: dict[str, object],
    *,
    probability: float = 0.9,
    resumed: bool = False,
    zero_gradient_family: str | None = None,
    entries_per_family: int = 64,
    entries: list[dict[str, object]] | None = None,
    selected_checkpoint_file_sha256: str | None = None,
) -> dict[str, object]:
    entries = (
        _entries(probability, entries_per_family=entries_per_family) if entries is None else entries
    )
    selected_nll = -math.log(probability)
    return build_semantic_t1_capacity_result(
        provenance=provenance,
        run_integrity=_integrity(
            resumed=resumed,
            selected_checkpoint_file_sha256=selected_checkpoint_file_sha256,
        ),
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
                "model_state_sha256": CHECKPOINT_MODEL_STATE_SHA256,
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
    entries = _entries(probability, entries_per_family=entries_per_family)
    _write(prepared_path, _prepared(cache, entries))
    provenance = build_semantic_t1_result_provenance(
        cache_completion_path=cache_path,
        gate_zero_evidence_path=gate_zero_path,
        prepared_input_path=prepared_path,
        runner_source_revision_sha256="a" * 64,
        execution_environment=_environment(),
        repo_root=ROOT,
    )
    selected_checkpoint_bytes = _selected_checkpoint_bytes(provenance)
    selected_checkpoint_path.write_bytes(selected_checkpoint_bytes)
    result = _result(
        provenance,
        probability=probability,
        resumed=resumed,
        zero_gradient_family=zero_gradient_family,
        entries_per_family=entries_per_family,
        entries=entries,
        selected_checkpoint_file_sha256=hashlib.sha256(selected_checkpoint_bytes).hexdigest(),
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
    assert observed["nonempty_cell_count"] == 17
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


def test_rehashed_result_missing_a_required_cell_cannot_reach_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decision_path, _decision = _physical_chain(tmp_path, monkeypatch)
    result_path = decision_path.parent / RESULT_FILENAME
    result = json.loads(result_path.read_bytes())
    roles = load_semantic_development_cell_roles()
    victim = next(
        cell for cell in roles.required_cell_ids if cell.endswith(":connected_nonleaf_death")
    )
    replacement = next(cell for cell in roles.required_cell_ids if cell.endswith(":leaf_death"))
    for entry in result["entry_metrics"]:
        if entry["semantic_cell_id"] == victim:
            entry["semantic_cell_id"] = replacement
    result["entry_metrics"] = sorted(
        result["entry_metrics"],
        key=lambda item: (
            RINGCORE_EDITING_FAMILIES.index(item["family"]),
            item["semantic_cell_id"],
            item["panel_entry_sha256"],
        ),
    )
    victim_metric = next(row for row in result["cell_metrics"] if row["semantic_cell_id"] == victim)
    replacement_metric = next(
        row for row in result["cell_metrics"] if row["semantic_cell_id"] == replacement
    )
    replacement_metric["entry_count"] += victim_metric["entry_count"]
    result["cell_metrics"] = [
        row for row in result["cell_metrics"] if row["semantic_cell_id"] != victim
    ]
    body = {key: value for key, value in result.items() if key != "result_sha256"}
    result["result_sha256"] = _sha(body)
    _write(result_path, result)

    with pytest.raises(
        SemanticT1DecisionError,
        match="entry inventory or metadata|exactly cover all frozen required",
    ):
        build_semantic_t1_capacity_decision(
            completion_path=decision_path.parent / COMPLETION_FILENAME,
            repo_root=ROOT,
        )


@pytest.mark.parametrize("mutation", ["missing", "foreign", "relabeled"])
def test_rehashed_result_cannot_change_exact_prepared_entry_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    decision_path, _decision = _physical_chain(tmp_path, monkeypatch)
    result_path = decision_path.parent / RESULT_FILENAME
    original = json.loads(result_path.read_bytes())
    entries = copy.deepcopy(original["entry_metrics"])
    if mutation == "missing":
        entries.pop(0)
    elif mutation == "foreign":
        entries[0]["panel_entry_sha256"] = "f" * 64
    else:
        target = next(entry for entry in entries if entry["family"] == "atom_delete")
        family = target["family"]
        replacement = next(
            cell
            for cell in load_semantic_development_cell_roles().required_cell_ids
            if cell.rsplit(":", 2)[1] == family and cell != target["semantic_cell_id"]
        )
        target["semantic_cell_id"] = replacement
    entries.sort(
        key=lambda item: (
            RINGCORE_EDITING_FAMILIES.index(item["family"]),
            item["semantic_cell_id"],
            item["panel_entry_sha256"],
        )
    )
    provenance = copy.deepcopy(original["provenance"])
    provenance["panel_entry_binding_count"] = len(entries)
    (
        provenance["panel_entry_inventory_sha256"],
        provenance["panel_entry_metadata_sha256"],
    ) = _entry_lineage_hashes(entries)
    mutated = build_semantic_t1_capacity_result(
        provenance=provenance,
        run_integrity=original["run_integrity"],
        evaluation_trajectory=original["evaluation_trajectory"],
        entry_metrics=entries,
        gradient_evidence=original["gradient_evidence"],
    )
    _write(result_path, mutated)

    with pytest.raises(SemanticT1DecisionError, match="exact prepared-panel"):
        build_semantic_t1_capacity_decision(
            completion_path=decision_path.parent / COMPLETION_FILENAME,
            repo_root=ROOT,
        )


def test_result_aggregate_cannot_disagree_with_exact_entries() -> None:
    entries = _entries()
    provenance = _result_provenance(entries)
    result = _result(provenance, entries=entries)
    broken = copy.deepcopy(result)
    broken["family_metrics"][0]["teacher_successor_probability"] = 1.0
    body = {key: value for key, value in broken.items() if key != "result_sha256"}
    broken["result_sha256"] = _sha(body)
    with pytest.raises(SemanticT1DecisionError, match="recompute from entries"):
        validate_semantic_t1_capacity_result(broken)


def test_sparse_report_point_trajectory_selects_the_actual_optimizer_step() -> None:
    entries = _entries()
    provenance = _result_provenance(entries)
    integrity = _integrity()
    integrity.update(
        {
            "optimizer_steps_completed": 10,
            "evaluation_steps": [0, 1, 10],
            "selected_step": 10,
        }
    )
    probability = 0.9
    result = build_semantic_t1_capacity_result(
        provenance=provenance,
        run_integrity=integrity,
        evaluation_trajectory=[
            {
                "step": 0,
                "minimum_entry_teacher_successor_probability": 0.2,
                "mean_entry_canonical_successor_nll": -math.log(0.2),
                "model_state_sha256": "d" * 64,
            },
            {
                "step": 1,
                "minimum_entry_teacher_successor_probability": 0.5,
                "mean_entry_canonical_successor_nll": -math.log(0.5),
                "model_state_sha256": "e" * 64,
            },
            {
                "step": 10,
                "minimum_entry_teacher_successor_probability": probability,
                "mean_entry_canonical_successor_nll": -math.log(probability),
                "model_state_sha256": CHECKPOINT_MODEL_STATE_SHA256,
            },
        ],
        entry_metrics=entries,
        gradient_evidence=_gradients(),
    )
    assert result["run_integrity"]["selected_step"] == 10
    assert [row["step"] for row in result["evaluation_trajectory"]] == [0, 1, 10]


def test_sparse_report_point_trajectory_refuses_an_unregistered_step() -> None:
    entries = _entries()
    provenance = _result_provenance(entries)
    integrity = _integrity()
    integrity.update(
        {
            "optimizer_steps_completed": 10,
            "evaluation_steps": [0, 1, 9],
            "selected_step": 9,
        }
    )
    with pytest.raises(SemanticT1DecisionError, match="run-integrity"):
        build_semantic_t1_capacity_result(
            provenance=provenance,
            run_integrity=integrity,
            evaluation_trajectory=[
                {
                    "step": step,
                    "minimum_entry_teacher_successor_probability": probability,
                    "mean_entry_canonical_successor_nll": -math.log(probability),
                    "model_state_sha256": CHECKPOINT_MODEL_STATE_SHA256,
                }
                for step, probability in ((0, 0.2), (1, 0.5), (9, 0.9))
            ],
            entry_metrics=entries,
            gradient_evidence=_gradients(),
        )


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
        ("SEMANTIC_T1_SELECTED_CHECKPOINT.pt", "physical SHA-256"),
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


@pytest.mark.parametrize(
    ("mutation", "failure_fragment"),
    [
        ("selected_step", "step, stream, or runtime"),
        ("model_state", "model-state identity"),
        ("identity", "differs from reopened provenance"),
    ],
)
def test_selected_checkpoint_semantics_are_bound_to_result_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    failure_fragment: str,
) -> None:
    decision_path, _decision = _physical_chain(tmp_path, monkeypatch)
    run = decision_path.parent
    checkpoint_path = run / "SEMANTIC_T1_SELECTED_CHECKPOINT.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if mutation == "selected_step":
        payload["selected_step"] = 2
    elif mutation == "model_state":
        payload["model_state"] = {"fixture.weight": torch.tensor([2.0])}
    else:
        payload["identity"]["capacity_policy_sha256"] = "f" * 64
    torch.save(payload, checkpoint_path)

    result_path = run / RESULT_FILENAME
    result = json.loads(result_path.read_bytes())
    result["run_integrity"]["selected_checkpoint_file_sha256"] = hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()
    body = {key: value for key, value in result.items() if key != "result_sha256"}
    result["result_sha256"] = _sha(body)
    _write(result_path, result)

    with pytest.raises(SemanticT1DecisionError, match=failure_fragment):
        build_semantic_t1_capacity_completion(
            artifact_directory=run,
            result_path=result_path,
            cache_completion_path=run / "SEMANTIC_T1_SUCCESSOR_CACHE_COMPLETE.json",
            gate_zero_evidence_path=run / "GATE_ZERO_EVIDENCE.json",
            prepared_input_path=run / "SEMANTIC_T1_PREPARED_INPUTS.json",
            selected_checkpoint_path=checkpoint_path,
            repo_root=ROOT,
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
