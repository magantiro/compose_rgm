from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import compose_v4.experiments.editing_t1_atom_insert_schedule_v3 as v3
from compose_v4.experiments.editing_p50_gate import state_dict_semantic_sha256
from compose_v4.experiments.editing_t1_uniform_successor_capacity_v2 import (
    EXPECTED_V1_THRESHOLDS,
    stable_sha256,
)
from compose_v4.experiments.successor_micro_overfit import SuccessorMicroOverfitError

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "editing_t1_atom_insert_schedule_v3.json"


class _ToyBatch:
    def __init__(self, count: int) -> None:
        self.teacher_rates = torch.ones(count)
        self.importance_weights = torch.ones(count)

    def to(self, device: torch.device) -> _ToyBatch:
        self.teacher_rates = self.teacher_rates.to(device)
        self.importance_weights = self.importance_weights.to(device)
        return self


class _ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = torch.nn.Parameter(torch.tensor(0.0))
        self.family_head = torch.nn.Linear(1, 1, bias=False)
        self.grow_root_head = torch.nn.Linear(1, 1, bias=False)
        self.grow_query = torch.nn.Linear(1, 1, bias=False)
        self.grow_option = torch.nn.Embedding(1, 1)
        self.total_hazard_head = torch.nn.Linear(1, 1)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.zero_()


def _toy_probability(model: _ToyModel) -> torch.Tensor:
    score = (
        model.body
        + model.family_head.weight[0, 0]
        + model.grow_root_head.weight[0, 0]
        + model.grow_query.weight[0, 0]
        + model.grow_option.weight[0, 0]
    )
    return torch.sigmoid(score)


def _toy_forward(model, batch, fibers):
    del fibers
    probability = _toy_probability(model)
    values = probability.expand(batch.teacher_rates.shape[0])
    return SimpleNamespace(
        selected_productive_successor_log_probability=torch.log(values),
        total_hazard=torch.ones_like(values),
    )


def _toy_loss(prediction, batch):
    del batch
    return -prediction.selected_productive_successor_log_probability.mean()


def _toy_metrics(model, panel, *, include_per_example=True):
    probability = float(_toy_probability(model).detach())
    nll = -math.log(probability)
    rank = 1 if probability > 0.5 else 2
    result = {
        "n_examples": len(panel.examples),
        "families": ["atom_insert"],
        "canonical_successor_nll": nll,
        "teacher_successor_probability": probability,
        "teacher_successor_top1_recall": float(rank == 1),
    }
    if include_per_example:
        result["per_example"] = [
            {
                "teacher_successor_probability": probability,
                "teacher_successor_nll": nll,
                "teacher_successor_rank": rank,
            }
            for _ in panel.examples
        ]
    return result


def _toy_panel(count: int = 64):
    return SimpleNamespace(
        examples=tuple(SimpleNamespace(family_name="atom_insert") for _ in range(count)),
        batch=_ToyBatch(count),
        fibers=tuple(range(count)),
    )


@pytest.fixture
def toy_runtime(monkeypatch):
    monkeypatch.setattr(v3, "forward_teacher_successor_batch", _toy_forward)
    monkeypatch.setattr(v3, "factorized_successor_identity_loss", _toy_loss)
    monkeypatch.setattr(v3, "successor_panel_metrics", _toy_metrics)


def _run_toy(
    root: Path,
    *,
    expected_boundary: str | None,
    resume: Path | None = None,
):
    model = _ToyModel()
    initial = state_dict_semantic_sha256(model.state_dict())
    report = v3.train_atom_insert_schedule_v3(
        model,
        _toy_panel(),
        phase1_updates=2,
        phase1_learning_rate=0.01,
        phase2_updates=2,
        phase2_learning_rate=0.001,
        expected_initial_model_sha256=initial,
        expected_phase1_model_sha256=expected_boundary,
        weight_decay=0.0,
        scope="all",
        seed=20260730,
        gradient_clip_norm=10.0,
        thresholds=EXPECTED_V1_THRESHOLDS,
        checkpoint_root=root,
        checkpoint_identity={"contract": "test", "panel": "fixed"},
        checkpoint_interval=1,
        resume_checkpoint=resume,
    )
    return model, report


def test_v3_contract_is_sealed_schedule_only_and_non_authorizing() -> None:
    contract = v3.load_atom_insert_schedule_v3_contract(CONTRACT)
    optimization = contract["optimization_law"]
    assert contract["training_authorized"] is False
    assert contract["bounded_p50_authorized"] is False
    assert contract["family"] == "atom_insert"
    assert optimization["phase_1"]["completed_updates"] == 495
    assert optimization["phase_1"]["learning_rate"] == 0.001
    assert optimization["phase_2"]["maximum_additional_updates"] == 250
    assert optimization["phase_2"]["learning_rate"] == 0.0001
    assert optimization["retain_optimizer_state_across_phase_boundary"] is True
    assert optimization["patience_or_plateau_stop"] is False
    assert contract["evaluation"]["thresholds"] == EXPECTED_V1_THRESHOLDS


def test_v3_contract_rejects_rehashed_schedule_mutation(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_bytes())
    payload["optimization_law"]["phase_2"]["learning_rate"] = 0.0002
    body = {key: value for key, value in payload.items() if key != "contract_sha256"}
    payload["contract_sha256"] = stable_sha256(body)
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SuccessorMicroOverfitError, match="identity|optimization"):
        v3.load_atom_insert_schedule_v3_contract(path)


def test_checkpoint_roundtrip_binds_model_optimizer_rng_and_identity(tmp_path: Path) -> None:
    model = _ToyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    cpu_rng = torch.get_rng_state().clone()
    body = {
        "schema": v3.V3_CHECKPOINT_SCHEMA,
        "schema_version": v3.V3_CHECKPOINT_VERSION,
        "status": "COMPLETE_ATOM_INSERT_V3_CONTINUATION_STATE",
        "training_authorized": False,
        "bounded_p50_authorized": False,
        "identity": {"contract": "fixed"},
        "checkpoint_reason": "TEST",
        "completed_update": 495,
        "phase": "phase_2",
        "model_state": copy.deepcopy(model.state_dict()),
        "model_state_sha256": state_dict_semantic_sha256(model.state_dict()),
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "optimizer_state_sha256": v3.checkpoint_semantic_sha256(optimizer.state_dict()),
        "cpu_rng_state": cpu_rng,
        "cuda_rng_states": [],
        "rng_state_sha256": v3.checkpoint_semantic_sha256({"cpu": cpu_rng, "cuda": []}),
        "trajectory": [],
        "selected_criterion_point": {"update": 0},
        "selected_update": 0,
        "selected_state": copy.deepcopy(model.state_dict()),
        "finite_nonzero_gradient_seen": {},
        "gradient_update_counts": {},
        "component_gradient_update_counts": {},
        "optimizer_steps_with_nonzero_gradient": 0,
        "initial_metrics": {},
        "initial_model_state_sha256": state_dict_semantic_sha256(model.state_dict()),
        "initial_hazard_state": {},
        "checkpoint_receipts": [],
    }
    receipt = v3.publish_v3_checkpoint(tmp_path, body)
    assert Path(receipt["checkpoint_local_path"]).is_file()
    assert Path(receipt["receipt_local_path"]).is_file()
    loaded = v3.load_v3_checkpoint(
        Path(receipt["checkpoint_local_path"]), expected_identity={"contract": "fixed"}
    )
    assert loaded["completed_update"] == 495
    assert loaded["model_state_sha256"] == body["model_state_sha256"]
    with pytest.raises(SuccessorMicroOverfitError, match="identity"):
        v3.load_v3_checkpoint(
            Path(receipt["checkpoint_local_path"]), expected_identity={"contract": "changed"}
        )


def test_two_phase_checkpoint_resume_is_trajectory_equivalent(
    tmp_path: Path,
    toy_runtime,
) -> None:
    _, derivation = _run_toy(tmp_path / "derive", expected_boundary=None)
    boundary_receipt = next(
        receipt
        for receipt in derivation["checkpoint_receipts"]
        if receipt["reason"] == "PHASE_BOUNDARY"
    )
    boundary_hash = boundary_receipt["model_state_sha256"]
    boundary_state = v3.load_v3_checkpoint(
        Path(boundary_receipt["checkpoint_local_path"]),
        expected_identity={"contract": "test", "panel": "fixed"},
    )
    assert boundary_state["optimizer_state"]["param_groups"][0]["lr"] == 0.001
    assert {
        int(value["step"]) for value in boundary_state["optimizer_state"]["state"].values()
    } == {2}

    _, uninterrupted = _run_toy(tmp_path / "uninterrupted", expected_boundary=boundary_hash)
    resume_receipt = next(
        receipt
        for receipt in uninterrupted["checkpoint_receipts"]
        if receipt["completed_update"] == 3 and receipt["reason"] == "PERIODIC"
    )
    _, resumed = _run_toy(
        tmp_path / "resumed",
        expected_boundary=boundary_hash,
        resume=Path(resume_receipt["checkpoint_local_path"]),
    )
    assert resumed["trajectory_sha256"] == uninterrupted["trajectory_sha256"]
    assert resumed["selected_update"] == uninterrupted["selected_update"]
    assert resumed["selected_model_state_sha256"] == uninterrupted["selected_model_state_sha256"]
    assert resumed["terminal_model_state_sha256"] == uninterrupted["terminal_model_state_sha256"]
    assert resumed["threshold_checks"] == uninterrupted["threshold_checks"]

    contract = copy.deepcopy(v3.load_atom_insert_schedule_v3_contract(CONTRACT))
    contract["optimization_law"]["phase_1"] = {
        **contract["optimization_law"]["phase_1"],
        "completed_updates": 2,
        "learning_rate": 0.01,
        "required_terminal_model_state_sha256": boundary_hash,
    }
    contract["optimization_law"]["phase_2"] = {
        **contract["optimization_law"]["phase_2"],
        "maximum_additional_updates": 2,
        "learning_rate": 0.001,
    }
    contract["optimization_law"]["maximum_total_updates"] = 4
    assert (
        v3.validate_atom_insert_v3_training_report(uninterrupted, contract=contract)
        is uninterrupted
    )


def test_phase_boundary_hash_mismatch_fails_before_refinement(
    tmp_path: Path,
    toy_runtime,
) -> None:
    with pytest.raises(SuccessorMicroOverfitError, match="update-495 model hash"):
        _run_toy(tmp_path, expected_boundary="f" * 64)
