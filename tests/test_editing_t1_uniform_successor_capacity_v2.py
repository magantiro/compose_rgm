from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import modal_apps.run_editing_t1_uniform_successor_capacity_v2 as modal_runner
import compose_v4.experiments.editing_t1_uniform_successor_capacity_v2 as capacity_v2
import scripts.run_editing_t1_uniform_successor_capacity_v2 as v2_runner
from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY
from compose_v4.experiments.cycle_op_prior import build_cycle_op_records
from compose_v4.experiments.successor_micro_overfit import (
    SuccessorMicroOverfitError,
    examples_from_traces,
    prepare_successor_panel,
)
from compose_v4.model.factorized_tracelet_rate_model import FactorizedTraceletRateModel
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "editing_t1_uniform_successor_capacity_v2.json"


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
        self.cycle_open_head = torch.nn.Linear(1, 1, bias=False)
        self.total_hazard_head = torch.nn.Linear(1, 1, bias=True)
        with torch.no_grad():
            self.family_head.weight.zero_()
            self.cycle_open_head.weight.zero_()
            self.total_hazard_head.weight.fill_(3.25)
            self.total_hazard_head.bias.fill_(-1.75)


def _toy_probability(model: _ToyModel) -> torch.Tensor:
    score = model.body + model.family_head.weight[0, 0] + model.cycle_open_head.weight[0, 0]
    return torch.sigmoid(score)


def _toy_forward(model, batch, fibers):
    del fibers
    probability = _toy_probability(model)
    values = probability.expand(batch.teacher_rates.shape[0])
    return SimpleNamespace(
        selected_productive_successor_log_probability=torch.log(values),
        total_hazard=torch.ones_like(values),
    )


def _toy_metrics(model, panel, *, include_per_example=True):
    probability = float(_toy_probability(model).detach())
    nll = -math.log(probability)
    rank = 1 if probability > 0.5 else 2
    metrics = {
        "n_examples": len(panel.examples),
        "families": ["cycle_attach"],
        "canonical_successor_nll": nll,
        "teacher_successor_probability": probability,
        "teacher_successor_top1_recall": float(rank == 1),
    }
    if include_per_example:
        metrics["per_example"] = [
            {
                "teacher_successor_probability": probability,
                "teacher_successor_nll": nll,
                "teacher_successor_rank": rank,
            }
            for _ in panel.examples
        ]
    return metrics


def _toy_panel(count: int = 64):
    examples = tuple(SimpleNamespace(family_name="cycle_attach") for _ in range(count))
    return SimpleNamespace(examples=examples, batch=_ToyBatch(count), fibers=tuple(range(count)))


def _sealed_outer_result(report, contract):
    digest = "a" * 64
    active8 = {field: digest for field in v2_runner.ACTIVE8_T1_IDENTITY_FIELDS}
    body = {
        "schema": capacity_v2.UNIFORM_V2_RESULT_SCHEMA,
        "schema_version": capacity_v2.UNIFORM_V2_RESULT_VERSION,
        "status": capacity_v2.UNIFORM_V2_RESULT_STATUS,
        "training_authorized": False,
        "bounded_p50_authorized": False,
        "gate_decision": None,
        "diagnostic_scope": "INDEPENDENT_FAMILY_LOCAL_SCRATCH_MODEL",
        "source_commit": "b" * 40,
        "source_commit_binding": (
            "CLEAN_LAUNCHER_HEAD_ATTESTATION_PLUS_WORKER_VERIFIED_SERIALIZED_TREE"
        ),
        "serialized_source_tree_sha256": "c" * 64,
        "uniform_v2_contract_sha256": contract["contract_sha256"],
        "uniform_v2_contract_file_sha256": digest,
        "parent_v1_uniform_contract_sha256": contract["parents"]["v1_contract_sha256"],
        "parent_v1_uniform_contract_file_sha256": contract["parents"]["v1_contract_file_sha256"],
        "parent_t1_runtime_contract_sha256": contract["parents"]["runtime_contract_sha256"],
        "parent_t1_runtime_contract_file_sha256": contract["parents"][
            "runtime_contract_file_sha256"
        ],
        "gate_zero_runtime_contract_sha256": digest,
        "panel_artifact_sha256": contract["parents"]["panel_artifact_sha256"],
        "panel_selection_sha256": digest,
        "panel_census_sha256": digest,
        "panel_capacity_strata_sha256": digest,
        **active8,
        "family": "cycle_attach",
        "panel_kind": "unique_state",
        "scope": "all",
        "example_count": 64,
        "unique_progress_address_count": 64,
        "source_row_sha256s": [f"{index:064x}" for index in range(64)],
        "operator_identity": {
            "operator_capability_fingerprint": "test",
            "enable_ring_system_delete": False,
            "enable_ring_grow_macro": False,
            "enable_cycle_ops": True,
        },
        "weight_policy": {
            "teacher_rate": 1.0,
            "importance_coefficient": 1.0,
            "original_coefficient_audit": {"count": 64},
        },
        "training_report": report,
        "optimization_law_sha256": capacity_v2.stable_sha256(contract["optimization_law"]),
        "selected_update": report["selected_update"],
        "terminal_update": report["terminal_update"],
        "thresholds": report["thresholds"],
        "threshold_checks": report["threshold_checks"],
        "per_example_floor_metrics": report["per_example_floor_metrics"],
        "all_threshold_checks_pass": report["all_threshold_checks_pass"],
        "initial_model_state_sha256": report["initial_model_state_sha256"],
        "terminal_model_state_sha256": report["terminal_model_state_sha256"],
        "selected_model_state_sha256": report["selected_model_state_sha256"],
        "returned_model_state_sha256": report["returned_model_state_sha256"],
        "cache_manifest_receipt": {"test": True},
        "cache_receipts": [{"test": True}],
        "runner_file_sha256": digest,
        "optimizer_implementation_file_sha256": digest,
    }
    return {**body, "result_sha256": capacity_v2.stable_sha256(body)}


def test_v2_contract_is_self_hashed_non_authorizing_and_preserves_v1() -> None:
    contract = capacity_v2.load_uniform_capacity_v2_contract(CONTRACT)
    assert contract["training_authorized"] is False
    assert contract["bounded_p50_authorized"] is False
    assert contract["panel_policy"]["examples_per_family"] == 64
    assert contract["optimization_law"]["teacher_rate"] == 1.0
    assert contract["optimization_law"]["importance_coefficient"] == 1.0
    assert contract["optimization_law"]["maximum_full_panel_updates"] == 500
    assert contract["optimization_law"]["seed"] == 20260730
    assert contract["optimization_law"]["excluded_parameter_prefixes"] == ["total_hazard_head."]
    assert contract["evaluation"]["thresholds"] == capacity_v2.EXPECTED_V1_THRESHOLDS
    assert contract["post_v1_rationale"]["v1_claim_boundary"].startswith("V2 does not reinterpret")


def test_v2_contract_rejects_rehashed_post_freeze_mutation(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_bytes())
    payload["optimization_law"]["maximum_full_panel_updates"] = 501
    body = {key: value for key, value in payload.items() if key != "contract_sha256"}
    payload["contract_sha256"] = capacity_v2.stable_sha256(body)
    mutated = tmp_path / "mutated.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SuccessorMicroOverfitError, match="identity"):
        capacity_v2.load_uniform_capacity_v2_contract(mutated)


def test_serialized_source_tree_hash_is_content_addressed(tmp_path: Path) -> None:
    for subtree in capacity_v2.SERIALIZED_SOURCE_SUBTREES:
        source = tmp_path / subtree
        source.mkdir(parents=True, exist_ok=True)
        (source / "identity.txt").write_text(subtree, encoding="utf-8")
    before = capacity_v2.serialized_source_tree_sha256(tmp_path)
    (tmp_path / "src" / "identity.txt").write_text("changed", encoding="utf-8")
    after = capacity_v2.serialized_source_tree_sha256(tmp_path)
    assert before != after


def test_checkpoint_selection_uses_exact_lexicographic_order() -> None:
    incumbent = {
        "update": 3,
        "minimum_teacher_successor_probability": 0.7,
        "mean_canonical_successor_nll": 0.2,
    }
    assert capacity_v2.checkpoint_is_better(
        {
            "update": 4,
            "minimum_teacher_successor_probability": 0.71,
            "mean_canonical_successor_nll": 9.0,
        },
        incumbent,
    )
    assert capacity_v2.checkpoint_is_better(
        {
            "update": 4,
            "minimum_teacher_successor_probability": 0.7,
            "mean_canonical_successor_nll": 0.19,
        },
        incumbent,
    )
    assert not capacity_v2.checkpoint_is_better(
        {
            "update": 4,
            "minimum_teacher_successor_probability": 0.7,
            "mean_canonical_successor_nll": 0.2,
        },
        incumbent,
    )
    assert capacity_v2.checkpoint_is_better(
        {
            "update": 9,
            "minimum_teacher_successor_probability": 0.0,
            "minimum_teacher_successor_log_probability": -100.0,
            "mean_canonical_successor_nll": 9.0,
        },
        {
            "update": 1,
            "minimum_teacher_successor_probability": 0.0,
            "minimum_teacher_successor_log_probability": -101.0,
            "mean_canonical_successor_nll": 0.1,
        },
    )


def test_probability_floor_proves_all_v1_probability_nll_top1_checks() -> None:
    probabilities = torch.tensor([0.8, 0.9, 0.95], dtype=torch.float64)
    checks, metrics = capacity_v2.probability_implication_checks(
        probabilities,
        thresholds=capacity_v2.EXPECTED_V1_THRESHOLDS,
    )
    assert all(checks.values())
    assert metrics["minimum_teacher_successor_probability"] == 0.8
    failed, _ = capacity_v2.probability_implication_checks(
        torch.tensor([0.79, 0.99], dtype=torch.float64),
        thresholds=capacity_v2.EXPECTED_V1_THRESHOLDS,
    )
    assert failed["minimum_every_example_teacher_successor_probability"] is False


def test_v2_optimizer_stops_only_after_probability_and_gradient_gates(
    monkeypatch,
) -> None:
    monkeypatch.setattr(capacity_v2, "forward_teacher_successor_batch", _toy_forward)
    monkeypatch.setattr(capacity_v2, "successor_panel_metrics", _toy_metrics)
    model = _ToyModel()
    hazard_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("total_hazard_head.")
    }
    report = capacity_v2.train_uniform_successor_capacity_v2(
        model,
        _toy_panel(),
        maximum_updates=10,
        learning_rate=0.4,
        weight_decay=0.0,
        scope="all",
        seed=7,
        gradient_clip_norm=10.0,
        thresholds=capacity_v2.EXPECTED_V1_THRESHOLDS,
    )

    assert report["early_stopped"] is True
    assert 1 <= report["completed_full_panel_updates"] < 10
    assert report["trajectory_state_count"] == report["completed_full_panel_updates"] + 1
    assert report["trajectory"][0]["all_early_stop_checks_pass"] is False
    assert report["trajectory"][-1]["all_early_stop_checks_pass"] is True
    assert report["trajectory"][-1]["optimizer_update_executed_after_evaluation"] is False
    assert report["selected_update"] == report["terminal_update"]
    assert report["hazard_freeze"]["bit_identity_pass"] is True
    assert report["hazard_freeze"]["optimizer_included"] is False
    assert report["all_threshold_checks_pass"] is True
    for name, parameter in model.named_parameters():
        if name in hazard_before:
            assert parameter.requires_grad is False
            assert torch.equal(parameter.detach(), hazard_before[name])
    capacity_v2.validate_v2_training_report(
        report,
        family="cycle_attach",
        maximum_updates=10,
        learning_rate=0.4,
        weight_decay=0.0,
        gradient_clip_norm=10.0,
        seed=7,
    )


def test_v2_report_rejects_continuing_after_proven_early_stop(monkeypatch) -> None:
    monkeypatch.setattr(capacity_v2, "forward_teacher_successor_batch", _toy_forward)
    monkeypatch.setattr(capacity_v2, "successor_panel_metrics", _toy_metrics)
    report = capacity_v2.train_uniform_successor_capacity_v2(
        _ToyModel(),
        _toy_panel(),
        maximum_updates=10,
        learning_rate=0.4,
        weight_decay=0.0,
        scope="all",
        seed=7,
        gradient_clip_norm=10.0,
        thresholds=capacity_v2.EXPECTED_V1_THRESHOLDS,
    )
    mutated = copy.deepcopy(report)
    mutated["trajectory"][0]["all_early_stop_checks_pass"] = True
    mutated["trajectory_sha256"] = capacity_v2.stable_sha256(mutated["trajectory"])
    with pytest.raises(SuccessorMicroOverfitError, match="proof|continued"):
        capacity_v2.validate_v2_training_report(
            mutated,
            family="cycle_attach",
            maximum_updates=10,
            learning_rate=0.4,
            weight_decay=0.0,
            gradient_clip_norm=10.0,
            seed=7,
        )


def test_v2_optimizer_uses_maximum_stop_reason_when_floor_is_not_reached(
    monkeypatch,
) -> None:
    monkeypatch.setattr(capacity_v2, "forward_teacher_successor_batch", _toy_forward)
    monkeypatch.setattr(capacity_v2, "successor_panel_metrics", _toy_metrics)
    report = capacity_v2.train_uniform_successor_capacity_v2(
        _ToyModel(),
        _toy_panel(),
        maximum_updates=1,
        learning_rate=1e-6,
        weight_decay=0.0,
        scope="all",
        seed=7,
        gradient_clip_norm=10.0,
        thresholds=capacity_v2.EXPECTED_V1_THRESHOLDS,
    )
    assert report["early_stopped"] is False
    assert report["stop_reason"] == "MAXIMUM_FULL_PANEL_UPDATES_REACHED"
    assert report["terminal_update"] == 1
    capacity_v2.validate_v2_training_report(
        report,
        family="cycle_attach",
        maximum_updates=1,
        learning_rate=1e-6,
        weight_decay=0.0,
        gradient_clip_norm=10.0,
        seed=7,
    )


def test_v2_optimizer_runs_real_successor_bridge_and_freezes_hazard() -> None:
    torch.manual_seed(31)
    model = FactorizedTraceletRateModel(
        build_typed_ring_catalog(()),
        hidden_dim=12,
        message_passing_steps=1,
        enable_cyclic_graft=True,
        enable_heteroatom_scan=True,
        enable_cycle_ops=True,
        enable_ring_grow_macro=False,
        atom_vocabulary=ORGANIC_VOCABULARY,
    )
    records, attempted = build_cycle_op_records(
        ("C1CCCCC1",),
        n_slots=12,
        seed=4,
        max_bonds_per_molecule=2,
    )
    assert attempted > 0
    examples = examples_from_traces(
        (record.path.trace for record in records),
        families=("cycle_attach",),
        maximum_per_family=1,
        unique_source_molecules=True,
    )
    assert len(examples) == 1
    panel = prepare_successor_panel(model, examples)
    hazard_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("total_hazard_head.")
    }
    report = capacity_v2.train_uniform_successor_capacity_v2(
        model,
        panel,
        maximum_updates=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        scope="all",
        seed=7,
        gradient_clip_norm=10.0,
        thresholds=capacity_v2.EXPECTED_V1_THRESHOLDS,
    )
    assert report["completed_full_panel_updates"] == 1
    assert report["trajectory_state_count"] == 2
    assert report["hazard_freeze"]["bit_identity_pass"] is True
    for name, parameter in model.named_parameters():
        if name in hazard_before:
            assert torch.equal(parameter.detach(), hazard_before[name])


def test_modal_v2_family_selection_and_receipt_are_non_authorizing() -> None:
    assert modal_runner.REMOTE_PROJECT_ROOT == modal_runner.REMOTE_ROOT
    assert modal_runner._selected_families("", all_families=True) == modal_runner.FAMILIES
    assert modal_runner._selected_families(
        "cycle_attach,bond_reroute",
        all_families=False,
    ) == ("cycle_attach", "bond_reroute")
    receipt = modal_runner._build_receipt(
        run_label="uniform-v2-test",
        source_commit="a" * 40,
        source_tree_sha256="d" * 64,
        active8_inventory="/artifacts/inventory.json",
        active8_inventory_file_sha256="b" * 64,
        requested_families=("cycle_attach",),
        stage="GPU_RESULTS",
        cpu_preflight_bindings=({"family": "cycle_attach"},),
        results=({"family": "cycle_attach", "result_sha256": "c" * 64},),
        failures=(),
    )
    assert receipt["status"] == modal_runner.COMPLETE_RECEIPT_STATUS
    assert receipt["training_authorized"] is False
    assert receipt["bounded_p50_authorized"] is False
    assert receipt["gate_decision"] is None
    assert receipt["serialized_source_tree_sha256"] == "d" * 64


def test_outer_result_validator_binds_law_revision_and_selected_state(monkeypatch) -> None:
    monkeypatch.setattr(capacity_v2, "forward_teacher_successor_batch", _toy_forward)
    monkeypatch.setattr(capacity_v2, "successor_panel_metrics", _toy_metrics)
    contract = capacity_v2.load_uniform_capacity_v2_contract(CONTRACT)
    optimization = contract["optimization_law"]
    report = capacity_v2.train_uniform_successor_capacity_v2(
        _ToyModel(),
        _toy_panel(),
        maximum_updates=optimization["maximum_full_panel_updates"],
        learning_rate=optimization["learning_rate"],
        weight_decay=optimization["weight_decay"],
        scope=optimization["scope"],
        seed=optimization["seed"],
        gradient_clip_norm=optimization["gradient_clip_norm"],
        thresholds=capacity_v2.EXPECTED_V1_THRESHOLDS,
    )
    sealed = _sealed_outer_result(report, contract)
    assert (
        v2_runner.validate_uniform_capacity_v2_result(
            sealed,
            contract=contract,
        )
        == sealed
    )

    wrong_lr = copy.deepcopy(sealed)
    wrong_lr["training_report"]["learning_rate"] = 0.5
    wrong_lr["result_sha256"] = capacity_v2.stable_sha256(
        {key: value for key, value in wrong_lr.items() if key != "result_sha256"}
    )
    with pytest.raises(v2_runner.EditingT1RuntimeError, match="training report"):
        v2_runner.validate_uniform_capacity_v2_result(wrong_lr, contract=contract)

    wrong_binding = copy.deepcopy(sealed)
    wrong_binding["source_commit_binding"] = "UNVERIFIED"
    wrong_binding["result_sha256"] = capacity_v2.stable_sha256(
        {key: value for key, value in wrong_binding.items() if key != "result_sha256"}
    )
    with pytest.raises(v2_runner.EditingT1RuntimeError, match="identity"):
        v2_runner.validate_uniform_capacity_v2_result(wrong_binding, contract=contract)
