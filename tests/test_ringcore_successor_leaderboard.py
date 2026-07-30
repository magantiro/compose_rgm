from __future__ import annotations

import copy
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from compose_v4.experiments.production_successor_kernel import (
    FactorizedMarkedLaw,
    SuccessorKernelDiagnostics,
    SuccessorKernelResult,
)
from compose_v4.experiments.ringcore_successor_leaderboard import (
    CURRENT_STATE_SOURCE,
    PRIMARY_METRIC,
    SECONDARY_METRIC,
    SnapshotEvaluationSpec,
    SuccessorLeaderboardError,
    TeacherSuccessor,
    aggregate_validation_rows,
    encode_semantic_cell,
    inventory_self_hash,
    load_current_snapshot_model,
    load_json_object,
    prepare_snapshot_specs,
    readiness_summary,
    score_teacher_successor,
    validate_current_snapshot_payload,
    validate_inventory,
    validate_leaderboard_config,
)
from compose_v4.experiments.successor_kernel import (
    CanonicalSuccessor,
    KernelIdentity,
    SuccessorBatch,
)
from compose_v4.model.factorized_tracelet_rate_model import MARK_RULE_NAMES

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "ringcore_v1_successor_leaderboard_v1.json"
INVENTORY_PATH = (
    ROOT / "diagnostics" / "coherence" / "ringcore_v1_scientific_a7546e2_frozen_inventory.json"
)


@pytest.fixture
def config():
    return load_json_object(CONFIG_PATH)


@pytest.fixture
def inventory():
    return load_json_object(INVENTORY_PATH)


def test_protocol_prepares_all_32_current_validation_snapshots(config, inventory):
    validate_leaderboard_config(config)
    specs = prepare_snapshot_specs(config, inventory)

    assert len(specs) == 32
    assert [spec.step for spec in specs] == list(range(500, 16001, 500))
    assert {spec.state_source for spec in specs} == {CURRENT_STATE_SOURCE}
    assert {spec.partition for spec in specs} == {"validation"}
    assert len({spec.sha256 for spec in specs}) == 32


def test_exact_inventory_self_hash_is_locally_enforceable(config, inventory):
    assert inventory_self_hash(inventory) == inventory["inventory_sha256"]
    validate_inventory(config, inventory, require_exact_self_hash=True)

    projection = copy.deepcopy(inventory)
    projection.pop("checkpoint_metadata")
    for row in projection["snapshots"]:
        row.pop("metadata")
    with pytest.raises(SuccessorLeaderboardError, match="compact projection"):
        validate_inventory(config, projection, require_exact_self_hash=True)
    with pytest.raises(SuccessorLeaderboardError, match="compact projection"):
        prepare_snapshot_specs(config, projection)


def test_stale_family_balanced_secondary_is_refused(config):
    stale = copy.deepcopy(config)
    stale["reported_metrics"]["metric_ids"]["secondary"] = "balanced_family_canonical_successor_nll"
    with pytest.raises(SuccessorLeaderboardError, match="semantic cells"):
        validate_leaderboard_config(stale)

    promoted = copy.deepcopy(config)
    promoted["selection_rule"]["metric_order"] = [
        SECONDARY_METRIC,
        PRIMARY_METRIC,
    ]
    with pytest.raises(SuccessorLeaderboardError, match="metric order"):
        validate_leaderboard_config(promoted)


def test_semantic_cell_encoder_is_complete_order_stable_and_versioned():
    labels = {
        "capability_regime": "topology_change",
        "evidence_origin": "real_analogue",
        "path_scale": "short",
        "cardinality_topology_delta": "cycle_rank_minus_one",
        "chemistry_charge_stratum": "broad_organic_neutral",
        "split_unit": "held_scaffold",
    }
    reversed_labels = dict(reversed(tuple(labels.items())))
    assert encode_semantic_cell(labels) == encode_semantic_cell(reversed_labels)
    assert encode_semantic_cell(labels).startswith("joint_semantic_axes_v1:")
    incomplete = dict(labels)
    incomplete.pop("path_scale")
    with pytest.raises(SuccessorLeaderboardError, match="axes mismatch"):
        encode_semantic_cell(incomplete)


def test_test_partition_and_best_state_are_refused(config):
    bad_partition = copy.deepcopy(config)
    bad_partition["validation_data"]["partition"] = "test"
    with pytest.raises(SuccessorLeaderboardError, match="validation"):
        validate_leaderboard_config(bad_partition)

    bad_state = copy.deepcopy(config)
    bad_state["run"]["snapshot_model_state"] = "best_state_dict"
    with pytest.raises(SuccessorLeaderboardError, match="current_state_dict"):
        validate_leaderboard_config(bad_state)


def test_validation_stream_and_semantic_census_are_fully_bound(config):
    validate_leaderboard_config(config)

    wrong_bins = copy.deepcopy(config)
    wrong_bins["validation_data"]["record_sampling"]["curriculum_bin_edges"] = [4, 8, 12]
    with pytest.raises(
        SuccessorLeaderboardError,
        match="record-sampling contract mismatch",
    ):
        validate_leaderboard_config(wrong_bins)

    missing_sampler_hash = copy.deepcopy(config)
    missing_sampler_hash["validation_data"]["record_sampling"]["implementation_sha256"] = None
    with pytest.raises(
        SuccessorLeaderboardError,
        match="implementation SHA-256",
    ):
        validate_leaderboard_config(missing_sampler_hash)

    stale_census = copy.deepcopy(config)
    stale_census["panels"]["semantic_cells"]["full_validation_census"]["nonempty_cells"] = 111
    with pytest.raises(
        SuccessorLeaderboardError,
        match="census counts drifted",
    ):
        validate_leaderboard_config(stale_census)


def test_inventory_step_or_hash_tampering_is_refused(config, inventory):
    wrong_step = copy.deepcopy(inventory)
    wrong_step["snapshots"][3]["completed_steps"] = 2001
    with pytest.raises(SuccessorLeaderboardError, match="payload step"):
        validate_inventory(config, wrong_step, require_exact_self_hash=False)

    wrong_hash = copy.deepcopy(inventory)
    wrong_hash["snapshots"][0]["sha256"] = "not-a-hash"
    with pytest.raises(SuccessorLeaderboardError, match="full SHA-256"):
        validate_inventory(config, wrong_hash, require_exact_self_hash=False)


def test_payload_validator_returns_current_not_embedded_best():
    current = {"weight": torch.tensor([2.0])}
    best = {"weight": torch.tensor([1.0])}
    spec = SnapshotEvaluationSpec(
        step=500,
        name="checkpoint.step500.pt",
        sha256="a" * 64,
        bytes=1,
    )
    payload = {
        "checkpoint_kind": "exact_training_recovery",
        "completed_steps": 500,
        "current_state_dict": current,
        "best_state_dict": best,
        "history": [{"step": 500.0}],
        "provenance_sha256": "b" * 64,
    }

    selected = validate_current_snapshot_payload(
        payload,
        spec,
        expected_run_identity_sha256="b" * 64,
    )

    assert selected is current
    assert selected is not best


def test_current_snapshot_loader_strictly_installs_current_weights(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "checkpoint.step500.pt"
    current = {"weight": torch.tensor([2.0])}
    best = {"weight": torch.tensor([1.0])}
    torch.save(
        {
            "checkpoint_kind": "exact_training_recovery",
            "completed_steps": 500,
            "current_state_dict": current,
            "best_state_dict": best,
            "history": [{"step": 500.0}],
            "provenance_sha256": "b" * 64,
        },
        path,
    )
    from compose_v4.experiments import ringcore_successor_leaderboard as module
    from compose_v4.experiments.checkpoint_evaluator import file_sha256

    spec = SnapshotEvaluationSpec(
        step=500,
        name=path.name,
        sha256=file_sha256(path),
        bytes=path.stat().st_size,
    )

    class FakeModel:
        loaded = None
        strict = None
        evaluated = False

        def load_state_dict(self, state, *, strict):
            self.loaded = state
            self.strict = strict

        def eval(self):
            self.evaluated = True
            return self

    model = FakeModel()
    monkeypatch.setattr(
        module,
        "_load_production_checkpoint",
        lambda _path, *, expected_scope_hash: (model, {"source": "embedded_best"}),
    )
    monkeypatch.setattr(
        module,
        "prepare_snapshot_specs",
        lambda _config, _inventory: (spec,),
    )

    loaded, audit = load_current_snapshot_model(
        path,
        spec,
        config={},
        inventory={
            "checkpoint_metadata": {"corpus_scope_hash": "scope"},
            "manifest_summary": {"run_identity_sha256": "b" * 64},
        },
    )

    assert loaded is model
    assert model.loaded is current or torch.equal(model.loaded["weight"], current["weight"])
    assert not torch.equal(model.loaded["weight"], best["weight"])
    assert model.strict is True
    assert model.evaluated is True
    assert audit.state_source == CURRENT_STATE_SOURCE
    assert audit.step == 500


def test_current_snapshot_loader_rejects_a_free_floating_spec(monkeypatch):
    from compose_v4.experiments import ringcore_successor_leaderboard as module

    frozen = SnapshotEvaluationSpec(
        step=500,
        name="checkpoint.step500.pt",
        sha256="a" * 64,
        bytes=100,
    )
    forged = replace(frozen, sha256="b" * 64)
    monkeypatch.setattr(
        module,
        "prepare_snapshot_specs",
        lambda _config, _inventory: (frozen,),
    )
    with pytest.raises(SuccessorLeaderboardError, match="inventory-bound"):
        load_current_snapshot_model(
            "/does/not/matter/checkpoint.step500.pt",
            forged,
            config={},
            inventory={},
        )


def _kernel_result() -> SuccessorKernelResult:
    family_logp = tuple(
        math.log(0.4) if name == "cycle_attach" else math.log(0.6 / (len(MARK_RULE_NAMES) - 1))
        for name in MARK_RULE_NAMES
    )
    law = FactorizedMarkedLaw(
        source_key="source",
        marks=(),
        total_hazard=2.0,
        family_log_probabilities=family_logp,
        enabled_families=tuple(MARK_RULE_NAMES),
    )
    batch = SuccessorBatch(
        source_key="source",
        successors=(
            CanonicalSuccessor(
                key="teacher",
                state=None,  # type: ignore[arg-type]
                probability=0.6,
                alias_count=3,
            ),
            CanonicalSuccessor(
                key="other",
                state=None,  # type: ignore[arg-type]
                probability=0.4,
                alias_count=1,
            ),
        ),
        identity=KernelIdentity("test"),
        virtual_mass=0.1,
    )
    diagnostics = SuccessorKernelDiagnostics(
        raw_mark_count=5,
        productive_mark_count=4,
        virtual_self_mark_count=1,
        canonical_successor_count=2,
        raw_productive_mass=0.9,
        virtual_self_mass=0.1,
        alias_multiplicities=(3, 1),
        marks_by_family=dict.fromkeys(MARK_RULE_NAMES, 0),
        productive_marks_by_family=dict.fromkeys(MARK_RULE_NAMES, 0),
    )
    return SuccessorKernelResult(batch, diagnostics, law)


def _teacher(*, draw: int = 0, cell: str = "cell_a", weight: float = 1.0):
    return TeacherSuccessor(
        panel_id="production_law",
        draw_index=draw,
        partition="validation",
        family="cycle_attach",
        semantic_cell_id=cell,
        teacher_successor_key="teacher",
        importance_weight=weight,
        teacher_mark_log_probability=math.log(0.2),
    )


def test_teacher_scoring_uses_productive_canonical_probability_and_diagnostics():
    row = score_teacher_successor(_kernel_result(), _teacher())

    assert row.canonical_successor_probability == pytest.approx(0.6)
    assert row.canonical_successor_nll == pytest.approx(-math.log(0.6))
    assert row.canonical_successor_rank == 1
    assert row.uniform_canonical_successor_nll == pytest.approx(math.log(2))
    assert row.learned_minus_uniform_log_likelihood == pytest.approx(math.log(1.2))
    assert row.training_target_nll == pytest.approx(-math.log(0.2))
    assert row.selected_mark_minus_successor_nll_gap == pytest.approx(math.log(3))
    assert row.teacher_successor_alias_multiplicity == 3
    assert row.maximum_alias_multiplicity == 3
    assert row.raw_legal_mark_count == 5
    assert row.productive_mark_count == 4
    assert row.canonical_successor_count == 2
    assert row.virtual_self_mass == pytest.approx(0.1)
    assert row.productive_mass == pytest.approx(0.9)
    assert row.teacher_family_probability == pytest.approx(0.4)
    assert row.family_choice_top1 is True


def test_teacher_missing_after_grouping_fails_instead_of_skipping():
    teacher = replace(_teacher(), teacher_successor_key="absent")
    with pytest.raises(SuccessorLeaderboardError, match="exactly once"):
        score_teacher_successor(_kernel_result(), teacher)


def test_aggregation_preserves_production_weights_but_balances_semantic_cells():
    base = score_teacher_successor(_kernel_result(), _teacher())
    # cell A has two high-weight/easy observations; cell B has one hard observation.
    rows = (
        replace(
            base,
            draw_index=0,
            semantic_cell_id="cell_a",
            importance_weight=3.0,
            canonical_successor_nll=1.0,
        ),
        replace(
            base,
            draw_index=1,
            semantic_cell_id="cell_a",
            importance_weight=1.0,
            canonical_successor_nll=3.0,
        ),
        replace(
            base,
            draw_index=2,
            semantic_cell_id="cell_b",
            importance_weight=1.0,
            canonical_successor_nll=5.0,
        ),
    )

    metrics = aggregate_validation_rows(
        rows,
        required_semantic_cells=("cell_a", "cell_b"),
        expected_nonterminal_draw_indices=(0, 1, 2),
    )

    assert metrics[PRIMARY_METRIC] == pytest.approx((3 * 1 + 1 * 3 + 1 * 5) / 5)
    assert metrics["per_semantic_cell"]["cell_a"]["canonical_successor_nll"] == pytest.approx(1.5)
    assert metrics[SECONDARY_METRIC] == pytest.approx((1.5 + 5.0) / 2)
    assert metrics["fraction_teacher_successors_aliased"] == 1.0


def test_aggregation_refuses_missing_cells_duplicates_and_test_rows():
    row = score_teacher_successor(_kernel_result(), _teacher())
    with pytest.raises(SuccessorLeaderboardError, match="census mismatch"):
        aggregate_validation_rows(
            (row,),
            required_semantic_cells=("cell_a", "cell_b"),
            expected_nonterminal_draw_indices=(0,),
        )
    with pytest.raises(SuccessorLeaderboardError, match="not unique"):
        aggregate_validation_rows(
            (row, row),
            required_semantic_cells=("cell_a",),
            expected_nonterminal_draw_indices=(0,),
        )
    with pytest.raises(SuccessorLeaderboardError, match="non-validation"):
        aggregate_validation_rows(
            (replace(row, partition="test"),),
            required_semantic_cells=("cell_a",),
            expected_nonterminal_draw_indices=(0,),
        )
    with pytest.raises(SuccessorLeaderboardError, match="panel census mismatch"):
        aggregate_validation_rows(
            (row,),
            required_semantic_cells=("cell_a",),
            expected_nonterminal_draw_indices=(0, 1),
        )
    with pytest.raises(SuccessorLeaderboardError, match="different panel"):
        aggregate_validation_rows(
            (replace(row, panel_id="forensics"),),
            required_semantic_cells=("cell_a",),
            expected_nonterminal_draw_indices=(0,),
        )


def test_local_readiness_report_never_claims_evaluation_or_selection(config, inventory):
    report = readiness_summary(config, inventory)
    stored = load_json_object(
        ROOT
        / "diagnostics"
        / "coherence"
        / "ringcore_v1_successor_leaderboard_readiness_2026-07-30.json"
    )

    assert report["ready_to_execute"] is False
    assert report["snapshot_specs_prepared"] == 32
    assert report["selection_partition"] == "validation"
    assert report["snapshot_state_source"] == CURRENT_STATE_SOURCE
    assert "no checkpoint evaluated" in report["explicit_non_actions"]
    assert "no checkpoint selected" in report["explicit_non_actions"]
    assert report["selection_safety"]["automatic_ranking_code_present"] is False
    assert report["selection_safety"]["automatic_selection_code_present"] is False
    assert report["selection_safety"]["test_partition_input_accepted"] is False
    assert report["blockers"]
    for field in (
        "ready_to_execute",
        "snapshot_specs_prepared",
        "snapshot_steps",
        "snapshot_state_source",
        "selection_partition",
        "primary_metric",
        "secondary_metric",
        "selection_safety",
        "blockers",
        "explicit_non_actions",
    ):
        assert stored[field] == report[field]


def test_legacy_diagnostic_cannot_emit_a_ranking_or_best_checkpoint():
    source = (ROOT / "scripts" / "ring_core_checkpoint_selection.py").read_text()
    assert '"ranking_by_primary_metric"' not in source
    assert '"best_by_primary_metric"' not in source
    assert '"selection_performed"] = False' in source
    assert 'if mode != "current"' in source
    assert '"state_source": "current_state_dict"' in source
