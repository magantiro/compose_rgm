from __future__ import annotations

import copy
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from compose_v4.experiments.ringcore_successor_leaderboard import (
    SEMANTIC_CELL_AXES,
    encode_semantic_cell,
    load_json_object,
)
from compose_v4.experiments.ringcore_validation_panel import (
    FAMILY_FORENSICS_PANEL_ID,
    STATE_DIGEST_SCHEMA,
    ValidationPanelError,
    bind_panel_metric_vector,
    family_forensics_expansion_decision,
    paired_weighted_bootstrap,
    panel_artifact_self_hash,
    production_panel_expansion_decision,
    seal_validation_panel,
    validate_panel_metric_vector,
    validate_validation_panel_artifact,
    write_validation_panel,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "ringcore_v1_successor_leaderboard_v1.json"
INVENTORY_PATH = (
    ROOT
    / "diagnostics"
    / "coherence"
    / "ringcore_v1_scientific_a7546e2_frozen_inventory.json"
)


@pytest.fixture
def inventory():
    return load_json_object(INVENTORY_PATH)


def _config(*, production_draws=(2, 4), forensic_minima=(1, 2)):
    config = load_json_object(CONFIG_PATH)
    config["panels"]["production_law"]["initial_draws"] = production_draws[0]
    config["panels"]["production_law"]["expanded_draws"] = production_draws[1]
    config["panels"]["family_forensics"]["active_families"] = [
        "atom_insert",
        "cycle_attach",
    ]
    config["panels"]["family_forensics"][
        "initial_minimum_nonterminal_examples_per_family"
    ] = forensic_minima[0]
    config["panels"]["family_forensics"][
        "expanded_minimum_nonterminal_examples_per_ambiguous_family"
    ] = forensic_minima[1]
    config["panels"]["family_forensics"]["maximum_stream_draws"] = 16
    return config


def _axes(*, regime="local_edit"):
    return {
        axis: {
            "capability_regime": regime,
            "evidence_origin": "real_analogue",
            "path_scale": "short",
            "cardinality_topology_delta": "cardinality_same",
            "chemistry_charge_stratum": "broad_organic_neutral",
            "split_unit": "held_scaffold",
        }[axis]
        for axis in SEMANTIC_CELL_AXES
    }


def _row(
    row_index: int,
    *,
    stream_draw_index: int | None = None,
    family: str = "atom_insert",
    terminal: bool = False,
):
    progress = 1 if terminal else 0
    axes = None if terminal else _axes(regime=family)
    return {
        "row_index": row_index,
        "stream_draw_index": (
            row_index if stream_draw_index is None else stream_draw_index
        ),
        "partition": "validation",
        "layer": "general_corruption",
        "record_key": f"record-{row_index}",
        "path_length": 1,
        "progress_index": progress,
        "time": 0.37,
        "importance_weight": 1.0 + row_index,
        "terminal": terminal,
        "teacher_family": None if terminal else family,
        "teacher_rule_name": None if terminal else family,
        "teacher_successor_key": None if terminal else f"successor-{row_index}",
        "teacher_action_sha256": None if terminal else "a" * 64,
        "semantic_axis_values": axes,
        "semantic_cell_id": None if terminal else encode_semantic_cell(axes),
        "source_state_key": f"source-{row_index}",
        "source_state_sha256": "b" * 64,
        "exact_state_ref": {
            "shard_name": "corruption/validation/shard_0000.jsonl.gz",
            "shard_sha256": "c" * 64,
            "record_index": row_index,
            "progress_index": progress,
            "state_digest_schema": STATE_DIGEST_SCHEMA,
        },
    }


def _rehash(panel):
    panel = copy.deepcopy(panel)
    for row in panel["rows"]:
        raw = dict(row)
        raw.pop("row_sha256", None)
        from compose_v4.experiments.ringcore_successor_leaderboard import (
            stable_json_sha256,
        )

        row["row_sha256"] = stable_json_sha256(raw)
    panel["artifact_sha256"] = panel_artifact_self_hash(panel)
    return panel


def _metric_pair(
    panel,
    config,
    inventory,
    left,
    right,
    *,
    family_filter=None,
    metric_id="canonical_successor_nll",
    right_source_kind="uniform_canonical_successor",
):
    return (
        bind_panel_metric_vector(
            left,
            metric_id=metric_id,
            source_kind="snapshot_current_state",
            panel=panel,
            config=config,
            inventory=inventory,
            family_filter=family_filter,
            snapshot_step=500,
        ),
        bind_panel_metric_vector(
            right,
            metric_id=metric_id,
            source_kind=right_source_kind,
            panel=panel,
            config=config,
            inventory=inventory,
            family_filter=family_filter,
            snapshot_step=1000 if right_source_kind == "snapshot_current_state" else None,
        ),
    )


def test_production_panel_is_self_hashed_exact_validation_prefix(inventory, tmp_path):
    config = _config()
    panel = seal_validation_panel(
        panel_kind="production_law",
        rows=(_row(0), _row(1, terminal=True)),
        draw_contract={
            "stage": "initial",
            "requested_stream_draws": 2,
            "stream_draws_consumed": 2,
        },
        config=config,
        inventory=inventory,
    )

    validate_validation_panel_artifact(panel, config=config, inventory=inventory)
    assert panel["partition"] == "validation"
    assert panel["panel_kind"] == "production_law"
    assert panel["draw_contract"]["nonterminal_draw_indices"] == [0]
    assert panel["draw_contract"]["terminal_row_count"] == 1
    assert panel["artifact_sha256"] == panel_artifact_self_hash(panel)

    path = write_validation_panel(
        panel,
        tmp_path / "panel.json",
        config=config,
        inventory=inventory,
    )
    loaded = load_json_object(path)
    validate_validation_panel_artifact(loaded, config=config, inventory=inventory)
    assert (
        write_validation_panel(
            panel,
            path,
            config=config,
            inventory=inventory,
        )
        == path
    )
    changed = copy.deepcopy(panel)
    changed["rows"][0]["importance_weight"] = 9.0
    changed = _rehash(changed)
    with pytest.raises(FileExistsError):
        write_validation_panel(
            changed,
            path,
            config=config,
            inventory=inventory,
        )


def test_panel_rejects_test_rows_tampering_missing_draws_and_scores(inventory):
    config = _config()
    panel = seal_validation_panel(
        panel_kind="production_law",
        rows=(_row(0), _row(1)),
        draw_contract={
            "stage": "initial",
            "requested_stream_draws": 2,
            "stream_draws_consumed": 2,
        },
        config=config,
        inventory=inventory,
    )

    test_row = copy.deepcopy(panel)
    test_row["rows"][0]["partition"] = "test"
    test_row = _rehash(test_row)
    with pytest.raises(ValidationPanelError, match="validation only"):
        validate_validation_panel_artifact(test_row, config=config, inventory=inventory)

    missing = copy.deepcopy(panel)
    missing["rows"].pop()
    missing = _rehash(missing)
    with pytest.raises(ValidationPanelError, match="draw census mismatch|exact deterministic"):
        validate_validation_panel_artifact(missing, config=config, inventory=inventory)

    tampered = copy.deepcopy(panel)
    tampered["rows"][0]["importance_weight"] = 99.0
    tampered["artifact_sha256"] = panel_artifact_self_hash(tampered)
    with pytest.raises(ValidationPanelError, match="row self-hash"):
        validate_validation_panel_artifact(
            tampered,
            config=config,
            inventory=inventory,
        )

    scored = copy.deepcopy(panel)
    scored["checkpoint_scores"] = {"step500": 1.0}
    scored["artifact_sha256"] = panel_artifact_self_hash(scored)
    with pytest.raises(ValidationPanelError, match="top-level schema"):
        validate_validation_panel_artifact(scored, config=config, inventory=inventory)

    wrong_scalar_type = copy.deepcopy(panel)
    wrong_scalar_type["rows"][0]["progress_index"] = "0"
    wrong_scalar_type = _rehash(wrong_scalar_type)
    with pytest.raises(ValidationPanelError, match="progress index.*integer"):
        validate_validation_panel_artifact(
            wrong_scalar_type,
            config=config,
            inventory=inventory,
        )

    truthy_terminal = copy.deepcopy(panel)
    truthy_terminal["rows"][0]["terminal"] = 0
    truthy_terminal = _rehash(truthy_terminal)
    with pytest.raises(ValidationPanelError, match="JSON boolean"):
        validate_validation_panel_artifact(
            truthy_terminal,
            config=config,
            inventory=inventory,
        )


def test_panel_requires_exact_corpus_provenance_and_protocol_hash(inventory):
    config = _config()
    panel = seal_validation_panel(
        panel_kind="production_law",
        rows=(_row(0), _row(1)),
        draw_contract={
            "stage": "initial",
            "requested_stream_draws": 2,
            "stream_draws_consumed": 2,
        },
        config=config,
        inventory=inventory,
    )
    wrong = copy.deepcopy(panel)
    wrong["provenance"]["operator_registry_hash"] = "wrong"
    wrong["artifact_sha256"] = panel_artifact_self_hash(wrong)
    with pytest.raises(ValidationPanelError, match="corpus provenance"):
        validate_validation_panel_artifact(wrong, config=config, inventory=inventory)

    changed_protocol = copy.deepcopy(config)
    changed_protocol["panels"]["production_law"]["seed"] += 1
    with pytest.raises(ValidationPanelError, match="protocol hash"):
        validate_validation_panel_artifact(
            panel,
            config=changed_protocol,
            inventory=inventory,
        )


def test_family_forensics_is_sparse_nonterminal_and_exact_per_family(inventory):
    config = _config()
    panel = seal_validation_panel(
        panel_kind=FAMILY_FORENSICS_PANEL_ID,
        rows=(
            _row(0, stream_draw_index=1, family="atom_insert"),
            _row(1, stream_draw_index=4, family="cycle_attach"),
        ),
        draw_contract={
            "requested_nonterminal_examples_by_family": {
                "atom_insert": 1,
                "cycle_attach": 1,
            },
            "stream_draws_consumed": 5,
        },
        config=config,
        inventory=inventory,
    )

    validate_validation_panel_artifact(panel, config=config, inventory=inventory)
    assert panel["draw_contract"]["family_nonterminal_census"] == {
        "atom_insert": 1,
        "cycle_attach": 1,
    }
    assert panel["draw_contract"]["terminal_row_count"] == 0

    wrong_targets = copy.deepcopy(panel)
    wrong_targets["draw_contract"]["requested_nonterminal_examples_by_family"][
        "cycle_attach"
    ] = 2
    wrong_targets["artifact_sha256"] = panel_artifact_self_hash(wrong_targets)
    with pytest.raises(ValidationPanelError, match="exact family targets"):
        validate_validation_panel_artifact(
            wrong_targets,
            config=config,
            inventory=inventory,
        )


def test_paired_bootstrap_is_panel_bound_deterministic_and_nonselecting(inventory):
    config = _config(production_draws=(3, 6))
    panel = seal_validation_panel(
        panel_kind="production_law",
        rows=(_row(0), _row(1), _row(2)),
        draw_contract={
            "stage": "initial",
            "requested_stream_draws": 3,
            "stream_draws_consumed": 3,
        },
        config=config,
        inventory=inventory,
    )
    left = {0: 1.0, 1: 3.0, 2: 2.0}
    right = {0: 2.0, 1: 2.0, 2: 2.0}
    left_vector, right_vector = _metric_pair(
        panel,
        config,
        inventory,
        left,
        right,
    )

    first = paired_weighted_bootstrap(
        left_vector,
        right_vector,
        panel=panel,
        interval_kind="two_sided",
        config=config,
        inventory=inventory,
    )
    reordered_vectors = _metric_pair(
        panel,
        config,
        inventory,
        dict(reversed(tuple(left.items()))),
        right,
    )
    second = paired_weighted_bootstrap(
        reordered_vectors[0],
        reordered_vectors[1],
        panel=panel,
        interval_kind="two_sided",
        config=config,
        inventory=inventory,
    )

    assert asdict(first) == asdict(second)
    assert first.observed_difference == pytest.approx(1.0 / 6.0)
    assert first.contains_zero is True
    assert first.ranking_performed is False
    assert first.selection_performed is False
    assert not hasattr(first, "checkpoint_order")

    lower_vectors = _metric_pair(
        panel,
        config,
        inventory,
        {key: value + 1.0 for key, value in right.items()},
        right,
    )
    lower = paired_weighted_bootstrap(
        lower_vectors[0],
        lower_vectors[1],
        panel=panel,
        interval_kind="one_sided_lower",
        config=config,
        inventory=inventory,
    )
    assert lower.lower_bound == pytest.approx(1.0)
    assert lower.upper_bound is None
    assert lower.contains_zero is None

    with pytest.raises(ValidationPanelError, match="exact nonterminal"):
        bind_panel_metric_vector(
            {0: 2.0},
            metric_id="canonical_successor_nll",
            source_kind="uniform_canonical_successor",
            panel=panel,
            config=config,
            inventory=inventory,
        )


def test_metric_vectors_reject_test_and_embedded_best_state(inventory):
    config = _config()
    panel = seal_validation_panel(
        panel_kind="production_law",
        rows=(_row(0), _row(1)),
        draw_contract={
            "stage": "initial",
            "requested_stream_draws": 2,
            "stream_draws_consumed": 2,
        },
        config=config,
        inventory=inventory,
    )
    vector = bind_panel_metric_vector(
        {0: 1.0, 1: 2.0},
        metric_id="canonical_successor_nll",
        source_kind="snapshot_current_state",
        panel=panel,
        config=config,
        inventory=inventory,
        snapshot_step=500,
    )
    assert vector.partition == "validation"
    assert vector.state_source == "current_state_dict"
    assert vector.snapshot_sha256 == inventory["snapshots"][0]["sha256"]

    with pytest.raises(ValidationPanelError, match="validation only"):
        validate_panel_metric_vector(
            replace(vector, partition="test"),
            panel=panel,
            config=config,
            inventory=inventory,
        )
    with pytest.raises(ValidationPanelError, match="current snapshot state"):
        validate_panel_metric_vector(
            replace(vector, state_source="best_state_dict"),
            panel=panel,
            config=config,
            inventory=inventory,
        )
    with pytest.raises(ValidationPanelError, match="cannot carry a snapshot"):
        bind_panel_metric_vector(
            {0: 1.0, 1: 2.0},
            metric_id="canonical_successor_nll",
            source_kind="uniform_canonical_successor",
            panel=panel,
            config=config,
            inventory=inventory,
            snapshot_step=500,
        )


def test_production_expansion_uses_frozen_triggers_without_ranking(inventory):
    config = _config(production_draws=(4, 8))
    panel = seal_validation_panel(
        panel_kind="production_law",
        rows=(
            _row(0, family="atom_insert"),
            _row(1, family="cycle_attach"),
            _row(2, family="atom_insert"),
            _row(3, family="cycle_attach"),
        ),
        draw_contract={
            "stage": "initial",
            "requested_stream_draws": 4,
            "stream_draws_consumed": 4,
        },
        config=config,
        inventory=inventory,
    )
    initial_vectors = _metric_pair(
        panel,
        config,
        inventory,
        {0: 1.0, 1: -1.0, 2: 1.0, 3: -1.0},
        {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
        right_source_kind="snapshot_current_state",
    )
    interval = paired_weighted_bootstrap(
        initial_vectors[0],
        initial_vectors[1],
        panel=panel,
        interval_kind="two_sided",
        config=config,
        inventory=inventory,
    )
    decision = production_panel_expansion_decision(
        panel=panel,
        primary_interval=interval,
        inconclusive_hard_gates=("ring_open_support",),
        config=config,
        inventory=inventory,
    )

    assert decision.expand is True
    assert decision.target_draws == 8
    assert "primary_paired_interval_contains_zero" in decision.reasons
    assert "active_family_below_64:atom_insert" in decision.reasons
    assert "active_family_below_64:cycle_attach" in decision.reasons
    assert decision.ranking_performed is False
    assert decision.selection_performed is False

    config_decisive = _config(production_draws=(128, 256))
    rows = tuple(
        _row(
            index,
            family="atom_insert" if index < 64 else "cycle_attach",
        )
        for index in range(128)
    )
    decisive_panel = seal_validation_panel(
        panel_kind="production_law",
        rows=rows,
        draw_contract={
            "stage": "initial",
            "requested_stream_draws": 128,
            "stream_draws_consumed": 128,
        },
        config=config_decisive,
        inventory=inventory,
    )
    decisive_vectors = _metric_pair(
        decisive_panel,
        config_decisive,
        inventory,
        {index: 1.0 for index in range(128)},
        {index: 0.0 for index in range(128)},
        right_source_kind="snapshot_current_state",
    )
    decisive = paired_weighted_bootstrap(
        decisive_vectors[0],
        decisive_vectors[1],
        panel=decisive_panel,
        interval_kind="two_sided",
        config=config_decisive,
        inventory=inventory,
    )
    no_expand = production_panel_expansion_decision(
        panel=decisive_panel,
        primary_interval=decisive,
        inconclusive_hard_gates=(),
        config=config_decisive,
        inventory=inventory,
    )
    assert no_expand.expand is False
    assert no_expand.target_draws == 128

    expanded_panel = seal_validation_panel(
        panel_kind="production_law",
        rows=tuple(
            _row(
                index,
                family="atom_insert" if index % 2 == 0 else "cycle_attach",
            )
            for index in range(256)
        ),
        draw_contract={
            "stage": "expanded",
            "requested_stream_draws": 256,
            "stream_draws_consumed": 256,
        },
        config=config_decisive,
        inventory=inventory,
    )
    exhausted = production_panel_expansion_decision(
        panel=expanded_panel,
        primary_interval=None,
        inconclusive_hard_gates=("ring_open_support",),
        config=config_decisive,
        inventory=inventory,
    )
    assert exhausted.expand is False
    assert exhausted.target_draws == 256
    assert "primary_paired_interval_unavailable" in exhausted.reasons
    assert (
        "hard_gate_numerically_inconclusive:ring_open_support"
        in exhausted.reasons
    )


def test_family_expansion_targets_ambiguous_without_verdict(inventory):
    config = _config(forensic_minima=(1, 2))
    panel = seal_validation_panel(
        panel_kind=FAMILY_FORENSICS_PANEL_ID,
        rows=(
            _row(0, stream_draw_index=1, family="atom_insert"),
            _row(1, stream_draw_index=4, family="cycle_attach"),
        ),
        draw_contract={
            "requested_nonterminal_examples_by_family": {
                "atom_insert": 1,
                "cycle_attach": 1,
            },
            "stream_draws_consumed": 5,
        },
        config=config,
        inventory=inventory,
    )
    atom_vectors = _metric_pair(
        panel,
        config,
        inventory,
        {1: 1.0},
        {1: 0.0},
        family_filter="atom_insert",
        metric_id="canonical_successor_log_likelihood",
    )
    atom_interval = paired_weighted_bootstrap(
        atom_vectors[0],
        atom_vectors[1],
        panel=panel,
        interval_kind="one_sided_lower",
        config=config,
        inventory=inventory,
    )
    cycle_vectors = _metric_pair(
        panel,
        config,
        inventory,
        {4: 0.0},
        {4: 0.0},
        family_filter="cycle_attach",
        metric_id="canonical_successor_log_likelihood",
    )
    cycle_interval = paired_weighted_bootstrap(
        cycle_vectors[0],
        cycle_vectors[1],
        panel=panel,
        interval_kind="one_sided_lower",
        config=config,
        inventory=inventory,
    )
    decision = family_forensics_expansion_decision(
        panel=panel,
        one_sided_intervals={
            "atom_insert": atom_interval,
            "cycle_attach": cycle_interval,
        },
        config=config,
        inventory=inventory,
    )

    assert decision.expand is True
    assert dict(decision.target_counts) == {
        "atom_insert": 1,
        "cycle_attach": 2,
    }
    assert decision.statistically_ambiguous_families == ("cycle_attach",)
    assert decision.ranking_performed is False
    assert decision.selection_performed is False

    expanded_panel = seal_validation_panel(
        panel_kind=FAMILY_FORENSICS_PANEL_ID,
        rows=(
            _row(0, stream_draw_index=0, family="atom_insert"),
            _row(1, stream_draw_index=1, family="atom_insert"),
            _row(2, stream_draw_index=2, family="cycle_attach"),
            _row(3, stream_draw_index=3, family="cycle_attach"),
        ),
        draw_contract={
            "requested_nonterminal_examples_by_family": {
                "atom_insert": 2,
                "cycle_attach": 2,
            },
            "stream_draws_consumed": 4,
        },
        config=config,
        inventory=inventory,
    )
    atom_final_vectors = _metric_pair(
        expanded_panel,
        config,
        inventory,
        {0: -1.0, 1: -1.0},
        {0: 0.0, 1: 0.0},
        family_filter="atom_insert",
        metric_id="canonical_successor_log_likelihood",
    )
    atom_final = paired_weighted_bootstrap(
        atom_final_vectors[0],
        atom_final_vectors[1],
        panel=expanded_panel,
        interval_kind="one_sided_lower",
        config=config,
        inventory=inventory,
    )
    cycle_final_vectors = _metric_pair(
        expanded_panel,
        config,
        inventory,
        {2: -1.0, 3: -1.0},
        {2: 0.0, 3: 0.0},
        family_filter="cycle_attach",
        metric_id="canonical_successor_log_likelihood",
    )
    cycle_final = paired_weighted_bootstrap(
        cycle_final_vectors[0],
        cycle_final_vectors[1],
        panel=expanded_panel,
        interval_kind="one_sided_lower",
        config=config,
        inventory=inventory,
    )
    final = family_forensics_expansion_decision(
        panel=expanded_panel,
        one_sided_intervals={
            "atom_insert": atom_final,
            "cycle_attach": cycle_final,
        },
        config=config,
        inventory=inventory,
    )
    assert final.expand is False
    assert final.statistically_ambiguous_families == (
        "atom_insert",
        "cycle_attach",
    )
    assert not hasattr(final, "eligible_checkpoint")
