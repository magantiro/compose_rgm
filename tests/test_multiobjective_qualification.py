"""Known-answer and adversarial tests for the multiobjective qualification layer.

Project rule: every new analysis metric ships with a fixture in which its
intended conclusion is FALSE, and a test asserting the metric says so. Seven
statistics with signs fixed by construction have already been caught in this
project, one of them in new code written by an agent who had already read the
rule. So each guard below has a paired fixture that makes it return the other
answer.
"""

from __future__ import annotations

import json

import pytest

from compose_v4.experiments.multiobjective_qualification import (
    ADMISSIBLE,
    COMPOSE_INTERNAL_AXES,
    INCOMPARABLE_INTERNAL_AXIS,
    INCOMPARABLE_SECONDARY_AXIS,
    INCOMPARABLE_SURROGATE,
    PANEL_A,
    PANEL_B,
    PANEL_NONE,
    RESOURCE_AXES,
    FrozenScaleError,
    PanelAdmissionError,
    ResourceAxisError,
    SourceConditioningRecord,
    assert_not_cross_panel,
    assert_reference_is_frozen,
    canonical_axis,
    load_frozen_scales,
    oracle_efficiency_verdict,
    panel_admission,
    reconcile_ledger,
)

CITE_HNGFN_ROLLOUT = (
    "HN-GFN main.py:145 `m = BlockMoleculeDataExtended()` -- every rollout "
    "starts from an empty block molecule"
)
CITE_COMPOSE_START = (
    "modal_apps/pareto_control_app.py: `start = canonical_state_key(...task['source']...)`"
)


# ---------------------------------------------------------------------------
# Counter reconciliation
# ---------------------------------------------------------------------------


def test_the_three_vocabularies_agree_on_the_algorithmic_axis():
    """Lane 4's `raw_oracle_calls` and Workstream D's `oracle_requests` are one axis."""
    assert (
        canonical_axis("raw_oracle_calls", "pareto_control.CostLedger")
        == canonical_axis("oracle_requests", "oracle_accounting.OracleCounts")
        == "algorithmic_oracle_requests"
    )


def test_native_and_unique_are_the_same_axis_across_lanes():
    assert (
        canonical_axis("native_oracle_calls", "pareto_control.CostLedger")
        == canonical_axis(
            "unique_valid_canonical_evaluations", "oracle_accounting.OracleCounts"
        )
        == "unique_oracle_evaluations"
    )


def test_the_json_key_and_the_module_name_for_post_hoc_scoring_agree():
    """`harness_only_requests` in the committed JSON is `benchmark_eval_requests`."""
    assert (
        canonical_axis("harness_only_requests", "pareto_oracle_semantics")
        == canonical_axis("benchmark_eval_requests", "pareto_oracle_semantics")
        == "benchmark_eval_requests"
    )


def test_an_undeclared_counter_raises_rather_than_being_dropped():
    with pytest.raises(ResourceAxisError, match="no declared meaning"):
        canonical_axis("mystery_calls", "pareto_control.CostLedger")


def test_reconcile_keeps_the_raw_instrument_record_out_of_the_axes():
    """The verbatim execution record is a parity object, not a resource axis.

    Mapping it onto the algorithmic axis would attribute our harness overhead to
    the method -- the exact error `pareto_oracle_semantics` exists to correct.
    """
    ledger = {
        "raw_instrument_oracle_requests": 623333,
        "algorithmic_oracle_requests": 623328,
        "harness_only_requests": 5,
        "native_oracle_calls": 269884,
        "kernel_calls": 393,
    }
    out = reconcile_ledger(ledger, "pareto_oracle_semantics")
    assert out["algorithmic_oracle_requests"] == 623328
    assert out["benchmark_eval_requests"] == 5
    assert "raw_instrument_oracle_requests" not in out
    assert out["passthrough"]["raw_instrument_oracle_requests"] == 623333


def test_every_declared_axis_has_a_stated_meaning():
    for axis, meaning in RESOURCE_AXES.items():
        assert meaning.strip(), f"{axis} has no semantics"


# ---------------------------------------------------------------------------
# Panel admission -- with the fixture where the intended conclusion is false
# ---------------------------------------------------------------------------


def test_a_source_conditioned_preference_method_enters_panel_a():
    """The POSITIVE fixture: the guard must be able to say YES."""
    record = SourceConditioningRecord(
        method="COMPOSE-fixed-preference",
        supplied_source="YES",
        edit_budget="YES",
        preference_conditioned="YES",
        evidence={
            "supplied_source": CITE_COMPOSE_START,
            "edit_budget": "pareto_control_app.py: BUDGET = 6, frozen by PROTOCOL 7.1",
            "preference_conditioned": (
                "pareto_control_app.py: PREFERENCES = (0.1, 0.3, 0.5, 0.7, 0.9)"
            ),
        },
    )
    assert panel_admission(record) == PANEL_A


def test_a_de_novo_preference_method_is_panel_b_not_panel_a():
    """The ADVERSARIAL fixture: a globally free method must NOT reach Panel A.

    HN-GFN is preference-conditioned and strong, and it is exactly the method a
    reader would expect to see beside COMPOSE. It still cannot enter Panel A,
    because it does not start from a supplied molecule.
    """
    record = SourceConditioningRecord(
        method="HN-GFN",
        supplied_source="NO",
        edit_budget="NO",
        preference_conditioned="YES",
        evidence={
            "supplied_source": CITE_HNGFN_ROLLOUT,
            "edit_budget": (
                "HN-GFN mol_mdp_ext.py:169 `add_block_to` is the only trajectory "
                "action; there is no source and so no trust region"
            ),
            "preference_conditioned": (
                "HN-GFN main.py:227 `raw_reward = (weights*score).sum()`; "
                "model_pred_hyper.py conditions the output heads on the weights"
            ),
        },
    )
    assert panel_admission(record) == PANEL_B
    assert panel_admission(record) != PANEL_A


def test_a_method_that_is_neither_is_not_admissible():
    record = SourceConditioningRecord(
        method="OP-GFN",
        supplied_source="NO",
        edit_budget="NO",
        preference_conditioned="NO",
        evidence={
            "supplied_source": (
                "OP-GFN algo/graph_sampling.py:79 `graphs = [self.env.new() ...]` "
                "with envs/graph_building_env.py:139 `def new(self): return Graph()`"
            ),
            "edit_budget": "no source exists, so no trust region is definable",
            "preference_conditioned": (
                "OP-GFN tasks/seh_frag_moo.py:398,414 `preference_type` is None "
                "unless --type pref, and --type pref is the PC-GFN baseline"
            ),
        },
    )
    assert panel_admission(record) == PANEL_NONE


def test_unverified_keeps_a_method_out_of_the_main_table():
    """Unresolved is not the same as absent, and it is not a free pass either."""
    record = SourceConditioningRecord(
        method="AReUReDi",
        supplied_source="UNVERIFIED",
        edit_budget="UNVERIFIED",
        preference_conditioned="UNVERIFIED",
    )
    assert panel_admission(record) == PANEL_NONE


def test_an_asserted_capability_without_evidence_is_refused():
    with pytest.raises(PanelAdmissionError, match="no evidence"):
        SourceConditioningRecord(
            method="wishful",
            supplied_source="YES",
            edit_budget="YES",
            preference_conditioned="YES",
        )


def test_cross_panel_comparison_raises():
    panels = {"COMPOSE": PANEL_A, "HN-GFN": PANEL_B}
    with pytest.raises(PanelAdmissionError, match="different tasks"):
        assert_not_cross_panel(panels, ("COMPOSE", "HN-GFN"))


def test_same_panel_comparison_is_allowed():
    """The paired fixture: the guard must be able to permit something."""
    panels = {"COMPOSE-global": PANEL_B, "HN-GFN": PANEL_B}
    assert_not_cross_panel(panels, ("COMPOSE-global", "HN-GFN"))


# ---------------------------------------------------------------------------
# Efficiency comparability
# ---------------------------------------------------------------------------


def test_surrogate_asymmetry_makes_an_oracle_comparison_inadmissible():
    """The substantive guard.

    HN-GFN's published budget is 200 + 8x100 = 1000 true-oracle evaluations
    (main_mobo.py:50-52), spent while its GFlowNet consults a learned proxy.
    COMPOSE pays the frozen evaluator directly. On `unique_oracle_evaluations`
    alone HN-GFN looks over two orders of magnitude cheaper, and that number
    would be reporting amortization, not efficiency.
    """
    compose = {"unique_oracle_evaluations": 269884, "surrogate_calls": 0}
    hn_gfn = {"unique_oracle_evaluations": 1000, "surrogate_calls": 40000}
    verdict, reason = oracle_efficiency_verdict(
        "unique_oracle_evaluations", compose, hn_gfn
    )
    assert verdict == INCOMPARABLE_SURROGATE
    assert "amortization" in reason


def test_two_surrogate_free_methods_are_comparable():
    """The ADVERSARIAL fixture for the surrogate guard: it must be able to pass.

    A guard that can only ever return INCOMPARABLE has not measured anything.
    """
    compose = {"unique_oracle_evaluations": 269884, "surrogate_calls": 0}
    gen_rank = {"unique_oracle_evaluations": 57, "surrogate_calls": 0}
    verdict, _ = oracle_efficiency_verdict(
        "unique_oracle_evaluations", compose, gen_rank
    )
    assert verdict == ADMISSIBLE


def test_two_surrogate_using_methods_are_comparable_to_each_other():
    left = {"unique_oracle_evaluations": 1000, "surrogate_calls": 40000}
    right = {"unique_oracle_evaluations": 1000, "surrogate_calls": 12000}
    verdict, _ = oracle_efficiency_verdict("unique_oracle_evaluations", left, right)
    assert verdict == ADMISSIBLE


def test_compose_internal_axes_may_not_carry_an_external_claim():
    for axis in COMPOSE_INTERNAL_AXES:
        verdict, _ = oracle_efficiency_verdict(
            axis, {axis: 1, "surrogate_calls": 0}, {axis: 2, "surrogate_calls": 0}
        )
        assert verdict == INCOMPARABLE_INTERNAL_AXIS


def test_wall_time_may_not_be_a_primary_axis():
    verdict, _ = oracle_efficiency_verdict(
        "wall_core_seconds",
        {"wall_core_seconds": 10, "surrogate_calls": 0},
        {"wall_core_seconds": 20, "surrogate_calls": 0},
    )
    assert verdict == INCOMPARABLE_SECONDARY_AXIS


def test_an_undeclared_axis_raises():
    with pytest.raises(ResourceAxisError):
        oracle_efficiency_verdict("vibes", {}, {})


# ---------------------------------------------------------------------------
# The frozen hypervolume normalization
# ---------------------------------------------------------------------------


def test_frozen_scales_load_the_adopted_pair(tmp_path):
    census = {
        "adopted_pair": "potency_vs_developability",
        "pairs": [
            {
                "pair": "potency_vs_developability",
                "objective_a": "P",
                "objective_b": "D",
            }
        ],
        "frozen_scales": {
            "utopia_p99": {"P": 2.7315048451066857, "D": 0.555419816046901},
            "reference_p5": {"P": -1.2689119285385548, "D": -1.3547257562637012},
        },
    }
    path = tmp_path / "census.json"
    path.write_text(json.dumps(census))
    scales = load_frozen_scales(path)
    assert scales["objective_a"] == "P"
    assert scales["nadir_p5"] == [-1.2689119285385548, -1.3547257562637012]
    assert scales["utopia_p99"] == [2.7315048451066857, 0.555419816046901]
    assert scales["is_a_cap"] is False


def test_the_real_committed_census_matches_the_frozen_constants():
    """Known-answer test against the artifact actually committed to the repo."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    census = repo / "diagnostics" / "pareto_tradeoff_census.json"
    if not census.exists():  # pragma: no cover - artifact not present on this branch
        pytest.skip("pareto_tradeoff_census.json not on this branch")
    scales = load_frozen_scales(census)
    assert scales["pair"] == "potency_vs_developability"
    assert scales["objective_a"] == "P"
    assert scales["objective_b"] == "D"
    assert scales["utopia_p99"] == pytest.approx(
        [2.7315048451066857, 0.555419816046901]
    )
    assert scales["nadir_p5"] == pytest.approx(
        [-1.2689119285385548, -1.3547257562637012]
    )


def test_a_reference_minted_from_a_front_is_refused():
    with pytest.raises(FrozenScaleError, match="own achievement"):
        assert_reference_is_frozen("nadir taken from the COMPOSE verified front")


def test_the_frozen_provenance_is_accepted():
    """The paired fixture: the guard must be able to accept the real freeze."""
    assert_reference_is_frozen(
        "diagnostics/pareto_tradeoff_census.json :: frozen_scales"
    )
