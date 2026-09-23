"""The action is the MACRO, not the primitives inside it.

These pin the semi-Markov contract the controller is built on: one coherent COMPOSE
program is one action, it earns one reward `u(G') - u(G)`, and it is REPRESENTED by what
kind of transformation it is and how much structure it moved -- never by how many
primitive edits it happened to compile into.
"""
from __future__ import annotations

import numpy as np

from compose_v4.control.pmo_contextual_macro import (
    MACRO_FAMILIES,
    POST_EXECUTION_BLOCKS,
    edge_features,
    macro_families,
    macro_scale,
)
from compose_v4.control.pmo_reward_adaptive import FamilyLedger

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
PHENOL = "Oc1ccccc1"


def _candidate(modules, changes=None, rule_histogram=None):
    return {
        "provenance": {
            "metadata": {"modules": modules},
            "actual_changes": changes or {},
        },
        # Present on real candidates and deliberately NOT a source of family labels.
        "program_rule_histogram": rule_histogram or {},
    }


def test_macro_family_comes_from_the_program_not_the_executor_rules():
    """Four executor rule names are spelled like macro families.

    Reading the rule histogram would therefore half-work -- `cycle_close` would fire the
    right one-hot for the wrong reason, from a primitive count. The extractor must ignore
    it entirely, which is what makes the family label a statement about the ACTION.
    """
    candidate = _candidate(
        [{"family": "segment_grow", "parameters": {}, "primitive_edits": 4}],
        rule_histogram={"cycle_close": 3, "bond_reroute": 2, "atom_insert": 9},
    )
    families = macro_families(candidate)
    assert families == ("segment_grow",)
    for rule in ("cycle_close", "bond_reroute", "atom_insert"):
        assert rule not in families


def test_a_region_replacement_names_the_option_that_rebuilt_it():
    """Otherwise all fifteen rebuild families collapse into one label and no preference
    among them is learnable."""
    candidate = _candidate(
        [
            {
                "family": "region_replace",
                "parameters": {"rebuild_option": "fuse_ring", "excised_atoms": 6},
                "primitive_edits": 11,
            }
        ]
    )
    families = macro_families(candidate)
    assert "region_replace" in families            # the operation
    assert "region_replace:fuse_ring" in families  # the specific option
    assert "fuse_ring" in families                 # shared with ring builds reached elsewhere


def test_every_emitted_family_is_nameable_by_the_value_model():
    """A label the feature vector has no slot for is invisible: the model cannot prefer or
    avoid it. Both anchored rebuild options must be nameable."""
    for option in ("regrow", "ring_then_grow", "fuse_ring", "append_ring"):
        candidate = _candidate(
            [{"family": "region_replace", "parameters": {"rebuild_option": option}}]
        )
        for family in macro_families(candidate):
            assert family.split(":")[0] in MACRO_FAMILIES, family


def test_two_macros_of_identical_length_are_distinguishable():
    """The whole reason to represent macros: an eleven-primitive coherent ring operation
    and an eleven-primitive chain regrow must not share a feature vector."""
    shared = {"parent": ASPIRIN, "endpoint": PHENOL, "primitives": 11, "module_count": 1}
    ring = _candidate(
        [
            {
                "family": "region_replace",
                "parameters": {"rebuild_option": "fuse_ring", "excised_atoms": 7},
                "primitive_edits": 11,
            }
        ],
        changes={"heavy_atom_delta": -1, "ring_delta": 1},
    )
    chain = _candidate(
        [
            {
                "family": "region_replace",
                "parameters": {"rebuild_option": "regrow", "excised_atoms": 2},
                "primitive_edits": 11,
            }
        ],
        changes={"heavy_atom_delta": -1, "ring_delta": 0},
    )
    rows = []
    for candidate in (ring, chain):
        rows.append(
            edge_features(
                {
                    **shared,
                    "families": list(macro_families(candidate)),
                    "realized_families": list(macro_families(candidate)),
                    **macro_scale(candidate),
                },
                blocks=POST_EXECUTION_BLOCKS,
            )
        )
    assert not np.allclose(rows[0], rows[1])


def test_realized_scale_survives_when_length_is_held_fixed():
    """Guards the specific collapse: if the realized block were primitives+modules only,
    these two would be identical."""
    base = {
        "parent": ASPIRIN,
        "endpoint": PHENOL,
        "primitives": 8,
        "module_count": 2,
        "families": ["segment_replace"],
        "realized_families": ["segment_replace"],
    }
    small = edge_features(
        {**base, "excised_atoms": 1, "rebuild_capacity": 2, "d_rings": 0},
        blocks=POST_EXECUTION_BLOCKS,
    )
    large = edge_features(
        {**base, "excised_atoms": 13, "rebuild_capacity": 13, "d_rings": 2},
        blocks=POST_EXECUTION_BLOCKS,
    )
    assert not np.allclose(small, large)


def test_family_audit_reports_every_required_column():
    """The audit exists to read a discovered preference off a scored run; a missing column
    is a question that cannot be answered afterwards."""
    ledger = FamilyLedger()
    grow = {"families": ["segment_grow"]}
    ring = {"families": ["region_replace", "fuse_ring"]}
    ledger.snapshot([grow, ring], [0.5, 0.5], "before")
    ledger.propose([grow, ring])
    ledger.query(grow, 0.4)
    ledger.query(ring, 0.9)
    ledger.outcome(grow, -0.1)
    ledger.outcome(ring, 0.3)
    ledger.snapshot([grow, ring], [0.2, 0.8], "after")

    report = ledger.report()["families"]
    for column in (
        "n_proposed",
        "n_queried",
        "expected_delta_u",
        "p_delta_positive",
        "mean_predicted_value",
        "proposal_mass_shift",
    ):
        assert column in report["segment_grow"], column
    assert report["fuse_ring"]["expected_delta_u"] == 0.3
    assert report["segment_grow"]["p_delta_positive"] == 0.0
    # The shift is what shows reward moved the policy rather than merely being recorded.
    assert report["fuse_ring"]["proposal_mass_shift"] > 0
    assert report["segment_grow"]["proposal_mass_shift"] < 0
