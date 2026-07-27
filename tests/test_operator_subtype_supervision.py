"""Regression for the ring_system_grow supervision gap: a production-enabled operator SUBTYPE with zero
positive selected targets must fail the gate, even when every atom (element,valence) class is supervised.

This reproduces the earlier false-positive condition (atom-class-only audit passing while ring_system_grow
had zero supervision) and proves the subtype/template-level gate catches it."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from operator_subtype_supervision_gate import (  # noqa: E402
    check_operator_subtype_supervision,
    subtype_target_counts,
)

# production-enabled families (manifest operator_registry.production_enabled)
_ENABLED = (
    "atom_insert", "atom_delete", "atom_restate", "bond_reorder", "bond_reroute",
    "ring_system_grow", "ring_system_delete", "ring_system_restate",
)


def _recipe_like_sequences():
    """Teacher-mark family sequences matching the ACTUAL locked recipe: corruption (restate/delete/reorder/
    reroute/ring_system_delete/ring_system_restate) + MMP (atom insert/delete). NO ring_system_grow."""
    return [
        ["atom_restate", "atom_delete"],
        ["bond_reorder", "bond_reroute"],
        ["ring_system_delete"],           # ring OPENING is taught (trim-only)
        ["ring_system_restate"],
        ["atom_insert", "atom_delete"],   # MMP
        ["atom_insert", "atom_restate"],
    ]


def test_subtype_gate_catches_unsupervised_ring_system_grow():
    ok, report = check_operator_subtype_supervision(_recipe_like_sequences(), _ENABLED)
    assert not ok, "gate must FAIL: ring_system_grow has zero positive targets under the real recipe"
    assert report["unsupervised_enabled_families"] == ["ring_system_grow"]
    # every OTHER enabled family is supervised (the atom-class audit would have passed)
    assert report["positive_targets_by_family"].get("ring_system_grow", 0) == 0
    assert report["positive_targets_by_family"]["ring_system_delete"] >= 1


def test_subtype_gate_passes_when_grow_is_supervised():
    seqs = _recipe_like_sequences() + [["ring_system_grow"], ["ring_system_grow", "atom_restate"]]
    ok, report = check_operator_subtype_supervision(seqs, _ENABLED)
    assert ok, f"gate should pass once grow is supervised: {report['unsupervised_enabled_families']}"
    assert report["positive_targets_by_family"]["ring_system_grow"] == 2


def test_atom_class_coverage_alone_does_not_imply_family_coverage():
    # the old false-positive: full atom-class coverage but a whole ring family unsupervised
    counts = subtype_target_counts(_recipe_like_sequences())
    assert counts["atom_insert"] and counts["atom_restate"] and counts["atom_delete"]  # atoms fully covered
    assert counts.get("ring_system_grow", 0) == 0  # yet a production family is not


# Compositional ring ops: build_cycle_op_records records EXECUTOR rule_names (bond_insert/bond_delete) that
# the rate model scores under the cycle_insert/cycle_attach families (via _CYCLE_OP_EXECUTOR_TO_FAMILY). The
# gate must alias them, else a fully-supervised cycle op reads as unsupervised.
_CYCLE_ALIAS = {"bond_insert": "cycle_insert", "bond_delete": "cycle_attach"}


def test_cycle_op_executor_marks_alias_to_their_scoring_family():
    seqs = [["bond_delete"], ["bond_insert"], ["atom_restate", "bond_reorder"]]
    counts = subtype_target_counts(seqs, _CYCLE_ALIAS)
    assert counts["cycle_attach"] == 1 and counts["cycle_insert"] == 1
    assert "bond_delete" not in counts and "bond_insert" not in counts


def test_cycle_op_supervision_passes_only_with_the_alias():
    enabled = ("atom_restate", "bond_reorder", "cycle_insert", "cycle_attach")
    seqs = [["bond_delete"], ["bond_insert"], ["atom_restate"], ["bond_reorder"]]
    # without the alias the cycle families read as unsupervised
    ok_no_alias, report_no = check_operator_subtype_supervision(seqs, enabled)
    assert not ok_no_alias
    assert set(report_no["unsupervised_enabled_families"]) == {"cycle_insert", "cycle_attach"}
    # with the alias every enabled family is supervised
    ok_alias, _ = check_operator_subtype_supervision(seqs, enabled, _CYCLE_ALIAS)
    assert ok_alias
