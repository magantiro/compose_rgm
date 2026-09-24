"""The reservoir rule must be target-blind, deterministic, and keep the oracle head.

The motivating measurement: thiothixene's defining thioxanthene tricycle is absent from
the top 40 of its oracle ranking and its best donor sits at rank 41. A rule that only
deepens top-N uniformly would still miss the scaffold families; a rule that only
diversifies would throw away the strongest molecules. These tests pin both halves.
"""

from __future__ import annotations

import ast
import inspect

import pytest

from compose_v4.control import prescreen_reservoir as pr

# Two tight scaffold families plus three structurally distinct molecules. The families are
# ranked ABOVE the distinct ones, so a pure top-N rule returns only family members.
FAMILY_A = ["c1ccccc1C", "c1ccccc1CC", "c1ccccc1CCC", "c1ccccc1CCCC", "c1ccccc1CCCCC"]
FAMILY_B = ["c1ccncc1C", "c1ccncc1CC", "c1ccncc1CCC", "c1ccncc1CCCC"]
DISTINCT = ["C1CCOc2ccccc21", "O=C1NC(=O)CS1", "c1ccc2Sc3ccccc3Cc2c1"]
RANKED = [(1.0 - 0.01 * i, s) for i, s in enumerate(FAMILY_A + FAMILY_B + DISTINCT)]


def test_reaches_structures_a_pure_top_n_rule_would_miss():
    picked = pr.select_reservoir(RANKED, count=6, head_fraction=0.25)
    smiles = {e.smiles for e in picked}
    assert smiles & set(DISTINCT), "diversity selection reached none of the distinct scaffolds"
    # A pure top-N of the same size would be family members only.
    assert not {s for _, s in RANKED[:6]} & set(DISTINCT)


def test_keeps_the_oracle_head():
    picked = pr.select_reservoir(RANKED, count=8, head_fraction=0.5)
    head = [e for e in picked if e.selected_by == "oracle_head"]
    assert len(head) == 4
    assert [e.smiles for e in head] == [s for _, s in RANKED[:4]]
    assert head[0].oracle_rank == 1


def test_is_deterministic():
    a = [e.smiles for e in pr.select_reservoir(RANKED, count=7)]
    b = [e.smiles for e in pr.select_reservoir(RANKED, count=7)]
    assert a == b


def test_never_repeats_a_molecule():
    picked = pr.select_reservoir(RANKED + RANKED, count=9)
    assert len({e.smiles for e in picked}) == len(picked)


def test_prefix_bounds_what_can_be_reached():
    picked = pr.select_reservoir(RANKED, count=5, prefix=5, head_fraction=0.2)
    assert {e.smiles for e in picked} <= set(FAMILY_A)


def test_unparsable_entries_are_skipped_not_fatal():
    picked = pr.select_reservoir([(1.0, "not_a_molecule"), *RANKED], count=4)
    assert all(e.smiles != "not_a_molecule" for e in picked)


def test_refuses_a_nonpositive_size():
    with pytest.raises(ValueError):
        pr.select_reservoir(RANKED, count=0)


def test_module_cannot_consume_a_reference():
    """Target-blindness checked on the CODE, not on prose.

    A text search over the source is the wrong instrument: this module's own docstring
    says "never against a reference", which a naive match flags. Assert the two things
    that actually constrain it -- no function here accepts a target/reference argument,
    and the maximum-common-substructure machinery is not imported at all.
    """
    params = set(inspect.signature(pr.select_reservoir).parameters)
    assert params == {"ranked", "count", "prefix", "head_fraction"}

    tree = ast.parse(inspect.getsource(pr))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            names = {a.arg for a in node.args.args} | {a.arg for a in node.args.kwonlyargs}
            leaked = {n for n in names if "target" in n or "reference" in n or n == "ref"}
            assert not leaked, f"{node.name} accepts reference-bearing argument(s) {leaked}"
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "rdFMCS" not in alias.name
        if isinstance(node, ast.ImportFrom):
            assert "rdFMCS" not in {a.name for a in node.names}
