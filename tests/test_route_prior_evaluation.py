"""Guards for the held-out comparison harness and its statistics.

The harness makes three promises that decide whether its numbers mean anything:
the arms are MATCHED, the parent states are the ones the production proposal
path uses, and the control statistics are exact rather than approximated. Each
is checked here against an expectation computed independently of the code under
test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from route_prior_paired_summary import sign_test
from route_prior_region_ranking import (
    uniform_recall_at,
    uniform_reciprocal_rank,
)

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.control.dynamic_program_synthesis import (
    GENERIC_MODULES,
    compile_generic_module,
)
from compose_v4.experiments.route_program_prior_eval import (
    PROPOSAL_SLOTS,
    parent_state,
    run_declared_arm,
    uniform_families,
)

_LEAD = "CN(C)Cc3ccc2c(CNC(=O)c1cccn12)c3"


# ---- The state semantics trap -------------------------------------------


def test_the_parent_is_padded_to_the_proposal_path_slot_capacity():
    """48 slots, which is NOT the editing corpus's 40 and NOT the 40-heavy-atom
    endpoint ceiling. Three distinct numbers that have been conflated before."""

    padded = parent_state(_LEAD)
    tight = smiles_to_molecular_graph(_LEAD)
    assert padded.n_atoms == PROPOSAL_SLOTS == 48
    assert tight.n_atoms < PROPOSAL_SLOTS
    assert len(padded.atom_types) == PROPOSAL_SLOTS
    assert padded.n_real_atoms == tight.n_real_atoms


def test_a_tight_graph_cannot_grow_and_a_padded_one_can():
    """MUTATION: dropping `pad_molecular_graph` from `parent_state` turns this
    red. Atom birth needs a free slot, so on a tight graph the whole growth half
    of the module vocabulary silently disappears -- which has invalidated three
    earlier measurements in this repository."""

    padded = parent_state(_LEAD)
    tight = smiles_to_molecular_graph(_LEAD)
    _product, stage = compile_generic_module(
        padded, np.random.default_rng(0), "segment_grow"
    )
    assert any(a["executor_rule"] == "atom_insert" for a in stage["actions"])
    with pytest.raises((ValueError, IndexError)):
        compile_generic_module(tight, np.random.default_rng(0), "segment_grow")


# ---- The arms are matched ------------------------------------------------


def test_a_declared_arm_uses_the_module_counts_it_is_given():
    """MUTATION: drawing K inside `run_declared_arm` instead of consuming
    `module_counts` turns this red. Matched K is what stops an arm winning by
    declaring shorter, easier programs."""

    source = parent_state(_LEAD)
    requested = [1, 3, 2, 1]
    seen = []

    def families_for(graph, rng, module_count):
        seen.append(module_count)
        return uniform_families(graph, rng, module_count)

    run_declared_arm(
        source,
        draws=len(requested),
        seed=5,
        families_for=families_for,
        region_law=None,
        module_counts=requested,
    )
    assert seen == requested


def test_uniform_families_only_names_production_families():
    graph = parent_state(_LEAD)
    families = uniform_families(graph, np.random.default_rng(1), 3)
    assert len(families) == 3
    assert set(families) <= set(GENERIC_MODULES)


def test_a_declared_arm_counts_work_even_when_every_draw_is_refused():
    """A refused draw is still work, and an arm that reported zero work for its
    failures would look cheaper precisely where it performed worst."""

    source = parent_state(_LEAD)
    tally = run_declared_arm(
        source,
        draws=3,
        seed=11,
        families_for=lambda g, rng, k: ("append_ring",) * k,
        region_law=None,
        module_counts=[3, 3, 3],
    )
    assert tally.draws == 3
    assert tally.module_compiles_attempted >= 9


# ---- The control statistics are exact ------------------------------------


def test_uniform_recall_matches_a_hand_computed_hypergeometric():
    """Expectations are hand-computed, not recomputed from the function."""

    assert uniform_recall_at(4, 1, 1) == pytest.approx(0.25)
    assert uniform_recall_at(4, 1, 2) == pytest.approx(0.5)
    assert uniform_recall_at(4, 1, 4) == pytest.approx(1.0)
    # two targets out of four, top two: 1 - C(2,2)/C(4,2) = 1 - 1/6
    assert uniform_recall_at(4, 2, 2) == pytest.approx(5.0 / 6.0)
    assert uniform_recall_at(10, 0, 3) == 0.0


def test_uniform_reciprocal_rank_matches_the_enumerated_expectation():
    """With one target among four, every rank is equally likely."""

    expected = (1.0 + 0.5 + 1.0 / 3.0 + 0.25) / 4.0
    assert uniform_reciprocal_rank(4, 1) == pytest.approx(expected)
    # With every position a target the first is always rank one.
    assert uniform_reciprocal_rank(4, 4) == pytest.approx(1.0)


def test_sign_test_matches_the_binomial_by_hand():
    assert sign_test([1.0] * 5)["p_value"] == pytest.approx(2.0 / 32.0)
    assert sign_test([-1.0] * 5)["p_value"] == pytest.approx(2.0 / 32.0)
    assert sign_test([0.0] * 5)["p_value"] == 1.0
    balanced = sign_test([1.0, -1.0, 1.0, -1.0])
    assert balanced["p_value"] == pytest.approx(1.0)
    assert balanced["positive"] == 2 and balanced["negative"] == 2
    assert sign_test([1.0, 1.0, 0.0])["ties"] == 1
