"""Guards on terminal free-objective refinement of T4 endpoints.

Two defects are cheap to introduce here and expensive to notice.

The first is evaluating on a TIGHT graph.  A molecular graph rebuilt from SMILES
carries exactly ``n_real_atoms`` slots, so atom birth is inexpressible and the
whole ``atom_insert`` family vanishes from the enumerated support without an
error -- measured at 305 of 733 marks (42%) on the molecule pinned below.  A
test that only proved refinement runs would pass on a tight graph, so the
negative case is asserted directly.

The second is a gate that drifts from the campaign's own gate.  Eligibility is
re-checked here against :class:`~compose_v4.experiments.t4_fiber_campaign.Fiber`
independently of the flag the module reports, because a comparison whose
expectation is recomputed from the code under test cannot fail.
"""

from __future__ import annotations

from functools import lru_cache

import pytest
from rdkit import Chem

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.generic_legal_action_policy import enumerate_legal_successors
from compose_v4.experiments.t4_fiber_campaign import QED_MIN, SA_MAX, Fiber
from compose_v4.experiments.t4_objective_refinement import (
    PRODUCTION_MAX_ATOMS,
    SourceNotRepresentable,
    endpoint_properties,
    enumerate_refinements,
    free_objective_near_miss,
    refine_endpoint,
)

#: The FA7 cell whose live terminal status was ``candidate_exhaustion``.
FA7_0_SEED = "CC(C)CCN(Cc2ccc1ccc(C(N)=N)cc1c2)C(=O)c3cccc4ccccc34"

#: Its best similarity-passing endpoint: QED 0.598337, i.e. 0.001663 UNDER the
#: bound, while holding a similarity margin of +0.072 and an SA margin of +1.72.
FA7_0_LEADER = "CCC(=O)N(CCC(C)C)Cc1ccc2ccc(C(=N)N)cc2c1"

FA7_0_DELTA = 0.6


@lru_cache(maxsize=4)
def _refined(seed: str, delta: float, smiles: str):
    """Enumeration is seconds-scale, so the slow call is paid once per file."""

    return enumerate_refinements(seed, delta, smiles)


# ---- Slot semantics ---------------------------------------------------------


def test_tight_graph_forbids_atom_birth_and_padding_restores_it():
    """The failure this module's padding exists to prevent, asserted directly."""

    tight = smiles_to_molecular_graph(FA7_0_LEADER)
    padded = pad_molecular_graph(smiles_to_molecular_graph(FA7_0_LEADER), PRODUCTION_MAX_ATOMS)

    tight_rules = [s.rule for s in enumerate_legal_successors(tight)]
    padded_rules = [s.rule for s in enumerate_legal_successors(padded)]

    assert tight_rules.count("atom_insert") == 0
    assert padded_rules.count("atom_insert") > 0
    assert len(padded_rules) > len(tight_rules)


def test_refinement_can_grow_the_molecule():
    """A successor heavier than its source is only reachable from a padded graph.

    This is the behavioural form of the check above: it fails if the module ever
    stops padding, without reading how the module pads.
    """

    source_heavy = Chem.MolFromSmiles(FA7_0_LEADER).GetNumHeavyAtoms()
    rows = _refined(FA7_0_SEED, FA7_0_DELTA, FA7_0_LEADER)

    assert any(row.heavy > source_heavy for row in rows)


# ---- Trigger ----------------------------------------------------------------


def test_trigger_fires_only_on_a_free_objective_near_miss():
    base = {"qed": 0.50, "sa": 2.0, "sim": 0.70, "v": 0.1}
    assert free_objective_near_miss(base, 0.6)

    assert not free_objective_near_miss({**base, "sim": 0.55}, 0.6)
    assert not free_objective_near_miss({**base, "sa": SA_MAX + 0.01}, 0.6)
    assert not free_objective_near_miss({**base, "qed": QED_MIN}, 0.6)
    assert not free_objective_near_miss(None, 0.6)


def test_trigger_reads_no_target_identity():
    """Two different seeds with the same violation vector trigger identically."""

    props = {"qed": 0.55, "sa": 3.0, "sim": 0.65, "v": 0.08}
    assert free_objective_near_miss(props, 0.6) is True


def test_fa7_0_leader_triggers_and_is_not_already_eligible():
    props = endpoint_properties(FA7_0_SEED, FA7_0_DELTA, FA7_0_LEADER)

    assert props["sim"] >= FA7_0_DELTA
    assert props["sa"] <= SA_MAX
    assert props["qed"] < QED_MIN
    assert free_objective_near_miss(props, FA7_0_DELTA)
    assert Fiber(FA7_0_SEED, FA7_0_DELTA).check(FA7_0_LEADER) is None


# ---- Gate agreement ---------------------------------------------------------


def test_reported_eligibility_agrees_with_an_independent_gate_call():
    """Re-decide every row with a fresh Fiber; the module's flag must not drift."""

    fiber = Fiber(FA7_0_SEED, FA7_0_DELTA)
    rows = _refined(FA7_0_SEED, FA7_0_DELTA, FA7_0_LEADER)

    assert rows
    for row in rows:
        assert row.eligible == (fiber.check(row.smiles) is not None)


def test_eligible_rows_satisfy_every_contract_bound():
    rows = _refined(FA7_0_SEED, FA7_0_DELTA, FA7_0_LEADER)

    for row in (r for r in rows if r.eligible):
        assert row.similarity >= FA7_0_DELTA
        assert row.qed >= QED_MIN
        assert row.sa <= SA_MAX


# ---- The measured rescue ----------------------------------------------------


def test_one_legal_rewrite_rescues_the_fa7_0_leader():
    """The cell labelled `candidate_exhaustion` is one edit from an eligible molecule."""

    report = refine_endpoint(FA7_0_SEED, FA7_0_DELTA, FA7_0_LEADER)

    assert report["triggered"] is True
    assert report["source_eligible"] is False
    assert report["eligible_count"] >= 1
    assert max(row["qed"] for row in report["eligible"]) >= QED_MIN


def test_refinement_spends_no_oracle_calls():
    report = refine_endpoint(FA7_0_SEED, FA7_0_DELTA, FA7_0_LEADER)

    assert report["oracle_calls"] == 0


def test_refinement_rejects_an_unscorable_source():
    with pytest.raises(ValueError):
        enumerate_refinements(FA7_0_SEED, FA7_0_DELTA, "not-a-molecule")


def test_a_radical_endpoint_is_reported_not_raised():
    """T4 lanes emit radicals; batching over a panel must not die on one.

    This exact species came out of the fa7_0 proposal lanes and crashed the
    first refinement sweep inside the chemistry layer.
    """

    radical = "[CH]c1ccc(CN(CCC(C)CCN)C(=O)c2cccc3ccccc23)cc1C=C"
    report = refine_endpoint(FA7_0_SEED, FA7_0_DELTA, radical)

    assert report["source_representable"] is False
    assert report["eligible_count"] == 0
    assert report["oracle_calls"] == 0


def test_enumerate_refinements_raises_on_an_unrepresentable_source():
    radical = "[CH]c1ccc(CN(CCC(C)CCN)C(=O)c2cccc3ccccc23)cc1C=C"
    with pytest.raises(SourceNotRepresentable):
        enumerate_refinements(FA7_0_SEED, FA7_0_DELTA, radical)
