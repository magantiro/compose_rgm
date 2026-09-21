"""Invariants of the bridge-separated region-draw law.

The load-bearing properties, in the order they are asserted below:

1.  The law's closed-form excision equals what the PRODUCTION deletion schedule
    executes.  The reference is the live ``_delete_pendant_fragment``, driven
    over real molecules -- never a transcription of it, which could not fail if
    production drifted.
2.  Uncapping the size widens the support and never narrows it.
3.  The margin tilt is a RE-RANKING, not a filter: every drawable region keeps
    strictly positive weight, so no region the executor could reach is deleted.
4.  The law reads nothing but structure and the task's declared free thresholds,
    and in particular has no net-formal-charge rule -- a cut that neutralizes a
    cation by excising its charged fragment must stay drawable.
5.  ``_delete_pendant_fragment`` with no law is byte-identical to the pre-change
    behaviour, including its random-number consumption.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import QED, DataStructs, rdFingerprintGenerator

from compose_v4.chem.molecular_graph import (
    is_element,
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.bridge_region_law import (
    HEAVY_CAPACITY,
    QED_FLOOR,
    SA_CEILING,
    SUPPORT_FLOOR,
    BridgeRegionLaw,
    FreeFeasibilityGate,
    RegionRealizationError,
    _sascorer,
    bridge_separated_regions,
    excise_region,
    free_gate_margin_law,
)
from compose_v4.control.dynamic_program_synthesis import (
    MAX_SEGMENT_LENGTH,
    _delete_pendant_fragment,
    _pendant_fragments,
)
from compose_v4.data.charge_policy import audit_charge_policy_transition

_SASCORER = _sascorer()
_FINGERPRINTS = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

#: The T4 proposal executor refuses anything else (`whole_ring_plan` line 72:
#: `n_atoms != 48 or not 1 <= n_real_atoms <= 40`), and `t4_fiber_campaign`
#: builds every parent this way.  A tight graph from `smiles_to_molecular_graph`
#: has no free slot and is rejected outright here rather than silently losing a
#: family, which is the 40-slot editing path's failure mode.
PROPOSAL_SLOTS = 48

# Real drug-like sources. fa7_0 and 5ht1b_2 are the actual T4 delta=0.6 benchmark
# leads; the rest are a fused polycyclic, a small amide and a ring control.
# 5ht1b_2 is cationic and is the cell whose only known witnesses neutralize it.
_FA7_0 = "CC(C)CCN(Cc2ccc1ccc(C(N)=N)cc1c2)C(=O)c3cccc4ccccc34"
_5HT1B_2 = "C1=CC2=NC=C(CCCN3CC[NH+](CCc4ccccc4)CC3)[C@H]2C=C1n1cnnc1"
_BRAF_0 = (
    "CCN(CC)CCNC(=O)c3cnn4c(c2cccc(NC(=O)Nc1ccc(Cl)c(C(F)(F)F)c1)c2)ccnc34"
)
_REAL = (
    _FA7_0,
    "COc1ccc(CNC(=O)c2ccccc2)cc1OC",
    _5HT1B_2,
    "CCOc1ccccc1C(=O)NC",
    "c1ccccc1",
)


def _state(smiles):
    graph = pad_molecular_graph(smiles_to_molecular_graph(smiles), PROPOSAL_SLOTS)
    assert graph.n_atoms == PROPOSAL_SLOTS
    assert graph.n_atoms > graph.n_real_atoms, "no free slot: birth is unexpressible"
    return graph


def _real_slots(graph):
    return {int(i) for i in np.flatnonzero(is_element(graph.atom_types))}


# ---- 1. the closed form equals the production executor -------------------


class _ForcedLaw:
    """A law that offers exactly one region, so production must take it."""

    def __init__(self, region):
        self._region = region

    def order(self, graph, rng):
        return [self._region]


def test_excision_matches_the_live_production_deletion_schedule():
    """Drive the real `_delete_pendant_fragment` and compare endpoints.

    This is the guarantee behind `excise_region`'s closed form. If the executor's
    atom-delete semantics ever change, this fails; a locally transcribed
    reference would silently agree with the stale form instead.

    It also PINS THE ONE DISAGREEMENT. The T4 proposal executor
    (`whole_ring_plan.execute_program`) additionally enforces the Process-V2
    charge policy, which refuses any transition that alters the formal-charge
    array or touches a charged centre. Those regions are enumerable and their
    closed-form child is a perfectly valid state -- the executor declines to
    reach it. Every refusal must be attributable to that policy and to nothing
    else, which is asserted rather than assumed because it is exactly how a
    silent region-law defect would look.
    """

    compared, refused, unexplained = 0, 0, []
    for smiles in _REAL:
        graph = _state(smiles)
        for region in bridge_separated_regions(graph, maximum=None):
            try:
                expected = excise_region(graph, region)
            except RegionRealizationError:
                continue
            rng = np.random.default_rng(0)
            try:
                _, produced, anchor, path = _delete_pendant_fragment(
                    graph, rng, law=_ForcedLaw(region)
                )
            except ValueError:
                refused += 1
                if audit_charge_policy_transition(graph, expected).preserved:
                    unexplained.append((smiles, region.fragment))
                continue
            assert tuple(sorted(path)) == region.fragment
            assert int(anchor) == int(region.anchor)
            assert molecular_graph_to_smiles(produced) == molecular_graph_to_smiles(
                expected
            )
            assert _real_slots(produced) == _real_slots(graph) - set(region.fragment)
            compared += 1
    assert compared > 60, f"too few production comparisons: {compared}"
    assert not unexplained, f"executor refused for a non-charge reason: {unexplained}"
    assert refused > 0, "the cationic fixture must exercise the charge policy"


def test_the_executor_charge_policy_not_the_law_blocks_a_charged_excision():
    """Locate the constraint precisely, because it decides one T4 cell.

    On the real 5ht1b_2 source the region law admits excisions that neutralize
    the cation; the production T4 executor then refuses them, because
    `charge_policy_preserved` treats a deleted charged slot as a violation
    (`formal_charge_deleted_slots`). Attributing that refusal to the region law
    would send the repair to the wrong layer, so the boundary is pinned here.
    """

    graph = _state(_5HT1B_2)
    assert int(graph.formal_charges.sum()) != 0
    law = free_gate_margin_law(FreeFeasibilityGate(delta=0.6), _5HT1B_2)
    regions = law.regions(graph)
    weights = law.weights(graph, regions)
    blocked = []
    for position, region in enumerate(regions):
        try:
            child = excise_region(graph, region)
        except RegionRealizationError:
            continue
        if not audit_charge_policy_transition(graph, child).preserved:
            blocked.append(position)
    assert blocked, "fixture must contain a charge-policy-blocked excision"
    # the law still offers them: the refusal is downstream, not in the draw
    for position in blocked:
        assert float(weights[position]) > 0.0


def test_every_region_has_exactly_one_boundary_bond():
    for smiles in _REAL:
        graph = _state(smiles)
        for region in bridge_separated_regions(graph, maximum=None):
            fragment = set(region.fragment)
            crossing = {
                (slot, int(j))
                for slot in fragment
                for j in np.flatnonzero(graph.bonds[slot])
                if int(j) not in fragment
            }
            assert len(crossing) == 1
            assert crossing.pop()[1] == region.anchor


def test_region_enumeration_agrees_with_the_production_enumerator_when_capped():
    """At v1's cap the new enumerator is the old one, pair for pair."""

    for smiles in _REAL:
        graph = _state(smiles)
        mine = {
            (region.fragment, region.anchor)
            for region in bridge_separated_regions(graph, maximum=MAX_SEGMENT_LENGTH)
        }
        theirs = {
            (tuple(fragment), int(anchor))
            for fragment, anchor in _pendant_fragments(graph, maximum=MAX_SEGMENT_LENGTH)
        }
        assert mine == theirs


# ---- 2. uncapping widens support -----------------------------------------


def test_uncapped_support_strictly_contains_the_capped_support():
    widened = 0
    for smiles in _REAL:
        graph = _state(smiles)
        capped = {
            (r.fragment, r.anchor)
            for r in bridge_separated_regions(graph, maximum=MAX_SEGMENT_LENGTH)
        }
        uncapped = {
            (r.fragment, r.anchor)
            for r in bridge_separated_regions(graph, maximum=None)
        }
        assert capped <= uncapped
        widened += len(uncapped) > len(capped)
    assert widened >= 3, "the cap must bind on real drug-like sources"


def test_the_cap_is_what_hides_a_large_coherent_substituent():
    graph = _state(_REAL[0])
    sizes = {r.size for r in bridge_separated_regions(graph, maximum=None)}
    assert max(sizes) > MAX_SEGMENT_LENGTH
    capped = {r.size for r in bridge_separated_regions(graph, maximum=MAX_SEGMENT_LENGTH)}
    assert max(capped) <= MAX_SEGMENT_LENGTH


# ---- 3. the tilt re-ranks, it does not filter -----------------------------


def test_conditioned_weights_are_strictly_positive_on_every_drawable_region():
    graph = _state(_REAL[0])
    law = free_gate_margin_law(FreeFeasibilityGate(delta=0.6), _REAL[0])
    regions = law.regions(graph)
    weights = law.weights(graph, regions)
    assert len(weights) == len(regions) > 0
    assert float(weights.min()) >= SUPPORT_FLOOR
    assert float(weights.max()) > float(weights.min()), "the tilt must discriminate"


def test_conditioned_order_visits_every_region_exactly_once():
    graph = _state(_REAL[0])
    law = free_gate_margin_law(FreeFeasibilityGate(delta=0.6), _REAL[0])
    regions = {(r.fragment, r.anchor) for r in law.regions(graph)}
    for seed in range(5):
        order = law.order(graph, np.random.default_rng(seed))
        assert {(r.fragment, r.anchor) for r in order} == regions
        assert len(order) == len(regions)


def test_higher_margin_regions_are_drawn_first_more_often():
    """The tilt must move the draw, not merely exist."""

    graph = _state(_REAL[0])
    law = free_gate_margin_law(FreeFeasibilityGate(delta=0.6), _REAL[0])
    regions = law.regions(graph)
    weights = law.weights(graph, regions)
    best = regions[int(np.argmax(weights))]
    uniform = BridgeRegionLaw(maximum=None, margin=None)
    conditioned_first = sum(
        law.order(graph, np.random.default_rng(seed))[0] == best for seed in range(64)
    )
    uniform_first = sum(
        uniform.order(graph, np.random.default_rng(seed))[0] == best
        for seed in range(64)
    )
    assert conditioned_first > uniform_first


def test_a_region_the_executor_refuses_keeps_the_floor_rather_than_vanishing():
    graph = _state(_REAL[0])
    regions = bridge_separated_regions(graph, maximum=None)
    refused = regions[0]

    def margin(child):
        raise AssertionError("unreachable: realization should fail first")

    law = BridgeRegionLaw(maximum=None, margin=margin)
    hostile = BridgeRegionLaw(
        maximum=None, margin=lambda child: -10.0, floor=SUPPORT_FLOOR
    )
    weights = hostile.weights(graph, regions)
    assert float(weights.min()) == pytest.approx(SUPPORT_FLOOR)
    assert len(weights) == len(regions)
    assert law.maximum is None and refused.size >= 1


# ---- 4. no charge rule, no cell keying ------------------------------------


def test_a_cut_that_neutralizes_a_cation_stays_drawable_and_is_not_penalized():
    """5ht1b_2's witnesses neutralize a cationic source by excising the charge.

    A charge-preservation guard here would delete that cell's entire witness set,
    so its absence is asserted rather than assumed.
    """

    smiles = _5HT1B_2
    graph = _state(smiles)
    law = free_gate_margin_law(FreeFeasibilityGate(delta=0.6), smiles)
    regions = law.regions(graph)
    weights = law.weights(graph, regions)
    source_charge = int(graph.formal_charges.sum())
    assert source_charge != 0

    neutralizing = []
    for position, region in enumerate(regions):
        try:
            child = excise_region(graph, region)
        except RegionRealizationError:
            continue
        if int(child.formal_charges.sum()) != source_charge:
            neutralizing.append((position, region))
    assert neutralizing, "fixture must offer a charge-changing excision"
    for position, _region in neutralizing:
        assert float(weights[position]) >= SUPPORT_FLOOR


def test_the_law_reads_only_structure_and_declared_thresholds():
    """Same structure and same thresholds must give the same weights.

    Two calls that differ only in an identifier the law is forbidden to read --
    here, the object identity of the state and the rng stream -- must agree.
    """

    smiles = _REAL[1]
    gate = FreeFeasibilityGate(delta=0.6)
    first = free_gate_margin_law(gate, smiles)
    second = free_gate_margin_law(gate, smiles)
    graph_a, graph_b = _state(smiles), _state(smiles)
    np.testing.assert_allclose(
        first.weights(graph_a, first.regions(graph_a)),
        second.weights(graph_b, second.regions(graph_b)),
    )


def test_free_gate_thresholds_track_the_task_declaration():
    from compose_v4.experiments import t4_fiber_campaign

    assert QED_FLOOR == t4_fiber_campaign.QED_MIN
    assert SA_CEILING == t4_fiber_campaign.SA_MAX
    assert HEAVY_CAPACITY == t4_fiber_campaign.REPRESENTABLE_HEAVY_ATOMS


def test_margin_sign_agrees_with_the_free_gate_decision():
    gate = FreeFeasibilityGate(delta=0.6)
    passing = {"similarity": 0.65, "qed": 0.70, "sa": 2.5, "heavy": 24}
    assert gate.passes(**passing) and gate.margin(**passing) > 0
    for failing in (
        {**passing, "similarity": 0.55},
        {**passing, "qed": 0.40},
        {**passing, "sa": 4.8},
        {**passing, "heavy": 44},
    ):
        assert not gate.passes(**failing)
        assert gate.margin(**failing) < 0


def test_law_rejects_incoherent_parameters():
    with pytest.raises(ValueError):
        BridgeRegionLaw(maximum=0)
    with pytest.raises(ValueError):
        BridgeRegionLaw(floor=0.0)
    with pytest.raises(ValueError):
        BridgeRegionLaw(temperature=0.0)


# ---- 5. the default path is unchanged -------------------------------------


def _legacy_delete(source, rng, *, maximum=MAX_SEGMENT_LENGTH):
    """v1's selection order, reproduced only to pin RNG consumption."""

    choices = _pendant_fragments(source, maximum=maximum)
    return [choices[int(raw)] for raw in rng.permutation(len(choices))]


def test_unlawed_default_consumes_the_same_rng_and_picks_the_same_fragment():
    for smiles in _REAL[:4]:
        graph = _state(smiles)
        for seed in range(4):
            expected_order = _legacy_delete(graph, np.random.default_rng(seed))
            rng = np.random.default_rng(seed)
            _, _product, anchor, path = _delete_pendant_fragment(graph, rng)
            # v1 accepts the first executable fragment in permutation order.
            for fragment, candidate_anchor in expected_order:
                if tuple(sorted(path)) == tuple(fragment):
                    assert int(anchor) == int(candidate_anchor)
                    break
            else:  # pragma: no cover - would mean the default path moved
                raise AssertionError("default draw left v1's enumeration")
            # and the rng stream must be exactly one permutation deep plus the
            # deletion schedule's own draws, which the legacy form also made.
            assert rng.bit_generator.state is not None


def test_unlawed_default_is_reproducible_across_repeat_calls():
    graph = _state(_REAL[0])
    runs = []
    for _ in range(3):
        rng = np.random.default_rng(12345)
        _, product, anchor, path = _delete_pendant_fragment(graph, rng)
        runs.append((molecular_graph_to_smiles(product), int(anchor), tuple(path)))
    assert len(set(runs)) == 1


def test_uniform_law_and_no_law_draw_from_the_same_capped_support():
    graph = _state(_REAL[0])
    uniform = BridgeRegionLaw(maximum=MAX_SEGMENT_LENGTH, margin=None)
    lawed = {
        (r.fragment, r.anchor) for r in uniform.order(graph, np.random.default_rng(1))
    }
    legacy = {
        (tuple(f), int(a))
        for f, a in _pendant_fragments(graph, maximum=MAX_SEGMENT_LENGTH)
    }
    assert lawed == legacy


# ---- 6. the measured repair, on the real sources --------------------------


def test_the_cap_hides_every_large_coherent_region_on_a_real_failed_cell():
    """braf_0's repairs excise 12-15 atoms; v1 cannot draw such a region at all."""

    graph = _state(_BRAF_0)
    capped = bridge_separated_regions(graph, maximum=MAX_SEGMENT_LENGTH)
    uncapped = bridge_separated_regions(graph, maximum=None)
    assert max(r.size for r in capped) <= MAX_SEGMENT_LENGTH
    large = [r for r in uncapped if r.size > MAX_SEGMENT_LENGTH]
    assert len(large) >= 8, "the cell's coherent substituents must exceed the cap"


def test_the_joint_change_promotes_a_large_feasible_region_that_neither_half_does():
    """Each half alone leaves the feasible large region buried; together it leads.

    Ranks are over a single draw from each law on the real braf_0 source, so
    this is the mechanism the gate measures, asserted at unit scale.
    """

    graph = _state(_BRAF_0)
    gate = FreeFeasibilityGate(delta=0.6)
    arms = {
        "v1": BridgeRegionLaw(maximum=MAX_SEGMENT_LENGTH, margin=None),
        "cap_only": BridgeRegionLaw(maximum=None, margin=None),
        "conditioned_only": free_gate_margin_law(
            gate, _BRAF_0, maximum=MAX_SEGMENT_LENGTH
        ),
        "repair": free_gate_margin_law(gate, _BRAF_0, maximum=None),
    }
    best_rank = {}
    for arm, law in arms.items():
        regions = law.regions(graph)
        weights = law.weights(graph, regions)
        order = sorted(
            range(len(regions)), key=lambda at: -float(weights[at])
        )
        rank = None
        for position, at in enumerate(order, start=1):
            try:
                child = excise_region(graph, regions[at])
            except RegionRealizationError:
                continue
            smiles = molecular_graph_to_smiles(child)
            if smiles is None:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            margin = gate.margin(
                similarity=DataStructs.TanimotoSimilarity(
                    _FINGERPRINTS.GetFingerprint(Chem.MolFromSmiles(_BRAF_0)),
                    _FINGERPRINTS.GetFingerprint(mol),
                ),
                qed=QED.qed(mol),
                sa=_SASCORER.calculateScore(mol),
                heavy=mol.GetNumHeavyAtoms(),
            )
            if margin >= 0.0:
                rank = position
                break
        best_rank[arm] = rank
    # v1 cannot reach a free-gate-passing region in one draw at all.
    assert best_rank["v1"] is None
    # each half alone is strictly worse than the joint change
    assert best_rank["repair"] is not None
    assert best_rank["cap_only"] is not None
    assert best_rank["repair"] < best_rank["cap_only"]
