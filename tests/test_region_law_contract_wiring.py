"""The contract field that selects a region-draw law, and proof the runtime reads it.

WHAT THESE TESTS ARE GUARDING
-----------------------------
``bridge_region_law`` is validated on its own; ``dynamic_program_synthesis``
accepts a ``region_law=`` keyword.  Neither fact makes the repair reachable.  A
contract field that is declared and never read is the defect class this
repository has already paid for twice -- a launch receipt reading
``SPAWNED_ALL`` while nothing ran, and a contract declaring
``charged_calls_per_task: 250`` beside a runtime using a module constant of
1000 -- and in both cases a signature check would have passed.

So the load-bearing test here is :func:`test_the_production_shallow_lane_consults_the_contract_law`.
It drives the PRODUCTION ``t4_fiber_campaign.expand`` and observes the draw site
consulting the law through a counter this file owns.  Nothing in the expectation
is recomputed from the code under test: the counter is external, and the
assertion is that it moved.  Removing ``region_law=region_law`` from ``expand``,
from ``synthesize_dynamic_program``, or from ``compile_generic_module`` each
leaves the counter at zero and turns this test red.

:func:`test_the_consumption_check_fails_when_the_draw_ignores_the_law` is its
negative control: a draw that discards the law must make the guard RAISE.  A
guard that cannot fail proves nothing, so the failing direction is asserted too.
"""

from __future__ import annotations

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.bridge_region_law import BridgeRegionLaw, free_gate_margin_law
from compose_v4.control.dynamic_program_synthesis import (
    MAX_SEGMENT_LENGTH,
    compile_generic_module,
    synthesize_dynamic_program,
)
from compose_v4.control.region_law_contract import (
    CONTRACT_FIELD,
    CONTRACT_LANE,
    FREE_GATE_MARGIN_V1,
    RegionLawContractError,
    RegionLawNotConsumed,
    assert_no_unconsumable_region_law_request,
    assert_region_law_is_consumed,
    build_region_law,
    region_law_request,
    resolve_region_law,
)
from compose_v4.experiments.t4_fiber_campaign import Fiber, expand

#: The T4 proposal executor requires 48 slots; a 40-slot editing-corpus state is
#: rejected outright. `t4_fiber_campaign` builds every parent exactly this way.
PROPOSAL_SLOTS = 48

#: braf_0, one of the five cells that terminated at `candidate_exhaustion`. Its
#: known eligible witnesses are 12-15 atom excisions, which v1's cap cannot draw.
_BRAF_0 = "CCN(CC)CCNC(=O)c3cnn4c(c2cccc(NC(=O)Nc1ccc(Cl)c(C(F)(F)F)c1)c2)ccnc34"
_DELTA = 0.6

#: Deterministic seeds. The module order is a permutation over thirteen families
#: and only two route through the law, so one draw may legitimately miss it;
#: fixed seeds keep the result reproducible rather than flaky.
_PROBE_SEEDS = range(16)


def _state(smiles):
    graph = pad_molecular_graph(smiles_to_molecular_graph(smiles), PROPOSAL_SLOTS)
    assert graph.n_atoms > graph.n_real_atoms, "no free slot: birth is unexpressible"
    return graph


def _repair_law():
    from compose_v4.control.bridge_region_law import FreeFeasibilityGate

    return free_gate_margin_law(FreeFeasibilityGate(delta=_DELTA), _BRAF_0, maximum=None)


def _contract(region_law=None, *, lane=CONTRACT_LANE):
    payload = {
        "delta": _DELTA,
        "support": "compose_valid",
        "proposal": {
            "shallow": {"draws": 480, "horizon": 3},
            "anchored_replacement": {"draws": 512, "horizon": 3},
        },
    }
    if region_law is not None:
        payload["proposal"][lane][CONTRACT_FIELD] = region_law
    return payload


class _CountingLaw:
    """A real law that records every time the production draw site consults it.

    The counter lives in this file, so the expectation is not recomputed from
    the code under test: the assertion is simply that the production path moved
    a number only the draw site can move.
    """

    def __init__(self, inner):
        self._inner = inner
        self.consulted = 0
        self.largest_drawn = 0

    def regions(self, graph):
        return self._inner.regions(graph)

    def weights(self, graph, regions):
        return self._inner.weights(graph, regions)

    def order(self, graph, rng):
        self.consulted += 1
        ordered = self._inner.order(graph, rng)
        if ordered:
            self.largest_drawn = max(self.largest_drawn, ordered[0].size)
        return ordered


# ---- 1. reading the field ------------------------------------------------


def test_an_absent_field_is_the_off_state_and_resolves_to_no_law():
    payload = _contract()
    assert region_law_request(payload) is None
    assert resolve_region_law(payload, delta=_DELTA, reference_smiles=_BRAF_0) is None


def test_the_named_law_resolves_to_the_measured_repair_configuration():
    law = resolve_region_law(
        _contract(FREE_GATE_MARGIN_V1), delta=_DELTA, reference_smiles=_BRAF_0
    )
    assert isinstance(law, BridgeRegionLaw)
    # The gate harness measured `free_gate_margin_law(gate, source, maximum=None)`:
    # uncapped support AND a conditioned tilt. Either half alone was insufficient.
    assert law.maximum is None
    assert law.conditioned


def test_a_block_may_pin_the_tilt_the_rescue_runs_under():
    law = build_region_law(
        {"law": FREE_GATE_MARGIN_V1, "maximum": 12, "floor": 0.2, "temperature": 0.25},
        delta=_DELTA,
        reference_smiles=_BRAF_0,
    )
    assert (law.maximum, law.floor, law.temperature) == (12, 0.2, 0.25)


@pytest.mark.parametrize(
    "request_value",
    ["uniform_bounded_v1", "", {"law": "nope"}, {"maximum": None}, 7, ["a"]],
)
def test_an_unhonourable_request_raises_rather_than_degrading_to_v1(request_value):
    with pytest.raises(RegionLawContractError):
        build_region_law(request_value, delta=_DELTA, reference_smiles=_BRAF_0)


def test_an_unknown_tilt_parameter_is_refused():
    with pytest.raises(RegionLawContractError):
        build_region_law(
            {"law": FREE_GATE_MARGIN_V1, "cap": 8},
            delta=_DELTA,
            reference_smiles=_BRAF_0,
        )


def test_the_field_is_refused_on_a_lane_that_cannot_consume_it():
    payload = _contract(FREE_GATE_MARGIN_V1, lane="anchored_replacement")
    with pytest.raises(RegionLawContractError):
        assert_no_unconsumable_region_law_request(payload)


def test_the_field_is_refused_at_the_top_level_where_no_reader_exists():
    payload = _contract()
    payload[CONTRACT_FIELD] = FREE_GATE_MARGIN_V1
    with pytest.raises(RegionLawContractError):
        assert_no_unconsumable_region_law_request(payload)


def test_a_correctly_placed_field_passes_the_placement_guard():
    assert_no_unconsumable_region_law_request(_contract(FREE_GATE_MARGIN_V1))


# ---- 2. the runtime actually consumes it ---------------------------------


def test_the_production_shallow_lane_consults_the_contract_law():
    """THE wiring test: the law reaches the draw site through `expand` itself.

    This fails if any hop drops the keyword -- `expand` ->
    `synthesize_dynamic_program` -> `compile_generic_module` ->
    `_delete_pendant_fragment` -- and it fails if the resolver returns `None`
    for a declared field. It is driven by the production functions, not by a
    transcription of them.
    """

    law = _CountingLaw(_repair_law())
    fiber = Fiber(_BRAF_0, _DELTA, support="compose_valid")
    for seed in _PROBE_SEEDS:
        expand(
            _BRAF_0,
            0.0,
            fiber,
            np.random.default_rng(seed),
            draws=1,
            multi_region=False,
            horizon=3,
            proposal_lane="shallow",
            region_law=law,
        )
        if law.consulted:
            break
    assert law.consulted > 0, (
        "the campaign's shallow lane never consulted the contract's region law; "
        "the field is declared but not reaching the draw site"
    )


def test_the_consumption_guard_reports_success_on_the_production_path():
    fiber = Fiber(_BRAF_0, _DELTA, support="compose_valid")

    def draw(law, attempt):
        return expand(
            _BRAF_0,
            0.0,
            fiber,
            np.random.default_rng(1_000_003 * (attempt + 1)),
            draws=1,
            multi_region=False,
            horizon=3,
            proposal_lane="shallow",
            region_law=law,
        )

    assert assert_region_law_is_consumed(draw) >= 1


def test_the_consumption_check_fails_when_the_draw_ignores_the_law():
    """The negative control. A guard that cannot fail is not a guard."""

    def deaf_draw(law, attempt):
        return []

    with pytest.raises(RegionLawNotConsumed):
        assert_region_law_is_consumed(deaf_draw, attempts=3)


def test_a_law_is_refused_on_a_campaign_lane_that_cannot_thread_it():
    fiber = Fiber(_BRAF_0, _DELTA, support="compose_valid")
    with pytest.raises(ValueError, match="only consumable on the 'shallow' lane"):
        expand(
            _BRAF_0,
            0.0,
            fiber,
            np.random.default_rng(0),
            draws=1,
            proposal_lane="anchored_replacement",
            region_law=_repair_law(),
        )


# ---- 3. the law changes what the production draw site can reach -----------


def test_only_the_contract_law_lets_the_production_module_excise_past_the_v1_cap():
    """OFF can never draw a region larger than eight atoms; ON does.

    Both arms call the PRODUCTION `compile_generic_module` on the real braf_0
    source, so this is a property of the shipped draw site rather than of the
    law object. It is the mechanism the gate measured, asserted at unit scale:
    every known eligible witness for this cell is a 12-15 atom excision.
    """

    graph = _state(_BRAF_0)
    off_largest, on_largest = 0, 0
    for seed in _PROBE_SEEDS:
        for law, sink in ((None, "off"), (_repair_law(), "on")):
            probe = None if law is None else _CountingLaw(law)
            try:
                product, _ = compile_generic_module(
                    graph,
                    np.random.default_rng(seed),
                    "substituent_delete",
                    region_law=probe,
                )
            except ValueError:
                continue
            # Heavy-atom loss across the executed module: the size of the region
            # the draw site actually excised, read off the product state rather
            # than off any bookkeeping the module reports about itself.
            removed = int(graph.n_real_atoms) - int(product.n_real_atoms)
            if sink == "off":
                off_largest = max(off_largest, removed)
            else:
                on_largest = max(on_largest, removed)
    assert off_largest <= MAX_SEGMENT_LENGTH, "v1 cannot exceed its own cap"
    assert on_largest > MAX_SEGMENT_LENGTH, (
        "the contract law must make a coherent substituent larger than the v1 cap "
        "drawable in one module"
    )


# ---- 4. off stays byte-identical -----------------------------------------


def test_passing_the_off_value_explicitly_does_not_perturb_the_rng_stream():
    """`region_law=None` must be the same draw as not passing it at all.

    Byte-identity is why the field can default OFF: `_delete_pendant_fragment`
    consumes `rng.permutation` when unlawed and `rng.random` under any law, so
    an "off" that installed a uniform law would move every existing run.
    """

    graph = _state(_BRAF_0)
    bare = synthesize_dynamic_program(graph, np.random.default_rng(11), max_modules=3)
    explicit = synthesize_dynamic_program(
        graph, np.random.default_rng(11), max_modules=3, region_law=None
    )
    assert bare[3]["states"] == explicit[3]["states"]
    assert bare[3]["actions"] == explicit[3]["actions"]


def test_the_campaign_off_path_is_unchanged_by_the_new_keyword():
    fiber = Fiber(_BRAF_0, _DELTA, support="compose_valid")
    bare = expand(_BRAF_0, 0.0, fiber, np.random.default_rng(5), draws=3)
    explicit = expand(
        _BRAF_0, 0.0, fiber, np.random.default_rng(5), draws=3, region_law=None
    )
    assert [row["smiles"] for row in bare] == [row["smiles"] for row in explicit]
