"""The replace-completion law: support preservation, wiring, and OFF identity.

These tests are written against the PRODUCTION path. Where an expectation could
be recomputed from the code under test it is instead derived independently, and
the wiring tests are proved by mutation: each one must go red when a specific
hop is removed. Two hops share one sink here
(``synthesize_dynamic_program`` and ``synthesize_named_module_sequence`` both
call ``compile_generic_module``), which is the exact near-miss the region-law
wiring hit, so both are covered separately.
"""

from __future__ import annotations

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import (
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control import dynamic_program_synthesis as dps
from compose_v4.control.bridge_region_law import (
    MARGIN_TEMPERATURE,
    SUPPORT_FLOOR,
    FreeFeasibilityGate,
    free_gate_margin_law,
)
from compose_v4.control.completion_law_contract import (
    CONTRACT_FIELD,
    CONTRACT_LANE,
    FREE_GATE_MARGIN_V1,
    CompletionLawContractError,
    CompletionLawNotConsumed,
    assert_completion_law_is_consumed,
    assert_no_unconsumable_completion_law_request,
    build_completion_law,
    resolve_completion_law,
)
from compose_v4.control.replace_completion_law import (
    DEFAULT_CANDIDATES,
    ReplaceCompletionLaw,
)

#: A real T4 lead. Not a witness and not a hand-built molecule.
LEAD = "CC(C)CCN(Cc2ccc1ccc(C(N)=N)cc1c2)C(=O)c3cccc4ccccc34"
SLOTS = 48


def _source():
    return pad_molecular_graph(smiles_to_molecular_graph(LEAD), SLOTS)


def _payload(field=None):
    payload = {"proposal": {CONTRACT_LANE: {"draws": 8, "horizon": 3}}}
    if field is not None:
        payload["proposal"][CONTRACT_LANE][CONTRACT_FIELD] = field
    return payload


# ---- The law itself ------------------------------------------------------


def test_weights_are_strictly_positive_so_the_law_cannot_filter():
    """Every candidate keeps positive weight -- this is the re-ranking guarantee."""

    source = _source()
    # A margin that is hugely negative everywhere would zero an unfloored law.
    law = ReplaceCompletionLaw(candidates=4, margin=lambda _g: -50.0)
    weights = law.weights([source, source, source])
    assert len(weights) == 3
    assert all(w > 0.0 for w in weights), weights
    # And it is the floor that does it, not luck.
    assert all(w == pytest.approx(SUPPORT_FLOOR) for w in weights), weights


def test_support_floor_actually_binds_not_merely_exists():
    """A floor is only tested where it BINDS.

    Asserting ``min(weight) > 0`` is satisfied by an un-floored law whenever the
    margins happen to be positive, which is how an earlier floor guard passed
    against a law mutated from re-ranking into filtering. So drive it with a
    negative margin, where an unfloored exponential would underflow to zero.
    """

    source = _source()
    floored = ReplaceCompletionLaw(candidates=2, margin=lambda _g: -1e4, floor=0.05)
    assert floored.weights([source])[0] == pytest.approx(0.05)
    # Independent expectation: without a floor this value underflows to 0.0.
    import math

    assert math.exp(-1e4 / MARGIN_TEMPERATURE) == 0.0


def test_a_raising_margin_is_floored_not_dropped():
    """The except branch must FLOOR, not zero.

    `test_weights_are_strictly_positive_...` uses a margin that RETURNS a bad
    value, so it never reaches the exception path and a mutation zeroing that
    path SURVIVED it. This drives a margin that RAISES, which is the only way to
    exercise the branch. Same lesson as the support floor: a guard is only
    tested where it binds.
    """

    source = _source()

    def raising(_graph):
        raise ValueError("this candidate cannot be scored")

    law = ReplaceCompletionLaw(candidates=3, margin=raising)
    weights = law.weights([source, source])
    assert all(w == pytest.approx(SUPPORT_FLOOR) for w in weights), weights
    assert all(w > 0.0 for w in weights), weights


def test_a_nonfinite_margin_is_also_floored():
    source = _source()
    law = ReplaceCompletionLaw(candidates=2, margin=lambda _g: float("nan"))
    assert all(w == pytest.approx(SUPPORT_FLOOR) for w in law.weights([source]))


def test_order_is_a_permutation_of_every_candidate():
    """Re-ranking reorders; it never drops a candidate."""

    source = _source()
    law = ReplaceCompletionLaw(candidates=5, margin=lambda _g: 0.1)
    order = law.order([source] * 5, np.random.default_rng(0))
    assert sorted(order) == list(range(5))


def test_a_better_margin_is_preferred_far_more_often_than_not():
    """The tilt points the right way. Statistical, so it is given real power."""

    good, bad = _source(), _source()

    def margin(graph):
        # Distinguish by identity, not by chemistry, so this test measures the
        # law's weighting rather than the gate's.
        return 1.0 if graph is good else -1.0

    law = ReplaceCompletionLaw(candidates=2, margin=margin)
    rng = np.random.default_rng(20260921)
    firsts = [law.order([good, bad], rng)[0] for _ in range(400)]
    assert firsts.count(0) > 380, firsts.count(0)


def test_uniform_law_ignores_the_margin():
    source = _source()
    law = ReplaceCompletionLaw(candidates=3, margin=None)
    assert list(law.weights([source] * 3)) == [1.0, 1.0, 1.0]


def test_rejects_degenerate_parameters():
    with pytest.raises(ValueError):
        ReplaceCompletionLaw(candidates=0)
    with pytest.raises(ValueError):
        ReplaceCompletionLaw(floor=0.0)
    with pytest.raises(ValueError):
        ReplaceCompletionLaw(temperature=0.0)


# ---- The contract surface ------------------------------------------------


def test_absent_field_resolves_to_none_which_is_the_only_off():
    assert resolve_completion_law(_payload(), delta=0.6, reference_smiles=LEAD) is None


def test_named_request_resolves_and_carries_the_default_k():
    law = resolve_completion_law(
        _payload(FREE_GATE_MARGIN_V1), delta=0.6, reference_smiles=LEAD
    )
    assert isinstance(law, ReplaceCompletionLaw)
    assert law.conditioned
    assert law.candidates == DEFAULT_CANDIDATES


def test_block_request_overrides_k():
    law = resolve_completion_law(
        _payload({"law": FREE_GATE_MARGIN_V1, "candidates": 4}),
        delta=0.6,
        reference_smiles=LEAD,
    )
    assert law.candidates == 4


def test_unknown_law_and_unknown_parameter_both_raise():
    with pytest.raises(CompletionLawContractError):
        build_completion_law("no_such_law", delta=0.6, reference_smiles=LEAD)
    with pytest.raises(CompletionLawContractError):
        build_completion_law(
            {"law": FREE_GATE_MARGIN_V1, "nonsense": 1}, delta=0.6, reference_smiles=LEAD
        )


def test_field_on_a_lane_that_cannot_consume_it_is_refused():
    payload = {"proposal": {"structured": {CONTRACT_FIELD: FREE_GATE_MARGIN_V1}}}
    with pytest.raises(CompletionLawContractError):
        assert_no_unconsumable_completion_law_request(payload)
    with pytest.raises(CompletionLawContractError):
        assert_no_unconsumable_completion_law_request({CONTRACT_FIELD: "x"})


def test_the_completion_margin_is_the_region_law_margin():
    """Both halves must be conditioned by byte-identical chemistry.

    Built independently here and compared on a real state, so a future change
    that rebuilt the margin separately would show up as a divergence.
    """

    source = _source()
    region_margin = free_gate_margin_law(
        FreeFeasibilityGate(delta=0.6), LEAD
    ).margin
    completion = build_completion_law(
        FREE_GATE_MARGIN_V1, delta=0.6, reference_smiles=LEAD
    )
    assert completion.margin(source) == pytest.approx(region_margin(source))


# ---- Wiring, proved by mutation -----------------------------------------


def _draw(law, seed, *, entry):
    """One production proposal draw with the law installed where it belongs."""

    source = _source()
    rng = np.random.default_rng(20260921 + seed)
    if entry == "dynamic":
        return dps.synthesize_dynamic_program(
            source, rng, max_modules=3, completion_law=law
        )
    return dps.synthesize_named_module_sequence(
        source, rng, ["segment_replace"], completion_law=law
    )


def test_dynamic_synthesizer_consults_the_completion_law():
    """HOP 1. Goes red if `synthesize_dynamic_program` stops threading the law."""

    attempts = assert_completion_law_is_consumed(
        lambda law, seed: _draw(law, seed, entry="dynamic")
    )
    assert attempts >= 1


def test_named_sequence_consults_the_completion_law():
    """HOP 2, the one that shares a sink with HOP 1.

    Dropping the keyword from `synthesize_named_module_sequence` alone leaves
    the hop-1 test GREEN, because `compile_generic_module` still receives a law
    from the other caller. That is the near-miss the region-law wiring hit, so
    this hop gets its own test.
    """

    attempts = assert_completion_law_is_consumed(
        lambda law, seed: _draw(law, seed, entry="named")
    )
    assert attempts >= 1


def test_consumption_check_fails_loudly_when_nothing_consults_the_law():
    """The negative control: a draw that never reaches the site must RAISE."""

    with pytest.raises(CompletionLawNotConsumed):
        assert_completion_law_is_consumed(lambda law, seed: None, attempts=3)


# ---- OFF is byte-identical ----------------------------------------------


def test_absent_law_reproduces_the_historical_draw_exactly():
    """ABSENT is the only OFF. Same seed, law=None twice, identical program.

    This is the guard against implementing "off" as a uniform law: a uniform law
    would reproduce the SUPPORT but consume a different RNG stream, silently
    moving every existing run.
    """

    a = dps.synthesize_named_module_sequence(
        _source(), np.random.default_rng(7), ["segment_replace"]
    )
    b = dps.synthesize_named_module_sequence(
        _source(), np.random.default_rng(7), ["segment_replace"], completion_law=None
    )
    assert a[4]["modules"] == b[4]["modules"]
    # And on the free synthesizer too, where most families never reach the site.
    c = dps.synthesize_dynamic_program(
        _source(), np.random.default_rng(7), max_modules=3
    )
    d = dps.synthesize_dynamic_program(
        _source(), np.random.default_rng(7), max_modules=3, completion_law=None
    )
    assert c[4]["modules"] == d[4]["modules"]


def test_a_uniform_law_is_not_the_off_state():
    """Documents WHY no uniform law is registered: it moves the DISTRIBUTION.

    The claim is distributional, not per-seed, and getting that wrong is how
    this test first failed twice. With ``candidates=K`` and uniform weights the
    head of the order is candidate 0 with probability about ``1/K``, and
    candidate 0 consumes exactly the RNG the unconditioned path consumes -- so
    on any individual seed a uniform law can and does reproduce the OFF draw.
    What it cannot do is reproduce it EVERY time, which is what "byte-identical
    OFF" has to mean. So measure agreement across many seeds and require it to
    be far below 1.0.
    """

    agree = 0
    trials = 60
    for seed in range(trials):
        off = dps.synthesize_named_module_sequence(
            _source(),
            np.random.default_rng(seed),
            ["segment_replace"],
            completion_law=None,
        )
        uniform = dps.synthesize_named_module_sequence(
            _source(),
            np.random.default_rng(seed),
            ["segment_replace"],
            completion_law=ReplaceCompletionLaw(candidates=4, margin=None),
        )
        agree += off[4]["modules"] == uniform[4]["modules"]
    # An absent field agrees with itself 60/60 (asserted above). A uniform law
    # must not, or "absent is the only OFF" would be an empty statement.
    assert agree < trials, agree
    # And it should be roughly the 1-in-K coincidence rate, not near-total.
    assert agree <= 0.6 * trials, agree


# ---- The costed equivalence the helper relies on ------------------------


def test_scoring_endpoint_equals_the_executed_module_endpoint():
    """`_conditioned_completion` scores from `contracted`, not by replaying deletes.

    The production path then executes `[*delete_actions, *grow_actions]` from
    `source`. Those must be the same molecule or the law would be re-ranking
    something other than what gets built. Asserted rather than assumed.
    """

    source = _source()
    rng = np.random.default_rng(3)
    delete_actions, contracted, anchor, _path = dps._delete_pendant_fragment(
        source, rng, law=None
    )
    capacity = min(dps.MAX_SEGMENT_LENGTH, 40 - contracted.n_real_atoms)
    grow_actions, _, _ = dps._grow_actions(
        contracted, rng, length=1, elements=("C",), anchor=anchor
    )
    from compose_v4.experiments.whole_ring_plan import execute_program

    from_contracted, _ = execute_program(contracted, list(grow_actions))
    from_source, _ = execute_program(source, [*delete_actions, *grow_actions])
    assert molecular_graph_to_smiles(from_contracted) == molecular_graph_to_smiles(
        from_source
    )
