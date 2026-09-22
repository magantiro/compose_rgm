"""The learned chemical prior must reach the CONSTRUCTION draw, per family.

A prior that is accepted by a signature and dropped one hop later is inert, and
a module whose tests all pass proves nothing about that: the region law shipped
measured, tested and unreachable because no production caller passed it.  These
tests therefore

* drive the REAL production entry points (``synthesize_dynamic_program``,
  ``compile_generic_module``, ``DynamicProgramOptimizer._mutate``) rather than a
  transcription of them;
* assert CONSUMPTION with a probe that raises a ``BaseException`` from inside
  ``order``, because ``synthesize_dynamic_program`` catches ``ValueError`` per
  family and would swallow anything ordinary;
* assert BEHAVIOUR separately for every construction family, because
  ``segment_grow``, ``functionalize`` and the grow half of ``segment_replace``
  share one sink (``_grow_actions``) and a single consultation test stays green
  when only one of the three hops is dropped;
* assert the support is a PERMUTATION of the uniform law's candidate set, so the
  prior can never become a filter;
* assert ``successor_prior=None`` consumes the historical RNG stream exactly,
  which is what "absent is byte-identical off" means operationally.

No torch and no checkpoint: the arms are separated by stub priors with declared
preferences, so a failure is a wiring defect rather than a model result.
"""

from __future__ import annotations

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import (
    ELEMENT_TO_IDX,
    IDX_TO_ELEMENT,
    is_element,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.dynamic_program_synthesis import (
    CONSTRUCTION_LAW_TAG,
    DynamicProgramOptimizer,
    _grow_actions,
    _terminal_shrink,
    compile_generic_module,
    synthesize_dynamic_program,
    synthesize_named_module_sequence,
)
from compose_v4.control.learned_successor_prior import (
    PriorNotConsumed,
    assert_prior_is_consumed,
)

SOURCE_SMILES = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"


def _source(smiles: str = SOURCE_SMILES):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), 48)


class _ForcedPrior:
    """A prior whose ranking is declared rather than learned.

    ``order`` is a pure re-ranking of the candidate list it is handed, exactly as
    the production prior's is, so a family that consults it produces a declared
    outcome and a family that drops it does not.
    """

    def __init__(self, *, element: str | None = None, prefer_high_vertex: bool = False):
        self.element = element
        self.prefer_high_vertex = prefer_high_vertex
        self.seen: list[tuple[str, int]] = []

    def _rank(self, family, action):
        if family == "atom_insert":
            wanted = (
                self.element is not None
                and int(action.atom_type) == ELEMENT_TO_IDX[self.element]
            )
            site = int(action.neighbors[0][0]) if action.neighbors else -1
            return (not wanted, -site if self.prefer_high_vertex else site)
        if family == "atom_delete":
            v = int(action.v)
            return (False, -v if self.prefer_high_vertex else v)
        return (False, 0)

    def order(self, source, rng, *, family, actions):
        candidates = tuple(actions)
        self.seen.append((family, len(candidates)))
        return tuple(sorted(candidates, key=lambda a: self._rank(family, a)))

    def weights(self, source, *, family, actions):
        n = len(tuple(actions))
        return np.full(n, 1.0 / n, dtype=float)

    def probability_of(self, source, *, family, actions, action):
        return 1.0 / len(tuple(actions))


# ---- Consumption: the production path must reach the prior ----


def test_synthesize_dynamic_program_consults_the_prior():
    source = _source()

    def draw(prior, seed):
        synthesize_dynamic_program(
            source,
            np.random.default_rng(np.random.SeedSequence([seed, 11])),
            max_modules=3,
            successor_prior=prior,
        )

    assert assert_prior_is_consumed(draw, attempts=24) >= 1


def test_named_module_sequence_consults_the_prior():
    source = _source()

    def draw(prior, seed):
        synthesize_named_module_sequence(
            source,
            np.random.default_rng(np.random.SeedSequence([seed, 5])),
            ("segment_grow",),
            successor_prior=prior,
        )

    assert assert_prior_is_consumed(draw, attempts=8) >= 1


class _StubConfig:
    max_primitives = 32
    max_blocks = 8


class _StubOptimizer(DynamicProgramOptimizer):
    """The smallest object ``DynamicProgramOptimizer._mutate`` needs.

    ``_mutate`` is called UNBOUND on exactly this shape by
    ``pmo_online_memory`` and ``dynamic_program_synthesis_v21``, so driving it
    this way runs the production method body, not a copy of it.
    """

    def __init__(self, rng, prior):
        self.rng = rng
        self.config = _StubConfig()
        self.construction_prior = prior

    def _continuation_lineage(self, entry):
        return {}


def _entry(source):
    from compose_v4.rewrite.trace_shard import encode_state

    return {"trace": {"states": [encode_state(source)]}}


def test_the_optimizer_mutate_hop_passes_its_prior_to_synthesis():
    """The hop the production controller actually uses, driven for real."""

    source = _source()
    entry = _entry(source)

    def draw(prior, seed):
        # A seed whose first `random()` falls below FRESH_SYNTHESIS_PROBABILITY,
        # so the method takes the fresh-synthesis branch rather than super()'s.
        rng = np.random.default_rng(np.random.SeedSequence([seed, 97]))
        while True:
            state = rng.bit_generator.state
            if rng.random() < 0.5:
                rng.bit_generator.state = state
                break
        DynamicProgramOptimizer._mutate(_StubOptimizer(rng, prior), entry)

    assert assert_prior_is_consumed(draw, attempts=24) >= 1


def test_the_optimizer_defaults_to_no_prior():
    assert DynamicProgramOptimizer.construction_prior is None


def test_the_consumption_probe_can_fail():
    """A negative control: a draw that never consults must be reported unwired."""

    with pytest.raises(PriorNotConsumed):
        assert_prior_is_consumed(lambda prior, seed: None, attempts=3)


# ---- Behaviour: every construction family separately ----


@pytest.mark.parametrize(
    "family, read_elements",
    [
        ("segment_grow", lambda p: list(p["elements"])),
        ("functionalize", lambda p: [p["element"]]),
        ("segment_replace", lambda p: list(p["elements"])),
    ],
)
def test_each_growth_family_builds_the_element_its_prior_ranks_first(
    family, read_elements
):
    """Three hops share ``_grow_actions``; each is bound on its own here."""

    source = _source()
    for element in ("N", "O"):
        prior = _ForcedPrior(element=element)
        built = None
        for seed in range(40):
            try:
                _product, stage = compile_generic_module(
                    source,
                    np.random.default_rng(np.random.SeedSequence([seed, 23])),
                    family,
                    successor_prior=prior,
                )
            except ValueError:
                continue
            built = read_elements(stage["parameters"])
            break
        assert built, f"{family} never compiled a growth under the prior"
        assert set(built) == {element}, f"{family} ignored its prior: {built}"


def test_segment_shrink_deletes_the_terminal_atom_its_prior_ranks_first():
    source = _source()
    prior = _ForcedPrior(prefer_high_vertex=True)
    _product, stage = compile_generic_module(
        source,
        np.random.default_rng(np.random.SeedSequence([3, 29])),
        "segment_shrink",
        successor_prior=prior,
    )
    terminal = [
        int(i)
        for i in np.flatnonzero(is_element(source.atom_types))
        if int(np.count_nonzero(source.bonds[i])) == 1
    ]
    assert stage["parameters"]["path"][0] == max(terminal)
    assert ("atom_delete", len(terminal)) in prior.seen


def test_carbonyl_insert_attaches_where_its_prior_ranks_first():
    source = _source()
    anchors = [
        int(i)
        for i in np.flatnonzero(is_element(source.atom_types))
        if int(source.atom_types[i]) == ELEMENT_TO_IDX["C"]
        and int(source.implicit_h_counts[i]) >= 2
    ]
    assert len(anchors) > 1, "the fixture must offer a real choice of anchor"
    for prefer_high in (False, True):
        prior = _ForcedPrior(prefer_high_vertex=prefer_high)
        _product, stage = compile_generic_module(
            source,
            np.random.default_rng(np.random.SeedSequence([7, 31])),
            "carbonyl_insert",
            successor_prior=prior,
        )
        expected = max(anchors) if prefer_high else min(anchors)
        assert stage["parameters"]["anchor"] == expected


def test_the_local_families_record_the_learned_law():
    """``_local_module`` must forward the prior into ``current_state_program``."""

    source = _source()
    _product, stage = compile_generic_module(
        source,
        np.random.default_rng(np.random.SeedSequence([1, 37])),
        "heteroatom_substitute",
        successor_prior=_ForcedPrior(),
    )
    assert "learned editing model" in stage["parameters"]["proposal_law"]


def test_the_first_growth_step_ranks_placement_and_element_jointly():
    """Placement is the measured signal; a prior that cannot see it is inert."""

    source = _source()
    prior = _ForcedPrior(element="O")
    _grow_actions(
        source,
        np.random.default_rng(0),
        length=2,
        elements=("C", "N", "O"),
        successor_prior=prior,
    )
    anchors = [
        int(i)
        for i in np.flatnonzero(is_element(source.atom_types))
        if int(source.implicit_h_counts[i]) >= 1
    ]
    assert prior.seen[0] == ("atom_insert", len(anchors) * 3)
    # The chain is linear, so the second step offers the elements only.
    assert prior.seen[1] == ("atom_insert", 3)


# ---- Support identity: a re-ranking, never a filter ----


class _RecordingPrior(_ForcedPrior):
    def __init__(self):
        super().__init__()
        self.support: list[tuple[frozenset, frozenset]] = []

    def order(self, source, rng, *, family, actions):
        candidates = tuple(actions)
        ordered = super().order(source, rng, family=family, actions=candidates)
        self.support.append(
            (frozenset(map(repr, candidates)), frozenset(map(repr, ordered)))
        )
        assert len(ordered) == len(candidates)
        return ordered


def test_the_prior_returns_a_permutation_of_the_candidate_set():
    source = _source()
    prior = _RecordingPrior()
    for family in ("segment_grow", "functionalize", "segment_shrink", "carbonyl_insert"):
        for seed in range(6):
            try:
                compile_generic_module(
                    source,
                    np.random.default_rng(np.random.SeedSequence([seed, 41])),
                    family,
                    successor_prior=prior,
                )
            except ValueError:
                continue
    assert prior.support, "no construction draw reached the prior"
    for offered, returned in prior.support:
        assert offered == returned


# ---- Absent is byte-identical off ----


def test_absent_prior_consumes_the_historical_rng_stream_in_growth():
    """One anchor draw, then one element draw per inserted atom -- and nothing else."""

    source = _source()
    length, elements = 3, ("C", "N", "O")
    anchors = [
        int(i)
        for i in np.flatnonzero(is_element(source.atom_types))
        if int(source.implicit_h_counts[i]) >= 1
    ]
    live = np.random.default_rng(np.random.SeedSequence([20260921, 2]))
    _grow_actions(source, live, length=length, elements=elements)

    reference = np.random.default_rng(np.random.SeedSequence([20260921, 2]))
    reference.integers(len(anchors))
    for _ in range(length):
        reference.integers(len(elements))
    assert live.bit_generator.state == reference.bit_generator.state


def test_absent_prior_consumes_the_historical_rng_stream_in_shrink():
    source = _source()
    live = np.random.default_rng(np.random.SeedSequence([20260921, 4]))
    actions, _current, _anchor, path = _terminal_shrink(
        source, live, requested_length=1
    )
    terminal = [
        int(i)
        for i in np.flatnonzero(is_element(source.atom_types))
        if int(np.count_nonzero(source.bonds[i])) == 1
    ]
    reference = np.random.default_rng(np.random.SeedSequence([20260921, 4]))
    reference.integers(len(terminal))
    assert len(actions) == 1 and len(path) == 1
    assert live.bit_generator.state == reference.bit_generator.state


def test_absent_prior_leaves_no_construction_law_tag():
    """ABSENT is the only off, so an untagged artifact is provably v1."""

    source = _source()
    _source_out, _program, _binding, _trace, metadata = synthesize_dynamic_program(
        source, np.random.default_rng(np.random.SeedSequence([5, 13])), max_modules=2
    )
    assert "construction_law" not in metadata
    for module in metadata["modules"]:
        assert "construction_law" not in module["parameters"]


def test_a_prior_shaped_proposal_is_tagged_at_synthesis_time():
    source = _source()
    _s, _p, _b, _t, metadata = synthesize_dynamic_program(
        source,
        np.random.default_rng(np.random.SeedSequence([5, 13])),
        max_modules=2,
        successor_prior=_ForcedPrior(),
    )
    assert metadata["construction_law"] == CONSTRUCTION_LAW_TAG


# ---- The family choice is NOT weighted by the prior ----


def test_the_module_family_choice_never_consults_the_prior():
    """Measured: the model puts 0.38 on atom_restate against 0.063 on cycle_close,
    and cycle_close is the one family whose edits GAIN drug-likeness.  Family
    selection must therefore stay uniform."""

    from compose_v4.control import dynamic_program_synthesis as module
    import inspect

    text = inspect.getsource(module._weighted_module_order)
    assert "prior" not in text
    assert "successor_prior" not in inspect.signature(
        module._weighted_module_order
    ).parameters


def test_the_element_name_round_trips_through_the_prior_branch():
    """The ON branch reads its element back off the executed action."""

    source = _source()
    prior = _ForcedPrior(element="N")
    _actions, _at, chosen = _grow_actions(
        source,
        np.random.default_rng(1),
        length=2,
        elements=("C", "N", "O"),
        successor_prior=prior,
    )
    assert chosen == ["N", "N"]
    assert IDX_TO_ELEMENT[ELEMENT_TO_IDX["N"]] == "N"
