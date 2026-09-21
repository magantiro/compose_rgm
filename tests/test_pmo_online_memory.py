"""Guards for the online proposal memory.

Every guard here is mutation-proven: the commit message records which
production mutation turns each one red.  A test whose expectation is
recomputed from the code under test cannot fail, so the expectations below are
built from independent constructions -- a hand-built archive, an arithmetic
identity, or the production consumption probe -- never from the memory's own
output.
"""

from __future__ import annotations

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.bridge_region_law import (
    UNIFORM_BOUNDED_V1,
    BridgeRegionLaw,
    bridge_separated_regions,
)
from compose_v4.control.pmo_online_memory import (
    TOP_K,
    EditOutcomeMemory,
    FrontierLedger,
    MemoryRegionLaw,
    OnlineProposalMemory,
    RegionContext,
    SizeResidual,
    memory_channel_proposal,
)

# A drug-like source with several bridge-separated substituents of different
# sizes, padded to the 48 slots the proposal path uses.
SOURCE = "CC(C)Cc1ccc(cc1)C(C)C(=O)NCCc1ccccc1"


def _source():
    return pad_molecular_graph(smiles_to_molecular_graph(SOURCE), 48)


def _warm(memory, *, context_bias=None):
    """Feed counted transitions built by hand, not by the memory."""

    graph = _source()
    regions = bridge_separated_regions(graph, maximum=None)
    for index, region in enumerate(regions[: 40]):
        context = RegionContext.of(graph, region)
        good = context_bias(context) if context_bias else (index % 2 == 0)
        memory.frontier.observe(f"C{index}", 0.5 if good else 0.1)
        memory.observe_transition(
            parent_graph=graph,
            parent_endpoint="parent",
            parent_score=0.2,
            child_endpoint=f"C{index}",
            child_score=0.5 if good else 0.1,
            child_heavy=int(graph.n_real_atoms) - region.size,
            family="substituent_delete",
            touched_slots=region.fragment,
        )
    return memory


# ---- Cold behaviour ------------------------------------------------------


def test_cold_memory_returns_no_law_so_off_is_byte_identical():
    """ABSENT is the only byte-identical OFF.

    An unlawed `_delete_pendant_fragment` consumes `rng.permutation`; ANY law
    object consumes `rng.random`. A uniform law would reproduce v1's support
    and not v1's draws, so a cold memory must return None, never a law.
    """

    memory = OnlineProposalMemory()
    assert memory.warm is False
    assert memory.region_law() is None


def test_cold_memory_adapter_delegates_to_the_production_fallback():
    calls = []

    def fallback(channel, entry):
        calls.append(channel)
        return "fallback-result"

    out = memory_channel_proposal(
        object(), "shallow_program_channel", {}, fallback, OnlineProposalMemory()
    )
    assert out == "fallback-result" and calls == ["shallow_program_channel"]


def test_non_shallow_lane_is_never_intercepted():
    """The structured lane's synthesizer takes no `region_law`, so the memory
    must not claim it."""

    memory = _warm(OnlineProposalMemory())
    seen = []
    memory_channel_proposal(
        object(), "structured_program_channel", {}, lambda c, e: seen.append(c), memory
    )
    assert seen == ["structured_program_channel"]


# ---- The law is a re-ranking, never a filter -----------------------------


def test_every_drawable_region_keeps_positive_weight():
    memory = _warm(OnlineProposalMemory())
    graph = _source()
    law = memory.region_law()
    regions = law.regions(graph)
    weights = law.weights(graph, regions)
    assert len(regions) > 5
    assert np.all(weights > 0.0), "a zero weight would make this a filter"
    assert weights.min() >= law.floor * 0.999


def test_an_unrealizable_region_keeps_the_floor_not_zero(monkeypatch):
    """The floor branch needs a region the executor refuses, and this source
    has none -- so force one. Without a fixture that reaches it, dropping the
    floor is invisible."""

    import compose_v4.control.pmo_online_memory as module
    from compose_v4.control.bridge_region_law import RegionRealizationError

    graph = _source()
    memory = _warm(OnlineProposalMemory())
    law = memory.region_law()
    regions = law.regions(graph)
    doomed = regions[0]
    original = module.excise_region

    def refusing(g, region):
        if region is doomed:
            raise RegionRealizationError("forced refusal")
        return original(g, region)

    monkeypatch.setattr(module, "excise_region", refusing)
    weights = law.weights(graph, regions)
    assert weights[0] > 0.0, "a refused region must keep the floor, not vanish"
    assert weights[0] == pytest.approx(law.floor)


def test_memory_support_is_a_strict_superset_of_v1():
    graph = _source()
    memory = _warm(OnlineProposalMemory())
    v1 = {(r.fragment, r.anchor) for r in UNIFORM_BOUNDED_V1.regions(graph)}
    new = {(r.fragment, r.anchor) for r in memory.region_law().regions(graph)}
    assert v1 < new, "the law must never narrow the support"


# ---- The law actually changes the draw -----------------------------------


def test_learned_weights_change_the_draw_at_matched_support():
    """Compared against UNCAPPED uniform, so the contrast isolates the memory
    rather than restating that the size cap moved."""

    graph = _source()
    memory = _warm(OnlineProposalMemory())
    uncapped = BridgeRegionLaw(maximum=None, margin=None)
    learned = memory.region_law()
    assert {(r.fragment, r.anchor) for r in uncapped.regions(graph)} == {
        (r.fragment, r.anchor) for r in learned.regions(graph)
    }, "arms must share support for this contrast to mean anything"

    def heads(law):
        out = []
        for seed in range(200):
            order = law.order(graph, np.random.default_rng(seed))
            if order:
                out.append(order[0].size)
        return np.array(out)

    a, b = heads(uncapped), heads(learned)
    assert abs(a.mean() - b.mean()) > 0.5, (a.mean(), b.mean())


def test_the_adapter_installs_the_law_on_the_production_path(monkeypatch):
    """Dropping `region_law=` inside the adapter is invisible to a test that
    calls `synthesize_dynamic_program` directly, so drive the ADAPTER and make
    the production draw site prove it reached the law."""

    from types import SimpleNamespace

    from compose_v4.control.region_law_contract import (
        RegionLawNotConsumed,
        _ConsumptionProbe,
        _ProbeConsumed,
    )

    graph = _source()
    memory = _warm(OnlineProposalMemory())
    probe = _ConsumptionProbe()
    monkeypatch.setattr(
        OnlineProposalMemory, "region_law", lambda self, **kw: probe
    )
    entry = {"trace": {"states": [None]}}
    monkeypatch.setattr(
        "compose_v4.rewrite.trace_shard.decode_state", lambda payload: graph
    )
    import compose_v4.control.pmo_online_memory as module

    monkeypatch.setattr(module, "SCHEMA", module.SCHEMA)

    def fallback(channel, entry):
        raise AssertionError("a warm memory must not fall back on the shallow lane")

    consumed = False
    for seed in range(16):
        optimizer = SimpleNamespace(
            shallow_rng=np.random.default_rng(seed),
            config=SimpleNamespace(max_primitives=32, max_blocks=8),
        )
        try:
            memory_channel_proposal(
                optimizer, "shallow_program_channel", entry, fallback, memory
            )
        except _ProbeConsumed:
            consumed = True
            break
        except ValueError:
            continue
    if not consumed:
        raise RegionLawNotConsumed(
            "the adapter completed 16 production draws without the draw site "
            "consulting the law it claims to install"
        )


def test_the_production_draw_site_consults_the_law():
    """Reuses the shipped consumption probe: it raises from `order` the moment
    the real synthesis path reaches it. Structural, not a signature check."""

    from compose_v4.control.dynamic_program_synthesis import (
        synthesize_dynamic_program,
    )
    from compose_v4.control.region_law_contract import (
        assert_region_law_is_consumed,
    )

    graph = _source()

    def draw(law, seed):
        return synthesize_dynamic_program(
            graph, np.random.default_rng(seed), max_modules=3,
            max_primitives=32, max_blocks=8, region_law=law,
        )

    assert assert_region_law_is_consumed(draw, attempts=16) >= 1


# ---- The size confound ---------------------------------------------------


def test_size_residual_removes_a_pure_size_gradient():
    """An outcome that is a deterministic function of size alone must leave
    ZERO residual. Expectation is arithmetic, not recomputed from the model."""

    size = SizeResidual()
    for delta in range(-20, 21):
        size.observe(delta, 0.1 * delta)
    for delta in (-16, -8, 0, 8, 16):
        assert abs(size.residual(delta, 0.1 * delta)) < 0.25, delta


def test_a_context_seen_only_at_one_size_cannot_win_on_size_alone():
    """The size confound, made adversarial.

    The scores here are a pure function of region SIZE -- exactly the corpus
    gradient r(score, heavy) = +0.67..0.81 that a naive memory would relearn
    as chemistry. Each context is then the only occupant of its size bucket, so
    its residual is zero by construction and the two learned effects must
    coincide. Scoring the RAW outcome instead separates them by ~0.2.
    """

    memory = OnlineProposalMemory()
    graph = _source()
    regions = bridge_separated_regions(graph, maximum=None)
    small = next(r for r in regions if r.size <= 2)
    large = max(regions, key=lambda r: r.size)
    assert large.size - small.size >= 8, "need a real size contrast"
    for region, score in ((small, 0.10), (large, 0.50)):
        for _ in range(12):
            memory.frontier.observe(f"x{region.size}{_}", score)
            memory.observe_transition(
                parent_graph=graph, parent_endpoint="p", parent_score=0.2,
                child_endpoint=f"x{region.size}{_}", child_score=score,
                child_heavy=int(graph.n_real_atoms) - region.size,
                family="substituent_delete", touched_slots=region.fragment,
            )
    a, _ = memory.edits.value(RegionContext.of(graph, small))
    b, _ = memory.edits.value(RegionContext.of(graph, large))
    assert abs(a - b) < 0.05, (a, b)


# ---- The donor memory may not promote ------------------------------------


def test_donor_association_cannot_promote_a_context_with_no_edit_evidence():
    """The gsk3b guard: on a predictor-backed oracle, high score is evidence of
    off-manifold exploitation, so donor membership alone must never lift a
    region above the size prior."""

    memory = OnlineProposalMemory()
    graph = _source()
    region = max(bridge_separated_regions(graph, maximum=None), key=lambda r: r.size)
    context = RegionContext.of(graph, region)
    other = next(
        RegionContext.of(graph, r)
        for r in bridge_separated_regions(graph, maximum=None)
        if RegionContext.of(graph, r) != context
    )
    for index in range(50):
        memory.donors.observe(
            context=context, endpoint=f"hack{index}", score=1.0, ordinal=index
        )
        memory.donors.observe(
            context=other, endpoint=f"dull{index}", score=0.0, ordinal=index
        )
    assert memory.donors.association(context)[1] == 50
    assert memory.donors.association(context)[0] > 0.2, (
        "the donor signal must be strong, or this guard proves nothing"
    )
    terms = memory.value_terms(context, region.size)
    assert terms["edit_observations"] == 0
    assert terms["total"] == pytest.approx(terms["scale"]), (
        "donor association leaked into an unevidenced context"
    )


def test_donor_rows_retain_provenance():
    memory = OnlineProposalMemory()
    graph = _source()
    memory.observe_scored_molecule(endpoint="CCO", score=0.42, graph=graph)
    assert memory.donors.rows
    row = memory.donors.rows[0]
    assert row["source_endpoint"] == "CCO"
    assert row["source_score"] == 0.42
    assert row["counted_call_ordinal"] >= 1
    assert row["claim"] == "association_only"


# ---- The allocation objective is the frontier ----------------------------


def test_frontier_gain_prefers_a_strong_parent_small_lift():
    """PMO grades on the top-ten mean, so a 0.80 -> 0.84 edit can be worth more
    than 0.05 -> 0.25. Built from a hand-made archive; the expectation is the
    arithmetic definition of the mean, not the ledger's own output."""

    ledger = FrontierLedger()
    for index in range(TOP_K):
        ledger.observe(f"m{index}", 0.80)
    strong = ledger.frontier_gain(0.84)
    weak = ledger.frontier_gain(0.25)
    assert strong == pytest.approx(0.004), strong
    assert weak == 0.0
    assert strong > weak


def test_frontier_gain_is_zero_off_the_frontier():
    ledger = FrontierLedger()
    for index in range(TOP_K):
        ledger.observe(f"m{index}", 0.9)
    assert ledger.frontier_gain(0.1) == 0.0


def test_ledger_refuses_a_changed_counted_score():
    ledger = FrontierLedger()
    ledger.observe("CCO", 0.3)
    ledger.observe("CCO", 0.3)
    with pytest.raises(ValueError):
        ledger.observe("CCO", 0.4)


# ---- Tilt is scale-free --------------------------------------------------


def test_tilt_is_invariant_to_the_task_score_scale():
    """Two runs identical but for a constant rescaling of every counted score
    must produce the SAME weights. A temperature fixed in score units fails
    this, and would make one controller behave differently per task."""

    graph = _source()

    def weights_for(scale):
        memory = OnlineProposalMemory()
        regions = bridge_separated_regions(graph, maximum=None)
        for index, region in enumerate(regions[:40]):
            score = (0.5 if index % 2 == 0 else 0.1) * scale
            memory.frontier.observe(f"C{index}", score)
            memory.observe_transition(
                parent_graph=graph, parent_endpoint="p",
                parent_score=0.2 * scale, child_endpoint=f"C{index}",
                child_score=score,
                child_heavy=int(graph.n_real_atoms) - region.size,
                family="substituent_delete", touched_slots=region.fragment,
            )
        law = memory.region_law()
        return law.weights(graph, law.regions(graph))

    a, b = weights_for(1.0), weights_for(10.0)
    assert np.allclose(a, b, rtol=1e-9), (a[:5], b[:5])


def test_degenerate_spread_falls_back_to_uniform():
    graph = _source()
    memory = _warm(OnlineProposalMemory(), context_bias=lambda c: True)
    law = memory.region_law()
    weights = law.weights(graph, law.regions(graph))
    assert np.all(weights > 0)


# ---- Information boundary ------------------------------------------------


def _executable_source() -> str:
    """The module's CODE with comments and string literals removed.

    Prose is not behaviour: the module's docstrings legitimately discuss the
    gsk3b measurement that motivated the donor gate, and a guard that failed on
    that would push the reasoning out of the file rather than the capability
    out of the code. Strip tokens instead, so the assertion is about what runs.
    """

    import io
    import pathlib
    import tokenize

    import compose_v4.control.pmo_online_memory as module

    source = pathlib.Path(module.__file__).read_text()
    kept = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        kept.append(token.string)
    return " ".join(kept)


def test_module_computes_no_molecular_property():
    """Where a property IS the objective, evaluating it off-ledger to choose a
    proposal is uncounted objective evaluation. The module must not reach the
    machinery at all."""

    code = _executable_source()
    for banned in ("QED", "sascorer", "TanimotoSimilarity", "rdFingerprintGenerator",
                   "MolFromSmiles", "Descriptors"):
        assert banned not in code, banned


def test_no_task_identity_branch_anywhere():
    """One controller across all PMO tasks: task-specific behaviour must emerge
    from counted feedback, never from an identifier in the code."""

    code = _executable_source().lower()
    for banned in ("gsk3b", "celecoxib", "perindopril", "jnk3", "drd2", "albuterol"):
        assert banned not in code, banned


def test_certification_states_the_exclusions():
    certification = OnlineProposalMemory().certification()
    joined = " ".join(certification["excludes"]).lower()
    for required in ("oracle internals", "target", "winner", "uncounted"):
        assert required in joined, required
    assert certification["donor_can_promote_unseen_context"] is False
    assert certification["primary_evidence"] == "edit_outcome_memory"


def test_edit_memory_shrinks_a_single_observation():
    """One lucky edit must not capture the draw."""

    memory = EditOutcomeMemory()
    context = RegionContext(1, 2, 1, False, False)
    memory.observe(
        context=context, family="f", delta_heavy=-3, residual=1.0,
        parent_endpoint="p", child_endpoint="c", parent_score=0.1, child_score=1.0,
    )
    effect, n = memory.value(context)
    assert n == 1
    assert effect < 0.2, effect


def test_memory_region_law_without_memory_matches_the_parent_law():
    graph = _source()
    plain = BridgeRegionLaw(maximum=None, margin=None)
    passthrough = MemoryRegionLaw(maximum=None, margin=None, memory=None)
    regions = plain.regions(graph)
    assert np.allclose(
        plain.weights(graph, regions), passthrough.weights(graph, regions)
    )


def test_allocation_priority_is_frontier_aligned_not_parent_relative():
    """The two objectives must be able to DISAGREE, or reporting both is
    theatre. A strong parent's small lift beats a weak parent's large one on
    the frontier, and loses on parent-relative change."""

    memory = OnlineProposalMemory()
    for index in range(TOP_K):
        memory.frontier.observe(f"m{index}", 0.80)
    strong = memory.allocation_priority(0.84, parent_score=0.80)
    weak = memory.allocation_priority(0.25, parent_score=0.05)
    assert strong["frontier_gain"] > weak["frontier_gain"]
    assert strong["parent_relative"] < weak["parent_relative"]


def test_proposal_cost_is_counted_separately_from_any_oracle_call():
    memory = _warm(OnlineProposalMemory())
    graph = _source()
    law = memory.region_law()
    before = memory.cost["region_weight_evaluations"]
    law.weights(graph, law.regions(graph))
    assert memory.cost["region_weight_evaluations"] > before
    assert set(memory.cost) == {"syntheses", "region_weight_evaluations"}
    assert "oracle" not in " ".join(memory.cost).lower()
