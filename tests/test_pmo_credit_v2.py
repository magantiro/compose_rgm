"""Adversarial invariants for the PMO-v2 hierarchical allocation prototype.

These are the deliverable of the v2 allocation work, not a smoke test.  Each one states a
property the FROZEN v1 rule provably fails (or, for the last three, provably keeps) and
pins the v2 behaviour against regression.  `scripts/pmo_credit_v2.py` is a prototype that
nothing pinned imports; `src/compose_v4/control/pmo_credit.py` is untouched and is
imported here only as the reference arm.

Charges no oracle call and launches nothing: every reward is a literal.

The five properties:

  1. duplicating untried cells must not enlarge their region's EXPLOITATION mass;
  2. a descendant of a productive basin inherits credit instead of the flat prior;
  3. that inherited credit is revocable once the descendant is measured and is bad;
  4. every live cell keeps a strictly positive share, isolated at one slot per draw;
  5. an exceptional joint overrides its marginals once `n_c` is large.

Two of them are regression tests for bugs found by the randomized fuzzer in
`scripts/pmo_credit_v2_audit.py` rather than by design: the region a proliferating cell
belongs to is a property of the CHAIN (`test_region_is_chain_aware_not_axis_aware`), and
the degenerate all-zero-value branch must still pool the untried children
(`test_zero_value_fallback_still_pools_untried`).
"""

from __future__ import annotations

import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from pmo_credit_v2 import (
    CHAINS,
    DEFAULT_KAPPA,
    PRODUCTION_CHAIN,
    HierarchicalCredit,
)

from compose_v4.control.pmo_credit import CreditKey, PopulationCredit

TOL = 1e-12


def _cell(basin="B", parent="p", family="atom_insert", scale="refine") -> CreditKey:
    return CreditKey(basin=basin, parent=parent, family=family, scale=scale)


def _productive(cls=HierarchicalCredit, **kw):
    """A 60-trial +5.0 lineage in PROVEN and a 60-trial +0.0 lineage in BARREN."""
    credit = cls(**kw)
    anchor = _cell(basin="PROVEN", parent="gen0")
    barren = _cell(basin="BARREN", parent="b0")
    for _ in range(60):
        credit.observe(anchor, 5.0)
        credit.observe(barren, 0.0)
    return credit, anchor, barren


def _v1_exploitation(credit: PopulationCredit, keys) -> np.ndarray:
    """v1's pre-floor credit vector -- the quantity its `allocate` mixes with `eps`."""
    values = np.asarray([max(0.0, credit.value(key)) for key in keys], dtype=float)
    total = float(values.sum())
    return values / total if total > 0.0 else np.full(len(keys), 1.0 / len(keys))


# ---- 1. Cell-proliferation invariance ------------------------------------------------


def test_untried_cell_duplication_does_not_enlarge_region_exploitation_mass():
    """THE central invariant.

    Precisely: let `v` be a node of the allocation tree and `U_N` a set of `N` untried
    cells that are all children of `v`.  Writing `E(U_N)` for their summed exploitation
    share,

        E(U_N) == E(U_1)   for every N >= 1,

    because untried cells contribute no trials and no improvement sum, so they move
    neither the node's value, nor its share, nor the denominator it is divided by.  The
    `eps` floor is deliberately excluded: it is reserved exploration, not credit.
    """
    credit = HierarchicalCredit()
    productive = _cell(basin="PROD", parent="p0")
    for _ in range(40):
        credit.observe(productive, 4.0)

    masses = []
    for fragments in (1, 2, 5, 10, 20, 30, 60, 200):
        keys = [productive] + [
            _cell(basin=f"FRAG{i}", parent=f"q{i}", family="cycle_close", scale="jump")
            for i in range(fragments)
        ]
        exploit = credit.exploitation_shares(keys)
        masses.append(float(exploit[1:].sum()))

    assert max(masses) - min(masses) < 1e-12, masses
    # and the region never reaches the N-times growth that v1 shows
    assert masses[-1] < 2.0 * masses[0]


def test_v1_is_the_arm_that_fails_proliferation():
    """The negative control: without this, invariant 1 could be vacuously true."""
    credit = PopulationCredit()
    productive = _cell(basin="PROD", parent="p0")
    for _ in range(40):
        credit.observe(productive, 4.0)

    def region(fragments):
        keys = [productive] + [
            _cell(basin=f"FRAG{i}", parent=f"q{i}", family="cycle_close", scale="jump")
            for i in range(fragments)
        ]
        return float(_v1_exploitation(credit, keys)[1:].sum())

    one, many = region(1), region(60)
    # Scaling the optimism bonus to the run's own measured reward fixed the ORDERING
    # between one untried cell and one cell with delivered evidence, but it does NOT
    # fix proliferation: enough barren cells still outbid a single productive one,
    # because their number grows while each keeps a positive share. That collective
    # failure is what the hierarchical backoff in this module exists to address.
    assert many > one, (one, many)
    keys = [productive] + [
        _cell(basin=f"FRAG{i}", parent=f"q{i}", family="cycle_close", scale="jump")
        for i in range(30)
    ]
    shares = credit.allocate(keys)
    assert float(shares[1:].sum()) > float(shares[0])


def test_proliferation_invariance_holds_at_every_level_of_the_chain():
    """Fresh `entry_id`s, fresh families and fresh basins attach at different nodes."""
    credit = HierarchicalCredit()
    for _ in range(30):
        credit.observe(_cell(basin="B0", parent="p0"), 3.0)
    credit.observe(_cell(basin="B0", parent="p1", family="cycle_close"), 1.0)

    cases = {
        "parent": lambda i: _cell(basin="B0", parent=f"new{i}"),
        "family": lambda i: _cell(basin="B0", parent="p0", family=f"fam{i}"),
        "basin": lambda i: _cell(basin=f"NEW{i}", parent=f"np{i}"),
    }
    base = [_cell(basin="B0", parent="p0"), _cell(basin="B0", parent="p1", family="cycle_close")]
    for axis, make in cases.items():
        attachment = credit.untried_attachment(make(0))
        masses = []
        for count in (1, 3, 12, 50):
            keys = base + [make(i) for i in range(count)]
            exploit = credit.exploitation_shares(keys)
            masses.append(
                float(
                    sum(
                        share
                        for key, share in zip(keys, exploit, strict=True)
                        if credit.untried_attachment(key) == attachment
                    )
                )
            )
        assert max(masses) - min(masses) < 1e-12, (axis, masses)


def test_region_is_chain_aware_not_axis_aware():
    """Regression: the region a proliferating cell joins depends on the CHAIN.

    Found by the fuzzer.  Under `context_only` the root's children are contexts, so a
    fresh BASIN attaches to the root alongside untried contexts of already-tried basins;
    a region defined per-axis measures a strict subset of that pool and reports dilution
    as a violation.  `untried_attachment` is the identity that makes the statement exact.
    """
    credit = HierarchicalCredit(chain="context_only")
    for _ in range(20):
        credit.observe(_cell(basin="B0", parent="p0"), 2.0)
    fresh_basin = _cell(basin="B9", parent="q0")
    untried_context_in_tried_basin = _cell(basin="B0", parent="p0", family="cycle_close")
    assert credit.untried_attachment(fresh_basin) == credit.untried_attachment(
        untried_context_in_tried_basin
    )
    # both are paid from the SAME pool, so the per-axis reading would miss half of it
    assert credit.untried_attachment(fresh_basin) == (0, ())


def test_zero_value_fallback_still_pools_untried():
    """Regression: the degenerate branch must not become count-proportional.

    Reachable only at `prior_weight == 0`, where the root backoff is 0 and every measured
    value is 0 too.  A plain uniform split over children would reopen the proliferation
    hole in exactly that corner.
    """
    credit = HierarchicalCredit(prior_weight=0.0)
    for _ in range(10):
        credit.observe(_cell(basin="B0", parent="p0"), 0.0)
    masses = []
    for count in (1, 5, 25):
        keys = [_cell(basin="B0", parent="p0")] + [
            _cell(basin=f"N{i}", parent=f"n{i}") for i in range(count)
        ]
        exploit = credit.exploitation_shares(keys)
        masses.append(float(exploit[1:].sum()))
    assert max(masses) - min(masses) < 1e-12, masses


# ---- 2. Generational inheritance -----------------------------------------------------


def test_descendant_of_productive_basin_inherits_credit():
    """An unseen child in PROVEN must outvalue an unseen child in BARREN."""
    credit, anchor, barren = _productive()
    good = _cell(basin="PROVEN", parent="gen1")
    bad = _cell(basin="BARREN", parent="b1")

    assert credit.shrunk_value(good) > credit.shrunk_value(bad)
    assert credit.shrunk_value(good) > 4.0
    assert credit.shrunk_value(bad) < 0.1

    shares = credit.allocate([anchor, good, barren, bad])
    assert shares[1] > shares[3]


def test_v1_gives_both_descendants_the_identical_flat_prior():
    """The negative control for inheritance: v1 cannot tell the two children apart.

    The prior is now scaled to the run's measured reward rather than fixed at 0.25, but
    it is still FLAT: both descendants receive the same value regardless of which parent
    they came from, which is exactly the inheritance the backoff supplies.
    """
    credit, _anchor, _barren = _productive(cls=PopulationCredit)
    good = _cell(basin="PROVEN", parent="gen1")
    bad = _cell(basin="BARREN", parent="b1")
    assert credit.value(good) == credit.value(bad) == pytest.approx(credit.observed_scale())


def test_inheritance_survives_every_kappa():
    """An untried cell has `n = 0`, so it inherits exactly for every `kappa > 0`."""
    good = _cell(basin="PROVEN", parent="gen1")
    bad = _cell(basin="BARREN", parent="b1")
    for kappa in (0.01, 0.25, 1.0, 4.0, 64.0, 1000.0):
        credit, _a, _b = _productive(kappa=kappa)
        assert credit.shrunk_value(good) > credit.shrunk_value(bad), kappa


# ---- 3. Negative evidence overrides inheritance --------------------------------------


def test_bad_descendant_loses_its_inherited_credit():
    """Backoff is a loan, not a subsidy: measured failure must repay it."""
    child = _cell(basin="PROVEN", parent="gen1")
    values = []
    for bad_trials in (0, 1, 2, 5, 10, 20, 40, 80, 160):
        credit, _anchor, _barren = _productive()
        for _ in range(bad_trials):
            credit.observe(child, 0.0)
        values.append(credit.shrunk_value(child))

    assert all(b <= a + TOL for a, b in pairwise(values)), values
    assert values[0] > 4.0
    assert values[-1] < 0.05 * values[0]


def test_inherited_credit_is_not_lost_by_a_single_unlucky_trial():
    """The converse guard: one zero must not erase a 60-trial lineage's evidence.

    Without this the test above would be satisfied by a rule that simply discards the
    prior on first contact, which is the opposite failure.

    The bound is the MECHANISM, not a tuned threshold: after one zero-improvement trial
    the child sits at exactly `(1 - 1/(1+kappa))` of its context value.  Note the context
    value must be read AFTER the observation -- observing the child also adds a trial to
    its context, basin and root, which nudges the backoff down.  Reading it before gives
    2.4590 against 2.5000 and looks like a broken identity when it is a real second-order
    effect of the same observation.
    """
    credit, _anchor, _barren = _productive()
    child = _cell(basin="PROVEN", parent="gen1")
    before = credit.shrunk_value(child)
    credit.observe(child, 0.0)
    after = credit.shrunk_value(child)
    context_after = credit.shrunk_values(child)[-2]

    single_observation_weight = 1.0 / (1.0 + DEFAULT_KAPPA)
    assert after < before
    assert after == pytest.approx((1.0 - single_observation_weight) * context_after)
    # the ancestor nudge is second order: the loss is the single-observation weight
    assert after > 0.9 * (1.0 - single_observation_weight) * before


def test_a_single_observation_never_outweighs_an_established_lineage():
    """The DESIGN CONSTRAINT that selects kappa, pinned separately from the mechanism.

    `select_kappa` takes the smallest kappa whose single-observation weight is at most
    0.5.  Stating it here means a future retune that violates it fails a test rather than
    silently changing what a first measurement is allowed to do.
    """
    assert 1.0 / (1.0 + DEFAULT_KAPPA) <= 0.5


# ---- 4. Exploration floor ------------------------------------------------------------


def test_every_cell_keeps_a_positive_share_at_one_slot_per_draw():
    """Isolate the floor: with one slot the only route off the winner is the floor.

    A multi-slot draw hides a floor failure because the winning cell simply exhausts its
    candidates and the next cell is taken for lack of an alternative.
    """
    credit = HierarchicalCredit()
    winner = _cell(basin="WIN", parent="w0")
    for _ in range(5000):
        credit.observe(winner, 1000.0)
    keys = [winner] + [
        _cell(basin=f"B{i}", parent=f"p{i}", family="cycle_close", scale="jump")
        for i in range(14)
    ]
    shares = credit.allocate(keys)
    bound = credit.exploration_floor / len(keys)

    assert float(shares.min()) >= bound - TOL
    assert all(share > 0.0 for share in shares)
    assert float(shares.sum()) == pytest.approx(1.0)

    rng = np.random.default_rng(20260920)
    counts = np.bincount(rng.choice(len(keys), size=6000, p=shares), minlength=len(keys))
    assert counts.min() > 0, counts


def test_total_exploration_budget_is_exactly_eps_however_many_cells_appear():
    """Exploration stays a RESERVED budget rather than an accident of proliferation."""
    credit = HierarchicalCredit()
    for _ in range(50):
        credit.observe(_cell(basin="B0", parent="p0"), 3.0)
    for size in (2, 5, 20, 100, 400):
        keys = [_cell(basin="B0", parent="p0")] + [
            _cell(basin=f"N{i}", parent=f"n{i}") for i in range(size - 1)
        ]
        shares = credit.allocate(keys)
        exploit = credit.exploitation_shares(keys)
        budget = float(shares.sum() - (1.0 - credit.exploration_floor) * exploit.sum())
        assert budget == pytest.approx(credit.exploration_floor, abs=1e-9), size
        assert float(shares.min()) >= credit.exploration_floor / size - TOL


# ---- 5. Interaction recovery ---------------------------------------------------------


def test_exceptional_joint_overrides_its_marginals_once_measured():
    """Shrinkage must vanish as `n_c` grows, or a real interaction is unlearnable."""
    peers = [_cell(basin="B", parent=f"peer{i}") for i in range(6)]
    star = _cell(basin="B", parent="star")

    first_win = None
    for n_star in (0, 1, 2, 4, 8, 16, 64, 256, 1024):
        credit = HierarchicalCredit()
        for peer in peers:
            for _ in range(50):
                credit.observe(peer, 0.2)
        for _ in range(n_star):
            credit.observe(star, 9.0)
        shrunk = credit.shrunk_value(star)
        if first_win is None and shrunk > credit.shrunk_value(peers[0]):
            first_win = n_star
        if n_star >= 256:
            assert shrunk == pytest.approx(9.0, rel=0.02), n_star

    assert first_win is not None and first_win <= 4, first_win


def test_shrinkage_weight_is_exactly_n_over_n_plus_kappa():
    """The mechanism, not just its consequence -- so a rewrite cannot drift."""
    kappa = DEFAULT_KAPPA
    peers = [_cell(basin="B", parent=f"peer{i}") for i in range(4)]
    star = _cell(basin="B", parent="star")
    for n_star in (1, 3, 9, 27):
        credit = HierarchicalCredit(kappa=kappa)
        for peer in peers:
            for _ in range(40):
                credit.observe(peer, 1.0)
        for _ in range(n_star):
            credit.observe(star, 7.0)
        backoff = credit.shrunk_values(star)[-2]
        weight = n_star / (n_star + kappa)
        assert credit.shrunk_value(star) == pytest.approx(
            weight * 7.0 + (1.0 - weight) * backoff
        )


# ---- structural guards ---------------------------------------------------------------


def test_v2_inherits_v1_evidence_semantics_unchanged():
    """v2 replaces the ALLOCATION rule and nothing else."""
    v1, v2 = PopulationCredit(), HierarchicalCredit()
    key = _cell()
    for value in (2.0, -1.0, 0.0, 4.5):
        v1.observe(key, value)
        v2.observe(key, value)
    assert v1.cell(key).payload() == v2.cell(key).payload()
    assert v1.marginal("basin")[key.basin].payload() == v2.marginal("basin")[key.basin].payload()
    assert v1.payload() == v2.payload()


def test_every_chain_is_a_strictly_nested_root_to_cell_hierarchy():
    for name, chain in CHAINS.items():
        assert chain[0] == (), name
        assert set(chain[-1]) == {"basin", "parent", "family", "scale"}, name
        for coarse, fine in pairwise(chain):
            assert set(coarse) < set(fine), (name, coarse, fine)


@pytest.mark.parametrize("chain", sorted(CHAINS))
def test_allocation_is_a_probability_vector_on_every_chain(chain):
    credit = HierarchicalCredit(chain=chain)
    for _ in range(12):
        credit.observe(_cell(basin="B0", parent="p0"), 2.0)
    keys = [
        _cell(basin=f"B{i % 3}", parent=f"p{i}", family="cycle_close" if i % 2 else "atom_insert")
        for i in range(9)
    ]
    shares = credit.allocate(keys)
    assert float(shares.sum()) == pytest.approx(1.0)
    assert float(shares.min()) > 0.0


def test_production_chain_is_the_one_the_audit_selected():
    assert PRODUCTION_CHAIN == "basin_scale_context_parent"
    assert CHAINS[PRODUCTION_CHAIN][-1] == ("basin", "family", "scale", "parent")


# ---- randomized fuzz, bounded so it can live in the suite ---------------------------


def test_fuzz_invariants_on_randomized_worlds():
    """A bounded slice of the audit's fuzzer, so a regression fails here and not only
    in a diagnostic nobody runs.  The full sweep lives in
    `scripts/pmo_credit_v2_audit.py`.
    """
    import pmo_credit_v2_audit as audit

    report = audit.run_fuzz(worlds=60, randomize_settings=False)
    for name, row in report["checks"].items():
        assert row["violation_rate"] == 0.0, (name, report["counterexamples"].get(name))
        assert row["worlds_checked"] > 30, (name, row["worlds_checked"])


def test_fuzz_controls_separate_the_two_mechanisms():
    """The fuzzer must be able to FAIL: pooling carries P1, the hierarchy carries P2."""
    import pmo_credit_v2_audit as audit

    report = audit.run_fuzz_controls(worlds=40)
    arms = report["arms"]
    assert arms["v2_production"]["p1_proliferation_invariance"]["violation_rate"] == 0.0
    assert arms["v2_production"]["p2_lineage_inheritance"]["violation_rate"] == 0.0
    # hierarchy without pooling loses P1 but keeps P2
    assert arms["v2_hierarchy_without_untried_pooling"]["p1_proliferation_invariance"][
        "violation_rate"
    ] > 0.5
    assert (
        arms["v2_hierarchy_without_untried_pooling"]["p2_lineage_inheritance"]["violation_rate"]
        == 0.0
    )
    # pooling without the hierarchy keeps P1 but loses P2
    assert arms["v2_flat_chain_no_hierarchy"]["p1_proliferation_invariance"]["violation_rate"] == 0.0
    assert arms["v2_flat_chain_no_hierarchy"]["p2_lineage_inheritance"]["violation_rate"] > 0.5
    # frozen v1 loses both
    assert arms["v1_frozen"]["p1_proliferation_invariance"]["violation_rate"] > 0.5
    assert arms["v1_frozen"]["p2_lineage_inheritance"]["violation_rate"] > 0.5


# ---- characterized counterexample, found by the fuzzer -------------------------------


def _inheritance_gap(prior_weight: float, kappa: float) -> tuple[float, float]:
    """A near-unmeasured basin beside a well-measured but barely-productive one.

    Minimal reproducer for the only P2 violation the fuzzer found: basin S carries 50
    trials at a small +0.19 upside, basin W carries a single 0.0 trial.  S is therefore
    the "productive" context by raw upside and W is not -- but S has enough evidence to be
    pulled DOWN to its true low value while W, being nearly unmeasured, keeps whatever the
    backoff hands it.
    """
    credit = HierarchicalCredit(prior_weight=prior_weight, kappa=kappa)
    for _ in range(50):
        credit.observe(_cell(basin="S", parent="s0"), 0.19)
    credit.observe(_cell(basin="W", parent="w0"), 0.0)
    return (
        credit.shrunk_value(_cell(basin="S", parent="unseen")),
        credit.shrunk_value(_cell(basin="W", parent="unseen")),
    )


def test_inheritance_holds_at_the_production_settings():
    """P2 is safe at `prior_weight = 0.25` for every kappa the audit would pick."""
    for kappa in (0.5, 1.0, 2.0, 4.0, 16.0):
        strong, weak = _inheritance_gap(0.25, kappa)
        assert strong > weak, (kappa, strong, weak)


def test_inheritance_can_invert_when_the_backoff_sits_above_the_truth():
    """The characterized failure, pinned so it cannot be rediscovered as a surprise.

    Inversion needs the backoff held ABOVE both measured values -- by an inflated
    `prior_weight`, or by a `kappa` so large the root prior still dominates after 50
    trials -- plus a nearly-unmeasured sibling that keeps it.  It is a TUNING artifact of
    those two constants, not a structural property of the hierarchy: it disappears at the
    production settings, which the test above pins.
    """
    # an inflated prior (20x the production 0.25) inverts from kappa = 4 upward
    assert _inheritance_gap(5.0, 1.0)[0] > _inheritance_gap(5.0, 1.0)[1]
    for kappa in (4.0, 16.0, 64.0):
        strong, weak = _inheritance_gap(5.0, kappa)
        assert weak > strong, (kappa, strong, weak)
    # at the production prior it takes kappa = 64 to reach the same inversion
    strong, weak = _inheritance_gap(0.25, 64.0)
    assert weak > strong, (strong, weak)
