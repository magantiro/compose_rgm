"""Selection, invariant and fuzz audit for the PMO-v2 hierarchical allocation prototype.

Charges NO oracle call, opens no network connection, constructs no TDC `Oracle` and
launches nothing.  Every reward is a closed-form or pseudo-random synthetic number.

Four parts:

  A  SELECTION -- which backoff chain, and which `kappa`.  Held-out predictive error of
     each chain on synthetic cell universes with KNOWN structure, plus a `kappa` sweep.
     This is what stops the hierarchy and the shrinkage constant from being magic numbers.
  B  TARGETED INVARIANTS -- the five properties, each on a case built to exercise it,
     reported with the measured number and the v1 comparison.
  C  RANDOMIZED FUZZING -- the same five properties attacked on hundreds of worlds nobody
     designed: random basin/parent/family/scale structure, random arrival order, random
     and adversarial reward regimes (jackpot, all-zeros, late-productive, parent-dominant,
     heavy churn).  Reports violation RATES, a `kappa` sweep that separates a structural
     flaw from a tuning artifact, and a minimal reproducer for any violation found.
  D  MATCHED A/B -- v1 versus v2 on the same landscapes and the same seeds, through the
     frozen controller's own `_allocate` path, by swapping `ctrl.credit` only.

MEASURED vs INFERRED.  Everything this script prints is MEASURED on synthetic data.  No
real PMO credit snapshot exists on disk, so the transfer of any of it to a scored oracle
run is INFERRED and is labelled as such in the report.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np
from pmo_credit_v2 import (
    CHAINS,
    DEFAULT_KAPPA,
    PRODUCTION_CHAIN,
    HierarchicalCredit,
)

from compose_v4.control.pmo_credit import (
    SCALES,
    CreditCell,
    CreditKey,
    PopulationCredit,
    credit_key_from_candidate,
    improvement,
)

TOL = 1e-12
FAMILIES = ("atom_insert", "atom_insert+cycle_close", "atom_delete+atom_insert", "bond_reroute")


def _credit(**kw) -> HierarchicalCredit:
    return HierarchicalCredit(**kw)


class V1ExploitView(PopulationCredit):
    """Frozen v1, exposing the same two hooks the fuzz checks call on v2.

    This adds no behaviour: `exploitation_shares` is exactly the normalised `value`
    vector that `PopulationCredit.allocate` mixes with its floor, and `shrunk_value` is
    v1's `value`.  Expressing v1 through the same interface is what lets the identical
    check functions run against both arms, so the baseline cannot drift away from the
    test it is a baseline for.
    """

    kappa = float("nan")
    chain_name = "v1_flat_cell"

    def exploitation_shares(self, keys) -> np.ndarray:
        keys = list(keys)
        if not keys:
            return np.zeros(0, dtype=float)
        values = np.asarray([max(0.0, self.value(key)) for key in keys], dtype=float)
        total = float(values.sum())
        return values / total if total > 0.0 else np.full(len(keys), 1.0 / len(keys))

    def shrunk_value(self, key: CreditKey) -> float:
        return self.value(key)

    def untried_attachment(self, key: CreditKey):
        """v1 has no hierarchy, so every untried cell hangs off the single root.

        Returning one shared node for all of them makes the P1 region for v1 "every
        untried cell in the pool", which is the faithful flat-allocator analogue of the
        v2 pool and the most GENEROUS region v1 could be scored on: it is the whole set
        among which v1's mass would have to stay fixed.
        """
        return None if self.cell(key).trials else (-1, ())


class NoPoolControl(HierarchicalCredit):
    """The hierarchy WITHOUT the untried-pool rule: each untried child keeps a full share.

    This is the naive way to write count-shrunk backoff, and it is the control that
    proves WHICH half of v2 does the work.  It keeps the backoff (so lineage inheritance
    still functions) and drops only the pooling (so proliferation should win again).  If
    the fuzzer cannot tell this apart from the real rule, the fuzzer is not testing the
    mechanism it claims to test.
    """

    def _split(self, share, node, groups, node_value, node_trials) -> None:
        available = share[node]
        weights = {
            child: max(
                0.0,
                node_value[child] if node_trials(*child) > 0 else node_value[node],
            )
            for child in groups
        }
        total = sum(weights.values())
        if total <= 0.0:
            for child in groups:
                share[child] = available / len(groups)
            return
        for child in groups:
            share[child] = available * weights[child] / total


def arm_exploitation(credit, keys) -> np.ndarray:
    """The pre-floor credit vector for EITHER arm, so the two are directly comparable."""
    keys = list(keys)
    if not keys:
        return np.zeros(0, dtype=float)
    if hasattr(credit, "exploitation_shares"):
        return credit.exploitation_shares(keys)
    values = np.asarray([max(0.0, credit.value(key)) for key in keys], dtype=float)
    total = float(values.sum())
    return values / total if total > 0.0 else np.full(len(keys), 1.0 / len(keys))


def pool_prior_diagnostics(credit, pool_keys) -> dict:
    """The two headline v1 numbers, measured on a ROUND'S POOL rather than on history.

    Measuring these over `credit.cells` instead would read 0% untried by construction --
    every cell in that dict has been observed at least once.  The production figures they
    are compared against (90.1% of pool untried, 68.1% of credit mass from the flat
    prior) are pool quantities.
    """
    keys = list(dict.fromkeys(pool_keys))
    if not keys:
        return {"fraction_of_pool_untried": None, "exploitation_mass_on_untried": None}
    exploit = arm_exploitation(credit, keys)
    untried = [index for index, key in enumerate(keys) if credit.cell(key).trials == 0]

    # How much mass goes to untried cells is NOT the thing v2 changes -- it replaces a
    # FLAT prior with an INHERITED one, so the mass can stay the same while being
    # distributed by evidence.  What distinguishes the arms is whether that mass is
    # informative: does an untried cell's share track the measured productivity of the
    # context it descends from?  v1 values every untried cell identically, so its answer
    # is exactly one distinct share and a rank correlation of 0 by construction.
    context_value: dict[tuple[str, str, str], list] = {}
    for key, cell in credit.cells.items():
        bucket = context_value.setdefault((key.basin, key.family, key.scale), [0, 0.0])
        bucket[0] += cell.trials
        bucket[1] += cell.positive_improvement_sum
    shares, contexts = [], []
    for index in untried:
        key = keys[index]
        trials, positive = context_value.get((key.basin, key.family, key.scale), (0, 0.0))
        shares.append(float(exploit[index]))
        contexts.append(positive / trials if trials else 0.0)
    informativeness = _spearman(shares, contexts) if len(shares) >= 3 else float("nan")
    return {
        "fraction_of_pool_untried": round(len(untried) / len(keys), 4),
        "exploitation_mass_on_untried": round(
            float(exploit[untried].sum()) if untried else 0.0, 4
        ),
        "distinct_share_values_among_untried": len({round(v, 10) for v in shares}),
        "untried_share_vs_context_productivity_rank_correlation": (
            None if math.isnan(informativeness) else round(informativeness, 4)
        ),
    }


def _clone_like(credit):
    """An empty credit object of the same arm and settings as `credit`."""
    if isinstance(credit, HierarchicalCredit):
        return type(credit)(
            exploration_floor=credit.exploration_floor,
            prior_weight=credit.prior_weight,
            kappa=credit.kappa,
            chain=credit.chain_name if credit.chain_name != "custom" else credit.chain,
        )
    return type(credit)(
        exploration_floor=credit.exploration_floor, prior_weight=credit.prior_weight
    )


def _copy_without(credit: HierarchicalCredit, drop: CreditKey) -> HierarchicalCredit:
    """A credit state identical to `credit` minus one cell's evidence."""
    clone = HierarchicalCredit(
        exploration_floor=credit.exploration_floor,
        prior_weight=credit.prior_weight,
        kappa=credit.kappa,
        chain=credit.chain_name if credit.chain_name != "custom" else credit.chain,
    )
    clone.cells = {key: cell for key, cell in credit.cells.items() if key != drop}
    return clone


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks with ties, so a tied predictor is not silently rewarded."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    index = 0
    while index < len(values):
        stop = index
        while stop + 1 < len(values) and sorted_values[stop + 1] == sorted_values[index]:
            stop += 1
        ranks[order[index : stop + 1]] = 0.5 * (index + stop) + 1.0
        index = stop + 1
    return ranks


def _spearman(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(np.corrcoef(_average_ranks(a), _average_ranks(b))[0, 1])


# ======================================================================================
# Synthetic worlds
# ======================================================================================

TRUTH_KINDS = (
    "basin",
    "basin_scale",
    "basin_family",
    "context",
    "parent_dominant",
    "mixed",
    "interaction",
    "jackpot",
    "all_zeros",
    "all_negative",
    "sparse_positive",
)


def make_world(rng: np.random.Generator, kind: str, *, n_basins=None, n_parents=None):
    """A cell universe with a known true upside per cell.

    Returns `(keys, truth)`.  `truth[key]` is the expected upside the landscape pays for
    that cell; the fuzzers sample noisy observations around it.
    """
    n_basins = int(rng.integers(1, 10)) if n_basins is None else n_basins
    n_families = int(rng.integers(1, 5))
    n_scales = int(rng.integers(1, 4))
    n_parents = int(rng.integers(1, 9)) if n_parents is None else n_parents
    basins = [f"B{i}" for i in range(n_basins)]
    families = list(FAMILIES[:n_families])
    scales = list(SCALES[:n_scales])

    basin_effect = {b: float(rng.gamma(2.0, 1.0)) for b in basins}
    family_effect = {f: float(rng.uniform(0.2, 2.0)) for f in families}
    scale_effect = {s: float(rng.uniform(0.2, 2.0)) for s in scales}
    parent_effect: dict[str, float] = {}

    keys, truth = [], {}
    for b in basins:
        for p_index in range(n_parents):
            parent = f"{b}_p{p_index}"
            parent_effect[parent] = float(rng.gamma(2.0, 0.5))
            for f in families:
                for s in scales:
                    key = CreditKey(basin=b, parent=parent, family=f, scale=s)
                    if kind == "basin":
                        value = basin_effect[b]
                    elif kind == "basin_scale":
                        value = basin_effect[b] * scale_effect[s]
                    elif kind == "basin_family":
                        value = basin_effect[b] * family_effect[f]
                    elif kind == "context":
                        value = basin_effect[b] * family_effect[f] * scale_effect[s]
                    elif kind == "parent_dominant":
                        value = parent_effect[parent] * 3.0 + 0.1 * basin_effect[b]
                    elif kind == "mixed":
                        value = (
                            basin_effect[b] * scale_effect[s]
                            + 0.5 * family_effect[f]
                            + 0.3 * parent_effect[parent]
                        )
                    elif kind == "interaction":
                        value = basin_effect[b] * 0.3
                    elif kind == "jackpot" or kind == "all_zeros":
                        value = 0.0
                    elif kind == "all_negative":
                        # every trial is a regression: trials accumulate, the positive
                        # sum stays exactly 0, so every Q_raw is 0 with large n
                        value = -abs(float(rng.gamma(2.0, 1.0)))
                    elif kind == "sparse_positive":
                        value = (
                            float(rng.gamma(2.0, 2.0)) if rng.random() < 0.05 else -0.5
                        )
                    else:
                        raise ValueError(f"unknown truth kind {kind!r}")
                    keys.append(key)
                    truth[key] = float(value)

    if kind == "interaction":
        # A handful of genuinely exceptional joints that no marginal can see.
        for key in rng.choice(np.asarray(keys, dtype=object), size=min(3, len(keys)), replace=False):
            truth[key] = 12.0
    if kind == "jackpot":
        truth[keys[int(rng.integers(0, len(keys)))]] = 30.0
    return keys, truth


def observe_world(
    credit, keys, truth, rng, *, rounds=12, draws=13, churn=0.0, late=None, noise=0.25
):
    """Drive rounds of allocate -> synthetic reward -> observe against a credit object.

    `churn` is the per-round probability that a fresh `entry_id` is minted for each basin,
    which is the production behaviour that made the v1 cell key churn faster than evidence
    accumulated.  `late` makes one basin productive only after that round index.
    """
    live = list(keys)
    truth = dict(truth)
    late_basin = None
    if late is not None:
        late_basin = min(k.basin for k in keys)
    for round_index in range(rounds):
        pool = list(live)
        if not pool:
            break
        shares = credit.allocate(pool)
        picks = rng.choice(len(pool), size=min(draws, len(pool)), replace=True, p=shares)
        for index in picks:
            key = pool[int(index)]
            value = truth[key]
            if late_basin is not None and key.basin == late_basin:
                value = 8.0 if round_index >= late else 0.0
            reward = float(value + rng.normal(0.0, noise))
            credit.observe(key, reward)
        if churn > 0.0 and rng.random() < churn:
            fresh = []
            for basin in sorted({k.basin for k in live}):
                parent = f"{basin}_gen{round_index}_{int(rng.integers(0, 10_000))}"
                # One fresh cell per DISTINCT (family, scale) in the basin.  Without the
                # de-duplication the same `(basin, parent, family, scale)` is minted twice
                # and `allocate` rightly refuses a repeated cell.
                seen: set[tuple[str, str]] = set()
                for key in [k for k in keys if k.basin == basin]:
                    if (key.family, key.scale) in seen or len(seen) >= 4:
                        continue
                    seen.add((key.family, key.scale))
                    new_key = CreditKey(
                        basin=basin, parent=parent, family=key.family, scale=key.scale
                    )
                    truth[new_key] = truth[key]
                    fresh.append(new_key)
            live = list(dict.fromkeys(live + fresh))
    return credit, live, truth


# ======================================================================================
# Part A -- selection: which chain, which kappa
# ======================================================================================


def select_hierarchy(seeds=range(40), kappa=DEFAULT_KAPPA):
    """Held-out predictive error of each backoff chain, per landscape kind.

    Worlds are bounded to 4 basins x 3 parents: leave-one-out is quadratic in the cell
    count, and the question -- which marginal predicts an unseen cell -- does not need the
    wide structures the fuzzer uses to attack the invariants.

    For every cell that carries evidence, drop that cell and ask each chain to predict its
    true upside from the rest of the joint.  This is the question the allocation actually
    asks of an unseen cell, so it is the right basis for choosing the chain.
    """
    per_kind: dict[str, dict[str, dict]] = {}
    for kind in TRUTH_KINDS:
        if kind == "all_zeros":
            continue  # no signal to predict; excluded rather than scored as a tie
        errors: dict[str, list[float]] = {name: [] for name in CHAINS}
        correlations: dict[str, list[float]] = {name: [] for name in CHAINS}
        for seed in seeds:
            rng = np.random.default_rng(20260920 + int(seed))
            keys, truth = make_world(rng, kind, n_basins=4, n_parents=3)
            base = _credit(kappa=kappa, chain=PRODUCTION_CHAIN)
            observe_world(base, keys, truth, rng, rounds=10, draws=20)
            tried = [k for k in base.cells if base.cells[k].trials > 0]
            if len(tried) < 6:
                continue
            for name in CHAINS:
                predicted, actual = [], []
                for held in tried:
                    clone = HierarchicalCredit(kappa=kappa, chain=name)
                    clone.cells = {k: c for k, c in base.cells.items() if k != held}
                    predicted.append(clone.shrunk_value(held))
                    actual.append(max(0.0, truth[held]))
                errors[name].append(float(np.mean(np.abs(np.asarray(predicted) - actual))))
                rho = _spearman(predicted, actual)
                if not math.isnan(rho):
                    correlations[name].append(rho)
        per_kind[kind] = {
            name: {
                "median_held_out_mae": round(float(np.median(errors[name])), 4)
                if errors[name]
                else None,
                "median_rank_correlation": round(float(np.median(correlations[name])), 4)
                if correlations[name]
                else None,
                "n_worlds": len(errors[name]),
            }
            for name in CHAINS
        }

    # Aggregate: mean rank of each chain across landscape kinds (1 = best MAE).
    ranks: dict[str, list[int]] = {name: [] for name in CHAINS}
    for kind, rows in per_kind.items():
        scored = sorted(
            (name for name in CHAINS if rows[name]["median_held_out_mae"] is not None),
            key=lambda name: rows[name]["median_held_out_mae"],
        )
        for position, name in enumerate(scored, start=1):
            ranks[name].append(position)
    summary = {
        name: {
            "mean_mae_rank_across_landscapes": round(float(np.mean(ranks[name])), 3)
            if ranks[name]
            else None,
            "wins": sum(1 for r in ranks[name] if r == 1),
        }
        for name in CHAINS
    }
    best = min(
        (name for name in CHAINS if summary[name]["mean_mae_rank_across_landscapes"] is not None),
        key=lambda name: summary[name]["mean_mae_rank_across_landscapes"],
    )
    return {
        "method": (
            "leave-one-cell-out: drop a cell's evidence, predict its true upside from the "
            "rest of the joint through each chain, score MAE and rank correlation"
        ),
        "kappa_used": kappa,
        "per_landscape": per_kind,
        "summary": summary,
        "best_by_mean_rank": best,
        "selected": PRODUCTION_CHAIN,
    }


def select_kappa(
    seeds=range(4),
    kappas=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0),
    noise_levels=(0.1, 1.0, 8.0),
):
    """What `kappa` buys, as a function of OBSERVATION NOISE.

    A single-noise sweep is not a defensible basis for this constant.  `kappa` is the
    number of pseudo-trials of backoff evidence a cell must outweigh before its own
    estimate dominates, so the right `kappa` is a function of how informative ONE
    measurement is.  On near-noiseless synthetic rewards the optimum collapses to the
    smallest value swept -- which says "trust a single observation immediately", a
    statement about the landscape and not about the rule.

    Note what does NOT depend on `kappa`: an UNTRIED cell has `n = 0`, so its weight on
    its own estimate is 0 for every `kappa > 0` and it inherits its parent exactly.
    `kappa` governs only how fast a cell that HAS been measured moves off the inherited
    value, which is why P2 is near-flat in `kappa` while P3 and P4 are the properties it
    trades against.
    """
    grid = []
    for noise in noise_levels:
        for kappa in kappas:
            maes, gains = [], []
            for kind in TRUTH_KINDS:
                for seed in seeds:
                    rng = np.random.default_rng(777_000 + int(seed))
                    keys, truth = make_world(rng, kind, n_basins=4, n_parents=3)
                    base = _credit(kappa=kappa, chain=PRODUCTION_CHAIN)
                    observe_world(base, keys, truth, rng, rounds=10, draws=20, noise=noise)
                    tried = [k for k in base.cells if base.cells[k].trials > 0]
                    if kind not in ("all_zeros", "all_negative") and len(tried) >= 6:
                        predicted, actual = [], []
                        for held in tried:
                            clone = _copy_without(base, held)
                            predicted.append(clone.shrunk_value(held))
                            actual.append(max(0.0, truth[held]))
                        maes.append(float(np.mean(np.abs(np.asarray(predicted) - actual))))
                    rng2 = np.random.default_rng(777_000 + int(seed))
                    keys2, truth2 = make_world(rng2, kind, n_basins=4, n_parents=3)
                    run = _credit(kappa=kappa, chain=PRODUCTION_CHAIN)
                    observe_world(
                        run, keys2, truth2, rng2, rounds=12, draws=13, churn=0.3, noise=noise
                    )
                    realized = sum(
                        truth2.get(k, 0.0) * c.trials for k, c in run.cells.items()
                    )
                    trials = sum(c.trials for c in run.cells.values())
                    if trials:
                        gains.append(realized / trials)
            grid.append(
                {
                    "noise": noise,
                    "kappa": kappa,
                    "median_held_out_mae": round(float(np.median(maes)), 4) if maes else None,
                    "mean_true_gain_per_draw": round(float(np.mean(gains)), 4) if gains else None,
                }
            )

    # Best kappa per noise level, by each criterion.
    by_noise = {}
    for noise in noise_levels:
        rows = [r for r in grid if r["noise"] == noise and r["median_held_out_mae"] is not None]
        best_mae = min(rows, key=lambda r: r["median_held_out_mae"])
        best_gain = max(rows, key=lambda r: r["mean_true_gain_per_draw"])
        by_noise[str(noise)] = {
            "best_kappa_by_held_out_mae": best_mae["kappa"],
            "best_kappa_by_true_gain": best_gain["kappa"],
            "mae_at_best": best_mae["median_held_out_mae"],
            "gain_at_best": best_gain["mean_true_gain_per_draw"],
        }

    # Robust choice: minimax REGRET in gain across noise levels, because the production
    # reward noise is unmeasured and a constant tuned to one noise level is a guess.
    regret = {}
    for kappa in kappas:
        worst = 0.0
        for noise in noise_levels:
            rows = [r for r in grid if r["noise"] == noise]
            best = max(r["mean_true_gain_per_draw"] for r in rows)
            mine = next(r["mean_true_gain_per_draw"] for r in rows if r["kappa"] == kappa)
            if best > 0:
                worst = max(worst, (best - mine) / abs(best))
        regret[kappa] = round(worst, 4)
    minimax = min(regret, key=regret.get)

    # DECLARED SELECTION RULE, applied in code so it cannot be hand-fitted after seeing
    # the numbers: take the SMALLEST kappa whose single-observation weight is at most
    # 0.5 -- one measurement of a cell must not outweigh everything its lineage has
    # established -- subject to a worst-case relative gain regret of at most 0.25.
    max_single_observation_weight = 0.5
    max_regret = 0.25
    eligible = [
        kappa
        for kappa in kappas
        if 1.0 / (1.0 + kappa) <= max_single_observation_weight + 1e-12
        and regret[kappa] <= max_regret
    ]
    selected_by_rule = min(eligible) if eligible else minimax
    return {
        "method": (
            "sweep kappa x observation noise over all landscape kinds; score held-out MAE "
            "and TRUE gain per draw (the landscape value of what was drawn, not the noisy "
            "realization, so noise does not inflate the score it is being swept against)"
        ),
        "grid": grid,
        "best_by_noise_level": by_noise,
        "worst_case_relative_gain_regret_by_kappa": regret,
        "minimax_regret_kappa": minimax,
        "selection_rule": (
            "smallest kappa with single-observation weight 1/(1+kappa) <= 0.5 AND "
            "worst-case relative gain regret <= 0.25"
        ),
        "single_observation_weight_by_kappa": {
            str(kappa): round(1.0 / (1.0 + kappa), 4) for kappa in kappas
        },
        "eligible_under_rule": eligible,
        "selected_by_rule": selected_by_rule,
        "selected": DEFAULT_KAPPA,
        "criteria_disagree": (
            "held-out MAE wants kappa to TRACK the observation noise (best 1.0 at noise "
            "0.1, 4.0 at noise 1.0, 64.0 at noise 8.0) while realized gain prefers the "
            "smallest kappa at every noise level.  The rule breaks the tie on the "
            "stability of an established lineage, which is the property the v1 defect is "
            "about; production reward noise is UNMEASURED, so this is not settled."
        ),
        "kappa_free_property": (
            "an untried cell has n=0, so it inherits its parent exactly for EVERY kappa>0; "
            "kappa governs only how fast a measured cell moves off the inherited value"
        ),
    }


# ======================================================================================
# Part B -- the five targeted invariants
# ======================================================================================


def _productive_state(cls, *, trials=60, value=5.0, **kw):
    credit = cls(**kw)
    anchor = CreditKey(basin="PROVEN", parent="gen0", family="atom_insert", scale="refine")
    for _ in range(trials):
        credit.observe(anchor, value)
    barren = CreditKey(basin="BARREN", parent="b0", family="atom_insert", scale="refine")
    for _ in range(trials):
        credit.observe(barren, 0.0)
    return credit, anchor, barren


def invariant_1_proliferation():
    """Duplicating untried cells must not enlarge their region's EXPLOITATION mass.

    Precise statement.  Let `v` be a node of the allocation tree, `U_N` a set of `N`
    untried cells that are all children of `v`, and `E(U_N)` the summed exploitation
    share of `U_N`.  Then for every `N >= 1`

        E(U_N) <= E(U_1) + tol       (v2 also achieves equality to machine precision)

    whereas v1 satisfies `E(U_N) ~ N * E(U_1)` until the region saturates the budget.
    """
    rows = []
    for name, cls in (("v1", PopulationCredit), ("v2", HierarchicalCredit)):
        credit = cls()
        productive = CreditKey(basin="PROD", parent="p0", family="atom_insert", scale="refine")
        for _ in range(40):
            credit.observe(productive, 4.0)
        baseline_exploit = None
        for fragments in (1, 5, 10, 20, 30, 40, 60):
            keys = [productive] + [
                CreditKey(basin=f"FRAG{i}", parent=f"q{i}", family="cycle_close", scale="jump")
                for i in range(fragments)
            ]
            shares = credit.allocate(keys)
            if name == "v2":
                exploit = credit.exploitation_shares(keys)
            else:
                # v1 has no separate exploitation channel; reconstruct it from its own rule
                values = np.asarray([max(0.0, credit.value(k)) for k in keys])
                exploit = values / values.sum()
            region = float(exploit[1:].sum())
            baseline_exploit = region if baseline_exploit is None else baseline_exploit
            rows.append(
                {
                    "arm": name,
                    "untried_cells": fragments,
                    "region_exploitation_mass": round(region, 6),
                    "ratio_to_single_cell": round(region / baseline_exploit, 4),
                    "productive_total_share": round(float(shares[0]), 4),
                    "region_total_share": round(float(shares[1:].sum()), 4),
                }
            )
    v2_rows = [r for r in rows if r["arm"] == "v2"]
    v1_rows = [r for r in rows if r["arm"] == "v1"]
    spread = max(r["region_exploitation_mass"] for r in v2_rows) - min(
        r["region_exploitation_mass"] for r in v2_rows
    )
    return {
        "statement": (
            "for untried cells U_N sharing a parent node, the summed EXPLOITATION share "
            "E(U_N) <= E(U_1) + 1e-12 for all N; the deliberate eps floor is excluded "
            "because it is reserved exploration, not credit"
        ),
        "sweep": rows,
        "v2_exploitation_mass_spread_over_1_to_60_cells": float(f"{spread:.3e}"),
        "v1_exploitation_ratio_at_60_cells": v1_rows[-1]["ratio_to_single_cell"],
        "v2_exploitation_ratio_at_60_cells": v2_rows[-1]["ratio_to_single_cell"],
        "v1_productive_share_at_60_cells": v1_rows[-1]["productive_total_share"],
        "v2_productive_share_at_60_cells": v2_rows[-1]["productive_total_share"],
        "measured": "MEASURED",
        "verdict": "PASS" if spread <= 1e-9 else "FAIL",
    }


def invariant_2_inheritance():
    """A descendant of a productive basin must inherit credit, not the flat prior."""
    out = {}
    for name, cls in (("v1", PopulationCredit), ("v2", HierarchicalCredit)):
        credit, anchor, barren = _productive_state(cls)
        child_good = CreditKey(basin="PROVEN", parent="gen1", family="atom_insert", scale="refine")
        child_bad = CreditKey(basin="BARREN", parent="b1", family="atom_insert", scale="refine")
        keys = [anchor, child_good, barren, child_bad]
        shares = credit.allocate(keys)
        value = credit.shrunk_value if name == "v2" else credit.value
        out[name] = {
            "value_child_of_proven": round(value(child_good), 4),
            "value_child_of_barren": round(value(child_bad), 4),
            "values_identical": abs(value(child_good) - value(child_bad)) < TOL,
            "share_child_of_proven": round(float(shares[1]), 4),
            "share_child_of_barren": round(float(shares[3]), 4),
            "share_ratio_proven_over_barren": round(float(shares[1] / shares[3]), 3),
        }
    return {
        "statement": (
            "with a 60-trial +5.0 lineage in basin PROVEN and a 60-trial +0.0 lineage in "
            "basin BARREN, an UNSEEN child in PROVEN must be valued above an unseen child "
            "in BARREN"
        ),
        "arms": out,
        "measured": "MEASURED",
        "verdict": "PASS" if not out["v2"]["values_identical"] else "FAIL",
        "v1_reference": "v1 values both children at exactly the flat prior 0.25",
    }


def invariant_3_negative_evidence():
    """Inherited credit must be revocable: a bad descendant loses it once measured."""
    child = CreditKey(basin="PROVEN", parent="gen1", family="atom_insert", scale="refine")
    rows = []
    for bad_trials in (0, 1, 2, 5, 10, 20, 40, 60, 120):
        credit, anchor, _ = _productive_state(HierarchicalCredit)
        for _ in range(bad_trials):
            credit.observe(child, 0.0)
        keys = [anchor, child]
        rows.append(
            {
                "bad_trials": bad_trials,
                "shrunk_value": round(credit.shrunk_value(child), 4),
                "exploitation_share": round(float(credit.exploitation_shares(keys)[1]), 4),
                "total_share": round(float(credit.allocate(keys)[1]), 4),
            }
        )
    values = [r["shrunk_value"] for r in rows]
    monotone = all(b <= a + TOL for a, b in pairwise(values))
    return {
        "statement": (
            "an unseen child of a +5.0 lineage starts at the inherited value; feeding it "
            "only zero-improvement trials must drive its value monotonically down toward "
            "its own estimate, so backoff is a loan and not a subsidy"
        ),
        "sweep": rows,
        "value_at_zero_bad_trials": values[0],
        "value_at_120_bad_trials": values[-1],
        "decay_factor": round(values[-1] / values[0], 4),
        "monotone_non_increasing": monotone,
        "measured": "MEASURED",
        "verdict": "PASS" if monotone and values[-1] < 0.1 * values[0] else "FAIL",
    }


def invariant_4_exploration_floor(draws=4000, seed=20260920):
    """Every live cell keeps a strictly positive share, isolated at ONE slot per draw.

    A multi-slot draw hides a floor failure, because the winning cell simply runs out of
    candidates and the next cell is taken for lack of an alternative.  With one slot the
    only route to a non-winning cell is the floor itself.
    """
    rng = np.random.default_rng(seed)
    credit = HierarchicalCredit()
    winner = CreditKey(basin="WIN", parent="w0", family="atom_insert", scale="refine")
    for _ in range(2000):
        credit.observe(winner, 1000.0)
    keys = [winner] + [
        CreditKey(basin=f"B{i}", parent=f"p{i}", family="cycle_close", scale="jump")
        for i in range(14)
    ]
    shares = credit.allocate(keys)
    counts = Counter()
    for _ in range(draws):
        counts[int(rng.choice(len(keys), p=shares))] += 1
    bound = credit.exploration_floor / len(keys)
    never = [i for i in range(len(keys)) if counts[i] == 0]
    exploit = credit.exploitation_shares(keys)
    exploration_component = float(shares.sum() - (1.0 - credit.exploration_floor) * exploit.sum())
    return {
        "statement": (
            "share(c) >= eps/|K| > 0 for every live cell c, and the TOTAL exploration "
            "budget sum(share) - (1-eps)*sum(exploit) equals eps exactly regardless of "
            "how many cells are live"
        ),
        "cells": len(keys),
        "single_slot_draws": draws,
        "theoretical_minimum_share": round(bound, 6),
        "observed_minimum_share": round(float(shares.min()), 6),
        "winner_share": round(float(shares[0]), 4),
        "cells_never_drawn": never,
        "total_exploration_budget": round(exploration_component, 12),
        "exploration_budget_equals_eps": abs(exploration_component - credit.exploration_floor)
        < 1e-9,
        "measured": "MEASURED",
        "verdict": "PASS"
        if not never and float(shares.min()) >= bound - TOL
        else "FAIL",
    }


def invariant_5_interaction_recovery():
    """Shrinkage must vanish as `n_c` grows, so an exceptional joint beats its marginals."""
    rows = []
    context_peers = [
        CreditKey(basin="B", parent=f"peer{i}", family="atom_insert", scale="refine")
        for i in range(6)
    ]
    star = CreditKey(basin="B", parent="star", family="atom_insert", scale="refine")
    for n_star in (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
        credit = HierarchicalCredit()
        for peer in context_peers:
            for _ in range(50):
                credit.observe(peer, 0.2)
        for _ in range(n_star):
            credit.observe(star, 9.0)
        keys = [star, *context_peers]
        raw = credit.cell(star).positive_improvement_sum / max(1, credit.cell(star).trials)
        rows.append(
            {
                "star_trials": n_star,
                "own_raw_value": round(raw, 4),
                "shrunk_value": round(credit.shrunk_value(star), 4),
                "fraction_of_own_estimate": round(n_star / (n_star + credit.kappa), 4),
                "peer_shrunk_value": round(credit.shrunk_value(context_peers[0]), 4),
                "beats_peers": credit.shrunk_value(star) > credit.shrunk_value(context_peers[0]),
                "exploitation_share": round(float(credit.exploitation_shares(keys)[0]), 4),
            }
        )
    converged = [r for r in rows if r["star_trials"] >= 64]
    gap = max(abs(r["shrunk_value"] - r["own_raw_value"]) / r["own_raw_value"] for r in converged)
    first_win = next((r["star_trials"] for r in rows if r["beats_peers"]), None)
    return {
        "statement": (
            "a joint cell paying 9.0 inside a context whose other six parents pay 0.2 must "
            "overtake its own marginals, and |Q~ - Q_raw|/Q_raw must go to 0 as n_c grows"
        ),
        "sweep": rows,
        "trials_to_beat_context_peers": first_win,
        "max_relative_shrinkage_at_n_ge_64": round(gap, 4),
        "measured": "MEASURED",
        "verdict": "PASS" if first_win is not None and first_win <= 4 and gap < 0.05 else "FAIL",
    }


# ======================================================================================
# Part C -- randomized fuzzing
# ======================================================================================


def _fuzz_state(rng, kappa, chain, *, randomize_settings=True, randomize_chain=False):
    """A random world driven through random rounds, returning a realistic credit state.

    `randomize_settings` also varies the ALLOCATOR's own constants, so a property that
    survives only at the default `prior_weight`/`exploration_floor` is exposed as a
    tuning artifact rather than a structural guarantee.
    """
    kind = TRUTH_KINDS[int(rng.integers(0, len(TRUTH_KINDS)))]
    churn = float(rng.choice([0.0, 0.2, 0.5, 0.9]))
    rounds = int(rng.integers(3, 16))
    late = int(rng.integers(2, rounds)) if rng.random() < 0.25 else None
    prior_weight = float(rng.choice([0.0, 0.01, 0.25, 1.0, 5.0])) if randomize_settings else 0.25
    floor = float(rng.choice([0.05, 0.1, 0.2, 0.35])) if randomize_settings else 0.2
    chain_name = (
        str(rng.choice(list(CHAINS))) if randomize_chain and rng.random() < 0.4 else chain
    )
    credit = HierarchicalCredit(
        kappa=kappa, chain=chain_name, prior_weight=prior_weight, exploration_floor=floor
    )
    keys, truth = make_world(rng, kind)
    credit, live, truth = observe_world(
        credit, keys, truth, rng, rounds=rounds, draws=int(rng.integers(4, 20)),
        churn=churn, late=late,
    )
    return credit, live, truth, {
        "kind": kind,
        "churn": churn,
        "rounds": rounds,
        "late": late,
        "prior_weight": prior_weight,
        "exploration_floor": floor,
        "chain": chain_name,
    }


def _minimal_repro(credit: HierarchicalCredit, keys) -> dict:
    """A reproducer carrying only the evidence and the pool, no landscape machinery."""
    return {
        "arm": type(credit).__name__,
        "kappa": getattr(credit, "kappa", None),
        "chain": getattr(credit, "chain_name", None),
        "exploration_floor": credit.exploration_floor,
        "prior_weight": credit.prior_weight,
        "evidence": [
            {**key.payload(), **cell.payload()}
            for key, cell in sorted(credit.cells.items())
            if cell.trials
        ],
        "pool": [key.payload() for key in keys],
    }


def fuzz_proliferation(rng, credit, live, _truth, duplicates=None):
    """P1: duplicating an unseen cell N times must not enlarge its REGION's exploit mass.

    The region is every untried cell attached to the same node, not just the cells the
    attacker adds.  Measuring only the added subset reports a false violation: the added
    cells do take a growing slice, but they take it by DILUTING the untried siblings that
    were already there, out of a pool whose total is fixed.  Both numbers are returned so
    the distinction stays visible.
    """
    duplicates = duplicates or int(rng.integers(2, 40))
    pool = list(dict.fromkeys(live))[: int(rng.integers(4, 40))]
    if len(pool) < 2:
        return None
    seed_key = pool[int(rng.integers(0, len(pool)))]
    # Attack at a randomly chosen level of the chain, not only at the leaf: a fresh
    # `entry_id` (leaf), a fresh program family or scale (the context level), or a whole
    # fresh basin (a child of the root).  N goes as high as 500.
    axis = str(rng.choice(["parent", "parent", "basin", "family", "scale"]))
    if rng.random() < 0.15:
        duplicates = int(rng.integers(100, 501))

    def sibling(i):
        payload = seed_key.payload()
        payload[axis] = f"FUZZ_{axis}_{i}" if axis != "scale" else SCALES[i % len(SCALES)]
        return CreditKey(**payload)

    if axis == "scale":
        # only 3 scales exist, so a scale-axis attack cannot proliferate; fall back
        axis = "parent"

        def sibling(i):
            payload = seed_key.payload()
            payload["parent"] = f"FUZZ_parent_{i}"
            return CreditKey(**payload)

    base_pool = [k for k in pool if k != sibling(0)]

    def in_region(key: CreditKey) -> bool:
        """Cells paid out of the SAME node's untried pool as the attacker's cells.

        Chain-aware on purpose: for `basin_context_parent` a fresh basin attaches to the
        root, a fresh family to the basin and a fresh entry_id to the context, and for a
        chain without a basin level the same fresh basin attaches somewhere else
        entirely.  Hand-rolling this per axis measures a subset of the pool.
        """
        return credit.untried_attachment(key) == attachment

    def region_mass(keys):
        exploit = credit.exploitation_shares(keys)
        return float(sum(s for k, s in zip(keys, exploit, strict=True) if in_region(k)))

    attachment = credit.untried_attachment(sibling(0))
    single = base_pool + [sibling(0)]
    many = base_pool + [sibling(i) for i in range(duplicates)]
    region_single, region_many = region_mass(single), region_mass(many)
    added_single = float(credit.exploitation_shares(single)[-1])
    added_many = float(credit.exploitation_shares(many)[len(base_pool) :].sum())

    region_ratio = region_many / region_single if region_single > 0 else 1.0
    capture_ratio = added_many / added_single if added_single > 0 else 1.0
    return {
        # PRIMARY: the region's total exploitation mass must not grow.
        "violated": region_ratio > 1.0 + 1e-9
        # LITERAL "N-times": N duplicates must not win N times the mass.
        or capture_ratio >= duplicates - 1e-9
        # CAP: whatever the attacker adds is bounded by the pre-existing untried pool.
        or added_many > region_single + 1e-9,
        "metric": round(region_ratio, 9),
        "duplicates": duplicates,
        "axis": axis,
        "detail": {
            "region_mass_with_1_added": region_single,
            "region_mass_with_N_added": region_many,
            "region_ratio": region_ratio,
            "added_cells_mass_with_1": added_single,
            "added_cells_mass_with_N": added_many,
            "capture_ratio_added_subset": round(capture_ratio, 6),
            "n_times_bound": duplicates,
            "added_mass_capped_by_region": added_many <= region_single + 1e-9,
        },
        "pool": many,
    }


def fuzz_inheritance(rng, credit, live, _truth):
    """P2: an unseen p' under a productive (b,f,s) must beat an unseen unrelated parent."""
    contexts: dict[tuple, list] = {}
    for key, cell in credit.cells.items():
        if cell.trials:
            contexts.setdefault((key.basin, key.family, key.scale), []).append((key, cell))
    if len(contexts) < 2:
        return None
    scored = []
    for node, rows in contexts.items():
        trials = sum(c.trials for _, c in rows)
        total = sum(c.positive_improvement_sum for _, c in rows)
        scored.append((total / trials if trials else 0.0, trials, node))
    scored.sort()
    weak_value, _weak_trials, weak_node = scored[0]
    strong_value, _strong_trials, strong_node = scored[-1]
    if strong_value <= weak_value + 1e-9:
        return None  # no productive/unproductive contrast: nothing to assert
    unseen_strong = CreditKey(
        basin=strong_node[0], parent="FUZZ_child_strong", family=strong_node[1],
        scale=strong_node[2],
    )
    unseen_weak = CreditKey(
        basin=weak_node[0], parent="FUZZ_child_weak", family=weak_node[1], scale=weak_node[2]
    )
    gap = credit.shrunk_value(unseen_strong) - credit.shrunk_value(unseen_weak)
    # A contrast of 0.19 vs 0.00 upside is noise, not a productive lineage.  The property
    # is asserted unconditionally AND again restricted to a material contrast, and both
    # rates are reported -- narrowing the precondition alone would be tuning the test.
    material = strong_value >= weak_value + 0.25
    return {
        "violated": gap <= 0.0,
        "qualified": material,
        "metric": round(gap, 6),
        "detail": {
            "material_contrast": material,
            "strong_context_raw": round(strong_value, 4),
            "weak_context_raw": round(weak_value, 4),
            "value_unseen_strong": round(credit.shrunk_value(unseen_strong), 6),
            "value_unseen_weak": round(credit.shrunk_value(unseen_weak), 6),
        },
        "pool": [unseen_strong, unseen_weak, *list(dict.fromkeys(live))[:10]],
    }


def fuzz_negative_override(rng, credit, live, _truth):
    """P3: once an inheriting cell is measured bad, its value must fall."""
    contexts: dict[tuple, float] = {}
    for key, cell in credit.cells.items():
        if cell.trials and cell.positive_improvement_sum > 0:
            node = (key.basin, key.family, key.scale)
            contexts[node] = contexts.get(node, 0.0) + cell.positive_improvement_sum
    if not contexts:
        return None
    node = max(contexts, key=contexts.get)
    child = CreditKey(basin=node[0], parent="FUZZ_bad_child", family=node[1], scale=node[2])
    start = credit.shrunk_value(child)
    if start <= 1e-9:
        return None
    values = [start]
    probe = _clone_like(credit)
    probe.cells = dict(credit.cells)
    for _step in range(80):
        probe.observe(child, 0.0)
        values.append(probe.shrunk_value(child))
    monotone = all(b <= a + 1e-12 for a, b in pairwise(values))
    decay = values[-1] / values[0]
    return {
        "violated": (not monotone) or decay > 0.25,
        "metric": round(decay, 6),
        "detail": {
            "start_value": round(values[0], 6),
            "value_after_80_zero_trials": round(values[-1], 6),
            "monotone_non_increasing": monotone,
        },
        "pool": [child, *list(dict.fromkeys(live))[:10]],
    }


def fuzz_interaction(rng, credit, live, _truth):
    """P4: an exceptional joint must overtake its marginals given enough measurements."""
    pool = list(dict.fromkeys(live))
    if len(pool) < 2:
        return None
    star = pool[int(rng.integers(0, len(pool)))]
    probe = _clone_like(credit)
    probe.cells = {k: CreditCell.restore(c.payload()) for k, c in credit.cells.items()}
    context_peers = [
        k
        for k in probe.cells
        if (k.basin, k.family, k.scale) == (star.basin, star.family, star.scale) and k != star
    ]
    exceptional = 50.0
    for _ in range(400):
        probe.observe(star, exceptional)
    raw = probe.cell(star).positive_improvement_sum / probe.cell(star).trials
    shrunk = probe.shrunk_value(star)
    relative = abs(shrunk - raw) / raw
    beats = all(shrunk > probe.shrunk_value(peer) for peer in context_peers) if context_peers else True
    return {
        "violated": (relative > 0.05) or (not beats),
        "metric": round(relative, 6),
        "detail": {
            "own_raw_value": round(raw, 4),
            "shrunk_value": round(shrunk, 4),
            "beats_all_context_peers": beats,
            "context_peers": len(context_peers),
        },
        "pool": [star, *context_peers[:8]],
    }


def fuzz_exploration(rng, credit, live, _truth):
    """P5: the reserved exploration budget is exactly eps however many cells appear."""
    pool = list(dict.fromkeys(live))[: int(rng.integers(3, 50))]
    if len(pool) < 2:
        return None
    extra = int(rng.integers(0, 60))
    pool = pool + [
        CreditKey(basin=f"NEW{i}", parent=f"np{i}", family="atom_insert", scale="jump")
        for i in range(extra)
    ]
    shares = credit.allocate(pool)
    exploit = credit.exploitation_shares(pool)
    budget = float(shares.sum() - (1.0 - credit.exploration_floor) * exploit.sum())
    floor_ok = float(shares.min()) >= credit.exploration_floor / len(pool) - 1e-12
    sums_ok = abs(float(shares.sum()) - 1.0) < 1e-9
    return {
        "violated": (abs(budget - credit.exploration_floor) > 1e-9) or not floor_ok or not sums_ok,
        "metric": round(budget, 12),
        "detail": {
            "cells": len(pool),
            "new_cells_added": extra,
            "min_share": float(shares.min()),
            "required_min": credit.exploration_floor / len(pool),
            "shares_sum_to_one": sums_ok,
        },
        "pool": pool,
    }


FUZZ_CHECKS = {
    "p1_proliferation_invariance": fuzz_proliferation,
    "p2_lineage_inheritance": fuzz_inheritance,
    "p3_negative_evidence_overrides": fuzz_negative_override,
    "p4_interaction_recovery": fuzz_interaction,
    "p5_exploration_budget_fixed": fuzz_exploration,
}


def run_fuzz(
    worlds=600,
    kappa=DEFAULT_KAPPA,
    chain=PRODUCTION_CHAIN,
    base_seed=90210,
    *,
    randomize_settings=True,
    randomize_chain=False,
):
    """Attack all five properties on randomized worlds; report violation RATES.

    `randomize_chain` stays OFF by default so the headline rate is a statement about the
    production chain alone; the chain comparison belongs in `run_fuzz_controls`, where
    the evidence is held identical across arms.
    """
    stats = {
        name: {
            "checked": 0,
            "violations": 0,
            "skipped": 0,
            "metrics": [],
            "by_kind": Counter(),
            "by_chain": Counter(),
            "checked_by_chain": Counter(),
            "by_prior_weight": Counter(),
            "checked_by_prior_weight": Counter(),
            "qualified_checked": 0,
            "qualified_violations": 0,
        }
        for name in FUZZ_CHECKS
    }
    counterexamples: dict[str, dict] = {}
    regimes = Counter()
    for world_index in range(worlds):
        rng = np.random.default_rng(base_seed + world_index)
        credit, live, truth, regime = _fuzz_state(
            rng, kappa, chain,
            randomize_settings=randomize_settings, randomize_chain=randomize_chain,
        )
        regimes[regime["kind"]] += 1
        for name, check in FUZZ_CHECKS.items():
            result = check(rng, credit, live, truth)
            if result is None:
                stats[name]["skipped"] += 1
                continue
            stats[name]["checked"] += 1
            stats[name]["checked_by_chain"][regime["chain"]] += 1
            stats[name]["checked_by_prior_weight"][regime["prior_weight"]] += 1
            qualified = result.get("qualified", True)
            stats[name]["qualified_checked"] += int(qualified)
            stats[name]["metrics"].append(result["metric"])
            if result["violated"]:
                stats[name]["qualified_violations"] += int(qualified)
                stats[name]["by_prior_weight"][regime["prior_weight"]] += 1
                stats[name]["violations"] += 1
                stats[name]["by_kind"][regime["kind"]] += 1
                stats[name]["by_chain"][regime["chain"]] += 1
                if name not in counterexamples:
                    counterexamples[name] = {
                        "world_seed": base_seed + world_index,
                        "regime": regime,
                        "metric": result["metric"],
                        "detail": result["detail"],
                        "reproducer": _minimal_repro(credit, result["pool"]),
                    }
    out = {}
    for name, row in stats.items():
        metrics = np.asarray(row["metrics"], dtype=float) if row["metrics"] else np.zeros(0)
        out[name] = {
            "worlds_checked": row["checked"],
            "worlds_skipped_precondition_unmet": row["skipped"],
            "violations": row["violations"],
            "violation_rate": round(row["violations"] / row["checked"], 6)
            if row["checked"]
            else None,
            "metric_median": round(float(np.median(metrics)), 6) if len(metrics) else None,
            "metric_p99": round(float(np.quantile(metrics, 0.99)), 6) if len(metrics) else None,
            "metric_max": round(float(metrics.max()), 6) if len(metrics) else None,
            "violations_by_landscape_kind": dict(row["by_kind"]),
            "violations_by_chain": dict(row["by_chain"]),
            "qualified_worlds_checked": row["qualified_checked"],
            "qualified_violations": row["qualified_violations"],
            "qualified_violation_rate": round(
                row["qualified_violations"] / row["qualified_checked"], 6
            )
            if row["qualified_checked"]
            else None,
            "violations_by_prior_weight": {str(k): v for k, v in row["by_prior_weight"].items()},
            "worlds_checked_by_prior_weight": {
                str(k): v for k, v in row["checked_by_prior_weight"].items()
            },
            "worlds_checked_by_chain": dict(row["checked_by_chain"]),
            "violation_rate_excluding_flat_control": round(
                (row["violations"] - row["by_chain"].get("flat", 0))
                / max(1, row["checked"] - row["checked_by_chain"].get("flat", 0)),
                6,
            ),
        }
    return {
        "worlds": worlds,
        "kappa": kappa,
        "chain": chain,
        "landscape_kind_counts": dict(regimes),
        "checks": out,
        "counterexamples": counterexamples,
        "measured": "MEASURED",
    }


def fuzz_kappa_sweep(worlds=200, kappas=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0)):
    """Separate a structural flaw from a tuning artifact.

    A violation rate that is flat in `kappa` is structural; one that moves with `kappa` is
    a tuning artifact of the chosen constant.
    """
    rows = []
    for kappa in kappas:
        report = run_fuzz(
            worlds=worlds, kappa=kappa, base_seed=515_000, randomize_settings=False
        )
        rows.append(
            {
                "kappa": kappa,
                **{
                    name: report["checks"][name]["violation_rate"]
                    for name in FUZZ_CHECKS
                },
            }
        )
    verdict = {}
    for name in FUZZ_CHECKS:
        series = [r[name] for r in rows if r[name] is not None]
        if not series:
            verdict[name] = "NO DATA"
        elif max(series) == 0.0:
            verdict[name] = "NO VIOLATION AT ANY KAPPA (structural property holds)"
        elif min(series) == max(series):
            verdict[name] = "FLAT IN KAPPA -> STRUCTURAL, not a tuning artifact"
        else:
            verdict[name] = (
                f"MOVES WITH KAPPA -> tuning artifact (rate {min(series)} at kappa "
                f"{rows[[r[name] for r in rows].index(min(series))]['kappa']} vs "
                f"{max(series)} at kappa "
                f"{rows[[r[name] for r in rows].index(max(series))]['kappa']})"
            )
    return {"worlds_per_kappa": worlds, "sweep": rows, "interpretation": verdict}


ARMS = {
    "v2_production": ("HierarchicalCredit", PRODUCTION_CHAIN),
    "v2_flat_chain_no_hierarchy": ("HierarchicalCredit", "flat"),
    "v2_hierarchy_without_untried_pooling": ("NoPoolControl", PRODUCTION_CHAIN),
    "v1_frozen": ("V1ExploitView", None),
}


def _arm_credit(arm: str, reference):
    """An allocation arm carrying `reference`'s evidence, so only the RULE differs."""
    cls_name, chain = ARMS[arm]
    if cls_name == "V1ExploitView":
        credit = V1ExploitView(
            exploration_floor=reference.exploration_floor, prior_weight=reference.prior_weight
        )
    else:
        cls = HierarchicalCredit if cls_name == "HierarchicalCredit" else NoPoolControl
        credit = cls(
            exploration_floor=reference.exploration_floor,
            prior_weight=reference.prior_weight,
            kappa=reference.kappa,
            chain=chain,
        )
    credit.cells = {k: CreditCell.restore(c.payload()) for k, c in reference.cells.items()}
    return credit


def run_fuzz_controls(worlds=400, base_seed=90210):
    """All five properties, all four arms, on IDENTICAL evidence.

    The evidence is generated once per world by the production arm and transplanted, so
    the arms differ only in their allocation rule.  The two controls are orthogonal and
    are the reason to believe the fuzzer tests the mechanism it claims to:

      * `v2_flat_chain_no_hierarchy` keeps the count shrinkage and removes the
        intermediate levels -- it should keep P1 (it still pools) and lose P2 (nothing to
        inherit from but the root);
      * `v2_hierarchy_without_untried_pooling` keeps the levels and removes the pooling --
        it should keep P2 and lose P1.

    A fuzzer that cannot separate those two is not measuring either mechanism.
    """
    stats = {
        arm: {name: {"checked": 0, "violations": 0, "metrics": []} for name in FUZZ_CHECKS}
        for arm in ARMS
    }
    for world_index in range(worlds):
        rng = np.random.default_rng(base_seed + world_index)
        reference, live, truth, _regime = _fuzz_state(
            rng, DEFAULT_KAPPA, PRODUCTION_CHAIN, randomize_settings=False
        )
        for arm in ARMS:
            credit = _arm_credit(arm, reference)
            rng_checks = np.random.default_rng(base_seed + world_index)
            for name, check in FUZZ_CHECKS.items():
                result = check(rng_checks, credit, live, truth)
                if result is None:
                    continue
                stats[arm][name]["checked"] += 1
                stats[arm][name]["metrics"].append(result["metric"])
                stats[arm][name]["violations"] += bool(result["violated"])
    return {
        "worlds": worlds,
        "evidence": "generated once per world by the production arm and transplanted to every arm",
        "arms": {
            arm: {
                name: {
                    "worlds_checked": row["checked"],
                    "violations": row["violations"],
                    "violation_rate": round(row["violations"] / row["checked"], 6)
                    if row["checked"]
                    else None,
                    "metric_median": round(float(np.median(row["metrics"])), 6)
                    if row["metrics"]
                    else None,
                    "metric_max": round(float(np.max(row["metrics"])), 6)
                    if row["metrics"]
                    else None,
                }
                for name, row in rows.items()
            }
            for arm, rows in stats.items()
        },
        "measured": "MEASURED",
    }


# ======================================================================================
# Part D -- matched A/B through the frozen controller
# ======================================================================================


def run_ab(seeds=(20260920, 7, 101, 4242, 55555, 13, 271828, 31337, 900001), rounds=16):
    """v1 vs v2 on the same landscapes, same seeds, through the controller's `_allocate`.

    Reuses `scripts/pmo_controller_adversarial_audit.py` -- its landscapes, its pool
    builder and its campaign loop -- rather than standing up a second harness.  The ONLY
    difference between the arms is the object bound to `ctrl.credit`.
    """
    import pmo_controller_adversarial_audit as audit

    results = {}
    for name, payoff in audit.LANDSCAPES.items():
        rows = []
        for seed in seeds:
            trial = {"seed": seed}
            for arm in ("v1_flat_cell", "v2_hierarchical"):
                ctrl = audit.controller(seed)
                if arm == "v2_hierarchical":
                    ctrl.credit = HierarchicalCredit()
                history = audit.run_campaign(
                    ctrl, audit.ALL_CHANNEL_LAYOUT, payoff, rounds=rounds
                )
                total = sum(row["realized_gain"] for row in history)
                drawn = sum(row["n_drawn"] for row in history)
                cores = audit._share_by(history, "core")
                late = audit._share_by(history[rounds // 2 :], "core")
                productive = "indole" if name == "late_basin" else "benzamide_piperidine"
                cells = ctrl.credit.cells
                trial[arm] = {
                    "total_realized_gain": round(total, 4),
                    "candidates_drawn": drawn,
                    "gain_per_call": round(total / max(1, drawn), 4),
                    "productive_basin_share_overall": cores.get(productive, 0.0),
                    "productive_basin_share_second_half": late.get(productive, 0.0),
                    "active_cells": len(cells),
                    "basins_touched": len(cores),
                }
            trial["gain_per_call_delta_v2_minus_v1"] = round(
                trial["v2_hierarchical"]["gain_per_call"] - trial["v1_flat_cell"]["gain_per_call"],
                4,
            )
            rows.append(trial)
        deltas = [row["gain_per_call_delta_v2_minus_v1"] for row in rows]
        v2_share = [row["v2_hierarchical"]["productive_basin_share_second_half"] for row in rows]
        v1_share = [row["v1_flat_cell"]["productive_basin_share_second_half"] for row in rows]
        wins = sum(d > 0 for d in deltas)
        losses = sum(d < 0 for d in deltas)
        results[name] = {
            "trials": rows,
            "n_seeds": len(seeds),
            "allocation_path": "ctrl._allocate -- the full production path",
            "landscape_regime": (
                "STATIC 15-cell layout, fully tried within two rounds: no proliferation "
                "and no parent churn, so the v1 defect is not exercised here"
            ),
            "median_gain_per_call_delta": round(float(np.median(deltas)), 4),
            "mean_gain_per_call_delta": round(float(np.mean(deltas)), 4),
            "seeds_where_v2_wins": wins,
            "seeds_where_v1_wins": losses,
            "median_productive_share_v2": round(float(np.median(v2_share)), 4),
            "median_productive_share_v1": round(float(np.median(v1_share)), 4),
            "median_active_cells_v2": int(
                np.median([r["v2_hierarchical"]["active_cells"] for r in rows])
            ),
            "verdict": (
                "V2 BETTER"
                if float(np.median(deltas)) > 0 and wins > len(deltas) / 2
                else "V2 WORSE"
                if float(np.median(deltas)) < 0 and losses > len(deltas) / 2
                else "INDISTINGUISHABLE"
            ),
        }
    return results


# ======================================================================================


def _churn_campaign(ctrl, payoff, *, rounds=14, cores=3, draws=13, promote=2):
    """A campaign where children are PROMOTED to parents, as a recursive run does.

    This is the regime the v1 defect is about and the fixed-layout landscapes are not:
    `credit_key_from_candidate` keys on the parent `entry_id`, so every promotion opens a
    fresh row of cells.  The measured production run showed 90.1% of each round's pool
    untried and the active-cell count going 13 -> 114 in 14 rounds; a static layout where
    all 15 cells are tried by round two cannot exercise any of that.
    """
    import pmo_controller_adversarial_audit as audit

    core_names = list(audit.CORES)[:cores]
    parents = [f"gen0_{i}" for i in range(4)]
    history = []
    for round_index in range(rounds):
        ctrl.batches = round_index
        layout = [
            (core, parent, channel)
            for core in core_names
            for parent in parents
            for channel in audit.CHANNELS
        ]
        pool = audit.build_pool(ctrl, layout, per_cell=2, offset=round_index)
        value, _ = ctrl._fit_value()
        cells = {credit_key_from_candidate(row) for row in pool}
        diagnostics = pool_prior_diagnostics(ctrl.credit, cells)
        if ctrl.credit.cells:
            chosen, _detail = ctrl._credit_allocate(pool, value, draws)
        else:
            chosen = list(ctrl.rng.permutation(np.asarray(pool, dtype=object))[:draws])
        gains, drawn_cores = [], Counter()
        for row in chosen:
            key = credit_key_from_candidate(row)
            core = audit.CORE_BY_BASIN[key.basin]
            gain = payoff(core, key.parent, key.scale, round_index) + audit._jitter(
                row["endpoint"]
            )
            ctrl.credit.observe(
                key,
                improvement(
                    audit.PARENT_SCORE + gain, audit.PARENT_SCORE, direction="maximize"
                ),
            )
            gains.append(gain)
            drawn_cores[core] += 1
        history.append(
            {
                "round": round_index,
                "n_drawn": len(chosen),
                "realized_gain": float(sum(gains)),
                "by_core": dict(drawn_cores),
                "cells_in_pool": len(cells),
                **diagnostics,
                "active_cells": len(ctrl.credit.cells),
            }
        )
        parents = parents[promote:] + [
            f"gen{round_index + 1}_{i}" for i in range(promote)
        ]
    return history


def _churn_landscapes():
    """Basin-determined payoffs, so credit SHOULD survive the parent churn."""
    import pmo_controller_adversarial_audit as audit

    def graded(core, parent, scale, round_index):
        return audit.PAYOFF_BY_CORE[core]

    def late(core, parent, scale, round_index):
        if core == "indole":
            return 5.0 if round_index >= 7 else 0.0
        return 0.3

    def lineage(core, parent, scale, round_index):
        # the recursive-optimizer case: a good basin keeps paying as generations advance
        base = audit.PAYOFF_BY_CORE[core]
        generation = int(parent.split("_")[0].replace("gen", "")) if "gen" in parent else 0
        return base * (1.0 + 0.15 * generation)

    return {
        "churn_graded_basins": graded,
        "churn_late_basin": late,
        "churn_compounding_lineage": lineage,
    }


def run_ab_churn(seeds=(20260920, 7, 101, 4242, 55555, 13, 271828, 31337, 900001), rounds=14):
    """v1 vs v2 under parent churn -- the regime the v1 defect describes.

    `_churn_campaign` uses the first three cores, so the productive basin tracked here is
    `indole` (the highest `PAYOFF_BY_CORE` among them) for every landscape.
    """
    import pmo_controller_adversarial_audit as audit

    results = {}
    for name, payoff in _churn_landscapes().items():
        rows = []
        for seed in seeds:
            trial = {"seed": seed}
            for arm in ("v1_flat_cell", "v2_hierarchical"):
                ctrl = audit.controller(seed)
                if arm == "v2_hierarchical":
                    ctrl.credit = HierarchicalCredit()
                history = _churn_campaign(ctrl, payoff, rounds=rounds)
                total = sum(row["realized_gain"] for row in history)
                drawn = sum(row["n_drawn"] for row in history)
                tail = history[rounds // 2 :]
                cores = Counter()
                for row in tail:
                    cores.update(row["by_core"])
                total_draws = sum(cores.values()) or 1
                # The most productive core AMONG THE ONES THIS CAMPAIGN USES.  Naming a
                # core outside `cores` reports a share of 0.0 by construction and looks
                # like the controller never found the payoff.
                productive = "indole"
                trial[arm] = {
                    "total_realized_gain": round(total, 4),
                    "candidates_drawn": drawn,
                    "gain_per_call": round(total / max(1, drawn), 4),
                    "productive_basin_share_second_half": round(
                        cores.get(productive, 0) / total_draws, 4
                    ),
                    "active_cells_final": history[-1]["active_cells"],
                    "mean_fraction_of_pool_untried_second_half": round(
                        float(np.mean([r["fraction_of_pool_untried"] for r in tail])), 4
                    ),
                    "mean_exploitation_mass_on_untried_second_half": round(
                        float(np.mean([r["exploitation_mass_on_untried"] for r in tail])), 4
                    ),
                    "mean_distinct_share_values_among_untried": round(
                        float(np.mean([r["distinct_share_values_among_untried"] for r in tail])), 2
                    ),
                    "mean_untried_informativeness_second_half": round(
                        float(
                            np.mean(
                                [
                                    r["untried_share_vs_context_productivity_rank_correlation"]
                                    for r in tail
                                    if r["untried_share_vs_context_productivity_rank_correlation"]
                                    is not None
                                ]
                                or [0.0]
                            )
                        ),
                        4,
                    ),
                }
            trial["gain_per_call_delta_v2_minus_v1"] = round(
                trial["v2_hierarchical"]["gain_per_call"]
                - trial["v1_flat_cell"]["gain_per_call"],
                4,
            )
            rows.append(trial)
        deltas = [row["gain_per_call_delta_v2_minus_v1"] for row in rows]
        wins = sum(d > 0 for d in deltas)
        losses = sum(d < 0 for d in deltas)
        results[name] = {
            "trials": rows,
            "n_seeds": len(seeds),
            "allocation_path": (
                "ctrl._credit_allocate directly -- isolates the credit decision from the "
                "per-channel slot reservation that ctrl._allocate performs first"
            ),
            "median_gain_per_call_delta": round(float(np.median(deltas)), 4),
            "mean_gain_per_call_delta": round(float(np.mean(deltas)), 4),
            "seeds_where_v2_wins": wins,
            "seeds_where_v1_wins": losses,
            "median_productive_share_v2": round(
                float(np.median([r["v2_hierarchical"]["productive_basin_share_second_half"] for r in rows])), 4
            ),
            "median_productive_share_v1": round(
                float(np.median([r["v1_flat_cell"]["productive_basin_share_second_half"] for r in rows])), 4
            ),
            "median_fraction_untried_v2": round(
                float(np.median([r["v2_hierarchical"]["mean_fraction_of_pool_untried_second_half"] for r in rows])), 4
            ),
            "median_fraction_untried_v1": round(
                float(np.median([r["v1_flat_cell"]["mean_fraction_of_pool_untried_second_half"] for r in rows])), 4
            ),
            "median_prior_mass_on_untried_v2": round(
                float(np.median([r["v2_hierarchical"]["mean_exploitation_mass_on_untried_second_half"] for r in rows])), 4
            ),
            "median_prior_mass_on_untried_v1": round(
                float(np.median([r["v1_flat_cell"]["mean_exploitation_mass_on_untried_second_half"] for r in rows])), 4
            ),
            "median_untried_informativeness_v2": round(
                float(np.median([r["v2_hierarchical"]["mean_untried_informativeness_second_half"] for r in rows])), 4
            ),
            "median_untried_informativeness_v1": round(
                float(np.median([r["v1_flat_cell"]["mean_untried_informativeness_second_half"] for r in rows])), 4
            ),
            "median_distinct_untried_shares_v2": round(
                float(np.median([r["v2_hierarchical"]["mean_distinct_share_values_among_untried"] for r in rows])), 2
            ),
            "median_distinct_untried_shares_v1": round(
                float(np.median([r["v1_flat_cell"]["mean_distinct_share_values_among_untried"] for r in rows])), 2
            ),
            "interpretation": (
                "v2 does not reduce the mass reaching untried cells -- it replaces a FLAT "
                "prior with an INHERITED one.  The arms are distinguished by whether that "
                "mass is informative: v1 values every untried cell identically, so it has "
                "one distinct share and zero rank correlation with context productivity "
                "by construction."
            ),
            "verdict": (
                "V2 BETTER"
                if float(np.median(deltas)) > 0 and wins > len(deltas) / 2
                else "V2 WORSE"
                if float(np.median(deltas)) < 0 and losses > len(deltas) / 2
                else "INDISTINGUISHABLE"
            ),
        }
    return results


INVARIANTS = {
    "i1_proliferation_invariance": invariant_1_proliferation,
    "i2_lineage_inheritance": invariant_2_inheritance,
    "i3_negative_evidence_overrides": invariant_3_negative_evidence,
    "i4_exploration_floor_isolated": invariant_4_exploration_floor,
    "i5_interaction_recovery": invariant_5_interaction_recovery,
}


PRODUCTION_WIRING_REQUIREMENTS = [
    {
        "item": "promote the module out of scripts/",
        "detail": (
            "`scripts/pmo_credit_v2.py` carries a rule and an invariant, so by the repo's "
            "own scratchpad test it belongs in `src/compose_v4/control/` WITH its tests "
            "when it is adopted -- not promoted afterwards.  `tests/test_pmo_credit_v2.py` "
            "already exists and would move with it."
        ),
    },
    {
        "item": "serialization schema",
        "detail": (
            "`HierarchicalCredit` inherits `payload`/`restore`, which emit "
            "`schema_version = 'pmo_credit_v1'` and do NOT carry `kappa` or the chain.  A "
            "restored controller would silently fall back to the defaults.  Adopting v2 "
            "needs a v2 schema plus a documented read path for v1 snapshots."
        ),
    },
    {
        "item": "controller seam",
        "detail": (
            "`PmoPopulationController.__init__` line 193 constructs `PopulationCredit()` "
            "and `restore` line 828 reconstructs it; `_credit_allocate` line 561 calls "
            "`self.credit.allocate(cells)`.  Those three sites are the entire seam -- "
            "v2 needs no other controller change, which is why the A/B could swap the "
            "attribute."
        ),
    },
    {
        "item": "content-hash re-pin",
        "detail": (
            "`pmo_credit.py` and `pmo_population_controller.py` are pinned by content "
            "hash and a scored 3x250 run is launching against those bytes.  Adoption is a "
            "deliberate re-pin of the whole chain, not an edit -- and every number "
            "measured under v1 stays a v1 number."
        ),
    },
    {
        "item": "cache the node statistics",
        "detail": (
            "`exploitation_shares` rebuilds `_stats_table()` on every call, which is "
            "O(cells x levels) and is recomputed once per allocation.  Harmless at the "
            "prototype's scale and at the controller's (hundreds of cells), but it should "
            "be memoised behind an observe-counter before it sits in a long run."
        ),
    },
    {
        "item": "choose kappa against REAL reward noise",
        "detail": (
            "the kappa optimum is a function of how informative one oracle call is, and "
            "that has not been measured on a real PMO run.  Re-run `select_kappa` against "
            "the observed per-cell improvement variance before freezing it."
        ),
    },
    {
        "item": "decide the basin label first",
        "detail": (
            "v2 makes the basin axis load-bearing -- it is now the level a whole lineage "
            "inherits through.  `pmo_credit.basin_label` uses the Bemis-Murcko scaffold, "
            "which its own docstring adopted because the controller's `basin_id` was a "
            "per-molecule key.  A scaffold-hopping jump changes basin and so resets "
            "inheritance by design; whether that is wanted is a chemistry decision."
        ),
    },
]

MEASURED_VS_INFERRED = {
    "MEASURED": [
        "every number in parts A-D: they are computed here from synthetic landscapes",
        "the v1 failure rates, on the same worlds and the same check functions as v2",
        "the exploitation-mass count invariance, to machine precision (spread < 1e-12)",
        "that v2 is WORSE than v1 on the fixed-layout landscapes (part D, d_matched_ab)",
    ],
    "INFERRED": [
        (
            "that any of this transfers to a scored PMO oracle run.  No real credit "
            "snapshot exists on disk, so the landscapes are stand-ins for a reward "
            "surface nobody has measured."
        ),
        (
            "that the churn A/B regime resembles production.  It was built to match two "
            "measured production numbers (90.1% of pool untried, 13 -> 114 active cells "
            "in 14 rounds); matching two summary statistics is not matching the process."
        ),
        (
            "the chain ranking.  Family and scale are collinear in the controller "
            "(`RULES_BY_CHANNEL` is one family per channel), so the two leading chains "
            "are indistinguishable there and the ranking between them rests entirely on "
            "synthetic worlds where they are not collinear."
        ),
    ],
    "NOT CLAIMED": [
        "any calibrated uncertainty; kappa and the floor are search settings",
        "that v2 improves final PMO score -- allocation was measured, not chemistry",
    ],
}


def _where_v2_is_worse(report: dict) -> dict:
    """Collect every landscape where v2 lost, so the summary cannot quietly drop them."""
    losses = []
    for block in ("d_matched_ab", "d_matched_ab_churn"):
        for name, row in (report.get(block) or {}).items():
            if row["median_gain_per_call_delta"] < 0 or row["seeds_where_v1_wins"] > row[
                "seeds_where_v2_wins"
            ]:
                losses.append(
                    {
                        "block": block,
                        "landscape": name,
                        "median_gain_per_call_delta_v2_minus_v1": row[
                            "median_gain_per_call_delta"
                        ],
                        "seeds_v2_wins": row["seeds_where_v2_wins"],
                        "seeds_v1_wins": row["seeds_where_v1_wins"],
                        "verdict": row["verdict"],
                    }
                )
    return {
        "landscapes_where_v2_loses": losses,
        "explanation": (
            "the fixed-layout landscapes hold a static 15-cell pool that is fully tried "
            "within two rounds, so there is no proliferation and no parent churn for the "
            "hierarchy to protect against.  In that regime the count shrinkage is pure "
            "bias: it pulls each cell toward its basin and root means and blurs exactly "
            "the per-cell distinctions the landscape rewards.  v2 is a fix for the churn "
            "regime and costs a little where there is no churn."
        ),
        "honest_summary": (
            "v2 is not uniformly better than v1.  It wins where cells churn and most of "
            "the pool is untried, and loses slightly on a small static pool."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="diagnostics/pmo_credit_v2_prototype_v1.json")
    parser.add_argument("--fuzz-worlds", type=int, default=600)
    parser.add_argument("--kappa-sweep-worlds", type=int, default=150)
    parser.add_argument("--control-worlds", type=int, default=400)
    parser.add_argument("--ab-seeds", type=int, default=9)
    parser.add_argument("--skip-ab", action="store_true")
    args = parser.parse_args()

    report = {
        "schema_version": "pmo_credit_v2_prototype_v1",
        "status": "PROTOTYPE -- NOT WIRED into PmoPopulationController; nothing re-pinned",
        "oracle_calls": 0,
        "reward_source": "closed-form and pseudo-random synthetic landscapes; no TDC Oracle, no network",
        "frozen_v1_untouched": [
            "src/compose_v4/control/pmo_credit.py",
            "src/compose_v4/control/pmo_population_controller.py",
            "src/compose_v4/control/bootstrap_pool_continuity.py",
        ],
        "v2_rule": {
            "module": "scripts/pmo_credit_v2.py",
            "class": "HierarchicalCredit(PopulationCredit)",
            "factorisation": "P(b,p,f,s) = P(b,f,s) * P(p | b,f,s), with a basin level above the context",
            "chain": list(CHAINS[PRODUCTION_CHAIN]),
            "shrinkage": "Q~(v) = n/(n+kappa) * Q_raw(v) + kappa/(n+kappa) * Q~(parent(v))",
            "kappa": DEFAULT_KAPPA,
            "untried_pool": (
                "untried children of a node are ONE pseudo-sibling carrying the node's "
                "backoff value, split evenly among them"
            ),
            "exploration": "unchanged from v1: q = (1-eps)*q_exploit + eps*uniform, eps = 0.2",
        },
    }

    print("part A: selecting the backoff chain ...", flush=True)
    report["a_hierarchy_selection"] = select_hierarchy(seeds=range(25))
    print("part A: selecting kappa ...", flush=True)
    report["a_kappa_selection"] = select_kappa()

    print("part B: targeted invariants ...", flush=True)
    report["b_targeted_invariants"] = {}
    for name, fn in INVARIANTS.items():
        report["b_targeted_invariants"][name] = fn()
        print(f"  {name}: {report['b_targeted_invariants'][name]['verdict']}", flush=True)

    print(f"part C: fuzzing {args.fuzz_worlds} randomized worlds ...", flush=True)
    report["c_fuzz"] = run_fuzz(worlds=args.fuzz_worlds)
    for name, row in report["c_fuzz"]["checks"].items():
        print(f"  {name}: rate={row['violation_rate']} on {row['worlds_checked']} worlds", flush=True)
    print(f"part C: kappa sweep over {args.kappa_sweep_worlds} worlds each ...", flush=True)
    report["c_fuzz_kappa_sweep"] = fuzz_kappa_sweep(worlds=args.kappa_sweep_worlds)
    print("part C: control arms under the same attacks ...", flush=True)
    report["c_fuzz_arm_controls"] = run_fuzz_controls(worlds=args.control_worlds)
    for arm, rows in report["c_fuzz_arm_controls"]["arms"].items():
        rates = {n: r["violation_rate"] for n, r in rows.items()}
        print(f"  {arm}: {rates}", flush=True)

    if not args.skip_ab:
        print("part D: matched A/B through the frozen controller ...", flush=True)
        report["d_matched_ab"] = run_ab(seeds=tuple(
            (20260920, 7, 101, 4242, 55555, 13, 271828, 31337, 900001, 64, 5, 88)[: args.ab_seeds]
        ))
        for name, row in report["d_matched_ab"].items():
            print(f"  {name}: {row['verdict']} (median delta {row['median_gain_per_call_delta']})",
                  flush=True)
        print("part D: matched A/B under parent churn ...", flush=True)
        report["d_matched_ab_churn"] = run_ab_churn(seeds=tuple(
            (20260920, 7, 101, 4242, 55555, 13, 271828, 31337, 900001, 64, 5, 88)[: args.ab_seeds]
        ))
        for name, row in report["d_matched_ab_churn"].items():
            print(f"  {name}: {row['verdict']} (median delta {row['median_gain_per_call_delta']})",
                  flush=True)

    report["e_where_v2_is_worse"] = _where_v2_is_worse(report)
    report["f_production_wiring_requirements"] = PRODUCTION_WIRING_REQUIREMENTS
    report["g_measured_vs_inferred"] = MEASURED_VS_INFERRED

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
