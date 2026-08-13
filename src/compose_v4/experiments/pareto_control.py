"""Target-free Pareto / preference control over the frozen COMPOSE process.

Four arms and one scalarization, with the cost ledger that makes them
comparable. NO learned h_phi: this lane establishes whether future-aware
preference control buys anything at all before anyone amortizes it, because
building the amortizer first is a mistake this project has already made once.

WHY THE PROCESS IS INJECTED
---------------------------
The arms take a `Process` and an `Objectives` protocol rather than importing the
kernel. That is not decoration -- it is what lets the control logic be tested
EXACTLY, on a toy state graph with hand-computable optima, without RDKit, the
rewrite kernel or the frozen R_theta checkpoint. Two of the five instrument
defects already found in this project ("a verified arm that was secretly
greedy", "an action selected by argmax V_G then scored by V_G") were bugs in
control logic that a toy-graph test would have caught immediately.

THE COST LEDGER IS THE POINT, NOT BOOKKEEPING
---------------------------------------------
Hypervolume has a well-known inflation channel: an arm that simply generates
more molecules scores better without controlling anything better. So every arm
returns a `CostLedger` with THREE separate counters, and a comparison is only
admissible at matched budget:

    native_oracle_calls   distinct molecules scored -- the benchmark convention
    raw_oracle_calls      every scoring invocation, including cache hits and
                          every candidate evaluated inside a lookahead rollout
    kernel_calls          successor enumerations; dominates wall time

Native counting alone hides what verified control spends inside rollouts; raw
counting alone hides caching efficiency. Reporting one without the other is a
reporting choice that can decide the winner, so both are always carried.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np

#: Augmented-Chebyshev tie-breaking weight. Small enough not to reorder the max
#: term, large enough to reject weakly-dominated points.
CHEBYSHEV_RHO = 1e-3


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

class Process(Protocol):
    """The frozen COMPOSE process: legal successors and their reference law."""

    def successors(self, key: str) -> list[tuple[str, float]]:
        """Distinct canonical successors of `key` with R_theta probabilities."""


class Objectives(Protocol):
    """Objective vector for a molecule, in held-in IQR units, MAXIMIZED."""

    def z(self, key: str) -> np.ndarray:
        ...


# ---------------------------------------------------------------------------
# Cost ledger
# ---------------------------------------------------------------------------

@dataclass
class CostLedger:
    native_oracle_calls: int = 0
    raw_oracle_calls: int = 0
    kernel_calls: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"native_oracle_calls": self.native_oracle_calls,
                "raw_oracle_calls": self.raw_oracle_calls,
                "kernel_calls": self.kernel_calls}

    def __add__(self, other: "CostLedger") -> "CostLedger":
        return CostLedger(self.native_oracle_calls + other.native_oracle_calls,
                          self.raw_oracle_calls + other.raw_oracle_calls,
                          self.kernel_calls + other.kernel_calls)


class MeteredProcess:
    """Caches enumeration and scoring, and counts both conventions honestly.

    The cache is what makes native and raw counts differ, and that difference is
    a reported quantity, not an implementation detail.
    """

    def __init__(self, process: Process, objectives: Objectives) -> None:
        self._process = process
        self._objectives = objectives
        self._successors: dict[str, list[tuple[str, float]]] = {}
        self._z: dict[str, np.ndarray] = {}
        self.ledger = CostLedger()

    def successors(self, key: str) -> list[tuple[str, float]]:
        if key not in self._successors:
            self.ledger.kernel_calls += 1
            self._successors[key] = list(self._process.successors(key))
        return self._successors[key]

    def z(self, key: str) -> np.ndarray:
        self.ledger.raw_oracle_calls += 1
        if key not in self._z:
            self.ledger.native_oracle_calls += 1
            self._z[key] = np.asarray(self._objectives.z(key), dtype=float)
        return self._z[key]

    def z_many(self, keys: Sequence[str]) -> np.ndarray:
        return np.stack([self.z(k) for k in keys]) if keys else np.zeros((0, 2))


# ---------------------------------------------------------------------------
# Scalarization
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Scalarization:
    """Augmented weighted Chebyshev, MINIMIZED.

    Chebyshev rather than a weighted sum because every Pareto point is the
    Chebyshev optimum for SOME weight, whereas a weighted sum can only select
    points on the convex hull of the front. An experiment scalarized only by a
    weighted sum would systematically miss nonconvex regions and would report a
    smaller achievable front than actually exists.
    """

    utopia: np.ndarray
    rho: float = CHEBYSHEV_RHO

    def __call__(self, z: np.ndarray, weight: float) -> np.ndarray:
        z = np.atleast_2d(z)
        gap = self.utopia[None, :] - z
        w = np.array([weight, 1.0 - weight])
        return np.max(w[None, :] * gap, axis=1) + self.rho * np.sum(gap, axis=1)

    def binding_index(self, z: np.ndarray, weight: float) -> np.ndarray:
        gap = self.utopia[None, :] - np.atleast_2d(z)
        w = np.array([weight, 1.0 - weight])
        return np.argmax(w[None, :] * gap, axis=1)


@dataclass(frozen=True)
class WeightedSum:
    """Secondary scalarization, reported for diagnosis only. MINIMIZED, so its
    sign matches Chebyshev and the two can be compared without a flip."""

    def __call__(self, z: np.ndarray, weight: float) -> np.ndarray:
        z = np.atleast_2d(z)
        w = np.array([weight, 1.0 - weight])
        return -np.sum(w[None, :] * z, axis=1)


# ---------------------------------------------------------------------------
# Trajectory record
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    arm: str
    weight: float
    states: list[str]
    endpoint: str
    endpoint_z: np.ndarray
    complete: bool
    ledger: CostLedger
    overrode_greedy: int = 0
    decisions: list[dict] = field(default_factory=list)

    @property
    def actions(self) -> tuple[str, ...]:
        """The committed action sequence -- the object instrument gate D2
        compares between two arms. Two arms whose action sequences are identical
        on every source are the same arm, whatever they are called."""
        return tuple(self.states[1:])


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def _argmin_stable(values: np.ndarray, keys: Sequence[str]) -> int:
    """Deterministic argmin: ties break on the canonical key, never on
    enumeration order, so a rerun commits the same trajectory."""
    best = int(np.argmin(values))
    tied = np.flatnonzero(values <= values[best] + 1e-12)
    return int(min(tied, key=lambda i: keys[i]))


def unguided_run(metered: MeteredProcess, start: str, budget: int,
                 seed: int) -> Trajectory:
    """Arm 1. Sample `budget` edits from the frozen reference law.

    Preference-blind by construction: whatever coverage this achieves over the
    preference grid is chance, which is exactly what makes it the floor for P1.
    """
    rng = random.Random(seed)
    states = [start]
    current = start
    complete = True
    for _ in range(budget):
        rows = metered.successors(current)
        if not rows:
            complete = False
            break
        keys = [r[0] for r in rows]
        probs = np.array([r[1] for r in rows], dtype=float)
        total = probs.sum()
        probs = probs / total if total > 0 else np.full(len(keys), 1.0 / len(keys))
        current = keys[rng.choices(range(len(keys)), weights=probs, k=1)[0]]
        states.append(current)
    return Trajectory("unguided", float("nan"), states, current,
                      metered.z(current), complete, metered.ledger)


def greedy_preference_run(metered: MeteredProcess, start: str, budget: int,
                          weight: float, scalarize: Scalarization) -> Trajectory:
    """Arm 3. Commit the fiber argmin of the scalarization at every state."""
    states = [start]
    current = start
    complete = True
    for _ in range(budget):
        rows = metered.successors(current)
        if not rows:
            complete = False
            break
        keys = [r[0] for r in rows]
        scores = scalarize(metered.z_many(keys), weight)
        current = keys[_argmin_stable(scores, keys)]
        states.append(current)
    return Trajectory("greedy_pref", weight, states, current,
                      metered.z(current), complete, metered.ledger)


def _greedy_landing(metered: MeteredProcess, key: str, remaining: int,
                    weight: float, scalarize: Scalarization) -> tuple[str, float]:
    """V_G: the deterministic greedy continuation under the SAME weight.

    Returns the LANDING state's value, not the best value seen on the way. A
    Chebyshev objective over a bounded region can be entered and then left
    again, so "best seen" would credit a trajectory for a state it did not
    commit to.
    """
    current = key
    for _ in range(remaining):
        rows = metered.successors(current)
        if not rows:
            break
        keys = [r[0] for r in rows]
        scores = scalarize(metered.z_many(keys), weight)
        current = keys[_argmin_stable(scores, keys)]
    return current, float(scalarize(metered.z(current), weight)[0])


def verified_preference_run(metered: MeteredProcess, start: str, budget: int,
                            weight: float, scalarize: Scalarization,
                            *, top_immediate: int = 4, top_reference: int = 2,
                            n_random: int = 2, seed: int = 0) -> Trajectory:
    """Arm 4. Remaining-budget lookahead under strict improvement.

    Shortlist strata 4/2/2, as the sealed controller used. The RANDOM stratum is
    deliberately retained: a candidate universe defined entirely by the greedy
    score cannot demonstrate that a non-greedy action was worth taking, because
    the non-greedy actions would never be in it.

    STRICT IMPROVEMENT: the lookahead overrides greedy only when it is strictly
    better; ties keep greedy. This is what makes `verified` a genuine superset
    of `greedy` rather than a coin flip between them.

    A NOTE ON WHAT MAY BE REPORTED FROM THIS ARM. The committed action is
    argmin V_G, so V_G(committed) <= V_G(greedy) holds BY CONSTRUCTION and a
    "the lookahead won" statistic computed from V_G has no falsifying range.
    Only two things are admissible: (a) the ENDPOINT comparison against a
    separately-run greedy arm, and (b) the top-1 disagreement rate, which says
    the controller would act differently and can be zero. `decisions` records
    both, and records nothing that could be mistaken for the circular statistic.
    """
    rng = random.Random(seed)
    states = [start]
    current = start
    complete = True
    overrode = 0
    decisions: list[dict] = []

    for step in range(budget):
        remaining = budget - step
        rows = metered.successors(current)
        if not rows:
            complete = False
            break
        keys = [r[0] for r in rows]
        reference = np.array([r[1] for r in rows], dtype=float)
        immediate = scalarize(metered.z_many(keys), weight)

        greedy_index = _argmin_stable(immediate, keys)
        picked, seen = [greedy_index], {greedy_index}
        for order, count in ((np.argsort(immediate, kind="stable"), top_immediate),
                             (np.argsort(-reference, kind="stable"), top_reference)):
            for index in order[:count]:
                if int(index) not in seen:
                    picked.append(int(index))
                    seen.add(int(index))
        rest = [i for i in range(len(keys)) if i not in seen]
        if rest and n_random > 0:
            for index in rng.sample(rest, min(n_random, len(rest))):
                picked.append(index)
                seen.add(index)

        futures = {i: _greedy_landing(metered, keys[i], remaining - 1,
                                      weight, scalarize)[1] for i in picked}
        best_index = min(picked, key=lambda i: (futures[i], keys[i]))
        chosen = best_index if futures[best_index] < futures[greedy_index] else greedy_index

        decisions.append({
            "step": step, "remaining": remaining, "candidates": len(picked),
            # Non-circular: this says the controller WOULD act differently, and
            # it can be zero. It does not say the difference paid off.
            "top1_disagreement": bool(best_index != greedy_index),
            "overrode_greedy": bool(chosen != greedy_index),
        })
        overrode += int(chosen != greedy_index)
        current = keys[chosen]
        states.append(current)

    return Trajectory("verified_pref", weight, states, current,
                      metered.z(current), complete, metered.ledger,
                      overrode_greedy=overrode, decisions=decisions)


def generate_then_rank(metered: MeteredProcess, start: str, budget: int,
                       preferences: Sequence[float], scalarize: Scalarization,
                       n_trajectories: int, seed: int) -> list[Trajectory]:
    """Arm 2. Open-loop: generate first, choose afterwards.

    `n_trajectories` is NOT a free parameter -- it is set by
    `trajectories_for_budget` so that this arm's native oracle-call count
    matches the closed-loop arm it is contrasted against. Without that, the
    contrast would vary controller AND budget, which is the parity defect the
    main lane's audit already caught once.
    """
    pool: list[Trajectory] = []
    for k in range(n_trajectories):
        pool.append(unguided_run(metered, start, budget, seed * 1000 + k))
    if not pool:
        raise ValueError("generate_then_rank needs at least one trajectory")

    endpoints = [t.endpoint for t in pool]
    z = metered.z_many(endpoints)
    out = []
    for weight in preferences:
        scores = scalarize(z, weight)
        pick = pool[_argmin_stable(scores, endpoints)]
        out.append(Trajectory("gen_rank", weight, pick.states, pick.endpoint,
                              pick.endpoint_z, pick.complete, metered.ledger))
    return out


def branch_from_common_prefix(metered: MeteredProcess, start: str, prefix_steps: int,
                              branch_steps: int, preferences: Sequence[float],
                              scalarize: Scalarization,
                              prefix_weight: float = 0.5) -> dict[str, Any]:
    """Contrast P6: one realized history, then a fan of preference-dependent futures.

    The prefix is committed under a SINGLE policy and is finished before any
    branch preference is injected, so the five branches share `x_0..x_prefix`
    exactly. Without that separation the branches would differ from step zero and
    the figure would show five trajectories rather than one history with five
    futures -- which is a different and much weaker statement.

    The retargeting protocol made the same separation physical (generate and
    hash all prefixes, then inject the goal). That is not ceremony: it is what
    proves the prefix was not chosen with knowledge of the branch preference.
    """
    prefix = greedy_preference_run(metered, start, prefix_steps, prefix_weight,
                                   scalarize)
    branches = {}
    for weight in preferences:
        branches[weight] = greedy_preference_run(metered, prefix.endpoint,
                                                 branch_steps, weight, scalarize)
    return {"prefix": prefix, "branch_point": prefix.endpoint, "branches": branches}


def trajectories_for_budget(target_native_calls: int, budget: int,
                            mean_fiber_width: float) -> int:
    """How many unguided trajectories fit a control arm's NATIVE ORACLE budget.

    An unguided trajectory scores only its own committed states plus its
    endpoint, so it is cheap per trajectory; a control arm scores the whole
    fiber at every step. This function is what converts one arm's cost into the
    other's trajectory count so that P3 and P4 vary controller ONLY.
    """
    per_trajectory = max(budget + 1, 1)
    return max(1, int(round(target_native_calls / per_trajectory)))


def trajectories_for_kernel_budget(target_kernel_calls: int, budget: int) -> int:
    """How many unguided trajectories fit a control arm's KERNEL budget.

    WHY BOTH AXES EXIST, AND WHY NEITHER MAY BE QUOTED ALONE
    --------------------------------------------------------
    One kernel call yields ~600 candidate molecules, so the two budget axes buy
    wildly different amounts of search:

      * matched on NATIVE ORACLE CALLS, generate-then-rank is handed roughly
        600x the closed-loop arms' kernel budget -- which is the hypervolume
        inflation channel wearing a benchmark convention's clothes;
      * matched on KERNEL CALLS, generate-then-rank sees far fewer distinct
        molecules than the closed-loop arms, which understates it on the axis
        the multi-objective literature actually budgets.

    Neither matching is "the fair one". PROTOCOL.md therefore requires the
    contrast to be reported as a BRACKET: gen_rank at both matchings, with the
    kernel-call ratio stated. A single number here would be a reporting choice
    that decides the winner.
    """
    per_trajectory = max(budget, 1)
    return max(1, int(round(target_kernel_calls / per_trajectory)))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def pareto_front_indices(z: np.ndarray) -> np.ndarray:
    """Nondominated indices for MAXIMIZED objectives."""
    n = len(z)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        keep &= ~(np.all(z <= z[i], axis=1) & np.any(z < z[i], axis=1))
    return np.flatnonzero(keep)


def hypervolume(z: np.ndarray, reference: np.ndarray) -> float:
    """2-D hypervolume dominated by `z` above `reference`, MAXIMIZED objectives.

    Computed over an arm's COMMITTED ENDPOINTS ONLY -- exactly one per
    preference branch -- so every arm contributes the same number of points and
    the "generate more molecules, score a bigger HV" channel is closed
    structurally rather than adjusted for statistically.
    """
    pts = np.atleast_2d(z)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    pts = pts[np.all(pts > reference[None, :], axis=1)]
    if len(pts) == 0:
        return 0.0
    front = pts[pareto_front_indices(pts)]
    front = front[np.argsort(-front[:, 0])]
    area, prev_y = 0.0, reference[1]
    for x, y in front:
        if y > prev_y:
            area += (x - reference[0]) * (y - prev_y)
            prev_y = y
    return float(area)


def normalized_hypervolume(z: np.ndarray, reference: np.ndarray,
                           utopia: np.ndarray) -> float:
    """Hypervolume expressed in units of the frozen reference box `[r, z*]`.

    NOT BOUNDED BY 1, and it must not be clipped to be.

    `z*` is the held-in **p99**, not a maximum. A controlled trajectory can
    exceed the 99th percentile of the held-in pool, and on this task it does:
    the census's G4 potency reach fraction of 0.700 says a single-objective
    greedy rollout passes `z*_P` on 14 of 20 sources. So values above 1.0 are
    real achievement beyond the reference box, not error.

    Clipping endpoints at `z*` would suppress exactly that signal and would make
    ceiling effects invisible, so the box is a NORMALIZER and never a cap.

    Ratios are unaffected: `HV_star` uses the identical normalizer, so
    `HV(b) / HV_star`, `B_90` and `N_90` are invariant to the box constant.
    """
    box = float(np.prod(utopia - reference))
    return hypervolume(z, reference) / box if box > 0 else 0.0


def preference_coverage(z: np.ndarray, preferences: Sequence[float],
                        scalarize: Scalarization, keys: Sequence[str]) -> float:
    """Fraction of preferences whose endpoint is the UNIQUE Chebyshev argmin
    within this arm's own endpoint set.

    0.2 means all five preferences collapsed onto one molecule; 1.0 means every
    preference is best served by its own endpoint. It can take either value, so
    it is a measurement.
    """
    if len(z) == 0:
        return 0.0
    hits = 0
    for i, weight in enumerate(preferences):
        scores = scalarize(z, weight)
        best = _argmin_stable(scores, keys)
        if best == i and np.sum(scores <= scores[best] + 1e-12) == 1:
            hits += 1
    return hits / len(preferences)


def endpoint_diversity(similarity: np.ndarray) -> float:
    """1 - mean pairwise similarity among an arm's endpoints. The direct
    quantitative form of "one molecular history, many preference-dependent
    futures"."""
    n = len(similarity)
    if n < 2:
        return 0.0
    iu = np.triu_indices(n, k=1)
    return float(1.0 - np.nanmean(similarity[iu]))


def hv_auc(hv_trace: Sequence[float], calls: Sequence[float],
           total_budget: float) -> float:
    """Trapezoid AUC of best-so-far HV against an oracle-call axis, normalized
    by the budget so it lands in [0, 1].

    The caller supplies the axis. It MUST be run twice -- once on native calls
    and once on raw calls -- and both reported. Quoting one convention alone can
    decide which arm wins, so PROTOCOL.md forbids it.
    """
    if not hv_trace or total_budget <= 0:
        return 0.0
    best = np.maximum.accumulate(np.asarray(hv_trace, dtype=float))
    x = np.asarray(calls, dtype=float)
    if len(x) < 2:
        return float(best[-1] * x[-1] / total_budget) if len(x) else 0.0
    return float(np.trapezoid(best, x) / total_budget)


def source_paired_bootstrap(differences: Sequence[float], *, resamples: int = 2000,
                            seed: int = 20260813) -> dict[str, float]:
    """Paired bootstrap where the SOURCE is the resampling unit.

    Preference branches within a source are repeated measures, not independent
    examples; resampling branches would understate the interval. The caller must
    pass ONE number per source, already aggregated over that source's branches.
    """
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    boot = np.array([np.mean(rng.choice(d, n, replace=True)) for _ in range(resamples)])
    return {"n": n, "mean": float(d.mean()), "median": float(np.median(d)),
            "ci95_low": float(np.percentile(boot, 2.5)),
            "ci95_high": float(np.percentile(boot, 97.5)),
            "wins": int((d > 1e-9).sum()), "losses": int((d < -1e-9).sum()),
            "ties": int((np.abs(d) <= 1e-9).sum())}


# ---------------------------------------------------------------------------
# Reference front and efficiency metrics
#
# FROZEN 2026-08-13 02:27 EDT, BEFORE any hypervolume number existed. At the
# time of writing the authorized 12-source smoke had emitted 6 arm checkpoints
# and ZERO `DONE` lines, and `DONE` is the only place an HV value is printed.
# ---------------------------------------------------------------------------

class ReferenceFrontError(ValueError):
    """Raised when HV_ref would be derived from a single arm."""


def union_reference_front(fronts_by_method: dict[str, np.ndarray]) -> np.ndarray:
    """The UNION nondominated front across every method and arm.

    WHY HV_ref MAY NOT COME FROM COMPOSE
    ------------------------------------
    If the denominator of "reached 90% of the attainable front" is the best
    front COMPOSE itself produced, then the statement is partly about COMPOSE's
    own ceiling, and every efficiency curve inherits that bias. That is the same
    shape as the defects already caught in this project: a metric that cannot
    fully disappoint. A method that simply stopped early would drag the
    goalposts toward itself.

    So HV_ref is computed from the pooled endpoints of ALL arms and ALL external
    methods on the frozen evaluation set. Passing a single method raises, rather
    than warning -- a warning would eventually be ignored.
    """
    if len(fronts_by_method) < 2:
        raise ReferenceFrontError(
            "HV_ref must be the union nondominated front across all methods and "
            f"arms, not one arm's own front. Got {sorted(fronts_by_method)}. "
            "Use the frozen box normalisation instead if only one method exists.")
    pooled = np.vstack([np.atleast_2d(z) for z in fronts_by_method.values()])
    pooled = pooled[np.all(np.isfinite(pooled), axis=1)]
    if len(pooled) == 0:
        raise ReferenceFrontError("no finite endpoints in any method")
    return pooled[pareto_front_indices(pooled)]


def pooled_attainable_hypervolume(fronts_by_method: dict[str, np.ndarray],
                                  nadir: np.ndarray,
                                  utopia: np.ndarray) -> float:
    """`HV_star` -- the POOLED ATTAINABLE hypervolume.

    Normalized hypervolume of the pooled nondominated union of ALL
    predeclared methods at the common maximum evaluation budget.

    `nadir` is `r`, the frozen corner from the held-in scales (p5) used to
    COMPUTE hypervolume. It is a different object from `HV_star`, which is
    the TARGET whose 90% level defines budget-to-90%. Conflating the two is
    how a favourable ceiling survives review.

    Its numerical value is known only after every method has run, but the
    RULE is preregistered and METHOD-SYMMETRIC: nobody defines the ceiling
    from their own performance, and a strong method that expands the pooled
    frontier raises the bar for everyone including itself.
    """
    return normalized_hypervolume(
        union_reference_front(fronts_by_method), nadir, utopia)


#: Backwards-compatible alias. Prefer the frozen name.
hv_reference_value = pooled_attainable_hypervolume


#: Two DIFFERENT references, deliberately named apart. Conflating them is how
#: a favourable ceiling survives review.
#:
#:   r        the fixed NADIR corner used to COMPUTE hypervolume. Held-in p5,
#:            frozen from data before any outcome existed.
#:   HV_star  the TARGET hypervolume whose 90% level DEFINES budget-to-90%.
#:            The pooled nondominated union of ALL predeclared methods at the
#:            common maximum evaluation budget.
#:
#: HV_star is called "POOLED ATTAINABLE hypervolume" and never "reference HV"
#: or "COMPOSE's front". The name is where this kind of bias survives review.


def budget_to_ninety(hv_trace: Sequence[float], budgets: Sequence[float],
                     hv_star: float, fraction: float = 0.9) -> dict[str, Any]:
    """`B_90` / `N_90`: smallest budget at which best-so-far HV reaches
    `fraction * HV_star` -- 90% of the POOLED ATTAINABLE hypervolume.

    CENSORING. If the threshold is never reached inside the budget this returns
    ``{"censored": True, "value": None, "reported_as": "B_90 > B_max"}``. It
    does **not** substitute `B_max`.

    Substituting the maximum makes a method that never got there look like one
    that got there at the last moment -- a failure rendered as a success. That
    is the defect family this project has caught six times, and it is easier to
    commit here than anywhere else because the substitution looks like tidiness.
    """
    b = np.asarray(budgets, dtype=float)
    b_max = float(b[-1]) if len(b) else float("nan")
    if hv_star <= 0 or len(hv_trace) == 0:
        return {"value": None, "censored": True, "b_max": b_max,
                "reported_as": "B_90 > B_max", "reason": "no usable HV_star or trace"}
    best = np.maximum.accumulate(np.asarray(hv_trace, dtype=float))
    hit = np.flatnonzero(best >= fraction * hv_star)
    if len(hit) == 0:
        return {"value": None, "censored": True, "b_max": b_max,
                "reported_as": "B_90 > B_max",
                "best_fraction_of_hv_star": float(best[-1] / hv_star)}
    return {"value": float(b[hit[0]]), "censored": False, "b_max": b_max,
            "reported_as": "B_90"}


def summarize_b90(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate `B_90` across sources WITHOUT letting censoring vanish.

    The median is taken over uncensored sources only and is reported alongside
    the censored count. A median that silently pooled censored sources at
    `B_max` would be the same substitution one level up.
    """
    values = [r["value"] for r in results if not r["censored"]]
    censored = [r for r in results if r["censored"]]
    return {
        "median_uncensored": float(np.median(values)) if values else None,
        "n_uncensored": len(values),
        "n_censored": len(censored),
        "censoring_rate": len(censored) / max(len(results), 1),
        "note": ("censored sources are counted, never imputed at B_max; a median "
                 "over uncensored sources alone is meaningless without the "
                 "censoring rate beside it"),
    }


def preference_region_coverage(endpoints_z: np.ndarray,
                               reference_front: np.ndarray,
                               n_regions: int = 5) -> dict[str, Any]:
    """What fraction of the front's REGIONS does a method actually occupy?

    A method that produces one excellent potency-heavy cluster must not score
    well merely because its hypervolume is decent. The union front's extent on
    objective 0 is cut into `n_regions` equal bins, and coverage is the number
    of DISTINCT bins the method's endpoints land in, over `n_regions`.

    Range [1/n_regions, 1]. Five preferences collapsing into one bin gives 0.2;
    one endpoint per bin gives 1.0. Both are attainable, so it measures.
    """
    front = np.atleast_2d(reference_front)
    front = front[np.all(np.isfinite(front), axis=1)]
    z = np.atleast_2d(endpoints_z)
    z = z[np.all(np.isfinite(z), axis=1)]
    if len(front) == 0 or len(z) == 0:
        return {"coverage": 0.0, "occupied_regions": [], "n_regions": n_regions,
                "degenerate_front": True}
    lo, hi = float(np.min(front[:, 0])), float(np.max(front[:, 0]))
    if hi <= lo:
        return {"coverage": 0.0, "occupied_regions": [], "n_regions": n_regions,
                "degenerate_front": True}
    edges = np.linspace(lo, hi, n_regions + 1)
    bins = np.clip(np.digitize(z[:, 0], edges[1:-1]), 0, n_regions - 1)
    occupied = sorted({int(b) for b in bins})
    return {"coverage": len(occupied) / n_regions,
            "occupied_regions": occupied, "n_regions": n_regions,
            "region_edges_objective0": [float(e) for e in edges],
            "degenerate_front": False}


# ---------------------------------------------------------------------------
# THE METHOD UNIVERSE. "Pooled" is meaningless without a membership list, and a
# union that silently gains or loses a method between drafts changes every B_90
# in the table. So membership is declared here, frozen, and checked.
# ---------------------------------------------------------------------------

#: Contributors to HV_star_internal -- the pooled terminal union used for N_90
#: on the INTERNAL trajectory axis. COMPOSE arms only.
INTERNAL_METHOD_UNIVERSE = (
    "unguided",
    "gen_rank@greedy",
    "gen_rank@verified",
    "greedy_pref",
    "verified_pref",
)

#: Contributors to HV_star -- the pooled union used for B_90 on the EXTERNAL
#: evaluation/oracle axes. Internal arms PLUS every qualified external method.
#: External members are added only by the baselines lane; none is qualified yet,
#: so today this equals the internal universe and B_90 is internal-only.
EXTERNAL_METHOD_UNIVERSE: tuple[str, ...] = INTERNAL_METHOD_UNIVERSE


class MethodUniverseError(ValueError):
    """Raised when a pooled union does not match its declared membership."""


def check_method_universe(fronts_by_method: dict[str, np.ndarray],
                          declared: Sequence[str], *, label: str) -> dict[str, Any]:
    """Assert the pooled union is exactly the declared universe.

    A union that quietly gains a method inflates HV_star and pushes every B_90
    out; one that quietly loses a method deflates it and pulls every B_90 in.
    Either way the whole table moves without anything in the table changing, so
    the membership is checked rather than trusted.
    """
    present, want = set(fronts_by_method), set(declared)
    missing, unexpected = sorted(want - present), sorted(present - want)
    if missing or unexpected:
        raise MethodUniverseError(
            f"{label} pooled union does not match its declared membership. "
            f"missing={missing} unexpected={unexpected}. Update the declared "
            f"universe deliberately, or fix the inputs -- do not let the union "
            f"drift.")
    return {"label": label, "members": sorted(want), "n_members": len(want)}
