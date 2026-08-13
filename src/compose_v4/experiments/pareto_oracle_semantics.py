"""Two oracle-request counters, because two different questions are being asked.

    raw_instrument_oracle_requests   every request the PROGRAM made, verbatim.
                                     Never corrected, never rewritten. This is
                                     what a serial-vs-parallel parity replay
                                     compares, and its value is that it is a
                                     faithful record of an execution.

    algorithmic_oracle_requests      objective information the METHOD consumed
                                     TO MAKE DECISIONS. The main P3 oracle-demand
                                     axis, and what the paper uses.

    benchmark_eval_requests          evaluations performed ONLY AFTER generation,
                                     to score and report the produced molecules.
                                     Ours, not the method's.

THREE CONCEPTS, KEPT APART ON PURPOSE
--------------------------------------
Keeping only the first would attribute our bookkeeping to the algorithms.
Keeping only the second would destroy the exact target a parity replay needs.
Dropping the third would make an HV-vs-oracle plot say something true but easy
to misread: `unguided` reaches HV > 0 at `algorithmic = 0`. That is correct --
it never consulted the objective -- but WE still had to evaluate its molecules
afterwards to know their hypervolume. Without a name, that work vanishes.

So two different cost questions are reported separately, and never merged:

    "how much objective information did the ALGORITHM require?"
        -> algorithmic_oracle_requests

    "how much evaluator work was required to produce AND assess the returned set?"
        -> algorithmic + benchmark_eval, with native/cached shown separately

**Post-hoc evaluation is never added back into the algorithmic counter.**

AND A READING RULE, BECAUSE ZERO IS NOT A VICTORY
--------------------------------------------------
`unguided` corrects to zero algorithmic requests. That is the correct
characterisation of a preference-blind floor -- it is not an efficiency win over
COMPOSE, and may never be presented as one. An arm that ignores the objective
buys its cheapness by ignoring the objective, which is visible in its HV.

WHY THIS IS A POST-HOC DERIVATION AND NOT A RE-RUN
---------------------------------------------------
The harness-only request count is **exactly derivable from artifacts already
committed**, so no COMPOSE arm is re-executed, no trajectory changes, and no old
shard is rewritten. Re-running COMPOSE to fix a 0.02% bookkeeping artifact would
also have destroyed the serial baseline the fan-out parity replay must match.

THE HARNESS-ONLY REQUESTS, ENUMERATED FROM THE CALL GRAPH
----------------------------------------------------------
Every `metered.z` / `metered.z_many` site in `pareto_control.py`:

    line 233  greedy_preference_run   loop     ALGORITHMIC -- selects the action
    line 255  rollout                 loop     ALGORITHMIC -- selects the action
    line 257  rollout                 return   ALGORITHMIC -- the rollout VALUE,
                                               which drives verified selection
    line 298  verified_preference_run loop     ALGORITHMIC -- selects the action
    line 353  generate_then_rank      ranking  ALGORITHMIC -- selects the pick

    line 218  unguided_run            return   HARNESS ONLY
    line 237  greedy_preference_run   return   HARNESS ONLY
    line 331  verified_preference_run return   HARNESS ONLY

The three harness sites are the `Trajectory(...)` constructions. Each is the
**final statement of its function**, evaluated after the trajectory is already
committed, and exists only to populate the `endpoint_z` field we later read to
compute hypervolume. Exactly one fires per completed trajectory.

WHY THEY CANNOT CHANGE ANY DECISION -- INCLUDING THROUGH THE CACHE
-------------------------------------------------------------------
1. **Position.** All three are terminal returns. Generation, action selection,
   rollout selection and stopping have all finished. There is no subsequent
   branch inside the call for them to influence.
2. **Ranking.** `generate_then_rank` ranks on `z_many(endpoints)`, whose every
   element resolves the same cache key to the same array the harness call
   produced. The values are identical objects; the ranking is unchanged.
3. **Cache state.** `MeteredProcess.z` returns the same array whether or not the
   key was cached -- caching changes `native_oracle_calls`, never a returned
   value. So a harness request can make a later algorithmic request cheaper to
   COUNT, but cannot make it return anything different. No decision downstream
   of a cache hit can differ from one downstream of a cache miss.
4. **`native_oracle_calls` is order-invariant.** It counts distinct molecules
   evaluated, so which call site reached a molecule first does not change it.
   Only `raw` needs correcting.

THE CORRECTION, UNIFORM ACROSS ARMS
-----------------------------------
    algorithmic = raw_instrument - (completed trajectories scored at a
                                    Trajectory return site)

Verified on all 12 committed sources, no negative corrections:

    unguided            5.0  -   5.0  =        0.0     <- and this is CORRECT
    gen_rank@greedy     5.5  -   2.8  =        2.8
    gen_rank@verified 133.0  -  66.5  =       66.5
    greedy_pref      21004.0 -   5.0  =    20999.0
    verified_pref   443240.5 -   5.0  =   443235.5

**`unguided` correcting to exactly zero is the check that the definition is
right, not a bug.** An unguided sampler reads `R_theta` and its RNG and never
consults an objective; its algorithmic oracle demand *is* zero. Every request
recorded against it was our evaluation of its output.

And `gen_rank` corrects to exactly `n_trajectories` -- one ranking read per
candidate, which is what generate-and-rank actually demands.

The definition is applied to **every** affected arm, not only where it is
numerically large. It matters ~50% for gen_rank and 0.02% for COMPOSE, and the
definition must not depend on whether the error happens to matter.
"""

from __future__ import annotations

from typing import Any, Mapping

#: Arms whose five branches are one trajectory per preference.
_PREFERENCE_BRANCHED = ("unguided", "greedy_pref", "verified_pref")


def harness_only_requests(arm: str, arm_payload: Mapping[str, Any],
                          n_preferences: int) -> int:
    """Completed trajectories that were scored at a `Trajectory` return site.

    For a generate-and-rank arm this is the POOL size -- every pooled run came
    from `unguided_run` and was scored on its way out. For the preference-branched
    arms it is one per preference.
    """
    if arm.startswith("gen_rank"):
        n = arm_payload.get("n_trajectories")
        if n is None:
            raise ValueError(f"{arm} payload has no n_trajectories; the "
                             "correction cannot be derived without it")
        return int(n)
    if arm in _PREFERENCE_BRANCHED:
        return int(n_preferences)
    raise ValueError(f"unknown arm {arm!r}: refusing to guess its trajectory count")


def corrected_cost(arm: str, arm_payload: Mapping[str, Any],
                   n_preferences: int) -> dict[str, Any]:
    """Both counters, side by side. The raw one is passed through untouched."""
    cost = arm_payload["cost"]
    raw = int(cost["raw_oracle_calls"])
    harness = harness_only_requests(arm, arm_payload, n_preferences)
    algorithmic = raw - harness
    if algorithmic < 0:
        raise ValueError(
            f"{arm}: correction would go negative ({raw} - {harness}). The "
            "derivation does not hold for this artifact; do not silently clamp.")
    return {
        "raw_instrument_oracle_requests": raw,
        "algorithmic_oracle_requests": algorithmic,
        "benchmark_eval_requests": harness,
        "harness_only_requests": harness,   # legacy alias, same quantity
        "native_oracle_calls": int(cost["native_oracle_calls"]),
        "kernel_calls": int(cost["kernel_calls"]),
        "correction": ("raw minus one post-hoc endpoint scoring per completed "
                       "trajectory; see pareto_oracle_semantics for the call-graph "
                       "proof that those requests affect no decision"),
    }


def corrected_shard(shard: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Per-arm corrected accounting for one committed shard. Does NOT mutate it."""
    n_pref = len(shard["preferences"])
    return {arm: corrected_cost(arm, payload, n_pref)
            for arm, payload in shard["arms"].items()
            if isinstance(payload, dict) and "cost" in payload}
