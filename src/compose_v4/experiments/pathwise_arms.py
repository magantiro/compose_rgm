"""The pathwise-constraint arms as pure policies over an injected kernel.

The successor kernel only runs on Modal, but the POLICIES do not have to. Every
arm here takes `successors` and `utility` as callables, so the whole arm family
-- masking, best-of-N selection, dead ends, the verified controller -- is
exercised locally against a synthetic successor graph in
`tests/test_pathwise_constraint.py`. The Modal app wires the frozen R_theta and
the frozen goal language into the same functions; it does not reimplement them.

MASKING COSTS NO KERNEL CALLS. `successors` returns the full legal support and
the mask is a predicate on the returned keys, so a masked arm and an
unconstrained arm visiting the same state pay for one enumeration between them.

WHAT IS DEFINITIONAL HERE
-------------------------
`mask=True` arms cannot commit a violating state: that is the construction, not
a result. The quantities that can come out the other way live in
`pathwise_constraints.path_violation_summary` and in the mask census.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from compose_v4.experiments.pathwise_arm_names import (
    ALL_ARMS,
    STAGE_A,
    STAGE_B,
    STAGE_B_ALL,
    STAGE_B_CORRIDOR_ARMS,
    STAGE_B_DESCRIPTIVE_ARM,
    STAGE_B_SPEC,
)
from compose_v4.experiments.pathwise_constraints import (
    mask_successors,
    path_violation_summary,
)

__all__ = [
    "ALL_ARMS",
    "ARM_BUILDERS",
    "STAGE_A",
    "STAGE_B",
    "STAGE_B_ALL",
    "STAGE_B_CORRIDOR_ARMS",
    "STAGE_B_DESCRIPTIVE_ARM",
    "STAGE_B_SPEC",
    "ArmContext",
    "best_of_n",
    "build_stage_b_arm",
    "greedy_path",
    "stochastic_path",
    "verified_path",
]

Rows = list[tuple[str, float]]


@dataclass
class ArmContext:
    """Everything an arm needs, with the expensive parts injected."""

    successors: Callable[[str], Rows]
    """Full legal canonical successors of a state: (key, R_theta probability)."""
    utility: Callable[[str], tuple[float, float]]
    """Goal rank key. Lexicographic (worst margin, mean margin)."""
    motif_smarts: str
    rng: random.Random
    horizon: int = 6
    rollouts: int = 6
    shortlist: int = 3
    candidates_top_immediate: int = 4
    candidates_top_reference: int = 2
    candidates_random: int = 2
    feasible: Callable[[str], bool] | None = None
    """Per-state feasibility predicate. Defaults to exact labeled-subgraph
    preservation of `motif_smarts`. Stage A2 injects the cLogP corridor here so
    the same arm policies serve a corridor family without being rewritten."""

    def support(self, key: str, *, mask: bool) -> Rows:
        rows = self.successors(key)
        if not mask:
            return rows
        if self.feasible is not None:
            return [row for row in rows if self.feasible(row[0])]
        return mask_successors(self.motif_smarts, rows)[0]

    def ranked(self, rows: Rows) -> Rows:
        return sorted(rows, key=lambda row: (self.utility(row[0]), row[0]),
                      reverse=True)


def _enforce_here(step: int, horizon: int, *, mask: bool,
                  endpoint_only: bool) -> bool:
    """Where the constraint bites.

    `endpoint_only=True` applies the mask ONLY when choosing the final
    committed state, which is exactly what "the constraint is checked on the
    product" means operationally. `endpoint_only=False` applies it at every
    step. The two differ in nothing else, which is what makes the 2x2 of
    {where enforced} x {how navigated} an actual factorial rather than two
    unrelated policies.
    """
    if not mask:
        return False
    return (step == horizon - 1) if endpoint_only else True


def greedy_path(ctx: ArmContext, start: str, *, mask: bool,
                endpoint_only: bool = False) -> dict:
    """Commit the best immediate successor. Stops on an empty support.

    `endpoint_terminal_infeasible` records the case that matters for the
    endpoint arms: the controller reached the last step and no successor
    satisfied the constraint, so it cannot deliver a compliant product at all.
    Those trajectories leave the conditioning set of the primary estimand
    rather than being quietly counted as compliant.
    """
    path, current, dead, infeasible = [start], start, None, False
    for step in range(ctx.horizon):
        here = _enforce_here(step, ctx.horizon, mask=mask,
                             endpoint_only=endpoint_only)
        rows = ctx.support(current, mask=here)
        if not rows:
            dead = step
            infeasible = bool(here)
            break
        current = ctx.ranked(rows)[0][0]
        path.append(current)
    return {"trajectory": path, "dead_end_step": dead,
            "endpoint_terminal_infeasible": infeasible}


def stochastic_path(ctx: ArmContext, start: str, *, mask: bool,
                    goal_aware: bool) -> dict:
    """Sample R_theta, optionally restricted to the top-`shortlist` by utility.

    `shortlist == 1` IS greedy, so the stochastic arms degenerate to the greedy
    arms rather than forming a separate tunable policy family.
    """
    path, current, dead = [start], start, None
    for step in range(ctx.horizon):
        rows = ctx.support(current, mask=mask)
        if not rows:
            dead = step
            break
        pool = ctx.ranked(rows)[: ctx.shortlist] if goal_aware else list(rows)
        weights = [max(row[1], 0.0) for row in pool]
        if sum(weights) <= 0.0:
            weights = [1.0] * len(pool)
        current = ctx.rng.choices([row[0] for row in pool], weights=weights, k=1)[0]
        path.append(current)
    return {"trajectory": path, "dead_end_step": dead}


def verified_path(ctx: ArmContext, start: str, *, mask: bool,
                  endpoint_only: bool = False) -> dict:
    """Commit argmax V_G under strict improvement, then re-plan.

    Every lookahead rollout obeys the same mask as the committed step, so the
    controller plans INSIDE the feasible set rather than routing around it and
    discovering at commit time that its plan was illegal.

    `top1_disagreements` is recorded because it is the non-circular statistic:
    it says the controller would ACT differently. The utility gap is not, since
    greedy's action is always in the candidate set and strict improvement never
    commits a lower V_G.
    """
    current, overrides, disagreements, path, dead = start, 0, 0, [start], None
    infeasible = False
    for step in range(ctx.horizon):
        remaining = ctx.horizon - step
        here = _enforce_here(step, ctx.horizon, mask=mask,
                             endpoint_only=endpoint_only)
        rows = ctx.support(current, mask=here)
        if not rows:
            dead = step
            infeasible = bool(here)
            break
        keys = [row[0] for row in rows]
        reference = [row[1] for row in rows]
        scores = [ctx.utility(key) for key in keys]
        order = sorted(range(len(keys)), key=lambda i: (scores[i], keys[i]),
                       reverse=True)
        greedy_index = order[0]
        picked, seen = [greedy_index], {greedy_index}
        for index in order[: ctx.candidates_top_immediate]:
            if index not in seen:
                picked.append(index)
                seen.add(index)
        by_reference = sorted(range(len(keys)), key=lambda i: -reference[i])
        for index in by_reference[: ctx.candidates_top_reference]:
            if index not in seen:
                picked.append(index)
                seen.add(index)
        rest = [i for i in range(len(keys)) if i not in seen]
        if rest and ctx.candidates_random > 0:
            for index in ctx.rng.sample(rest, min(ctx.candidates_random, len(rest))):
                picked.append(index)
                seen.add(index)

        # The lookahead plans over the REMAINING horizon under the same
        # enforcement rule, so an endpoint-only controller's rollouts also
        # enforce only at the true final step and a pathwise controller's
        # rollouts stay inside the feasible set the whole way.
        lookahead = ArmContext(**{**ctx.__dict__, "horizon": remaining - 1})
        futures = {
            i: ctx.utility(
                greedy_path(lookahead, keys[i], mask=mask,
                            endpoint_only=endpoint_only)["trajectory"][-1])
            for i in picked
        }
        best = max(picked, key=lambda i: (futures[i], keys[i]))
        chosen = best if futures[best] > futures[greedy_index] else greedy_index
        disagreements += int(best != greedy_index)
        overrides += int(chosen != greedy_index)
        current = keys[chosen]
        path.append(current)
    return {"trajectory": path, "dead_end_step": dead, "overrides": overrides,
            "top1_disagreements": disagreements,
            "endpoint_terminal_infeasible": infeasible}


def best_of_n(ctx: ArmContext, start: str, *, mask: bool, goal_aware: bool,
              endpoint_filter: bool) -> dict:
    """`rollouts` independent stochastic rollouts, then a single selection.

    `endpoint_filter=True` is the ENDPOINT-ONLY arm: it keeps a rollout on the
    strength of its final molecule alone and never inspects the intermediates.
    That is precisely the handling this workstream claims is insufficient, so
    the discarded intermediates are retained in `rollouts` for the audit.
    """
    rolls = [
        stochastic_path(ctx, start, mask=mask, goal_aware=goal_aware)
        for _ in range(ctx.rollouts)
    ]
    for roll in rolls:
        roll["audit"] = path_violation_summary(ctx.motif_smarts, roll["trajectory"])
        roll["utility"] = ctx.utility(roll["trajectory"][-1])
    pool = [r for r in rolls if r["audit"]["endpoint_valid"]] if endpoint_filter else rolls
    if not pool:
        return {"trajectory": [start], "dead_end_step": 0, "selection_failed": True,
                "rollouts": rolls, "rollouts_offered": len(rolls),
                "rollouts_admissible": 0}
    chosen = max(pool, key=lambda r: (r["utility"], r["trajectory"][-1]))
    return {**chosen, "selection_failed": False, "rollouts": rolls,
            "rollouts_offered": len(rolls), "rollouts_admissible": len(pool)}


#: Arm name -> builder. The names and the stage partition live in
#: `pathwise_arm_names`, which has no RDKit dependency, because `modal run`
#: imports the app module in the launcher's interpreter.
def build_stage_b_arm(name: str, ctx: ArmContext, start: str) -> dict:
    """Dispatch a stage-B arm from its declared (mask, endpoint_only, controller).

    Reading the arm's behaviour off the spec table rather than hand-writing
    five closures is what keeps the 2x2 an actual factorial: `endpoint_greedy`
    and `pathwise_greedy` are provably the same code path differing in one
    boolean, so a terminal-cost difference between them cannot be an artefact
    of two separately written policies.
    """
    mask, endpoint_only, controller = STAGE_B_SPEC[name]
    runner = verified_path if controller == "verified" else greedy_path
    return runner(ctx, start, mask=mask, endpoint_only=endpoint_only)


ARM_BUILDERS: dict[str, Callable[[ArmContext, str], dict]] = {
    "unconstrained_greedy":
        lambda ctx, s: greedy_path(ctx, s, mask=False),
    "endpoint_only":
        lambda ctx, s: best_of_n(ctx, s, mask=False, goal_aware=True,
                                 endpoint_filter=True),
    "pathwise_greedy":
        lambda ctx, s: greedy_path(ctx, s, mask=True),
    "pathwise_stochastic":
        lambda ctx, s: best_of_n(ctx, s, mask=True, goal_aware=True,
                                 endpoint_filter=False),
    "mask_only_sampling":
        lambda ctx, s: best_of_n(ctx, s, mask=True, goal_aware=False,
                                 endpoint_filter=False),
    "unconstrained_verified":
        lambda ctx, s: verified_path(ctx, s, mask=False),
    "pathwise_verified":
        lambda ctx, s: verified_path(ctx, s, mask=True),
}
