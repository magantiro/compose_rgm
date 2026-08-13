"""Closed-loop, METERED generate-and-rank matching for the Pareto P3/P4 contrasts.

WHAT WENT WRONG, AND WHY IT WENT WRONG IN OUR FAVOUR
-----------------------------------------------------
`trajectories_for_kernel_budget` is an OPEN-LOOP estimate:

    per_trajectory = max(budget, 1)                    # assume 6 fresh calls
    return max(1, int(round(target_kernel_calls / per_trajectory)))

It hands out a trajectory count and hopes the arm spends the budget. Unguided
trajectories all start at the SAME root, so they collide in the enumeration
cache and cannot spend it. Measured on the committed 12-source smoke, realized
attainment against the COMPOSE arm each was matched to:

    gen_rank@greedy      65.5%          gen_rank@verified      41.7%

On source 000 the baseline was handed 61 trajectories against `verified_pref`'s
368 kernel calls and spent 60 -- **16%**. The error runs in the direction that
flatters COMPOSE, which is exactly why it may not be left in place.

THE FIX: METER, DO NOT ESTIMATE
-------------------------------
Keep generating until the REALIZED ledger reaches the frozen per-source target.
No estimate of what a trajectory "should" cost enters anywhere.

NO SCIENTIFIC CEILING
---------------------
A per-source cap would recreate the defect: on the hard sources -- exactly the
ones where the cache saturates -- the baseline would again be stopped early and
COMPOSE would look better because we quit funding its competitor. So the only
limits here are OPERATIONAL, and if either fires the source's kernel-matched
comparison is marked failed rather than reported at whatever it reached.

    MATCHING_FAILED        the runaway guard fired. A bug or pathological run.
    MATCHING_UNREACHABLE   the ledger STALLED: the reachable set from this root
                           saturated the cache, so generate-and-rank structurally
                           CANNOT spend COMPOSE's kernel budget here.

`MATCHING_UNREACHABLE` is not a failure of this code. It is a finding, and a
sharp one: it would mean the two methods cannot be equated on the kernel axis at
all from a fixed root, and the resource frontier -- not a matched point -- is the
only honest comparison. Neither status may be reported as a scientific result.

STOPPING CONVENTION, PREDECLARED
--------------------------------
A trajectory moves the ledger from below the target to above it; there is no
exact landing. The convention is **the first t at which the realized ledger
reaches or exceeds the target**, i.e. slight OVER-funding of the baseline.
Chosen because it errs AGAINST us: an undershoot convention would reproduce the
bug being fixed. The overshoot is recorded so its size is visible.

THE WHOLE PREFIX IS RECORDED, NOT JUST THE MATCHED POINT
--------------------------------------------------------
Every completed trajectory emits `(t, kernel_calls, native_oracle_calls,
raw_oracle_calls, HV)`. That is what lets the paper show the resource frontier

    HV vs kernel calls     HV vs objective evaluations     HV vs trajectories

instead of being hostage to whether the last trajectory overshot by a few calls.
If closed-loop control wins on one axis and loses on another, the prefix is what
lets us report that rather than pick the flattering axis.

HARNESS OVERHEAD IN THE BASELINE'S REQUEST COUNT -- AUDITED, NOT ASSUMED
------------------------------------------------------------------------
`gen_rank` records `raw_oracle_calls == 2 * n_trajectories` on all 24 committed
arm-instances. That is NOT automatically over-counting: the frozen convention
says duplicate algorithmic requests count, and a cache hit incrementing the raw
counter is that convention working as designed.

The admissible question is whether the ALGORITHM asked twice or the HARNESS did.
Traced in `docs/PARETO_ORACLE_REQUEST_AUDIT.md`: `unguided_run` never reads an
objective during generation -- it samples from `R_theta` alone -- and scores its
endpoint only afterwards, to populate `Trajectory.endpoint_z`. The ranking step
then calls `z_many` over the pool, re-requesting values the harness is already
holding; `np.stack([t.endpoint_z for t in pool])` yields the identical matrix
with zero further requests. A competent generate-and-rank scores each generated
molecule once, and this comparator is one WE author, so its recorded demand must
reflect the method rather than our data structure.

So this module reads `endpoint_z` and the per-t hypervolumes come from vectors
already paid for. Note the direction: correcting it makes the baseline look
CHEAPER, against COMPOSE.

The same one-request-per-trajectory pattern exists in the COMPOSE arms at their
`Trajectory` construction sites, where it is 0.02% and 0.00% of their totals. It
is disclosed and deliberately NOT corrected -- doing so would require re-running
those arms and would destroy the serial baseline the fan-out parity replay must
match exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

#: Operational only. Not a scientific budget -- see the module docstring.
RUNAWAY_TRAJECTORY_GUARD = 5000
#: Consecutive trajectories adding zero kernel calls before the ledger is
#: declared stalled. Generous: the marginal yield is spiky, not smooth.
STALL_WINDOW = 250

MATCHED = "MATCHED"
MATCHING_FAILED = "MATCHING_FAILED"
MATCHING_UNREACHABLE = "MATCHING_UNREACHABLE"


@dataclass
class MatchedGenRank:
    """The matched arm plus the whole resource prefix that produced it."""

    status: str
    target_kernel_calls: int
    realized_kernel_calls: int
    n_trajectories: int
    overshoot_kernel_calls: int
    prefix: list[dict[str, Any]] = field(default_factory=list)
    selected: list[Any] = field(default_factory=list)
    note: str = ""

    @property
    def is_scientific(self) -> bool:
        """Only a MATCHED source may enter a claim-bearing P3/P4 comparison."""
        return self.status == MATCHED

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status,
                "target_kernel_calls": self.target_kernel_calls,
                "realized_kernel_calls": self.realized_kernel_calls,
                "n_trajectories": self.n_trajectories,
                "overshoot_kernel_calls": self.overshoot_kernel_calls,
                "attainment": (self.realized_kernel_calls / self.target_kernel_calls
                               if self.target_kernel_calls else None),
                "is_scientific": self.is_scientific,
                "note": self.note,
                "prefix": self.prefix}


def generate_then_rank_metered(
    metered,
    start: str,
    budget: int,
    preferences: Sequence[float],
    scalarize,
    target_kernel_calls: int,
    *,
    seed: int,
    reference: np.ndarray,
    utopia: np.ndarray,
    unguided_run,
    hypervolume,
    argmin_stable,
    trajectory_cls,
    runaway_guard: int = RUNAWAY_TRAJECTORY_GUARD,
    stall_window: int = STALL_WINDOW,
) -> MatchedGenRank:
    """Generate until the METERED ledger reaches `target_kernel_calls`.

    Dependencies are injected rather than imported so this module can be
    exercised against a synthetic process in tests without the Modal kernel.
    """
    if target_kernel_calls <= 0:
        raise ValueError("target_kernel_calls must be positive")

    pool: list[Any] = []
    prefix: list[dict[str, Any]] = []
    z_rows: list[np.ndarray] = []
    since_progress = 0
    last_kernel = metered.ledger.kernel_calls
    status = MATCHED
    note = ""

    while metered.ledger.kernel_calls < target_kernel_calls:
        if len(pool) >= runaway_guard:
            status, note = MATCHING_FAILED, (
                f"runaway guard fired at {runaway_guard} trajectories with the "
                f"ledger at {metered.ledger.kernel_calls}/{target_kernel_calls}")
            break
        if since_progress >= stall_window:
            status, note = MATCHING_UNREACHABLE, (
                f"the ledger added no kernel call over {stall_window} consecutive "
                f"trajectories and stalled at {metered.ledger.kernel_calls}/"
                f"{target_kernel_calls}: the reachable set from this root "
                f"saturated the enumeration cache, so generate-and-rank cannot "
                f"spend COMPOSE's kernel budget here. This is a FINDING about "
                f"the two methods' resource structures, not a bug.")
            break

        run = unguided_run(metered, start, budget, seed * 1000 + len(pool))
        pool.append(run)
        # ALREADY PAID FOR. `unguided_run` scores its own endpoint to build
        # `endpoint_z`, so re-reading it through the meter would bill the
        # baseline twice -- which is exactly the defect the committed smoke
        # carries: `raw_oracle_calls == 2 * n_trajectories` on all 24 gen_rank
        # arm-instances, because the original arm called `z_many` over the pool
        # after `unguided_run` had already scored every endpoint.
        z_rows.append(np.asarray(run.endpoint_z, dtype=float))
        # Frontier from vectors ALREADY PAID FOR -- read, never re-metered.
        prefix.append({
            "t": len(pool),
            "kernel_calls": int(metered.ledger.kernel_calls),
            "native_oracle_calls": int(metered.ledger.native_oracle_calls),
            "raw_oracle_calls": int(metered.ledger.raw_oracle_calls),
            "hypervolume": float(hypervolume(np.stack(z_rows), reference, utopia)),
        })
        if metered.ledger.kernel_calls > last_kernel:
            last_kernel, since_progress = metered.ledger.kernel_calls, 0
        else:
            since_progress += 1

    realized = int(metered.ledger.kernel_calls)
    result = MatchedGenRank(
        status=status,
        target_kernel_calls=int(target_kernel_calls),
        realized_kernel_calls=realized,
        n_trajectories=len(pool),
        overshoot_kernel_calls=max(0, realized - int(target_kernel_calls)),
        prefix=prefix,
        note=note,
    )
    if not pool:
        result.status = MATCHING_FAILED
        result.note = "no trajectory completed"
        return result

    # Selection is unchanged from the original arm: rank the pool afterwards,
    # once per requested preference. Purpose applied ONLY at the end -- that is
    # the mechanism this comparator exists to isolate.
    z = np.stack(z_rows)
    endpoints = [t.endpoint for t in pool]
    for weight in preferences:
        scores = scalarize(z, weight)
        pick = pool[argmin_stable(scores, endpoints)]
        result.selected.append(
            trajectory_cls("gen_rank", weight, pick.states, pick.endpoint,
                           pick.endpoint_z, pick.complete, metered.ledger))
    return result
