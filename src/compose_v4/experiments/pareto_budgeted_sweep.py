"""Budgeted preference sweep: R_theta-prioritized oracle spend under a 10k cap.

Preregistered in `docs/BUDGETED_PREFERENCE_SWEEP_PREREGISTRATION.md`, committed
BEFORE this implementation existed and before any quality outcome.

THE ONE CHANGE FROM THE FULL SWEEP
----------------------------------
The full continuum sweep calls the true objective oracle on EVERY successor of
EVERY expanded state -- median 613 unique evaluations per expansion, 75,910 per
source, against a frozen budget of 10,000.

Here, at each newly expanded state:

    enumerate the COMPLETE canonical fiber          (unchanged)
    score all successors with frozen R_theta        (already carried as
                                                     s.probability)
    call the true oracle ONLY on the top K_SWEEP
        PREVIOUSLY UNEVALUATED successors
    reuse every cached objective value for free
    partition EXACTLY over the scored subset        (unchanged semantics)

Everything else is preserved bit-for-bit: the partition, `_argmin_stable` ties,
the traversal, frozen objectives and scaling, and the 240-expansion guard.

K_SWEEP IS DERIVED FROM CONSTANTS THAT PREDATE P0c
--------------------------------------------------
    K_SWEEP = floor(BUDGET / GUARD) = floor(10000 / 240) = 41

Both inputs were frozen before P0c ran, so K cannot have been tuned to an
outcome that did not yet exist. The guard is used rather than the observed
median expansion count deliberately: at K=41 the worst traversal the guard
permits spends 240 * 41 = 9,840, so the global hard cap NEVER binds and the
result cannot depend on DFS order. Using the observed median (136 -> K=73)
would fit a median source but hit the cap mid-traversal on a 240-expansion
source, which is exactly the artifact this design must avoid.

BARRED: objective-aware pruning, aspiration control, diversity bonuses, learned
surrogates, relaxing the guard, raising the budget, and any sweep over K.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

#: Frozen fairness rule: unique true-oracle evaluations permitted per source.
ORACLE_BUDGET = 10_000

#: Frozen expansion guard, unchanged from P0c.
EXPANSION_GUARD = 240

#: Derived from the two constants above. Not tunable, not swept.
K_SWEEP = ORACLE_BUDGET // EXPANSION_GUARD  # == 41


def select_for_evaluation(
    rows: Sequence[Sequence[Any]],
    already_scored: Iterable[str],
    k: int = K_SWEEP,
) -> tuple[list[str], list[str]]:
    """Choose which successors to spend true-oracle calls on at one state.

    `rows` is the COMPLETE canonical fiber as the kernel produced it:
    `[[key, r_theta_probability], ...]`. It must already be canonicalized and
    alias-resolved -- shortlisting a pre-canonical fiber would silently change
    the candidate set, which is what test Q2 guards.

    Returns `(to_evaluate, scored_subset)`:

    * `to_evaluate` -- the top-`k` successors by frozen `R_theta` likelihood
      among those with no cached objective value. These are the only keys that
      cost a true-oracle call.
    * `scored_subset` -- every key that will have an objective value at this
      state: the cached ones (free) plus the newly evaluated ones. The exact
      preference partition is computed over this set.

    Cached values are free under the frozen oracle semantics, so a successor
    seen at an earlier state costs nothing here and never displaces a fresh
    candidate from the shortlist.

    Ties in `R_theta` probability are broken by canonical key, so the selection
    is deterministic and independent of fiber ordering -- the same discipline
    `_argmin_stable` applies to the controller's own choice.
    """
    seen = set(already_scored)
    unevaluated = [(float(r[1]), str(r[0])) for r in rows if str(r[0]) not in seen]
    # Highest likelihood first; canonical key ascending breaks ties.
    unevaluated.sort(key=lambda t: (-t[0], t[1]))
    to_evaluate = [key for _, key in unevaluated[:max(0, int(k))]]

    cached_here = [str(r[0]) for r in rows if str(r[0]) in seen]
    fresh = set(to_evaluate)
    # Preserve the kernel's own fiber order in the returned subset so the
    # partition sees candidates in exactly the order the full sweep would.
    scored_subset = [
        str(r[0]) for r in rows
        if str(r[0]) in seen or str(r[0]) in fresh
    ]
    assert len(scored_subset) == len(cached_here) + len(to_evaluate)
    return to_evaluate, scored_subset


class OracleLedger:
    """Counts UNIQUE canonical molecules actually evaluated, and hard-stops.

    The frozen denominator is unique canonical molecules evaluated -- never raw
    requests -- so a repeat request for an already-evaluated key is free and
    does not advance the ledger.

    `would_exceed` lets the traversal stop cleanly at the cap instead of
    discovering it mid-state. Under K_SWEEP the cap should never bind; this is
    a backstop, and `binding` records whether it ever did.
    """

    def __init__(self, budget: int = ORACLE_BUDGET) -> None:
        self.budget = int(budget)
        self._evaluated: set[str] = set()
        self.binding = False

    @property
    def spent(self) -> int:
        return len(self._evaluated)

    @property
    def remaining(self) -> int:
        return self.budget - self.spent

    def would_exceed(self, keys: Sequence[str]) -> bool:
        fresh = {k for k in keys if k not in self._evaluated}
        return self.spent + len(fresh) > self.budget

    def charge(self, keys: Sequence[str]) -> list[str]:
        """Register evaluations, refusing to cross the cap. Returns keys billed."""
        billed = []
        for k in keys:
            if k in self._evaluated:
                continue  # cached: free
            if self.spent >= self.budget:
                self.binding = True
                break
            self._evaluated.add(k)
            billed.append(k)
        return billed
