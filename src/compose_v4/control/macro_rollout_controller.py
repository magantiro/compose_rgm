"""Choose a macro transition by CONTINUATION value, not by the reward it hands back.

THE MEASUREMENT THIS EXISTS FOR.  Five matched 64-call thiothixene arms, identical
controller, seed, budget and contract, differing only in the molecule each started from:

    local leader (no macro)   0.4839 -> 0.4839   +0.0000
    bare ring graft           0.5076 -> 0.5810   +0.0734
    ring+junction  Cl         0.5118 -> 0.5118   +0.0000
    ring+junction  F          0.5118 -> 0.5118   +0.0000
    ring+junction  N-linked   0.5000 -> 0.5966   +0.0966

Within ONE operator family the three candidates started within 0.012 of each other and
continued +0.0000, +0.0000 and +0.0966 -- and the best arm in the whole experiment is the
one with the LOWEST immediate score.  Structural proximity does not rank them either: the
two dead arms recover 2 of 2 of the reference's ring systems and the productive bare graft
recovers 1 of 2.  So neither immediate reward nor reference similarity orders these states,
and a controller that argmaxes either one picks a dead basin.

WHAT IS SELECTED ON.  The value ordinary COMPOSE search REACHES from a candidate, measured
by actually running it for a few charged calls.  This is a bandit over temporally extended
actions; successive halving keeps it cheap by spending most of the budget on the arms that
survive early rungs.

THE NO-MACRO ARM IS ALWAYS PRESENT and is not special-cased anywhere: it is an ordinary
candidate whose endpoint is the incumbent.  A macro therefore has to BEAT staying put on
the same measurement, rather than being adopted because a stagnation counter fired.  On the
data above it would have won at thiothixene and lost at any state where local search is
still productive.

EVERY ROLLOUT CALL IS CHARGED.  `advance` is expected to query the run's own ledger.  The
controller never evaluates a candidate off-ledger, because a selection made against
uncounted scores is objective evaluation the budget did not pay for.
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: Spend a little on many arms, then more on the survivors. Each entry is
#: (charged steps to give every surviving arm at this rung, how many arms to keep after).
DEFAULT_RUNGS = ((4, 4), (8, 1))


@dataclass(frozen=True)
class MacroCandidate:
    """One temporally extended action, identified by the state it lands in."""

    label: str
    endpoint: str
    family: str = "macro"
    detail: dict = field(default_factory=dict)

    @property
    def is_no_macro(self) -> bool:
        return self.family == "no_macro"


@dataclass(frozen=True)
class ArmOutcome:
    """What a short charged continuation from one candidate reached."""

    value: float
    calls: int


@dataclass(frozen=True)
class RolloutSelection:
    winner: MacroCandidate
    value: float
    calls_spent: int
    rungs: tuple
    considered: int
    no_macro_value: float | None


@dataclass
class StagnationDetector:
    """Fires when the frontier has not improved for `patience` consecutive updates.

    Deliberately keyed on the FRONTIER, not on the last batch's mean: a run whose best is
    frozen while its top-ten keeps filling is exactly the state the thiothixene and
    celecoxib leaders were in, and a mean-based trigger does not see it.
    """

    patience: int = 3
    tolerance: float = 0.0
    best: float | None = None
    idle: int = 0

    def update(self, frontier_value: float) -> bool:
        if self.best is None or frontier_value > self.best + self.tolerance:
            self.best = frontier_value
            self.idle = 0
        else:
            self.idle += 1
        return self.idle >= self.patience

    def reset(self) -> None:
        self.idle = 0


def rollout_cost(n_candidates: int, rungs=DEFAULT_RUNGS) -> int:
    """Exact charged cost, so a caller can refuse before spending anything."""
    alive, total = n_candidates, 0
    for steps, keep in rungs:
        total += alive * steps
        alive = min(alive, keep)
        if alive <= 0:
            break
    return total


def select_by_continuation(candidates, advance, *, rungs=DEFAULT_RUNGS, budget=None):
    """Run successive halving over `candidates` and return the arm that got furthest.

    `advance(endpoint, steps) -> ArmOutcome` must run exactly `steps` CHARGED local search
    calls from `endpoint` and report the best value reached. It is supplied by the caller
    so that this module never needs a task, an oracle or a ledger of its own.

    Raises rather than silently truncating when the budget cannot cover the schedule: a
    partially run tournament selects on unequal evidence, which is the failure this whole
    module exists to avoid.
    """
    candidates = list(candidates)
    if not candidates:
        raise ValueError("no candidates")
    if not any(c.is_no_macro for c in candidates):
        raise ValueError(
            "the no-macro arm is mandatory: without it a macro is adopted because "
            "stagnation fired rather than because it beat staying put")
    required = rollout_cost(len(candidates), rungs)
    if budget is not None and required > budget:
        raise ValueError(f"rollout needs {required} charged calls, budget is {budget}")

    alive = candidates
    spent, history, values = 0, [], {}
    for steps, keep in rungs:
        scored = []
        for candidate in alive:
            outcome = advance(candidate.endpoint, steps)
            if outcome.calls != steps:
                raise ValueError(
                    f"arm {candidate.label} charged {outcome.calls} of {steps} requested; "
                    "unequal evidence makes the comparison meaningless")
            spent += outcome.calls
            # An arm's value is the best it has reached across ALL rungs it survived, so
            # surviving arms are never penalised for a weak later rung.
            values[candidate.label] = max(
                values.get(candidate.label, float("-inf")), outcome.value)
            scored.append((values[candidate.label], candidate))
        scored.sort(key=lambda row: -row[0])
        history.append(tuple((c.label, round(v, 6)) for v, c in scored))
        alive = [c for _v, c in scored[:keep]]
        if len(alive) <= 1:
            break

    best_value, winner = max(
        ((values[c.label], c) for c in candidates if c.label in values),
        key=lambda row: row[0])
    no_macro = next((values.get(c.label) for c in candidates if c.is_no_macro), None)
    return RolloutSelection(winner=winner, value=best_value, calls_spent=spent,
                            rungs=tuple(history), considered=len(candidates),
                            no_macro_value=no_macro)


def macro_candidates(incumbent, proposals, *, limit=7, diversity=None):
    """The incumbent as the mandatory no-macro arm, plus up to `limit` macro proposals.

    `diversity`, when given, is called as `diversity(chosen_endpoints, endpoint) -> bool`
    and must return True to accept. Two near-identical candidates waste a rung: the
    thiothixene Cl and F arms differ by one halogen, scored identically, and both were
    dead, while the structurally distinct sibling carried the entire gain.
    """
    chosen = [MacroCandidate(label="no_macro", endpoint=incumbent, family="no_macro")]
    kept: list[str] = []
    for proposal in proposals:
        endpoint = getattr(proposal, "endpoint", None) or proposal["endpoint"]
        if endpoint == incumbent or endpoint in kept:
            continue
        if diversity is not None and not diversity(kept, endpoint):
            continue
        family = getattr(proposal, "family", None) or proposal.get("family", "macro")
        kept.append(endpoint)
        chosen.append(MacroCandidate(
            label=f"{family}:{len(kept)}", endpoint=endpoint, family=family,
            detail=dict(getattr(proposal, "detail", None) or {})))
        if len(kept) >= limit:
            break
    return chosen
