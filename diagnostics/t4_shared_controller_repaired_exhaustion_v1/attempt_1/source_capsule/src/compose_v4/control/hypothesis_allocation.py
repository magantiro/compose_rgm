"""Budget allocation over structural hypotheses, driven by counted docking outcomes.

Everything below the hypothesis is already built and is NOT re-implemented here:
`complete_region_program` holds the protected program whose intermediate states stay
internal, `structural_subgoal` is the address-free structural object, and
`objective_program_search` owns the two proposal lanes and the shared archive. This
module adds only the missing decision layer -- which coherent structural hypothesis
deserves the next few oracle calls.

The unit is a PATH from coarse to fine, for example

    ("structured", "ring_construction", "six_membered", "two_nitrogen")

so that credit for a docking outcome reaches the decision that actually mattered rather
than being smeared over the fourteen primitives the compiler emitted. A coarse node
accumulates every outcome beneath it, which is what lets a family be abandoned before its
parameterisations have each been paid for separately.

Two measured facts shape the design and are worth stating because they rule out the
obvious alternatives. Program SIZE must earn no credit on its own: autonomous programs of
16 and 17 edits exist and score -7.2 to -7.8, while bank programs of the same size reach
-11.7, so size is necessary and not sufficient. And the value of a branch is NOT the
predicted docking score of an undocked molecule -- every such surrogate tried here failed,
the best reaching 8.2% top-10 against 6.1% by chance. What is estimated instead is the
probability that spending more calls under a hypothesis yields a new frontier molecule,
which is a statistic over observed outcomes rather than a structural prediction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

SCHEMA_VERSION = "hypothesis_allocation_v1"


@dataclass
class HypothesisNode:
    """One node of the coarse-to-fine hypothesis tree, with its own outcome record."""

    path: tuple[str, ...]
    calls: int = 0
    frontier_advances: int = 0
    improvements: int = 0
    best_score: float = float("inf")
    children: dict[str, HypothesisNode] = field(default_factory=dict)

    def child(self, label: str) -> HypothesisNode:
        if label not in self.children:
            self.children[label] = HypothesisNode(path=(*self.path, label))
        return self.children[label]

    @property
    def advance_rate(self) -> float:
        return self.frontier_advances / self.calls if self.calls else 0.0


class HypothesisTree:
    """Outcome statistics over hypothesis paths, plus a widening gate.

    Progressive widening keeps the branching factor survivable: a node's children are
    only opened once it has earned enough calls, so the controller never enumerates
    region x family x topology x heteroatom x attachment up front.
    """

    def __init__(self, *, widen_after: int = 4, widen_exponent: float = 0.5):
        self.root = HypothesisNode(path=())
        self.widen_after = widen_after
        self.widen_exponent = widen_exponent
        self.calls = 0

    def nodes_on(self, path) -> list[HypothesisNode]:
        node, chain = self.root, [self.root]
        for label in path:
            node = node.child(label)
            chain.append(node)
        return chain

    def record(
        self, path, *, score: float, parent_score: float | None, incumbent: float | None
    ) -> dict:
        """Credit one counted docking outcome to every level of its hypothesis path.

        `incumbent` is the best score standing BEFORE this call, so a frontier advance is
        judged against what was actually known at the time, never against the final best.
        """
        advanced = incumbent is not None and score < incumbent
        improved = parent_score is not None and score < parent_score
        self.calls += 1
        for node in self.nodes_on(path):
            node.calls += 1
            node.frontier_advances += int(advanced)
            node.improvements += int(improved)
            node.best_score = min(node.best_score, score)
        return {"advanced": advanced, "improved": improved}

    def may_widen(self, node: HypothesisNode) -> bool:
        """Open a node's children only once it has paid for them."""
        return node.calls >= self.widen_after and (
            len(node.children) <= node.calls**self.widen_exponent
        )

    def open_branches(self, candidates) -> list[tuple[str, ...]]:
        """Candidate paths currently eligible for budget, honouring the widening gate."""
        allowed = []
        for path in candidates:
            chain = self.nodes_on(path[:-1]) if len(path) > 1 else [self.root]
            parent = chain[-1]
            if len(path) == 1 or parent.calls == 0 or self.may_widen(parent):
                allowed.append(path)
        return allowed or list(candidates)


def thompson_scores(
    tree: HypothesisTree,
    paths,
    rng,
    *,
    prior_a=1.0,
    prior_b=4.0,
    budget_remaining=None,
    horizon=None,
) -> np.ndarray:
    """Posterior samples of each branch's frontier-advance rate.

    Beta-Bernoulli over "did a call under this hypothesis advance the frontier". The
    prior is deliberately pessimistic (mean 0.2) because most branches do not advance
    anything, and an optimistic prior would spend the whole budget opening new families.

    When `budget_remaining` and `horizon` are supplied the sample is shifted toward
    exploitation as the budget runs down: with few calls left there is no value in
    resolving uncertainty that cannot be acted on. This is the budget-aware part, and it
    is a schedule over exploration, not a prediction of any molecule's score.
    """
    samples = np.empty(len(paths))
    late = 0.0
    if budget_remaining is not None and horizon:
        late = float(np.clip(1.0 - budget_remaining / horizon, 0.0, 1.0))
    for i, path in enumerate(paths):
        node = tree.nodes_on(path)[-1]
        a = prior_a + node.frontier_advances
        b = prior_b + max(node.calls - node.frontier_advances, 0)
        draw = rng.beta(a, b)
        # late in the run, interpolate the draw toward the observed mean
        samples[i] = (1 - late) * draw + late * (a / (a + b))
    return samples


def upper_confidence(tree: HypothesisTree, paths, *, exploration: float = 1.0) -> np.ndarray:
    """UCB alternative to Thompson sampling, for a deterministic comparison arm."""
    total = max(tree.calls, 1)
    out = np.empty(len(paths))
    for i, path in enumerate(paths):
        node = tree.nodes_on(path)[-1]
        if node.calls == 0:
            out[i] = float("inf")
            continue
        out[i] = node.advance_rate + exploration * math.sqrt(math.log(total) / node.calls)
    return out


def branch_best_scores(tree: HypothesisTree, paths, rng, *, exploration: float = 0.35,
                       budget_remaining=None, horizon=None) -> np.ndarray:
    """Rank branches by the best molecule each has produced, with an exploration bonus.

    Frontier advances are far too rare for a rate to be estimable: measured on the runs
    Dynamic won, the winning branch advanced ONCE in 39 calls (BRAF) and once in 64
    (5HT1B), and every branch that ever wins does so with a single advance. A Bernoulli
    rate is then 1/calls, which perversely PUNISHES whichever branch has been invested
    in, and the rate-based bandit duly starved the winner on six of seven replayed runs.

    This is an extreme-value problem -- what matters is the best a branch can reach, not
    its average -- so the statistic is the branch maximum. The bonus is optimism for
    thinly sampled branches and decays as the budget runs down.
    """
    total = max(tree.calls, 1)
    late = 0.0
    if budget_remaining is not None and horizon:
        late = float(np.clip(1.0 - budget_remaining / horizon, 0.0, 1.0))
    seen = [tree.nodes_on(path)[-1] for path in paths]
    finite = [n.best_score for n in seen if math.isfinite(n.best_score)]
    if not finite:
        return rng.random(len(paths))
    floor, ceiling = max(finite), min(finite)
    spread = max(floor - ceiling, 1e-6)
    out = np.empty(len(paths))
    for i, node in enumerate(seen):
        if not math.isfinite(node.best_score):
            out[i] = 1.0 + rng.random() * 0.1
            continue
        # lower docking score is better, so map the branch maximum into [0, 1]
        quality = (floor - node.best_score) / spread
        bonus = exploration * (1 - late) * math.sqrt(math.log(total + 1) / (node.calls + 1))
        out[i] = quality + bonus
    return out


def allocate(
    tree: HypothesisTree,
    candidates,
    rng,
    *,
    rule: str = "thompson",
    budget_remaining=None,
    horizon=None,
) -> tuple[str, ...]:
    """Choose the hypothesis path that receives the next counted call."""
    paths = tree.open_branches(list(candidates))
    if not paths:
        raise ValueError("no hypothesis branch is available for allocation")
    if rule == "thompson":
        scores = thompson_scores(
            tree, paths, rng, budget_remaining=budget_remaining, horizon=horizon
        )
    elif rule == "ucb":
        scores = upper_confidence(tree, paths)
    elif rule == "branch_best":
        scores = branch_best_scores(tree, paths, rng, budget_remaining=budget_remaining,
                                    horizon=horizon)
    elif rule == "uniform":
        scores = rng.random(len(paths))
    else:
        raise ValueError(f"unknown allocation rule {rule!r}")
    return paths[int(np.argmax(scores))]
