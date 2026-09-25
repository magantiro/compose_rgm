"""Exact uniform sampling over the legal Active8 mark set, without enumerating it.

WHY THIS EXISTS
---------------
"Draw one legal edit uniformly at random" had been implemented as "construct every
legal edit, then draw one".  Measured on a 40-heavy-atom PMO parent that costs 6.57 s
per step, of which 6.17 s is two rules whose admission masks are expensive
(`cycle_close` 4.39 s, `atom_restate_semantic` 1.78 s), while executing the ONE chosen
edit costs 9.9 ms.  Enumeration is the cost of DISCOVERY, not of sampling.

THE CONSTRUCTION, AND WHY IT IS EXACT
-------------------------------------
Each rule's production enumerator iterates a cheaply-describable TENTATIVE set and
keeps the elements its admission predicate accepts:

    cycle_close            combinations(real, 2) x MICRO_BOND_CLASSES
                           admitted by `resolve_cycle_close_edge(...).admitted`
    atom_restate_semantic  real x range(len(ORGANIC_VOCABULARY))
                           admitted by `.admitted and successor is not None and
                           canonical_state_key(successor) != source_key`

So every legal mark of those rules appears EXACTLY ONCE in its tentative set, by the
enumerator's own construction rather than by our re-derivation.

Sample a family with probability proportional to |S_f|, then an element uniformly
within it, then reject if inadmissible.  The probability of drawing any particular
tentative element is

    (|S_f| / sum_g |S_g|) * (1 / |S_f|) = 1 / sum_g |S_g|,

i.e. uniform over the UNION of tentative sets.  Conditioning on acceptance therefore
gives exactly the uniform law over the union of legal marks:

    P(a | accepted) = 1 / |A(x)|.

Choosing a family uniformly instead would define a DIFFERENT law, which is why the
weights are tentative-set SIZES.

The remaining rules are enumerated exhaustively because their enumerators are already
cheap; for them the tentative set IS the legal set and nothing is ever rejected, which
leaves the union law unchanged.

CORRECTNESS IS CHECKED, NOT ASSERTED
------------------------------------
`tests`/the equivalence probe compare this sampler against exhaustive enumeration on
tractable states: identical support, and draw frequencies flat within sampling error.
The admission predicate here is the production resolver, never a cheaper surrogate.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from compose_v4.chem.molecular_graph import (
    ORGANIC_VOCABULARY,
    is_element,
)
from compose_v4.rewrite.operators import (
    MICRO_BOND_CLASSES,
    CycleCloseEdge,
    SemanticAtomRestate,
    resolve_cycle_close_edge,
    resolve_semantic_atom_restate_action,
)

#: Rules whose production enumerator is cheap; enumerated exhaustively.
CHEAP_RULES = (
    "atom_delete",
    "atom_insert",
    "bond_reorder",
    "bond_reroute",
    "cycle_open",
    "ring_system_restate",
)
#: Rules sampled by rejection from their enumerator's own tentative set.
LAZY_RULES = ("cycle_close", "atom_restate_semantic")


def _real_slots(graph) -> tuple[int, ...]:
    return tuple(int(v) for v in np.flatnonzero(is_element(graph.atom_types)))


class LegalMarkSampler:
    """Uniform draws over `A(x)` for one FIXED state `x`.

    Contexts and cheap enumerations are prepared once per state, so repeated draws
    from the same molecule cost only the per-candidate admission checks.
    """

    def __init__(self, graph):
        from compose_v4.experiments.pmo_legal_action_policy import _candidate_actions
        from compose_v4.rewrite.kernel import canonical_state_key
        from compose_v4.rewrite.semantic_atom_restate import (
            prepare_semantic_atom_restate_context,
        )
        from compose_v4.rewrite.semantic_cycle_close import (
            prepare_semantic_cycle_close_context,
        )

        self.graph = graph
        self.real = _real_slots(graph)
        self._canonical = canonical_state_key
        self.cheap: dict[str, tuple] = {
            rule: tuple(_candidate_actions(graph, rule)) for rule in CHEAP_RULES
        }
        self._cc_pairs = tuple(combinations(self.real, 2))
        self._cc_orders = tuple(MICRO_BOND_CLASSES)
        self._ar_classes = len(ORGANIC_VOCABULARY)
        self._cc_context = None
        self._ar_context = None
        self._prepare_cc = prepare_semantic_cycle_close_context
        self._prepare_ar = prepare_semantic_atom_restate_context
        self.sizes: dict[str, int] = {
            **{rule: len(actions) for rule, actions in self.cheap.items()},
            "cycle_close": len(self._cc_pairs) * len(self._cc_orders),
            "atom_restate_semantic": len(self.real) * self._ar_classes,
        }
        self.total = sum(self.sizes.values())
        self._families = tuple(self.sizes)
        weights = np.asarray([self.sizes[f] for f in self._families], dtype=float)
        self._weights = weights / weights.sum() if weights.sum() else weights
        self.attempts = 0
        self.rejections = 0

    def _cycle_close_candidate(self, index: int):
        pair_index, order_index = divmod(index, len(self._cc_orders))
        a, b = self._cc_pairs[pair_index]
        action = CycleCloseEdge(a, b, self._cc_orders[order_index])
        if self._cc_context is None:
            self._cc_context = self._prepare_cc(self.graph)
        resolution = resolve_cycle_close_edge(
            self.graph, action, context=self._cc_context
        )
        if resolution is None or not resolution.admitted:
            return None
        return action

    def _atom_restate_candidate(self, index: int):
        vertex_index, class_index = divmod(index, self._ar_classes)
        action = SemanticAtomRestate(int(self.real[vertex_index]), int(class_index))
        if self._ar_context is None:
            self._ar_context = self._prepare_ar(self.graph)
        resolution = resolve_semantic_atom_restate_action(
            self.graph, action, context=self._ar_context
        )
        if (
            resolution is None
            or not resolution.admitted
            or resolution.successor is None
            or self._canonical(resolution.successor) == self._ar_context.source_key
        ):
            return None
        return action

    def draw(self, rng, *, max_attempts: int = 512):
        """One uniform legal mark as ``(rule, action)``, or ``None`` if exhausted."""
        if not self.total:
            return None
        for _ in range(int(max_attempts)):
            self.attempts += 1
            family = self._families[int(rng.choice(len(self._families), p=self._weights))]
            size = self.sizes[family]
            if not size:
                self.rejections += 1
                continue
            index = int(rng.integers(size))
            if family == "cycle_close":
                action = self._cycle_close_candidate(index)
            elif family == "atom_restate_semantic":
                action = self._atom_restate_candidate(index)
            else:
                action = self.cheap[family][index]
            if action is None:
                self.rejections += 1
                continue
            return family, action
        return None
