"""Step-level learned chemical prior over the legal successors of one state.

The PMO proposal path draws a legal successor UNIFORMLY
(``current_state_edits.current_state_program`` line 53:
``action = actions[int(rng.integers(len(actions)))]``, which honestly declares
``conditional_action_probability = 1 / len(actions)``).  The legal fiber is
astronomically larger than the drug-like manifold, so a uniform draw over it
walks off that manifold; validity closure is a VALENCE guarantee and says
nothing about chemical plausibility.

This module re-weights that one draw by a TRAINED editing model
``Q_theta(y | x, t)`` -- the same normalized marked law every molecular
experiment in this repository scores -- while leaving every other part of the
proposal untouched.

What it is and is not
---------------------
* It is a re-ranking of the legal candidate list, never a filter.  Every legal
  candidate keeps at least ``floor`` relative weight (default 0.05, the value
  the T4 ``BridgeRegionLaw`` uses for the same reason), so the supported set is
  byte-identical to the uniform law's supported set and exploration survives.
* It is TASK-INDEPENDENT.  No oracle, no task identity, no target, and no
  objective value is read here.  One prior serves every benchmark.
* It is NOT a QED/SA screen.  No property calculator is imported.  The weights
  come only from the learned marked law.
* ``prior is None`` at the call site preserves the historical ``rng.integers``
  draw BYTE-IDENTICALLY, so "off" is the absence of the object rather than a
  uniform law object (a uniform law would reproduce the SUPPORT but not the
  DRAWS, because it consumes a different number of RNG values).

The join, and why it is not uniform across families
---------------------------------------------------
The proposal path enumerates candidates with the production *operator*
enumerators (``ENUMERATORS`` in ``current_state_edits``); the model scores
marks built by its own coordinate decoder.  The two describe the same rewrites
with different payload classes, so each candidate must be joined to its mark.

Measured on real production PMO parents, the coordinate join reaches the SAME
canonical successor as the model mark for

    atom_restate_semantic, cycle_close, bond_reroute, ring_system_restate

and does NOT for ``cycle_open``: opening an AROMATIC ring leaves the hydrogen
placement ambiguous, and ``CycleOpenEdge`` and ``BondDelete`` resolve the
residual Kekule structure differently (e.g. ``C=CC(Br)=CC(Cl)=CNC...`` against
``CC=C(Br)C=C(Cl)CNC...`` from one source and one slot pair).  Those are
different molecules, not different spellings.

``cycle_open`` is therefore joined by executing both sides and comparing
canonical successor keys.  That is the expensive path and it is used only where
it is needed: ``cycle_open`` carries ~8 candidates per parent against ~100 for
``atom_restate_semantic``.  ``tests/test_learned_successor_prior.py`` re-derives
the certified set from real parents and fails if any certified family stops
being successor-exact, so the fast path can never silently drift.

Invariants maintained
---------------------
* ``order`` returns a permutation of the candidate list -- same objects, same
  multiset, only the order changes.  Nothing is added and nothing is dropped.
* Every returned weight is ``>= floor / (1 + floor * n)``; no legal candidate
  reaches probability zero.
* A candidate the model cannot reach (no joined mark) keeps exactly the floor
  weight rather than being discarded.
* The head of ``order`` is one draw from the weighted law and the tail is that
  law conditioned on the head's rejection (Efraimidis-Spirakis weighted
  sampling without replacement), matching ``BridgeRegionLaw.order``.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.rewrite.semantic_atom_restate import _target_restate

# ---- Family wiring ----------------------------------------------------------

#: Proposal-path family -> the executor rule the model scores it under.
FAMILY_TO_EXECUTOR_RULE = {
    "atom_restate_semantic": "atom_restate",
    "bond_reroute": "bond_reroute",
    "cycle_close": "bond_insert",
    "cycle_open": "bond_delete",
    "ring_system_restate": "ring_system_restate",
}

#: Families whose coordinate join is certified to reach the model mark's own
#: canonical successor.  Certified by measurement on real production parents
#: and re-derived by the test suite; see the module docstring.
SUCCESSOR_EXACT_COORDINATE_JOIN = frozenset(
    {
        "atom_restate_semantic",
        "bond_reroute",
        "cycle_close",
        "ring_system_restate",
    }
)

DEFAULT_FLOOR = 0.05
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TIME = 0.5


class LearnedSuccessorPriorError(RuntimeError):
    """The prior was asked for something it cannot answer honestly."""


def _coordinate_key(rule: str, action: Any) -> Any:
    """The join key for one model mark, by executor rule."""

    if rule == "atom_restate":
        return (
            int(action.v),
            int(action.atom_type),
            int(action.formal_charge),
            int(action.implicit_h_count),
        )
    if rule == "bond_insert":
        return (int(action.a), int(action.b), int(action.order))
    if rule == "bond_delete":
        return (int(action.a), int(action.b))
    return action


def _candidate_key(state: MolecularGraph, family: str, action: Any) -> Any:
    """The join key for one proposal-path candidate, in the model's coordinates."""

    if family == "atom_restate_semantic":
        restate = _target_restate(
            state,
            vertex=int(action.v),
            target_class_index=int(action.target_class_index),
        )
        if restate is None:
            return None
        return (
            int(restate.v),
            int(restate.atom_type),
            int(restate.formal_charge),
            int(restate.implicit_h_count),
        )
    if family == "cycle_close":
        return (int(action.a), int(action.b), int(action.order))
    if family == "cycle_open":
        return (int(action.a), int(action.b))
    return action


# ---- The prior --------------------------------------------------------------


@dataclass
class LearnedSuccessorPrior:
    """Re-rank one state's legal candidates by a trained editing model.

    ``model`` is a ``FactorizedTraceletRateModel``; it is taken as an opaque
    scorer so this module never depends on how it was constructed.  It is typed
    ``Any`` so importing this module does not pull in torch.
    """

    model: Any
    floor: float = DEFAULT_FLOOR
    temperature: float = DEFAULT_TEMPERATURE
    time: float = DEFAULT_TIME
    cache_entries: int = 64
    verify_successors: bool = True
    _law_cache: OrderedDict = field(default_factory=OrderedDict, repr=False)
    _weight_cache: OrderedDict = field(default_factory=OrderedDict, repr=False)
    _stats: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.floor <= 1.0:
            raise ValueError(f"support floor must lie in (0, 1], not {self.floor}")
        if not self.temperature > 0.0:
            raise ValueError(f"temperature must be positive, not {self.temperature}")
        if not 0.0 < self.time < 1.0:
            raise ValueError(f"time must lie in the open unit interval, not {self.time}")
        if int(self.cache_entries) < 1:
            raise ValueError("cache_entries must be at least 1")
        self._stats.setdefault("model_forwards", 0)
        self._stats.setdefault("candidates_scored", 0)
        self._stats.setdefault("candidates_joined", 0)
        self._stats.setdefault("successor_executions", 0)
        self._stats.setdefault("orders_drawn", 0)

    # -- statistics (load-independent work counters, never wall clock) --

    @property
    def statistics(self) -> dict:
        return dict(self._stats)

    def reset_statistics(self) -> None:
        for key in self._stats:
            self._stats[key] = 0

    # -- the marked law, cached per source state --

    @staticmethod
    def _state_identity(source: MolecularGraph) -> str:
        """A content address for one state, with no objective dependency.

        This is byte-for-byte the computation `control.docking_value.identity`
        performs, inlined so a chemistry prior never has to import a module that
        also carries docking and scoring machinery.  Keep the two in step; the
        test suite asserts this module imports nothing objective-bearing.
        """
        from compose_v4.rewrite.trace_shard import encode_state

        return hashlib.sha256(
            json.dumps(
                encode_state(source),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()

    def _marked_law(self, source: MolecularGraph):
        from compose_v4.experiments.production_successor_kernel import (
            enumerate_factorized_marked_law,
        )

        key = self._state_identity(source)
        cached = self._law_cache.get(key)
        if cached is not None:
            self._law_cache.move_to_end(key)
            return cached
        law = enumerate_factorized_marked_law(self.model, source, self.time)
        self._stats["model_forwards"] += 1
        self._law_cache[key] = law
        while len(self._law_cache) > int(self.cache_entries):
            self._law_cache.popitem(last=False)
        return law

    # -- weights --

    def weights(
        self,
        source: MolecularGraph,
        *,
        family: str,
        actions,
    ) -> np.ndarray:
        """Normalized draw weights, one per candidate, in candidate order.

        Every entry is strictly positive: an unreachable or unjoined candidate
        keeps the floor rather than being removed from the support.
        """

        if family not in FAMILY_TO_EXECUTOR_RULE:
            raise LearnedSuccessorPriorError(
                f"no learned-prior wiring for proposal family {family!r}"
            )
        candidates = tuple(actions)
        if not candidates:
            raise LearnedSuccessorPriorError("cannot weight an empty candidate list")

        # `order` and `probability_of` ask for the same weights back to back, and
        # on the verified-successor family that means re-running real executor
        # calls.  Memoize on the exact inputs.
        cache_key = (
            self._state_identity(source),
            family,
            len(candidates),
            float(self.floor),
            float(self.temperature),
        )
        cached = self._weight_cache.get(cache_key)
        if cached is not None:
            self._weight_cache.move_to_end(cache_key)
            return cached.copy()

        rule = FAMILY_TO_EXECUTOR_RULE[family]
        law = self._marked_law(source)
        table: dict = {}
        for mark in law.marks:
            if mark.executor_rule_name != rule:
                continue
            try:
                table[_coordinate_key(rule, mark.action)] = mark
            except (AttributeError, TypeError):
                continue

        exact = family in SUCCESSOR_EXACT_COORDINATE_JOIN or not self.verify_successors
        probabilities = np.zeros(len(candidates), dtype=float)
        for index, action in enumerate(candidates):
            self._stats["candidates_scored"] += 1
            try:
                key = _candidate_key(source, family, action)
            except (AttributeError, TypeError):
                key = None
            if key is None:
                continue
            mark = table.get(key)
            if mark is None:
                continue
            if not exact and not self._same_successor(source, family, action, mark):
                continue
            probabilities[index] = float(mark.probability)
            self._stats["candidates_joined"] += 1

        peak = float(probabilities.max())
        if peak <= 0.0:
            result = np.full(len(candidates), 1.0 / len(candidates), dtype=float)
        else:
            relative = np.power(probabilities / peak, 1.0 / float(self.temperature))
            weighted = np.maximum(float(self.floor), relative)
            result = weighted / float(weighted.sum())
        self._weight_cache[cache_key] = result
        while len(self._weight_cache) > int(self.cache_entries):
            self._weight_cache.popitem(last=False)
        return result.copy()

    def _same_successor(self, source, family: str, action, mark) -> bool:
        """Do the candidate and its joined mark reach the same molecule?"""

        from compose_v4.experiments.production_successor_kernel import (
            _default_rewrite_system,
        )
        from compose_v4.experiments.whole_ring_plan import execute_program
        from compose_v4.rewrite.action_codec_v4 import encode_action
        from compose_v4.rewrite.kernel import canonical_state_key

        self._stats["successor_executions"] += 1
        try:
            candidate_state, _receipt = execute_program(
                source, [encode_action(family, action)]
            )
            mark_state = _default_rewrite_system(self.model).apply(
                source, mark.executor_rule_name, mark.action
            )
        except (ValueError, RuntimeError, KeyError, IndexError, TypeError):
            # An executor refusal on either side is a definite answer: the two
            # rewrites do not reach the same molecule.  RuntimeError is listed
            # because the pinned rdkit raises it (Canon.cpp invariant
            # violation) on Kekule-degenerate hypervalent rings.
            return False
        return canonical_state_key(candidate_state) == canonical_state_key(mark_state)

    # -- the draw --

    def order(self, source: MolecularGraph, rng, *, family: str, actions) -> tuple:
        """Every candidate, ordered by one weighted draw without replacement.

        The head is a single draw from the weighted law; the tail is that law
        conditioned on the preceding entries having been rejected, so a caller
        that retries on executor refusal stays on-distribution.
        """

        candidates = tuple(actions)
        weights = self.weights(source, family=family, actions=candidates)
        # Efraimidis-Spirakis: key_i = u_i ** (1 / w_i), take descending keys.
        draws = rng.random(len(candidates))
        keys = np.power(np.clip(draws, 1e-300, 1.0), 1.0 / weights)
        self._stats["orders_drawn"] += 1
        return tuple(candidates[index] for index in np.argsort(-keys))

    def probability_of(
        self,
        source: MolecularGraph,
        *,
        family: str,
        actions,
        action,
    ) -> float:
        """The draw probability this prior assigns one candidate.

        Reported at the call site in place of the uniform ``1 / len(actions)``
        so the artifact records the law the proposal was actually drawn from.
        """

        candidates = tuple(actions)
        weights = self.weights(source, family=family, actions=candidates)
        for index, candidate in enumerate(candidates):
            if candidate is action or candidate == action:
                return float(weights[index])
        raise LearnedSuccessorPriorError("action is not a member of the candidate list")

    # -- second consumer: rank a constructed multi-step program --

    def path_log_likelihood(self, states, actions) -> dict:
        """Score an already-constructed edit path under the same learned law.

        The region-grounded construction lane builds a multi-step program and
        needs to rank the results before attachment; a chain that the model
        never saw (peroxide, N-O linkers) should rank below one it did.  This
        returns the reference log-likelihood of that path under
        ``Q_theta``:  ``sum_i log Q_theta(x_{i+1} | x_i, t)``.

        ``states`` is the executed state sequence (length ``n + 1``) and
        ``actions`` the ``n`` executed ``(family, action)`` pairs, exactly as a
        production trace records them.

        COMPARABILITY WARNING.  ``total`` falls monotonically with path length,
        because every step multiplies in a probability below one.  Rank paths of
        DIFFERENT lengths by ``per_step`` (the geometric mean per step) and
        paths of the SAME length by ``total``.  ``unjoined_steps`` counts steps
        whose action could not be joined to a model mark; a path with any
        unjoined step is scored on strictly less evidence, so report it rather
        than silently comparing across different amounts of support.
        """

        import math

        sequence = tuple(states)
        steps = tuple(actions)
        if len(sequence) != len(steps) + 1:
            raise LearnedSuccessorPriorError(
                f"path has {len(sequence)} states for {len(steps)} actions; "
                "expected exactly one more state than actions"
            )
        if not steps:
            raise LearnedSuccessorPriorError("cannot score an empty path")

        total = 0.0
        unjoined = 0
        per_step_logs = []
        for state, (family, action) in zip(sequence[:-1], steps, strict=True):
            if family not in FAMILY_TO_EXECUTOR_RULE:
                unjoined += 1
                per_step_logs.append(None)
                continue
            rule = FAMILY_TO_EXECUTOR_RULE[family]
            law = self._marked_law(state)
            match = None
            try:
                key = _candidate_key(state, family, action)
            except (AttributeError, TypeError):
                key = None
            if key is not None:
                for mark in law.marks:
                    if mark.executor_rule_name != rule:
                        continue
                    try:
                        if _coordinate_key(rule, mark.action) == key:
                            match = mark
                            break
                    except (AttributeError, TypeError):
                        continue
            if match is None:
                unjoined += 1
                per_step_logs.append(None)
                continue
            total += float(match.log_probability)
            per_step_logs.append(float(match.log_probability))

        scored = len(steps) - unjoined
        return {
            "total_log_likelihood": total,
            "per_step_log_likelihood": (total / scored) if scored else float("-nan"),
            "per_step_probability": math.exp(total / scored) if scored else 0.0,
            "scored_steps": scored,
            "unjoined_steps": unjoined,
            "path_length": len(steps),
            "step_log_likelihoods": tuple(per_step_logs),
        }


# ---- Consumption probe ------------------------------------------------------


class PriorNotConsumed(BaseException):
    """The production draw never consulted the prior it was handed.

    Deliberately a ``BaseException`` and NOT a ``ValueError`` / ``RuntimeError``
    / ``KeyError`` / ``IndexError`` / ``TypeError``: the proposal path catches
    all five per family and per draw, so any of them would be swallowed by the
    very code the probe is trying to observe, and the probe would report a pass
    it never earned.
    """


class _ProbeSignal(BaseException):
    """Raised from inside a probe prior's ``order`` to prove it was reached."""


class _ConsumptionProbePrior:
    """A stand-in prior that raises the moment the draw site consults it."""

    def order(self, source, rng, *, family, actions):
        raise _ProbeSignal

    def weights(self, source, *, family, actions):
        raise _ProbeSignal

    def probability_of(self, source, *, family, actions, action):
        raise _ProbeSignal


def assert_prior_is_consumed(draw, *, attempts: int = 16) -> int:
    """Prove a caller's own draw closure reaches the prior it was given.

    ``draw`` takes ``(prior, seed)`` and must perform the caller's REAL
    production draw with that prior.  Returns the 1-based attempt on which the
    prior was reached.  Seeds are fixed rather than random so the check is
    reproducible: for a given code state it always passes or always fails, which
    makes a failure a wiring defect instead of an unlucky draw.

    Bounded attempts are required because the family is chosen before the
    candidate list, and only some families route through the prior.
    """

    probe = _ConsumptionProbePrior()
    for attempt in range(1, int(attempts) + 1):
        try:
            draw(probe, attempt)
        except _ProbeSignal:
            return attempt
        except Exception:  # noqa: BLE001,S112 - an ordinary failure just means retry
            continue
    raise PriorNotConsumed(
        f"the production draw did not consult the prior in {attempts} attempts; "
        "the keyword is almost certainly dropped at one of the call sites"
    )
