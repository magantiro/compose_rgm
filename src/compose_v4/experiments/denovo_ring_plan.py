"""Realize a globally-sampled ring plan through the production sampler.

The competitor observation this implements: neither GenMol nor InVirtuoGen owns
a ring controller.  They pin a GLOBAL marginal before generation -- InVirtuoGen
factorizes ``p(x) = p(n) p_theta(x | n)`` with ``p(n)`` the empirical ZINC250k
length law, GenMol draws its mask count from an empirical training
distribution -- and their string/fragment representation then lets ring content
arrive from the data distribution.  COMPOSE builds through a legal fiber that
changes as it builds, so the distribution available at step ``t`` drifts away
from the corpus.  Length is not the only global variable whose marginal needs
pinning.

So this module factorizes the de-novo law the same way::

    p(G) = p(R) * p_theta(G | R)

``R`` is a ring-system multiset drawn ONCE, before generation, from the corpus
law (``compose_v4.eval.denovo_ring_marginal.RingSystemPlanPrior``).  It is
realized at ``t = 0`` -- on the pristine all-carbon tree, where
``_eligible_grow_host_graph`` offers its largest possible host -- using the
EXISTING ``ring_system_grow`` macro and the checkpoint's own ring catalog.  The
ordinary learned process then runs from the resulting state.

Two properties are load-bearing and are the reason this is not a rewrite of the
ring machinery:

* **The production sampler chooses.**  Restriction is expressed through the two
  attributes ``FactorizedTraceletRateModel.sample_rewrite_mark_conditioned``
  already reads -- ``disabled_sampling_rule_names`` and
  ``excluded_sampling_ring_template_indices`` -- so template selection,
  placement and the semantic electronic decoding all run the shipped code path
  verbatim.  Nothing here transcribes it, and a change to that path moves these
  results instead of silently disagreeing with them.
* **Nothing is rejected after the fact.**  The latent is drawn before
  generation; an endpoint is never filtered on how its rings came out.  Plans
  the host cannot carry are RECORDED as unrealized, never retried until they
  look right.

The arms this supports:

``A`` production sampler, no plan.
``B`` the ring-system COUNT drawn from the corpus law and realized at ``t = 0``,
      each template chosen freely from the model's own support.  Isolates
      *earliness plus count* from the pinned size law.
``C`` the full signature drawn from the corpus law and realized at ``t = 0``.
      Adds the pinned sizes on top of ``B``.

``B`` is a substitute for the ``exact_early_ring`` arm a matched-checkpoint
comparison cannot express: that scheduler reorders TEACHER TRACES at training
time, so obtaining it requires a retrain, not an inference flag.  ``B`` is the
inference-time analogue of its intent -- commit ring transactions at the
earliest state that can carry them -- and it is labelled as a substitution
wherever it is reported.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.eval.denovo_ring_marginal import (
    MoleculeRingSignature,
    RingSystemSignature,
)
from compose_v4.eval.ring_calibration import ring_template_cycle_sizes
from compose_v4.eval.ring_support_host import template_host_atoms
from compose_v4.model.factorized_tracelet_rate_model import MARK_RULE_NAMES
from compose_v4.rewrite.kernel import de_novo_rewrite_system
from compose_v4.rewrite.typed_ring_catalog import ring_system_template

RING_GROW_RULE = "ring_system_grow"

#: Why one requested ring system was not installed.  ``signature_absent`` is a
#: property of the CATALOG, ``no_support`` of the HOST at that moment; they are
#: separate codes because they point at different repairs.
UNREALIZED_SIGNATURE_ABSENT = "signature_absent_from_catalog"
UNREALIZED_NO_SUPPORT = "no_host_support"


# ---- Catalog index ----------------------------------------------------------


@dataclass(frozen=True)
class CatalogSignatureIndex:
    """Which templates realize which ring-system signature, and what they cost.

    ``host_atoms`` is the number of host atoms a signature's CHEAPEST template
    consumes.  It is the scheduling key: the host is the resource every ring
    commitment spends, so the plan is realized most-demanding-first.
    """

    template_count: int
    indices: dict[RingSystemSignature, tuple[int, ...]]
    host_atoms: dict[RingSystemSignature, int]

    @classmethod
    def build(cls, templates: Sequence[Any]) -> CatalogSignatureIndex:
        indices: dict[RingSystemSignature, list[int]] = {}
        host: dict[RingSystemSignature, int] = {}
        for index, template in enumerate(templates):
            signature = ring_template_cycle_sizes(template)
            indices.setdefault(signature, []).append(index)
            need = template_host_atoms(template)
            host[signature] = min(host.get(signature, need), need)
        return cls(
            template_count=len(templates),
            indices={key: tuple(value) for key, value in indices.items()},
            host_atoms=host,
        )

    def excluded_for(self, signature: RingSystemSignature) -> tuple[int, ...]:
        """Every template index that does NOT realize this signature."""

        keep = set(self.indices.get(signature, ()))
        if not keep:
            raise KeyError(f"signature {signature!r} is absent from the catalog")
        return tuple(index for index in range(self.template_count) if index not in keep)

    def order_plan(self, plan: MoleculeRingSignature) -> tuple[RingSystemSignature, ...]:
        """Most host-demanding first, deterministically.

        A large ring system needs a large contiguous carbon tree; a three-ring
        needs three atoms.  Installing the cheap systems first would spend the
        contiguity the expensive ones require, which is the same ordinal
        mechanism the unplanned process suffers from -- so the plan is
        scheduled against the resource rather than in draw order.
        """

        return tuple(
            sorted(
                plan,
                key=lambda signature: (
                    -self.host_atoms.get(signature, len(signature)),
                    signature,
                ),
            )
        )


# ---- Sampling restriction ---------------------------------------------------


@contextmanager
def restricted_ring_sampling(
    model: Any,
    *,
    only_ring_grow: bool = False,
    disable_ring_grow: bool = False,
    excluded_template_indices: Sequence[int] = (),
):
    """Temporarily narrow the production sampler's family and template choice.

    Both knobs already exist on the shipped model and are read inside
    ``sample_rewrite_mark_conditioned``; this only sets and restores them, so
    the selection itself is never reimplemented here.  Restoration is in a
    ``finally`` because a raising draw must not strand the model in a
    restricted state for every later trajectory in the same worker.
    """

    if only_ring_grow and disable_ring_grow:
        raise ValueError("cannot both require and forbid ring_system_grow")
    previous_rules = getattr(model, "disabled_sampling_rule_names", ())
    previous_templates = getattr(model, "excluded_sampling_ring_template_indices", ())
    if only_ring_grow:
        model.disabled_sampling_rule_names = tuple(
            name for name in MARK_RULE_NAMES if name != RING_GROW_RULE
        )
    elif disable_ring_grow:
        model.disabled_sampling_rule_names = (RING_GROW_RULE,)
    model.excluded_sampling_ring_template_indices = tuple(
        int(index) for index in excluded_template_indices
    )
    try:
        yield
    finally:
        model.disabled_sampling_rule_names = previous_rules
        model.excluded_sampling_ring_template_indices = previous_templates


# ---- Realization ------------------------------------------------------------


@dataclass
class RingPlanOutcome:
    """What the plan asked for and what the host could carry."""

    requested: MoleculeRingSignature
    order: tuple[RingSystemSignature, ...]
    realized: tuple[RingSystemSignature, ...] = ()
    unrealized: tuple[tuple[RingSystemSignature, str], ...] = ()
    events: int = 0
    bin_provenance: dict = field(default_factory=dict)

    @property
    def fully_realized(self) -> bool:
        return not self.unrealized

    def to_json(self) -> dict:
        return {
            "requested": [list(system) for system in self.requested],
            "order": [list(system) for system in self.order],
            "realized": [list(system) for system in self.realized],
            "unrealized": [
                [list(system), reason] for system, reason in self.unrealized
            ],
            "events": self.events,
            "fully_realized": self.fully_realized,
            "bin_provenance": dict(self.bin_provenance),
        }


def install_one_ring_system(
    model: Any,
    state: MolecularGraph,
    *,
    rng: np.random.Generator,
    time_value: float,
    index: CatalogSignatureIndex | None = None,
    signature: RingSystemSignature | None = None,
    runtime: Any | None = None,
) -> tuple[MolecularGraph | None, RingSystemSignature | None, str | None]:
    """Commit one ``ring_system_grow``, optionally pinned to one signature.

    ``signature is None`` leaves the template choice entirely to the model over
    its own support -- the arm-B behaviour.  The realized signature is read back
    from the EXECUTED action through ``ring_system_template``, never copied from
    the request, so "the plan was realized" is an observation rather than an
    echo of the specification.
    """

    runtime = runtime or de_novo_rewrite_system()
    if signature is None:
        excluded: tuple[int, ...] = ()
    else:
        if index is None:
            raise ValueError("a pinned signature needs a catalog index")
        if signature not in index.indices:
            return None, None, UNREALIZED_SIGNATURE_ABSENT
        excluded = index.excluded_for(signature)
    with restricted_ring_sampling(
        model, only_ring_grow=True, excluded_template_indices=excluded
    ):
        sampled = model.sample_rewrite_mark(state, float(time_value), rng)
    if sampled.action is None:
        return None, None, UNREALIZED_NO_SUPPORT
    if sampled.rule_name != RING_GROW_RULE:
        raise RuntimeError(
            f"ring-plan realization drew {sampled.rule_name!r}, not {RING_GROW_RULE!r}"
        )
    realized = ring_template_cycle_sizes(ring_system_template(state, sampled.action))
    successor = runtime.apply(state, RING_GROW_RULE, sampled.action)
    return successor, realized, None


def realize_ring_plan(
    model: Any,
    state: MolecularGraph,
    plan: MoleculeRingSignature,
    *,
    rng: np.random.Generator,
    time_value: float,
    index: CatalogSignatureIndex,
    pin_signatures: bool = True,
    runtime: Any | None = None,
) -> tuple[MolecularGraph, RingPlanOutcome]:
    """Install the whole plan at one state, most host-demanding first.

    ``pin_signatures=False`` realizes only the plan's LENGTH, letting the model
    choose each template from its full support -- the arm-B ablation that
    separates "commit the rings early" from "commit the corpus's rings".

    A system the host cannot carry is recorded and the plan CONTINUES: a later,
    cheaper system may still fit, and abandoning the remainder would confound
    the realization rate with the ordering.
    """

    runtime = runtime or de_novo_rewrite_system()
    order = index.order_plan(plan)
    outcome = RingPlanOutcome(requested=plan, order=order)
    realized: list[RingSystemSignature] = []
    unrealized: list[tuple[RingSystemSignature, str]] = []
    for signature in order:
        successor, got, reason = install_one_ring_system(
            model,
            state,
            rng=rng,
            time_value=time_value,
            index=index,
            signature=signature if pin_signatures else None,
            runtime=runtime,
        )
        if successor is None:
            unrealized.append((signature, str(reason)))
            continue
        state = successor
        realized.append(got if got is not None else signature)
        outcome.events += 1
    outcome.realized = tuple(realized)
    outcome.unrealized = tuple(unrealized)
    return state, outcome


# ---- The arms ---------------------------------------------------------------

#: Arm labels.  Two ingredients vary independently: WHERE the plan is realized
#: (``0`` = at ``t = 0``, ``1`` = at the model's own first ring event) and WHAT
#: is pinned (``B`` = the ring-system COUNT only, ``C`` = the full signature).
#: ``A`` is the shipped process.  ``A -> B`` attributes earliness plus the
#: count; ``B -> C`` attributes the size law; ``0 -> 1`` attributes the
#: realization point.
ARMS = ("A", "B0", "C0", "B1", "C1")

#: ``t = 0`` is the literal reading of "as early as the support allows": the
#: host is maximal on the pristine carbon tree.  But a ring-rich state at an
#: early conditioning time is OFF-DISTRIBUTION -- in training, ring-rich states
#: occur only late, where the process is winding down -- and the hazard head is
#: a function of ``(state, time)``, so the continuation reads "nearly done".
#: MEASURED on matched states: total hazard 21.1 on the bare tree against 5.2
#: after installing the plan, a ~4x collapse that shows up as ~5x fewer
#: subsequent events.  ``first_ring_event`` realizes the same plan at the state
#: and time the model ITSELF first decides to commit a ring, which keeps the
#: state on-distribution while the host is still largely intact.
REALIZE_AT_START = "t0"
REALIZE_AT_FIRST_RING_EVENT = "first_ring_event"


def arm_configuration(arm: str) -> tuple[bool, str] | None:
    """``(pin_signatures, realization_point)`` for a planned arm, else ``None``."""

    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; known: {ARMS}")
    if arm == "A":
        return None
    return (
        arm.startswith("C"),
        REALIZE_AT_START if arm.endswith("0") else REALIZE_AT_FIRST_RING_EVENT,
    )


def initial_frozen_time(time_step: float) -> float:
    """The conditioning time the production loop uses on its first interval.

    Read off ``sample_tracelet_ancestral``'s own ``frozen_time`` expression at
    ``operational_time = 0`` so a plan event is scored at the same time the
    first ordinary event would have been, rather than at an invented ``t``.
    """

    return 1.0 - float(np.exp(-(0.0 + float(time_step)) / 2.0))


class FirstRingEventPlanInterceptor:
    """Realize the whole plan when the model first decides to commit a ring.

    The trigger is the MODEL's own draw, not a clock: whatever state and
    conditioning time the process has reached when it first chooses
    ``ring_system_grow`` is where the plan lands.  That keeps the plan at the
    earliest on-distribution moment rather than the earliest possible one.

    Ring growth is disabled immediately afterwards, so the realized ring
    marginal is the plan's and not the plan's plus whatever the process adds.
    """

    def __init__(
        self,
        model: Any,
        plan: MoleculeRingSignature,
        *,
        index: CatalogSignatureIndex,
        pin_signatures: bool,
        runtime: Any | None = None,
    ) -> None:
        self._model = model
        self._plan = plan
        self._index = index
        self._pin = pin_signatures
        self._runtime = runtime or de_novo_rewrite_system()
        self.outcome: RingPlanOutcome | None = None
        self.fired = False

    def __call__(self, state, sampled, frozen_time, rng):
        if self.fired or sampled.rule_name != RING_GROW_RULE:
            return None
        self.fired = True
        successor, outcome = realize_ring_plan(
            self._model,
            state,
            self._plan,
            rng=rng,
            time_value=float(frozen_time),
            index=self._index,
            pin_signatures=self._pin,
            runtime=self._runtime,
        )
        outcome.bin_provenance = {"realized_at_frozen_time": float(frozen_time)}
        self.outcome = outcome
        # From here the ordinary process continues WITHOUT ring growth.
        self._model.disabled_sampling_rule_names = (RING_GROW_RULE,)
        return successor, (RING_GROW_RULE,) * outcome.events


def sample_denovo_arm(
    model: Any,
    *,
    arm: str,
    rng: np.random.Generator,
    source_prior: Any,
    index: CatalogSignatureIndex | None = None,
    plan_prior: Any | None = None,
    n_slots: int = 40,
    operational_horizon: float = 16.0,
    time_step: float = 0.1,
    max_events: int = 128,
    fiber_cache: dict | None = None,
    rate_cache: dict | None = None,
    runtime: Any | None = None,
) -> dict:
    """One de-novo trajectory under one arm, sharing the ``t = 0`` tree.

    Every arm draws its initial tree as the FIRST use of ``rng``, so matched
    trajectory seeds give matched initial states and the arms diverge only
    afterwards.  Arm A delegates to the production sampler untouched.
    """

    from compose_v4.chem.molecular_graph import molecular_graph_to_smiles
    from compose_v4.chem.state import is_connected_or_null, is_valid_state
    from compose_v4.experiments.tracelet_conditional import sample_tracelet_ancestral

    configuration = arm_configuration(arm)
    common = {
        "rng": rng,
        "n_slots": n_slots,
        "operational_horizon": operational_horizon,
        "time_step": time_step,
        "max_events": max_events,
        "fiber_cache": fiber_cache,
        "rate_cache": rate_cache,
    }
    plan_json: dict | None = None
    plan_events = 0
    trigger_fired: bool | None = None
    if configuration is None:
        rollout = sample_tracelet_ancestral(model, source_prior=source_prior, **common)
    else:
        if index is None or plan_prior is None:
            raise ValueError(f"arm {arm} needs a catalog index and a plan prior")
        pin_signatures, realization_point = configuration
        state = source_prior.sample(rng, n_slots=n_slots)
        host_atoms = int(np.count_nonzero(state.atom_types > 0))
        plan, provenance = plan_prior.sample_with_provenance(rng, host_atoms)
        provenance = {**provenance, "host_atoms": host_atoms}
        if realization_point == REALIZE_AT_START:
            state, outcome = realize_ring_plan(
                model,
                state,
                plan,
                rng=rng,
                time_value=initial_frozen_time(time_step),
                index=index,
                pin_signatures=pin_signatures,
                runtime=runtime,
            )
            outcome.bin_provenance = provenance
            with restricted_ring_sampling(model, disable_ring_grow=True):
                rollout = sample_tracelet_ancestral(
                    model, initial_state=state, **common
                )
        else:
            interceptor = FirstRingEventPlanInterceptor(
                model,
                plan,
                index=index,
                pin_signatures=pin_signatures,
                runtime=runtime,
            )
            # The context manager restores whatever the interceptor set, so a
            # trajectory cannot leave ring growth disabled for the next one.
            with restricted_ring_sampling(model):
                rollout = sample_tracelet_ancestral(
                    model, initial_state=state, transition_interceptor=interceptor, **common
                )
            trigger_fired = interceptor.fired
            outcome = interceptor.outcome or RingPlanOutcome(
                requested=plan, order=index.order_plan(plan)
            )
            outcome.bin_provenance = {**provenance, **outcome.bin_provenance}
        plan_json = outcome.to_json()
        plan_events = outcome.events
    record = {
        "arm": arm,
        "smiles": molecular_graph_to_smiles(rollout.final_state),
        "events": len(rollout.event_times),
        "plan_events": plan_events,
        "process_events": len(rollout.event_times) - (
            plan_events if configuration and configuration[1] != REALIZE_AT_START else 0
        ),
        "event_rules": list(rollout.event_rules),
        "valid_state": bool(is_valid_state(rollout.final_state)),
        "connected": bool(is_connected_or_null(rollout.final_state)),
        "exhausted_event_budget": bool(
            getattr(rollout, "exhausted_event_budget", False)
        ),
        "ring_plan": plan_json,
    }
    if trigger_fired is not None:
        record["plan_trigger_fired"] = bool(trigger_fired)
    if configuration is not None and configuration[1] == REALIZE_AT_START:
        # Plan events happen before the CTMC starts, so they are not in
        # ``rollout.event_times``; total work is the sum.
        record["events"] = len(rollout.event_times) + plan_events
        record["process_events"] = len(rollout.event_times)
    return record
