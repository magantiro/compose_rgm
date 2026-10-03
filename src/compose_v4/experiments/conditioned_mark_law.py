"""Public normalized conditioning of the factorized marked rewrite law.

This module is the supported boundary for applications that need to restrict a
frozen reference process to a declared, current-state-only action set.  The
application supplies a family set and an action predicate.  We enumerate the
model's exact legal discrete marks, retain the predicate-approved marks, and
renormalize their *original model probabilities*.  There is no rejection loop
and no discarded proposal RNG draw.

The first public surface covers the finite primitive tables used by structured
program controllers.  Structured ring-system macros have conditional decoders
of their own and are intentionally excluded until their full electronic law is
exposed through the same contract.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from math import exp
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from compose_v4.chem.molecular_graph import (
    BOND_CLASS_TO_H_CHANGE,
    NULL_IDX,
    MolecularGraph,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    MARK_RULE_TO_INDEX,
    FactorizedTraceletRateModel,
    _masked_family_logits,
    prepare_factorized_mark_batch,
)
from compose_v4.rewrite.operators import (
    AtomDelete,
    AtomInsert,
    AtomRestate,
    BondReorder,
)
from compose_v4.rewrite.scaffold_construction import ScaffoldContext


PUBLIC_DISCRETE_FAMILIES = frozenset(
    {
        "atom_insert",
        "atom_delete",
        "atom_restate",
        "bond_reorder",
        "cycle_insert",
        "cycle_attach",
    }
)


@dataclass(frozen=True)
class ConditionedMarkCandidate:
    """One legal mark and its probability before and after conditioning."""

    rule_name: str
    action: Any
    base_log_probability: float
    conditioned_log_probability: float

    @property
    def base_probability(self) -> float:
        return exp(self.base_log_probability)

    @property
    def conditioned_probability(self) -> float:
        return exp(self.conditioned_log_probability)


@dataclass(frozen=True)
class ConditionedSampledRewriteMark:
    """A draw from :class:`ConditionedMarkedLaw` with an auditable density."""

    total_hazard: float
    rule_name: str
    action: Any
    base_log_probability: float
    conditioned_log_probability: float
    conditioning_log_mass: float
    condition_id: str


@dataclass(frozen=True)
class ConditionedMarkedLaw:
    """Finite normalized law obtained by conditioning a frozen marked law."""

    total_hazard: float
    condition_id: str
    allowed_families: tuple[str, ...]
    conditioning_log_mass: float
    candidates: tuple[ConditionedMarkCandidate, ...]

    @property
    def conditioning_mass(self) -> float:
        return exp(self.conditioning_log_mass)

    def sample(self, rng: np.random.Generator) -> ConditionedSampledRewriteMark:
        if not self.candidates:
            raise RuntimeError("cannot sample an empty conditioned marked law")
        probabilities = np.asarray(
            tuple(item.conditioned_probability for item in self.candidates),
            dtype=np.float64,
        )
        probabilities /= probabilities.sum()
        candidate = self.candidates[
            int(rng.choice(len(self.candidates), p=probabilities))
        ]
        return ConditionedSampledRewriteMark(
            total_hazard=self.total_hazard,
            rule_name=candidate.rule_name,
            action=candidate.action,
            base_log_probability=candidate.base_log_probability,
            conditioned_log_probability=candidate.conditioned_log_probability,
            conditioning_log_mass=self.conditioning_log_mass,
            condition_id=self.condition_id,
        )


def _prepared_sampling_batch(
    model: FactorizedTraceletRateModel,
    state: MolecularGraph,
    time: float,
    *,
    property_values: tuple[float, ...] | None,
    property_mask: tuple[bool, ...] | None,
    scaffold_context: ScaffoldContext | None,
):
    if (property_values is None) != (property_mask is None):
        raise ValueError("property values and mask must be provided together")
    if property_values is not None:
        if model.property_condition_dim == 0:
            raise ValueError("unconditional model cannot accept property targets")
        if (
            len(property_values) != model.property_condition_dim
            or len(property_mask) != model.property_condition_dim
        ):
            raise ValueError("property target has the wrong dimension")
        values = torch.tensor((property_values,), dtype=torch.float32)
        value_mask = torch.tensor((property_mask,), dtype=torch.bool)
    else:
        values = None
        value_mask = None

    if model.scaffold_conditioning != (scaffold_context is not None):
        raise ValueError("sampler/model scaffold conditioning disagree")
    if scaffold_context is not None and any(
        (
            getattr(model, "virtualize_legacy_self_grafts", False),
            getattr(model, "disabled_sampling_rule_names", ()),
            getattr(model, "excluded_sampling_ring_template_indices", ()),
        )
    ):
        raise ValueError(
            "scaffold sampler cannot use unmatched inference-only support overrides"
        )

    cache_key = (
        model._state_cache_key(state)
        if scaffold_context is None
        else scaffold_context.state_cache_key(state)
    )
    cached = model._sampling_state_cache.get(cache_key)
    if cached is None:
        cached = prepare_factorized_mark_batch(
            (state,),
            (0.0,),
            (None,),
            (None,),
            (0.0,),
            use_aromatic_bond_view=True,
            ring_catalog=model.ring_catalog,
            compute_ring_grow_support=model.enable_ring_grow_macro,
            compute_ring_restates=model.enable_ring_restates,
            compute_cyclic_graft=model.enable_cyclic_graft,
            compute_ring_opening=model.enable_ring_opening,
            compute_ring_system_delete=model.enable_ring_system_delete,
            editing_process_semantics=model.editing_process_semantics,
            atom_restate_action_semantics=model.atom_restate_action_semantics,
            ring_restate_scorer_mode=model.ring_restate_scorer_mode,
            cycle_close_action_semantics=model.cycle_close_action_semantics,
            cycle_open_action_semantics=model.cycle_open_action_semantics,
            atom_delete_action_semantics=model.atom_delete_action_semantics,
            scaffold_contexts=(
                None if scaffold_context is None else (scaffold_context,)
            ),
            node_context_provider=model.node_context_provider,
        )
        model._sampling_state_cache[cache_key] = cached
        if len(model._sampling_state_cache) > model._sampling_state_cache_limit:
            model._sampling_state_cache.popitem(last=False)
    else:
        model._sampling_state_cache.move_to_end(cache_key)
    return replace(
        cached,
        times=torch.tensor((float(time),), dtype=torch.float32),
        property_condition_values=values,
        property_condition_mask=value_mask,
    ).to(model.device)


def _coordinates(logits: Tensor, mask: Tensor):
    valid = torch.nonzero(mask.reshape(-1), as_tuple=False).squeeze(-1)
    for flat_index in valid.tolist():
        coordinate = tuple(
            int(value) for value in np.unravel_index(int(flat_index), logits.shape)
        )
        yield coordinate, logits[coordinate]


def _family_actions(
    model: FactorizedTraceletRateModel,
    state: MolecularGraph,
    family: str,
    masks: dict[str, Tensor],
    logits: dict[str, Tensor],
):
    null_slots = tuple(int(v) for v in np.flatnonzero(state.atom_types == NULL_IDX))
    if family == "atom_insert":
        if not null_slots:
            return
        slot = null_slots[0]
        for (atom_index,), logit in _coordinates(
            logits["grow_root"][0], masks["grow_root"][0]
        ):
            atom_type = int(model.atom_vocabulary.element_of(atom_index))
            yield AtomInsert(
                slot, atom_type, 0, int(model.cnof_valences[atom_index]), ()
            ), logit
        for (neighbor, order_index, atom_index), logit in _coordinates(
            logits["grow_connected"][0], masks["grow_connected"][0]
        ):
            order = order_index + 1
            atom_type = int(model.atom_vocabulary.element_of(atom_index))
            yield AtomInsert(
                slot,
                atom_type,
                0,
                int(model.cnof_valences[atom_index]) - order,
                ((neighbor, order),),
            ), logit
        return
    if family == "atom_delete":
        for (vertex,), logit in _coordinates(
            logits["atom_delete"][0], masks["atom_delete"][0]
        ):
            yield AtomDelete(vertex), logit
        return
    if family == "atom_restate":
        for (vertex, atom_index), logit in _coordinates(
            logits["atom_restate"][0], masks["atom_restate"][0]
        ):
            atom_type = int(model.atom_vocabulary.element_of(atom_index))
            bond_valence = sum(
                int(BOND_CLASS_TO_H_CHANGE[int(order)])
                for order in state.bonds[vertex]
            )
            yield AtomRestate(
                vertex,
                atom_type,
                0,
                int(model.cnof_valences[atom_index]) - bond_valence,
            ), logit
        return
    if family == "bond_reorder":
        for (left, right, order_index), logit in _coordinates(
            logits["bond_reorder"][0], masks["bond_reorder"][0]
        ):
            yield BondReorder(left, right, order_index + 1), logit
        return
    if family == "cycle_insert":
        for (template_index,), logit in _coordinates(
            logits["cycle_insert"][0], masks["cycle_insert"][0]
        ):
            template = model.cycle_templates[template_index]
            yield template.instantiate(null_slots[: template.span]), logit
        return
    if family == "cycle_attach":
        for (anchor, template_index), logit in _coordinates(
            logits["cycle_attach"][0], masks["cycle_attach"][0]
        ):
            template = model.attach_templates[template_index]
            yield template.instantiate(anchor, null_slots[: template.span]), logit
        return
    raise ValueError(f"family is not exposed by the public discrete law: {family}")


@torch.no_grad()
def conditioned_mark_law(
    model: FactorizedTraceletRateModel,
    state: MolecularGraph,
    time: float,
    *,
    allowed_families: tuple[str, ...],
    condition_id: str,
    action_predicate: Callable[[str, Any], bool] | None = None,
    property_values: tuple[float, ...] | None = None,
    property_mask: tuple[bool, ...] | None = None,
    scaffold_context: ScaffoldContext | None = None,
) -> ConditionedMarkedLaw:
    """Return the exact normalized model law on a declared action subset.

    ``action_predicate`` must depend only on the current state, supplied
    condition, and candidate action.  The returned conditioning mass is the
    frozen model probability of the retained set before renormalization.
    """

    if not condition_id:
        raise ValueError("condition_id must be nonempty")
    families = tuple(dict.fromkeys(str(name) for name in allowed_families))
    if not families:
        raise ValueError("allowed_families must be nonempty")
    unsupported = set(families) - PUBLIC_DISCRETE_FAMILIES
    if unsupported:
        raise ValueError(
            f"families lack a public finite discrete law: {sorted(unsupported)}"
        )
    batch = _prepared_sampling_batch(
        model,
        state,
        time,
        property_values=property_values,
        property_mask=property_mask,
        scaffold_context=scaffold_context,
    )
    node, global_state, pair = model._encode_batch(batch)
    masks, logits, action_log_z = model._action_tables(
        batch,
        node,
        global_state,
        pair,
        require_exact_ring_support=scaffold_context is not None,
    )
    enabled = torch.isfinite(action_log_z[0]).clone()
    for disabled in getattr(model, "disabled_sampling_rule_names", ()):
        index = MARK_RULE_TO_INDEX.get(str(disabled))
        if index is not None:
            enabled[index] = False
    family_logits = _masked_family_logits(
        model._family_base_logits(batch, global_state)[0],
        action_log_z[0],
        enabled,
        rate_factorization=model.rate_factorization,
    )
    family_log_probabilities = torch.log_softmax(family_logits, dim=-1)

    retained: list[tuple[str, Any, Tensor]] = []
    predicate = action_predicate or (lambda _family, _action: True)
    for family in families:
        family_index = MARK_RULE_TO_INDEX[family]
        if not bool(enabled[family_index]):
            continue
        within_log_z = action_log_z[0, family_index]
        for action, action_logit in _family_actions(
            model, state, family, masks, logits
        ):
            if predicate(family, action):
                retained.append(
                    (
                        family,
                        action,
                        family_log_probabilities[family_index]
                        + action_logit
                        - within_log_z,
                    )
                )
    if not retained:
        raise ValueError(f"conditioning event {condition_id!r} has zero model support")
    base_log_probabilities = torch.stack(tuple(item[2] for item in retained))
    conditioning_log_mass_tensor = torch.logsumexp(base_log_probabilities, dim=0)
    conditioning_log_mass = float(conditioning_log_mass_tensor)
    candidates = tuple(
        ConditionedMarkCandidate(
            rule_name=family,
            action=action,
            base_log_probability=float(base_log_probability),
            conditioned_log_probability=float(
                base_log_probability - conditioning_log_mass_tensor
            ),
        )
        for family, action, base_log_probability in retained
    )
    total_hazard = float(F.softplus(model.total_hazard_head(global_state)[0, 0]))
    return ConditionedMarkedLaw(
        total_hazard=total_hazard,
        condition_id=condition_id,
        allowed_families=families,
        conditioning_log_mass=conditioning_log_mass,
        candidates=candidates,
    )


__all__ = [
    "PUBLIC_DISCRETE_FAMILIES",
    "ConditionedMarkCandidate",
    "ConditionedMarkedLaw",
    "ConditionedSampledRewriteMark",
    "conditioned_mark_law",
]
