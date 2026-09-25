"""Uniform native-mark law for the frozen superstructure reference ablation.

This adapter leaves the RingCore model's action masks and total hazard intact,
but replaces its family and action-coordinate probabilities. It is intentionally
specific to the non-virtualized, unconditioned RingCore fragment checkpoint.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F

from compose_v4.model.factorized_tracelet_rate_model import (
    MARK_RULE_NAMES,
    MARK_RULE_TO_INDEX,
    SampledRewriteMark,
    prepare_factorized_mark_batch,
)
from compose_v4.rewrite.operators import (
    BondDelete,
    BondInsert,
    CycleCloseEdge,
    CycleOpenEdge,
)


def _has_native_coordinate(masks: dict[str, torch.Tensor], family: str) -> bool:
    if family == "atom_insert":
        return bool(masks["grow_root"][0].any() or masks["grow_connected"][0].any())
    return bool(masks[family][0].any())


class UniformNativeMarkLaw:
    """Use uniform available-family and uniform within-family native marks."""

    def __init__(self, model):
        if bool(getattr(model, "virtualize_legacy_self_grafts", False)):
            raise ValueError("uniform native law does not support virtualized legacy grafts")
        if bool(model.enable_ring_grow_macro):
            raise ValueError("uniform native law requires the RingCore primitive-ring configuration")
        self.model = model

    def __getattr__(self, name):
        return getattr(self.model, name)

    @torch.no_grad()
    def sample_rewrite_mark(self, state, time: float, rng: np.random.Generator):
        return self.sample_rewrite_mark_conditioned(
            state, time, rng, property_values=None, property_mask=None
        )

    @torch.no_grad()
    def sample_rewrite_mark_conditioned(
        self,
        state,
        time: float,
        rng: np.random.Generator,
        *,
        property_values=None,
        property_mask=None,
        allowed_rule_names: frozenset[str] | None = None,
    ) -> SampledRewriteMark:
        model = self.model
        if property_values is not None or property_mask is not None:
            raise ValueError("superstructure native-law ablation is property-unconditioned")
        if allowed_rule_names is not None:
            unknown = allowed_rule_names.difference(MARK_RULE_TO_INDEX)
            if unknown:
                raise ValueError(f"unknown allowed families: {sorted(unknown)}")
        options = {
            "use_aromatic_bond_view": True,
            "ring_catalog": model.ring_catalog,
            "compute_ring_grow_support": model.enable_ring_grow_macro,
            "compute_ring_restates": model.enable_ring_restates,
            "compute_cyclic_graft": model.enable_cyclic_graft,
            "compute_ring_opening": model.enable_ring_opening,
            "compute_ring_system_delete": model.enable_ring_system_delete,
            "editing_process_semantics": model.editing_process_semantics,
            "atom_restate_action_semantics": model.atom_restate_action_semantics,
            "ring_restate_scorer_mode": model.ring_restate_scorer_mode,
            "cycle_close_action_semantics": model.cycle_close_action_semantics,
            "cycle_open_action_semantics": model.cycle_open_action_semantics,
            "atom_delete_action_semantics": model.atom_delete_action_semantics,
        }
        cache_key = model._state_cache_key(state)
        cached = model._sampling_state_cache.get(cache_key)
        if cached is None:
            cached = prepare_factorized_mark_batch(
                (state,), (0.0,), (None,), (None,), (0.0,), **options
            )
            model._sampling_state_cache[cache_key] = cached
            if len(model._sampling_state_cache) > model._sampling_state_cache_limit:
                model._sampling_state_cache.popitem(last=False)
        else:
            model._sampling_state_cache.move_to_end(cache_key)
        batch = replace(cached, times=torch.tensor((float(time),), dtype=torch.float32)).to(
            model.device
        )
        node, global_state, pair = model._encode_batch(batch)
        masks, logits, action_log_z = model._action_tables(
            batch, node, global_state, pair, require_exact_ring_support=False
        )
        # The native masks, not numerical mark scores, define this arm's
        # support. Verify that the learned sampler sees the same support.
        eligible = []
        disabled = frozenset(str(name) for name in getattr(model, "disabled_sampling_rule_names", ()))
        for family in MARK_RULE_NAMES:
            mask_available = _has_native_coordinate(masks, family)
            learned_available = bool(torch.isfinite(action_log_z[0, MARK_RULE_TO_INDEX[family]]))
            if mask_available != learned_available:
                raise RuntimeError(f"native mask/logit support disagreement for {family}")
            if mask_available and family not in disabled and (
                allowed_rule_names is None or family in allowed_rule_names
            ):
                eligible.append(family)
        uniform_logits = {name: torch.zeros_like(table) for name, table in logits.items()}
        while eligible:
            family = eligible[int(rng.integers(len(eligible)))]
            action = model._sample_action_from_family(
                family, state, batch, node[0], global_state[0], pair[0],
                masks, uniform_logits, rng
            )
            if action is None:
                eligible.remove(family)
                continue
            # Preserve the frozen holding rate and the original executor alias.
            hazard = float(F.softplus(model.total_hazard_head(global_state)[0, 0]))
            applied_rule = family
            if isinstance(action, BondInsert):
                applied_rule = "bond_insert"
            elif isinstance(action, BondDelete):
                applied_rule = "bond_delete"
            elif isinstance(action, CycleCloseEdge):
                applied_rule = "cycle_close"
            elif isinstance(action, CycleOpenEdge):
                applied_rule = "cycle_open"
            return SampledRewriteMark(hazard, applied_rule, action)
        return SampledRewriteMark(0.0, "<TERMINAL>", None)
