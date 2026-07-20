"""Quotient-correct distillation from a pre-quotient marked rewrite teacher.

The retained pancake checkpoint assigns rates to syntactic rewrite matches.
Several root-free Graft matches can describe the same molecular successor and
some symmetry matches leave the canonical molecule unchanged.  Deleting those
matches and renormalizing the remaining softmax changes every productive rate.

This module instead separates the teacher measure into

* productive rates on distinct canonical molecular successors; and
* virtual self/calibration mass that advances the proposal clock but is not a
  jump of the molecular CTMC.

The distillation objective matches the productive family rates and the
canonical Graft-successor conditional law.  Its optimum therefore has total
hazard equal to the teacher's productive hazard and preserves every retained
transition intensity without importing the rejected P1/P2 topology scalar.
Small-ring and joint ring-context calibration are deliberately outside this
base target and require a separate gate.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
from math import exp, isfinite
from typing import Any, Iterable, Protocol

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.model.factorized_tracelet_rate_model import (
    MARK_RULE_NAMES,
    MARK_RULE_TO_INDEX,
    FactorizedMarkPrediction,
    FactorizedTraceletRateModel,
    SampledRewriteMark,
    _legacy_prequotient_graft_tables,
    _masked_family_logits,
    factorized_mark_bregman_loss,
    molecular_state_cache_key,
    prepare_factorized_mark_batch,
)
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
from compose_v4.rewrite.operators import BondReroute


_GRAFT_GROUP_KEY_CACHE_LIMIT = 2048
_GRAFT_GROUP_KEY_CACHE: OrderedDict[
    tuple[bytes, bytes, bytes, bytes],
    tuple[tuple[str, ...], tuple[int, ...]],
] = OrderedDict()


@dataclass(frozen=True)
class MarkedRateRecord:
    """One syntactic marked rewrite and its nonnegative CTMC rate."""

    rule_name: str
    action: Any
    rate: float


@dataclass(frozen=True)
class CanonicalSuccessorAggregation:
    """Exact productive/virtual decomposition of explicit marked rates."""

    source_key: str
    successor_rates: dict[str, float]
    successor_alias_counts: dict[str, int]
    input_family_rates: dict[str, float]
    productive_family_rates: dict[str, float]
    virtual_self_family_rates: dict[str, float]
    virtual_backtrack_family_rates: dict[str, float]

    @property
    def input_total_rate(self) -> float:
        return float(sum(self.input_family_rates.values()))

    @property
    def productive_total_rate(self) -> float:
        return float(sum(self.successor_rates.values()))

    @property
    def virtual_self_rate(self) -> float:
        return float(sum(self.virtual_self_family_rates.values()))

    @property
    def virtual_backtrack_rate(self) -> float:
        return float(sum(self.virtual_backtrack_family_rates.values()))

    @property
    def mass_balance_error(self) -> float:
        return abs(
            self.input_total_rate
            - self.productive_total_rate
            - self.virtual_self_rate
            - self.virtual_backtrack_rate
        )


def aggregate_canonical_successor_rates(
    state: MolecularGraph,
    marked_rates: Iterable[MarkedRateRecord | tuple[str, Any, float]],
    *,
    previous_state_key: str | None = None,
) -> CanonicalSuccessorAggregation:
    """Aggregate aliases and route self/reverse rates without renormalizing.

    ``previous_state_key`` optionally augments the state with one accepted-step
    history.  A proposed immediate return to that key is accounted as virtual
    backtrack mass.  All other productive rates are numerically unchanged.
    """

    runtime = de_novo_rewrite_system()
    source_key = canonical_state_key(state)
    successor_rates: dict[str, float] = {}
    alias_counts: dict[str, int] = {}
    input_family: dict[str, float] = {}
    productive_family: dict[str, float] = {}
    virtual_self: dict[str, float] = {}
    virtual_backtrack: dict[str, float] = {}

    for raw_record in marked_rates:
        record = (
            raw_record
            if isinstance(raw_record, MarkedRateRecord)
            else MarkedRateRecord(*raw_record)
        )
        rate = float(record.rate)
        if not isfinite(rate) or rate < 0.0:
            raise ValueError("marked rewrite rates must be finite and nonnegative")
        if rate == 0.0:
            continue
        rule_name = str(record.rule_name)
        input_family[rule_name] = input_family.get(rule_name, 0.0) + rate
        successor = runtime.apply(state, rule_name, record.action)
        successor_key = canonical_state_key(successor)
        if successor_key == source_key:
            virtual_self[rule_name] = virtual_self.get(rule_name, 0.0) + rate
            continue
        if previous_state_key is not None and successor_key == previous_state_key:
            virtual_backtrack[rule_name] = (
                virtual_backtrack.get(rule_name, 0.0) + rate
            )
            continue
        successor_rates[successor_key] = successor_rates.get(successor_key, 0.0) + rate
        alias_counts[successor_key] = alias_counts.get(successor_key, 0) + 1
        productive_family[rule_name] = productive_family.get(rule_name, 0.0) + rate

    return CanonicalSuccessorAggregation(
        source_key=source_key,
        successor_rates=dict(sorted(successor_rates.items())),
        successor_alias_counts=dict(sorted(alias_counts.items())),
        input_family_rates=dict(sorted(input_family.items())),
        productive_family_rates=dict(sorted(productive_family.items())),
        virtual_self_family_rates=dict(sorted(virtual_self.items())),
        virtual_backtrack_family_rates=dict(sorted(virtual_backtrack.items())),
    )


@dataclass(frozen=True)
class PancakeQuotientCalibration:
    """Base calibration included in the permanent pancake distillation lane.

    Atom-delete thinning is a family-rate correction and is included.  The
    action-dependent small-ring multiplier is intentionally excluded here; it
    belongs to the separately gated ring-context extension.
    """

    atom_delete_log_rate_adjustment: float = -0.5

    def __post_init__(self) -> None:
        value = float(self.atom_delete_log_rate_adjustment)
        if not isfinite(value) or value > 0.0:
            raise ValueError("delete log-rate adjustment must be finite and non-positive")


@dataclass(frozen=True)
class FactorizedQuotientRateTable:
    """Differentiable family and canonical-Graft rate table for one state."""

    source_key: str
    graft_successor_keys: tuple[str, ...]
    total_hazard: Tensor
    family_rates: Tensor
    graft_successor_rates: Tensor
    enabled_families: Tensor
    raw_graft_mark_count: int
    productive_graft_mark_count: int
    graft_alias_counts: Tensor

    @property
    def graft_family_rate(self) -> Tensor:
        return self.family_rates[MARK_RULE_TO_INDEX["bond_reroute"]]


@dataclass(frozen=True)
class PancakeQuotientTarget:
    """Detached calibrated-teacher rates on the productive quotient."""

    source_key: str
    graft_successor_keys: tuple[str, ...]
    raw_total_hazard: Tensor
    productive_total_hazard: Tensor
    raw_family_rates: Tensor
    productive_family_rates: Tensor
    graft_successor_rates: Tensor
    virtual_family_rates: Tensor
    raw_graft_mark_count: int
    productive_graft_mark_count: int
    graft_alias_counts: Tensor

    @property
    def virtual_total_rate(self) -> Tensor:
        return self.virtual_family_rates.sum()

    @property
    def mass_balance_error(self) -> Tensor:
        return (
            self.raw_total_hazard
            - self.productive_total_hazard
            - self.virtual_total_rate
        ).abs()


@dataclass(frozen=True)
class PancakeQuotientStateFeatures:
    """Exact state-conditioned features needed by any learned correction."""

    raw_total_hazard: Tensor
    raw_family_rates: Tensor
    productive_survival_fractions: Tensor
    raw_graft_log_partition: Tensor
    productive_graft_log_partition: Tensor


@dataclass(frozen=True)
class _AnalyticPancakeQuotientContext:
    target: PancakeQuotientTarget
    features: PancakeQuotientStateFeatures
    batch: Any
    node: Tensor
    global_state: Tensor
    pair: Tensor
    masks: dict[str, Tensor]
    logits: dict[str, Tensor]


def _masked_logsumexp(values: Tensor, mask: Tensor) -> Tensor:
    if values.shape != mask.shape:
        raise ValueError("masked values and support must have equal shapes")
    if not bool(mask.any()):
        return values.new_tensor(float("-inf"))
    return torch.logsumexp(values.masked_select(mask), dim=0)


def _graft_group_keys(
    state: MolecularGraph,
    quotient_mask: Tensor,
    remove_neighbors: Tensor,
    successor_groups: Tensor,
) -> tuple[tuple[str, ...], Tensor]:
    state_cache_key = molecular_state_cache_key(state)
    cached = _GRAFT_GROUP_KEY_CACHE.get(state_cache_key)
    if cached is not None:
        _GRAFT_GROUP_KEY_CACHE.move_to_end(state_cache_key)
        keys, counts = cached
        return keys, torch.tensor(counts, dtype=torch.long)
    runtime = de_novo_rewrite_system()
    group_values = torch.unique(
        successor_groups.masked_select(quotient_mask),
        sorted=True,
    )
    keys: list[str] = []
    counts: list[int] = []
    for group_tensor in group_values:
        group = int(group_tensor)
        coordinates = torch.nonzero(
            quotient_mask & (successor_groups == group),
            as_tuple=False,
        )
        if not len(coordinates):
            raise RuntimeError("canonical Graft group is empty")
        moved, target = (int(value) for value in coordinates[0])
        removed = int(remove_neighbors[moved, target])
        successor = runtime.apply(
            state,
            "bond_reroute",
            BondReroute(a=moved, b=removed, u=moved, v=target),
        )
        key = canonical_state_key(successor)
        if key == canonical_state_key(state):
            raise RuntimeError("productive Graft group resolved to the source")
        keys.append(key)
        counts.append(len(coordinates))
    if len(keys) != len(set(keys)):
        raise RuntimeError("distinct Graft group ids resolved to one successor")
    result = (tuple(keys), tuple(counts))
    _GRAFT_GROUP_KEY_CACHE[state_cache_key] = result
    while len(_GRAFT_GROUP_KEY_CACHE) > _GRAFT_GROUP_KEY_CACHE_LIMIT:
        _GRAFT_GROUP_KEY_CACHE.popitem(last=False)
    return result[0], torch.tensor(result[1], dtype=torch.long)


def _factorized_rate_table(
    model: FactorizedTraceletRateModel,
    state: MolecularGraph,
    time: float,
    *,
    legacy_prequotient_graft: bool,
) -> FactorizedQuotientRateTable:
    """Return one differentiable model table under legacy or quotient support."""

    state_cache_key = model._state_cache_key(state)
    cached_batch = model._sampling_state_cache.get(state_cache_key)
    if cached_batch is None:
        cached_batch = prepare_factorized_mark_batch(
            (state,),
            (0.0,),
            (None,),
            (None,),
            (0.0,),
            ring_catalog=model.ring_catalog,
        )
        model._sampling_state_cache[state_cache_key] = cached_batch
        while len(model._sampling_state_cache) > model._sampling_state_cache_limit:
            model._sampling_state_cache.popitem(last=False)
    else:
        model._sampling_state_cache.move_to_end(state_cache_key)
    batch = replace(
        cached_batch,
        times=torch.tensor((float(time),), dtype=torch.float32),
    ).to(model.device)
    node, global_state, pair = model._encode_batch(batch)
    masks, logits, action_log_z = model._action_tables(
        batch,
        node,
        global_state,
        pair,
        require_exact_ring_support=False,
    )
    quotient_mask = masks["bond_reroute"][0]
    raw_mask_array, _ = _legacy_prequotient_graft_tables(state)
    raw_mask = torch.from_numpy(raw_mask_array).to(model.device)
    selected_mask = raw_mask if legacy_prequotient_graft else quotient_mask
    selected_log_z = _masked_logsumexp(logits["bond_reroute"][0], selected_mask)
    adjusted_log_z = action_log_z.clone()
    adjusted_log_z[0, MARK_RULE_TO_INDEX["bond_reroute"]] = selected_log_z
    enabled = torch.isfinite(adjusted_log_z)
    family_logits = _masked_family_logits(
        model._family_base_logits(batch, global_state),
        adjusted_log_z,
        enabled,
        rate_factorization=model.rate_factorization,
    )
    family_probabilities = torch.softmax(family_logits, dim=-1)[0]
    has_legal_mark = enabled[0].any()
    total_hazard = F.softplus(model.total_hazard_head(global_state)[0, 0]) * (
        has_legal_mark.to(global_state.dtype)
    )
    family_rates = total_hazard * family_probabilities

    group_keys, alias_counts = _graft_group_keys(
        state,
        quotient_mask.detach().cpu(),
        batch.graft_remove_neighbors[0].detach().cpu(),
        batch.graft_successor_groups[0].detach().cpu(),
    )
    group_rates: list[Tensor] = []
    denominator = _masked_logsumexp(logits["bond_reroute"][0], selected_mask)
    for group in range(len(group_keys)):
        group_mask = quotient_mask & (batch.graft_successor_groups[0] == group)
        group_log_mass = _masked_logsumexp(logits["bond_reroute"][0], group_mask)
        conditional_mass = torch.where(
            torch.isfinite(denominator),
            torch.exp(group_log_mass - denominator),
            group_log_mass.new_zeros(()),
        )
        group_rates.append(
            family_rates[MARK_RULE_TO_INDEX["bond_reroute"]] * conditional_mass
        )
    graft_rates = (
        torch.stack(group_rates)
        if group_rates
        else family_rates.new_zeros((0,))
    )
    return FactorizedQuotientRateTable(
        source_key=canonical_state_key(state),
        graft_successor_keys=group_keys,
        total_hazard=total_hazard,
        family_rates=family_rates,
        graft_successor_rates=graft_rates,
        enabled_families=enabled[0],
        raw_graft_mark_count=int(raw_mask.sum()),
        productive_graft_mark_count=int(quotient_mask.sum()),
        graft_alias_counts=alias_counts.to(model.device),
    )


def predict_factorized_quotient_rates(
    model: FactorizedTraceletRateModel,
    state: MolecularGraph,
    time: float,
) -> FactorizedQuotientRateTable:
    """Return the student's productive quotient rates for one state."""

    return _factorized_rate_table(
        model,
        state,
        time,
        legacy_prequotient_graft=False,
    )


def _build_analytic_pancake_quotient_context(
    teacher: FactorizedTraceletRateModel,
    state: MolecularGraph,
    time: float,
    *,
    calibration: PancakeQuotientCalibration,
) -> _AnalyticPancakeQuotientContext:
    """Compute the lossless quotient projection in one encoder/action pass."""

    if teacher.property_condition_dim:
        raise ValueError("analytic pancake projection requires an unconditional model")
    state_cache_key = teacher._state_cache_key(state)
    cached_batch = teacher._sampling_state_cache.get(state_cache_key)
    if cached_batch is None:
        cached_batch = prepare_factorized_mark_batch(
            (state,),
            (0.0,),
            (None,),
            (None,),
            (0.0,),
            ring_catalog=teacher.ring_catalog,
        )
        teacher._sampling_state_cache[state_cache_key] = cached_batch
        while len(teacher._sampling_state_cache) > teacher._sampling_state_cache_limit:
            teacher._sampling_state_cache.popitem(last=False)
    else:
        teacher._sampling_state_cache.move_to_end(state_cache_key)
    batch = replace(
        cached_batch,
        times=torch.tensor((float(time),), dtype=torch.float32),
    ).to(teacher.device)
    node, global_state, pair = teacher._encode_batch(batch)
    masks, logits, quotient_action_log_z = teacher._action_tables(
        batch,
        node,
        global_state,
        pair,
        require_exact_ring_support=False,
    )
    quotient_mask = masks["bond_reroute"][0]
    raw_mask_array, _ = _legacy_prequotient_graft_tables(state)
    raw_mask = torch.from_numpy(raw_mask_array).to(teacher.device)
    raw_graft_log_z = _masked_logsumexp(logits["bond_reroute"][0], raw_mask)
    productive_graft_log_z = _masked_logsumexp(
        logits["bond_reroute"][0],
        quotient_mask,
    )
    raw_action_log_z = quotient_action_log_z.clone()
    raw_action_log_z[0, MARK_RULE_TO_INDEX["bond_reroute"]] = raw_graft_log_z
    enabled = torch.isfinite(raw_action_log_z)
    raw_family_logits = _masked_family_logits(
        teacher._family_base_logits(batch, global_state),
        raw_action_log_z,
        enabled,
        rate_factorization=teacher.rate_factorization,
    )
    raw_family_probabilities = torch.softmax(raw_family_logits, dim=-1)[0]
    has_legal_mark = enabled[0].any()
    raw_total_hazard = F.softplus(teacher.total_hazard_head(global_state)[0, 0]) * (
        has_legal_mark.to(global_state.dtype)
    )
    raw_family_rates = raw_total_hazard * raw_family_probabilities

    survival = torch.ones_like(raw_family_rates)
    delete_index = MARK_RULE_TO_INDEX["atom_delete"]
    graft_index = MARK_RULE_TO_INDEX["bond_reroute"]
    survival[delete_index] = exp(float(calibration.atom_delete_log_rate_adjustment))
    if torch.isfinite(raw_graft_log_z) and torch.isfinite(productive_graft_log_z):
        survival[graft_index] = torch.exp(
            productive_graft_log_z - raw_graft_log_z
        )
    else:
        survival[graft_index] = 0.0
    productive_family_rates = raw_family_rates * survival

    group_keys, alias_counts = _graft_group_keys(
        state,
        quotient_mask.detach().cpu(),
        batch.graft_remove_neighbors[0].detach().cpu(),
        batch.graft_successor_groups[0].detach().cpu(),
    )
    group_rates: list[Tensor] = []
    for group in range(len(group_keys)):
        group_mask = quotient_mask & (batch.graft_successor_groups[0] == group)
        group_log_mass = _masked_logsumexp(logits["bond_reroute"][0], group_mask)
        group_rates.append(
            raw_family_rates[graft_index]
            * torch.exp(group_log_mass - raw_graft_log_z)
        )
    graft_successor_rates = (
        torch.stack(group_rates)
        if group_rates
        else raw_family_rates.new_zeros((0,))
    )
    productive_family_rates[graft_index] = graft_successor_rates.sum()
    productive_total_hazard = productive_family_rates.sum()
    virtual_family_rates = (raw_family_rates - productive_family_rates).clamp_min(0.0)
    target = PancakeQuotientTarget(
        source_key=canonical_state_key(state),
        graft_successor_keys=group_keys,
        raw_total_hazard=raw_total_hazard.detach(),
        productive_total_hazard=productive_total_hazard.detach(),
        raw_family_rates=raw_family_rates.detach(),
        productive_family_rates=productive_family_rates.detach(),
        graft_successor_rates=graft_successor_rates.detach(),
        virtual_family_rates=virtual_family_rates.detach(),
        raw_graft_mark_count=int(raw_mask.sum()),
        productive_graft_mark_count=int(quotient_mask.sum()),
        graft_alias_counts=alias_counts.to(teacher.device),
    )
    features = PancakeQuotientStateFeatures(
        raw_total_hazard=raw_total_hazard.detach(),
        raw_family_rates=raw_family_rates.detach(),
        productive_survival_fractions=survival.detach(),
        raw_graft_log_partition=raw_graft_log_z.detach(),
        productive_graft_log_partition=productive_graft_log_z.detach(),
    )
    return _AnalyticPancakeQuotientContext(
        target=target,
        features=features,
        batch=batch,
        node=node,
        global_state=global_state,
        pair=pair,
        masks=masks,
        logits=logits,
    )


def build_pancake_quotient_state_features(
    teacher: FactorizedTraceletRateModel,
    state: MolecularGraph,
    time: float,
    *,
    calibration: PancakeQuotientCalibration = PancakeQuotientCalibration(),
) -> PancakeQuotientStateFeatures:
    """Return the state-dependent survival signal missing from head-only fit."""

    with torch.no_grad():
        return _build_analytic_pancake_quotient_context(
            teacher,
            state,
            time,
            calibration=calibration,
        ).features


@dataclass
class AnalyticPancakeQuotientSampler:
    """Lossless direct sampler for the productive pancake quotient CTMC."""

    base_model: FactorizedTraceletRateModel
    calibration: PancakeQuotientCalibration = PancakeQuotientCalibration()
    context_cache_limit: int = 2048
    _context_cache: OrderedDict[
        tuple[tuple[bytes, bytes, bytes, bytes], float],
        _AnalyticPancakeQuotientContext,
    ] = field(default_factory=OrderedDict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.context_cache_limit <= 0:
            raise ValueError("analytic context cache limit must be positive")

    def _context(
        self,
        state: MolecularGraph,
        time: float,
    ) -> _AnalyticPancakeQuotientContext:
        key = (self.base_model._state_cache_key(state), float(time))
        cached = self._context_cache.get(key)
        if cached is not None:
            self._context_cache.move_to_end(key)
            return cached
        with torch.no_grad():
            context = _build_analytic_pancake_quotient_context(
                self.base_model,
                state,
                time,
                calibration=self.calibration,
            )
        self._context_cache[key] = context
        while len(self._context_cache) > self.context_cache_limit:
            self._context_cache.popitem(last=False)
        return context

    def rate_table(
        self,
        state: MolecularGraph,
        time: float,
    ) -> PancakeQuotientTarget:
        return self._context(state, time).target

    def state_features(
        self,
        state: MolecularGraph,
        time: float,
    ) -> PancakeQuotientStateFeatures:
        return self._context(state, time).features

    @torch.no_grad()
    def sample_rewrite_mark(
        self,
        state: MolecularGraph,
        time: float,
        rng: np.random.Generator,
    ) -> SampledRewriteMark:
        context = self._context(state, time)
        target = context.target
        hazard = float(target.productive_total_hazard)
        if hazard <= 1e-12:
            return SampledRewriteMark(0.0, "<TERMINAL>", None)
        probabilities = (
            target.productive_family_rates / target.productive_total_hazard
        ).float().cpu().numpy()
        family_index = int(rng.choice(len(MARK_RULE_NAMES), p=probabilities))
        rule_name = MARK_RULE_NAMES[family_index]
        if rule_name == "bond_reroute":
            group_probabilities = (
                target.graft_successor_rates
                / target.productive_family_rates[family_index]
            ).float().cpu().numpy()
            group = int(rng.choice(len(group_probabilities), p=group_probabilities))
            group_mask = context.masks["bond_reroute"][0] & (
                context.batch.graft_successor_groups[0] == group
            )
            coordinates = torch.nonzero(group_mask, as_tuple=False)
            if not len(coordinates):
                raise RuntimeError("sampled canonical Graft group is empty")
            moved, target_slot = (int(value) for value in coordinates[0])
            removed = int(
                context.batch.graft_remove_neighbors[0, moved, target_slot]
            )
            action = BondReroute(
                a=moved,
                b=removed,
                u=moved,
                v=target_slot,
            )
        else:
            action = self.base_model._sample_action_from_family(
                rule_name,
                state,
                context.batch,
                context.node[0],
                context.global_state[0],
                context.pair[0],
                context.masks,
                context.logits,
                rng,
            )
        if action is None:
            raise RuntimeError("positive analytic family rate had no executable action")
        return SampledRewriteMark(hazard, rule_name, action)


@dataclass(frozen=True)
class FrozenQEDResidualConfig:
    """Frozen valid-state conditional-control sidecar configuration.

    The sidecar has exactly two responsibilities: a log-rate residual for
    each rewrite family and normalized within-family residuals induced through
    shifted frozen action features.  There is deliberately no independent
    total-hazard head.  A missing condition is structurally gated to the exact
    analytic quotient base, including after the sidecar has been trained.
    """

    qed_mean: float
    qed_standard_deviation: float
    hidden_dim: int = 128

    def __post_init__(self) -> None:
        if not isfinite(float(self.qed_mean)):
            raise ValueError("QED mean must be finite")
        if (
            not isfinite(float(self.qed_standard_deviation))
            or float(self.qed_standard_deviation) <= 0.0
        ):
            raise ValueError("QED standard deviation must be finite and positive")
        if int(self.hidden_dim) <= 0:
            raise ValueError("conditional sidecar hidden dimension must be positive")


@dataclass(frozen=True)
class FrozenQEDResidualRateTable:
    """Differentiable conditional rates for one valid molecular state."""

    source_key: str
    total_hazard: Tensor
    family_rates: Tensor
    family_log_probabilities: Tensor
    family_log_hazard_residuals: Tensor
    enabled_families: Tensor


@dataclass(frozen=True)
class _FrozenQEDResidualContext:
    base: _AnalyticPancakeQuotientContext
    table: FrozenQEDResidualRateTable
    batch: Any
    node: Tensor
    global_state: Tensor
    pair: Tensor
    masks: dict[str, Tensor]
    logits: dict[str, Tensor]
    action_log_z: Tensor


def _zero_last_linear(module: nn.Sequential) -> None:
    final = module[-1]
    if not isinstance(final, nn.Linear):
        raise TypeError("zero-initialized residual network must end in Linear")
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)


class FrozenQEDResidualAdapter(nn.Module):
    """Trainable QED sidecar over a frozen analytic quotient generator.

    The unconditional model is intentionally not registered as a child module,
    so a sidecar checkpoint cannot silently duplicate or modify its weights.
    Its identity is supplied separately by the qualification/launch manifest.
    """

    adapter_kind = "frozen_valid_state_qed_residual_v1"

    def __init__(
        self,
        base_sampler: AnalyticPancakeQuotientSampler,
        config: FrozenQEDResidualConfig,
    ) -> None:
        super().__init__()
        base_model = base_sampler.base_model
        if int(base_model.property_condition_dim) != 0:
            raise ValueError("frozen residual sidecar requires an unconditional base")
        if base_model.empirical_mark_prior_mode != "none":
            raise ValueError("frozen residual sidecar prohibits empirical mark priors")
        if base_model.ring_family_mass_mode != "boolean":
            raise ValueError("frozen residual sidecar requires Boolean ring-family mass")
        for parameter in base_model.parameters():
            parameter.requires_grad_(False)
        base_model.eval()

        # Bypass nn.Module registration: the base is a separately qualified
        # artifact and must never be serialized into the trainable sidecar.
        object.__setattr__(self, "_base_sampler", base_sampler)
        self.config = config
        width = int(base_model.hidden_dim)
        adapter_width = int(config.hidden_dim)
        self.register_buffer(
            "qed_mean",
            torch.tensor(float(config.qed_mean), dtype=torch.float32),
        )
        self.register_buffer(
            "qed_standard_deviation",
            torch.tensor(float(config.qed_standard_deviation), dtype=torch.float32),
        )
        self.condition_encoder = nn.Sequential(
            nn.Linear(2, adapter_width),
            nn.SiLU(),
            nn.Linear(adapter_width, width),
        )
        self.global_residual = nn.Sequential(
            nn.Linear(2 * width, adapter_width),
            nn.SiLU(),
            nn.Linear(adapter_width, width),
        )
        self.node_residual = nn.Sequential(
            nn.Linear(3 * width, adapter_width),
            nn.SiLU(),
            nn.Linear(adapter_width, width),
        )
        self.pair_residual = nn.Sequential(
            nn.Linear(3 * width, adapter_width),
            nn.SiLU(),
            nn.Linear(adapter_width, width),
        )
        self.family_log_hazard_residual = nn.Sequential(
            nn.Linear(2 * width, adapter_width),
            nn.SiLU(),
            nn.Linear(adapter_width, len(MARK_RULE_NAMES)),
        )
        for module in (
            self.global_residual,
            self.node_residual,
            self.pair_residual,
            self.family_log_hazard_residual,
        ):
            _zero_last_linear(module)
        self.to(base_model.device)

    @property
    def base_sampler(self) -> AnalyticPancakeQuotientSampler:
        return self._base_sampler

    @property
    def base_model(self) -> FactorizedTraceletRateModel:
        return self._base_sampler.base_model

    def train(self, mode: bool = True) -> "FrozenQEDResidualAdapter":
        super().train(mode)
        self.base_model.eval()
        return self

    def sidecar_contract(self) -> dict[str, object]:
        return {
            "kind": self.adapter_kind,
            "property_names": ["qed"],
            "base_parameters_frozen": all(
                not parameter.requires_grad for parameter in self.base_model.parameters()
            ),
            "base_weights_in_sidecar_state_dict": any(
                key.startswith("_base_sampler.") or key.startswith("base_model.")
                for key in self.state_dict()
            ),
            "family_hazard_residual": True,
            "within_family_residual": True,
            "independent_total_hazard_residual": False,
            "missing_condition_identity": True,
            "qed_mean": float(self.qed_mean),
            "qed_standard_deviation": float(self.qed_standard_deviation),
        }

    def _condition(
        self,
        target_qed: float | None,
        *,
        reference: Tensor,
    ) -> tuple[Tensor, Tensor]:
        present = 0.0 if target_qed is None else 1.0
        value = 0.0 if target_qed is None else float(target_qed)
        if not isfinite(value):
            raise ValueError("target QED must be finite")
        normalized = (
            reference.new_tensor(value) - self.qed_mean.to(reference)
        ) / self.qed_standard_deviation.to(reference)
        condition = torch.stack(
            (normalized, reference.new_tensor(present)),
        ).unsqueeze(0)
        gate = reference.new_tensor(present).reshape(1, 1)
        return self.condition_encoder(condition) * gate, gate

    def _context(
        self,
        state: MolecularGraph,
        time: float,
        *,
        target_qed: float | None,
    ) -> _FrozenQEDResidualContext:
        base = self.base_sampler._context(state, time)
        base_node = base.node.detach()
        base_global = base.global_state.detach()
        base_pair = base.pair.detach()
        condition, gate = self._condition(target_qed, reference=base_global)
        n_slots = base_node.shape[1]
        global_expand_node = base_global.unsqueeze(1).expand(-1, n_slots, -1)
        condition_expand_node = condition.unsqueeze(1).expand(-1, n_slots, -1)
        node = base_node + self.node_residual(
            torch.cat((base_node, global_expand_node, condition_expand_node), dim=-1)
        ) * gate.unsqueeze(1)
        global_state = base_global + self.global_residual(
            torch.cat((base_global, condition), dim=-1)
        ) * gate
        global_expand_pair = base_global[:, None, None, :].expand(
            -1, n_slots, n_slots, -1
        )
        condition_expand_pair = condition[:, None, None, :].expand(
            -1, n_slots, n_slots, -1
        )
        pair = base_pair + self.pair_residual(
            torch.cat((base_pair, global_expand_pair, condition_expand_pair), dim=-1)
        ) * gate[:, None, None, :]
        masks, logits, action_log_z = self.base_model._action_tables(
            base.batch,
            node,
            global_state,
            pair,
            require_exact_ring_support=False,
        )
        for name, base_mask in base.masks.items():
            if name not in masks or not torch.equal(masks[name], base_mask):
                raise RuntimeError("conditional residual changed legal rewrite support")

        family_delta = self.family_log_hazard_residual(
            torch.cat((base_global, condition), dim=-1)
        )[0] * gate[0, 0]
        base_family_rates = base.target.productive_family_rates.to(
            device=family_delta.device,
            dtype=family_delta.dtype,
        )
        family_rates = base_family_rates * torch.exp(family_delta)
        total_hazard = family_rates.sum()
        enabled = base_family_rates > 0.0
        family_log_probabilities = torch.where(
            enabled & (total_hazard > 0.0),
            torch.log(family_rates.clamp_min(1e-30))
            - torch.log(total_hazard.clamp_min(1e-30)),
            family_rates.new_tensor(float("-inf")),
        )
        table = FrozenQEDResidualRateTable(
            source_key=base.target.source_key,
            total_hazard=total_hazard,
            family_rates=family_rates,
            family_log_probabilities=family_log_probabilities,
            family_log_hazard_residuals=family_delta,
            enabled_families=enabled,
        )
        return _FrozenQEDResidualContext(
            base=base,
            table=table,
            batch=base.batch,
            node=node,
            global_state=global_state,
            pair=pair,
            masks=masks,
            logits=logits,
            action_log_z=action_log_z,
        )

    def rate_table(
        self,
        state: MolecularGraph,
        time: float,
        *,
        target_qed: float | None,
    ) -> FrozenQEDResidualRateTable:
        return self._context(state, time, target_qed=target_qed).table

    def forward_mark_example(
        self,
        state: MolecularGraph,
        time: float,
        *,
        target_qed: float | None,
        teacher_rule_name: str | None,
        teacher_action: Any | None,
        teacher_rate: float,
        importance_weight: float = 1.0,
    ) -> tuple[FactorizedMarkPrediction, Any]:
        """Return a one-row prediction/batch consumable by the standard loss."""

        context = self._context(state, time, target_qed=target_qed)
        batch = replace(
            context.batch,
            teacher_rule_names=(teacher_rule_name,),
            teacher_actions=(teacher_action,),
            teacher_rates=torch.tensor(
                (float(teacher_rate),),
                dtype=torch.float32,
                device=self.base_model.device,
            ),
            importance_weights=torch.tensor(
                (float(importance_weight),),
                dtype=torch.float32,
                device=self.base_model.device,
            ),
        )
        selected = self.base_model._selected_mark_log_probability(
            batch,
            node=context.node,
            global_state=context.global_state,
            pair=context.pair,
            masks=context.masks,
            logits=context.logits,
            action_log_z=context.action_log_z,
            family_log_prob=context.table.family_log_probabilities.unsqueeze(0),
        )
        prediction = FactorizedMarkPrediction(
            total_hazard=context.table.total_hazard.unsqueeze(0),
            selected_mark_log_probability=selected,
            family_log_probabilities=(
                context.table.family_log_probabilities.unsqueeze(0)
            ),
            enabled_families=context.table.enabled_families.unsqueeze(0),
        )
        return prediction, batch

    def loss_for_mark_example(self, *args: Any, **kwargs: Any) -> Tensor:
        prediction, batch = self.forward_mark_example(*args, **kwargs)
        return factorized_mark_bregman_loss(prediction, batch)

    @torch.no_grad()
    def sample_rewrite_mark_conditioned(
        self,
        state: MolecularGraph,
        time: float,
        rng: np.random.Generator,
        *,
        target_qed: float | None,
    ) -> SampledRewriteMark:
        context = self._context(state, time, target_qed=target_qed)
        hazard = float(context.table.total_hazard)
        if hazard <= 1e-12:
            return SampledRewriteMark(0.0, "<TERMINAL>", None)
        probabilities = (
            context.table.family_rates / context.table.total_hazard
        ).float().cpu().numpy()
        family_index = int(rng.choice(len(MARK_RULE_NAMES), p=probabilities))
        rule_name = MARK_RULE_NAMES[family_index]
        if rule_name == "bond_reroute":
            group_scores = []
            quotient_mask = context.masks["bond_reroute"][0]
            for group in range(len(context.base.target.graft_successor_keys)):
                group_mask = quotient_mask & (
                    context.batch.graft_successor_groups[0] == group
                )
                group_scores.append(
                    _masked_logsumexp(
                        context.logits["bond_reroute"][0],
                        group_mask,
                    )
                )
            if not group_scores:
                raise RuntimeError("positive conditional Graft rate has no successor group")
            group_probabilities = torch.softmax(
                torch.stack(group_scores), dim=0
            ).float().cpu().numpy()
            group = int(rng.choice(len(group_probabilities), p=group_probabilities))
            group_mask = quotient_mask & (
                context.batch.graft_successor_groups[0] == group
            )
            coordinates = torch.nonzero(group_mask, as_tuple=False)
            if not len(coordinates):
                raise RuntimeError("sampled conditional Graft group is empty")
            moved, target_slot = (int(value) for value in coordinates[0])
            removed = int(
                context.batch.graft_remove_neighbors[0, moved, target_slot]
            )
            action = BondReroute(
                a=moved,
                b=removed,
                u=moved,
                v=target_slot,
            )
        else:
            action = self.base_model._sample_action_from_family(
                rule_name,
                state,
                context.batch,
                context.node[0],
                context.global_state[0],
                context.pair[0],
                context.masks,
                context.logits,
                rng,
            )
        if action is None:
            raise RuntimeError("positive conditional family rate had no executable action")
        return SampledRewriteMark(hazard, rule_name, action)

    @torch.no_grad()
    def sample_rewrite_mark(
        self,
        state: MolecularGraph,
        time: float,
        rng: np.random.Generator,
    ) -> SampledRewriteMark:
        return self.sample_rewrite_mark_conditioned(
            state,
            time,
            rng,
            target_qed=None,
        )


@dataclass(frozen=True)
class FrozenExactRDKitResidualConfig:
    """One deterministic RDKit scalar target using the frozen sidecar design."""

    property_name: str
    mean: float
    standard_deviation: float
    hidden_dim: int = 128

    def __post_init__(self) -> None:
        if self.property_name not in {"qed", "molecular_weight", "crippen_logp"}:
            raise ValueError("unsupported exact-RDKit conditional property")
        if not isfinite(float(self.mean)):
            raise ValueError("property mean must be finite")
        if (
            not isfinite(float(self.standard_deviation))
            or float(self.standard_deviation) <= 0.0
        ):
            raise ValueError("property standard deviation must be finite and positive")
        if int(self.hidden_dim) <= 0:
            raise ValueError("conditional sidecar hidden dimension must be positive")


class FrozenExactRDKitResidualAdapter(FrozenQEDResidualAdapter):
    """Property-named facade reusing the verified frozen residual sidecar."""

    adapter_kind = "frozen_valid_state_exact_rdkit_residual_v1"

    def __init__(
        self,
        base_sampler: AnalyticPancakeQuotientSampler,
        config: FrozenExactRDKitResidualConfig,
    ) -> None:
        object.__setattr__(self, "property_name", str(config.property_name))
        object.__setattr__(self, "property_config", config)
        super().__init__(
            base_sampler,
            FrozenQEDResidualConfig(
                qed_mean=float(config.mean),
                qed_standard_deviation=float(config.standard_deviation),
                hidden_dim=int(config.hidden_dim),
            ),
        )

    def sidecar_contract(self) -> dict[str, object]:
        contract = super().sidecar_contract()
        contract.update(
            {
                "kind": self.adapter_kind,
                "property_names": [self.property_name],
                "normalizer_mean": float(self.qed_mean),
                "normalizer_standard_deviation": float(
                    self.qed_standard_deviation
                ),
            }
        )
        return contract

    def rate_table_target(
        self,
        state: MolecularGraph,
        time: float,
        *,
        target_value: float | None,
    ) -> FrozenQEDResidualRateTable:
        return self.rate_table(state, time, target_qed=target_value)

    def sample_rewrite_mark_for_target(
        self,
        state: MolecularGraph,
        time: float,
        rng: np.random.Generator,
        *,
        target_value: float | None,
    ) -> SampledRewriteMark:
        return self.sample_rewrite_mark_conditioned(
            state,
            time,
            rng,
            target_qed=target_value,
        )


def build_calibrated_pancake_quotient_target(
    teacher: FactorizedTraceletRateModel,
    state: MolecularGraph,
    time: float,
    *,
    calibration: PancakeQuotientCalibration = PancakeQuotientCalibration(),
) -> PancakeQuotientTarget:
    """Project the frozen pancake marked law onto productive molecular jumps."""

    with torch.no_grad():
        raw = _factorized_rate_table(
            teacher,
            state,
            time,
            legacy_prequotient_graft=True,
        )
        productive_family_rates = raw.family_rates.clone()
        graft_index = MARK_RULE_TO_INDEX["bond_reroute"]
        delete_index = MARK_RULE_TO_INDEX["atom_delete"]
        productive_family_rates[graft_index] = raw.graft_successor_rates.sum()
        productive_family_rates[delete_index] = (
            raw.family_rates[delete_index]
            * exp(float(calibration.atom_delete_log_rate_adjustment))
        )
        productive_total = productive_family_rates.sum()
        virtual_family_rates = raw.family_rates - productive_family_rates
        if bool((virtual_family_rates < -1e-7).any()):
            raise RuntimeError("productive projection increased a teacher family rate")
        virtual_family_rates = virtual_family_rates.clamp_min(0.0)
        return PancakeQuotientTarget(
            source_key=raw.source_key,
            graft_successor_keys=raw.graft_successor_keys,
            raw_total_hazard=raw.total_hazard.detach(),
            productive_total_hazard=productive_total.detach(),
            raw_family_rates=raw.family_rates.detach(),
            productive_family_rates=productive_family_rates.detach(),
            graft_successor_rates=raw.graft_successor_rates.detach(),
            virtual_family_rates=virtual_family_rates.detach(),
            raw_graft_mark_count=raw.raw_graft_mark_count,
            productive_graft_mark_count=raw.productive_graft_mark_count,
            graft_alias_counts=raw.graft_alias_counts.detach(),
        )


def canonical_successor_distillation_loss(
    student: FactorizedQuotientRateTable,
    teacher: PancakeQuotientTarget,
    *,
    eps: float = 1e-12,
) -> Tensor:
    """Poisson-KL family loss plus exact canonical-Graft conditional loss."""

    if student.source_key != teacher.source_key:
        raise ValueError("student and teacher source states differ")
    if student.graft_successor_keys != teacher.graft_successor_keys:
        raise ValueError("student and teacher canonical Graft supports differ")
    target_family = teacher.productive_family_rates.to(
        device=student.family_rates.device,
        dtype=student.family_rates.dtype,
    )
    target_graft = teacher.graft_successor_rates.to(
        device=student.graft_successor_rates.device,
        dtype=student.graft_successor_rates.dtype,
    )
    family_loss = student.total_hazard - (
        target_family * torch.log(student.family_rates.clamp_min(eps))
    ).sum()
    graft_family_rate = student.graft_family_rate
    if len(target_graft) and bool((target_graft > 0).any()):
        graft_conditional = student.graft_successor_rates / graft_family_rate.clamp_min(
            eps
        )
        graft_loss = -(
            target_graft * torch.log(graft_conditional.clamp_min(eps))
        ).sum()
    else:
        graft_loss = family_loss.new_zeros(())
    return family_loss + graft_loss


def canonical_successor_distillation_optimum(
    teacher: PancakeQuotientTarget,
    *,
    eps: float = 1e-12,
) -> Tensor:
    """Return the teacher-only minimum used to report excess Poisson KL."""

    target_family = teacher.productive_family_rates
    optimum = teacher.productive_total_hazard - (
        target_family * torch.log(target_family.clamp_min(eps))
    ).sum()
    target_graft = teacher.graft_successor_rates
    graft_total = target_graft.sum()
    if len(target_graft) and bool((target_graft > 0).any()):
        optimum = optimum - (
            target_graft
            * torch.log((target_graft / graft_total.clamp_min(eps)).clamp_min(eps))
        ).sum()
    return optimum


def canonical_successor_distillation_excess(
    student: FactorizedQuotientRateTable,
    teacher: PancakeQuotientTarget,
    *,
    eps: float = 1e-12,
) -> Tensor:
    """Return nonnegative objective excess above the exact teacher optimum."""

    return canonical_successor_distillation_loss(
        student,
        teacher,
        eps=eps,
    ) - canonical_successor_distillation_optimum(teacher, eps=eps).to(
        device=student.total_hazard.device,
        dtype=student.total_hazard.dtype,
    )


class RewriteMarkSampler(Protocol):
    def sample_rewrite_mark(
        self,
        state: MolecularGraph,
        time: float,
        rng: np.random.Generator,
    ) -> SampledRewriteMark:
        ...


@dataclass
class CanonicalHistoryThinningSampler:
    """Exact non-renormalizing safety for self and immediate reverse jumps.

    The wrapper is trajectory-local.  Rejected events retain the base hazard
    and become recorded virtual jumps, so every other marked transition keeps
    its original CTMC intensity.  Distillation should make these rejections
    rare; this wrapper is a semantic safety boundary, not the learning signal.
    """

    base_sampler: RewriteMarkSampler
    suppress_immediate_backtracks: bool = True
    _previous_state_key: str | None = field(default=None, init=False, repr=False)
    _self_rejections: int = field(default=0, init=False, repr=False)
    _backtrack_rejections: int = field(default=0, init=False, repr=False)
    _accepted_events: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._runtime = de_novo_rewrite_system()

    def begin_rollout_audit(self) -> None:
        self._previous_state_key = None
        self._self_rejections = 0
        self._backtrack_rejections = 0
        self._accepted_events = 0
        begin = getattr(self.base_sampler, "begin_rollout_audit", None)
        if callable(begin):
            begin()

    def end_rollout_audit(self) -> dict[str, object]:
        end = getattr(self.base_sampler, "end_rollout_audit", None)
        base = end() if callable(end) else None
        return {
            "method": "canonical_history_exact_thinning_v1",
            "self_rejections": self._self_rejections,
            "immediate_backtrack_rejections": self._backtrack_rejections,
            "accepted_events": self._accepted_events,
            "base_sampler_diagnostics": base,
        }

    def sample_hazard_probe(
        self,
        state: MolecularGraph,
        time: float,
        rng: np.random.Generator,
    ) -> SampledRewriteMark:
        probe = getattr(self.base_sampler, "sample_hazard_probe", None)
        if callable(probe):
            return probe(state, time, rng)
        return self.base_sampler.sample_rewrite_mark(state, time, rng)

    def resample_rewrite_mark_after_event(
        self,
        state: MolecularGraph,
        time: float,
        rng: np.random.Generator,
        initial: SampledRewriteMark,
    ) -> SampledRewriteMark:
        resample = getattr(
            self.base_sampler,
            "resample_rewrite_mark_after_event",
            None,
        )
        sampled = (
            resample(state, time, rng, initial)
            if callable(resample)
            else initial
        )
        if sampled.action is None or float(sampled.total_hazard) <= 1e-12:
            return sampled
        source_key = canonical_state_key(state)
        successor = self._runtime.apply(state, sampled.rule_name, sampled.action)
        successor_key = canonical_state_key(successor)
        if successor_key == source_key:
            self._self_rejections += 1
            return SampledRewriteMark(
                float(sampled.total_hazard),
                "<VIRTUAL_CANONICAL_SELF>",
                None,
            )
        if (
            self.suppress_immediate_backtracks
            and self._previous_state_key is not None
            and successor_key == self._previous_state_key
        ):
            self._backtrack_rejections += 1
            return SampledRewriteMark(
                float(sampled.total_hazard),
                "<VIRTUAL_IMMEDIATE_BACKTRACK>",
                None,
            )
        self._previous_state_key = source_key
        self._accepted_events += 1
        return sampled

    def sample_rewrite_mark(
        self,
        state: MolecularGraph,
        time: float,
        rng: np.random.Generator,
    ) -> SampledRewriteMark:
        initial = self.sample_hazard_probe(state, time, rng)
        return self.resample_rewrite_mark_after_event(
            state,
            time,
            rng,
            initial,
        )


__all__ = [
    "AnalyticPancakeQuotientSampler",
    "CanonicalHistoryThinningSampler",
    "CanonicalSuccessorAggregation",
    "FactorizedQuotientRateTable",
    "FrozenExactRDKitResidualAdapter",
    "FrozenExactRDKitResidualConfig",
    "FrozenQEDResidualAdapter",
    "FrozenQEDResidualConfig",
    "FrozenQEDResidualRateTable",
    "MarkedRateRecord",
    "PancakeQuotientCalibration",
    "PancakeQuotientStateFeatures",
    "PancakeQuotientTarget",
    "aggregate_canonical_successor_rates",
    "build_calibrated_pancake_quotient_target",
    "build_pancake_quotient_state_features",
    "canonical_successor_distillation_excess",
    "canonical_successor_distillation_loss",
    "canonical_successor_distillation_optimum",
    "predict_factorized_quotient_rates",
]
