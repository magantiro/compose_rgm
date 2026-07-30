"""Differentiable canonical-successor supervision for the factorized editor.

The production model parameterizes executable mark probabilities.  This module
precompiles the subset of legal mark coordinates that execute to one teacher
molecule and aggregates their normalized probability with ``logsumexp`` during
training.  The executor and canonicalizer are used only during compilation;
the hot gradient path indexes existing action tables and does not replay
chemistry.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.chem.persistent_state_identity import (
    persistent_slot_state_sha256,
)
from compose_v4.experiments.production_successor_kernel import (
    ProductionSuccessorKernelError,
    enumerate_factorized_marked_law,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    _CYCLE_OP_EXECUTOR_TO_FAMILY,
    MARK_RULE_TO_INDEX,
    FactorizedMarkBatch,
    FactorizedTraceletRateModel,
    _masked_family_logits,
)
from compose_v4.rewrite.kernel import (
    RewriteSystem,
    canonical_state_key,
    de_novo_rewrite_system,
)


class SuccessorTrainingError(RuntimeError):
    """A teacher fiber or differentiable score is inconsistent with production."""


@dataclass(frozen=True, order=True)
class TeacherSuccessorAlias:
    """One factorized table coordinate in a teacher-successor fiber."""

    family_name: str
    table_name: str
    coordinate: tuple[int, ...]


@dataclass(frozen=True)
class StateProductiveSupport:
    """Precompiled coordinates whose execution is a canonical self-event.

    The editing kernel is conditioned on taking a productive molecular jump.
    Its normalizer is therefore one minus the aggregate probability of these
    virtual aliases.  An empty tuple means every legal mark is productive.
    """

    source_key: str
    source_state_sha256: str
    virtual_aliases: tuple[TeacherSuccessorAlias, ...] = ()

    def __post_init__(self) -> None:
        if not self.source_key:
            raise ValueError("state-support source key must be nonempty")
        if (
            len(self.source_state_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.source_state_sha256
            )
        ):
            raise ValueError(
                "state-support source_state_sha256 must be a lowercase SHA-256"
            )
        if len(set(self.virtual_aliases)) != len(self.virtual_aliases):
            raise ValueError("state support contains duplicate virtual coordinates")


@dataclass(frozen=True)
class TeacherSuccessorFiber:
    """All legal mark aliases that execute to one canonical teacher successor."""

    source_key: str
    target_key: str
    target_state_sha256: str
    aliases: tuple[TeacherSuccessorAlias, ...]
    state_support: StateProductiveSupport

    def __post_init__(self) -> None:
        if not self.source_key or not self.target_key:
            raise ValueError("teacher-successor keys must be nonempty")
        if self.source_key == self.target_key:
            raise ValueError("a teacher molecular jump cannot target the source state")
        if (
            len(self.target_state_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.target_state_sha256
            )
        ):
            raise ValueError(
                "teacher target_state_sha256 must be a lowercase SHA-256"
            )
        if not self.aliases:
            raise ValueError("a teacher-successor fiber must contain at least one alias")
        if len(set(self.aliases)) != len(self.aliases):
            raise ValueError("a teacher-successor fiber contains duplicate coordinates")
        if self.state_support.source_key != self.source_key:
            raise ValueError("teacher fiber and state support have different sources")
        if set(self.aliases) & set(self.state_support.virtual_aliases):
            raise ValueError("a teacher successor cannot also be a virtual self-event")


@dataclass(frozen=True)
class FactorizedSuccessorPrediction:
    """Marked and productive molecular quantities for successor supervision."""

    total_hazard: Tensor
    productive_hazard: Tensor
    selected_successor_log_probability: Tensor
    selected_productive_successor_log_probability: Tensor
    productive_log_probability: Tensor
    teacher_family_log_probability: Tensor
    selected_within_teacher_family_log_probability: Tensor
    family_log_probabilities: Tensor
    enabled_families: Tensor

    @property
    def selected_successor_rate(self) -> Tensor:
        return self.total_hazard * self.selected_successor_log_probability.exp()


def _alias(
    *,
    family_name: str,
    table_name: str,
    coordinate: tuple[int, ...],
) -> TeacherSuccessorAlias:
    return TeacherSuccessorAlias(
        family_name=family_name,
        table_name=table_name,
        coordinate=coordinate,
    )


def compile_teacher_successor_fiber(
    model: FactorizedTraceletRateModel,
    source: MolecularGraph,
    target: MolecularGraph,
    *,
    time: float = 0.5,
    system: RewriteSystem | None = None,
) -> TeacherSuccessorFiber:
    """Compile every production mark from ``source`` that canonicalizes to ``target``."""

    runtime = system or de_novo_rewrite_system()
    source_key = canonical_state_key(source)
    target_key = canonical_state_key(target)
    if source_key == target_key:
        raise SuccessorTrainingError(
            "teacher successor equals the source; virtual events are not molecular jumps"
        )
    try:
        marked_law = enumerate_factorized_marked_law(model, source, float(time))
    except ProductionSuccessorKernelError as error:
        raise SuccessorTrainingError(
            "could not enumerate the production marked law for teacher-fiber compilation"
        ) from error

    aliases: list[TeacherSuccessorAlias] = []
    virtual_aliases: list[TeacherSuccessorAlias] = []
    for mark in marked_law.marks:
        try:
            successor = runtime.apply(
                source,
                mark.executor_rule_name,
                mark.action,
            )
        except Exception as error:
            raise SuccessorTrainingError(
                "a production-enumerated mark failed during teacher-fiber compilation"
            ) from error
        successor_key = canonical_state_key(successor)
        compiled_alias = _alias(
            family_name=mark.family_name,
            table_name=mark.table_name,
            coordinate=mark.coordinate,
        )
        if successor_key == source_key:
            virtual_aliases.append(compiled_alias)
        if successor_key == target_key:
            aliases.append(compiled_alias)
    if not aliases:
        raise SuccessorTrainingError(
            "teacher successor is absent from the production marked support"
        )
    return TeacherSuccessorFiber(
        source_key=source_key,
        target_key=target_key,
        target_state_sha256=persistent_slot_state_sha256(target),
        aliases=tuple(sorted(aliases)),
        state_support=StateProductiveSupport(
            source_key=source_key,
            source_state_sha256=persistent_slot_state_sha256(source),
            virtual_aliases=tuple(sorted(virtual_aliases)),
        ),
    )


def compile_state_productive_support(
    model: FactorizedTraceletRateModel,
    source: MolecularGraph,
    *,
    time: float = 0.5,
    system: RewriteSystem | None = None,
) -> StateProductiveSupport:
    """Compile canonical self-event coordinates for a terminal/no-teacher row."""

    runtime = system or de_novo_rewrite_system()
    source_key = canonical_state_key(source)
    try:
        marked_law = enumerate_factorized_marked_law(model, source, float(time))
    except ProductionSuccessorKernelError as error:
        raise SuccessorTrainingError(
            "could not enumerate the production marked law for state-support compilation"
        ) from error
    virtual_aliases: list[TeacherSuccessorAlias] = []
    for mark in marked_law.marks:
        try:
            successor = runtime.apply(
                source,
                mark.executor_rule_name,
                mark.action,
            )
        except Exception as error:
            raise SuccessorTrainingError(
                "a production-enumerated mark failed during state-support compilation"
            ) from error
        if canonical_state_key(successor) == source_key:
            virtual_aliases.append(
                _alias(
                    family_name=mark.family_name,
                    table_name=mark.table_name,
                    coordinate=mark.coordinate,
                )
            )
    return StateProductiveSupport(
        source_key=source_key,
        source_state_sha256=persistent_slot_state_sha256(source),
        virtual_aliases=tuple(sorted(virtual_aliases)),
    )


def _log1mexp(log_probability: Tensor) -> Tensor:
    """Stable ``log(1 - exp(x))`` for log probabilities ``x <= 0``."""

    if bool((log_probability > 1e-6).any()):
        raise SuccessorTrainingError("virtual probability exceeds one")
    cutoff = -0.6931471805599453
    return torch.where(
        log_probability < cutoff,
        torch.log1p(-log_probability.exp()),
        torch.log(-torch.expm1(log_probability)),
    )


def forward_teacher_successor_batch(
    model: FactorizedTraceletRateModel,
    batch: FactorizedMarkBatch,
    fibers: tuple[TeacherSuccessorFiber | None, ...],
    *,
    state_supports: tuple[StateProductiveSupport | None, ...] | None = None,
) -> FactorizedSuccessorPrediction:
    """Score precompiled teacher-successor fibers on the differentiable action tables."""

    if len(fibers) != batch.batch_size:
        raise ValueError("teacher-successor fibers must align with the batch")
    if state_supports is not None and len(state_supports) != batch.batch_size:
        raise ValueError("state productive supports must align with the batch")
    nonterminal_flags = tuple(
        float(value) > 0.0
        for value in batch.teacher_rates.detach().cpu()
    )
    if batch.atom_types.device != model.device:
        batch = batch.to(model.device)

    node, global_state, pair = model._encode_batch(batch)
    masks, logits, action_log_z = model._action_tables(
        batch,
        node,
        global_state,
        pair,
    )
    enabled = torch.isfinite(action_log_z)
    has_legal_mark = enabled.any(dim=-1)
    family_logits = _masked_family_logits(
        model._family_base_logits(batch, global_state),
        action_log_z,
        enabled,
        rate_factorization=model.rate_factorization,
    )
    family_log_probability = torch.log_softmax(family_logits, dim=-1)

    selected = global_state.new_zeros(batch.batch_size)
    selected_productive = global_state.new_zeros(batch.batch_size)
    productive_log_probability = global_state.new_zeros(batch.batch_size)
    teacher_family_log_probability = global_state.new_zeros(batch.batch_size)
    selected_within_teacher_family_log_probability = global_state.new_zeros(
        batch.batch_size
    )

    def alias_log_probability(
        batch_index: int,
        alias: TeacherSuccessorAlias,
    ) -> tuple[Tensor, Tensor]:
        family_index = MARK_RULE_TO_INDEX.get(alias.family_name)
        if family_index is None:
            raise SuccessorTrainingError(
                f"teacher fiber references unknown family {alias.family_name!r}"
            )
        table_logits = logits.get(alias.table_name)
        table_mask = masks.get(alias.table_name)
        if table_logits is None or table_mask is None:
            raise SuccessorTrainingError(
                f"teacher fiber references unknown table {alias.table_name!r}"
            )
        key = (batch_index, *alias.coordinate)
        try:
            legal = table_mask[key]
            raw_logit = table_logits[key]
        except IndexError as error:
            raise SuccessorTrainingError(
                "teacher fiber coordinate lies outside the current action table"
            ) from error
        return (
            family_log_probability[batch_index, family_index]
            + raw_logit
            - action_log_z[batch_index, family_index],
            legal,
        )

    for batch_index, fiber in enumerate(fibers):
        is_nonterminal = nonterminal_flags[batch_index]
        if not is_nonterminal:
            if fiber is not None:
                raise SuccessorTrainingError(
                    "terminal training row carries a teacher-successor fiber"
                )
            support = (
                None
                if state_supports is None
                else state_supports[batch_index]
            )
            if support is None:
                raise SuccessorTrainingError(
                    "terminal training row is missing productive state support"
                )
            observed_source_digest = persistent_slot_state_sha256(
                batch.states[batch_index]
            )
            if observed_source_digest != support.source_state_sha256:
                raise SuccessorTrainingError(
                    "productive state support does not match the exact terminal batch state"
                )
            virtual_aliases = support.virtual_aliases
        else:
            if fiber is None:
                raise SuccessorTrainingError(
                    "nonterminal training row is missing its teacher-successor fiber"
                )
            observed_source_digest = persistent_slot_state_sha256(
                batch.states[batch_index]
            )
            if observed_source_digest != fiber.state_support.source_state_sha256:
                raise SuccessorTrainingError(
                    "teacher-successor fiber source does not match the exact batch state"
                )
            explicit_support = (
                None
                if state_supports is None
                else state_supports[batch_index]
            )
            if (
                explicit_support is not None
                and explicit_support != fiber.state_support
            ):
                raise SuccessorTrainingError(
                    "explicit productive support disagrees with the teacher fiber"
                )
            virtual_aliases = fiber.state_support.virtual_aliases

        virtual_log_probabilities: list[Tensor] = []
        virtual_legalities: list[Tensor] = []
        for alias in virtual_aliases:
            log_probability, legal = alias_log_probability(batch_index, alias)
            virtual_log_probabilities.append(log_probability)
            virtual_legalities.append(legal)
        if virtual_legalities and not bool(torch.stack(virtual_legalities).all()):
            raise SuccessorTrainingError(
                "precompiled virtual alias is no longer legal in the current batch"
            )
        virtual_log_probability = (
            global_state.new_tensor(float("-inf"))
            if not virtual_log_probabilities
            else torch.logsumexp(torch.stack(virtual_log_probabilities), dim=0)
        )
        productive_log_probability[batch_index] = _log1mexp(
            virtual_log_probability
        )

        if not is_nonterminal:
            continue

        alias_log_probabilities: list[Tensor] = []
        alias_legalities: list[Tensor] = []
        teacher_family_name = _CYCLE_OP_EXECUTOR_TO_FAMILY.get(
            batch.teacher_rule_names[batch_index],
            batch.teacher_rule_names[batch_index],
        )
        if teacher_family_name is None:
            alias_families = {alias.family_name for alias in fiber.aliases}
            if len(alias_families) != 1:
                raise SuccessorTrainingError(
                    "teacher family is absent and cannot be inferred uniquely "
                    "from a cross-family successor fiber"
                )
            teacher_family_name = next(iter(alias_families))
        teacher_family_index = MARK_RULE_TO_INDEX.get(teacher_family_name)
        if teacher_family_index is None:
            raise SuccessorTrainingError(
                f"teacher row references unknown family {teacher_family_name!r}"
            )
        within_teacher_family: list[Tensor] = []
        for alias in fiber.aliases:
            log_probability, legal = alias_log_probability(batch_index, alias)
            alias_legalities.append(legal)
            alias_log_probabilities.append(log_probability)
            if alias.family_name == teacher_family_name:
                within_teacher_family.append(log_probability)
        if not bool(torch.stack(alias_legalities).all()):
            raise SuccessorTrainingError(
                "precompiled teacher alias is no longer legal in the current batch"
            )
        selected[batch_index] = torch.logsumexp(
            torch.stack(alias_log_probabilities),
            dim=0,
        )
        selected_productive[batch_index] = (
            selected[batch_index] - productive_log_probability[batch_index]
        )
        if not within_teacher_family:
            raise SuccessorTrainingError(
                "teacher successor fiber has no alias in the recorded teacher family"
            )
        teacher_family_log_probability[batch_index] = family_log_probability[
            batch_index,
            teacher_family_index,
        ]
        selected_within_teacher_family_log_probability[batch_index] = (
            torch.logsumexp(torch.stack(within_teacher_family), dim=0)
            - teacher_family_log_probability[batch_index]
        )

    total_hazard = F.softplus(
        model.total_hazard_head(global_state).squeeze(-1)
    ) * has_legal_mark.to(global_state.dtype)
    productive_hazard = total_hazard * productive_log_probability.exp()
    return FactorizedSuccessorPrediction(
        total_hazard=total_hazard,
        productive_hazard=productive_hazard,
        selected_successor_log_probability=selected,
        selected_productive_successor_log_probability=selected_productive,
        productive_log_probability=productive_log_probability,
        teacher_family_log_probability=teacher_family_log_probability,
        selected_within_teacher_family_log_probability=(
            selected_within_teacher_family_log_probability
        ),
        family_log_probabilities=family_log_probability,
        enabled_families=enabled,
    )


def factorized_successor_bregman_loss(
    prediction: FactorizedSuccessorPrediction,
    batch: FactorizedMarkBatch,
) -> Tensor:
    """Poisson-KL GM loss on the canonical molecular-successor generator.

    Canonical self-events do not contribute to a state generator.  The positive
    term is therefore the productive hazard, while the selected rate remains
    ``total_hazard * raw_teacher_successor_mass``.
    """

    teacher_rates = batch.teacher_rates.to(prediction.total_hazard.device)
    weights = batch.importance_weights.to(prediction.total_hazard.device)
    log_hazard = torch.log(prediction.total_hazard.clamp_min(1e-12))
    nonterminal = teacher_rates > 0
    teacher_term = teacher_rates * (
        log_hazard + prediction.selected_successor_log_probability
    )
    per_example = prediction.productive_hazard - torch.where(
        nonterminal,
        teacher_term,
        torch.zeros_like(teacher_term),
    )
    return (per_example * weights).mean()


def factorized_successor_identity_loss(
    prediction: FactorizedSuccessorPrediction,
    batch: FactorizedMarkBatch,
) -> Tensor:
    """Editing-only NLL under the productive embedded molecular jump chain."""

    weights = batch.importance_weights.to(prediction.total_hazard.device)
    nonterminal = batch.teacher_rates.to(prediction.total_hazard.device) > 0
    if not bool(nonterminal.any()):
        raise SuccessorTrainingError(
            "successor-identity loss requires at least one molecular jump"
        )
    selected_weights = weights[nonterminal]
    denominator = selected_weights.sum()
    if not bool(denominator > 0):
        raise SuccessorTrainingError(
            "successor-identity loss has zero nonterminal importance weight"
        )
    return -(
        prediction.selected_productive_successor_log_probability[nonterminal]
        * selected_weights
    ).sum() / denominator


__all__ = [
    "FactorizedSuccessorPrediction",
    "StateProductiveSupport",
    "SuccessorTrainingError",
    "TeacherSuccessorAlias",
    "TeacherSuccessorFiber",
    "compile_state_productive_support",
    "compile_teacher_successor_fiber",
    "factorized_successor_bregman_loss",
    "factorized_successor_identity_loss",
    "forward_teacher_successor_batch",
]
