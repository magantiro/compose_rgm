"""Differentiable canonical-successor supervision for the factorized editor.

The production model parameterizes executable mark probabilities.  This module
precompiles the subset of legal mark coordinates that execute to one teacher
molecule and aggregates their normalized probability with ``logsumexp`` during
training.  The executor and canonicalizer are used only during compilation;
the hot gradient path indexes existing action tables and does not replay
chemistry.
"""

from __future__ import annotations

import hashlib
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
    SEMANTIC_EDITING_V2_PROCESS_SEMANTICS,
    FactorizedMarkBatch,
    FactorizedTraceletRateModel,
    _masked_family_logits,
)
from compose_v4.rewrite import action_codec as action_codec_v2
from compose_v4.rewrite import action_codec_v4
from compose_v4.rewrite.kernel import (
    RewriteSystem,
    canonical_state_key,
    de_novo_rewrite_system,
    editing_v2_semantic_rewrite_system,
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
        if len(self.source_state_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.source_state_sha256
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
        if len(self.target_state_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.target_state_sha256
        ):
            raise ValueError("teacher target_state_sha256 must be a lowercase SHA-256")
        if not self.aliases:
            raise ValueError(
                "a teacher-successor fiber must contain at least one alias"
            )
        if len(set(self.aliases)) != len(self.aliases):
            raise ValueError("a teacher-successor fiber contains duplicate coordinates")
        if self.state_support.source_key != self.source_key:
            raise ValueError("teacher fiber and state support have different sources")
        if set(self.aliases) & set(self.state_support.virtual_aliases):
            raise ValueError("a teacher successor cannot also be a virtual self-event")


@dataclass(frozen=True, order=True)
class CompiledSuccessorMark:
    """One enumerated mark tied to its alias and exact executor output."""

    alias: TeacherSuccessorAlias
    successor_state_sha256: str
    action_sha256: str

    def __post_init__(self) -> None:
        for name in ("successor_state_sha256", "action_sha256"):
            digest = getattr(self, name)
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")


@dataclass(frozen=True)
class CanonicalSuccessorAliasGroup:
    """Static enumerated marks for one canonical molecular successor.

    Each entry associates the factorized table coordinate, full canonical
    RewriteActionCodecV2 JSON SHA-256, and exact persistent-slot executor
    output.  The group carries no model score, probability, rate, weight, or
    time value and is discarded after cache-row projection.
    """

    target_key: str
    marks: tuple[CompiledSuccessorMark, ...]

    def __post_init__(self) -> None:
        if not self.target_key:
            raise ValueError("canonical successor alias-group key must be nonempty")
        if not self.marks:
            raise ValueError(
                "canonical successor alias group must retain an enumerated mark"
            )
        if tuple(sorted(set(self.marks))) != self.marks:
            raise ValueError("canonical successor marks must be sorted and unique")
        aliases = tuple(mark.alias for mark in self.marks)
        if len(set(aliases)) != len(aliases):
            raise ValueError(
                "canonical successor group repeats a factorized mark coordinate"
            )

    @property
    def successor_state_sha256s(self) -> tuple[str, ...]:
        return tuple(sorted({mark.successor_state_sha256 for mark in self.marks}))

    @property
    def aliases(self) -> tuple[TeacherSuccessorAlias, ...]:
        return tuple(sorted(mark.alias for mark in self.marks))

    def contains_exact_action(
        self,
        *,
        successor_state_sha256: str,
        action_sha256: str,
    ) -> bool:
        """Whether one enumerated mark has this action and exact output."""

        return any(
            mark.successor_state_sha256 == successor_state_sha256
            and mark.action_sha256 == action_sha256
            for mark in self.marks
        )


@dataclass(frozen=True)
class CompiledStateSuccessorMap:
    """Complete static successor-coordinate grouping for one exact source.

    This is deliberately a transient compilation object.  Persistent cache
    records retain only ``state_support`` and the one requested teacher fiber.
    """

    state_support: StateProductiveSupport
    successor_groups: tuple[CanonicalSuccessorAliasGroup, ...]
    virtual_marks: tuple[CompiledSuccessorMark, ...] = ()

    def __post_init__(self) -> None:
        target_keys = tuple(group.target_key for group in self.successor_groups)
        if target_keys != tuple(sorted(set(target_keys))):
            raise ValueError(
                "compiled successor groups must have sorted unique target keys"
            )
        if self.state_support.source_key in target_keys:
            raise ValueError(
                "compiled productive successor groups contain a canonical self-event"
            )
        productive_marks = [
            mark for group in self.successor_groups for mark in group.marks
        ]
        all_aliases = [mark.alias for mark in productive_marks]
        if len(all_aliases) != len(set(all_aliases)):
            raise ValueError(
                "one legal mark coordinate appears in multiple successor groups"
            )
        if tuple(sorted(set(self.virtual_marks))) != self.virtual_marks:
            raise ValueError("compiled virtual marks must be sorted and unique")
        virtual_aliases = tuple(sorted(mark.alias for mark in self.virtual_marks))
        if len(set(virtual_aliases)) != len(virtual_aliases):
            raise ValueError(
                "compiled virtual marks repeat a factorized mark coordinate"
            )
        if virtual_aliases != tuple(sorted(self.state_support.virtual_aliases)):
            raise ValueError(
                "compiled virtual mark identities disagree with state support"
            )
        if set(all_aliases) & set(virtual_aliases):
            raise ValueError("one legal mark coordinate is both productive and virtual")
        all_exact_digests = [
            digest
            for group in self.successor_groups
            for digest in group.successor_state_sha256s
        ]
        if len(all_exact_digests) != len(set(all_exact_digests)):
            raise ValueError(
                "one exact successor state appears in multiple canonical groups"
            )

    @property
    def source_key(self) -> str:
        return self.state_support.source_key

    @property
    def source_state_sha256(self) -> str:
        return self.state_support.source_state_sha256


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


def rewrite_action_codec_sha256(
    executor_rule_name: str,
    action: object,
    *,
    schema_version: int = action_codec_v2.SCHEMA_VERSION,
) -> str:
    """Full SHA-256 of one explicitly selected canonical action payload."""

    if schema_version == action_codec_v2.SCHEMA_VERSION:
        codec = action_codec_v2
        error_type = action_codec_v2.ActionCodecError
    elif schema_version == action_codec_v4.SCHEMA_VERSION:
        codec = action_codec_v4
        error_type = action_codec_v4.ActionCodecV4Error
    else:
        raise SuccessorTrainingError(
            f"unsupported rewrite action codec schema_version={schema_version}"
        )
    try:
        encoded = codec.encode_action(executor_rule_name, action)
    except error_type as error:
        raise SuccessorTrainingError(
            "rewrite action cannot be represented by the selected action codec"
        ) from error
    return hashlib.sha256(codec.canonical_json(encoded).encode("utf-8")).hexdigest()


def _uses_semantic_editing_process(model: FactorizedTraceletRateModel) -> bool:
    return model.editing_process_semantics == SEMANTIC_EDITING_V2_PROCESS_SEMANTICS


def compile_state_successor_map(
    model: FactorizedTraceletRateModel,
    source: MolecularGraph,
    *,
    time: float = 0.5,
    system: RewriteSystem | None = None,
) -> CompiledStateSuccessorMap:
    """Execute and group every legal mark once for one exact source state."""

    semantic_editing_process = _uses_semantic_editing_process(model)
    runtime = system or (
        editing_v2_semantic_rewrite_system()
        if semantic_editing_process
        else de_novo_rewrite_system()
    )
    action_codec_version = (
        action_codec_v4.SCHEMA_VERSION
        if semantic_editing_process
        else action_codec_v2.SCHEMA_VERSION
    )
    try:
        marked_law = enumerate_factorized_marked_law(model, source, float(time))
    except ProductionSuccessorKernelError as error:
        raise SuccessorTrainingError(
            "could not enumerate the production marked law for state-successor compilation"
        ) from error

    source_key = marked_law.source_key
    grouped_marks: dict[str, list[CompiledSuccessorMark]] = {}
    virtual_marks: list[CompiledSuccessorMark] = []
    for mark in marked_law.marks:
        try:
            successor = runtime.apply(
                source,
                mark.executor_rule_name,
                mark.action,
            )
        except Exception as error:
            raise SuccessorTrainingError(
                "a production-enumerated mark failed during state-successor compilation"
            ) from error
        successor_key = canonical_state_key(successor)
        compiled_alias = _alias(
            family_name=mark.family_name,
            table_name=mark.table_name,
            coordinate=mark.coordinate,
        )
        compiled_mark = CompiledSuccessorMark(
            alias=compiled_alias,
            successor_state_sha256=persistent_slot_state_sha256(successor),
            action_sha256=rewrite_action_codec_sha256(
                mark.executor_rule_name,
                mark.action,
                schema_version=action_codec_version,
            ),
        )
        if successor_key == source_key:
            virtual_marks.append(compiled_mark)
            continue
        grouped_marks.setdefault(successor_key, []).append(compiled_mark)

    state_support = StateProductiveSupport(
        source_key=source_key,
        source_state_sha256=persistent_slot_state_sha256(source),
        virtual_aliases=tuple(sorted(mark.alias for mark in virtual_marks)),
    )
    return CompiledStateSuccessorMap(
        state_support=state_support,
        successor_groups=tuple(
            CanonicalSuccessorAliasGroup(
                target_key=target_key,
                marks=tuple(sorted(marks)),
            )
            for target_key, marks in sorted(grouped_marks.items())
        ),
        virtual_marks=tuple(sorted(virtual_marks)),
    )


def teacher_successor_fiber_from_compiled_state(
    compiled: CompiledStateSuccessorMap,
    target: MolecularGraph,
) -> TeacherSuccessorFiber:
    """Derive the public teacher fiber for ``target`` without source replay."""

    target_key = canonical_state_key(target)
    if compiled.source_key == target_key:
        raise SuccessorTrainingError(
            "teacher successor equals the source; virtual events are not molecular jumps"
        )
    group = next(
        (
            candidate
            for candidate in compiled.successor_groups
            if candidate.target_key == target_key
        ),
        None,
    )
    if group is None:
        raise SuccessorTrainingError(
            "teacher successor is absent from the production marked support"
        )
    return TeacherSuccessorFiber(
        source_key=compiled.source_key,
        target_key=target_key,
        target_state_sha256=persistent_slot_state_sha256(target),
        aliases=group.aliases,
        state_support=compiled.state_support,
    )


def _exact_successor_group(
    compiled: CompiledStateSuccessorMap,
    target_state_sha256: str,
) -> CanonicalSuccessorAliasGroup:
    matches = tuple(
        group
        for group in compiled.successor_groups
        if target_state_sha256 in group.successor_state_sha256s
    )
    if len(matches) != 1:
        raise SuccessorTrainingError(
            "exact teacher successor is absent from the production marked support"
            if not matches
            else "exact teacher successor belongs to multiple canonical groups"
        )
    return matches[0]


def teacher_successor_fiber_from_exact_digest(
    compiled: CompiledStateSuccessorMap,
    target_state_sha256: str,
) -> TeacherSuccessorFiber:
    """Derive a fiber only when an exact executor output matches the target.

    This stricter builder path catches a stored trace transition that is merely
    canonically equivalent to, but not the exact persistent-slot result of, a
    legal production mark.
    """

    group = _exact_successor_group(compiled, target_state_sha256)
    return TeacherSuccessorFiber(
        source_key=compiled.source_key,
        target_key=group.target_key,
        target_state_sha256=target_state_sha256,
        aliases=group.aliases,
        state_support=compiled.state_support,
    )


def require_exact_successor_action_identity(
    compiled: CompiledStateSuccessorMap,
    *,
    target_state_sha256: str,
    action_sha256: str,
) -> None:
    """Require one enumerated action identity paired with this exact output."""

    group = _exact_successor_group(compiled, target_state_sha256)
    if not group.contains_exact_action(
        successor_state_sha256=target_state_sha256,
        action_sha256=action_sha256,
    ):
        raise SuccessorTrainingError(
            "teacher action identity is absent from the exact successor marked support"
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

    return teacher_successor_fiber_from_compiled_state(
        compile_state_successor_map(
            model,
            source,
            time=time,
            system=system,
        ),
        target,
    )


def compile_state_productive_support(
    model: FactorizedTraceletRateModel,
    source: MolecularGraph,
    *,
    time: float = 0.5,
    system: RewriteSystem | None = None,
) -> StateProductiveSupport:
    """Compile canonical self-event coordinates for a terminal/no-teacher row."""

    return compile_state_successor_map(
        model,
        source,
        time=time,
        system=system,
    ).state_support


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
        float(value) > 0.0 for value in batch.teacher_rates.detach().cpu()
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
            support = None if state_supports is None else state_supports[batch_index]
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
                None if state_supports is None else state_supports[batch_index]
            )
            if explicit_support is not None and explicit_support != fiber.state_support:
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
        productive_log_probability[batch_index] = _log1mexp(virtual_log_probability)

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
    return (
        -(
            prediction.selected_productive_successor_log_probability[nonterminal]
            * selected_weights
        ).sum()
        / denominator
    )


def factorized_hazard_bregman_loss(
    prediction: FactorizedSuccessorPrediction,
    batch: FactorizedMarkBatch,
) -> Tensor:
    """Poisson-KL total-hazard loss kept separate from successor identity."""

    teacher_rates = batch.teacher_rates.to(prediction.total_hazard.device)
    weights = batch.importance_weights.to(prediction.total_hazard.device)
    log_hazard = torch.log(prediction.total_hazard.clamp_min(1e-12))
    teacher_term = torch.where(
        teacher_rates > 0,
        teacher_rates * log_hazard,
        torch.zeros_like(teacher_rates),
    )
    return ((prediction.total_hazard - teacher_term) * weights).mean()


__all__ = [
    "CanonicalSuccessorAliasGroup",
    "CompiledStateSuccessorMap",
    "CompiledSuccessorMark",
    "FactorizedSuccessorPrediction",
    "StateProductiveSupport",
    "SuccessorTrainingError",
    "TeacherSuccessorAlias",
    "TeacherSuccessorFiber",
    "compile_state_productive_support",
    "compile_state_successor_map",
    "compile_teacher_successor_fiber",
    "factorized_hazard_bregman_loss",
    "factorized_successor_bregman_loss",
    "factorized_successor_identity_loss",
    "forward_teacher_successor_batch",
    "require_exact_successor_action_identity",
    "rewrite_action_codec_sha256",
    "teacher_successor_fiber_from_compiled_state",
    "teacher_successor_fiber_from_exact_digest",
]
