"""CPU-prepared conditional ring tables, bound to state, context and teacher.

Only the opt-in legacy Boolean-family scaffold lane is qualified here. These
tables contain chemistry, not learned logits, and can be reused across weights.
The existing prepared-batch serializer owns integrity checks on the payload.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any

from compose_v4.model.node_context import node_context_key
from compose_v4.rewrite.tracelets import RingSystemGrow


@dataclass(frozen=True)
class PreparedScaffoldTemplate:
    template_index: int
    placement_groups: tuple
    decoders: tuple


@dataclass(frozen=True)
class PreparedScaffoldRow:
    state_context_key: str
    model_support_key: str
    teacher_key: str
    ring_support: tuple[bool, ...]
    teacher_certificate: Any
    templates: tuple[PreparedScaffoldTemplate, ...]


def scaffold_support_key(model) -> str:
    """Configuration/catalog identity, deliberately independent of fitted weights."""
    value = (
        "scaffold_prepared_v2",
        model.ring_system_templates,
        model.ring_system_template_aliases,
        # AtomVocabulary uses object repr, which contains a process-local
        # memory address. Bind its ordered chemical classes, not its identity.
        model.atom_vocabulary.classes,
        model.operator_capabilities,
        model.ring_electronic_mode,
        model.ring_family_mass_mode,
        model.enable_cycle_ops,
    )
    return hashlib.sha256(repr(value).encode()).hexdigest()


def _teacher_key(action, family):
    return hashlib.sha256(repr((family, action)).encode()).hexdigest()


def _require_lane(model):
    from compose_v4.model.factorized_tracelet_rate_model import LEGACY_EDITING_PROCESS_SEMANTICS

    if (
        not model.scaffold_conditioning
        or model.enable_cycle_ops
        or model.editing_process_semantics != LEGACY_EDITING_PROCESS_SEMANTICS
        or model.ring_family_mass_mode != "boolean"
        or model.ring_electronic_mode not in {"factorized_local", "factorized_contextual"}
    ):
        raise ValueError(
            "prepared scaffold support requires the qualified legacy factorized lipid lane"
        )


def prepare_scaffold_mark_batch(model, batch):
    """Compile once on CPUs; never call from a GPU training step.

    The row binds the teacher too: an exact table for a different teacher is not
    silently accepted. Source-only legacy ring cache certificates are rejected.
    Partial/full ring support flags on legacy batches are not used as authority.
    """
    _require_lane(model)
    if model.device.type != "cpu" or batch.atom_types.device.type != "cpu":
        raise ValueError("scaffold support compilation is CPU-only")
    if batch.scaffold_prepared_rows is not None:
        raise ValueError("scaffold batch is already prepared")
    # Validate chemistry metadata without an unnecessary neural forward on CPU.
    model._validate_scaffold_batch(batch)
    support_key = scaffold_support_key(model)
    rows = []
    for state, context, action, family in zip(
        batch.states,
        batch.scaffold_contexts,
        batch.teacher_actions,
        batch.teacher_rule_names,
        strict=True,
    ):
        support = tuple(bool(v) for v in model._ring_grow_support(state, context))
        certificate, templates = None, []
        if isinstance(action, RingSystemGrow):
            certificate = model.ring_teacher_semantic_certificate(
                state, action, scaffold_context=context
            )
            for item in certificate.templates:
                groups = model._ring_template_placement_groups(state, item.template_index)
                templates.append(
                    PreparedScaffoldTemplate(
                        item.template_index,
                        groups,
                        tuple(
                            model._ring_semantic_decoder(state, group[0], context)
                            for group in groups
                        ),
                    )
                )
        rows.append(
            PreparedScaffoldRow(
                node_context_key(state, context),
                support_key,
                _teacher_key(action, family),
                support,
                certificate,
                tuple(templates),
            )
        )
    return replace(batch, scaffold_prepared_rows=tuple(rows))


def validate_prepared_scaffolds(model, batch):
    """Cheap CPU-metadata checks; no executor, enumeration or semantic DP."""
    _require_lane(model)
    rows = batch.scaffold_prepared_rows
    if rows is None or len(rows) != batch.batch_size or batch.scaffold_contexts is None:
        raise ValueError("prepared scaffold rows or conditions are missing")
    support_key = scaffold_support_key(model)
    for row, state, context, action, family in zip(
        rows,
        batch.states,
        batch.scaffold_contexts,
        batch.teacher_actions,
        batch.teacher_rule_names,
        strict=True,
    ):
        if row.state_context_key != node_context_key(state, context):
            raise ValueError("prepared scaffold state/context binding changed")
        if row.model_support_key != support_key:
            raise ValueError("prepared scaffold catalog or support configuration changed")
        if row.teacher_key != _teacher_key(action, family):
            raise ValueError("prepared scaffold teacher binding changed")
        if len(row.ring_support) != len(model.ring_system_templates):
            raise ValueError("prepared scaffold ring support has the wrong width")
        if isinstance(action, RingSystemGrow) != (row.teacher_certificate is not None):
            raise ValueError("prepared scaffold teacher certificate is missing or unexpected")
