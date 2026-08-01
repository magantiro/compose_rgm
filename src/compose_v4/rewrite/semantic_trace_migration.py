"""Offline whole-trace migration from historical actions to semantic Active8."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from compose_v4.chem.molecular_graph import (
    BOND_CLASS_TO_H_CHANGE,
    ORGANIC_VOCABULARY,
    MolecularGraph,
)
from compose_v4.rewrite.kernel import (
    RewriteSystem,
    canonical_state_key,
    de_novo_rewrite_system,
    editing_v2_semantic_rewrite_system,
)
from compose_v4.rewrite.operators import (
    AtomRestate,
    BondDelete,
    BondInsert,
    CycleCloseEdge,
    CycleOpenEdge,
    SemanticAtomRestate,
)
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace
from compose_v4.rewrite.trace_shard_v3 import encode_semantic_trace_record


class SemanticTraceMigrationRejectionCode(str, Enum):
    LEGACY_REPLAY_FAILED = "legacy_replay_failed"
    DISABLED_OR_UNKNOWN_RULE = "disabled_or_unknown_rule"
    TARGET_ATOM_CLASS_UNREPRESENTABLE = "target_atom_class_unrepresentable"
    SEMANTIC_ACTION_REJECTED = "semantic_action_rejected"
    CANONICAL_SUCCESSOR_MISMATCH = "canonical_successor_mismatch"
    FINAL_TARGET_MISMATCH = "final_target_mismatch"


@dataclass(frozen=True)
class SemanticTraceMigrationRejection:
    code: SemanticTraceMigrationRejectionCode
    step_index: int | None
    legacy_rule_name: str | None
    detail: str


@dataclass(frozen=True)
class SemanticTraceMigrationResult:
    admitted: bool
    trace: RewriteTrace | None
    rejection: SemanticTraceMigrationRejection | None


def _reject(
    code: SemanticTraceMigrationRejectionCode,
    *,
    step_index: int | None,
    legacy_rule_name: str | None,
    detail: str,
) -> SemanticTraceMigrationResult:
    return SemanticTraceMigrationResult(
        admitted=False,
        trace=None,
        rejection=SemanticTraceMigrationRejection(
            code=code,
            step_index=step_index,
            legacy_rule_name=legacy_rule_name,
            detail=detail,
        ),
    )


def _target_class_index(state: MolecularGraph, vertex: int) -> int | None:
    bond_order_sum = sum(
        int(BOND_CLASS_TO_H_CHANGE[int(order)]) for order in state.bonds[vertex]
    )
    return ORGANIC_VOCABULARY.class_index(
        int(state.atom_types[vertex]),
        bond_order_sum,
        int(state.implicit_h_counts[vertex]),
        int(state.formal_charges[vertex]),
    )


def _translate_step(
    step: RewriteStep,
    *,
    legacy_successor: MolecularGraph,
) -> RewriteStep | None:
    action = step.action
    if step.rule_name == "atom_restate" and type(action) is AtomRestate:
        vertex = int(action.v)
        target_class_index = _target_class_index(legacy_successor, vertex)
        if target_class_index is None:
            return None
        return RewriteStep(
            "atom_restate_semantic",
            SemanticAtomRestate(vertex, target_class_index),
        )
    if step.rule_name == "bond_insert" and type(action) is BondInsert:
        return RewriteStep(
            "cycle_close",
            CycleCloseEdge(
                min(int(action.a), int(action.b)),
                max(int(action.a), int(action.b)),
                int(action.order),
            ),
        )
    if step.rule_name == "bond_delete" and type(action) is BondDelete:
        return RewriteStep(
            "cycle_open",
            CycleOpenEdge(
                min(int(action.a), int(action.b)),
                max(int(action.a), int(action.b)),
            ),
        )
    if step.rule_name in {
        "atom_insert",
        "atom_delete",
        "bond_reorder",
        "bond_reroute",
        "ring_system_restate",
        "atom_restate_semantic",
        "cycle_close",
        "cycle_open",
    }:
        return step
    return None


def migrate_legacy_trace_to_editing_v2(
    trace: RewriteTrace,
    *,
    legacy_system: RewriteSystem | None = None,
) -> SemanticTraceMigrationResult:
    """Translate and replay a complete trace, rejecting it atomically on failure."""

    old_runtime = legacy_system or de_novo_rewrite_system()
    semantic_runtime = editing_v2_semantic_rewrite_system()
    legacy_state = trace.source
    semantic_state = trace.source
    migrated_steps: list[RewriteStep] = []

    for index, step in enumerate(trace.steps):
        try:
            legacy_successor = old_runtime.apply(
                legacy_state,
                step.rule_name,
                step.action,
            )
        except ValueError as error:
            return _reject(
                SemanticTraceMigrationRejectionCode.LEGACY_REPLAY_FAILED,
                step_index=index,
                legacy_rule_name=step.rule_name,
                detail=str(error),
            )
        translated = _translate_step(step, legacy_successor=legacy_successor)
        if translated is None:
            code = (
                SemanticTraceMigrationRejectionCode.TARGET_ATOM_CLASS_UNREPRESENTABLE
                if step.rule_name == "atom_restate"
                else SemanticTraceMigrationRejectionCode.DISABLED_OR_UNKNOWN_RULE
            )
            return _reject(
                code,
                step_index=index,
                legacy_rule_name=step.rule_name,
                detail="historical action has no declared semantic Active8 translation",
            )
        try:
            semantic_successor = semantic_runtime.apply(
                semantic_state,
                translated.rule_name,
                translated.action,
            )
        except ValueError as error:
            return _reject(
                SemanticTraceMigrationRejectionCode.SEMANTIC_ACTION_REJECTED,
                step_index=index,
                legacy_rule_name=step.rule_name,
                detail=str(error),
            )
        if canonical_state_key(semantic_successor) != canonical_state_key(
            legacy_successor
        ):
            return _reject(
                SemanticTraceMigrationRejectionCode.CANONICAL_SUCCESSOR_MISMATCH,
                step_index=index,
                legacy_rule_name=step.rule_name,
                detail=(
                    f"semantic={canonical_state_key(semantic_successor)!r}, "
                    f"legacy={canonical_state_key(legacy_successor)!r}"
                ),
            )
        migrated_steps.append(translated)
        legacy_state = legacy_successor
        semantic_state = semantic_successor

    if canonical_state_key(legacy_state) != canonical_state_key(trace.target):
        return _reject(
            SemanticTraceMigrationRejectionCode.FINAL_TARGET_MISMATCH,
            step_index=None,
            legacy_rule_name=None,
            detail="historical replay does not reach the recorded trace target",
        )
    if canonical_state_key(semantic_state) != canonical_state_key(trace.target):
        return _reject(
            SemanticTraceMigrationRejectionCode.FINAL_TARGET_MISMATCH,
            step_index=None,
            legacy_rule_name=None,
            detail="semantic replay does not preserve the recorded molecular target",
        )
    migrated = RewriteTrace(
        source=trace.source,
        target=semantic_state,
        steps=tuple(migrated_steps),
        metadata={
            **dict(trace.metadata),
            "semantic_migration": "historical_to_semantic_editing_v2_v1",
        },
    )
    return SemanticTraceMigrationResult(
        admitted=True,
        trace=migrated,
        rejection=None,
    )


def migrate_and_encode_semantic_trace(
    trace: RewriteTrace,
    *,
    trace_id: str,
    data_lane: str,
    split: str,
    source_address: Mapping[str, object],
    lineage: Mapping[str, object],
    legacy_system: RewriteSystem | None = None,
) -> tuple[dict[str, object] | None, SemanticTraceMigrationRejection | None]:
    """Return one trace-v3 record or one stable whole-trace rejection."""

    result = migrate_legacy_trace_to_editing_v2(
        trace,
        legacy_system=legacy_system,
    )
    if not result.admitted or result.trace is None:
        return None, result.rejection
    return (
        encode_semantic_trace_record(
            result.trace,
            trace_id=trace_id,
            data_lane=data_lane,
            split=split,
            source_address=source_address,
            lineage=lineage,
        ),
        None,
    )


__all__ = [
    "SemanticTraceMigrationRejection",
    "SemanticTraceMigrationRejectionCode",
    "SemanticTraceMigrationResult",
    "migrate_and_encode_semantic_trace",
    "migrate_legacy_trace_to_editing_v2",
]
