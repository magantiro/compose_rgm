"""Exact initialization and teacher-support checks before editing optimization.

The editing-v2 pilot must not spend an optimizer update until two independent
objects agree with their declared construction:

1. a warm-started state dict exactly matches its tensor/semantic-row transfer
   plan while every genuinely new tensor or row remains at fresh initialization;
2. every successor-training teacher is an executable production mark whose
   exact persistent-slot output, canonical successor fiber, and live action-table
   coordinates agree with the frozen cache row.

These checks intentionally contain no empirical performance tolerance. They
establish identity, legality, coverage, finiteness, and normalization only.
Capability-retention thresholds belong to the frozen development-probe contract,
not this structural gate.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from compose_v4.chem.persistent_state_identity import (
    persistent_slot_state_sha256,
)
from compose_v4.experiments.factorized_mark_conditional import (
    assert_teachers_in_exact_candidates,
)
from compose_v4.experiments.factorized_successor_data import (
    FactorizedSuccessorBatch,
)
from compose_v4.experiments.factorized_successor_training import (
    CompiledStateSuccessorMap,
    SuccessorTrainingError,
    compile_state_successor_map,
    forward_teacher_successor_batch,
    require_exact_successor_action_identity,
    rewrite_action_codec_sha256,
    teacher_successor_fiber_from_exact_digest,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    MARK_RULE_NAMES,
    _CYCLE_OP_EXECUTOR_TO_FAMILY,
    FactorizedTraceletRateModel,
)
from compose_v4.rewrite.kernel import (
    RewriteSystem,
    canonical_state_key,
    de_novo_rewrite_system,
)

# Categorical output tensors whose leading rows are indexed by the atom
# (element, valence) vocabulary. This registry is shared with the production
# semantic initializer; missing a widened inherited head must fail closed.
ATOM_VOCABULARY_HEAD_LAYOUT: Mapping[str, tuple[str, int]] = {
    "restate_head.weight": ("row", 1),
    "restate_head.bias": ("row", 1),
    "grow_root_head.weight": ("row", 1),
    "grow_root_head.bias": ("row", 1),
    "grow_option.weight": ("role_major", 3),
}


class EditingStepZeroGateError(RuntimeError):
    """Initialization or successor support is not safe for optimization."""


def semantic_row_map(
    source_classes: tuple[Any, ...],
    destination_classes: tuple[Any, ...],
) -> tuple[int | None, ...]:
    """Map destination vocabulary rows to source rows by semantic label."""

    if not source_classes or not destination_classes:
        raise EditingStepZeroGateError("semantic transfer vocabularies must be nonempty")
    if len(set(source_classes)) != len(source_classes):
        raise EditingStepZeroGateError("source vocabulary has duplicate class labels")
    if len(set(destination_classes)) != len(destination_classes):
        raise EditingStepZeroGateError("destination vocabulary has duplicate class labels")
    destination_index = {label: index for index, label in enumerate(destination_classes)}
    missing = tuple(label for label in source_classes if label not in destination_index)
    if missing:
        raise EditingStepZeroGateError(
            f"source vocabulary classes absent from destination: {missing!r}"
        )
    source_index = {label: index for index, label in enumerate(source_classes)}
    return tuple(source_index.get(label) for label in destination_classes)


def _state_schema(
    state: Mapping[str, torch.Tensor],
    *,
    name: str,
) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    if not isinstance(state, Mapping) or not state:
        raise EditingStepZeroGateError(f"{name} state dict must be nonempty")
    invalid = tuple(
        key
        for key, tensor in state.items()
        if not isinstance(key, str) or not key or not isinstance(tensor, torch.Tensor)
    )
    if invalid:
        raise EditingStepZeroGateError(
            f"{name} state dict contains malformed entries: {invalid[:10]!r}"
        )
    return tuple(
        sorted(
            (
                key,
                tuple(int(value) for value in tensor.shape),
                str(tensor.dtype),
            )
            for key, tensor in state.items()
        )
    )


@dataclass(frozen=True, order=True)
class TensorRowAssignment:
    """One destination row copied from a source row or kept fresh."""

    tensor_name: str
    destination_row: int
    source_row: int | None

    def __post_init__(self) -> None:
        if not self.tensor_name:
            raise ValueError("row assignment tensor name must be nonempty")
        if self.destination_row < 0:
            raise ValueError("destination row must be nonnegative")
        if self.source_row is not None and self.source_row < 0:
            raise ValueError("source row must be null or nonnegative")


@dataclass(frozen=True)
class InitializationTransferPlan:
    """Complete ownership of every destination tensor or leading row."""

    regime: str
    source_schema: tuple[tuple[str, tuple[int, ...], str], ...] | None
    destination_schema: tuple[tuple[str, tuple[int, ...], str], ...]
    exact_tensor_names: tuple[str, ...]
    row_assignments: tuple[TensorRowAssignment, ...]
    fresh_tensor_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.regime not in {"scratch", "compatible_warm_start"}:
            raise ValueError(f"unknown initialization regime {self.regime!r}")
        if not self.destination_schema:
            raise ValueError("initialization plan has no destination schema")
        if self.regime == "scratch":
            if self.source_schema is not None:
                raise ValueError("scratch initialization cannot claim a source")
            if self.exact_tensor_names or self.row_assignments:
                raise ValueError("scratch initialization cannot claim transfers")
        elif self.source_schema is None:
            raise ValueError("warm-start initialization lacks a source schema")
        exact = set(self.exact_tensor_names)
        fresh = set(self.fresh_tensor_names)
        row_tensors = {assignment.tensor_name for assignment in self.row_assignments}
        if (
            len(exact) != len(self.exact_tensor_names)
            or len(fresh) != len(self.fresh_tensor_names)
            or exact & fresh
            or exact & row_tensors
            or fresh & row_tensors
        ):
            raise ValueError("initialization plan tensor ownership overlaps or repeats")
        row_keys = tuple(
            (assignment.tensor_name, assignment.destination_row)
            for assignment in self.row_assignments
        )
        if len(row_keys) != len(set(row_keys)):
            raise ValueError("initialization plan repeats a destination row")
        destination_shapes = {name: shape for name, shape, _dtype in self.destination_schema}
        source_shapes = (
            {}
            if self.source_schema is None
            else {name: shape for name, shape, _dtype in self.source_schema}
        )
        owned = exact | fresh | row_tensors
        if owned != set(destination_shapes):
            raise ValueError(
                "initialization plan does not own every destination tensor exactly once"
            )
        source_only = set(source_shapes) - set(destination_shapes)
        if source_only:
            raise ValueError(
                f"initialization plan silently drops source tensors: {sorted(source_only)!r}"
            )
        for name in exact:
            if name not in source_shapes or source_shapes[name] != destination_shapes[name]:
                raise ValueError(f"exact tensor transfer {name!r} lacks a same-shaped source")
        for name in row_tensors:
            if name not in source_shapes:
                raise ValueError(f"row-mapped tensor {name!r} is absent from the source")
            source_shape = source_shapes[name]
            destination_shape = destination_shapes[name]
            if (
                not source_shape
                or not destination_shape
                or source_shape[1:] != destination_shape[1:]
            ):
                raise ValueError(f"row-mapped tensor {name!r} has incompatible schemas")
            assignments = tuple(
                assignment for assignment in self.row_assignments if assignment.tensor_name == name
            )
            if tuple(sorted(assignment.destination_row for assignment in assignments)) != tuple(
                range(destination_shape[0])
            ):
                raise ValueError(
                    f"row-mapped tensor {name!r} does not own every destination row exactly once"
                )
            mapped_source_rows = tuple(
                sorted(
                    assignment.source_row
                    for assignment in assignments
                    if assignment.source_row is not None
                )
            )
            if mapped_source_rows != tuple(range(source_shape[0])):
                raise ValueError(
                    f"row-mapped tensor {name!r} does not inherit every source row exactly once"
                )

    @property
    def sha256(self) -> str:
        payload = {
            "regime": self.regime,
            "source_schema": self.source_schema,
            "destination_schema": self.destination_schema,
            "exact_tensor_names": self.exact_tensor_names,
            "row_assignments": [
                {
                    "tensor_name": assignment.tensor_name,
                    "destination_row": assignment.destination_row,
                    "source_row": assignment.source_row,
                }
                for assignment in self.row_assignments
            ],
            "fresh_tensor_names": self.fresh_tensor_names,
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class InitializationParityReport:
    """Bit-exact result of applying one complete initialization plan."""

    regime: str
    transfer_plan_sha256: str
    exact_tensor_count: int
    copied_row_count: int
    fresh_row_count: int
    fresh_tensor_count: int
    exact_tensor_names: tuple[str, ...]
    fresh_tensor_names: tuple[str, ...]


def build_scratch_initialization_plan(
    fresh_target_state: Mapping[str, torch.Tensor],
) -> InitializationTransferPlan:
    """Declare every target tensor fresh and make no inheritance claim."""

    schema = _state_schema(fresh_target_state, name="fresh target")
    return InitializationTransferPlan(
        regime="scratch",
        source_schema=None,
        destination_schema=schema,
        exact_tensor_names=(),
        row_assignments=(),
        fresh_tensor_names=tuple(name for name, _shape, _dtype in schema),
    )


def build_compatible_initialization_plan(
    source_state: Mapping[str, torch.Tensor],
    fresh_target_state: Mapping[str, torch.Tensor],
    *,
    source_classes: tuple[Any, ...],
    destination_classes: tuple[Any, ...],
    explicitly_fresh_tensors: tuple[str, ...] = (),
    semantic_layouts: Mapping[str, tuple[str, int]] = ATOM_VOCABULARY_HEAD_LAYOUT,
) -> InitializationTransferPlan:
    """Classify every target tensor as exact, semantic-row, or explicit fresh.

    A shape-mismatched or destination-only tensor is never silently retained.
    The caller must name every intentionally fresh whole tensor. Widened
    vocabulary heads use the shared semantic layout even when their shapes
    happen to match, and must map every source row exactly once; new
    destination classes remain fresh.
    """

    source_schema = _state_schema(source_state, name="source")
    target_schema = _state_schema(fresh_target_state, name="fresh target")
    source_names = set(source_state)
    target_names = set(fresh_target_state)
    source_only = tuple(sorted(source_names - target_names))
    if source_only:
        raise EditingStepZeroGateError(
            f"source checkpoint contains parameters absent from the target: {source_only!r}"
        )
    explicit_fresh = tuple(explicitly_fresh_tensors)
    if len(explicit_fresh) != len(set(explicit_fresh)):
        raise EditingStepZeroGateError("explicit fresh-tensor declarations contain duplicates")
    unknown_fresh = set(explicit_fresh) - target_names
    if unknown_fresh:
        raise EditingStepZeroGateError(
            "explicit fresh-tensor declarations are absent from the target: "
            f"{sorted(unknown_fresh)!r}"
        )
    row_map = semantic_row_map(source_classes, destination_classes)
    exact: list[str] = []
    row_assignments: list[TensorRowAssignment] = []
    fresh_whole: list[str] = []
    used_explicit_fresh: set[str] = set()

    for name, destination in fresh_target_state.items():
        source = source_state.get(name)
        layout = semantic_layouts.get(name)
        if source is not None and layout is not None:
            if name in explicit_fresh:
                raise EditingStepZeroGateError(
                    f"semantic inherited tensor {name!r} cannot be silently reclassified as fresh"
                )
            layout_name, roles = layout
            if (
                layout_name not in {"row", "role_major"}
                or type(roles) is not int
                or roles <= 0
                or (layout_name == "row" and roles != 1)
            ):
                raise EditingStepZeroGateError(f"semantic layout for {name!r} is malformed")
            expected_source_rows = roles * len(source_classes)
            expected_destination_rows = roles * len(destination_classes)
            if (
                source.ndim == 0
                or destination.ndim == 0
                or source.shape[0] != expected_source_rows
                or destination.shape[0] != expected_destination_rows
                or source.shape[1:] != destination.shape[1:]
            ):
                raise EditingStepZeroGateError(
                    f"semantic tensor {name!r} has shapes incompatible with "
                    "its declared vocabulary layout"
                )
            mapped_source_rows: list[int] = []
            for role in range(roles):
                for destination_index, source_index in enumerate(row_map):
                    destination_row = role * len(destination_classes) + destination_index
                    source_row = (
                        None if source_index is None else role * len(source_classes) + source_index
                    )
                    row_assignments.append(
                        TensorRowAssignment(
                            tensor_name=name,
                            destination_row=destination_row,
                            source_row=source_row,
                        )
                    )
                    if source_row is not None:
                        mapped_source_rows.append(source_row)
            if tuple(sorted(mapped_source_rows)) != tuple(range(expected_source_rows)):
                raise EditingStepZeroGateError(
                    f"semantic tensor {name!r} does not map every inherited source row exactly once"
                )
            continue

        if source is not None and source.shape == destination.shape:
            if name in explicit_fresh:
                raise EditingStepZeroGateError(
                    f"shape-compatible inherited tensor {name!r} cannot be "
                    "silently reclassified as fresh"
                )
            exact.append(name)
            continue

        if name not in explicit_fresh:
            reason = (
                "is absent from the source"
                if source is None
                else (
                    f"has source shape {tuple(source.shape)} and target shape "
                    f"{tuple(destination.shape)} without a semantic mapping"
                )
            )
            raise EditingStepZeroGateError(
                f"target tensor {name!r} {reason}; declare it explicitly fresh "
                "or provide its inherited mapping"
            )
        used_explicit_fresh.add(name)
        fresh_whole.append(name)

    unused_fresh = set(explicit_fresh) - used_explicit_fresh
    if unused_fresh:
        raise EditingStepZeroGateError(
            "fresh-tensor declarations were not needed by the source/target "
            f"schemas: {sorted(unused_fresh)!r}"
        )
    return InitializationTransferPlan(
        regime="compatible_warm_start",
        source_schema=source_schema,
        destination_schema=target_schema,
        exact_tensor_names=tuple(sorted(exact)),
        row_assignments=tuple(sorted(row_assignments)),
        fresh_tensor_names=tuple(sorted(fresh_whole)),
    )


def assert_initialization_parity(
    plan: InitializationTransferPlan,
    *,
    fresh_target_state: Mapping[str, torch.Tensor],
    initialized_state: Mapping[str, torch.Tensor],
    source_state: Mapping[str, torch.Tensor] | None = None,
) -> InitializationParityReport:
    """Require the initialized state to equal the complete plan bit-for-bit."""

    if _state_schema(fresh_target_state, name="fresh target") != plan.destination_schema:
        raise EditingStepZeroGateError(
            "fresh target state no longer matches the transfer-plan schema"
        )
    if _state_schema(initialized_state, name="initialized target") != plan.destination_schema:
        raise EditingStepZeroGateError(
            "initialized state no longer matches the transfer-plan schema"
        )
    if plan.regime == "scratch":
        if source_state is not None:
            raise EditingStepZeroGateError(
                "scratch parity cannot receive or claim an inherited source"
            )
    else:
        if source_state is None:
            raise EditingStepZeroGateError("warm-start parity requires the inherited source state")
        if _state_schema(source_state, name="source") != plan.source_schema:
            raise EditingStepZeroGateError(
                "source state no longer matches the transfer-plan schema"
            )

    expected = {name: tensor.detach().clone() for name, tensor in fresh_target_state.items()}
    if source_state is not None:
        for name in plan.exact_tensor_names:
            expected[name] = (
                source_state[name]
                .detach()
                .to(
                    device=expected[name].device,
                    dtype=expected[name].dtype,
                )
                .clone()
            )
        for assignment in plan.row_assignments:
            if assignment.source_row is None:
                continue
            name = assignment.tensor_name
            expected[name][assignment.destination_row] = source_state[name][
                assignment.source_row
            ].to(
                device=expected[name].device,
                dtype=expected[name].dtype,
            )

    mismatches = tuple(
        name
        for name in expected
        if not torch.equal(
            initialized_state[name].detach().cpu(),
            expected[name].detach().cpu(),
        )
    )
    if mismatches:
        raise EditingStepZeroGateError(
            f"initialized parameters disagree with the exact transfer plan: {mismatches[:20]!r}"
        )
    nonfinite = tuple(
        name
        for name, tensor in initialized_state.items()
        if not bool(torch.isfinite(tensor.detach()).all())
    )
    if nonfinite:
        raise EditingStepZeroGateError(f"initialized parameters are nonfinite: {nonfinite[:20]!r}")
    copied_rows = sum(assignment.source_row is not None for assignment in plan.row_assignments)
    fresh_rows = len(plan.row_assignments) - copied_rows
    return InitializationParityReport(
        regime=plan.regime,
        transfer_plan_sha256=plan.sha256,
        exact_tensor_count=len(plan.exact_tensor_names),
        copied_row_count=copied_rows,
        fresh_row_count=fresh_rows,
        fresh_tensor_count=len(plan.fresh_tensor_names),
        exact_tensor_names=plan.exact_tensor_names,
        fresh_tensor_names=plan.fresh_tensor_names,
    )


@dataclass(frozen=True)
class StepZeroSuccessorSupportReport:
    """Exact support coverage observed across frozen successor batches."""

    batch_count: int
    row_count: int
    nonterminal_row_count: int
    terminal_row_count: int
    unique_source_state_count: int
    teacher_examples_by_family: Mapping[str, int]
    candidate_marks_by_family: Mapping[str, int]
    productive_successors_by_family: Mapping[str, int]
    examples_by_semantic_cell: Mapping[str, int]


def _family_name(rule_name: str | None) -> str | None:
    return None if rule_name is None else _CYCLE_OP_EXECUTOR_TO_FAMILY.get(rule_name, rule_name)


def _candidate_census(
    compiled: CompiledStateSuccessorMap,
) -> tuple[Counter[str], Counter[str]]:
    marks: Counter[str] = Counter()
    productive_successors: Counter[str] = Counter()
    for group in compiled.successor_groups:
        families = {mark.alias.family_name for mark in group.marks}
        for mark in group.marks:
            marks[mark.alias.family_name] += 1
        for family in families:
            productive_successors[family] += 1
    for mark in compiled.virtual_marks:
        marks[mark.alias.family_name] += 1
    return marks, productive_successors


def audit_successor_support_before_optimization(
    model: FactorizedTraceletRateModel,
    batches: Iterable[FactorizedSuccessorBatch],
    *,
    required_families: tuple[str, ...],
    required_semantic_cells: tuple[str, ...],
    system: RewriteSystem | None = None,
) -> StepZeroSuccessorSupportReport:
    """Recompile and verify frozen successor teachers without an update.

    Each unique exact source is enumerated once through the production marked
    law. Enumeration proves normalized finite mark mass; compilation executes
    every legal mark. Every batch teacher is then replayed and matched to both
    its exact target digest and its full canonical successor alias fiber.
    """

    if not required_families or len(required_families) != len(set(required_families)):
        raise ValueError("required_families must be nonempty and unique")
    unknown = set(required_families) - set(MARK_RULE_NAMES)
    if unknown:
        raise ValueError(f"unknown required families: {sorted(unknown)!r}")
    if (
        not required_semantic_cells
        or len(required_semantic_cells) != len(set(required_semantic_cells))
        or any(
            not isinstance(cell, str) or not cell
            for cell in required_semantic_cells
        )
    ):
        raise ValueError(
            "required_semantic_cells must be nonempty, unique text"
        )
    runtime = system or de_novo_rewrite_system()

    compiled_by_source: dict[str, CompiledStateSuccessorMap] = {}
    teachers: Counter[str] = Counter()
    candidates: Counter[str] = Counter()
    productive_successors: Counter[str] = Counter()
    semantic_cells: Counter[str] = Counter()
    batch_count = 0
    row_count = 0
    nonterminal_count = 0
    terminal_count = 0
    was_training = model.training
    model.eval()
    try:
        for batch_index, batch in enumerate(batches):
            batch_count += 1
            if not isinstance(batch, FactorizedSuccessorBatch):
                raise EditingStepZeroGateError(
                    "step-zero successor audit received another batch type"
                )
            teacher_rates = batch.mark_batch.teacher_rates.detach()
            if not bool(torch.isfinite(teacher_rates).all()) or bool((teacher_rates < 0).any()):
                raise EditingStepZeroGateError(
                    f"batch {batch_index} has nonfinite or negative teacher rates"
                )
            try:
                assert_teachers_in_exact_candidates(batch.mark_batch)
            except Exception as error:
                raise EditingStepZeroGateError(
                    f"batch {batch_index} has a teacher outside live candidates"
                ) from error

            rows = zip(
                batch.mark_batch.states,
                batch.mark_batch.teacher_rule_names,
                batch.mark_batch.teacher_actions,
                teacher_rates.cpu().tolist(),
                batch.fibers,
                batch.state_supports,
                batch.cache_addresses,
                batch.semantic_cell_ids,
                strict=True,
            )
            for row_index, (
                state,
                rule_name,
                action,
                teacher_rate,
                fiber,
                support,
                address,
                semantic_cell,
            ) in enumerate(rows):
                row_count += 1
                source_digest = persistent_slot_state_sha256(state)
                if source_digest != support.source_state_sha256:
                    raise EditingStepZeroGateError(
                        f"batch {batch_index} row {row_index} support has another "
                        "exact source state"
                    )
                compiled = compiled_by_source.get(source_digest)
                if compiled is None:
                    try:
                        compiled = compile_state_successor_map(
                            model,
                            state,
                            time=0.5,
                            system=runtime,
                        )
                    except SuccessorTrainingError as error:
                        raise EditingStepZeroGateError(
                            f"production support compilation failed for batch "
                            f"{batch_index} row {row_index}"
                        ) from error
                    compiled_by_source[source_digest] = compiled
                    state_candidates, state_successors = _candidate_census(compiled)
                    candidates.update(state_candidates)
                    productive_successors.update(state_successors)
                if compiled.state_support != support:
                    raise EditingStepZeroGateError(
                        f"batch {batch_index} row {row_index} cached productive "
                        "support differs from production enumeration"
                    )

                is_nonterminal = float(teacher_rate) > 0.0
                if is_nonterminal:
                    nonterminal_count += 1
                    if address.is_terminal or rule_name is None or action is None or fiber is None:
                        raise EditingStepZeroGateError(
                            f"batch {batch_index} row {row_index} is missing "
                            "nonterminal teacher support"
                        )
                    family = _family_name(rule_name)
                    if family is None:
                        raise EditingStepZeroGateError(
                            f"batch {batch_index} row {row_index} lacks a teacher family"
                        )
                    if not isinstance(semantic_cell, str) or not semantic_cell:
                        raise EditingStepZeroGateError(
                            f"batch {batch_index} row {row_index} lacks a "
                            "nonempty semantic cell"
                        )
                    teachers[family] += 1
                    semantic_cells[semantic_cell] += 1
                    try:
                        replayed = runtime.apply(state, rule_name, action)
                    except Exception as error:
                        raise EditingStepZeroGateError(
                            f"batch {batch_index} row {row_index} teacher is not executable"
                        ) from error
                    replayed_digest = persistent_slot_state_sha256(replayed)
                    if replayed_digest != fiber.target_state_sha256:
                        raise EditingStepZeroGateError(
                            f"batch {batch_index} row {row_index} teacher does "
                            "not execute to its stored exact successor"
                        )
                    if canonical_state_key(replayed) != fiber.target_key:
                        raise EditingStepZeroGateError(
                            f"batch {batch_index} row {row_index} teacher does "
                            "not execute to its stored canonical successor"
                        )
                    try:
                        action_sha256 = rewrite_action_codec_sha256(
                            rule_name,
                            action,
                        )
                        require_exact_successor_action_identity(
                            compiled,
                            target_state_sha256=replayed_digest,
                            action_sha256=action_sha256,
                        )
                        rebuilt_fiber = teacher_successor_fiber_from_exact_digest(
                            compiled,
                            replayed_digest,
                        )
                    except SuccessorTrainingError as error:
                        raise EditingStepZeroGateError(
                            f"batch {batch_index} row {row_index} teacher is "
                            "absent from exact production marked support"
                        ) from error
                    if rebuilt_fiber != fiber:
                        raise EditingStepZeroGateError(
                            f"batch {batch_index} row {row_index} cached teacher "
                            "fiber differs from production enumeration"
                        )
                else:
                    terminal_count += 1
                    if (
                        not address.is_terminal
                        or rule_name is not None
                        or action is not None
                        or fiber is not None
                        or semantic_cell is not None
                    ):
                        raise EditingStepZeroGateError(
                            f"batch {batch_index} row {row_index} has malformed terminal support"
                        )

            try:
                with torch.no_grad():
                    prediction = forward_teacher_successor_batch(
                        model,
                        batch.mark_batch,
                        batch.fibers,
                        state_supports=batch.state_supports,
                    )
            except SuccessorTrainingError as error:
                raise EditingStepZeroGateError(
                    f"batch {batch_index} cannot score its frozen successor support"
                ) from error
            if not bool(torch.isfinite(prediction.total_hazard).all()):
                raise EditingStepZeroGateError(f"batch {batch_index} has nonfinite total hazard")
            nonterminal = batch.mark_batch.teacher_rates > 0
            selected = prediction.selected_productive_successor_log_probability[
                nonterminal.to(prediction.selected_productive_successor_log_probability.device)
            ]
            if selected.numel() and not bool(torch.isfinite(selected).all()):
                raise EditingStepZeroGateError(
                    f"batch {batch_index} has nonfinite teacher-successor mass"
                )
    finally:
        model.train(was_training)

    if batch_count == 0:
        raise EditingStepZeroGateError("step-zero successor support audit received no batches")
    missing_teachers = tuple(family for family in required_families if teachers[family] == 0)
    if missing_teachers:
        raise EditingStepZeroGateError(
            "required families have no teacher examples in the step-zero "
            f"probe: {missing_teachers!r}"
        )
    missing_candidates = tuple(family for family in required_families if candidates[family] == 0)
    if missing_candidates:
        raise EditingStepZeroGateError(
            "required families have no production candidates in the step-zero "
            f"probe: {missing_candidates!r}"
        )
    missing_productive = tuple(
        family for family in required_families if productive_successors[family] == 0
    )
    if missing_productive:
        raise EditingStepZeroGateError(
            "required families have no productive molecular successors in the "
            f"step-zero probe: {missing_productive!r}"
        )
    missing_cells = tuple(
        cell
        for cell in required_semantic_cells
        if semantic_cells[cell] == 0
    )
    if missing_cells:
        raise EditingStepZeroGateError(
            "required semantic cells have no nonterminal examples in the "
            f"step-zero probe: {missing_cells!r}"
        )
    return StepZeroSuccessorSupportReport(
        batch_count=batch_count,
        row_count=row_count,
        nonterminal_row_count=nonterminal_count,
        terminal_row_count=terminal_count,
        unique_source_state_count=len(compiled_by_source),
        teacher_examples_by_family=dict(sorted(teachers.items())),
        candidate_marks_by_family=dict(sorted(candidates.items())),
        productive_successors_by_family=dict(sorted(productive_successors.items())),
        examples_by_semantic_cell=dict(sorted(semantic_cells.items())),
    )
