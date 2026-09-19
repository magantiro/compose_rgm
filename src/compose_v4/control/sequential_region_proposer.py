"""Bounded sequential complete-region proposals from a route expert.

The existing structural-goal proposer composes regions that all bind on one
root graph.  This additive path instead commits one exact complete region,
then binds the next address-free template on that exact successor.  It uses
only the generic templates and marginal law already present in a
``RouteDistilledGoalExpert`` checkpoint.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.control.complete_region_program import (
    execute_complete_region_program,
    program_from_structural_goal,
)
from compose_v4.control.route_distilled_goal_expert import RouteDistilledGoalExpert
from compose_v4.control.structural_subgoal import (
    StructuralGoal,
    StructuralSubgoal,
    instantiate_goal,
)
from compose_v4.control.structural_subgoal_policy import (
    StructuralDeltaTemplate,
    materialize_template,
    transfer_bindings,
)
from compose_v4.control.structural_subgoal_realizer import RealizerConfig
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state


@dataclass(frozen=True)
class SequentialRegionBudgets:
    """Fixed work limits for one deterministic sequential proposal call."""

    max_depth: int = 4
    beam_width: int = 32
    expansion_width: int = 24
    max_bindings_per_template: int = 4
    max_binding_visits: int = 16_384
    max_realization_attempts: int = 128
    maximum_primitives: int = 32
    maximum_expansions_per_realization: int = 4_000

    def __post_init__(self) -> None:
        names = (
            "max_depth",
            "beam_width",
            "expansion_width",
            "max_bindings_per_template",
            "max_binding_visits",
            "max_realization_attempts",
            "maximum_primitives",
            "maximum_expansions_per_realization",
        )
        for name in names:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"sequential region {name} must be a positive integer")
        if self.max_depth > 4:
            raise ValueError("sequential complete-region depth must be within 1..4")
        if self.maximum_primitives > 32:
            raise ValueError(
                "sequential complete-region paths are limited to 32 primitives"
            )


@dataclass(frozen=True)
class SequentialRegionStep:
    """Exact provenance for one materialized and committed region."""

    template: StructuralDeltaTemplate
    binding: tuple[int, ...]
    patch: StructuralSubgoal
    program_id: str
    endpoint_key: str
    endpoint_state: dict[str, Any]
    actions: tuple[dict[str, Any], ...]
    primitive_count: int
    compiler_strategy: str | None
    expanded: int
    attempted: int

    @property
    def template_id(self) -> str:
        return self.template.template_id


@dataclass(frozen=True)
class SequentialRegionProposal:
    """One complete one-to-four-region path ending at an exact successor."""

    endpoint: MolecularGraph
    steps: tuple[SequentialRegionStep, ...]
    log_probability: float

    @property
    def depth(self) -> int:
        return len(self.steps)

    @property
    def templates(self) -> tuple[StructuralDeltaTemplate, ...]:
        return tuple(step.template for step in self.steps)

    @property
    def total_primitives(self) -> int:
        return sum(step.primitive_count for step in self.steps)

    @property
    def endpoint_key(self) -> str:
        return canonical_state_key(self.endpoint)


@dataclass(frozen=True)
class SequentialRegionProposalBatch:
    """Bounded sequential proposals and auditable work accounting."""

    proposals: tuple[SequentialRegionProposal, ...]
    telemetry: dict[str, Any]


@dataclass(frozen=True)
class _MaterializedExtension:
    template: StructuralDeltaTemplate
    binding: tuple[int, ...]
    patch: StructuralSubgoal
    expected_endpoint_key: str
    log_probability: float


def _path_key(proposal: SequentialRegionProposal) -> tuple:
    return tuple(
        (step.template_id, step.binding, step.endpoint_key) for step in proposal.steps
    )


def _rank_key(proposal: SequentialRegionProposal) -> tuple:
    return (-proposal.log_probability, proposal.endpoint_key, _path_key(proposal))


def _materialized_extensions(
    prefix: SequentialRegionProposal,
    expert: RouteDistilledGoalExpert,
    budgets: SequentialRegionBudgets,
    telemetry: Counter,
) -> list[_MaterializedExtension]:
    """Bind and materialize one region on the prefix's exact successor."""

    templates = prefix.templates
    best_by_target: dict[str, _MaterializedExtension] = {}
    for template in expert.templates:
        census = transfer_bindings(
            template,
            prefix.endpoint,
            max_bindings=budgets.max_bindings_per_template,
            max_visits=budgets.max_binding_visits,
        )
        telemetry["binding_visits"] += census.visits
        telemetry["binding_truncations"] += int(census.truncated)
        telemetry["templates_with_bindings"] += int(bool(census.assignments))
        for binding in census.assignments:
            telemetry["materialization_attempts"] += 1
            try:
                patch = materialize_template(template, prefix.endpoint, binding)
                expected, _ = instantiate_goal(
                    prefix.endpoint,
                    StructuralGoal((patch,)),
                    (binding,),
                )
            except ValueError:
                telemetry["materialization_rejections"] += 1
                continue
            expected_key = canonical_state_key(expected)
            extension = _MaterializedExtension(
                template=template,
                binding=binding,
                patch=patch,
                expected_endpoint_key=expected_key,
                log_probability=expert.marginal.score((*templates, template)),
            )
            previous = best_by_target.get(expected_key)
            extension_key = (template.template_id, binding)
            if previous is not None:
                previous_key = (previous.template.template_id, previous.binding)
                if (-previous.log_probability, previous_key) <= (
                    -extension.log_probability,
                    extension_key,
                ):
                    telemetry["materialized_aliases"] += 1
                    continue
                telemetry["materialized_aliases"] += 1
            best_by_target[expected_key] = extension
    return sorted(
        best_by_target.values(),
        key=lambda row: (
            -row.log_probability,
            row.expected_endpoint_key,
            row.template.template_id,
            row.binding,
        ),
    )[: budgets.expansion_width]


def _realize_extension(
    prefix: SequentialRegionProposal,
    extension: _MaterializedExtension,
    budgets: SequentialRegionBudgets,
) -> tuple[SequentialRegionProposal | None, str]:
    remaining = budgets.maximum_primitives - prefix.total_primitives
    if remaining < 1:
        return None, "cumulative_primitive_budget_abstention"
    program = program_from_structural_goal(StructuralGoal((extension.patch,)))
    receipt = execute_complete_region_program(
        prefix.endpoint,
        program,
        resolved_bindings=(extension.binding,),
        config=RealizerConfig(
            maximum_primitives=remaining,
            maximum_expansions=budgets.maximum_expansions_per_realization,
        ),
    )
    status = str(receipt["status"])
    if status != "committed":
        return None, status
    endpoint_state = receipt["committed_endpoint_state"]
    if not isinstance(endpoint_state, dict):
        raise TypeError("sequential region runtime omitted its committed state")
    endpoint = decode_state(endpoint_state)
    endpoint_key = canonical_state_key(endpoint)
    if endpoint_key != extension.expected_endpoint_key:
        raise RuntimeError(
            "sequential complete-region realization changed its bound target"
        )
    primitive_count = int(receipt["realized_primitive_count"])
    actions = receipt.get("realized_actions")
    if not isinstance(actions, list) or len(actions) != primitive_count:
        raise RuntimeError(
            "sequential complete-region realization omitted exact actions"
        )
    if prefix.total_primitives + primitive_count > budgets.maximum_primitives:
        raise RuntimeError(
            "sequential complete-region runtime exceeded its primitive budget"
        )
    step = SequentialRegionStep(
        template=extension.template,
        binding=extension.binding,
        patch=extension.patch,
        program_id=program.program_id,
        endpoint_key=endpoint_key,
        endpoint_state=endpoint_state,
        actions=tuple(actions),
        primitive_count=primitive_count,
        compiler_strategy=receipt.get("compiler_strategy"),
        expanded=int(receipt.get("expanded", 0)),
        attempted=int(receipt.get("attempted", 0)),
    )
    return (
        SequentialRegionProposal(
            endpoint=endpoint,
            steps=(*prefix.steps, step),
            log_probability=extension.log_probability,
        ),
        status,
    )


def propose_sequential_region_paths(
    source: MolecularGraph,
    expert: RouteDistilledGoalExpert,
    *,
    budgets: SequentialRegionBudgets | None = None,
) -> SequentialRegionProposalBatch:
    """Propose exact paths whose later regions bind on earlier successors.

    No target, cell, route, teacher sequence, endpoint archive, or task score is
    an input.  Every expansion is derived from the current exact successor and
    the address-free templates plus marginal probabilities in ``expert``.
    """

    budgets = SequentialRegionBudgets() if budgets is None else budgets
    if not isinstance(budgets, SequentialRegionBudgets):
        raise TypeError("budgets must be SequentialRegionBudgets")
    telemetry: Counter = Counter()
    root = SequentialRegionProposal(source, (), 0.0)
    frontier = (root,)
    all_rows: list[SequentialRegionProposal] = []
    realization_attempts = 0

    for depth in range(1, budgets.max_depth + 1):
        next_by_endpoint: dict[str, SequentialRegionProposal] = {}
        for prefix in frontier:
            extensions = _materialized_extensions(prefix, expert, budgets, telemetry)
            telemetry["expansions_selected"] += len(extensions)
            for extension in extensions:
                if realization_attempts >= budgets.max_realization_attempts:
                    telemetry["realization_budget_exhausted"] = 1
                    break
                realization_attempts += 1
                telemetry["realization_attempts"] += 1
                try:
                    proposal, status = _realize_extension(prefix, extension, budgets)
                except ValueError:
                    telemetry["realization_status:invalid_target"] += 1
                    continue
                telemetry[f"realization_status:{status}"] += 1
                if proposal is None:
                    continue
                telemetry["realized_primitives"] += proposal.steps[-1].primitive_count
                previous = next_by_endpoint.get(proposal.endpoint_key)
                if previous is not None and _rank_key(previous) <= _rank_key(proposal):
                    telemetry["realized_aliases"] += 1
                    continue
                if previous is not None:
                    telemetry["realized_aliases"] += 1
                next_by_endpoint[proposal.endpoint_key] = proposal
            if realization_attempts >= budgets.max_realization_attempts:
                break
        next_rows = sorted(next_by_endpoint.values(), key=_rank_key)
        frontier = tuple(next_rows[: budgets.beam_width])
        telemetry[f"completed_depth_{depth}"] = len(frontier)
        if realization_attempts >= budgets.max_realization_attempts:
            telemetry["realization_budget_exhausted"] = 1
        all_rows.extend(frontier)
        if not frontier or realization_attempts >= budgets.max_realization_attempts:
            break

    best_by_endpoint: dict[str, SequentialRegionProposal] = {}
    for proposal in all_rows:
        previous = best_by_endpoint.get(proposal.endpoint_key)
        if previous is None or _rank_key(proposal) < _rank_key(previous):
            best_by_endpoint[proposal.endpoint_key] = proposal
    proposals = tuple(sorted(best_by_endpoint.values(), key=_rank_key))
    completed = sum(
        count
        for status, count in telemetry.items()
        if status == "realization_status:committed"
    )
    result_telemetry = {
        **dict(sorted(telemetry.items())),
        "max_depth": budgets.max_depth,
        "beam_width": budgets.beam_width,
        "expansion_width": budgets.expansion_width,
        "max_realization_attempts": budgets.max_realization_attempts,
        "maximum_primitives": budgets.maximum_primitives,
        "unique_exact_endpoints": len(proposals),
        "exact_realization_precision_numerator": completed,
        "exact_realization_precision_denominator": completed,
        "task_or_target_input_used": False,
        "endpoint_or_teacher_input_used": False,
    }
    return SequentialRegionProposalBatch(proposals, result_telemetry)


__all__ = [
    "SequentialRegionBudgets",
    "SequentialRegionProposal",
    "SequentialRegionProposalBatch",
    "SequentialRegionStep",
    "propose_sequential_region_paths",
]
