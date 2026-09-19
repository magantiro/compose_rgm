from __future__ import annotations

import math

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.control.route_distilled_goal_expert import make_route_expert
from compose_v4.control.sequential_region_proposer import (
    SequentialRegionBudgets,
    propose_sequential_region_paths,
)
from compose_v4.control.structural_subgoal_policy import (
    MarginalSubgoalPolicy,
    StructuralDeltaTemplate,
    transfer_bindings,
)
from compose_v4.rewrite.action_codec_v4 import decode_action
from compose_v4.rewrite.kernel import (
    canonical_state_key,
    editing_v2_semantic_rewrite_system,
)


def _methane() -> MolecularGraph:
    atom_types = np.zeros(48, dtype=np.int32)
    charges = np.zeros(48, dtype=np.int32)
    hydrogens = np.zeros(48, dtype=np.int32)
    bonds = np.zeros((48, 48), dtype=np.int32)
    atom_types[0] = 2
    hydrogens[0] = 4
    return MolecularGraph(atom_types, charges, hydrogens, bonds)


def _expert():
    root_append = StructuralDeltaTemplate(
        input_atoms=((2, 0, 4, 0),),
        input_bonds=((0,),),
        target_atoms=((2, 0, 3, 1),),
        output_atoms=((2, 0, 3, 1),),
        target_bonds=((0, 1), (1, 0)),
    )
    later_append = StructuralDeltaTemplate(
        input_atoms=((2, 0, 3, 1),),
        input_bonds=((0,),),
        target_atoms=((2, 0, 2, 2),),
        output_atoms=((2, 0, 3, 1),),
        target_bonds=((0, 1), (1, 0)),
    )
    template_ids = tuple(sorted((root_append.template_id, later_append.template_id)))
    marginal = MarginalSubgoalPolicy(
        template_ids=template_ids,
        probabilities=(0.5, 0.5),
        goal_count_probabilities=(0.25, 0.25, 0.25, 0.25),
        exploration_floor=0.1,
        training_identity="generic-marginal-fit",
    )
    return (
        make_route_expert(
            (root_append, later_append),
            marginal,
            training_evidence_identity="split-first-generic-training",
        ),
        root_append,
        later_append,
    )


def _budgets(**updates) -> SequentialRegionBudgets:
    values = {
        "max_depth": 4,
        "beam_width": 4,
        "expansion_width": 4,
        "max_bindings_per_template": 4,
        "max_binding_visits": 128,
        "max_realization_attempts": 32,
        "maximum_primitives": 32,
        "maximum_expansions_per_realization": 64,
    }
    values.update(updates)
    return SequentialRegionBudgets(**values)


def _signature(batch) -> tuple:
    return tuple(
        (
            proposal.endpoint_key,
            proposal.log_probability,
            tuple(
                (
                    step.template_id,
                    step.binding,
                    step.program_id,
                    step.endpoint_key,
                    step.actions,
                )
                for step in proposal.steps
            ),
        )
        for proposal in batch.proposals
    )


def test_later_region_binds_only_after_exact_first_successor() -> None:
    source = _methane()
    expert, root_append, later_append = _expert()
    assert transfer_bindings(later_append, source).assignments == ()

    batch = propose_sequential_region_paths(
        source, expert, budgets=_budgets(max_depth=2)
    )

    depth_two = [proposal for proposal in batch.proposals if proposal.depth == 2]
    assert len(depth_two) == 1
    proposal = depth_two[0]
    assert proposal.endpoint.n_real_atoms == 3
    assert proposal.templates == (root_append, later_append)
    assert proposal.log_probability == expert.marginal.score(proposal.templates)
    assert proposal.steps[0].endpoint_key != proposal.steps[1].endpoint_key
    assert all(step.actions for step in proposal.steps)
    assert batch.telemetry["completed_depth_2"] == 1


def test_sequential_region_output_is_deterministic_and_exactly_replayable() -> None:
    source = _methane()
    expert, _, _ = _expert()
    budgets = _budgets(max_depth=4)

    first = propose_sequential_region_paths(source, expert, budgets=budgets)
    second = propose_sequential_region_paths(source, expert, budgets=budgets)

    assert _signature(first) == _signature(second)
    assert first.telemetry == second.telemetry
    deepest = max(first.proposals, key=lambda row: row.depth)
    assert deepest.depth == 4
    system = editing_v2_semantic_rewrite_system()
    replay = source
    for step in deepest.steps:
        for action_record in step.actions:
            replay = system.apply(replay, *decode_action(action_record))
        assert canonical_state_key(replay) == step.endpoint_key
    assert canonical_state_key(replay) == deepest.endpoint_key
    assert (
        first.telemetry["exact_realization_precision_numerator"]
        == first.telemetry["exact_realization_precision_denominator"]
    )


@pytest.mark.parametrize("max_depth", [1, 2, 3, 4])
def test_sequential_region_depth_and_beam_limits_are_enforced(max_depth: int) -> None:
    source = _methane()
    expert, _, _ = _expert()

    batch = propose_sequential_region_paths(
        source,
        expert,
        budgets=_budgets(max_depth=max_depth, beam_width=1),
    )

    assert max(proposal.depth for proposal in batch.proposals) == max_depth
    assert all(proposal.depth <= max_depth for proposal in batch.proposals)
    assert all(
        batch.telemetry[f"completed_depth_{depth}"] <= 1
        for depth in range(1, max_depth + 1)
    )


def test_sequential_region_realization_and_primitive_budgets_stop_expansion() -> None:
    source = _methane()
    expert, _, _ = _expert()

    realization_limited = propose_sequential_region_paths(
        source,
        expert,
        budgets=_budgets(max_realization_attempts=1),
    )
    assert realization_limited.telemetry["realization_attempts"] == 1
    assert realization_limited.telemetry["realization_budget_exhausted"] == 1
    assert max(row.depth for row in realization_limited.proposals) == 1

    primitive_limited = propose_sequential_region_paths(
        source,
        expert,
        budgets=_budgets(maximum_primitives=1),
    )
    assert max(row.depth for row in primitive_limited.proposals) == 1
    assert primitive_limited.telemetry[
        "realization_status:cumulative_primitive_budget_abstention"
    ]

    binding_limited = propose_sequential_region_paths(
        source,
        expert,
        budgets=_budgets(max_binding_visits=1),
    )
    assert binding_limited.telemetry["binding_visits"] == 1
    assert binding_limited.telemetry["binding_visit_budget_exhausted"] == 1


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"max_depth": 0}, "positive integer"),
        ({"max_depth": 5}, "within 1..4"),
        ({"beam_width": 0}, "positive integer"),
        ({"expansion_width": 0}, "positive integer"),
        ({"max_realization_attempts": 0}, "positive integer"),
        ({"maximum_primitives": 33}, "limited to 32"),
    ],
)
def test_sequential_region_rejects_invalid_budgets(updates: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _budgets(**updates)


def test_route_expert_checkpoint_contains_no_runtime_task_or_teacher_fields() -> None:
    expert, _, _ = _expert()
    checkpoint = expert.checkpoint()

    def keys(value) -> set[str]:
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, (list, tuple)):
            return set().union(*(keys(item) for item in value), set())
        return set()

    forbidden = {
        "target_name",
        "cell",
        "cell_name",
        "route_id",
        "source_graph",
        "endpoint",
        "endpoints",
        "scores",
        "teacher_sequence",
        "teacher_sequences",
    }
    assert keys(checkpoint).isdisjoint(forbidden)
    assert math.isclose(sum(checkpoint["marginal"]["probabilities"]), 1.0)
