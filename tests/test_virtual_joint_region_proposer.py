from __future__ import annotations

import math

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import MolecularGraph, is_rdkit_valid
from compose_v4.chem.state import is_connected_or_null
from compose_v4.control.route_distilled_goal_expert import make_route_expert
from compose_v4.control.structural_subgoal import StructuralGoal, instantiate_goal
from compose_v4.control.structural_subgoal_policy import (
    MarginalSubgoalPolicy,
    StructuralDeltaTemplate,
    materialize_template,
    transfer_bindings,
)
from compose_v4.control.virtual_joint_region_proposer import (
    VirtualJointRegionBudgets,
    propose_virtual_joint_region_paths,
)
from compose_v4.data.charge_policy import charge_policy_preserved
from compose_v4.rewrite.action_codec_v4 import decode_action
from compose_v4.rewrite.kernel import (
    canonical_state_key,
    editing_v2_semantic_rewrite_system,
)


def _graph(
    atom_types: tuple[int, ...],
    hydrogens: tuple[int, ...],
    edges: tuple[tuple[int, int, int], ...],
) -> MolecularGraph:
    types = np.zeros(48, dtype=np.int32)
    charges = np.zeros(48, dtype=np.int32)
    hydrogen_counts = np.zeros(48, dtype=np.int32)
    bonds = np.zeros((48, 48), dtype=np.int32)
    types[: len(atom_types)] = atom_types
    hydrogen_counts[: len(hydrogens)] = hydrogens
    for left, right, order in edges:
        bonds[left, right] = bonds[right, left] = order
    return MolecularGraph(types, charges, hydrogen_counts, bonds)


def _expert(
    templates: tuple[StructuralDeltaTemplate, ...],
    *,
    weights: dict[str, float] | None = None,
):
    ordered_ids = tuple(sorted({row.template_id for row in templates}))
    if weights is None:
        probabilities = tuple(1.0 / len(ordered_ids) for _ in ordered_ids)
    else:
        total = sum(weights.values())
        probabilities = tuple(
            weights[template_id] / total for template_id in ordered_ids
        )
    marginal = MarginalSubgoalPolicy(
        template_ids=ordered_ids,
        probabilities=probabilities,
        goal_count_probabilities=(0.25, 0.25, 0.25, 0.25),
        exploration_floor=0.1,
        training_identity="generic-split-first-marginal",
    )
    unique = {row.template_id: row for row in templates}
    return make_route_expert(
        tuple(unique.values()),
        marginal,
        training_evidence_identity="generic-split-first-routes",
    )


def _co2_compensating_expert():
    source = _graph((2, 4, 4), (0, 0, 0), ((0, 1, 2), (0, 2, 2)))
    # Applied to either C=O arm in isolation, this gives the carbon two
    # hydrogens while leaving the other double bond intact, so RDKit rejects
    # the over-valent singleton.  Applying it to both arms jointly yields
    # methanediol, O-C-O, which is valid.
    reduce_one_carbonyl = StructuralDeltaTemplate(
        input_atoms=((2, 0, 0, 2), (4, 0, 0, 1)),
        input_bonds=((0, 2), (2, 0)),
        target_atoms=((2, 0, 2, 2), (4, 0, 1, 1)),
        output_atoms=(),
        target_bonds=((0, 1), (1, 0)),
    )
    return source, reduce_one_carbonyl, _expert((reduce_one_carbonyl,))


def _append_expert():
    source = _graph((2,), (4,), ())
    append = StructuralDeltaTemplate(
        input_atoms=((2, 0, 4, 0),),
        input_bonds=((0,),),
        target_atoms=((2, 0, 3, 1),),
        output_atoms=((2, 0, 3, 1),),
        target_bonds=((0, 1), (1, 0)),
    )
    return source, _expert((append,))


def _crowded_shallow_expert():
    source, joint_template, _ = _co2_compensating_expert()
    shallow = tuple(
        StructuralDeltaTemplate(
            input_atoms=((4, 0, 0, 1),),
            input_bonds=((0,),),
            target_atoms=(target,),
            output_atoms=(),
            target_bonds=((0,),),
        )
        for target in (
            (7, 0, 0, 1),
            (3, 0, 1, 1),
            (2, 0, 2, 1),
        )
    )
    weights = {joint_template.template_id: 0.7}
    weights.update({row.template_id: 0.1 for row in shallow})
    return source, _expert((joint_template, *shallow), weights=weights)


def _budgets(**updates) -> VirtualJointRegionBudgets:
    values = {
        "max_depth": 4,
        "beam_width": 16,
        "expansion_width": 16,
        "max_bindings_per_template": 8,
        "max_binding_visits": 128,
        "max_planning_expansions": 128,
        "max_targets": 16,
        "max_realization_attempts": 16,
        "max_realization_expansions": 256,
        "maximum_expansions_per_realization": 64,
        "maximum_primitives": 32,
    }
    values.update(updates)
    return VirtualJointRegionBudgets(**values)


def _signature(batch) -> tuple:
    return tuple(
        (
            proposal.endpoint_key,
            proposal.log_probability,
            proposal.program_id,
            proposal.actions,
            tuple(
                (
                    step.template_id,
                    step.binding,
                    step.constituent_key,
                    tuple(
                        (role.region_index, role.output_index, role.virtual_slot)
                        for role in step.created_roles
                    ),
                )
                for step in proposal.steps
            ),
        )
        for proposal in batch.proposals
    )


def test_invalid_single_regions_are_retained_until_a_valid_joint_stop() -> None:
    source, template, expert = _co2_compensating_expert()
    bindings = transfer_bindings(template, source).assignments
    assert bindings == ((0, 1), (0, 2))
    patches = tuple(materialize_template(template, source, row) for row in bindings)
    for patch, binding in zip(patches, bindings, strict=True):
        with pytest.raises(ValueError, match="outside molecular support"):
            instantiate_goal(source, StructuralGoal((patch,)), (binding,))
    joint, _ = instantiate_goal(source, StructuralGoal(patches), bindings)
    assert is_rdkit_valid(joint)
    assert canonical_state_key(joint) == "OCO"

    depth_one = propose_virtual_joint_region_paths(
        source, expert, budgets=_budgets(max_depth=1)
    )
    assert depth_one.proposals == ()
    assert depth_one.telemetry["standalone_invalid_constituents_retained"] == 2

    batch = propose_virtual_joint_region_paths(
        source, expert, budgets=_budgets(max_depth=2)
    )

    assert len(batch.proposals) == 1
    proposal = batch.proposals[0]
    assert proposal.depth == 2
    assert proposal.endpoint_key == "OCO"
    assert proposal.primitive_count == 2
    assert batch.telemetry["standalone_invalid_constituents_retained"] == 2
    assert batch.telemetry["valid_joint_endpoints_recovered"] == 1
    assert batch.telemetry["virtual_prefix_commits"] == 0
    assert batch.telemetry["partial_endpoint_evaluations"] == 0
    assert batch.telemetry["primitive_teacher_actions_used"] == 0
    assert batch.telemetry["final_target_abstentions"] == 2
    assert batch.telemetry["compiler_abstentions"] == 0
    assert batch.telemetry["exact_realization_precision_numerator"] == 1
    assert batch.telemetry["exact_realization_precision_denominator"] == 1


def test_shallow_target_cap_preserves_deeper_joint_target_coverage() -> None:
    source, expert = _crowded_shallow_expert()

    batch = propose_virtual_joint_region_paths(
        source,
        expert,
        budgets=_budgets(max_depth=2, max_targets=2),
    )

    assert batch.telemetry["valid_stop_targets_depth_1"] >= 3
    assert batch.telemetry["target_budget_drops_depth_1"] >= 1
    assert batch.telemetry["target_quota_depth_1"] == 1
    assert batch.telemetry["target_quota_depth_2"] == 1
    assert batch.telemetry["target_depths_retained"] == 2
    assert batch.telemetry["valid_joint_endpoints_recovered"] == 1
    assert any(row.depth == 2 and row.endpoint_key == "OCO" for row in batch.proposals)


def test_joint_execution_replays_exactly_through_only_valid_intermediates() -> None:
    source, _, expert = _co2_compensating_expert()
    batch = propose_virtual_joint_region_paths(
        source, expert, budgets=_budgets(max_depth=2)
    )
    proposal = batch.proposals[0]
    system = editing_v2_semantic_rewrite_system()
    replay = source
    for action_record in proposal.actions:
        replay = system.apply(replay, *decode_action(action_record))
        assert is_rdkit_valid(replay)
        assert is_connected_or_null(replay)
        assert charge_policy_preserved(source, replay)
    assert canonical_state_key(replay) == proposal.endpoint_key


def test_joint_planning_is_deterministic_and_globally_budgeted() -> None:
    source, _, expert = _co2_compensating_expert()
    budgets = _budgets(max_depth=2)

    first = propose_virtual_joint_region_paths(source, expert, budgets=budgets)
    second = propose_virtual_joint_region_paths(source, expert, budgets=budgets)

    assert _signature(first) == _signature(second)
    assert first.telemetry == second.telemetry
    assert first.telemetry["binding_visits"] <= budgets.max_binding_visits
    assert first.telemetry["planning_expansions"] <= budgets.max_planning_expansions
    assert first.telemetry["unique_valid_targets"] <= budgets.max_targets
    assert first.telemetry["realization_attempts"] <= budgets.max_realization_attempts
    assert first.telemetry["realizer_expansions"] <= budgets.max_realization_expansions

    planning_limited = propose_virtual_joint_region_paths(
        source,
        expert,
        budgets=_budgets(max_depth=2, max_planning_expansions=1),
    )
    assert planning_limited.proposals == ()
    assert planning_limited.telemetry["planning_expansions"] == 1
    assert planning_limited.telemetry["planning_expansion_budget_exhausted"] == 1

    realization_limited = propose_virtual_joint_region_paths(
        source,
        expert,
        budgets=_budgets(max_depth=2, max_realization_expansions=1),
    )
    assert realization_limited.proposals == ()
    assert realization_limited.telemetry["realizer_expansions"] == 1
    assert realization_limited.telemetry["realization_expansion_budget_exhausted"] == 1

    per_realization_limited = propose_virtual_joint_region_paths(
        source,
        expert,
        budgets=_budgets(
            max_depth=2,
            max_realization_expansions=256,
            maximum_expansions_per_realization=1,
        ),
    )
    assert per_realization_limited.proposals == ()
    assert per_realization_limited.telemetry["realizer_expansions"] == 1
    assert per_realization_limited.telemetry["compiler_abstentions"] == 1
    assert (
        per_realization_limited.telemetry["per_realization_expansion_cap_abstentions"]
        == 1
    )
    assert (
        per_realization_limited.telemetry.get(
            "realization_expansion_budget_exhausted", 0
        )
        == 0
    )


def test_expansion_width_truncation_reports_constituent_ranks() -> None:
    source, _, expert = _co2_compensating_expert()

    batch = propose_virtual_joint_region_paths(
        source,
        expert,
        budgets=_budgets(max_depth=2, expansion_width=1),
    )

    ranks = batch.telemetry["bound_constituent_ranks"]
    assert [row["rank"] for row in ranks] == [1, 2]
    assert sum(row["selected_for_expansion"] for row in ranks) == 1
    assert batch.telemetry["bound_constituents"] == 2
    assert batch.telemetry["selected_constituents"] == 1
    assert batch.telemetry["constituents_truncated_by_expansion_width"] == 1
    assert batch.proposals == ()


def test_created_role_to_virtual_slot_provenance_is_preserved() -> None:
    source, expert = _append_expert()

    batch = propose_virtual_joint_region_paths(
        source, expert, budgets=_budgets(max_depth=1)
    )

    assert len(batch.proposals) == 1
    proposal = batch.proposals[0]
    assert proposal.depth == 1
    assert len(proposal.created_roles) == 1
    role = proposal.created_roles[0]
    assert (role.region_index, role.output_index, role.virtual_slot) == (0, 0, 1)
    assert proposal.steps[0].binding == (0,)
    assert proposal.endpoint.n_real_atoms == 2


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"max_depth": 0}, "positive integer"),
        ({"max_depth": 5}, "within 1..4"),
        ({"max_binding_visits": 0}, "positive integer"),
        ({"max_planning_expansions": 0}, "positive integer"),
        ({"max_targets": 0}, "positive integer"),
        ({"max_depth": 4, "max_targets": 3}, "cover every requested depth"),
        ({"max_realization_expansions": 0}, "positive integer"),
        ({"maximum_expansions_per_realization": 0}, "positive integer"),
        ({"maximum_primitives": 33}, "limited to 32"),
    ],
)
def test_joint_region_rejects_invalid_budgets(updates: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _budgets(**updates)


def test_runtime_checkpoint_has_no_task_route_endpoint_or_teacher_fields() -> None:
    source, _, expert = _co2_compensating_expert()
    checkpoint = expert.checkpoint()

    def keys(value) -> set[str]:
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, (list, tuple)):
            return set().union(*(keys(item) for item in value), set())
        return set()

    forbidden = {
        "task",
        "target",
        "target_name",
        "cell",
        "cell_name",
        "route_id",
        "source_graph",
        "endpoint",
        "endpoints",
        "smiles",
        "scores",
        "teacher_action",
        "teacher_actions",
    }
    assert keys(checkpoint).isdisjoint(forbidden)
    assert math.isclose(sum(checkpoint["marginal"]["probabilities"]), 1.0)
    with pytest.raises(ValueError, match="ranker support is not implemented"):
        propose_virtual_joint_region_paths(source, expert, ranker=object())
