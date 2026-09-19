from __future__ import annotations

import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.control.edit_program import atom_signature, environment
from compose_v4.control.structural_subgoal import (
    StructuralGoal,
    StructuralSubgoal,
    instantiate_goal,
)
from compose_v4.control.structural_subgoal_policy import (
    ProposedStructuralGoal,
    StructuralDeltaTemplate,
    fit_marginal_subgoal_policy,
    materialize_template,
    minimize_subgoal,
    proposal_rewrite_event_count,
    proposal_rewrite_scale,
    propose_structural_goals,
    select_scale_balanced_proposals,
    structural_rewrite_event_count,
    template_features,
    transfer_bindings,
)
from compose_v4.rewrite.kernel import canonical_state_key


def _ethane(*, first_hydrogens: int = 3) -> MolecularGraph:
    atom_types = np.zeros(48, dtype=np.int32)
    charges = np.zeros(48, dtype=np.int32)
    hydrogens = np.zeros(48, dtype=np.int32)
    bonds = np.zeros((48, 48), dtype=np.int32)
    atom_types[:2] = 2
    hydrogens[:2] = (first_hydrogens, 3)
    bonds[0, 1] = bonds[1, 0] = 1
    return MolecularGraph(atom_types, charges, hydrogens, bonds)


def test_minimized_delta_preserves_bound_endpoint():
    source = _ethane()
    subgoal = StructuralSubgoal(
        input_atoms=(atom_signature(source, 0), atom_signature(source, 1)),
        input_bonds=((0, 1), (1, 0)),
        environments=(environment(source, 0), environment(source, 1)),
        target_atoms=((2, 0, 2, 2), atom_signature(source, 1)),
        output_atoms=((4, 0, 1, 1),),
        target_bonds=((0, 1, 1), (1, 0, 0), (1, 0, 0)),
    )
    full, _ = instantiate_goal(source, StructuralGoal((subgoal,)), ((0, 1),))

    template, indices = minimize_subgoal(subgoal)
    concrete = materialize_template(template, source, tuple((0, 1)[i] for i in indices))
    reduced, _ = instantiate_goal(
        source,
        StructuralGoal((concrete,)),
        (tuple((0, 1)[i] for i in indices),),
    )

    assert indices == (0,)
    assert canonical_state_key(reduced) == canonical_state_key(full)


def test_transfer_binding_uses_topology_and_adapts_current_hydrogens():
    source = _ethane(first_hydrogens=2)
    template = StructuralDeltaTemplate(
        input_atoms=((2, 0, 3, 1),),
        input_bonds=((0,),),
        target_atoms=((2, 0, 2, 2),),
        output_atoms=((4, 0, 1, 1),),
        target_bonds=((0, 1), (1, 0)),
    )

    census = transfer_bindings(template, source)
    concrete = materialize_template(template, source, census.assignments[0])

    assert census.assignments == ((0,), (1,))
    assert concrete.input_atoms[0] == (2, 0, 2, 1)
    assert concrete.target_atoms[0] == (2, 0, 1, 2)
    assert template_features(template).ndim == 1
    assert np.isfinite(template_features(template)).all()


def test_delta_template_round_trip_is_address_free():
    template = StructuralDeltaTemplate(
        input_atoms=((2, 0, 3, 1),),
        input_bonds=((0,),),
        target_atoms=(None,),
        output_atoms=((3, 0, 2, 1),),
        target_bonds=((0, 0), (0, 0)),
    )

    payload = template.payload()
    assert StructuralDeltaTemplate.from_payload(payload) == template
    serialized = str(payload)
    for forbidden in ("route_id", "program_id", "source_group", "endpoint", "actions"):
        assert forbidden not in serialized


def test_marginal_policy_proposes_rebound_valid_structural_goal():
    source = _ethane()
    template = StructuralDeltaTemplate(
        input_atoms=((2, 0, 3, 1),),
        input_bonds=((0,),),
        target_atoms=((2, 0, 2, 2),),
        output_atoms=((4, 0, 1, 1),),
        target_bonds=((0, 1), (1, 0)),
    )
    marginal = fit_marginal_subgoal_policy(
        [
            {
                "source_group": "source-a",
                "templates": (template,),
            }
        ],
        exploration_floor=0.1,
    )

    candidates, telemetry = propose_structural_goals(
        source,
        (template,),
        marginal,
        ranker=None,
        pool_size=4,
        beam_width=2,
        expansion_width=2,
        max_bindings_per_template=2,
    )

    assert candidates
    assert all(row.goal.subgoals for row in candidates)
    assert telemetry["valid_unique_single_goals"] >= 1


def test_rewrite_scale_counts_complete_target_graph_changes():
    template = StructuralDeltaTemplate(
        input_atoms=((2, 0, 3, 1),),
        input_bonds=((0,),),
        target_atoms=((2, 0, 2, 2),),
        output_atoms=tuple((2, 0, 2, 2) for _ in range(6)),
        target_bonds=tuple(
            tuple(1 if abs(left - right) == 1 else 0 for right in range(7)) for left in range(7)
        ),
    )
    placeholder = StructuralSubgoal(
        input_atoms=((2, 0, 3, 1),),
        input_bonds=((0,),),
        environments=(environment(_ethane(), 0),),
        target_atoms=((2, 0, 2, 2),),
        output_atoms=(),
        target_bonds=((0,),),
    )
    proposal = ProposedStructuralGoal(
        StructuralGoal((placeholder,)),
        ((0,),),
        (template,),
        _ethane(),
        ("constituent",),
        0.0,
    )

    assert structural_rewrite_event_count(template) == 13
    assert proposal_rewrite_event_count(proposal) == 13
    assert proposal_rewrite_scale(proposal) == "large"


def test_scale_balanced_selection_preserves_large_rewrite_support():
    placeholder = StructuralSubgoal(
        input_atoms=((2, 0, 3, 1),),
        input_bonds=((0,),),
        environments=(environment(_ethane(), 0),),
        target_atoms=((2, 0, 2, 2),),
        output_atoms=(),
        target_bonds=((0,),),
    )

    def proposal(outputs: int, score: float) -> ProposedStructuralGoal:
        size = outputs + 1
        template = StructuralDeltaTemplate(
            input_atoms=((2, 0, 3, 1),),
            input_bonds=((0,),),
            target_atoms=((2, 0, 2, 2),),
            output_atoms=tuple((2, 0, 2, 2) for _ in range(outputs)),
            target_bonds=tuple(
                tuple(1 if abs(left - right) == 1 else 0 for right in range(size))
                for left in range(size)
            ),
        )
        return ProposedStructuralGoal(
            StructuralGoal((placeholder,)),
            ((0,),),
            (template,),
            _ethane(),
            (f"constituent-{outputs}",),
            score,
        )

    ranked = [proposal(0, 4.0), proposal(0, 3.0), proposal(2, 2.0), proposal(6, 1.0)]
    selected = select_scale_balanced_proposals(ranked, 3)

    assert [proposal_rewrite_scale(row) for row in selected] == [
        "local",
        "medium",
        "large",
    ]
