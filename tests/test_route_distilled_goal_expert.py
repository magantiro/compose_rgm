import pytest

from compose_v4.control.route_distilled_goal_expert import (
    RouteDistilledGoalExpert,
    make_route_expert,
    propose_route_expert_candidates,
)
from compose_v4.control.structural_subgoal_policy import (
    MarginalSubgoalPolicy,
    StructuralDeltaTemplate,
)


def _expert():
    template = StructuralDeltaTemplate(
        input_atoms=((2, 0, 3, 1),),
        input_bonds=((0,),),
        target_atoms=((3, 0, 2, 1),),
        output_atoms=(),
        target_bonds=((0,),),
    )
    marginal = MarginalSubgoalPolicy(
        (template.template_id,),
        (1.0,),
        (0.97, 0.01, 0.01, 0.01),
        0.1,
        "marginal-fit",
    )
    return make_route_expert(
        (template,), marginal, training_evidence_identity="split-first-training"
    )


def test_route_expert_checkpoint_round_trip_has_no_route_payload():
    expert = _expert()
    payload = expert.checkpoint()
    assert RouteDistilledGoalExpert.from_checkpoint(payload) == expert
    serialized = str(payload)
    assert "route_id" not in serialized
    assert "source_group" not in serialized
    assert "endpoint" not in serialized


def test_route_expert_rejects_realization_limit_outside_pool():
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph

    source = pad_molecular_graph(smiles_to_molecular_graph("CC"), 48)
    with pytest.raises(ValueError, match="within the pool"):
        propose_route_expert_candidates(source, _expert(), pool_size=2, realization_limit=3)
