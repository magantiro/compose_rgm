import multiprocessing
import os
import threading
from types import SimpleNamespace

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


def _patch_proposals(monkeypatch, source, goal_names):
    template = _expert().templates[0]
    proposals = [
        SimpleNamespace(
            goal=goal_name,
            bindings={},
            endpoint=source,
            templates=(template,),
            score=1.0 / rank,
        )
        for rank, goal_name in enumerate(goal_names, 1)
    ]
    monkeypatch.setattr(
        "compose_v4.control.route_distilled_goal_expert.propose_structural_goals",
        lambda *args, **kwargs: (proposals, {"proposals": len(proposals)}),
    )
    monkeypatch.setattr(
        "compose_v4.control.route_distilled_goal_expert.program_from_structural_goal",
        lambda goal: SimpleNamespace(program_id=goal),
    )
    monkeypatch.setattr(
        "compose_v4.control.route_distilled_goal_expert.decode_state",
        lambda state: state,
    )
    monkeypatch.setattr(
        "compose_v4.control.route_distilled_goal_expert.proposal_rewrite_event_count",
        lambda proposal: 1,
    )
    monkeypatch.setattr(
        "compose_v4.control.route_distilled_goal_expert.proposal_rewrite_scale",
        lambda proposal: "small",
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


def test_route_expert_abstains_from_invalid_composed_target(monkeypatch):
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph

    source = pad_molecular_graph(smiles_to_molecular_graph("CC"), 48)

    def reject(*args, **kwargs):
        raise ValueError("overlapping subgoals disagree on a target bond")

    monkeypatch.setattr(
        "compose_v4.control.route_distilled_goal_expert.execute_complete_region_program",
        reject,
    )
    candidates, telemetry = propose_route_expert_candidates(
        source,
        _expert(),
        pool_size=2,
        realization_limit=2,
    )

    assert candidates == []
    assert telemetry["realization_status_counts"] == {"invalid_composed_target": 1}


def test_route_expert_default_realization_stays_in_calling_process(monkeypatch):
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph

    source = pad_molecular_graph(smiles_to_molecular_graph("CC"), 48)
    _patch_proposals(monkeypatch, source, ("quick",))
    observed = []

    def execute(candidate_source, program, *, resolved_bindings, config):
        observed.append((os.getpid(), program.program_id, config.maximum_expansions))
        return {
            "status": "committed",
            "committed_endpoint_state": candidate_source,
            "realized_primitive_count": 1,
            "compiler_strategy": "fixture",
        }

    monkeypatch.setattr(
        "compose_v4.control.route_distilled_goal_expert.execute_complete_region_program",
        execute,
    )
    candidates, telemetry = propose_route_expert_candidates(
        source,
        _expert(),
        pool_size=1,
        realization_limit=1,
        maximum_expansions=17,
    )

    assert observed == [(os.getpid(), "quick", 17)]
    assert [row["route_program_id"] for row in candidates] == ["quick"]
    assert "per_candidate_timeout_seconds" not in telemetry
    assert telemetry["realization_status_counts"] == {"committed": 1}


def test_route_expert_discloses_realized_actions_only_when_requested(monkeypatch):
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph

    source = pad_molecular_graph(smiles_to_molecular_graph("CC"), 48)
    _patch_proposals(monkeypatch, source, ("quick",))
    realized_action = {"executor_rule": "fixture"}

    def execute(candidate_source, program, *, resolved_bindings, config):
        return {
            "status": "committed",
            "committed_endpoint_state": candidate_source,
            "realized_primitive_count": 1,
            "realized_actions": [realized_action],
            "compiler_strategy": "fixture",
        }

    monkeypatch.setattr(
        "compose_v4.control.route_distilled_goal_expert.execute_complete_region_program",
        execute,
    )
    candidates, telemetry = propose_route_expert_candidates(
        source,
        _expert(),
        pool_size=1,
        realization_limit=1,
        include_realized_actions=True,
    )

    assert candidates[0]["realized_actions"] == [realized_action]
    assert telemetry["realized_actions_included"] is True


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="the monkeypatched hanging worker fixture requires fork",
)
def test_route_expert_timeout_preserves_later_candidates(monkeypatch):
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph

    source = pad_molecular_graph(smiles_to_molecular_graph("CC"), 48)
    _patch_proposals(monkeypatch, source, ("before", "hang", "after"))
    existing_child_pids = {child.pid for child in multiprocessing.active_children()}

    def execute(candidate_source, program, *, resolved_bindings, config):
        if program.program_id == "hang":
            threading.Event().wait()
        return {
            "status": "committed",
            "committed_endpoint_state": candidate_source,
            "realized_primitive_count": 1,
            "compiler_strategy": "fixture",
        }

    monkeypatch.setattr(
        "compose_v4.control.route_distilled_goal_expert.execute_complete_region_program",
        execute,
    )
    candidates, telemetry = propose_route_expert_candidates(
        source,
        _expert(),
        pool_size=3,
        realization_limit=3,
        per_candidate_timeout_seconds=0.05,
    )

    assert [row["route_program_id"] for row in candidates] == ["before", "after"]
    assert [row["route_proposal_rank"] for row in candidates] == [1, 3]
    assert telemetry["realization_status_counts"] == {
        "committed": 2,
        "realizer_candidate_timeout": 1,
    }
    assert telemetry["complete_programs_committed"] == 2
    assert telemetry["exact_realization_precision_numerator"] == 2
    assert telemetry["exact_realization_precision_denominator"] == 2
    assert {child.pid for child in multiprocessing.active_children()} <= existing_child_pids


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("nan"), float("inf"), True])
def test_route_expert_rejects_invalid_candidate_timeout(timeout):
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph

    source = pad_molecular_graph(smiles_to_molecular_graph("CC"), 48)
    with pytest.raises(ValueError, match="positive and finite"):
        propose_route_expert_candidates(
            source,
            _expert(),
            pool_size=1,
            realization_limit=1,
            per_candidate_timeout_seconds=timeout,
        )
