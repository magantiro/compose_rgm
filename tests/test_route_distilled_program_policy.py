from __future__ import annotations

import numpy as np
import pytest

from compose_v4.control.option_demonstrations import descriptor_menu
from compose_v4.control.option_features import structural_option_features
from compose_v4.control.option_policy import AdvantageWeightedOptionActor
from compose_v4.control.route_distilled_program_policy import (
    CHECKPOINT_SCHEMA,
    GENERIC_MODULES,
    STAGE_DESCRIPTOR_NAMES,
    RouteDistilledProgramPolicy,
    default_reference,
    stage_descriptor,
    synthesize_route_distilled_program,
)
from compose_v4.control.trajectory_value import molecule_features
from compose_v4.experiments.editing_v2_evaluation_semantics import (
    production_state_from_smiles,
)


def _checkpoint():
    source = production_state_from_smiles("CCO", max_atoms=48)
    state_dim = len(molecule_features("CCO"))
    option_dim = len(structural_option_features("generic"))
    actor = AdvantageWeightedOptionActor(state_dim, option_dim, hidden=8)
    for parameter in actor.parameters():
        parameter.data.zero_()
    return source, {
        "schema_version": CHECKPOINT_SCHEMA,
        "evidence": "fixture",
        "state_dim": state_dim,
        "option_dim": option_dim,
        "hidden": 8,
        "parameters": {
            name: value.detach().tolist()
            for name, value in sorted(actor.state_dict().items())
        },
        "parameter_identity": "fixture",
        "options": list(descriptor_menu()),
        "reference": list(default_reference()),
        "exploration_floor": 0.1,
        "module_count_probabilities": [0.8, 0.15, 0.05],
        "binding_prototypes": [
            {
                "family": family,
                "mean": [0.0] * len(STAGE_DESCRIPTOR_NAMES),
                "scale": [1.0] * len(STAGE_DESCRIPTOR_NAMES),
            }
            for family in sorted(GENERIC_MODULES)
        ],
        "training_identity": "fixture-training",
    }


def test_checkpoint_is_target_free_normalized_and_reconstructable():
    source, checkpoint = _checkpoint()
    policy = RouteDistilledProgramPolicy.from_checkpoint(checkpoint)
    probabilities = policy.option_probabilities(source)

    assert probabilities == pytest.approx(default_reference())
    assert sum(policy.family_probabilities(source).values()) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="forbidden teacher field"):
        RouteDistilledProgramPolicy.from_checkpoint({**checkpoint, "target": "jak2"})


def test_stage_descriptor_and_distilled_program_exact_replay():
    source, checkpoint = _checkpoint()
    policy = RouteDistilledProgramPolicy.from_checkpoint(checkpoint)
    descriptor = stage_descriptor(
        source,
        [
            {
                "executor_rule": "atom_insert",
                "payload": {
                    "slot": 3,
                    "atom_type": 2,
                    "neighbors": [[1, 1]],
                },
            }
        ],
    )
    assert descriptor.shape == (len(STAGE_DESCRIPTOR_NAMES),)
    assert np.isfinite(descriptor).all()
    _, program, _, trace, metadata = synthesize_route_distilled_program(
        source,
        np.random.default_rng(7),
        policy,
        max_modules=1,
    )

    assert trace["complete"]
    assert trace["primitive_edits"] == len(program.marks)
    assert metadata["runtime_teacher_lookup"] is False
    assert metadata["initial_stored_complete_routes"] == 0
    assert metadata["completed_module_count"] == 1
