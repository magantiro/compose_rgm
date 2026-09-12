import numpy as np
import pytest
import torch

from compose_v4.control.improvement_value import MonotoneImprovementModel
from compose_v4.control.option_controller import OptionBoundaryController
from compose_v4.control.option_features import (
    option_feature_names,
    option_state_features,
    structural_option_features,
)
from compose_v4.control.option_policy import AdvantageWeightedOptionActor
from compose_v4.control.region_replacement import PREFIX as REPLACEMENT_PREFIX
from compose_v4.control.ring_program import RingSpec, default_ring_options


def test_structural_features_are_compositional_not_default_menu_membership():
    custom = RingSpec("pendant", 6, (4, 2, 0), "aromatic", 2).option
    assert custom not in default_ring_options()
    custom_features = structural_option_features(custom)
    replacement = structural_option_features(REPLACEMENT_PREFIX + custom)
    generic = structural_option_features("generic")
    assert custom_features.shape == (len(option_feature_names()),)
    assert np.isfinite(custom_features).all()
    assert not np.array_equal(custom_features, replacement)
    assert not np.array_equal(custom_features, generic)


def test_option_state_features_include_local_global_and_option_clock():
    values = option_state_features(
        [1, 0, 1],
        current_score=0.4,
        incumbent=0.5,
        remaining_options=6,
        released_fraction=0.8,
        region_size=24,
        total_atoms=30,
        boundary_bonds=3,
        context_components=2,
        region_has_ring=True,
        ring_boundary_bonds=1,
    )
    assert values.shape == (12,)
    assert values[-6] == pytest.approx(0.8)


def test_integrated_controller_records_b_over_q_and_exact_terminal_boundary():
    option_dim = len(option_feature_names())
    actor = AdvantageWeightedOptionActor(2, option_dim, hidden=4)
    value = MonotoneImprovementModel(2, 2, 2, hidden=4)
    for model in (actor, value):
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
    controller = OptionBoundaryController(
        actor,
        value,
        actor_snapshot="actor-a",
        value_snapshot="value-a",
        horizons=(1, 2),
        thresholds=(0.01, 0.1),
    )
    selected = controller.decide(
        ("generic", "cyclize"), (0.7, 0.3), [0.2, 0.4], np.random.default_rng(2)
    )
    assert selected.reference_probability == pytest.approx(selected.proposal_probability)
    assert selected.controller_snapshot == "actor-a:value-a"
    intermediate = controller.log_potential([0.2, 0.4], remaining_options=2, incumbent=0.5)
    assert intermediate > 5
    assert controller.log_potential(
        [0.2, 0.4], remaining_options=0, incumbent=0.5, terminal_best=0.7
    ) == pytest.approx(7)
    with pytest.raises(ValueError, match="registered horizon"):
        controller.log_potential([0.2, 0.4], remaining_options=3, incumbent=0.5)


def test_controller_rejects_unbound_feature_dimensions():
    actor = AdvantageWeightedOptionActor(3, len(option_feature_names()), hidden=4)
    value = MonotoneImprovementModel(2, 1, 1, hidden=4)
    with pytest.raises(ValueError, match="configuration"):
        OptionBoundaryController(
            actor,
            value,
            actor_snapshot="actor-a",
            value_snapshot="value-a",
            horizons=(1,),
            thresholds=(0.1,),
        )
