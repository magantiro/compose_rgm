from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.source_prior import (
    DegreeBoundedCarbonTreePrior,
    FixedMolecularStatePrior,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.calibrated_rewrite_sampling import (
    ThinnedRateCalibrationSampler,
    ring_system_grow_contains_small_ring,
    ring_system_grow_cycle_sizes,
)
from compose_v4.experiments.tracelet_conditional import sample_tracelet_ancestral
from compose_v4.model.factorized_tracelet_rate_model import SampledRewriteMark
from compose_v4.rewrite.operators import AtomDelete
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.tracelets import RingSystemGrow
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target


def _ring_case(smiles: str) -> tuple[object, RingSystemGrow]:
    target = pad_molecular_graph(smiles_to_molecular_graph(smiles), 12)
    source = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,)).sample(
        np.random.default_rng(409),
        n_slots=12,
    )
    trace = compile_carbon_tree_to_target(
        source,
        target,
        use_bond_reroute=True,
        align_source=True,
    )
    path = TraceProgressCTMC(trace)
    progress, step = next(
        (index, item)
        for index, item in enumerate(trace.steps)
        if isinstance(item.action, RingSystemGrow)
    )
    return path.state_at(progress), step.action


@dataclass
class _FixedSampler:
    sampled: SampledRewriteMark

    def sample_rewrite_mark(self, state, time, rng):
        del state, time, rng
        return self.sampled


@dataclass
class _VirtualThenTerminalSampler:
    calls: int = 0

    def sample_rewrite_mark(self, state, time, rng):
        del state, time, rng
        self.calls += 1
        if self.calls == 1:
            return SampledRewriteMark(
                1.0e9,
                "<VIRTUAL_RATE_CALIBRATION:atom_delete>",
                None,
            )
        return SampledRewriteMark(0.0, "<TERMINAL>", None)


def test_ring_action_cycle_size_audit_distinguishes_small_and_common_rings() -> None:
    small_state, small_action = _ring_case("C1CC1")
    common_state, common_action = _ring_case("C1CCCCC1")

    assert ring_system_grow_cycle_sizes(small_state, small_action) == (3,)
    assert ring_system_grow_cycle_sizes(common_state, common_action) == (6,)
    assert ring_system_grow_contains_small_ring(small_state, small_action)
    assert not ring_system_grow_contains_small_ring(common_state, common_action)


def test_small_ring_thinning_rejects_only_the_targeted_ring_rate() -> None:
    small_state, small_action = _ring_case("C1CC1")
    common_state, common_action = _ring_case("C1CCCCC1")
    calibration = dict(small_ring_log_rate_adjustment=-100.0)

    small = ThinnedRateCalibrationSampler(
        _FixedSampler(SampledRewriteMark(2.0, "ring_system_grow", small_action)),
        **calibration,
    ).sample_rewrite_mark(small_state, 0.5, np.random.default_rng(1))
    common = ThinnedRateCalibrationSampler(
        _FixedSampler(SampledRewriteMark(2.0, "ring_system_grow", common_action)),
        **calibration,
    ).sample_rewrite_mark(common_state, 0.5, np.random.default_rng(1))

    assert small.rule_name == "<VIRTUAL_RATE_CALIBRATION:small_ring>"
    assert small.action is None
    assert small.total_hazard == pytest.approx(2.0)
    assert common.rule_name == "ring_system_grow"
    assert common.action is common_action


def test_family_thinning_scales_delete_without_renormalizing_the_clock() -> None:
    state = pad_molecular_graph(smiles_to_molecular_graph("CC"), 4)
    calibration = ThinnedRateCalibrationSampler(
        _FixedSampler(SampledRewriteMark(3.0, "atom_delete", AtomDelete(1))),
        family_log_rate_adjustments=(("atom_delete", -100.0),),
    )
    sampled = calibration.sample_rewrite_mark(
        state,
        0.5,
        np.random.default_rng(2),
    )

    assert sampled.rule_name == "<VIRTUAL_RATE_CALIBRATION:atom_delete>"
    assert sampled.action is None
    assert sampled.total_hazard == pytest.approx(3.0)


def test_positive_rate_adjustments_are_rejected() -> None:
    sampler = _FixedSampler(SampledRewriteMark(0.0, "<TERMINAL>", None))
    with pytest.raises(ValueError, match="non-positive"):
        ThinnedRateCalibrationSampler(
            sampler,
            family_log_rate_adjustments=(("atom_insert", 0.1),),
        )
    with pytest.raises(ValueError, match="non-positive"):
        ThinnedRateCalibrationSampler(
            sampler,
            small_ring_log_rate_adjustment=0.1,
        )


def test_ancestral_sampler_records_generic_virtual_calibration_jumps() -> None:
    source = pad_molecular_graph(smiles_to_molecular_graph("CC"), 4)
    rollout = sample_tracelet_ancestral(
        _VirtualThenTerminalSampler(),
        rng=np.random.default_rng(3),
        n_slots=4,
        operational_horizon=1.0,
        time_step=0.1,
        max_events=4,
        source_prior=FixedMolecularStatePrior(source),
    )

    assert rollout.event_rules == ()
    assert rollout.virtual_event_rules == (
        "<VIRTUAL_RATE_CALIBRATION:atom_delete>",
    )

