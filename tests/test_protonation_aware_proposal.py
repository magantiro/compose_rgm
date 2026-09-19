import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.protonation_aware_proposal import (
    ProtonationAwareProposalConfig,
    propose_protonation_aware_candidates,
)
from compose_v4.rewrite.kernel import canonical_state_key


def _disable_structural_lanes(monkeypatch):
    def no_shallow(*args, **kwargs):
        raise ValueError("fixture abstention")

    monkeypatch.setattr(
        "compose_v4.control.protonation_aware_proposal.synthesize_dynamic_program",
        no_shallow,
    )
    monkeypatch.setattr(
        "compose_v4.control.protonation_aware_proposal.enumerate_retained_core_prunes",
        lambda *args, **kwargs: (),
    )
    monkeypatch.setattr(
        "compose_v4.control.protonation_aware_proposal.propose_route_expert_candidates",
        lambda *args, **kwargs: ([], {"complete_programs_committed": 0}),
    )


def test_protonation_aware_expert_abstains_when_fiber_is_empty(monkeypatch):
    _disable_structural_lanes(monkeypatch)
    source = pad_molecular_graph(smiles_to_molecular_graph("CC"), 48)

    candidates, telemetry = propose_protonation_aware_candidates(
        source,
        object(),
        config=ProtonationAwareProposalConfig(shallow_draws=1),
    )

    assert candidates == []
    assert telemetry["protonation_actions_enumerated"] == 0
    assert telemetry["protonation_abstention"] is True
    assert telemetry["exact_execution_precision_denominator"] == 0


def test_protonation_aware_expert_exact_replays_charge_only_program(monkeypatch):
    _disable_structural_lanes(monkeypatch)
    source = pad_molecular_graph(smiles_to_molecular_graph("C[NH+](C)C"), 48)

    candidates, telemetry = propose_protonation_aware_candidates(
        source,
        object(),
        config=ProtonationAwareProposalConfig(shallow_draws=1),
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["program_kind"] == "charge_only"
    assert candidate["structural_lane"] == "charge_only"
    assert candidate["actions"][0]["executor_rule"] == "atom_protonation_restate"
    assert candidate["charge_transition"] == {
        "source_total_formal_charge": 1,
        "after_first_primitive_total_formal_charge": 0,
        "endpoint_total_formal_charge": 0,
    }
    assert telemetry["exact_execution_precision_numerator"] == 1
    assert telemetry["exact_execution_precision_denominator"] == 1
    assert telemetry["task_or_target_input_used"] is False
    assert telemetry["endpoint_or_teacher_input_used"] is False

    endpoint = pad_molecular_graph(smiles_to_molecular_graph(candidate["smiles"]), 48)
    assert canonical_state_key(endpoint) == candidate["endpoint_key"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"maximum_primitives": 31},
        {"maximum_active_atoms": 39},
        {"persistent_slots": 47},
        {"route_candidate_timeout_seconds": float("nan")},
    ],
)
def test_protonation_aware_config_rejects_support_changes(overrides):
    with pytest.raises(ValueError):
        ProtonationAwareProposalConfig(**overrides)
