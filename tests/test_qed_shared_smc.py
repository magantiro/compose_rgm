from types import SimpleNamespace

import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.qed_shared_smc import QEDSMCConfig, run_source


class TinyBoundValue:
    def properties(self, state):
        return (1.0, 1.0) if state.n_real_atoms == 2 else (0.5, 1.0)

    def in_target(self, state, _region):
        return state.n_real_atoms == 2

    def value(self, state, budget, _region):
        if self.in_target(state, _region):
            return 1.0
        return 0.0 if budget == 0 else 0.5


class TinyHead:
    budget_max = 1

    def __init__(self):
        self.metadata = {"source_split_sha256": "split-hash"}

    def for_source(self, _source):
        return TinyBoundValue()


class TinyReference:
    reference = SimpleNamespace(max_active_atoms=40)
    config = SimpleNamespace(persistent_slots=48)

    def __init__(self, *, terminal: bool = False):
        self.terminal = terminal

    def identity(self):
        return {"checkpoint_sha256": "shared-model"}

    def sample(self, _state, _rng):
        if self.terminal:
            return None
        return pad_molecular_graph(smiles_to_molecular_graph("CC"), 48)


def test_qed_smc_returns_terminal_successes_without_ranking_particles() -> None:
    config = QEDSMCConfig(horizon=1, particles=3, candidates=2)
    result = run_source(TinyReference(), TinyHead(), "C", config)
    assert result == run_source(TinyReference(), TinyHead(), "C", config)
    assert result["success"]
    assert len(result["candidates"]) == 2
    assert all(candidate["status"] == "OK" for candidate in result["candidates"])
    assert all(candidate["smiles"] == "CC" for candidate in result["candidates"])


def test_qed_smc_reports_extinction_as_a_failed_output_slot() -> None:
    result = run_source(
        TinyReference(terminal=True),
        TinyHead(),
        "C",
        QEDSMCConfig(horizon=1, particles=3, candidates=2),
    )
    assert not result["success"]
    assert all(candidate["status"] == "EXTINCT_NO_HIT" for candidate in result["candidates"])
    assert all(candidate["smiles"] == "C" for candidate in result["candidates"])
    assert all(not candidate["success"] for candidate in result["candidates"])


def test_qed_smc_rejects_budget_or_threshold_errors() -> None:
    with pytest.raises(ValueError, match="thresholds"):
        QEDSMCConfig(qed_minimum=float("nan"))
    with pytest.raises(ValueError, match="fitted value-head budget"):
        run_source(TinyReference(), TinyHead(), "C", QEDSMCConfig(horizon=2))
