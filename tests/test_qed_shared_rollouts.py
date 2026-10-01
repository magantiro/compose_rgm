from types import SimpleNamespace

import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.qed_shared_rollouts import QEDRolloutConfig, rollout_source


class TinyReference:
    reference = SimpleNamespace(max_active_atoms=40)
    config = SimpleNamespace(persistent_slots=48)

    def identity(self) -> dict[str, str]:
        return {"checkpoint_sha256": "shared-model"}

    def sample(self, state, _rng):
        if state.n_real_atoms == 4:
            return pad_molecular_graph(smiles_to_molecular_graph("CCO"), 48)
        return None


def test_rollout_preserves_original_source_and_resets_slots() -> None:
    source = "C[C@@H](O)F"
    config = QEDRolloutConfig(horizon=3, replicates=2)
    result = rollout_source(TinyReference(), source, 7, config)
    assert result == rollout_source(TinyReference(), source, 7, config)
    assert result["source_original"] == source
    assert "@" not in result["source_represented"]
    assert len(result["trajectories"]) == 2
    for trajectory in result["trajectories"]:
        assert trajectory["status"] == "TERMINAL"
        assert len(trajectory["path"]) == 2
        assert trajectory["path"][0]["similarity_to_source"] == 1.0


def test_rollout_configuration_rejects_empty_horizon() -> None:
    with pytest.raises(ValueError, match="horizon"):
        QEDRolloutConfig(horizon=0, replicates=1)
