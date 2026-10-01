from types import SimpleNamespace

import numpy as np
import pytest

from compose_v4.experiments.hphi_region_features import input_dim
from compose_v4.experiments.qed_shared_rollouts import rollout_seed
from compose_v4.experiments.qed_shared_training import (
    value_examples,
    value_examples_with_bellman,
    value_examples_with_regions,
)


class TinyReference:
    config = SimpleNamespace(persistent_slots=48)

    def identity(self):
        return {"checkpoint_sha256": "shared-model"}

    def encode(self, _state):
        return np.zeros(256, dtype=np.float64)


def example_rollout():
    return {
        "schema_version": "compose.qed.shared_rollout.v1",
        "source_role": "train",
        "source_original": "C",
        "source_represented": "C",
        "reference": TinyReference().identity(),
        "configuration": {"horizon": 1, "replicates": 1},
        "trajectories": [
            {
                "replicate": 0,
                "seed": rollout_seed("C", 0),
                "status": "HORIZON",
                "path": [
                    {"smiles": "C", "qed": 0.3597849378839701, "similarity_to_source": 1.0},
                    {"smiles": "CC", "qed": 0.3727855551576051, "similarity_to_source": 0.0},
                ],
            }
        ],
    }


def test_value_examples_label_future_hit_without_training_on_boundary() -> None:
    features, labels = value_examples(
        TinyReference(), example_rollout(), budget_max=1, regions=((0.365, 0.0),)
    )
    assert features.shape == (1, input_dim(1))
    assert labels.tolist() == [1.0]
    assert features[0, -1] == 1.0
    _, _, region_indices = value_examples_with_regions(
        TinyReference(),
        example_rollout(),
        budget_max=1,
        regions=((0.9, 0.4), (0.365, 0.0)),
    )
    assert region_indices.tolist() == [0, 1, 0]


def test_value_examples_label_terminal_no_hit() -> None:
    rollout = example_rollout()
    rollout["trajectories"][0]["status"] = "TERMINAL"
    rollout["trajectories"][0]["path"] = rollout["trajectories"][0]["path"][:1]
    features, labels = value_examples(
        TinyReference(), rollout, budget_max=1, regions=((0.365, 0.0),)
    )
    assert features.shape[0] == 1
    assert labels.tolist() == [0.0]


def test_bellman_links_follow_same_region_and_exact_goal_boundary() -> None:
    values = value_examples_with_bellman(
        TinyReference(),
        example_rollout(),
        budget_max=1,
        regions=((0.9, 0.4), (0.365, 0.0)),
    )
    assert values.region_indices.tolist() == [0, 1, 0]
    assert values.labels.tolist() == [0.0, 1.0, 0.0]
    assert values.next_row_indices.tolist() == [2, -1, -1]
    assert values.next_in_region.tolist() == [0, 1, 0]


def test_value_examples_reject_unmatched_reference_or_test_role() -> None:
    rollout = example_rollout()
    rollout["reference"] = {"checkpoint_sha256": "other-model"}
    with pytest.raises(ValueError, match="another reference"):
        value_examples(TinyReference(), rollout, budget_max=1)
    rollout = example_rollout()
    rollout["source_role"] = "test"
    with pytest.raises(ValueError, match="training or validation"):
        value_examples(TinyReference(), rollout, budget_max=1)


def test_value_examples_reject_metric_drift() -> None:
    rollout = example_rollout()
    rollout["trajectories"][0]["path"][1]["qed"] = 0.99
    with pytest.raises(ValueError, match="metric drift"):
        value_examples(TinyReference(), rollout, budget_max=1, regions=((0.365, 0.0),))
