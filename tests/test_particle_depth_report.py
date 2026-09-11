import pytest

from compose_v4.control.local_endpoint_selector import policy_identity
from tools.pmo_option_particle_report import proposal_depths, slot_policy_identity


def test_distinct_starts_with_shared_empty_chain_and_resampled_descendants():
    parents = [{"primitives": 23, "chain": []}, {"primitives": 0, "chain": []}]
    children = [
        {"id": "a", "primitives": 24, "primitive_count": 1, "chain": ["a"]},
        {"id": "b", "primitives": 5, "primitive_count": 5, "chain": ["b"]},
    ]
    assert proposal_depths(parents, children, [0, 0]) == [1, 5]
    next_child = {"id": "c", "primitives": 8, "primitive_count": 3, "chain": ["b", "c"]}
    assert proposal_depths([children[1], None], [next_child, None], [5, None]) == [8, None]


def test_depth_rejects_bad_parent_count():
    with pytest.raises(ValueError, match="exact parent"):
        proposal_depths(
            [{"primitives": 23, "chain": []}],
            [{"id": "a", "primitives": 21, "primitive_count": 1, "chain": ["a"]}],
            [0],
        )


def test_report_keeps_local_channel_workers_distinct_and_legacy_unchanged():
    config = {
        "seed": 20261008,
        "local_selector_arms": ["guided"],
        "endpoint_model_sha256": "frozen-model",
    }
    # The frozen experiment draws local at round 1 / slot 2, not slot 0.
    assert slot_policy_identity(config, "guided", 1, 2, "broad", "memory") == policy_identity(
        "memory", "frozen-model"
    )
    assert slot_policy_identity(config, "guided", 1, 0, "broad", "memory") == "broad"
    assert slot_policy_identity(config, "baseline", 1, 2, "broad", "memory") == "broad"
    assert slot_policy_identity({}, "legacy", 1, 2, None) is None


def test_report_rejects_local_identity_without_memory():
    config = {"seed": 20261008, "local_selector_arms": ["guided"]}
    with pytest.raises(ValueError, match="frozen donor memory"):
        slot_policy_identity(config, "guided", 1, 2, "broad")
