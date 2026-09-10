import pytest

from diagnostics.t4_route_diagnosis.analyze import flatten


def test_flatten_preserves_every_state_and_boundary():
    stages = [
        {"name": "first", "states": [0, 1, 2], "actions": ["a", "b"], "primitive_edits": 2},
        {"name": "second", "states": [2, 3], "actions": ["c"], "primitive_edits": 1},
    ]
    states, boundaries = flatten(0, stages)
    assert states == [0, 1, 2, 3]
    assert [b["step"] for b in boundaries] == [2, 3]


@pytest.mark.parametrize("states,actions", [([1, 2], ["a"]), ([0, 1, 2], ["a"])])
def test_flatten_rejects_discontinuity_or_missing_actions(states, actions):
    with pytest.raises(ValueError):
        flatten(0, [{"name": "broken", "states": states, "actions": actions, "primitive_edits": 1}])
