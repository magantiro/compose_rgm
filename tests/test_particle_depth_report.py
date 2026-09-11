import pytest

from tools.pmo_option_particle_report import proposal_depths


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
