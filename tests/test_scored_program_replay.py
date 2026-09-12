"""Source/endpoint balance for deterministic scored-program representation reuse."""

from collections import defaultdict

import pytest

from tools.prepare_scored_program_replay import normalize_endpoint_mass


def test_more_representations_do_not_multiply_source_or_endpoint_mass():
    data = [
        {"source_group": s, "endpoint": e}
        for s, e in (("a", "one"), ("a", "one"), ("a", "two"), ("b", "three"))
    ]
    weighted = normalize_endpoint_mass(data)
    totals = defaultdict(float)
    for row in weighted:
        totals[row["source_group"], row["endpoint"]] += row["structural_replay_weight"]
    assert totals == pytest.approx({("a", "one"): 0.25, ("a", "two"): 0.25, ("b", "three"): 0.5})
    assert normalize_endpoint_mass([]) == []
