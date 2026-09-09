"""Small diagnostic witnesses, not a controller-quality or release gate."""

import pytest

from compose_v4.experiments.winner_paths import PathConfig, find_path, replay


@pytest.mark.parametrize(
    "source,target",
    [
        ("CCO", "OCC"),
        ("CC", "CCC1CCCCC1"),
        ("c1ccccc1", "c1ccc2ccccc2c1"),
        ("C1CCCCC1", "C1CCCCCC1"),
    ],
)
def test_exact_replayed_witness(source, target):
    result = find_path(source, target, PathConfig())
    assert result["status"] == "witness_found"
    assert (
        replay(result["source_state"], result["actions"], result["target_2d"]) == result["states"]
    )
    assert result["primitive_lower_bound"] <= result["witness_steps"]
    assert all(mark["schema_version"] == 4 for mark in result["actions"])
    if source == "CCO":
        assert result["witness_steps"] == 0
    else:
        with pytest.raises(ValueError, match="endpoint"):
            replay(result["source_state"], result["actions"][:-1], result["target_2d"])


def test_support_and_abstention():
    import json

    import numpy as np

    from tools.ivg_winner_paths import canonical

    assert find_path("CCN", "CC[NH3+]", PathConfig())["status"] == "unreachable_charge_change"
    assert find_path("C", "C" * 49, PathConfig())["status"] == "unsupported_size"
    limited = find_path("CC", "CCC1CCCCC1", PathConfig(max_expansions=1))
    assert limited["status"] == "search_unresolved"
    assert all(attempt["expanded"] == 1 for attempt in limited["attempts"])
    assert json.loads(canonical({"residual": np.int64(3)})) == {"residual": 3}
    assert json.loads(canonical(limited))["status"] == "search_unresolved"


def test_config_refuses_support_changes():
    with pytest.raises(ValueError, match="support"):
        PathConfig(max_active=41)
    with pytest.raises(ValueError, match="positive"):
        PathConfig(max_expansions=0)


def test_annotation_and_census_identity():
    from compose_v4.control.region import enumerate_regions
    from compose_v4.experiments.t4_warm_continuation import canonical_slots, exact_context
    from compose_v4.rewrite.kernel import canonical_state_key
    from compose_v4.rewrite.trace_shard import decode_state
    from tools.ivg_winner_paths import annotate, mapped_context, pairs_from_census

    source, target = "CC", "CCC1CCCCC1"
    winner = {"canonical_smiles": target, "source_rows": [{"line": 2}]}
    cell = {
        "target": "parp1",
        "source_idx": 0,
        "source_smiles": source,
        "cell": "docking_parp1_idx0_thr4",
        "delta": 0.4,
        "runs": [{"run_seed": 0, "reported_docking_score": 1, "winners": [winner]}],
    }
    pairs = pairs_from_census(
        {"cells": [cell, {**cell, "delta": 0.6}]}, [{"target": "parp1", "smiles": source}]
    )
    assert len(pairs) == 1 and len(pairs[0]["references"]) == 2
    path = find_path(source, target, PathConfig())
    rows = annotate(pairs[0], path, None)
    assert len(rows) == path["witness_steps"] + 1
    assert rows[-1]["cycle_rank"] == 1
    assert all("value_abstention" in row for row in rows)
    assert all(
        row["next_edit"]["learned_mark_support_and_probability"] == "not_evaluated"
        for row in rows[:-1]
    )
    graph = decode_state(path["states"][-1])
    smiles = canonical_state_key(graph)
    mapping = canonical_slots(graph, smiles)
    for region in enumerate_regions(smiles):
        assert mapped_context(region, mapping) == exact_context(graph, smiles, region)
