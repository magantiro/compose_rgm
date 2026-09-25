from __future__ import annotations

import pytest

from compose_v4.benchmark.fragment_common_panel_ablation import (
    select_common_panel,
    supported_panel,
)


def _offers():
    rows = [{"draw": i, "status": "compiler_or_constraint_abstention"} for i in range(8)]
    rows[0] = {"draw": 0, "status": "model_supported", "endpoint": "C", "mean_log_mark": -1.0}
    rows[1] = {"draw": 1, "status": "model_supported", "endpoint": "CC", "mean_log_mark": -7.0}
    return rows


def test_common_support_is_invariant_to_finite_score_changes():
    first = _offers()
    changed = _offers()
    changed[0]["mean_log_mark"] = -100.0
    changed[1]["mean_log_mark"] = 100.0
    assert [endpoint for endpoint, _ in supported_panel(first)] == [
        endpoint for endpoint, _ in supported_panel(changed)
    ]
    for i in range(200):
        key = f"paired-{i}"
        assert select_common_panel(first, law="uniform", seed_material=key) == select_common_panel(
            changed, law="uniform", seed_material=key
        )


def test_learned_and_uniform_sample_only_common_support():
    panel = _offers()
    for law in ("learned", "uniform"):
        assert {
            select_common_panel(panel, law=law, seed_material=str(i)) for i in range(100)
        } <= {"C", "CC"}


def test_no_output_and_malformed_support_fail_closed():
    empty = [{"draw": i, "status": "compiler_or_constraint_abstention"} for i in range(8)]
    assert select_common_panel(empty, law="uniform", seed_material="x") is None
    malformed = _offers()
    malformed[1]["mean_log_mark"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        supported_panel(malformed)
