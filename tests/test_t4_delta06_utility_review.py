from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from compose_v4.experiments.t4_delta06_utility_launch import publish_once
from compose_v4.experiments.t4_delta06_utility_review import (
    RESULT_PATH,
    build_review,
)

ROOT = Path(__file__).resolve().parents[1]


def test_review_maps_every_locked_membership_and_preserves_negative_ivg_result() -> (
    None
):
    review = build_review(ROOT, code_revision="test-revision")

    assert review["status"] == "complete_negative_ivg_comparison"
    assert review["run"]["requests"] == 19
    assert review["run"]["memberships"] == 22
    assert review["run"]["failed_scores"] == 0
    summary = review["summary"]
    assert summary["cells_with_scores"] == 8
    assert summary["cell_abstentions"] == 7
    assert summary["policy_cells_with_scores"] == 13
    assert summary["policy_cell_published_source_reference_outcomes"] == {
        "better": 9,
        "tie": 3,
        "worse": 1,
        "abstain": 17,
    }
    assert summary["policy_cell_ivg_reported_mean_outcomes"] == {
        "better": 0,
        "tie": 0,
        "worse": 13,
        "abstain": 17,
    }
    assert summary["scored_cell_best_published_source_reference_outcomes"] == {
        "better": 6,
        "tie": 1,
        "worse": 1,
        "abstain": 0,
    }


def test_review_records_exact_cell_policy_margins() -> None:
    review = build_review(ROOT, code_revision="test-revision")
    rows = {
        (row["cell"], row["policy_id"]): row for row in review["cell_policy_results"]
    }

    braf = rows[("braf_2", "source_balanced_marginal_subgoal")]
    assert braf["best_locked_score"] == -10.4
    assert braf["gain_vs_published_source_reference"] == pytest.approx(0.6)
    assert braf["gain_vs_ivg_reported_mean"] == pytest.approx(-1 / 3)

    parp = rows[("parp1_0", "source_balanced_marginal_subgoal")]
    assert parp["best_locked_score"] == -11.4
    assert parp["gain_vs_published_source_reference"] == pytest.approx(4.1)
    assert parp["gain_vs_ivg_reported_mean"] == pytest.approx(-13 / 6)

    abstain = rows[("fa7_0", "graph_conditioned_subgoal")]
    assert abstain["best_locked_score"] is None
    assert abstain["published_source_reference_outcome"] == "abstain"


def test_review_rejects_tampered_or_incomplete_scored_result(tmp_path: Path) -> None:
    for relative in (
        "configs",
        "docs",
        "modal_apps",
        "diagnostics/t4_delta06_structural_subgoal_utility_lock/attempt_1",
        "diagnostics/t4_delta06_structural_subgoal_utility_launch/attempt_1",
    ):
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)

    required = (
        "configs/t4_frozen_program_benchmark_v2.json",
        "docs/GENMOL_T4_SEEDS.json",
        "modal_apps/genmol_t4_opt_app.py",
        "diagnostics/t4_delta06_structural_subgoal_utility_lock/attempt_1/abstention_ledger.json",
        "diagnostics/t4_delta06_structural_subgoal_utility_launch_spawn.json",
        "diagnostics/t4_delta06_structural_subgoal_utility_reduce_spawn.json",
    )
    for relative in required:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())

    original = json.loads((ROOT / RESULT_PATH).read_text())["payload"]
    incomplete = copy.deepcopy(original)
    incomplete["status"] = "incomplete_no_retry"
    publish_once(tmp_path / RESULT_PATH, incomplete)
    with pytest.raises(ValueError, match="one-shot launch boundary"):
        build_review(tmp_path, code_revision="test-revision")
