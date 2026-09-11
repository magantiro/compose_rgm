import pytest

from compose_v4.control.continuation_choice import choose_continuation


def test_downstream_value_changes_choice_without_new_oracle_or_support():
    first = [{"score": x} for x in (0.7, 0.2, 0.1, 0.1)]
    second = [[{"score": x}] * 4 for x in (0.7, 0.95, 0.1, 0.1)]
    result = choose_continuation(first, second, seed=0, snapshot_id="fixed")
    a, b = (result["decisions"][k] for k in ("immediate", "future"))
    assert a["selected"] == 0 and b["selected"] == 1
    assert a["best_witness"] == [0.7, 0.2, 0.1, 0.1]
    assert b["best_witness"] == [0.7, 0.95, 0.1, 0.1]
    assert a["reference"] == b["reference"] == [0.25] * 4
    assert a["acting_uniform"] == b["acting_uniform"]
    for decision in (a, b):
        assert decision["kl"] <= 1 + 1e-10
        assert min(decision["probabilities"]) >= 0.025


def test_failure_is_not_a_fictitious_molecule_and_bad_scores_fail():
    result = choose_continuation(
        [None] * 4, [[None] * 4 for _ in range(4)], seed=0, snapshot_id="failures"
    )
    assert all(d["abstained"] for d in result["decisions"].values())
    with pytest.raises(ValueError, match="failed first"):
        choose_continuation(
            [None] * 4, [[{"score": 1}] * 4 for _ in range(4)], seed=0, snapshot_id="invalid"
        )
    with pytest.raises(ValueError, match="finite"):
        choose_continuation(
            [{"score": float("nan")}] * 4,
            [[None] * 4 for _ in range(4)],
            seed=0,
            snapshot_id="invalid",
        )


def test_locked_three_round_adapter_preserves_draws_and_reuses_identical_choices(tmp_path):
    from pathlib import Path

    from compose_v4.experiments.pmo_archive_pilot import Store
    from compose_v4.experiments.pmo_continuation_choice import draw_slots, run_comparison

    store = Store(tmp_path, lambda: None)
    tasks_seen = []

    def parallel(tasks):
        tasks_seen.extend(tasks)
        for task in reversed(tasks):  # Deliberately unordered remote completion.
            key = task["worker_id"]
            yield {
                "worker_id": key,
                "phase": task["phase"],
                "slot": task["slot"],
                "replay_verified": True,
                "seconds": 0,
                "initialization_seconds": 0,
                "executor_calls": 0,
                "law_work": {"fresh_laws": 0},
                "attempts": [
                    {"draw": i, "status": "complete", "bundle": {"option": "generic"}}
                    for i in range(4)
                ],
                "candidates": [
                    {
                        "id": f"workers/{key}/draws/{i:02}",
                        "smiles": "C" * (i + 1) if task["phase"] == 1 else task["parent"]["smiles"],
                        "node": {"exact_fixture": i},
                    }
                    for i in range(4)
                ],
            }

    class Scores:
        def __init__(self):
            self.context = {}
            self.requests = []

        def score(self, smiles):
            assert Path(self.context["lock_path"]).exists()
            self.requests.append(smiles)
            return {"desirability": len(smiles) / 10}

    scores = Scores()
    data = {
        "parents": [{"id": str(i), "score": 0.1, "development_source": str(i)} for i in range(5)],
        "observed": {"C": 0.1},
    }
    contract = {"seed": 20260918, "arms": ["immediate", "future"]}
    result = run_comparison({}, contract, data, store, scores, {}, parallel)
    assert result["changed_choices"] == 0
    assert result["workers"] == 30  # 5 + 20 + 5, not duplicate fresh workers per arm.
    assert result["logical_planning_attempts"] == 100
    assert result["physical_fresh_attempts"] == 20
    assert "C" not in scores.requests  # Historical label reuse, no hidden rescore.
    assert result["fresh_paired_differences"] == [0] * 5
    assert [t["phase"] for t in tasks_seen].count(3) == 5
    bad = {
        "worker_id": "bad",
        "attempts": [{"draw": i, "status": "complete"} for i in range(4)],
        "candidates": [],
    }
    with pytest.raises(ValueError, match="missing"):
        draw_slots(bad)
