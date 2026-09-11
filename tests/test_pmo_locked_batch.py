import pytest

from compose_v4.benchmark.molleo_task3 import BudgetExceeded
from compose_v4.experiments.pmo_macro_probe import DurableScores
from compose_v4.experiments.t4_matched_pilot import seal, unseal


def make_scores(path, oracle, commit, budget=3):
    lock = path / "lock.json"
    seal(lock, {"smiles": ["CCO", "CCN", "CCC"]})
    scores = DurableScores(path, oracle, budget, commit, {})
    scores.context = {"role": "locked_test", "lock_path": str(lock)}
    return scores


def test_locked_batch_matches_serial_order_values_accounting_and_restore(tmp_path):
    molecules = ["OCC", "CCN", "CCO", "CCC"]
    expected = {"CCO": 0.4, "CCN": 0.6, "CCC": 0.2}
    runs = {}
    for mode in ("serial", "batch"):
        folder = tmp_path / mode
        calls, barriers = [], []

        def commit(barriers=barriers, folder=folder, calls=calls):
            barriers.append(
                (
                    len(list(folder.glob("oracle/*/started.json"))),
                    len(list(folder.glob("oracle/*/result.json"))),
                    len(calls),
                )
            )

        def oracle(s, mode=mode, barriers=barriers, calls=calls):
            if mode == "batch":
                assert barriers[0] == (3, 0, 0)
            calls.append(s)
            return expected[s]

        scores = make_scores(folder, oracle, commit)
        values = (
            scores.score_many(molecules)
            if mode == "batch"
            else [scores.score(s) for s in molecules]
        )
        runs[mode] = (values, list(calls), len(barriers), scores.meter.evaluated())
        assert scores.meter.spent == 3 and scores.meter.n_primed == 0
        before = len(barriers)
        assert scores.score_many(molecules) == values
        assert len(barriers) == before
        restored = DurableScores(
            folder, lambda s: pytest.fail("duplicate oracle call"), 3, commit, {}
        )
        assert restored.score_many(molecules) == values
        assert restored.meter.spent == 3
        assert [r["smiles"] for r in restored.rows] == ["CCO", "CCN", "CCC"]
    assert runs["serial"][0:2] == runs["batch"][0:2]
    assert runs["serial"][3] == runs["batch"][3]
    assert runs["serial"][2] == 6
    assert runs["batch"][2] == 2


def test_locked_batch_budget_and_invalid_input_fail_before_reservation(tmp_path):
    scores = make_scores(
        tmp_path,
        lambda s: pytest.fail("unexpected oracle call"),
        lambda: pytest.fail("unexpected remote barrier"),
        budget=1,
    )
    with pytest.raises(BudgetExceeded):
        scores.score_many(["CCO", "CCN"])
    with pytest.raises(ValueError, match="invalid SMILES"):
        scores.score_many(["CCO", "not-a-molecule"])
    assert not list(tmp_path.glob("oracle/*/started.json"))
    assert scores.meter.spent == 0


@pytest.mark.parametrize("error", [RuntimeError("oracle failed"), KeyboardInterrupt()])
def test_locked_batch_failure_and_interrupt_cannot_retry(tmp_path, error):
    calls = []

    def oracle(s):
        calls.append(s)
        if len(calls) == 2:
            raise error
        return 0.5

    scores = make_scores(tmp_path, oracle, lambda: None)
    with pytest.raises(type(error)):
        scores.score_many(["CCO", "CCN", "CCC"])
    assert calls == ["CCO", "CCN"]
    assert len(list(tmp_path.glob("oracle/*/started.json"))) == 3
    assert unseal(tmp_path / "oracle/0000/result.json")["score"] == 0.5
    with pytest.raises(RuntimeError, match="interrupted"):
        scores.score("CCC")
    with pytest.raises(RuntimeError, match="interrupted"):
        scores.score_many(["CCC"])
    with pytest.raises(RuntimeError, match="(unresolved|failed) oracle attempt"):
        DurableScores(tmp_path, lambda s: pytest.fail("implicit retry"), 3, lambda: None, {})


def test_locked_batch_requires_durable_reservation_before_calls(tmp_path):
    scores = make_scores(
        tmp_path,
        lambda s: pytest.fail("called before durable reservation"),
        lambda: (_ for _ in ()).throw(OSError("volume unavailable")),
    )
    with pytest.raises(OSError, match="volume unavailable"):
        scores.score_many(["CCO", "CCN"])
    assert scores.meter.spent == 0
    with pytest.raises(RuntimeError, match="unresolved oracle attempt"):
        DurableScores(tmp_path, lambda s: 0.4, 3, lambda: None, {})
