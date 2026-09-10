import json
from pathlib import Path

import pytest

from compose_v4.benchmark.molleo_task3 import BudgetExceeded
from compose_v4.experiments import pmo_macro_probe as probe
from compose_v4.experiments.t4_matched_pilot import seal


def test_scores_charge_canonical_once_and_restore(tmp_path):
    lock = tmp_path / "lock.json"
    seal(lock, {"smiles": "CCO"})
    calls = []
    scores = probe.DurableScores(tmp_path, lambda s: calls.append(s) or 0.4, 1, lambda: None, {})
    scores.context = {"role": "test", "lock_path": str(lock)}
    assert scores.score("OCC") == scores.score("CCO") == {"desirability": 0.4}
    assert calls == ["CCO"]
    assert scores.meter.spent == 1 and scores.meter.n_primed == 0
    with pytest.raises(BudgetExceeded):
        scores.score("CCC")
    restored = probe.DurableScores(
        tmp_path, lambda s: pytest.fail("unnecessary oracle repeat"), 1, lambda: None, {}
    )
    assert restored.score("CCO") == {"desirability": 0.4}
    assert restored.meter.spent == 1


def test_unlocked_and_interrupted_oracle_fail_closed(tmp_path):
    scores = probe.DurableScores(
        tmp_path, lambda s: pytest.fail("unlocked evaluation"), 1, lambda: None, {}
    )
    with pytest.raises(ValueError, match="durable candidate lock"):
        scores.score("CC")
    seal(tmp_path / "oracle/0000/started.json", {"smiles": "CC"})
    with pytest.raises(RuntimeError, match="unresolved oracle attempt"):
        probe.DurableScores(tmp_path, lambda s: 0.5, 1, lambda: None, {})


@pytest.mark.parametrize("arm", probe.ARMS)
def test_only_guided_arm_scores_before_search_lock(tmp_path, monkeypatch, arm):
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph

    graph = pad_molecular_graph(smiles_to_molecular_graph("CC"), 48)
    root = {"smiles": "CC", "state": probe.encode_state(graph)}
    context = {"in_search": False}
    calls = []

    def oracle(smiles):
        calls.append((smiles, context["in_search"]))
        return 0.5

    def search(node, hierarchy, *, config, score, save, **kwargs):
        candidate = {"smiles": "CCC", "attempt_id": "test"}
        attempt = {"status": "complete", "bundle": {"option": "generic"}, "candidate": candidate}
        save("levels/00/attempt_00_00", attempt)
        context["in_search"] = True
        if arm == "guided":
            assert score("CCC")["desirability"] == 0.5
        else:
            with pytest.raises(AssertionError, match="post-hoc"):
                score("CCC")
        context["in_search"] = False
        result = {"attempts": [attempt]}
        save("generation_lock", result)
        return result

    monkeypatch.setattr(probe, "run_search", search)
    scores = probe.DurableScores(tmp_path, oracle, 44, lambda: None, {})
    contract = {
        "roots": [root],
        "seed": 1,
        "search": {"depth": 3, "width": 2, "branches": 2, "primitive_budget": 110},
        "interpretation": "test",
    }
    result = probe.run_case(
        contract,
        {"arm": arm, "replicate": 0},
        None,
        tmp_path,
        scores,
        meter=None,
        commit=lambda: None,
        progress={},
    )
    assert calls == [("CC", False), ("CCC", arm == "guided")]
    assert result["oracle_calls"] == 2 and result["primed"] == 0


def test_frozen_recipe_and_jnk3_fixture():
    root = Path(__file__).resolve().parents[1]
    contract = probe.load_contract(root)
    assert len(probe.cases(contract)) == 12
    assert len({probe.case_name(c) for c in probe.cases(contract)}) == 12
    assert contract["oracle_budget"] == 4 + 4 * (2 + 4 + 4)
    parity = json.loads((root / contract["inputs"]["jnk3_parity"]["path"]).read_text())
    oracle = probe.make_oracle("jnk3", root, contract)
    assert [oracle(s) for s in parity["panel"]] == pytest.approx(
        parity["JNK3"]["frozen"], abs=1e-12
    )
