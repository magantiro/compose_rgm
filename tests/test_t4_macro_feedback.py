"""Bounded plumbing checks using saved states; no docking/model enumeration."""

import ast
import json
from pathlib import Path

import pytest

from compose_v4.control.docking_value import identity
from compose_v4.experiments import t4_macro_feedback as feedback
from compose_v4.experiments.t4_matched_pilot import unseal

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / "diagnostics/t4_proposal_docking/attempt_1/run"
OLD = ROOT / "diagnostics/t4_recovery_lookahead/attempt_2/archive.json"


def test_full_source_ledger_and_contract(tmp_path):
    contract = json.loads((ROOT / feedback.CONTRACT_PATH).read_text())
    assert (
        identity({k: v for k, v in contract.items() if k != "contract_sha256"})
        == contract["contract_sha256"]
    )
    for key, source in (
        ("archive", OLD),
        ("saved_lock", SAVED / "candidate_lock.json"),
        ("saved_docking", SAVED / "result.json"),
    ):
        target = tmp_path / contract[key]["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
    archive = feedback.initial_archive(contract, tmp_path)
    assert len(archive) == len({r["smiles"] for r in archive}) == 68
    assert min(r["ds"] for r in archive if r["ds"] is not None) == -9.9


def test_feedback_timing_exact_state_and_completed_resume(tmp_path):
    contract = json.loads((ROOT / feedback.CONTRACT_PATH).read_text())
    contract["search"]["rounds"] = 2
    old = unseal(OLD)["archive"]
    old = [{**r, "smiles": feedback.canonical_smiles(r["smiles"])} for r in old]
    new = unseal(SAVED / "candidate_lock.json")["take"][:8]
    task = {"cell": "a" * 64}
    proposals, calls, best_new = [], [], []

    def score_factory(model, _seed):
        def score(_smiles):
            return {
                "qed": 0.8,
                "sa": 2.0,
                "sim": 0.5,
                "v": 0.0,
                "oracle_eligible": True,
                "predicted_docking": -5.0,
                "desirability": 0.5,
            }

        return score

    def propose(number, parents, digest):
        before = tmp_path / f"rounds/{number:02}/before.json"
        assert feedback.sha256_file(before) == digest
        assert parents == unseal(before)["parents"]
        if number:
            assert best_new[0] in [r["smiles"] for r in parents]
        proposals.append(number)
        candidates = []
        for row in new[number * 4 : (number + 1) * 4]:
            candidate = feedback.root_candidate(row, 110)
            candidate["node"] = {**candidate["node"], "budget": 109}
            candidate["chain"] = ["synthetic-test-option"]
            candidates.append(candidate)
        return {
            "candidates": candidates,
            "attempts": 4,
            "options": {"test": 4},
            "proposal_seconds": 0.0,
        }

    def execute(round_id, index, digest):
        folder = tmp_path / "docking" / round_id
        lock = unseal(folder / "candidate_lock.json")
        assert feedback.sha256_file(folder / "candidate_lock.json") == digest
        assert (
            json.loads((folder / "docking_started.json").read_text())["candidate_lock_sha256"]
            == digest
        )
        round_number = lock["round"]
        prior = unseal(tmp_path / f"rounds/{round_number:02}/prior/value.json")
        assert all(r["round"] < prior["before_round"] for r in prior["training_rows"])
        row = lock["take"][index]
        assert row["node"]["budget"] == 109
        score = None if index == 3 else -20.0 + index - round_number
        if index == 0:
            best_new.append(row["smiles"])
        calls.append(row["smiles"])
        return {
            "index": index,
            "smiles": row["smiles"],
            "ds": score,
            "candidate_lock_sha256": digest,
        }

    arguments = {"commit": lambda: None, "progress": {}, "score_factory": score_factory}
    result = feedback.run_episode(contract, task, old, tmp_path, propose, execute, **arguments)
    assert result["new_oracle_attempts"] == 8
    assert len(set(calls)) == len(calls) == 8 and proposals == [0, 1]
    assert result["best"]["ds"] == -21.0
    assert sum(r["ds"] is None for s in result["rounds"] for r in s["docked"]) == 2
    resumed = feedback.run_episode(contract, task, old, tmp_path, propose, execute, **arguments)
    assert resumed == result and len(calls) == 8 and proposals == [0, 1]


def test_continuation_preserves_budget_and_rejects_refresh():
    row = unseal(SAVED / "candidate_lock.json")["take"][0]
    parent = feedback.root_candidate(row, 110)
    candidate = {**parent, "node": {**parent["node"], "budget": 109}}
    continued = feedback.extend_candidate(parent, candidate, "rounds/00/worker/00")
    assert continued["node"] == candidate["node"]
    assert continued["primitive_depth"] == continued["option_depth"] == 1
    with pytest.raises(ValueError, match="horizon"):
        feedback.extend_candidate(continued, parent, "bad-reset")


def test_launch_scope():
    tree = ast.parse((ROOT / "modal_apps/genmol_t4_opt_app.py").read_text())
    for name, timeout, containers in (
        ("t4_macro_feedback", 7200, 1),
        ("t4_macro_feedback_propose", 1200, 8),
    ):
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        settings = {
            k.arg: ast.literal_eval(k.value)
            for k in node.decorator_list[0].keywords
            if k.arg in ("timeout", "max_containers", "retries")
        }
        assert settings == {"timeout": timeout, "max_containers": containers, "retries": 0}
        if name == "t4_macro_feedback":
            assert "_runtime" not in ast.unparse(node)
