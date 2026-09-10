import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from compose_v4.experiments import t4_winner_route_docking as diagnostic
from compose_v4.experiments.continuation_profile import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def test_exact_saved_panel_and_limit(tmp_path):
    contract = json.loads((ROOT / diagnostic.CONTRACT_PATH).read_text())
    witness = json.loads((ROOT / contract["witness"]["path"]).read_text())
    seed = json.loads((ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())[0]["smiles"]
    previous = tmp_path / "previous.json"
    previous.write_text(
        json.dumps(
            {
                "status": "complete",
                "best": {"smiles": seed, "state": witness["source_state"], "ds": -11.0},
            }
        )
    )
    contract["incumbent"] = {"path": "previous.json", "sha256": sha256_file(previous)}
    lock = diagnostic.lock_panel(contract, ROOT, tmp_path, {"smiles": seed})
    assert len(lock["take"]) == 6
    assert lock["take"][0]["smiles"] == witness["target"]
    assert all(row["oracle_eligible"] for row in lock["take"])
    contract["compute"]["oracle_call_limit"] = 7
    with pytest.raises(ValueError, match="limit"):
        diagnostic.lock_panel(contract, ROOT, tmp_path, {"smiles": seed})


@pytest.mark.parametrize(
    "score,count,status",
    [
        (-13.0, 6, "complete"),
        (-10.0, 1, "winner_advantage_not_observed"),
        (None, 1, "winner_oracle_failed"),
    ],
)
def test_winner_precedes_conditional_panel(tmp_path, score, count, status):
    lock = {
        "take": [{"smiles": str(i), "role": str(i)} for i in range(6)],
        "winner_continue_below": -11.0,
        "interpretation": "diagnostic",
    }
    seen = []

    def execute(index, digest):
        assert (tmp_path / "candidate_lock.json").exists()
        assert (tmp_path / "docking_started.json").exists()
        seen.append(index)
        return {
            "index": index,
            "smiles": str(index),
            "ds": score if index == 0 else -9.0,
            "candidate_lock_sha256": digest,
        }

    result = diagnostic.run_panel(lock, tmp_path, execute, commit=lambda: None, progress={})
    assert seen == list(range(count))
    assert result["status"] == status and result["new_oracle_attempts"] == count
    assert result["oracle_failures"] == int(score is None)
    lock["take"][0]["smiles"] = "changed"
    with pytest.raises(ValueError, match="differs"):
        diagnostic.run_panel(lock, tmp_path, execute, commit=lambda: None, progress={})


def test_docking_row_identity_cannot_drift(tmp_path):
    lock = {"take": [{"smiles": "C", "role": "winner"}], "winner_continue_below": -11.0}
    with pytest.raises(ValueError, match="identity"):
        diagnostic.run_panel(
            lock,
            tmp_path,
            lambda i, h: {"index": i, "smiles": "N", "candidate_lock_sha256": h},
            commit=lambda: None,
            progress={},
        )


def test_launcher_uses_deployed_diagnostic_and_six_call_limit(tmp_path, monkeypatch):
    from modal_apps import run_process_v2_p50_app
    from tools import t4_launch

    calls = []

    def lookup(app, function):
        assert (app, function) == ("genmol-t4-opt", diagnostic.KIND)

        def spawn(task):
            calls.append(task)
            return SimpleNamespace(object_id="fixture-call")

        return SimpleNamespace(spawn=spawn)

    (tmp_path / "diagnostics").mkdir()
    monkeypatch.setattr(t4_launch, "ROOT", tmp_path)
    monkeypatch.setattr(t4_launch, "_sha256", lambda path: "b" * 64)
    monkeypatch.setattr(t4_launch.modal.Function, "from_name", lookup)
    monkeypatch.setattr(
        run_process_v2_p50_app,
        "local_image_revision",
        lambda **kw: {"commit": "a" * 40, "image_revision_sha256": "c" * 64},
    )
    t4_launch.launch_continuation_profile({"commit": "a" * 40}, winner_route_docking=True)
    receipt = json.loads((tmp_path / "diagnostics/t4_winner_route_docking_spawn.json").read_text())
    assert len(calls) == 1 and receipt["oracle_call_limit"] == 6
