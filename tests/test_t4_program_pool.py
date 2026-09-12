"""Bounded program-pool locking/accounting, with no network or real oracle calls."""

import json
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.experiments.t4_program_pool import APP, KIND, LOCK, run_remote

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_pool_excludes_reused_winner_and_binds_corrected_proposals():
    lock = unseal(ROOT / LOCK)
    assert len(lock["take"]) == 26
    assert lock["compute"]["new_oracle_call_limit"] == 28
    assert len({r["smiles"] for r in lock["take"]}) == 26
    assert all(r["oracle_eligible"] and r["trace"]["complete"] for r in lock["take"])
    controls = unseal(ROOT / lock["winner_controls"]["path"])
    assert not {r["smiles"] for r in controls} & {r["smiles"] for r in lock["take"]}
    assert all("attempt_1" not in p for p in lock["inputs_sha256"])
    assert {r["arm"] for r in lock["memberships"]} == {
        "uninterrupted_serial",
        "independent_multi_site",
        "joint_multi_site",
    }


def test_driver_scores_locked_rows_repeats_best_and_reuses_completed_result(tmp_path):
    # A fake scalar oracle checks orchestration only. It is not scientific data.
    root, artifacts = tmp_path / "repo", tmp_path / "artifacts"
    root.mkdir()
    (root / APP).parent.mkdir(parents=True)
    (root / APP).write_text("# model-free fixture\n")
    control_path = root / "controls.json"
    seal(control_path, [{"ds": -2.0}] * 3)
    lock = {
        "required_rdkit": "2024.03.5",
        "task": {},
        "take": [{"smiles": "CC", "docking_seed": 1701}, {"smiles": "CCC", "docking_seed": 1701}],
        "compute": {"new_oracle_call_limit": 4},
        "winner_controls": {"path": "controls.json", "sha256": sha256_file(control_path)},
    }
    seal(root / LOCK, lock)
    task = {
        "program_app_sha256": sha256_file(root / APP),
        "lock_sha256": sha256_file(root / LOCK),
        "app_sha256": "unused_mock",
        "image_revision": {},
    }
    task["run_id"] = identity(task)

    class Volume:
        def reload(self):
            pass

        def commit(self):
            pass

    calls = []

    def parallel(tasks):
        for item in tasks:
            actual_lock = unseal(
                artifacts / KIND / item["stage"] / item["run_id"] / "candidate_lock.json"
            )
            row = actual_lock["take"][item["index"]]
            calls.append((item["stage"], row["docking_seed"]))
            yield {"index": item["index"], "ds": -3.0 - item["index"]}

    result = run_remote(task, root, artifacts, Volume(), lambda _: None, parallel)
    assert result["new_oracle_calls"] == 4
    assert result["reused_control_calls"] == 3
    assert result["best"]["smiles"] == "CCC"
    assert calls == [("pool", 1701), ("pool", 1701), ("confirmation", 1702), ("confirmation", 1703)]
    again = run_remote(task, root, artifacts, Volume(), lambda _: None, parallel)
    assert again == json.loads(json.dumps(result))
    assert len(calls) == 4
