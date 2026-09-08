"""Focused zero-network saved-batch and indexed-oracle accounting checks."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from compose_v4.experiments import t4_partial_docking as partial
from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.experiments.t4_warm_continuation import payload_hash

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "diagnostics/t4_warm_continuation/attempt_2"


def configuration():
    contract = json.loads((ROOT / partial.CONTRACT_PATH).read_text())
    old = unseal(SOURCE / "round_2/parents/00.json")["task"]
    task = {
        **contract["task"],
        "smiles": old["smiles"],
        "target": old["target"],
        "code_revision": "fixture",
        "expected_input_sha256": contract["expected_input_sha256"],
    }
    return contract, task


def test_saved_lock_reuses_exact_states_and_excludes_seen_or_ineligible(monkeypatch):
    from compose_v4.rewrite.kernel import RewriteSystem

    monkeypatch.setattr(
        RewriteSystem, "apply", lambda *_: pytest.fail("saved docking executed rewrite")
    )
    contract, task = configuration()
    assert contract["contract_sha256"] == payload_hash(
        {k: v for k, v in contract.items() if k != "contract_sha256"}
    )
    lock = partial.lock_saved_batch(SOURCE, contract, task, qed_min=0.6, sa_max=4)
    assert len(lock["pool"]) == 25 and len(lock["take"]) == 13
    assert len({c["bundle_id"] for c in lock["take"]}) == 13
    assert {c["smiles"] for c in lock["take"]} == set(contract["authorized_smiles"])
    assert (
        sum("already_evaluated_or_seed" in c["endpoint_exclusion_reasons"] for c in lock["pool"])
        == 1
    )
    assert all(c["v"] == 0 and c["state"] for c in lock["take"])
    assert lock["new_executor_calls"] == lock["new_generator_calls"] == lock["oracle_calls"] == 0


def test_remote_recheck_can_shorten_but_not_backfill_batch(monkeypatch):
    contract, task = configuration()
    original = partial.feasible_endpoint
    count = 0

    def reject_first(properties):
        nonlocal count
        if original(properties):
            count += 1
            return count != 1
        return False

    monkeypatch.setattr(partial, "feasible_endpoint", reject_first)
    lock = partial.lock_saved_batch(SOURCE, contract, task, qed_min=0.6, sa_max=4)
    assert len(lock["take"]) == 12


@pytest.mark.parametrize("defect", ["hash", "budget", "candidate"])
def test_source_or_authorization_drift_fails_before_docking(defect):
    contract, task = configuration()
    if defect == "hash":
        contract["source"]["files"]["warm_start.json"] = "bad"
    elif defect == "budget":
        contract["compute"]["oracle_call_limit"] = 14
    else:
        contract["authorized_smiles"][0] = "C"
    with pytest.raises(ValueError):
        partial.lock_saved_batch(SOURCE, contract, task, qed_min=0.6, sa_max=4)


def test_out_of_order_parallel_results_preserve_identity_and_accounting(tmp_path):
    contract, task = configuration()
    calls = []

    def dock(lock, digest, output):
        assert sha256_file(output / "candidate_lock.json") == digest
        assert (output / "docking_started.json").exists()
        for i in reversed(range(len(lock["take"]))):
            calls.append(i)
            yield {
                "index": i,
                "smiles": lock["take"][i]["smiles"],
                "candidate_lock_sha256": digest,
                "ds": None if i == 0 else -8.0 - i / 100,
            }

    result = partial.run_batch(task, contract, SOURCE, tmp_path, dock, qed_min=0.6, sa_max=4)
    assert len(calls) == result["new_oracle_attempts"] == 13
    assert result["cumulative_guided_calls"] == 33 and result["remaining_new_call_allowance"] == 27
    assert result["oracle_failures"] == 1 and result["optimizer_round_completed"] is False
    assert [r["index"] for r in result["docked"]] == list(range(13))
    assert (
        partial.run_batch(task, contract, SOURCE, tmp_path, dock, qed_min=0.6, sa_max=4) == result
    )
    assert len(calls) == 13


def test_interrupted_batch_does_not_redock(tmp_path):
    contract, task = configuration()

    def interrupted(*_):
        raise RuntimeError("synthetic worker interruption")

    with pytest.raises(RuntimeError, match="synthetic worker"):
        partial.run_batch(task, contract, SOURCE, tmp_path, interrupted, qed_min=0.6, sa_max=4)
    with pytest.raises(RuntimeError, match="no implicit oracle retry"):
        partial.run_batch(
            task,
            contract,
            SOURCE,
            tmp_path,
            lambda *_: pytest.fail("retried"),
            qed_min=0.6,
            sa_max=4,
        )


def test_resource_and_no_generator_launch_boundary():
    tree = ast.parse((ROOT / "modal_apps/genmol_t4_opt_app.py").read_text())
    for name, cap, timeout in [("t4_partial_docking", 1, 3600), ("t4_partial_dock_one", 4, 600)]:
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        keywords = {
            k.arg: ast.literal_eval(k.value)
            for k in node.decorator_list[0].keywords
            if k.arg in ("cpu", "memory", "timeout", "max_containers", "retries")
        }
        assert keywords == {
            "cpu": (1.0, 1.0),
            "memory": 2048,
            "timeout": timeout,
            "max_containers": cap,
            "retries": 0,
        }
        assert "_runtime" not in ast.unparse(node)


def test_worker_uses_lock_and_never_repeats_started_or_completed_row(tmp_path, monkeypatch):
    contract, task = configuration()
    contract["required_rdkit"] = partial.rdBase.rdkitVersion  # local boundary fixture only
    lock = partial.lock_saved_batch(SOURCE, contract, task, qed_min=0.6, sa_max=4)
    run_id = "a" * 64
    root = tmp_path / partial.KIND / run_id
    digest = seal(root / "candidate_lock.json", lock)
    partial.publish_json(root / "docking_started.json", {"candidate_lock_sha256": digest})
    app = tmp_path / "modal_apps/genmol_t4_opt_app.py"
    app.parent.mkdir()
    app.write_text("# worker fixture\n")
    actual_verify = partial.verify_file

    def verify(path, expected):
        if not str(path).startswith("/opt/dock/"):
            actual_verify(path, expected)

    monkeypatch.setattr(partial, "verify_file", verify)
    payload = {
        "image_revision": {},
        "run_id": run_id,
        "index": 0,
        "app_sha256": sha256_file(app),
        "candidate_lock_sha256": digest,
    }
    volume = SimpleNamespace(reload=lambda: None, commit=lambda: None)
    calls = []

    def dock(smiles, _tag):
        assert (root / "rows/00/started.json").exists()
        calls.append(smiles)
        return -9.0

    first = partial.dock_saved_row(payload, tmp_path, tmp_path, volume, lambda _: None, dock)
    assert (
        partial.dock_saved_row(payload, tmp_path, tmp_path, volume, lambda _: None, dock) == first
    )
    partial.publish_json(root / "rows/01/started.json", {})
    with pytest.raises(RuntimeError, match="no implicit re-docking"):
        partial.dock_saved_row(
            {**payload, "index": 1}, tmp_path, tmp_path, volume, lambda _: None, dock
        )
    assert calls == [lock["take"][0]["smiles"]]
