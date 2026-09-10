import ast
import json
from pathlib import Path

import pytest

from compose_v4.experiments import t4_proposal_docking as proposal
from compose_v4.experiments.t4_matched_pilot import unseal

ROOT = Path(__file__).resolve().parents[1]


def test_complete_saved_census_has_exactly_the_locked_16(monkeypatch):
    contract = json.loads((ROOT / proposal.CONTRACT_PATH).read_text())
    source = ROOT / "diagnostics/t4_recovery_lookahead/attempt_2"
    path_map = {contract["archive"]["path"]: source / "archive.json"}
    for index, item in enumerate(contract["sources"]):
        for key, name in (("generation", "generation_lock.json"), ("scored", "scored_lock.json")):
            path_map[item[key]["path"]] = source / f"case_{index // 3}/roots/{index % 3:02d}/{name}"
    actual_verify = proposal.verify_file
    monkeypatch.setattr(
        proposal, "verify_file", lambda path, digest: actual_verify(path_map[str(path)], digest)
    )
    monkeypatch.setattr(proposal, "unseal", lambda path: unseal(path_map[str(path)]))
    task = {"smiles": json.loads((ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())[0]["smiles"]}
    lock = proposal.lock_batch(contract, Path(""), task)
    assert len(lock["take"]) == len({r["smiles"] for r in lock["take"]}) == 16
    assert {r["smiles"] for r in lock["take"]} == set(contract["authorized_smiles"])
    assert all(r["v"] == 0 and r["origins"] and r["state"] for r in lock["take"])
    assert lock["best_prior"]["ds"] == -9.7
    contract["compute"]["oracle_call_limit"] = 0
    with pytest.raises(ValueError, match="authorized 16"):
        proposal.lock_batch(contract, Path(""), task)


def test_lock_precedes_oracle_and_failure_counts(tmp_path):
    lock = {"take": [{"smiles": "C"}, {"smiles": "N"}], "best_prior": {"ds": -8.0}}

    def execute(index, digest):
        assert (tmp_path / "candidate_lock.json").exists()
        assert (tmp_path / "docking_started.json").exists()
        return {
            "index": index,
            "candidate_lock_sha256": digest,
            "smiles": lock["take"][index]["smiles"],
            "ds": None if index == 0 else -9.0,
        }

    result = proposal.run_locked_batch(lock, tmp_path, execute, commit=lambda: None, progress={})
    assert result["new_oracle_attempts"] == 2 and result["oracle_failures"] == 1
    assert result["best_batch"]["ds"] == -9.0
    assert result["cumulative_source_and_diagnostic_calls"] == 53
    lock["take"][0]["smiles"] = "O"
    with pytest.raises(ValueError, match="saved lock differs"):
        proposal.run_locked_batch(lock, tmp_path, execute, commit=lambda: None, progress={})


def test_resource_boundary_and_no_generator_initialization():
    tree = ast.parse((ROOT / "modal_apps/genmol_t4_opt_app.py").read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "t4_proposal_docking"
    )
    settings = {
        k.arg: ast.literal_eval(k.value)
        for k in node.decorator_list[0].keywords
        if k.arg in ("cpu", "memory", "timeout", "max_containers", "retries")
    }
    assert settings == {
        "cpu": (1.0, 1.0),
        "memory": 2048,
        "timeout": 1200,
        "max_containers": 1,
        "retries": 0,
    }
    assert "_runtime" not in ast.unparse(node)
