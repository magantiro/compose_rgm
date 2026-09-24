"""Preview reuse must be exact, complete, and independent of outcome quality."""

import json
import sys

import pytest

from tools.seed_fragment_joint_pilot import digest, main


def fixture(tmp_path):
    support = tmp_path / "support"
    (support / "rows").mkdir(parents=True)
    (support / "attempts").mkdir()
    hashes = {}
    for task in ("motif_extension", "scaffold_decoration"):
        for drug in range(10):
            attempts = []
            for index in range(2):
                attempt = {
                    "complete": True,
                    "attempt_index": index,
                    "offered": [{"draw": i} for i in range(8)],
                    "committed_smiles": None if drug == 0 else "CCO",
                    "rng_state_before": index,
                    "rng_state_after": index + 1,
                }
                path = support / "attempts" / f"{task}_{drug}_{index:03d}.json"
                path.write_text(json.dumps(attempt))
                attempts.append(attempt)
            relative = f"rows/{task}_{drug}.json"
            path = support / relative
            path.write_text(json.dumps({"task": task, "drug": str(drug), "attempts": attempts}))
            hashes[relative] = digest(path)
    (support / "manifest.json").write_text(
        json.dumps(
            {
                "mode": "support",
                "seed": 0,
                "attempts_per_prompt": 2,
                "candidate_attempts": 8,
                "inputs": {},
            }
        )
    )
    (support / "summary.json").write_text(
        json.dumps(
            {
                "mode": "support",
                "support_pass": True,
                "row_sha256": hashes,
            }
        )
    )
    return support


def invoke(monkeypatch, support, output):
    monkeypatch.setattr(
        sys, "argv", ["reuse", "--support-dir", str(support), "--output-dir", str(output)]
    )
    main()


def test_all_prefix_bytes_and_failures_are_reused(tmp_path, monkeypatch):
    support = fixture(tmp_path)
    output = tmp_path / "pilot"
    invoke(monkeypatch, support, output)
    files = list((output / "attempts").glob("*.json"))
    assert len(files) == 40
    assert sum(json.loads(p.read_text())["committed_smiles"] is None for p in files) == 4
    for path in files:
        assert path.read_bytes() == (support / "attempts" / path.name).read_bytes()
    assert not (output / "summary.json").exists()


def test_failed_gate_cannot_seed_pilot(tmp_path, monkeypatch):
    support = fixture(tmp_path)
    path = support / "summary.json"
    record = json.loads(path.read_text())
    record["support_pass"] = False
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="qualified"):
        invoke(monkeypatch, support, tmp_path / "pilot")


def test_changed_attempt_cannot_seed_pilot(tmp_path, monkeypatch):
    support = fixture(tmp_path)
    path = next((support / "attempts").glob("*.json"))
    record = json.loads(path.read_text())
    record["committed_smiles"] = "CCCC"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="identical"):
        invoke(monkeypatch, support, tmp_path / "pilot")
