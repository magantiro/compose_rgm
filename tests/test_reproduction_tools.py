"""Tests for the reproduction and preservation tools themselves.

These cover the TOOLS, not the science they read.  The existing experiment
registry tests exercise the registry; nothing exercised these scripts, so every
defect they had was found by running them by hand.

Each test here corresponds to a way one of them previously returned success
while checking nothing:

  - a manifest whose schema is unknown was read anyway and passed
  - an input with no hash counted as verified
  - `--verify` checked a fresh scan, so a deleted file was never looked for
  - a tracked-but-modified file counted as protected
  - "regenerable" was asserted from a filename with no recovery dependency
  - the AUC depression note used a closed form that is wrong below one
    logging period

Style note: these drive the real module functions.  A test that recomputes an
expectation from the code under test cannot fail, so expectations here are
either literals or come from a second, independent route.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True,
                   capture_output=True)


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    return root


# ---- verify_experiment_inputs ----


def _manifest(root: Path, task: str, document: dict) -> None:
    target = root / "experiments" / task / "manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document))


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def verifier():
    return _load("verify_experiment_inputs")


def test_unsupported_schema_is_rejected(tmp_path, verifier, capsys):
    root = tmp_path
    (root / "a.txt").write_text("x")
    _manifest(root, "frag", {
        "schema_version": "fragment_manifest_v3",
        "task": "frag",
        "inputs": [{"path": "a.txt", "sha256": _sha(root / "a.txt")}],
    })
    assert verifier.main(["--repo-root", str(root)]) == 1
    assert "REJECTED" in capsys.readouterr().out


def test_missing_hash_is_not_a_pass(tmp_path, verifier, capsys):
    root = tmp_path
    (root / "a.txt").write_text("x")
    _manifest(root, "t", {
        "schema_version": "experiment_task_manifest_v1",
        "task": "t",
        "inputs": [{"path": "a.txt", "role": "no hash at all"}],
    })
    assert verifier.main(["--repo-root", str(root)]) == 1
    assert "declares no hash" in capsys.readouterr().out


@pytest.mark.parametrize("value", ["zz", "abc", "a" * 63, "A" * 64 + "b"])
def test_malformed_hash_is_rejected(tmp_path, verifier, value):
    root = tmp_path
    (root / "a.txt").write_text("x")
    _manifest(root, "t", {
        "schema_version": "experiment_task_manifest_v1",
        "task": "t",
        "inputs": [{"path": "a.txt", "sha256": value}],
    })
    assert verifier.main(["--repo-root", str(root)]) == 1


def test_empty_inputs_is_rejected(tmp_path, verifier):
    root = tmp_path
    _manifest(root, "t", {
        "schema_version": "experiment_task_manifest_v1",
        "task": "t",
        "inputs": [],
    })
    assert verifier.main(["--repo-root", str(root)]) == 1


def test_zero_verified_overall_fails_even_with_no_failures(tmp_path, verifier, capsys):
    """Reaches main()'s total_ok guard, which the empty-inputs case never does.

    An informational pin that has drifted is not a failure, so a manifest whose
    every input drifted yields 0 verified, 1 drifted, 0 failed -- and an earlier
    version returned success for it. The empty-inputs test cannot cover this:
    it is rejected one hop earlier, inside check(). Two guards, one sink.
    """
    root = tmp_path
    target = root / "a.txt"
    target.write_text("now")
    _manifest(root, "t", {
        "schema_version": "experiment_task_manifest_v1",
        "task": "t",
        "inputs": [{"path": "a.txt", "pin": "informational", "sha256": "0" * 64}],
    })
    assert verifier.main(["--repo-root", str(root)]) == 1
    output = capsys.readouterr().out
    assert "drift" in output
    assert "NOTHING WAS VERIFIED" in output


def test_corrupted_input_fails_a_strict_pin(tmp_path, verifier, capsys):
    root = tmp_path
    target = root / "a.txt"
    target.write_text("original")
    _manifest(root, "t", {
        "schema_version": "experiment_task_manifest_v1",
        "task": "t",
        "inputs": [{"path": "a.txt", "pin": "strict", "sha256": _sha(target)}],
    })
    assert verifier.main(["--repo-root", str(root)]) == 0
    target.write_text("tampered")
    assert verifier.main(["--repo-root", str(root)]) == 1
    assert "CHANGED" in capsys.readouterr().out


def test_missing_file_fails(tmp_path, verifier, capsys):
    root = tmp_path
    target = root / "a.txt"
    target.write_text("x")
    digest = _sha(target)
    target.unlink()
    _manifest(root, "t", {
        "schema_version": "experiment_task_manifest_v1",
        "task": "t",
        "inputs": [{"path": "a.txt", "sha256": digest}],
    })
    assert verifier.main(["--repo-root", str(root)]) == 1
    assert "MISSING" in capsys.readouterr().out


def test_wellformed_manifest_passes(tmp_path, verifier):
    root = tmp_path
    (root / "a.txt").write_text("x")
    _manifest(root, "t", {
        "schema_version": "experiment_task_manifest_v1",
        "task": "t",
        "inputs": [{"path": "a.txt", "sha256": _sha(root / "a.txt")}],
    })
    assert verifier.main(["--repo-root", str(root)]) == 0


def test_shipped_manifests_verify():
    """The repo's own manifests must pass the strict verifier."""
    verifier = _load("verify_experiment_inputs")
    assert verifier.main([]) == 0


# ---- preservation_inventory ----


@pytest.fixture
def inventory():
    return _load("preservation_inventory")


def test_modified_tracked_file_is_not_protected(tmp_path, inventory):
    root = _repo(tmp_path)
    target = root / "a.txt"
    target.write_text("committed")
    _git(root, "add", "a.txt")
    _git(root, "commit", "-qm", "add")

    tracked = inventory.git_tracked(root)
    clean = inventory.walk(root, tracked, inventory.git_dirty(root))
    assert clean["at_risk"] == []

    target.write_text("modified, uncommitted")
    dirty = inventory.git_dirty(root)
    assert "a.txt" in dirty
    result = inventory.walk(root, tracked, dirty)
    assert [entry["path"] for entry in result["at_risk"]] == ["a.txt"]
    assert "MODIFIED" in result["at_risk"][0]["reason"]


def test_regenerable_claim_requires_its_dependency(tmp_path, inventory):
    """A capsule without its manifest is NOT regenerable."""
    root = _repo(tmp_path)
    capsule = root / "diagnostics" / "run_v1" / "source_capsule" / "deep" / "nest"
    capsule.mkdir(parents=True)
    (capsule / "mod.py").write_text("x")
    relative = "diagnostics/run_v1/source_capsule/deep/nest/mod.py"

    problem = inventory.recovery_dependency_ok(root, relative, "source_capsule")
    assert problem is not None and "manifest" in problem

    (root / "diagnostics" / "run_v1" / "source_capsule_manifest.json").write_text("{}")
    assert inventory.recovery_dependency_ok(root, relative, "source_capsule") is None


def test_deep_capsule_nesting_still_resolves(tmp_path, inventory):
    """Anchoring on a fixed parent depth demoted 5,136 recoverable files."""
    root = _repo(tmp_path)
    deep = root / "diagnostics" / "r" / "source_capsule"
    for extra in ("a", "b", "c", "d", "e", "f"):
        deep = deep / extra
    deep.mkdir(parents=True)
    (deep / "x.py").write_text("x")
    (root / "diagnostics" / "r" / "source_capsule_manifest.json").write_text("{}")
    relative = str((deep / "x.py").relative_to(root))
    assert inventory.recovery_dependency_ok(root, relative, "source_capsule") is None


def test_snapshot_without_receipts_is_demoted(tmp_path, inventory):
    root = _repo(tmp_path)
    rounds = root / "diagnostics" / "c" / "campaign" / "round_0001"
    rounds.mkdir(parents=True)
    (rounds / "complete.json").write_text("{}")
    relative = "diagnostics/c/campaign/round_0001/complete.json"
    assert inventory.recovery_dependency_ok(
        root, relative, "cumulative_round_snapshot") is not None

    receipt = root / "diagnostics" / "c" / "oracle" / "query_000000"
    receipt.mkdir(parents=True)
    (receipt / "result.json").write_text("{}")
    assert inventory.recovery_dependency_ok(
        root, relative, "cumulative_round_snapshot") is None


def test_verify_uses_the_saved_manifest_not_a_fresh_scan(tmp_path, inventory, capsys):
    """A recorded file that has since vanished must still be looked for."""
    root = _repo(tmp_path)
    (root / "keep.txt").write_text("tracked")
    _git(root, "add", "keep.txt")
    _git(root, "commit", "-qm", "c")
    loose = root / "loose.bin"
    loose.write_text("irreplaceable")

    assert inventory.main(["--repo-root", str(root), "--write-manifest"]) == 0
    saved = json.loads((root / inventory.MANIFEST).read_text())
    assert [e["path"] for e in saved["at_risk"]] == ["loose.bin"]

    backup = tmp_path / "backup"
    backup.mkdir()
    (backup / "loose.bin").write_text("irreplaceable")
    assert inventory.main(
        ["--repo-root", str(root), "--verify", str(backup)]) == 0

    # The file disappears locally. A fresh scan would not know it ever existed;
    # the saved manifest does, so an incomplete backup must still fail.
    loose.unlink()
    (backup / "loose.bin").unlink()
    assert inventory.main(
        ["--repo-root", str(root), "--verify", str(backup)]) == 1
    output = capsys.readouterr().out
    assert "ABSENT FROM BACKUP" in output
    assert "may be LOST" in output


def test_verify_detects_a_corrupted_backup(tmp_path, inventory, capsys):
    root = _repo(tmp_path)
    (root / "loose.bin").write_text("original")
    assert inventory.main(["--repo-root", str(root), "--write-manifest"]) == 0
    backup = tmp_path / "backup"
    backup.mkdir()
    (backup / "loose.bin").write_text("CORRUPTED")
    assert inventory.main(
        ["--repo-root", str(root), "--verify", str(backup)]) == 1
    assert "DIFFERS" in capsys.readouterr().out


def test_verify_without_a_manifest_refuses(tmp_path, inventory):
    root = _repo(tmp_path)
    backup = tmp_path / "backup"
    backup.mkdir()
    assert inventory.main(
        ["--repo-root", str(root), "--verify", str(backup)]) == 2


def test_non_repository_is_refused(tmp_path, inventory):
    assert inventory.main(["--repo-root", str(tmp_path)]) == 2


# ---- reproduce_pmo_tables ----


@pytest.fixture
def pmo():
    return _load("reproduce_pmo_tables")


@pytest.mark.parametrize(
    "budget,expected",
    [(64, 0.50), (96, 0.50), (100, 0.50), (112, 0.446), (200, 0.25),
     (250, 0.20), (1000, 0.05), (10000, 0.005)],
)
def test_auc_depression_is_measured_not_assumed(pmo, budget, expected):
    """The old closed form reported 78.1% at budget 64; the truth is 50%.

    Below one logging period the grid never fires and the whole run is a single
    trapezoid from (0, 0), which removes exactly half at ANY such budget.
    """
    auc = pmo.load_auc()
    assert pmo.measured_depression(auc, budget) == pytest.approx(expected, abs=5e-4)


def test_auc_self_test_guards_the_environment(pmo):
    pmo.self_test(pmo.load_auc())


def test_zero_charged_ledger_is_excluded_on_evidence(tmp_path, pmo):
    """A gate ledger is shaped like a scored run; only its report distinguishes it."""
    root = tmp_path
    ledger = root / "diagnostics" / "gate_v1" / "dryrun"
    (ledger / "oracle" / "query_000000").mkdir(parents=True)
    (ledger / "oracle" / "manifest.json").write_text(json.dumps(
        {"budget": 10, "task": {"kind": "pmo", "name": "celecoxib_rediscovery"}}))
    (ledger / "oracle" / "query_000000" / "result.json").write_text(json.dumps(
        {"index": 0, "endpoint": "CCO", "score": 0.9, "status": "complete"}))

    assert pmo.zero_charged_declaration(ledger, root) is None
    (root / "diagnostics" / "gate_v1" / "report.json").write_text(json.dumps(
        {"evidence_role": "zero_charged_oracle_integration_gate",
         "new_charged_oracle_calls": 0}))
    assert pmo.zero_charged_declaration(ledger, root) is not None


def test_incomplete_receipts_are_counted_not_scored(tmp_path, pmo):
    root = tmp_path
    ledger = root / "diagnostics" / "run"
    (ledger / "oracle").mkdir(parents=True)
    (ledger / "oracle" / "manifest.json").write_text(json.dumps(
        {"budget": 4, "task": {"kind": "pmo", "name": "qed"}}))
    for index, status in enumerate(["complete", "complete", "failed"]):
        query = ledger / "oracle" / f"query_{index:06d}"
        query.mkdir()
        (query / "result.json").write_text(json.dumps(
            {"index": index, "endpoint": "CCO", "score": 0.5,
             "status": status}))

    entry = pmo.read_ledger(ledger)
    assert entry is not None
    assert entry["incomplete"] == 1
    assert len(entry["scored"]) == 2

    scores = pmo.score_ledger(entry, pmo.load_auc())
    assert scores["charged"] == 2
    assert scores["auc_budget"] == 4


def test_unreadable_receipt_is_reported_not_silently_dropped(tmp_path, pmo):
    root = tmp_path
    ledger = root / "diagnostics" / "run"
    (ledger / "oracle" / "query_000000").mkdir(parents=True)
    (ledger / "oracle" / "manifest.json").write_text(json.dumps(
        {"budget": 2, "task": {"kind": "pmo", "name": "qed"}}))
    (ledger / "oracle" / "query_000000" / "result.json").write_text("{not json")
    entry = pmo.read_ledger(ledger)
    assert entry is None or entry["unreadable"] == 1


# ---- reproduce_t4_table ----


@pytest.fixture
def t4():
    return _load("reproduce_t4_table")


def test_t4_seal_detects_tampering(tmp_path, t4):
    envelope = {"payload": {"a": 1}, "payload_sha256": t4.canonical_sha256({"a": 1})}
    assert t4.check_seal(envelope)
    envelope["payload"]["a"] = 2
    with pytest.raises(t4.ReproductionError):
        t4.check_seal(envelope)


def test_submitted_combined_must_be_the_better_run(t4):
    document = {
        "summary": {},
        "rows": [{"cell": "x", "target": "x", "seed": 0, "delta": 0.4,
                  "new100": -9.0, "old200": -10.0, "combined": -9.0,
                  "genmol": -8.0, "verdict": "win"}],
    }
    _, failures = t4.check_submitted(document)
    assert any("is not the better of" in f for f in failures)


def test_submitted_verdict_must_follow_from_the_scores(t4):
    document = {
        "summary": {},
        "rows": [{"cell": "x", "target": "x", "seed": 0, "delta": 0.4,
                  "new100": -9.0, "old200": None, "combined": -9.0,
                  "genmol": -10.0, "verdict": "win"}],
    }
    _, failures = t4.check_submitted(document)
    assert any("implies" in f for f in failures)


def test_shipped_submitted_table_reproduces():
    """The real submitted artifact must verify against its own typeset macros."""
    t4 = _load("reproduce_t4_table")
    assert t4.main(["--submitted"]) == 0
