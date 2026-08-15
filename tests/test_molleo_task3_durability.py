"""Durability: a killed run must lose essentially nothing, and must resume EXACTLY.

The decisive test here is `test_a_killed_run_resumes_into_the_same_run`: a real
child process is killed with `os._exit` in the middle of a generation, and the
resumed run is required to end in the same state -- same evaluations, same
values, same archive, same RNG position -- as a run that was never interrupted.
Everything else in this file exists to make the failure modes of that mechanism
explicit.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from compose_v4.benchmark.oracles import DEFAULT_BUNDLE_DIR
from compose_v4.benchmark.run_store import RunStore, restore_rng

BUNDLE_PRESENT = (DEFAULT_BUNDLE_DIR / "molleo_task3_oracle_manifest.json").exists()
REPO_ROOT = Path(__file__).resolve().parents[1]

VALUES = (0.1, 0.2, 0.3, 0.4, 0.5)


def open_store(root: Path, **kwargs):
    return RunStore.open(root, seed=1, budget=100, policy="test", **kwargs)


# ---- the ledger -----------------------------------------------------------

def test_evaluations_survive_a_reopen(tmp_path):
    store, state = open_store(tmp_path)
    assert state is None
    for i in range(10):
        store.record(f"C{i}", VALUES)
    store.close()

    store, state = open_store(tmp_path)
    store.close()
    assert state is not None
    assert state.spent == 10
    assert state.evaluations["C7"] == VALUES


def test_a_half_written_final_line_is_forgiven_and_truncated(tmp_path):
    """The one corruption that is expected: a process killed mid-write.

    That record was never acknowledged, so dropping it is correct -- and the
    partial bytes MUST be truncated, or the next append would splice onto them.
    """
    store, _ = open_store(tmp_path)
    for i in range(5):
        store.record(f"C{i}", VALUES)
    store.close()
    with open(store.ledger_path, "a") as handle:
        handle.write('{"n":5,"smiles":"CCO","v":[0.1,0.2')   # killed here

    store, state = open_store(tmp_path)
    assert state.spent == 5
    store.record("C5", VALUES)
    store.close()

    lines = store.ledger_path.read_text().strip().split("\n")
    assert len(lines) == 6
    assert json.loads(lines[-1])["smiles"] == "C5"


def test_a_corrupt_line_that_is_not_the_last_one_raises(tmp_path):
    store, _ = open_store(tmp_path)
    for i in range(5):
        store.record(f"C{i}", VALUES)
    store.close()
    lines = store.ledger_path.read_text().split("\n")
    lines[2] = "{not json"
    store.ledger_path.write_text("\n".join(lines))
    with pytest.raises(ValueError, match="corrupt and is not the final line"):
        open_store(tmp_path)


def test_a_gap_in_the_sequence_raises(tmp_path):
    store, _ = open_store(tmp_path)
    for i in range(5):
        store.record(f"C{i}", VALUES)
    store.close()
    lines = [line for line in store.ledger_path.read_text().split("\n") if line]
    del lines[2]
    store.ledger_path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="records are missing"):
        open_store(tmp_path)


def test_the_same_molecule_recorded_twice_raises(tmp_path):
    """The meter charges once per canonical molecule, so a repeat means the
    ledger and the budget have stopped describing the same run."""
    store, _ = open_store(tmp_path)
    store.record("CCO", VALUES)
    store._sequence = 1
    store.record("CCO", VALUES)
    store.close()
    with pytest.raises(ValueError, match="twice"):
        open_store(tmp_path)


# ---- checkpoints ----------------------------------------------------------

def test_a_checkpoint_restores_step_archive_and_rng(tmp_path):
    store, _ = open_store(tmp_path)
    rng = np.random.default_rng(7)
    rng.random(13)                       # advance it somewhere non-trivial
    for i in range(4):
        store.record(f"C{i}", VALUES)
    store.checkpoint(step=3, archive=["CCO", "CCC"], rng=rng,
                     policy_state={"temperature": 0.5})
    expected_next = rng.random(5)
    store.close()

    store, state = open_store(tmp_path)
    store.close()
    assert state.step == 3
    assert state.archive == ["CCO", "CCC"]
    assert state.policy_state == {"temperature": 0.5}
    assert np.array_equal(restore_rng(state.rng_state).random(5), expected_next)


def test_evaluations_after_the_last_checkpoint_are_kept_and_flagged(tmp_path):
    """They were paid for, so they are honoured -- but the policy that asked for
    them did not survive, and a resuming policy is entitled to know that."""
    store, _ = open_store(tmp_path)
    for i in range(4):
        store.record(f"C{i}", VALUES)
    store.checkpoint(step=1, archive=[], rng=None)
    for i in range(4, 9):
        store.record(f"C{i}", VALUES)
    store.close()

    store, state = open_store(tmp_path)
    store.close()
    assert state.spent == 9
    assert state.uncheckpointed == 5


def test_an_edited_ledger_prefix_refuses_to_resume(tmp_path):
    """Extending a ledger is normal; rewriting one is not."""
    store, _ = open_store(tmp_path)
    for i in range(5):
        store.record(f"C{i}", VALUES)
    store.checkpoint(step=1, archive=[], rng=None)
    store.close()
    text = store.ledger_path.read_text().replace('"C3"', '"CX"')
    store.ledger_path.write_text(text)
    with pytest.raises(ValueError, match="modified, not merely extended"):
        open_store(tmp_path)


def test_a_truncated_ledger_refuses_to_resume(tmp_path):
    store, _ = open_store(tmp_path)
    for i in range(8):
        store.record(f"C{i}", VALUES)
    store.checkpoint(step=1, archive=[], rng=None)
    store.close()
    lines = [line for line in store.ledger_path.read_text().split("\n") if line]
    store.ledger_path.write_text("\n".join(lines[:3]) + "\n")
    with pytest.raises(ValueError, match="gone backwards"):
        open_store(tmp_path)


def test_a_destroyed_checkpoint_falls_back_to_the_previous_one(tmp_path):
    """Two generations of checkpoint, so a crash during the rename still leaves
    something to resume from."""
    store, _ = open_store(tmp_path)
    store.record("C0", VALUES)
    store.checkpoint(step=1, archive=["A"], rng=None)
    store.record("C1", VALUES)
    store.checkpoint(step=2, archive=["B"], rng=None)
    store.close()
    store.checkpoint_path.write_text("{ truncated")

    store, state = open_store(tmp_path)
    store.close()
    assert state.step == 1 and state.archive == ["A"]


def test_a_tampered_checkpoint_payload_is_rejected(tmp_path):
    store, _ = open_store(tmp_path)
    store.record("C0", VALUES)
    store.checkpoint(step=1, archive=["A"], rng=None)
    store.close()
    document = json.loads(store.checkpoint_path.read_text())
    document["payload"]["step"] = 99
    store.checkpoint_path.write_text(json.dumps(document))
    # The digest no longer matches, so this checkpoint is skipped; with no
    # previous one there is nothing to fall back to and the ledger stands alone.
    store, state = open_store(tmp_path)
    store.close()
    assert state.step == 0 and state.spent == 1


def test_arrays_survive_a_checkpoint(tmp_path):
    store, _ = open_store(tmp_path)
    store.checkpoint(step=0, archive=[], rng=None,
                     arrays={"scores": np.arange(6, dtype=float)})
    store.close()
    store, _ = open_store(tmp_path)
    assert np.array_equal(store.load_arrays()["scores"], np.arange(6, dtype=float))
    store.close()


# ---- the real thing -------------------------------------------------------

RUN_SCRIPT = textwrap.dedent('''
    """A tiny deterministic policy. Optionally dies mid-generation."""
    import json, os, sys
    sys.path.insert(0, {src!r})
    import numpy as np
    from compose_v4.benchmark.task3_run import Task3Run

    root, die_at, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    panel = json.loads(open({panel!r}).read())

    run = Task3Run.open(root, seed=11, budget=400, policy="deterministic-test")
    while run.step < steps:
        # The policy is a pure function of the RNG, so a resumed run proposes
        # exactly what the crashed one proposed.
        picks = [panel[i] for i in run.rng.integers(0, len(panel), size=8)]
        run.evaluate(picks)
        run.archive = sorted(set(run.archive) | set(picks))[:20]
        run.step += 1
        if run.step == die_at:
            run.store.flush()
            os._exit(9)            # kill -9, mid-generation, no cleanup
        run.checkpoint(policy_state={{"last_picks": picks}})
    run.close()
    print(json.dumps({{"spent": run.spent, "step": run.step,
                       "archive": run.archive,
                       "rng": json.loads(json.dumps(
                           run.rng.bit_generator.state, default=int)),
                       "evaluations": {{k: list(v) for k, v in
                                        run.meter.evaluated().items()}}}}))
''')


@pytest.mark.skipif(not BUNDLE_PRESENT, reason="frozen oracle bundle not present")
def test_a_killed_run_resumes_into_the_same_run(tmp_path):
    """Kill a real process mid-generation; the resumed run must be the SAME run.

    Not "close enough": the same molecules, the same float64 values, the same
    archive and the same RNG position as a run that was never interrupted. This
    is the property that makes a 10,000-call run safe to lose a machine over.
    """
    manifest = json.loads(
        (DEFAULT_BUNDLE_DIR / "molleo_task3_oracle_manifest.json").read_text())
    panel = [row["smiles"] for row in manifest["reference_scores"]][:60]
    panel_path = tmp_path / "panel.json"
    panel_path.write_text(json.dumps(panel))
    script = tmp_path / "run.py"
    script.write_text(RUN_SCRIPT.format(src=str(REPO_ROOT / "src"),
                                        panel=str(panel_path)))

    def execute(root: Path, die_at: int, steps: int) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(script), str(root),
                               str(die_at), str(steps)],
                              capture_output=True, text=True, cwd=REPO_ROOT)

    uninterrupted = tmp_path / "clean"
    done = execute(uninterrupted, 0, 12)
    assert done.returncode == 0, done.stderr
    expected = json.loads(done.stdout.strip().splitlines()[-1])

    crashed = tmp_path / "crashed"
    killed = execute(crashed, 5, 12)
    assert killed.returncode == 9, "the child was supposed to die"
    assert (crashed / "checkpoint.json").exists()

    resumed = execute(crashed, 0, 12)
    assert resumed.returncode == 0, resumed.stderr
    actual = json.loads(resumed.stdout.strip().splitlines()[-1])

    assert actual["spent"] == expected["spent"]
    assert actual["step"] == expected["step"]
    assert actual["archive"] == expected["archive"]
    assert actual["rng"] == expected["rng"], "the resumed run drew different randomness"
    assert actual["evaluations"] == expected["evaluations"], (
        "a resumed evaluation differs in the last bit; the oracle is not "
        "batch-invariant or the ledger is lossy")


@pytest.mark.skipif(not BUNDLE_PRESENT, reason="frozen oracle bundle not present")
def test_the_budget_survives_the_crash_rather_than_resetting(tmp_path):
    """The failure that matters most: a resumed run that thinks it is fresh
    would spend the budget twice and silently report an invalid result."""
    manifest = json.loads(
        (DEFAULT_BUNDLE_DIR / "molleo_task3_oracle_manifest.json").read_text())
    panel = [row["smiles"] for row in manifest["reference_scores"]][:60]
    panel_path = tmp_path / "panel.json"
    panel_path.write_text(json.dumps(panel))
    script = tmp_path / "run.py"
    script.write_text(RUN_SCRIPT.format(src=str(REPO_ROOT / "src"),
                                        panel=str(panel_path)))
    root = tmp_path / "run"
    killed = subprocess.run([sys.executable, str(script), str(root), "4", "12"],
                            capture_output=True, text=True, cwd=REPO_ROOT)
    assert killed.returncode == 9

    from compose_v4.benchmark.task3_run import Task3Run
    run = Task3Run.open(root, seed=11, budget=400, policy="deterministic-test")
    try:
        assert run.spent > 0
        assert run.spent == len(run.meter.evaluated())
        assert run.remaining == 400 - run.spent
        assert run.resumed is not None and run.resumed.uncheckpointed >= 0
    finally:
        run.close()
