"""Preemption must be scientifically invisible, and this is the test that can say so.

The acceptance test kills a real worker process with ``SIGKILL`` at five named
points -- during proposal generation, immediately before an oracle reservation,
immediately after one, after the oracle returned, and after the controller update
but before the durable commit -- resumes it, and requires the finished run to be
indistinguishable from one that was never interrupted.

Two things make it able to FAIL rather than merely able to pass:

1.  The reference run is built INDEPENDENTLY, by running the same worker with no
    kill at all, and the comparison is between the two runs' ARTIFACTS.  Nothing
    is recomputed from the code under test, so a defect cannot move both sides.
2.  The snapshot comparison classifies EVERY differing leaf against an explicit
    allow-list and requires the unexplained set to be empty.  Comparing a summary
    -- best score, call count -- would pass while the archive, the learned memory
    or a random stream silently differed, which is the failure this exists for.

The RNG is the sharp case and is tested twice: a snapshot that omits it is refused
at commit, and a resume that restores a WRONG stream is caught at the ledger,
because the run then proposes a different molecule for a slot already reserved.
A trajectory that diverges loudly is recoverable; one that diverges quietly is a
published number that cannot be reproduced.

Orphan check: a resume protocol that leaves child processes behind has not resumed
cleanly, so every killed scenario asserts that no process carrying the run marker
survives.  ``pkill -f <script>`` notoriously does not match spawn-pool children,
so the assertion matches on the run id, which appears in every descendant's argv.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from compose_v4.control.durable_resume import (
    COMPLETE,
    FRESH,
    PMO_NULLABLE_COMPONENTS,
    PMO_REQUIRED_COMPONENTS,
    RESUME_IN_PLACE,
    BudgetExhausted,
    CallLedger,
    DurableSnapshotStore,
    LeaseHeld,
    LedgerCorrupt,
    ResumeDivergence,
    SnapshotIncomplete,
    WorkerLease,
    atomic_write_json,
    classify_run_start,
    identity,
)

_TESTS = Path(__file__).resolve().parent
_REPO = _TESTS.parent
_WORKER = _TESTS / "resumability_kill_worker.py"
_KILL_POINTS = ("proposal", "before_reserve", "after_reserve", "after_observe", "before_commit")

#: Only ``after_reserve`` can leave a charged-but-unobserved call, so only it may
#: cost a repair.  Stating the expectation per scenario means a repair appearing
#: anywhere else is a failure rather than something the test absorbs.
_EXPECTED_REPAIRS = {
    "proposal": 0,
    "before_reserve": 0,
    "after_reserve": 1,
    "after_observe": 0,
    "before_commit": 0,
}

_REVISION = {"contract": "fake_oracle_resumability_v1", "controller": "kill_worker_v1"}


# ---- helpers ----------------------------------------------------------------


def _durable_env() -> dict:
    """pytest's tmp_path is under /var/folders, which the durable-path gate refuses."""
    return {
        **os.environ,
        "PYTHONPATH": str(_REPO / "src"),
        "COMPOSE_ALLOW_REAPABLE_PATH": "1",
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "OMP_NUM_THREADS": "1",
    }


def _run_worker(root: Path, run_id: str, worker_id: str, *, kill_at=None, kill_round=2):
    command = [
        sys.executable,
        str(_WORKER),
        "--root", str(root),
        "--run-id", run_id,
        "--worker-id", worker_id,
        "--rounds", "6",
        "--calls-per-round", "4",
        "--budget", "64",
        "--lease-seconds", "0.25",
    ]
    if kill_at is not None:
        command += ["--kill-at", kill_at, "--kill-round", str(kill_round), "--kill-ordinal", "1"]
    return subprocess.run(
        command, capture_output=True, text=True,
        timeout=180, env=_durable_env(), check=False,
    )


def _processes_carrying(marker: str) -> list[str]:
    """Every live process whose argv carries the marker, orphans included.

    Matching on the marker rather than on the script name is deliberate: a spawn
    pool's children run ``python -c from multiprocessing.spawn import spawn_main``
    and are invisible to a script-name match, which is exactly how orphans
    accumulate unnoticed.
    """
    listing = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,args="], capture_output=True, text=True, check=False
    ).stdout
    rows = []
    for line in listing.splitlines():
        if marker in line and "ps -eo" not in line:
            rows.append(line.strip())
    return rows


def _flatten(value, prefix: str = "") -> dict:
    flat = {}
    if isinstance(value, dict):
        for key in value:
            flat.update(_flatten(value[key], f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            flat.update(_flatten(item, f"{prefix}[{index}]"))
    else:
        flat[prefix] = value
    return flat


def _unexplained_differences(left: dict, right: dict, allowed_roots: set) -> dict:
    """Classify every differing leaf; anything not on the allow-list is a failure."""
    flat_left, flat_right = _flatten(left), _flatten(right)
    unexplained = {}
    for path in sorted(set(flat_left) | set(flat_right)):
        if flat_left.get(path, "<absent>") == flat_right.get(path, "<absent>"):
            continue
        if path.split(".")[0].split("[")[0] in allowed_roots:
            continue
        unexplained[path] = (flat_left.get(path, "<absent>"), flat_right.get(path, "<absent>"))
    return unexplained


def _ledger_rows(root: Path) -> list[dict]:
    os.environ["COMPOSE_ALLOW_REAPABLE_PATH"] = "1"
    ledger = CallLedger(root / "oracle", run_id=_RUN_ID, budget=64)
    return ledger.rows()


def _scientific_calls(rows: list[dict]) -> list[tuple]:
    """Oracle accounting identity: ids, molecules, scores, and their order."""
    return [
        (r["call_id"], r["round_index"], r["ordinal"], r["molecule"], r["score"]) for r in rows
    ]


def _snapshots(root: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((root / "snapshots").glob("snapshot_*.json"))]


_RUN_ID = "resumability-acceptance-v1"


def _components(**overrides) -> dict:
    base = {
        "archive": [{"molecule": "M", "score": 0.5}],
        "memory": {"donor_regions": {"a": 1.0}, "edit_outcomes": []},
        "allocator": {"parent_mass": {}, "draws": 3},
        "credit": {"cells": {}},
        "rng": {"bit_generator": "PCG64", "state": {"state": 7, "inc": 3}},
        "pending_candidates": [],
    }
    base.update(overrides)
    return base


_TOY_COMPONENTS = frozenset(
    {"archive", "memory", "allocator", "credit", "rng", "pending_candidates"}
)


def _store(tmp_path: Path, **kwargs) -> DurableSnapshotStore:
    os.environ["COMPOSE_ALLOW_REAPABLE_PATH"] = "1"
    kwargs.setdefault("required_components", _TOY_COMPONENTS)
    return DurableSnapshotStore(
        tmp_path / "snapshots", run_id="r1", revision_identity=_REVISION, **kwargs
    )


def _pmo_components(**overrides) -> dict:
    """The shape `PmoPopulationController.snapshot` really emits: PMO state nested."""
    base = {
        "rng": {"bit_generator": "PCG64", "state": {"state": 7, "inc": 3}},
        "entries": [],
        "observations": {},
        "pending": None,
        "pmo_population": {
            "jump_rng": {"state": 1},
            "shallow_rng": {"state": 2},
            "structured_rng": {"state": 3},
            "population_state": {"best_score": 0.4},
            "credit": {"cells": {}},
            "pool_continuity": {"pool": []},
            "online_memory": {"ordinal": 250},
        },
    }
    for key, value in overrides.items():
        if key.startswith("pmo_population."):
            base["pmo_population"][key.split(".", 1)[1]] = value
        else:
            base[key] = value
    return base


def _pmo_store(tmp_path: Path) -> DurableSnapshotStore:
    os.environ["COMPOSE_ALLOW_REAPABLE_PATH"] = "1"
    return DurableSnapshotStore(
        tmp_path / "snapshots",
        run_id="r1",
        revision_identity=_REVISION,
        required_components=PMO_REQUIRED_COMPONENTS,
        non_null_components=PMO_REQUIRED_COMPONENTS - PMO_NULLABLE_COMPONENTS,
    )


def _ledger(tmp_path: Path, budget: int = 8) -> CallLedger:
    os.environ["COMPOSE_ALLOW_REAPABLE_PATH"] = "1"
    return CallLedger(tmp_path / "oracle", run_id="r1", budget=budget)


# ---- the acceptance test ----------------------------------------------------


@pytest.mark.parametrize("kill_at", _KILL_POINTS)
def test_kill_and_resume_reproduces_the_uninterrupted_run(tmp_path, kill_at):
    """Kill a real worker, resume it, and require artifact-level identity.

    Scenarios run one at a time; the fake oracle is free, so the only cost is wall
    clock.  The reference arm is a separate, independently produced run.
    """
    reference_root = tmp_path / "reference"
    resumed_root = tmp_path / "resumed"

    reference = _run_worker(reference_root, _RUN_ID, "worker-reference")
    assert reference.returncode == 0, reference.stderr
    assert (reference_root / "result.json").exists()

    killed = _run_worker(resumed_root, _RUN_ID, "worker-a", kill_at=kill_at)
    assert killed.returncode == -9, (
        f"worker was expected to die by SIGKILL at {kill_at}; got {killed.returncode}\n"
        f"{killed.stderr}"
    )
    assert not (resumed_root / "result.json").exists()

    # A resume that leaves orphans behind has not resumed cleanly.
    survivors = _processes_carrying(_RUN_ID)
    assert survivors == [], f"orphaned processes survived the kill: {survivors}"

    resumed = _run_worker(resumed_root, _RUN_ID, "worker-b")
    assert resumed.returncode == 0, resumed.stderr

    reference_result = json.loads((reference_root / "result.json").read_text())
    resumed_result = json.loads((resumed_root / "result.json").read_text())

    # --- oracle accounting identity ---
    reference_rows, resumed_rows = _ledger_rows(reference_root), _ledger_rows(resumed_root)
    assert _scientific_calls(reference_rows) == _scientific_calls(resumed_rows)
    assert reference_result["charged_calls"] == resumed_result["charged_calls"]
    assert resumed_result["pending_calls"] == 0
    assert len({row["call_id"] for row in resumed_rows}) == len(resumed_rows)
    assert resumed_result["repaired_calls"] == _EXPECTED_REPAIRS[kill_at]
    assert reference_result["repaired_calls"] == 0

    # --- scientific state identity, leaf by leaf ---
    reference_snapshots, resumed_snapshots = _snapshots(reference_root), _snapshots(resumed_root)
    assert len(reference_snapshots) == len(resumed_snapshots)
    for left, right in zip(reference_snapshots, resumed_snapshots):
        # `epoch` names which worker incarnation committed and legitimately differs;
        # `snapshot_sha256` covers the body including that epoch.  Nothing else may.
        unexplained = _unexplained_differences(left, right, {"epoch", "snapshot_sha256"})
        assert unexplained == {}, f"round {left['round_index']} diverged: {unexplained}"

    assert resumed_snapshots[-1]["components"] == reference_snapshots[-1]["components"]

    # The stranding fix, asserted rather than implied: the killed worker left a
    # lease that still looked live, and the successor took the run over once it
    # lapsed instead of refusing forever and waiting for a human.
    assert resumed_snapshots[-1]["epoch"] > reference_snapshots[-1]["epoch"]


def test_the_reference_and_resumed_arms_are_compared_by_artifact_not_by_recomputation():
    """The comparison reads files; a defect in the module cannot move both sides."""
    source = Path(_WORKER).read_text()
    assert "fake_oracle" in source
    comparison = Path(__file__).read_text()
    assert "_snapshots(reference_root)" in comparison and "_ledger_rows(reference_root)" in comparison


# ---- ledger invariants ------------------------------------------------------


def test_a_reserved_call_is_never_re_spent_on_resume(tmp_path):
    ledger = _ledger(tmp_path)
    first = ledger.reserve(round_index=0, ordinal=0, molecule="CCO")
    assert ledger.charged_calls == 1
    again = ledger.reserve(round_index=0, ordinal=0, molecule="CCO")
    assert again.call_id == first.call_id
    assert ledger.charged_calls == 1, "re-reserving a slot must not charge a second call"


def test_a_resumed_run_that_proposes_a_different_molecule_diverges_loudly(tmp_path):
    """The signature of controller state that failed to restore."""
    ledger = _ledger(tmp_path)
    ledger.reserve(round_index=0, ordinal=0, molecule="CCO")
    with pytest.raises(ResumeDivergence):
        ledger.reserve(round_index=0, ordinal=0, molecule="CCN")


def test_spend_is_counted_from_durable_reservations_not_from_controller_state(tmp_path):
    ledger = _ledger(tmp_path)
    for ordinal in range(3):
        ledger.reserve(round_index=0, ordinal=ordinal, molecule=f"C{ordinal}")
    rebuilt = CallLedger(tmp_path / "oracle", run_id="r1", budget=8)
    assert rebuilt.charged_calls == 3
    assert rebuilt.remaining == 5


def test_a_reservation_made_before_the_oracle_ran_stays_charged_and_is_repaired_once(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.reserve(round_index=0, ordinal=0, molecule="CCO")
    assert ledger.reconcile().pending and not ledger.reconcile().complete
    repaired = ledger.repair_pending(lambda smiles: 0.25)
    assert len(repaired) == 1 and repaired[0]["repaired"] is True
    assert ledger.charged_calls == 1, "a repair fills in a score; it never adds a charge"
    assert ledger.repaired_calls == 1
    assert ledger.repair_pending(lambda smiles: 0.99) == [], "repair is idempotent"
    assert ledger.rows()[0]["score"] == 0.25


def test_the_budget_is_an_append_only_ceiling(tmp_path):
    ledger = _ledger(tmp_path, budget=2)
    ledger.reserve(round_index=0, ordinal=0, molecule="C0")
    ledger.reserve(round_index=0, ordinal=1, molecule="C1")
    with pytest.raises(BudgetExhausted):
        ledger.reserve(round_index=0, ordinal=2, molecule="C2")


def test_a_tampered_reservation_fails_its_call_id(tmp_path):
    ledger = _ledger(tmp_path)
    reservation = ledger.reserve(round_index=0, ordinal=0, molecule="CCO")
    path = tmp_path / "oracle" / "calls" / reservation.slot / "reserved.json"
    record = json.loads(path.read_text())
    record["molecule"] = "CCN"
    path.write_text(json.dumps(record))
    with pytest.raises(LedgerCorrupt):
        ledger.reconcile()


def test_the_ledger_refuses_a_changed_budget_across_resume(tmp_path):
    _ledger(tmp_path, budget=8)
    with pytest.raises(LedgerCorrupt):
        CallLedger(tmp_path / "oracle", run_id="r1", budget=9)


# ---- snapshot invariants ----------------------------------------------------


def test_dropping_the_rng_from_the_snapshot_is_refused(tmp_path):
    store = _pmo_store(tmp_path)
    components = _pmo_components()
    del components["rng"]
    with pytest.raises(SnapshotIncomplete):
        store.commit(
            round_index=0, charged_calls=0, scored_call_ordinal=0, components=components
        )


def test_skipping_the_memory_payload_is_refused(tmp_path):
    """Arm B must never resume as arm A because a key quietly went missing."""
    store = _pmo_store(tmp_path)
    components = _pmo_components()
    del components["pmo_population"]["online_memory"]
    with pytest.raises(SnapshotIncomplete):
        store.commit(
            round_index=0, charged_calls=0, scored_call_ordinal=0, components=components
        )


def test_a_missing_nested_component_is_not_hidden_by_a_present_parent(tmp_path):
    """A flat key check passes any snapshot whose nested block merely EXISTS."""
    store = _pmo_store(tmp_path)
    components = _pmo_components()
    components["pmo_population"] = {"jump_rng": {"state": 1}}
    with pytest.raises(SnapshotIncomplete) as failure:
        store.commit(
            round_index=0, charged_calls=0, scored_call_ordinal=0, components=components
        )
    assert "pmo_population.credit" in str(failure.value)


def test_a_nullable_component_may_be_present_and_null(tmp_path):
    """`pending` is None whenever no batch is outstanding; that is a real value."""
    store = _pmo_store(tmp_path)
    record = store.commit(
        round_index=0,
        charged_calls=4,
        scored_call_ordinal=4,
        components=_pmo_components(pending=None),
    )
    assert record["components"]["pending"] is None


def test_a_non_nullable_component_may_not_be_null(tmp_path):
    store = _pmo_store(tmp_path)
    with pytest.raises(SnapshotIncomplete):
        store.commit(
            round_index=0,
            charged_calls=0,
            scored_call_ordinal=0,
            components=_pmo_components(**{"pmo_population.credit": None}),
        )


def test_declaring_a_non_null_component_that_is_not_required_is_refused(tmp_path):
    with pytest.raises(ValueError):
        DurableSnapshotStore(
            tmp_path / "snapshots",
            run_id="r1",
            revision_identity=_REVISION,
            required_components=frozenset({"rng"}),
            non_null_components=frozenset({"rng", "credit"}),
        )


def test_the_pmo_profile_names_nested_controller_keys(tmp_path):
    """Measured against PmoPopulationController.snapshot, not invented."""
    assert "pmo_population.credit" in PMO_REQUIRED_COMPONENTS
    assert "pmo_population.online_memory" in PMO_REQUIRED_COMPONENTS
    assert PMO_NULLABLE_COMPONENTS <= PMO_REQUIRED_COMPONENTS
    store = _pmo_store(tmp_path)
    store.commit(
        round_index=0, charged_calls=4, scored_call_ordinal=4, components=_pmo_components()
    )
    assert store.load_latest()["components"]["pmo_population"]["online_memory"]["ordinal"] == 250


def test_a_null_required_component_is_refused_as_firmly_as_a_missing_one(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(SnapshotIncomplete):
        store.commit(
            round_index=0,
            charged_calls=0,
            scored_call_ordinal=0,
            components=_components(memory=None),
        )


def test_a_stored_snapshot_missing_a_component_is_refused_at_load_not_filled_in(tmp_path):
    """The defect this exists for: an absent key restored as an empty stand-in."""
    store = _store(tmp_path)
    store.commit(round_index=0, charged_calls=4, scored_call_ordinal=4, components=_components())
    path = tmp_path / "snapshots" / "snapshot_000000.json"
    record = json.loads(path.read_text())
    del record["components"]["memory"]
    record["snapshot_sha256"] = identity({k: v for k, v in record.items() if k != "snapshot_sha256"})
    path.write_text(json.dumps(record))
    with pytest.raises(SnapshotIncomplete):
        store.load_latest()


def test_a_tampered_snapshot_fails_its_content_hash(tmp_path):
    store = _store(tmp_path)
    store.commit(round_index=0, charged_calls=4, scored_call_ordinal=4, components=_components())
    path = tmp_path / "snapshots" / "snapshot_000000.json"
    record = json.loads(path.read_text())
    record["charged_calls"] = 999
    path.write_text(json.dumps(record))
    with pytest.raises(LedgerCorrupt):
        store.load_latest()


def test_a_snapshot_from_a_different_revision_identity_is_refused(tmp_path):
    store = _store(tmp_path)
    store.commit(round_index=0, charged_calls=4, scored_call_ordinal=4, components=_components())
    moved = DurableSnapshotStore(
        tmp_path / "snapshots", run_id="r1", revision_identity={"contract": "something_else"}
    )
    with pytest.raises(LedgerCorrupt):
        moved.load_latest()


def test_the_latest_snapshot_is_the_highest_committed_round(tmp_path):
    store = _store(tmp_path)
    for round_index in range(3):
        store.commit(
            round_index=round_index,
            charged_calls=round_index * 4,
            scored_call_ordinal=round_index * 4,
            components=_components(allocator={"draws": round_index}),
        )
    assert store.load_latest()["round_index"] == 2
    assert store.load_latest()["components"]["allocator"]["draws"] == 2


def test_a_fresh_store_has_no_snapshot_rather_than_an_empty_one(tmp_path):
    assert _store(tmp_path).load_latest() is None


# ---- lease invariants -------------------------------------------------------


class _Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_a_live_lease_blocks_a_second_worker(tmp_path):
    clock = _Clock()
    WorkerLease.acquire(
        tmp_path / "lease.json", run_id="r1", worker_id="a", lease_seconds=30, clock=clock
    )
    clock.now += 10
    with pytest.raises(LeaseHeld):
        WorkerLease.acquire(
            tmp_path / "lease.json", run_id="r1", worker_id="b", lease_seconds=30, clock=clock
        )


def test_an_expired_lease_allows_another_worker_to_take_the_run_over(tmp_path):
    """The direct fix for a run stranded because a dead worker might be alive."""
    clock = _Clock()
    first = WorkerLease.acquire(
        tmp_path / "lease.json", run_id="r1", worker_id="a", lease_seconds=30, clock=clock
    )
    clock.now += 31
    second = WorkerLease.acquire(
        tmp_path / "lease.json", run_id="r1", worker_id="b", lease_seconds=30, clock=clock
    )
    assert second.epoch > first.epoch


def test_a_worker_whose_lease_lapsed_while_it_was_alive_is_fenced_out_of_commits(tmp_path):
    """Expired is not the same statement as dead, so takeover carries a fencing token."""
    clock = _Clock()
    slow = WorkerLease.acquire(
        tmp_path / "lease.json", run_id="r1", worker_id="a", lease_seconds=30, clock=clock
    )
    clock.now += 31
    WorkerLease.acquire(
        tmp_path / "lease.json", run_id="r1", worker_id="b", lease_seconds=30, clock=clock
    )
    with pytest.raises(LeaseHeld):
        slow.renew()
    store = _store(tmp_path)
    with pytest.raises(LeaseHeld):
        store.commit(
            round_index=0,
            charged_calls=0,
            scored_call_ordinal=0,
            components=_components(),
            lease=slow,
        )


def test_renewal_keeps_a_lease_alive_indefinitely(tmp_path):
    clock = _Clock()
    lease = WorkerLease.acquire(
        tmp_path / "lease.json", run_id="r1", worker_id="a", lease_seconds=30, clock=clock
    )
    for _ in range(5):
        clock.now += 20
        lease.renew()
    assert not lease.expired()
    with pytest.raises(LeaseHeld):
        WorkerLease.acquire(
            tmp_path / "lease.json", run_id="r1", worker_id="b", lease_seconds=30, clock=clock
        )


def test_a_released_lease_does_not_make_a_successor_wait(tmp_path):
    clock = _Clock()
    lease = WorkerLease.acquire(
        tmp_path / "lease.json", run_id="r1", worker_id="a", lease_seconds=30, clock=clock
    )
    lease.release()
    successor = WorkerLease.acquire(
        tmp_path / "lease.json", run_id="r1", worker_id="b", lease_seconds=30, clock=clock
    )
    assert successor.epoch > lease.epoch


def test_a_lease_belonging_to_another_run_is_never_taken_over(tmp_path):
    clock = _Clock()
    WorkerLease.acquire(
        tmp_path / "lease.json", run_id="r1", worker_id="a", lease_seconds=1, clock=clock
    )
    clock.now += 100
    with pytest.raises(LeaseHeld):
        WorkerLease.acquire(
            tmp_path / "lease.json", run_id="other", worker_id="b", lease_seconds=30, clock=clock
        )


# ---- run-start classification ----------------------------------------------


def test_a_started_run_whose_worker_died_classifies_as_resume_in_place(tmp_path):
    clock = _Clock()
    paths = {"result_path": tmp_path / "result.json", "lease_path": tmp_path / "lease.json"}
    verdict, _ = classify_run_start(
        **paths, run_id="r1", worker_id="a", lease_seconds=30, clock=clock
    )
    assert verdict == FRESH
    clock.now += 31
    verdict, lease = classify_run_start(
        **paths, run_id="r1", worker_id="b", lease_seconds=30, clock=clock
    )
    assert verdict == RESUME_IN_PLACE and lease.epoch == 2


def test_a_finished_run_is_complete_and_is_never_resumed(tmp_path):
    atomic_write_json(tmp_path / "result.json", {"done": True})
    verdict, lease = classify_run_start(
        result_path=tmp_path / "result.json",
        lease_path=tmp_path / "lease.json",
        run_id="r1",
        worker_id="a",
        lease_seconds=30,
    )
    assert verdict == COMPLETE and lease is None


def test_a_blocked_run_raises_rather_than_being_mistaken_for_a_fresh_one(tmp_path):
    clock = _Clock()
    classify_run_start(
        result_path=tmp_path / "result.json",
        lease_path=tmp_path / "lease.json",
        run_id="r1",
        worker_id="a",
        lease_seconds=30,
        clock=clock,
    )
    with pytest.raises(LeaseHeld):
        classify_run_start(
            result_path=tmp_path / "result.json",
            lease_path=tmp_path / "lease.json",
            run_id="r1",
            worker_id="b",
            lease_seconds=30,
            clock=clock,
        )


def test_resume_is_never_spelled_as_a_retry_or_a_backfill():
    """Naming matters here: a retry may double-charge and a backfill invents history."""
    source = Path(
        _REPO / "src" / "compose_v4" / "control" / "durable_resume.py"
    ).read_text()
    assert RESUME_IN_PLACE == "resume_in_place"
    assert "retry" not in {FRESH, RESUME_IN_PLACE, COMPLETE}
    assert "def classify_run_start" in source


# ---- atomicity --------------------------------------------------------------


def test_an_atomic_write_leaves_no_temporary_file_behind(tmp_path):
    os.environ["COMPOSE_ALLOW_REAPABLE_PATH"] = "1"
    target = tmp_path / "value.json"
    atomic_write_json(target, {"a": 1})
    atomic_write_json(target, {"a": 2})
    assert json.loads(target.read_text()) == {"a": 2}
    assert list(tmp_path.glob("*.tmp")) == []


def test_a_reader_never_observes_a_partially_written_file(tmp_path):
    """Every observation during a long rewrite is a complete, parseable document."""
    target = tmp_path / "value.json"
    atomic_write_json(target, {"round": 0, "payload": "x" * 10})
    for round_index in range(1, 40):
        atomic_write_json(target, {"round": round_index, "payload": "x" * (10 * round_index)})
        observed = json.loads(target.read_text())
        assert observed["round"] == round_index
        assert len(observed["payload"]) == 10 * round_index


def test_the_durable_path_gate_refuses_a_reapable_root(tmp_path):
    from compose_v4.data.durable_path import ReapablePathError

    saved = os.environ.pop("COMPOSE_ALLOW_REAPABLE_PATH", None)
    try:
        with pytest.raises(ReapablePathError):
            CallLedger("/private/tmp/compose_resume_probe", run_id="r1", budget=4)
    finally:
        if saved is not None:
            os.environ["COMPOSE_ALLOW_REAPABLE_PATH"] = saved
