"""Bounded checks for duplicate barriers and contended periodic flushes."""

import threading

import pytest

from compose_v4.experiments import pmo_archive_pilot as pilot
from compose_v4.experiments.pmo_branch_policy import _frozen_save


def test_immutable_save_has_one_barrier_even_when_periodic_flush_is_due(tmp_path, monkeypatch):
    clock, calls = [0.0], []
    monkeypatch.setattr(pilot, "perf_counter", lambda: clock[0])
    store = pilot.Store(tmp_path, lambda: calls.append("commit"))
    clock[0] = 31.0
    _frozen_save(store, "candidate_lock", {"candidates": ["CC"]})
    assert store.read("candidate_lock") == {"candidates": ["CC"]}
    assert calls == ["commit"] and store.timings["commits"] == 1
    with pytest.raises(ValueError, match="changed locked"):
        _frozen_save(store, "candidate_lock", {"candidates": ["CCC"]})
    # Explicit oracle barriers stay unconditional, even without a Store.save.
    store.flush(force=True)
    store.flush(force=True)
    assert len(calls) == 3


def test_waiting_periodic_flush_rechecks_after_forced_barrier(tmp_path, monkeypatch):
    clock, calls, errors = [0.0], [], []
    entered, release, periodic_started = threading.Event(), threading.Event(), threading.Event()
    monkeypatch.setattr(pilot, "perf_counter", lambda: clock[0])

    def commit():
        calls.append("commit")
        entered.set()
        if not release.wait(2):
            raise TimeoutError("fixture commit was not released")
        clock[0] = 32.0

    store = pilot.Store(tmp_path, commit)
    clock[0] = 31.0

    def flush(force):
        try:
            if not force:
                periodic_started.set()
            store.flush(force=force)
        except TimeoutError as error:
            errors.append(error)

    forced = threading.Thread(target=flush, args=(True,))
    periodic = threading.Thread(target=flush, args=(False,))
    forced.start()
    assert entered.wait(2)
    periodic.start()
    assert periodic_started.wait(2)
    release.set()
    forced.join(2)
    periodic.join(2)
    assert not forced.is_alive() and not periodic.is_alive() and not errors
    assert calls == ["commit"] and store.timings["commits"] == 1
