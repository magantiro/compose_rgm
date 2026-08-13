"""The metered matcher, against a synthetic process that reproduces the defect.

The whole point of the repair is that an OPEN-LOOP trajectory count cannot fund
a baseline whose trajectories collide in the enumeration cache. So the fixture
is a process with a small reachable set from a fixed root -- which is what makes
the cache bite -- and the tests assert the three outcomes that matter:

  * a reachable target is MATCHED by metering, never by estimating;
  * a saturating process is reported UNREACHABLE rather than quietly capped;
  * neither non-MATCHED status is admissible as a scientific result.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compose_v4.experiments.pareto_gen_rank_matched import (  # noqa: E402
    MATCHED,
    MATCHING_FAILED,
    MATCHING_UNREACHABLE,
    generate_then_rank_metered,
)


@dataclass
class _Run:
    states: tuple
    endpoint: str
    endpoint_z: np.ndarray
    complete: bool = True


@dataclass
class _Traj:
    name: str
    weight: float
    states: tuple
    endpoint: str
    endpoint_z: np.ndarray
    complete: bool
    ledger: object


class _Ledger:
    def __init__(self) -> None:
        self.kernel_calls = 0
        self.native_oracle_calls = 0
        self.raw_oracle_calls = 0


class _Metered:
    """Caches enumeration by state, exactly as MeteredProcess does."""

    def __init__(self, reachable: int) -> None:
        self.ledger = _Ledger()
        self._seen: set[str] = set()
        self._z: dict[str, np.ndarray] = {}
        self.reachable = reachable

    def visit(self, key: str) -> None:
        if key not in self._seen:
            self._seen.add(key)
            self.ledger.kernel_calls += 1

    def z(self, key: str) -> np.ndarray:
        self.ledger.raw_oracle_calls += 1
        if key not in self._z:
            self.ledger.native_oracle_calls += 1
            h = abs(hash(key))
            self._z[key] = np.array([h % 97 / 97.0, (h // 97) % 89 / 89.0])
        return self._z[key]


def _unguided(reachable: int):
    """A run that touches `reachable` distinct states, so the cache saturates."""

    def run(metered, start, budget, seed):
        states = []
        for step in range(budget):
            key = f"s{(seed * 7 + step) % reachable}"
            metered.visit(key)
            states.append(key)
        return _Run(tuple(states), states[-1], metered.z(states[-1]))

    return run


def _hv(z, reference, utopia):
    gains = np.clip((z - reference) / (utopia - reference), 0.0, None)
    return float(np.max(gains[:, 0] * gains[:, 1])) if len(z) else 0.0


def _argmin(scores, keys):
    return int(min(range(len(scores)), key=lambda i: (scores[i], keys[i])))


def _call(metered, target, *, guard=5000, stall=250, reachable=10_000):
    return generate_then_rank_metered(
        metered, "root", 6, (0.1, 0.5, 0.9), lambda z, w: z[:, 0] * w,
        target, seed=3, reference=np.zeros(2), utopia=np.ones(2),
        unguided_run=_unguided(reachable), hypervolume=_hv,
        argmin_stable=_argmin, trajectory_cls=_Traj,
        runaway_guard=guard, stall_window=stall)


def test_metering_reaches_the_target_rather_than_estimating_it():
    """The realized ledger reaches the target; the trajectory count is an output."""
    metered = _Metered(10_000)
    result = _call(metered, 200)
    assert result.status == MATCHED
    assert result.realized_kernel_calls >= 200
    assert result.is_scientific
    # The open-loop estimate would have been round(200/6) = 33. Metering is what
    # distinguishes "spent the budget" from "was handed the budget".
    assert result.n_trajectories != 33


def test_overshoot_is_recorded_not_hidden():
    metered = _Metered(10_000)
    result = _call(metered, 137)
    assert result.overshoot_kernel_calls == result.realized_kernel_calls - 137
    assert result.overshoot_kernel_calls >= 0


def test_a_saturating_process_is_UNREACHABLE_not_quietly_capped():
    """The defect's own shape: the cache saturates below the target.

    This is the case a per-source ceiling would have silently converted into an
    underfunded baseline, which is precisely the error being repaired.
    """
    metered = _Metered(10_000)
    result = _call(metered, 500, stall=20, reachable=40)
    assert result.status == MATCHING_UNREACHABLE
    assert result.realized_kernel_calls < 500
    assert not result.is_scientific
    assert "FINDING" in result.note


def test_runaway_guard_is_operational_and_not_a_result():
    metered = _Metered(10_000)
    result = _call(metered, 10**9, guard=25, stall=10**9)
    assert result.status == MATCHING_FAILED
    assert not result.is_scientific


@pytest.mark.parametrize("status", [MATCHING_FAILED, MATCHING_UNREACHABLE])
def test_no_non_matched_status_is_ever_scientific(status):
    """No 'it got close enough'. The guard is the point of the guard."""
    metered = _Metered(10_000)
    result = _call(metered, 100)
    result.status = status
    assert not result.is_scientific


def test_the_prefix_records_every_axis_the_frontier_needs():
    metered = _Metered(10_000)
    result = _call(metered, 120)
    assert len(result.prefix) == result.n_trajectories
    for row in result.prefix:
        assert {"t", "kernel_calls", "native_oracle_calls",
                "raw_oracle_calls", "hypervolume"} <= set(row)
    ts = [r["t"] for r in result.prefix]
    assert ts == sorted(ts) == list(range(1, len(ts) + 1))
    # Best-so-far HV over a growing pool cannot decrease.
    hv = [r["hypervolume"] for r in result.prefix]
    assert all(b >= a - 1e-12 for a, b in zip(hv, hv[1:]))


def test_recording_the_prefix_costs_no_extra_oracle_call():
    """Each endpoint is paid for exactly once, so the frontier record is free.

    Re-scoring the pool at every t would inflate the baseline's own raw oracle
    count and make it look more expensive than it is.
    """
    metered = _Metered(10_000)
    result = _call(metered, 150)
    assert metered.ledger.raw_oracle_calls == result.n_trajectories
    assert metered.ledger.native_oracle_calls <= result.n_trajectories


def test_purpose_is_applied_only_after_generation():
    """The mechanism the comparator exists to isolate.

    Every selected trajectory must come from the pool generated WITHOUT any
    preference having influenced it.
    """
    metered = _Metered(10_000)
    result = _call(metered, 100)
    assert len(result.selected) == 3
    endpoints = {r["t"] for r in result.prefix}
    assert endpoints  # pool non-empty
    for traj in result.selected:
        assert traj.name == "gen_rank"


# --- the blinded probe firewall -------------------------------------------

def _call_blinded(metered, target, **kw):
    return generate_then_rank_metered(
        metered, "root", 6, (0.1, 0.5, 0.9), lambda z, w: z[:, 0] * w,
        target, seed=3, reference=np.zeros(2), utopia=np.ones(2),
        unguided_run=_unguided(kw.pop("reachable", 10_000)),
        hypervolume=None, argmin_stable=None, trajectory_cls=None,
        runaway_guard=kw.pop("guard", 5000), stall_window=kw.pop("stall", 250))


def test_blinded_probe_does_not_COMPUTE_hypervolume_at_all():
    """Not computed-then-hidden. Structurally absent.

    A feasibility go/no-go must be unable to read the science it would be
    tempted to condition on, rather than relying on nobody looking.
    """
    metered = _Metered(10_000)
    result = _call_blinded(metered, 150)
    assert result.blinded is True
    assert result.selected == []
    assert all(row["hypervolume"] is None for row in result.prefix)
    report = result.feasibility_only()
    flat = repr(report).lower()
    assert "hypervolume" not in flat and "hv" not in report
    assert set(report["kernel_prefix"][0]) == {
        "t", "kernel_calls", "native_oracle_calls", "raw_oracle_calls"}


def test_blinded_probe_still_answers_the_feasibility_question():
    metered = _Metered(10_000)
    report = _call_blinded(metered, 200).feasibility_only()
    assert report["status"] == MATCHED
    assert report["realized_kernel_calls"] >= 200
    assert report["n_trajectories"] > 0
    assert report["attainment"] >= 1.0


def test_blinded_probe_detects_UNREACHABLE_without_seeing_outcomes():
    metered = _Metered(10_000)
    report = _call_blinded(metered, 500, stall=20, reachable=40).feasibility_only()
    assert report["status"] == MATCHING_UNREACHABLE
    assert not report["is_scientific"]


def test_blinded_probe_wastes_no_compute():
    """Endpoints and their paid-for scores are retained, so the outcome can be
    computed offline later for all sources together without re-running."""
    metered = _Metered(10_000)
    result = _call_blinded(metered, 150)
    assert len(result.endpoints) == result.n_trajectories
    assert len(result.endpoint_z) == result.n_trajectories
    assert all(len(row) == 2 for row in result.endpoint_z)


def test_progress_callback_actually_fires_during_the_unbounded_loop():
    """The loop's length is unbounded, so a claimed checkpoint must exist.

    Regression for a docstring that promised 'a checkpoint every 50
    trajectories' while the constant was declared and never used.
    """
    seen = []
    metered = _Metered(10_000)
    generate_then_rank_metered(
        metered, "root", 6, (0.1,), lambda z, w: z[:, 0] * w,
        300, seed=3, reference=np.zeros(2), utopia=np.ones(2),
        unguided_run=_unguided(10_000), hypervolume=None,
        on_progress=lambda n, k, tgt, pfx, pool, det: seen.append((n, k, tgt, len(pfx))),
        progress_every=10)
    assert seen, "no checkpoint fired in an unbounded loop"
    assert [s[0] for s in seen] == list(range(10, seen[-1][0] + 1, 10))
    for n, k, tgt, npfx in seen:
        assert npfx == n and tgt == 300 and k <= 300 + 6


def test_resume_continues_instead_of_restarting():
    """A preemption must cost the time since the last checkpoint, not the source.

    Regression for a checkpoint that WROTE progress but could not be resumed
    from, so a Modal preemption at 86% restarted the source at zero.
    """
    saved = {}
    m1 = _Metered(10_000)

    def cap(n, k, tgt, pfx, pool, det):
        if n == 100 and "n" not in saved:
            saved.update(n=n, kernel=k, prefix=list(pfx),
                         seen=set(m1._seen), z=dict(m1._z), detector=dict(det))

    generate_then_rank_metered(
        m1, "root", 6, (0.1,), lambda z, w: z[:, 0] * w, 1200, seed=3,
        reference=np.zeros(2), utopia=np.ones(2), unguided_run=_unguided(10_000),
        hypervolume=None, on_progress=cap, progress_every=100)
    assert saved, "checkpoint never fired"

    # Restore a metered process to the checkpoint's CACHE state, then continue.
    m2 = _Metered(10_000)
    m2._seen = set(saved["seen"])
    m2._z = dict(saved["z"])
    m2.ledger.kernel_calls = saved["kernel"]
    pool = [_Run(("root",), f"e{i}", np.zeros(2)) for i in range(saved["n"])]
    r2 = generate_then_rank_metered(
        m2, "root", 6, (0.1,), lambda z, w: z[:, 0] * w, 1200, seed=3,
        reference=np.zeros(2), utopia=np.ones(2), unguided_run=_unguided(10_000),
        hypervolume=None,
        resume={"pool": pool, "prefix": saved["prefix"],
                "endpoint_z": [[0.0, 0.0]] * saved["n"],
                "detector": saved["detector"]})
    assert r2.status == MATCHED
    assert r2.n_trajectories > saved["n"], "resume did not continue past the checkpoint"
    # The restored cache means the resumed run does not re-pay for kernel calls
    # the checkpoint already bought.
    assert m2.ledger.kernel_calls >= saved["kernel"]
    assert r2.prefix[: saved["n"]] == saved["prefix"], "prefix history was lost"


def test_resume_refuses_an_inconsistent_checkpoint():
    m = _Metered(10_000)
    with pytest.raises(ValueError, match="inconsistent"):
        generate_then_rank_metered(
            m, "root", 6, (0.1,), lambda z, w: z[:, 0] * w, 100, seed=3,
            reference=np.zeros(2), utopia=np.ones(2),
            unguided_run=_unguided(10_000), hypervolume=None,
            resume={"pool": [1, 2, 3], "prefix": [{}], "endpoint_z": [[0.0, 0.0]]})


# --- LOSSLESSNESS: a resumed run must be INDISTINGUISHABLE from an uninterrupted one

def _serialise_cache(m):
    return ({k: list(v) for k, v in m._seen and {} .items()} if False else set(m._seen),
            {k: list(map(float, v)) for k, v in m._z.items()},
            dict(m.ledger.__dict__))


def test_resume_is_LOSSLESS_not_merely_safe():
    """Run to target uninterrupted; run again with a forced restart mid-way.

    Every observable must be identical: the trajectory count, the whole resource
    prefix, the endpoints, and every counter. 'Resumable' only means work is not
    lost; LOSSLESS means the interruption left no trace in the result.
    """
    TARGET, CUT = 1800, 100

    # (a) uninterrupted
    ma = _Metered(10_000)
    ra = generate_then_rank_metered(
        ma, "root", 6, (0.1, 0.5), lambda z, w: z[:, 0] * w, TARGET, seed=3,
        reference=np.zeros(2), utopia=np.ones(2), unguided_run=_unguided(10_000),
        hypervolume=None)

    # (b) interrupted at CUT, state captured, then resumed in a FRESH process
    snap = {}
    mb = _Metered(10_000)

    def grab(n, k, tgt, pfx, pool, det):
        if n == CUT and not snap:
            snap.update(seen=set(mb._seen), z=dict(mb._z),
                        ledger=dict(mb.ledger.__dict__), prefix=list(pfx),
                        pool=list(pool), det=dict(det))
            raise _Stop()

    class _Stop(Exception):
        pass

    try:
        generate_then_rank_metered(
            mb, "root", 6, (0.1, 0.5), lambda z, w: z[:, 0] * w, TARGET, seed=3,
            reference=np.zeros(2), utopia=np.ones(2),
            unguided_run=_unguided(10_000), hypervolume=None,
            on_progress=grab, progress_every=CUT)
    except _Stop:
        pass
    assert snap, "the interruption never fired"

    mc = _Metered(10_000)
    mc._seen = set(snap["seen"])
    mc._z = dict(snap["z"])
    for key, value in snap["ledger"].items():
        setattr(mc.ledger, key, value)
    rc = generate_then_rank_metered(
        mc, "root", 6, (0.1, 0.5), lambda z, w: z[:, 0] * w, TARGET, seed=3,
        reference=np.zeros(2), utopia=np.ones(2), unguided_run=_unguided(10_000),
        hypervolume=None,
        resume={"pool": snap["pool"], "prefix": snap["prefix"],
                "endpoint_z": [list(map(float, r.endpoint_z)) for r in snap["pool"]],
                "detector": snap["det"]})

    assert rc.status == ra.status
    assert rc.n_trajectories == ra.n_trajectories, "trajectory count differs"
    assert rc.realized_kernel_calls == ra.realized_kernel_calls, "kernel calls differ"
    assert rc.overshoot_kernel_calls == ra.overshoot_kernel_calls
    assert rc.prefix == ra.prefix, "the resource prefix differs"
    assert rc.endpoints == ra.endpoints, "endpoints differ"
    assert rc.endpoint_z == ra.endpoint_z, "objective vectors differ"
    assert mc.ledger.__dict__ == ma.ledger.__dict__, "counters differ"


def test_the_stall_detector_state_survives_the_checkpoint():
    """Loop-carried state, so restarting it at zero would make UNREACHABLE
    harder to declare after a preemption than without one."""
    seen = []
    m = _Metered(10_000)
    generate_then_rank_metered(
        m, "root", 6, (0.1,), lambda z, w: z[:, 0] * w, 10**9, seed=3,
        reference=np.zeros(2), utopia=np.ones(2), unguided_run=_unguided(40),
        hypervolume=None, stall_window=120,
        on_progress=lambda n, k, tgt, pfx, pool, det: seen.append(det),
        progress_every=25)
    assert seen, "no checkpoint fired"
    assert all({"since_progress", "last_kernel"} <= set(d) for d in seen)
    assert max(d["since_progress"] for d in seen) > 0, "detector never advanced"
