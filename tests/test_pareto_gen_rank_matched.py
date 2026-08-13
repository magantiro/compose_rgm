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
        on_progress=lambda n, k, tgt, pfx: seen.append((n, k, tgt, len(pfx))),
        progress_every=10)
    assert seen, "no checkpoint fired in an unbounded loop"
    assert [s[0] for s in seen] == list(range(10, seen[-1][0] + 1, 10))
    for n, k, tgt, npfx in seen:
        assert npfx == n and tgt == 300 and k <= 300 + 6
