"""Mechanics tests for the pathwise / hard-support arms. No controller needed.

Every controller here is a toy stub. That is the point: if these tests required
region-`h_phi`, the module would not actually be controller-agnostic.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from compose_v4.experiments.constrained_control import (
    ConstrainedTrajectory,
    Enforcement,
    noninferiority_verdict,
    p_hidden,
    run_constrained_trajectory,
    source_clustered_bootstrap,
)


class ScriptedController:
    """Walks a fixed script, skipping steps the support restriction forbids."""

    def __init__(self, script: list[str]) -> None:
        self.script = script
        self.i = 0

    def propose(self, state, source, budget_remaining, rng,
                allowed: Callable[[str], bool] | None = None):
        while self.i < len(self.script):
            cand = self.script[self.i]
            self.i += 1
            if allowed is None or allowed(cand):
                return cand
        return None


def in_corridor(smi: str) -> bool:
    """Toy predicate: uppercase conforms, lowercase violates."""
    return smi.isupper()


def test_pathwise_never_commits_a_forbidden_state():
    ctrl = ScriptedController(["A", "b", "C", "d", "E"])
    t = run_constrained_trajectory(
        ctrl, in_corridor, Enforcement.PATHWISE, "S", 3,
        np.random.default_rng(0))
    assert t.path == ("S", "A", "C", "E")
    assert t.violations == ()
    assert not t.visited_forbidden


def test_endpoint_only_records_violations_it_does_not_prevent():
    ctrl = ScriptedController(["A", "b", "C"])
    t = run_constrained_trajectory(
        ctrl, in_corridor, Enforcement.ENDPOINT_ONLY, "S", 3,
        np.random.default_rng(0))
    assert t.path == ("S", "A", "b", "C")
    assert t.violations == (2,)          # the hidden intermediate, recorded
    assert t.visited_forbidden


def test_source_is_never_counted_as_a_violation():
    # A lowercase source violates the predicate but is not a committed edit.
    ctrl = ScriptedController(["A"])
    t = run_constrained_trajectory(
        ctrl, in_corridor, Enforcement.ENDPOINT_ONLY, "s", 1,
        np.random.default_rng(0))
    assert t.violations == ()


def test_nonconforming_controller_fails_loudly_under_pathwise():
    class Rogue:
        def propose(self, state, source, budget_remaining, rng, allowed=None):
            return "b"                    # ignores the restriction entirely

    with pytest.raises(RuntimeError, match="violates the support restriction"):
        run_constrained_trajectory(
            Rogue(), in_corridor, Enforcement.PATHWISE, "S", 2,
            np.random.default_rng(0))


def test_halting_early_is_recorded():
    t = run_constrained_trajectory(
        ScriptedController([]), in_corridor, Enforcement.NONE, "S", 4,
        np.random.default_rng(0))
    assert t.halted_early and t.path == ("S",) and t.edits == 0


def _traj(path: str, violations: tuple[int, ...]) -> ConstrainedTrajectory:
    return ConstrainedTrajectory(
        source=path[0], path=tuple(path), enforcement=Enforcement.ENDPOINT_ONLY,
        halted_early=False, violations=violations)


def test_p_hidden_counts_only_conforming_endpoints_with_bad_intermediates():
    trajs = [
        _traj("SabC", (1, 2)),   # hidden: endpoint fine, path was not
        _traj("SABC", ()),       # clean throughout
        _traj("SABd", (3,)),     # endpoint itself violates -> NOT hidden
        _traj("SaBc", (1, 3)),   # endpoint violates -> NOT hidden
    ]
    assert p_hidden(trajs) == pytest.approx(0.25)


def test_p_hidden_refuses_an_empty_sequence():
    # A silent 0.0 would read as "no hidden-path behaviour".
    with pytest.raises(ValueError, match="undefined"):
        p_hidden([])


def test_bootstrap_resamples_sources_not_observations():
    # One source with an extreme value must be able to drop out entirely,
    # which is only possible if resampling happens at the source level.
    data = {"a": [0.0] * 20, "b": [0.0] * 20, "c": [100.0] * 20}
    point, lo, hi = source_clustered_bootstrap(
        data, lambda v: float(v.mean()), np.random.default_rng(0),
        resamples=2000)
    assert point == pytest.approx(100.0 / 3.0)
    assert lo == pytest.approx(0.0)      # all three draws missed source "c"
    assert hi > point                     # and sometimes drew it repeatedly


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(7)
    data = {f"s{i}": rng.normal(0.5, 1.0, 8).tolist() for i in range(40)}
    point, lo, hi = source_clustered_bootstrap(
        data, lambda v: float(v.mean()), np.random.default_rng(1),
        resamples=2000)
    assert lo < point < hi


def test_noninferiority_is_a_strict_inequality_on_the_lower_bound():
    assert noninferiority_verdict(-0.24, 0.25)
    assert not noninferiority_verdict(-0.26, 0.25)
    assert not noninferiority_verdict(-0.25, 0.25)   # boundary does NOT pass


def test_stricter_sensitivity_can_disagree_with_the_primary():
    # The preregistration requires reporting both; this is the case that makes
    # the requirement matter.
    assert noninferiority_verdict(-0.22, 0.25)
    assert not noninferiority_verdict(-0.22, 0.20)
