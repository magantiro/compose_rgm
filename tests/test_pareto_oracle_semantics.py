"""The semantic oracle correction, against a CLEANLY INSTRUMENTED fixture.

The correction claims that `raw - n_completed_trajectories` recovers what the
method actually demanded. That claim is only worth anything if it is checked
against a run where every request was TAGGED at its call site as algorithmic or
harness-only, rather than against the derivation restating itself.

So the fixture below re-implements the four arms in miniature with a meter that
records a tag per request, and the tests assert the post-hoc derivation lands on
the independently tagged count.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compose_v4.experiments.pareto_oracle_semantics import (  # noqa: E402
    corrected_cost,
    corrected_shard,
    harness_only_requests,
)

N_PREF = 5


class TaggedMeter:
    """Counts every request AND what it was for. The independent ground truth."""

    def __init__(self) -> None:
        self.algorithmic = 0
        self.harness = 0
        self.native = 0
        self._seen: set[str] = set()

    def z(self, key: str, *, tag: str) -> float:
        assert tag in ("algorithmic", "harness")
        setattr(self, tag, getattr(self, tag) + 1)
        if key not in self._seen:
            self._seen.add(key)
            self.native += 1
        return (abs(hash(key)) % 1000) / 1000.0

    @property
    def raw(self) -> int:
        return self.algorithmic + self.harness


def _unguided_run(meter: TaggedMeter, seed: int, budget: int = 6) -> str:
    """Samples from the reference law. Reads NO objective while generating."""
    end = f"m{seed}"
    meter.z(end, tag="harness")          # line 218: terminal, populates endpoint_z
    return end


def _greedy_run(meter: TaggedMeter, weight: int, budget: int = 6,
                fiber: int = 40) -> str:
    current = "root"
    for step in range(budget):
        for c in range(fiber):           # line 233: selects the action
            meter.z(f"c{weight}_{step}_{c}", tag="algorithmic")
        current = f"c{weight}_{step}_0"
    meter.z(current, tag="harness")      # line 237: terminal
    return current


def _verified_run(meter: TaggedMeter, weight: int, budget: int = 6,
                  fiber: int = 40, shortlist: int = 3) -> str:
    current = "root"
    for step in range(budget):
        for c in range(fiber):           # line 298
            meter.z(f"v{weight}_{step}_{c}", tag="algorithmic")
        for s in range(shortlist):       # lines 255/257: rollout value
            meter.z(f"r{weight}_{step}_{s}", tag="algorithmic")
        current = f"v{weight}_{step}_0"
    meter.z(current, tag="harness")      # line 331: terminal
    return current


def _gen_rank(meter: TaggedMeter, n_pool: int) -> list[str]:
    pool = [_unguided_run(meter, seed=i) for i in range(n_pool)]
    for end in pool:                     # line 353: ranking, one read per candidate
        meter.z(end, tag="algorithmic")
    return pool


def _payload(meter: TaggedMeter, **extra) -> dict:
    return {"cost": {"raw_oracle_calls": meter.raw,
                     "native_oracle_calls": meter.native,
                     "kernel_calls": 0}, **extra}


def test_unguided_corrects_to_exactly_zero():
    """An unguided sampler never consults an objective, so it demands nothing.

    This is the check that the DEFINITION is right. If the correction left
    `unguided` with a positive algorithmic demand, the definition would be
    attributing our evaluation of its output to the method.
    """
    meter = TaggedMeter()
    for i in range(N_PREF):
        _unguided_run(meter, seed=i)
    out = corrected_cost("unguided", _payload(meter), N_PREF)
    assert meter.algorithmic == 0
    assert out["algorithmic_oracle_requests"] == 0 == meter.algorithmic
    assert out["raw_instrument_oracle_requests"] == meter.raw == N_PREF


@pytest.mark.parametrize("n_pool", [1, 2, 61, 400])
def test_gen_rank_corrects_to_one_ranking_read_per_candidate(n_pool):
    meter = TaggedMeter()
    _gen_rank(meter, n_pool)
    out = corrected_cost("gen_rank@verified",
                         _payload(meter, n_trajectories=n_pool), N_PREF)
    assert meter.raw == 2 * n_pool                     # the committed pattern
    assert out["algorithmic_oracle_requests"] == n_pool == meter.algorithmic


def test_greedy_and_verified_agree_with_the_tagged_count():
    for name, run in (("greedy_pref", _greedy_run), ("verified_pref", _verified_run)):
        meter = TaggedMeter()
        for w in range(N_PREF):
            run(meter, weight=w)
        out = corrected_cost(name, _payload(meter), N_PREF)
        assert out["algorithmic_oracle_requests"] == meter.algorithmic, name
        assert out["harness_only_requests"] == meter.harness == N_PREF, name


def test_the_raw_counter_is_passed_through_untouched():
    """Parity depends on it, so the correction may never alter it."""
    meter = TaggedMeter()
    _gen_rank(meter, 7)
    payload = _payload(meter, n_trajectories=7)
    before = payload["cost"]["raw_oracle_calls"]
    out = corrected_cost("gen_rank@greedy", payload, N_PREF)
    assert out["raw_instrument_oracle_requests"] == before
    assert payload["cost"]["raw_oracle_calls"] == before   # not mutated


def test_native_calls_are_order_invariant_and_uncorrected():
    """Which call site reached a molecule first cannot change evaluator work."""
    a, b = TaggedMeter(), TaggedMeter()
    a.z("x", tag="harness"); a.z("x", tag="algorithmic")
    b.z("x", tag="algorithmic"); b.z("x", tag="harness")
    assert a.native == b.native == 1
    assert a.raw == b.raw == 2


def test_a_negative_correction_raises_and_is_never_clamped():
    meter = TaggedMeter()
    _unguided_run(meter, seed=0)
    with pytest.raises(ValueError, match="negative"):
        corrected_cost("gen_rank@greedy", _payload(meter, n_trajectories=99), N_PREF)


def test_unknown_arm_refuses_to_guess():
    with pytest.raises(ValueError, match="refusing to guess"):
        harness_only_requests("some_new_arm", {"cost": {}}, N_PREF)


def test_gen_rank_without_a_trajectory_count_refuses():
    with pytest.raises(ValueError, match="n_trajectories"):
        harness_only_requests("gen_rank@verified", {"cost": {}}, N_PREF)


def test_corrected_shard_covers_every_arm_and_does_not_mutate():
    shard = {"preferences": [0.1, 0.3, 0.5, 0.7, 0.9], "arms": {}}
    m = TaggedMeter(); [_unguided_run(m, seed=i) for i in range(N_PREF)]
    shard["arms"]["unguided"] = _payload(m)
    m2 = TaggedMeter(); _gen_rank(m2, 61)
    shard["arms"]["gen_rank@verified"] = _payload(m2, n_trajectories=61)
    snapshot = {a: dict(p["cost"]) for a, p in shard["arms"].items()}
    out = corrected_shard(shard)
    assert set(out) == {"unguided", "gen_rank@verified"}
    assert out["unguided"]["algorithmic_oracle_requests"] == 0
    assert out["gen_rank@verified"]["algorithmic_oracle_requests"] == 61
    assert {a: dict(p["cost"]) for a, p in shard["arms"].items()} == snapshot


# --- the metered top-up arm scores ONCE, so the correction must not double-count

def test_metered_topup_arm_keeps_its_ranking_demand():
    """Regression for a real miscount caught by source 000's ledger.

    The metered matcher reads `endpoint_z` instead of re-requesting it, so
    raw == n rather than 2n. Subtracting n as harness overhead would report
    generate-and-rank's algorithmic demand as ZERO. The double-billing went
    away; the demand -- one ranking read per candidate -- did not.
    """
    n = 1025
    old = {"cost": {"raw_oracle_calls": 2 * n, "native_oracle_calls": 224,
                    "kernel_calls": 369}, "n_trajectories": n}
    new = {"cost": {"raw_oracle_calls": n, "native_oracle_calls": 224,
                    "kernel_calls": 369}, "n_trajectories": n,
           "shares_scoring_with_ranking": True}
    a_old = corrected_cost("gen_rank@verified", old, N_PREF)
    a_new = corrected_cost("gen_rank@verified", new, N_PREF)
    assert a_old["algorithmic_oracle_requests"] == n
    assert a_new["algorithmic_oracle_requests"] == n, "the metered arm lost its demand"
    assert a_old["raw_instrument_oracle_requests"] == 2 * n
    assert a_new["raw_instrument_oracle_requests"] == n
    # The benchmark evaluation is free in the metered arm because it shares the
    # ranking's request, and is a separate request in the original.
    assert a_old["benchmark_eval_requests"] == n
    assert a_new["benchmark_eval_requests"] == 0
