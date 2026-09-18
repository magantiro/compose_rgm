"""Guards on wide free expansion.

Width was the binding constraint on the first autonomous campaign: at delta=0.6 roughly
1.1% of proposed endpoints are feasible, so 90 draws yields about ten candidates and the
observed pools were 9 to 77. A selector cannot choose a molecule the pool never contained.

The property that matters here is that buying width with cores does not change WHAT is
produced. A shard boundary may change how long a pool takes and, through the extra draws,
how large it is -- it may not change the law the endpoints are drawn from, and it may not
let an infeasible endpoint through.
"""

from __future__ import annotations

import numpy as np
import pytest

from compose_v4.experiments.t4_fiber_campaign import Fiber, expand
from compose_v4.experiments.t4_fiber_expansion import (
    MINIMUM_DRAWS_PER_SHARD,
    _shard_plan,
    branch_value,
    expand_wide,
)

JAK2_ROOT = "COC(=O)CC2Nc1ccccc1c3ccnc4[nH]cc2c34"


# ---- Sharding arithmetic ----


@pytest.mark.parametrize("draws, shards", [(720, 8), (90, 8), (1, 8), (45, 8), (7, 3)])
def test_the_shard_plan_conserves_the_draw_count(draws, shards):
    plan = _shard_plan(draws, shards)
    assert sum(plan) == draws
    assert 1 <= len(plan) <= shards


def test_the_shard_plan_refuses_shards_below_the_startup_floor():
    # Each shard pays an interpreter start and an RDKit import, so splitting 60 draws
    # eight ways would spend most of the wall clock on startup.
    assert len(_shard_plan(60, 8)) == 1
    assert len(_shard_plan(8 * MINIMUM_DRAWS_PER_SHARD, 8)) == 8


def test_expansion_rejects_an_empty_draw_request():
    with pytest.raises(ValueError):
        _shard_plan(0, 4)


# ---- The law is unchanged by sharding ----


def test_one_shard_reproduces_the_serial_expansion_exactly():
    """The equivalence that makes width safe to buy.

    A single shard is the serial expansion with a known seed offset, so the two must agree
    endpoint for endpoint. If this drifts, every pool-size comparison in the campaign is
    comparing two different proposal laws.
    """
    serial = expand(JAK2_ROOT, -8.0, Fiber(JAK2_ROOT, 0.6), np.random.default_rng(1235),
                    draws=6, horizon=3)
    sharded = expand_wide(JAK2_ROOT, -8.0, JAK2_ROOT, 0.6, draws=6, shards=1, seed=1234)
    assert {r["smiles"] for r in sharded} == {r["smiles"] for r in serial}


def test_every_endpoint_from_a_wide_pool_passes_the_exact_gate():
    fiber = Fiber(JAK2_ROOT, 0.6)
    pool = expand_wide(JAK2_ROOT, -8.0, JAK2_ROOT, 0.6, draws=90, shards=2, seed=5)
    assert pool, "the root must yield some feasible endpoints"
    assert all(fiber.check(record["smiles"]) is not None for record in pool)
    assert len({record["smiles"] for record in pool}) == len(pool), "pool must be deduplicated"


# ---- Branch value ----


class _PreferOne:
    """A value model that likes exactly one molecule, so the rollout's job is visible."""

    def __init__(self, favourite: str):
        self.favourite, self.weights, self.n = favourite, np.ones(15), 100
        self.seen: list[str] = []

    def predict(self, features):
        return np.zeros(np.asarray(features).shape[0])


def test_branch_value_reports_its_own_score_at_zero_levels():
    from compose_v4.control.fiber_control import SearchState, program_features

    state = SearchState(archive={JAK2_ROOT: -8.0})
    candidates = [{
        "smiles": "C", "parent_score": -9.0, "similarity": 0.62, "qed": 0.66, "sa": 3.4,
        "regions": 1, "created": 1, "deleted": 0, "delta": 0.6, "families": ["base"],
    }]
    candidates[0]["features"] = program_features(candidates[0], state)
    reported = branch_value(candidates, _PreferOne("C"), state, JAK2_ROOT, 0.6,
                            beam=1, draws=1, levels=0)
    assert reported["C"]["branch"] == reported["C"]["own"] == pytest.approx(9.0)
    assert reported["C"]["explored"] == 0


def test_branch_value_refuses_a_degenerate_request():
    from compose_v4.control.fiber_control import SearchState

    with pytest.raises(ValueError):
        branch_value([], _PreferOne("C"), SearchState(), JAK2_ROOT, 0.6, beam=0, draws=1)
