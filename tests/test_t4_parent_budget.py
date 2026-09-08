"""Small exact accounting and administrative-stop regression tests."""

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.continuation import ContinuationBudgetExceeded
from compose_v4.experiments.continuation_profile import ExecutorMeter
from compose_v4.experiments.t4_parent_budget import ParentExecutorShare
from compose_v4.rewrite.kernel import InvalidRewrite, RewriteSystem
from compose_v4.rewrite.operators import AtomDelete


def test_equal_shares_preserve_global_count_and_next_parent(monkeypatch):
    graph = pad_molecular_graph(smiles_to_molecular_graph("CC"), 48)
    monkeypatch.setattr(RewriteSystem, "apply", lambda self, state, *_: state)
    system = RewriteSystem.__new__(RewriteSystem)
    committed, shares = [], []
    with ExecutorMeter(24).instrument() as total:
        for parent in range(8):
            share = ParentExecutorShare(3)
            with share.instrument():
                for _ in range(10):
                    try:
                        result = system.apply(graph, "atom_delete", AtomDelete(1))
                    except Exception:  # noqa: BLE001 - prove legacy rejection cannot swallow cancellation
                        pytest.fail("budget stop was caught as chemical failure")
                    committed.append((parent, result))
            shares.append(share.receipt())
    assert total.calls == len(total.attempts) == len(committed) == 24
    assert all(share["executor_calls"] == 3 and share["exhausted"] for share in shares)
    assert {parent for parent, _ in committed} == set(range(8))
    assert all(np.array_equal(result.atom_types, graph.atom_types) for _, result in committed)


def test_nested_calls_count_and_interruption_is_recorded(monkeypatch):
    graph = pad_molecular_graph(smiles_to_molecular_graph("CC"), 48)

    def nested(self, state, rule, action):
        if action.v == 0:
            return self.apply(state, rule, AtomDelete(1))
        return state

    monkeypatch.setattr(RewriteSystem, "apply", nested)
    system = RewriteSystem.__new__(RewriteSystem)
    share = ParentExecutorShare(1)
    with ExecutorMeter(5).instrument() as total, share.instrument():
        system.apply(graph, "atom_delete", AtomDelete(0))
        pytest.fail("interrupted nested call committed a candidate")
    assert share.exhausted and total.calls == share.calls == 1
    assert total.attempts[0]["status"] == "budget_interrupted"


@pytest.mark.parametrize("failure", [InvalidRewrite("invalid"), ValueError("runtime")])
def test_share_does_not_swallow_errors(monkeypatch, failure):
    def fail(*_):
        raise failure

    monkeypatch.setattr(RewriteSystem, "apply", fail)
    with pytest.raises(type(failure), match=str(failure)), ParentExecutorShare(1).instrument():
        RewriteSystem.__new__(RewriteSystem).apply(None, "unused", None)


def test_share_does_not_swallow_total_ceiling(monkeypatch):
    graph = pad_molecular_graph(smiles_to_molecular_graph("CC"), 48)
    monkeypatch.setattr(RewriteSystem, "apply", lambda self, state, *_: state)
    with (
        pytest.raises(ContinuationBudgetExceeded),
        ExecutorMeter(0).instrument(),
        ParentExecutorShare(3).instrument(),
    ):
        RewriteSystem.__new__(RewriteSystem).apply(graph, "atom_delete", AtomDelete(1))


@pytest.mark.parametrize("fixture_limit", [1, 15])
def test_real_population_moves_to_next_parent_after_share_stop(
    monkeypatch, tmp_path, fixture_limit
):
    from test_t4_matched_pilot import population_fixture

    from compose_v4.control import option_selector, region, region_selector
    from compose_v4.experiments import t4_parent_budget
    from compose_v4.experiments.t4_warm_continuation import payload_hash
    from compose_v4.rewrite.trace_shard import encode_state

    prepare = population_fixture(monkeypatch, tmp_path)
    # Small fixture budget only. Production entrypoint still validates 8 x 2500.
    monkeypatch.setattr(
        t4_parent_budget, "ParentExecutorShare", lambda _: ParentExecutorShare(fixture_limit)
    )
    if fixture_limit > 1:
        monkeypatch.setattr(
            option_selector,
            "sample_option",
            lambda options, *_a, **_kw: option_selector.OptionChoice(
                "generic", tuple(options), tuple(1 / len(options) for _ in options)
            ),
        )
    smiles = "c1ccccc1"
    graph = pad_molecular_graph(smiles_to_molecular_graph(smiles), 48)
    largest = max(region.enumerate_regions(smiles), key=lambda r: r.size)
    monkeypatch.setattr(region_selector, "sample_region", lambda *_a, **_kw: (largest, None))
    warm = {
        "archive": [{"smiles": smiles, "state": encode_state(graph), "ds": None}],
        "round": 0,
        "rng_state": np.random.default_rng(1000).bit_generator.state,
    }
    task = {
        "smiles": smiles,
        "delta": 0.4,
        "target": "parp1",
        "budget": 20,
        "seed_rng": 1000,
        "lineages": 8,
        "regions_per_lineage": 3,
        "workers": 1,
        "particles_per_region": 1,
        "include_fused": True,
        "prepare_only": True,
        "primitive_guidance": "reference",
        "max_executor_applications": 20000,
        "executor_calls_per_parent": 2500,
        "warm_start_sha256": payload_hash(warm),
    }
    units = []
    with ExecutorMeter(20000).instrument() as meter:
        lock = prepare(task, units.append, warm_start=warm)
    assert len(units) == len(lock["work"]) == 8
    assert meter.calls <= 8 * fixture_limit and lock["oracle_calls"] == 0
    assert all(unit["parent_budget"]["exhausted"] for unit in lock["work"])
    if fixture_limit == 1:
        assert all(unit["unselected_region_draws"] == 3 for unit in lock["work"])
        assert not lock["take"] and not lock["pool"]
    else:
        assert lock["take"] and all(c["option"] == "generic" for c in lock["take"])
        from compose_v4.experiments.t4_warm_continuation import endpoint

        for candidate in lock["take"]:
            assert candidate["state"] == endpoint(lock, candidate)
