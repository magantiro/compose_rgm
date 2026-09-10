from pathlib import Path

import numpy as np
import pytest
from test_carbonyl_option import initial, kernel

from compose_v4.control.archive_allocation import ArchiveCredit, parent_distribution, top10_sum
from compose_v4.control.molecular_search_codec import encode_search_state
from compose_v4.control.molecular_task_search import MolecularHierarchy, MolecularSearchState
from compose_v4.experiments import pmo_archive_pilot as pilot
from compose_v4.experiments.continuation_profile import ExecutorMeter
from compose_v4.experiments.t4_macro_beam import WitnessIndex
from compose_v4.rewrite.trace_shard import encode_state


def test_canonical_parent_mass_and_equal_score_ties():
    archive = [{"smiles": "CC"}, {"smiles": "CCC"}]
    observed = {"CC": 0.4, "CCC": 0.8}
    p = parent_distribution(archive, observed)
    q = parent_distribution([archive[0], archive[1], archive[1]], observed)
    assert q[0] == p[0] and q[1] + q[2] == p[1]
    assert p[1] > p[0] and np.all(p >= 0.1)
    assert np.allclose(parent_distribution(archive, {"CC": 0.4, "CCC": 0.4}), [0.5, 0.5])
    assert top10_sum({str(i): 0.5 for i in range(11)}) == 5


def test_delayed_credit_changes_allocation_not_trial_count():
    credit = ArchiveCredit()
    credit.observe("e0", parent_chain=[], scale="global", option="grow", reward=0)
    credit.observe("e1", parent_chain=[], scale="local", option="local", reward=0)
    reference = [0.5, 0.5]
    p, _ = credit.distribution(reference, "where", ["global", "local"], adaptive=True)
    assert np.allclose(p, reference)
    credit.observe("e2", parent_chain=["e0"], scale="local", option="cyclize", reward=0.8)
    assert credit.edges["e0"]["direct"] == 0
    assert credit.edges["e0"]["credit"] == pytest.approx(0.72)
    assert credit.values("what", ["grow"])[1] == [1]
    q, audit = credit.distribution(reference, "what", ["grow", "local"], adaptive=True)
    assert q[0] > q[1] and audit["kl"] <= 1 and np.all(q >= 0.1 * np.array(reference))
    p, audit = credit.distribution(reference, "what", ["grow", "local"], adaptive=False)
    assert np.array_equal(p, reference) and audit["eta"] == 0
    with pytest.raises(ValueError, match="duplicate"):
        credit.observe("e2", parent_chain=[], scale="local", option="local", reward=0)


def run_option(output, *, interrupt=False):
    process = kernel()
    hierarchy = MolecularHierarchy(process, lazy_applicability=True, include_carbonyl_options=True)
    root = MolecularSearchState.start(initial("CCC").graph, budget=0, root_id="fixture")
    parent = {"id": "root", "node": encode_search_state(root), "chain": [], "primitives": 50}
    store = pilot.Store(output, lambda: None)
    save = store.save
    did_interrupt = False

    def maybe_stop(name, value):
        nonlocal did_interrupt
        save(name, value)
        if interrupt and not did_interrupt and value.get("status") == "running":
            did_interrupt = True
            raise InterruptedError("after durable primitive/selection")

    store.save = maybe_stop
    meter = ExecutorMeter(None)
    with meter.instrument():
        record = pilot.execute_option(
            parent,
            hierarchy,
            ArchiveCredit(),
            False,
            np.random.default_rng(2011),
            "attempt",
            store,
            WitnessIndex(meter),
            {},
        )
    return record


def test_real_option_renews_local_clock_and_replays_same_path_after_interrupt(tmp_path):
    full = run_option(tmp_path / "full")
    with pytest.raises(InterruptedError):
        run_option(tmp_path / "resume", interrupt=True)
    resumed = run_option(tmp_path / "resume")
    for key in ("status", "source", "node", "events", "bundle", "rng_state", "candidate"):
        assert full[key] == resumed[key]
    assert full["candidate"]["primitive_count"] >= 1
    assert full["candidate"]["primitives"] > 50
    assert full["candidate"]["node"]["stage"] == "where"


def test_persistent_feedback_resume_does_not_recharge_or_reveal_future_labels(
    tmp_path, monkeypatch
):
    def fake_option(parent, hierarchy, credit, adaptive, rng, name, store, witnesses, progress):
        prior = store.read(name)
        if prior:
            return prior
        index = int(name.split("/")[-1])
        graph = initial("C" * (index + 4)).graph
        node = MolecularSearchState.start(graph, budget=10, root_id="fixture")
        candidate = {
            "id": name,
            "node": encode_search_state(node),
            "smiles": "C" * (index + 4),
            "chain": parent["chain"] + [name],
            "primitives": parent["primitives"] + 1,
        }
        record = {
            "status": "complete",
            "candidate": candidate,
            "events": [],
            "bundle": {"option": "grow"},
            "proposal_seconds": 0,
            "stage_seconds": {},
        }
        store.save(name, record)
        witnesses.meter.attempts.append({"status": "invalid_rewrite", "fixture_attempt": index})
        return record

    monkeypatch.setattr(pilot, "execute_option", fake_option)
    root = {"smiles": "CCC", "state": encode_state(initial("CCC").graph)}
    contract = {
        "contract_sha256": "fixture",
        "roots": [root],
        "seed": 44,
        "oracle_budget": 6,
        "search": {
            "option_budget": 11,
            "max_attempts": 10,
            "parent_exploration": 0.2,
            "credit_discount": 0.9,
            "credit_pseudocount": 2,
        },
        "compute": {"profile_first_attempts": 4},
        "interpretation": "fixture only",
    }
    calls = []

    def run(folder, interrupted=False):
        store = pilot.Store(folder, lambda: None)
        save = store.save

        def stop(name, value):
            save(name, value)
            if interrupted and name == "feedback/0001":
                raise InterruptedError("after score, before next allocation")

        store.save = stop
        scores = pilot.DurableScores(
            folder, lambda s: calls.append(s) or len(s) / 10, 6, lambda: None, {}
        )
        return pilot.run_case(
            contract,
            {"arm": "adaptive", "replicate": 0},
            None,
            store,
            scores,
            meter=ExecutorMeter(None),
            progress={},
        )

    full = run(tmp_path / "full")
    calls.clear()
    with pytest.raises(InterruptedError):
        run(tmp_path / "resume", interrupted=True)
    resumed = run(tmp_path / "resume")
    assert pilot.Store(tmp_path / "resume", lambda: None).read("executor/0000") == [
        {"status": "invalid_rewrite", "fixture_attempt": 0}
    ]
    assert calls == ["C" * i for i in range(3, 9)]
    for key in ("best", "curve", "credit", "archive", "attempts", "auc_top10"):
        assert full[key] == resumed[key]
    assert resumed["oracle_calls"] == 6 and resumed["primed"] == 0
    assert len(resumed["archive"]) == 6


def test_contract_is_bounded_and_label_free():
    c = pilot.load_contract(Path(__file__).resolve().parents[1])
    assert len(pilot.cases(c)) == 2 and c["oracle_budget"] == 100
    assert not any(
        c[k] for k in ("prescreen", "winner_inputs", "training_authorized", "docking_authorized")
    )
    assert all("score" not in r for r in c["roots"])
    assert len(c["law_caches"]) == 1


def test_shared_region_context_is_exactly_the_per_option_context():
    from compose_v4.experiments.t4_warm_continuation import exact_context
    from compose_v4.rewrite.kernel import canonical_state_key

    hierarchy = MolecularHierarchy(kernel(), lazy_applicability=True, include_carbonyl_options=True)
    node = MolecularSearchState.start(initial("CCC").graph, budget=11, root_id="fixture")
    where = hierarchy.row(node)
    selected = max(where.successors, key=lambda n: n.region.released_fraction)
    context = exact_context(selected.graph, canonical_state_key(selected.graph), selected.region)
    what = hierarchy.row(selected)
    for option in what.labels:
        shared = hierarchy.option_state(selected, option, context=context)
        separate = hierarchy.option_state(selected, option)
        assert shared.key() == separate.key()
