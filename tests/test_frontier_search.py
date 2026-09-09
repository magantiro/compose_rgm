"""Small behavioral tests, no task data, trained models or docking."""

import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from compose_v4.control.continuation import ContinuationBudgetExceeded
from compose_v4.control.frontier_search import FrontierSearch, payload_hash
from compose_v4.control.task_search import SearchRow


def tree(state):
    return SearchRow((state + (0,), state + (1,)), ("left", "right"), (0.5, 0.5), (0.5, 0.5), 0.1)


def planner(preferred=1, reference=tree):
    return FrontierSearch(
        reference,
        lambda s: float(s[0] == preferred) if len(s) == 3 else None,
        lambda s: len(s) == 6,
        lambda s: s,
        reference_id="synthetic-reference",
        snapshot_id=f"synthetic-objective-{preferred}",
        seed=11,
    )


def encode(state):
    return {"path": list(state)}


def decode(payload):
    return tuple(payload["path"])


def test_short_feedback_before_full_horizon_and_all_three_levels():
    search = planner()
    # Materialize one preferred option path, leaving the full horizon unfinished.
    for node in ((), (1,), (1, 1)):
        search.decision(node)
    assert search.work.paths_completed == 0
    assert search.work.endpoint_evaluations == 2
    assert search.decision(())["probabilities"][1] > 0.65
    assert search.receipt()["exact_doob"] is False
    for node in ((), (1,), (1, 1)):
        decision = search.decision(node)
        assert decision["kl"] <= 1 + 1e-10
        assert min(decision["probabilities"]) >= 0.05
        assert decision["value_semantics"].endswith("not_expected_return")


def test_split_resume_matches_uninterrupted_rng_cursor_values_and_cached_rows():
    uninterrupted = planner()
    uninterrupted.advance((), 9)
    split = planner()
    split.advance((), 4)
    saved = json.loads(json.dumps(split.checkpoint(encode), allow_nan=False))
    resumed = planner()
    resumed.restore(saved, decode)
    resumed.advance((), 5)
    assert resumed.cursor == uninterrupted.cursor
    assert resumed.states == uninterrupted.states
    assert resumed.best == uninterrupted.best
    assert resumed.leaves == uninterrupted.leaves
    assert resumed.rng.bit_generator.state == uninterrupted.rng.bit_generator.state
    assert resumed.work.planning_transitions == 9
    assert resumed.work.row_attempts == uninterrupted.work.row_attempts
    assert resumed.decision(()) == uninterrupted.decision(())


def test_executor_pause_retains_prefix_and_rng_instead_of_failed_return():
    blocked = False

    def reference(s):
        if blocked and len(s) >= 1:
            raise ContinuationBudgetExceeded("fixture pause")
        return tree(s)

    search = planner(reference=reference)
    search.advance((), 1)
    cursor, rng = search.cursor, copy.deepcopy(search.rng.bit_generator.state)
    blocked = True
    receipt = search.advance((), 3)
    assert receipt["status"] == "executor_paused"
    assert search.cursor == cursor and search.rng.bit_generator.state == rng
    assert not search.best and not hasattr(search, "returns")
    blocked = False
    search.advance((), 2)
    assert search.work.planning_transitions == 3
    assert search.work.paths_completed == 0


def test_local_global_floor_zero_support_and_no_backup_through_zero_mass():
    row = SearchRow(
        (0, 1, 2), ("local", "global", "unsupported"), (0.999, 0.001, 0), (0.5, 0.5, 0), 0.2
    )
    search = FrontierSearch(
        lambda _: row,
        lambda s: float(s == 2),
        lambda _: False,
        lambda s: s,
        reference_id="floor",
        snapshot_id="floor",
        seed=0,
    )
    decision = search.decision("root")
    assert decision["probabilities"][2] == 0
    assert min(decision["probabilities"][:2]) >= 0.1
    assert decision["kl"] <= 1 + 1e-10
    assert search.best.get(search.ids["root"], 0) == 0


def test_snapshot_refresh_reuses_exact_rows_and_reverses_existing_evidence():
    original = planner()
    for node in ((), (0,), (0, 0), (1,), (1, 1)):
        original.decision(node)
    assert original.decision(())["probabilities"][1] > 0.65

    def forbidden(_):
        raise AssertionError("cached exact rows must not be re-enumerated")

    refreshed = planner(0, forbidden)
    refreshed.restore(original.checkpoint(encode), decode)
    assert refreshed.work.snapshot_invalidations == 1
    assert refreshed.decision(())["probabilities"][0] > 0.65
    assert refreshed.states == original.states
    assert refreshed.rng.bit_generator.state == original.rng.bit_generator.state


@pytest.mark.parametrize("defect", ["hash", "reference", "duplicate", "negative_index"])
def test_corrupt_or_incompatible_checkpoint_fails(defect):
    search = planner()
    search.advance((), 2)
    saved = search.checkpoint(encode)
    if defect == "hash":
        saved["cursor"] = 10000
    else:
        if defect == "reference":
            saved["reference_id"] = "other"
        elif defect == "duplicate":
            saved["states"].append(saved["states"][-1])
        else:
            saved["rows"][0]["children"][0] = -1
        saved["checkpoint_sha256"] = payload_hash(
            {k: v for k, v in saved.items() if k != "checkpoint_sha256"}
        )
    with pytest.raises(ValueError, match="checkpoint"):
        planner().restore(saved, decode)


def test_molecular_state_roundtrip_keeps_unfinished_phase_and_sparse_slots():
    from test_option_continuation import fixture_law, source

    from compose_v4.control.molecular_search_codec import decode_search_state, encode_search_state
    from compose_v4.control.molecular_task_search import MolecularHierarchy, MolecularSearchState
    from compose_v4.control.option_continuation import OptionContinuationKernel
    from compose_v4.rewrite.kernel import editing_v2_rewrite_system

    kernel = OptionContinuationKernel(
        fixture_law, editing_v2_rewrite_system(), max_executor_applications=200
    )
    hierarchy = MolecularHierarchy(kernel, generic_horizon=2, ring_options=())
    root = MolecularSearchState.start(source().graph, budget=4, root_id="codec-fixture")
    order = np.asarray([0, 1, 5, 3, 4, 2])
    graph = root.graph
    sparse_graph = replace(
        graph,
        atom_types=graph.atom_types[order],
        bonds=graph.bonds[np.ix_(order, order)],
        formal_charges=graph.formal_charges[order],
        implicit_h_counts=graph.implicit_h_counts[order],
    )
    sparse = MolecularSearchState.start(sparse_graph, budget=4, root_id="sparse-fixture")
    assert set(sparse.lineage.id_of) == {0, 1, 3, 5}
    assert (
        decode_search_state(json.loads(json.dumps(encode_search_state(sparse)))).key()
        == sparse.key()
    )
    where = hierarchy.row(root)
    chosen = max(where.successors, key=lambda s: s.region.size)
    what = hierarchy.row(chosen)
    assert "generic" in what.labels
    assert what.reference[what.labels.index("generic")] >= 0.1 / len(what.labels)
    generic = what.successors[what.labels.index("generic")]
    middle = hierarchy.row(generic).successors[0]
    assert middle.stage == "how" and middle.active.remaining == 1
    for state in (root, chosen, generic, middle):
        encoded = json.loads(json.dumps(encode_search_state(state)))
        assert decode_search_state(encoded).key() == state.key()
    damaged = encode_search_state(middle)
    damaged["lineage"]["next_id"] = 0
    with pytest.raises(ValueError, match="lineage"):
        decode_search_state(damaged)
    decision = FrontierSearch(
        hierarchy.row,
        lambda _: None,
        lambda s: s.budget == 0,
        MolecularSearchState.key,
        reference_id="fixture",
        snapshot_id="fixture",
        seed=0,
    ).decision(root)
    assert np.all(np.asarray(decision["probabilities"]) >= 0.2 * np.asarray(where.floor) - 1e-12)


def test_existing_option_decoder_retains_ring_expansion_progress():
    from compose_v4.control.molecular_search_codec import decode_option
    from compose_v4.control.option_continuation import OptionContinuationKernel
    from compose_v4.experiments.continuation_profile import state_payload
    from compose_v4.rewrite.kernel import editing_v2_rewrite_system
    from tools.ring_expansion_audit import fixture_law, initial

    start = initial("C1CCCCC1")
    process = OptionContinuationKernel(
        fixture_law, editing_v2_rewrite_system(), max_executor_applications=128
    )
    middle = process.row(start).successors[0]
    assert middle.expansion_progress is not None
    assert decode_option(json.loads(json.dumps(state_payload(middle)))).key() == middle.key()
