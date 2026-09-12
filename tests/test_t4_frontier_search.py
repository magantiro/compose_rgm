"""Synthetic T4 labels, tiny legal reference, production molecular executor."""

import copy
import json
import math

import pytest

from compose_v4.chem.molecular_graph import ELEMENT_TO_IDX, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.frontier_search import payload_hash
from compose_v4.control.molecular_search_codec import decode_search_state
from compose_v4.control.option_continuation import exact_graph_key
from compose_v4.experiments.t4_frontier_search import FrontierConfig, prepare_slice
from compose_v4.rewrite import action_codec, action_codec_v4
from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_rewrite_system
from compose_v4.rewrite.operators import AtomInsert
from compose_v4.rewrite.trace_shard import encode_state


def archive():
    rows = []
    for i in range(1, 18):
        graph = pad_molecular_graph(smiles_to_molecular_graph("C" * i), 48)
        rows.append(
            {
                "smiles": canonical_state_key(graph),
                "state": encode_state(graph),
                "ds": None if i == 1 else -float(i),
                "round": 0 if i == 1 else 1,
            }
        )
    return {
        "schema_version": "t4_exact_archive_v1",
        "round": 1,
        "archive": rows,
        "oracle_attempts": 16,
    }


def fixture_law(graph):
    return (
        ("atom_insert",),
        (AtomInsert(graph.n_real_atoms, ELEMENT_TO_IDX["C"], 0, 3, ((0, 1),)),),
        (1.0,),
    )


def run(warm, **kwargs):
    return prepare_slice(
        warm,
        source_sha256="a" * 64,
        input_sha256={"synthetic_reference": "b" * 64},
        code_revision="0" * 40,
        enumerate_law=kwargs.pop("enumerate_law", fixture_law),
        system=editing_v2_rewrite_system(),
        config=kwargs.pop(
            "config",
            FrontierConfig(lineages=1, primitive_budget=4, planning_transitions=1),
        ),
        **kwargs,
    )


def test_prepare_resume_keeps_full_path_rng_and_checkpoint_without_docking():
    warm = archive()
    continuous = run(warm, events_per_parent=6)
    progress = []
    first = run(warm, events_per_parent=2, progress=progress.append)
    saved = json.loads(json.dumps(first["checkpoint"], allow_nan=False))
    state = decode_search_state(saved["frontier"][0]["current"])
    assert state.stage == "how" and state.active.step == 0
    frozen_checkpoint = copy.deepcopy(saved)
    split = run(warm, checkpoint=saved, events_per_parent=4)
    assert saved == frozen_checkpoint  # no mutation of the caller's checkpoint
    whole, resumed = (
        continuous["checkpoint"]["frontier"][0],
        split["checkpoint"]["frontier"][0],
    )
    for key in (
        "current",
        "root",
        "path",
        "candidates",
        "acting_rng",
        "planning_remaining",
    ):
        assert resumed[key] == whole[key]
    assert resumed["planner"]["rng_state"] == whole["planner"]["rng_state"]
    assert resumed["planner"]["states"] == whole["planner"]["states"]
    assert split["pool"] == continuous["pool"]
    assert split["new_oracle_calls"] == 0 and not split["oracle_authorized"]
    assert progress and progress[0] != progress[-1]
    assert progress[0]["frontier"][0]["path"] == []
    assert split["checkpoint"]["value_snapshot"]["before_round"] == 2
    assert split["checkpoint"]["prior_oracle_attempts"] == 16
    system = editing_v2_rewrite_system()
    primitive_count = 0
    for event in resumed["path"]:
        source, product = (
            decode_search_state(event["source"]),
            decode_search_state(event["product"]),
        )
        if source.stage == "how":
            mark = event["mark"]
            codec = action_codec_v4 if mark["schema_version"] == 4 else action_codec
            rule, action = codec.decode_action(mark)
            actual = system.apply(source.graph, rule, action)
            assert exact_graph_key(actual) == exact_graph_key(product.graph)
            primitive_count += 1
        else:
            assert exact_graph_key(source.graph) == exact_graph_key(product.graph)
        assert event["decision"]["kl"] <= 1 + 1e-10
    assert primitive_count >= 1
    assert resumed["candidates"]  # option completions are reported even when endpoint-infeasible
    for candidate in split["pool"]:
        assert candidate["program_complete"]
        assert "r_release" in candidate and "r_coherent" in candidate
        assert "d_cycle_rank" in candidate and "d_ring_systems" in candidate


def test_new_oracle_round_keeps_unfinished_exact_frontier_without_chemical_recompute():
    warm = archive()
    first = run(warm, events_per_parent=2)
    saved = first["checkpoint"]
    new = copy.deepcopy(warm)
    graph = pad_molecular_graph(smiles_to_molecular_graph("C" * 18), 48)
    new["archive"].append(
        {
            "smiles": canonical_state_key(graph),
            "state": encode_state(graph),
            "ds": -2.0,
            "round": 2,
        }
    )
    new.update(round=2, oracle_attempts=17)

    def forbidden(_):
        raise AssertionError("a snapshot-only refresh must not execute new chemistry")

    refreshed = run(new, checkpoint=saved, events_per_parent=0, enumerate_law=forbidden)
    old_unit, new_unit = saved["frontier"][0], refreshed["checkpoint"]["frontier"][0]
    for field in ("current", "root", "path", "acting_rng", "planning_remaining"):
        assert new_unit[field] == old_unit[field]
    assert new_unit["planner"]["states"] == old_unit["planner"]["states"]
    assert new_unit["planner"]["work"]["snapshot_invalidations"] == 1
    assert refreshed["checkpoint"]["round"] == 3
    assert refreshed["executor_calls"] == 0
    bad = copy.deepcopy(new)
    bad["archive"][1]["ds"] -= 1
    with pytest.raises(ValueError, match="append-only"):
        run(bad, checkpoint=saved, events_per_parent=0)


def test_executor_cap_pauses_without_erasing_the_planning_or_committed_frontier():
    warm = archive()
    full = run(warm, events_per_parent=6)
    partial = run(warm, events_per_parent=6, executor_limit=1)
    unit = partial["checkpoint"]["frontier"][0]
    assert unit["status"] == "executor_paused"
    assert partial["executor_calls"] == 1
    # The pending slice carries per-lineage targets, not just a shared loop count.
    assert partial["checkpoint"]["pending_slice"]["target_events"] == [6]
    resumed = run(warm, checkpoint=partial["checkpoint"], events_per_parent=6)
    assert resumed["checkpoint"]["frontier"][0]["path"] == full["checkpoint"]["frontier"][0]["path"]
    assert resumed["new_oracle_calls"] == 0
    payload = partial["checkpoint"]
    assert (
        payload_hash({k: v for k, v in payload.items() if k != "checkpoint_sha256"})
        == payload["checkpoint_sha256"]
    )


def test_partial_multi_lineage_slice_does_not_repeat_finished_parents():
    warm, progress = archive(), []
    config = FrontierConfig(lineages=2, primitive_budget=4, planning_transitions=1)
    full = run(warm, config=config, events_per_parent=3, progress=progress.append)
    partial = next(
        record
        for record in progress
        if len(record["frontier"][0]["path"]) == 3 and not record["frontier"][1]["path"]
    )
    assert partial["pending_slice"]["target_events"] == [3, 3]
    resumed = run(warm, config=config, events_per_parent=3, checkpoint=partial)
    for actual, expected in zip(
        resumed["checkpoint"]["frontier"], full["checkpoint"]["frontier"], strict=True
    ):
        assert actual["path"] == expected["path"]
        assert actual["acting_rng"] == expected["acting_rng"]
    assert resumed["checkpoint"]["pending_slice"] is None
    with pytest.raises(ValueError, match="original events_per_parent"):
        run(warm, config=config, events_per_parent=2, checkpoint=partial)


def test_legacy_oracle_audit_does_not_accept_new_unverified_preparation():
    from compose_v4.experiments.t4_task_search_audit import verify_lock

    warm = archive()
    result = run(warm, events_per_parent=0)
    with pytest.raises(ValueError):
        verify_lock(result["checkpoint"], warm, editing_v2_rewrite_system())


def test_post_hoc_has_no_generation_time_value_route(monkeypatch):
    from compose_v4.control.docking_value import DockingValue

    def forbidden(*args, **kwargs):
        raise AssertionError("post-hoc generation must not call task desirability")

    monkeypatch.setattr(DockingValue, "desirability", forbidden)
    result = run(
        archive(),
        events_per_parent=12,
        config=FrontierConfig(
            lineages=1, primitive_budget=4, planning_transitions=0, guidance="post_hoc"
        ),
    )
    unit = result["checkpoint"]["frontier"][0]
    assert unit["planner"]["work"]["endpoint_evaluations"] == 0
    assert unit["planner"]["work"]["planning_transitions"] == 0
    for event in unit["path"]:
        assert event["decision"]["probabilities"] == pytest.approx(event["decision"]["reference"])
    with pytest.raises(ValueError, match="post-hoc"):
        FrontierConfig(guidance="post_hoc", planning_transitions=2)


def test_frontier_selected_path_audit_and_tamper_rejection():
    from compose_v4.experiments.t4_frontier_audit import verify_preparation

    warm = archive()
    full = run(warm, events_per_parent=12)
    verified = verify_preparation(full, warm, editing_v2_rewrite_system())
    assert verified["schema_version"] == "t4_frontier_oracle_lock_v1"
    assert verified["audit"]["selected_path_executor_calls"] > 0
    assert verified["audit"]["pool_count"] == len(full["pool"])
    bad = copy.deepcopy(full)
    bad["pool"][0]["r_coherent"] += 1
    with pytest.raises(ValueError, match="geometry"):
        verify_preparation(bad, warm, editing_v2_rewrite_system())
    partial = run(warm, events_per_parent=2)
    with pytest.raises(ValueError, match="horizon"):
        verify_preparation(partial, warm, editing_v2_rewrite_system())


@pytest.mark.parametrize("guidance", ["post_hoc", "in_loop"])
def test_partition_reduction_preserves_serial_paths_rng_probabilities_and_pool(
    guidance,
):
    from compose_v4.control.docking_value import LEGACY_RECIPE, identity
    from compose_v4.experiments.t4_frontier_audit import verify_preparation
    from compose_v4.experiments.t4_frontier_compare import merge_preparations

    warm = archive()
    config = FrontierConfig(
        lineages=2,
        primitive_budget=4,
        planning_transitions=0 if guidance == "post_hoc" else 1,
        guidance=guidance,
    )
    serial = run(warm, config=config, events_per_parent=12)
    parts = [run(warm, config=config, events_per_parent=12, lineage_indices=(i,)) for i in range(2)]
    merged = merge_preparations(list(reversed(parts)), warm, editing_v2_rewrite_system())
    for actual, expected in zip(
        merged["checkpoint"]["frontier"], serial["checkpoint"]["frontier"], strict=True
    ):
        for field in (
            "current",
            "root",
            "path",
            "acting_rng",
            "candidates",
            "planning_remaining",
        ):
            assert actual[field] == expected[field]
    assert merged["pool"] == serial["pool"]
    assert merged["proposed_for_audit"] == serial["proposed_for_audit"]
    assert merged["executor_calls"] == 0
    assert merged["checkpoint"]["executor_calls"] == sum(
        p["checkpoint"]["executor_calls"] for p in parts
    )
    compatible = copy.deepcopy(parts)
    snapshot = compatible[1]["checkpoint"]["value_snapshot"]
    snapshot["recipe"] = LEGACY_RECIPE
    snapshot["coefficients"][0] = math.nextafter(snapshot["coefficients"][0], math.inf)
    snapshot["snapshot_sha256"] = identity(
        {k: v for k, v in snapshot.items() if k != "snapshot_sha256"}
    )
    legacy_snapshot_id = snapshot["snapshot_sha256"]
    for unit in compatible[1]["checkpoint"]["frontier"]:
        if unit["planner"] is None:
            continue
        unit["planner"]["snapshot_id"] = legacy_snapshot_id
        unit["planner"]["checkpoint_sha256"] = payload_hash(
            {k: v for k, v in unit["planner"].items() if k != "checkpoint_sha256"}
        )
        for event in unit["path"]:
            event["decision"]["snapshot_id"] = legacy_snapshot_id
    compatible[1]["checkpoint"]["checkpoint_sha256"] = payload_hash(
        {k: v for k, v in compatible[1]["checkpoint"].items() if k != "checkpoint_sha256"}
    )
    reused = merge_preparations(compatible, warm, editing_v2_rewrite_system())
    assert reused["source_value_snapshot_equivalence"]["unique_snapshot_count"] == 2
    assert all(
        row["equivalent"] for row in reused["source_value_snapshot_equivalence"]["comparisons"]
    )
    assert (
        verify_preparation(reused, warm, editing_v2_rewrite_system())["take"]
        == reused["proposed_for_audit"]
    )
    incompatible = copy.deepcopy(compatible)
    snapshot = incompatible[1]["checkpoint"]["value_snapshot"]
    snapshot["coefficients"][0] += 1e-8
    snapshot["snapshot_sha256"] = identity(
        {k: v for k, v in snapshot.items() if k != "snapshot_sha256"}
    )
    incompatible[1]["checkpoint"]["checkpoint_sha256"] = payload_hash(
        {k: v for k, v in incompatible[1]["checkpoint"].items() if k != "checkpoint_sha256"}
    )
    with pytest.raises(ValueError, match="drift exceeds"):
        merge_preparations(incompatible, warm, editing_v2_rewrite_system())
    with pytest.raises(ValueError, match="missing lineage"):
        merge_preparations(parts[:1], warm, editing_v2_rewrite_system())
    with pytest.raises(ValueError, match="duplicate"):
        merge_preparations([parts[0], parts[0]], warm, editing_v2_rewrite_system())
    with pytest.raises(ValueError, match="partition"):
        run(warm, config=config, lineage_indices=(0, 0))
    with pytest.raises(ValueError, match="mismatch"):
        run(
            warm,
            config=config,
            checkpoint=parts[0]["checkpoint"],
            events_per_parent=0,
            lineage_indices=(1,),
        )
