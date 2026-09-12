"""Measured-feedback plumbing, variable support and resume with model-free fixtures."""

import json
from dataclasses import replace

import numpy as np
import pytest

from compose_v4.control.adaptive_program_optimizer import ProgramOptimizer, ProgramSearchConfig
from compose_v4.control.edit_program import extract_program
from compose_v4.control.edit_program_graph import compile_program_graph, execute_program_graph
from compose_v4.control.molecular_task_search import dispatch_complete_proposal
from tests.test_edit_program import ring_and_carbonyl


def optimizer(*, adaptive=True):
    source, stages = ring_and_carbonyl()
    program, binding = extract_program(source, stages)
    _, trace = execute_program_graph(source, compile_program_graph(program), binding)
    config = ProgramSearchConfig(
        seed=871,
        adaptive=adaptive,
        attempts_per_batch=32,
        candidates_per_batch=2,
        require_broad_runtime=False,
    )
    search = ProgramOptimizer(
        config, source_group="fixture", oracle_protocol="fixture:no-real-oracle"
    )
    record = {
        "source_group": search.source_group,
        "oracle_protocol": search.oracle_protocol,
        "source_state": trace["states"][0],
        "program": program.payload(),
        "assignment": list(binding),
        "trace": trace,
        "endpoint": trace["endpoint"],
    }
    search.add_measured_program(record, receipt_id="fixture:original", score=-1)
    return search


def eligible(row):
    return {"smiles": row["smiles"], "oracle_eligible": True}


def outcomes(search, batch, score):
    return [
        {
            "candidate_id": c["candidate_id"],
            "receipt_id": f"fixture:{c['candidate_id']}",
            "score": score - i,
            "oracle_protocol": search.oracle_protocol,
        }
        for i, c in enumerate(batch["candidates"])
    ]


def test_dispatch_does_not_enter_where_for_program_and_retains_broad_branch():
    rng, events = np.random.default_rng(17), []
    for _ in range(80):
        channel, product = dispatch_complete_proposal(
            rng,
            program_sampler=lambda c: events.append(c),
            broad_sampler=lambda: events.append("where"),
        )
        assert product is None
        assert events[-1] == ("where" if channel == "broad" else channel)
    assert set(events) == {"mutation", "recombination", "where"}
    with pytest.raises(ValueError, match="requires its broad hierarchy"):
        ProgramOptimizer(ProgramSearchConfig(), source_group="fixture", oracle_protocol="fixture")


def test_broad_branch_records_and_replays_real_executor_witnesses():
    from test_option_controller_runtime import runtime

    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.control.molecular_task_search import MolecularSearchState

    graph = pad_molecular_graph(smiles_to_molecular_graph("CCCC"), 48)
    root = MolecularSearchState.start(graph, budget=1, root_id="broad-fixture")
    hierarchy = runtime().hierarchy
    program, binding, detail = hierarchy.complete_reference_program(
        root, np.random.default_rng(1), max_options=1
    )
    endpoint, trace = execute_program_graph(graph, compile_program_graph(program), binding)
    assert endpoint.n_real_atoms == 5
    assert detail == {"complete_options": 1, "primitive_edits": 1}
    assert len(trace["actions"]) == 1
    assert all(block.label != "build_ring_system" for block in program.blocks)
    assert graph.n_real_atoms == 4


def test_variable_programs_produce_unseen_endpoints_and_score_feedback_changes_allocation():
    adaptive, static = optimizer(), optimizer(adaptive=False)
    first, control = adaptive.propose_batch(eligible), static.propose_batch(eligible)
    assert first["candidates"] and first["candidates"] == control["candidates"]
    assert all(
        c["endpoint"] != next(iter(adaptive.entries.values()))["endpoint"]
        for c in first["candidates"]
    )
    assert first["new_oracle_calls"] == 0
    adaptive.observe_batch(first["batch_id"], outcomes(adaptive, first, -5))
    static.observe_batch(control["batch_id"], outcomes(static, control, -5))
    keys, weights = adaptive.selection()
    _, fixed = static.selection()
    new = {c["endpoint"] for c in first["candidates"]}
    assert sum(
        w for k, w in zip(keys, weights, strict=True) if adaptive.entries[k]["endpoint"] in new
    ) > sum(w for k, w in zip(keys, fixed, strict=True) if static.entries[k]["endpoint"] in new)


def test_snapshot_resume_and_pending_lock_are_exact_and_output_is_not_mutable_state():
    search = optimizer()
    batch = search.propose_batch(eligible)
    original_endpoint = batch["candidates"][0]["endpoint"]
    batch["candidates"][0]["endpoint"] = "tampered return value"
    assert search.pending["candidates"][0]["endpoint"] == original_endpoint
    snapshot = search.snapshot()
    restored = ProgramOptimizer.restore(json.loads(json.dumps(snapshot)))
    labels = outcomes(search, search.pending, -3)
    search.observe_batch(snapshot["pending"]["batch_id"], labels)
    restored.observe_batch(snapshot["pending"]["batch_id"], labels)
    assert search.snapshot() == restored.snapshot()
    one, two = search.propose_batch(eligible), restored.propose_batch(eligible)
    assert one["batch_id"] == two["batch_id"]
    assert one["attempts"] == two["attempts"]
    with pytest.raises(ValueError, match="pending batch"):
        search.propose_batch(eligible)


def test_incomplete_wrong_domain_and_missing_scores_do_not_update_archive():
    search = optimizer()
    batch = search.propose_batch(eligible)
    before = search.snapshot()
    with pytest.raises(ValueError, match="exactly one outcome"):
        search.observe_batch(batch["batch_id"], [])
    wrong = outcomes(search, batch, -4)
    wrong[-1]["oracle_protocol"] = "another-protocol"
    with pytest.raises(ValueError, match="actual oracle protocol"):
        search.observe_batch(batch["batch_id"], wrong)
    assert search.snapshot() == before
    labels = [
        {**r, "score": None, "failure": "cache_miss_not_evaluated"}
        for r in outcomes(search, batch, -4)
    ]
    search.observe_batch(batch["batch_id"], labels)
    assert len(search.entries) == 1
    assert not search.failed_endpoints


def test_no_forced_padding_to_maximum_blocks_and_configuration_is_explicit():
    config = optimizer().config
    assert replace(config, max_primitives=40, max_blocks=12).max_blocks == 12
    with pytest.raises(ValueError, match="sum to one"):
        replace(config, channel_probabilities=(1.0, 0.0, 0.0))
