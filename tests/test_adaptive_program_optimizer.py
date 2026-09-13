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


def test_program_only_mode_is_explicit_and_never_enters_reference(monkeypatch):
    search = optimizer()
    search.config = replace(
        ProgramSearchConfig.program_only_recipe(seed=871),
        attempts_per_batch=32,
        candidates_per_batch=2,
    )

    def forbidden(*args):
        pytest.fail("program-only optimization must not enter the reference hierarchy")

    monkeypatch.setattr(search, "_broad", forbidden)
    batch = search.propose_batch(eligible)
    assert batch["candidates"]
    assert {r["channel"] for r in batch["attempts"]} <= {"mutation", "recombination"}
    assert search.config.current_state_edit_probability == 0.25
    assert ProgramOptimizer.restore(search.snapshot()).config == search.config
    with pytest.raises(ValueError, match="match the proposal mode"):
        replace(search.config, proposal_mode="mixed")
    with pytest.raises(ValueError, match="cannot require"):
        replace(search.config, require_broad_runtime=True)
    assert ProgramSearchConfig().channel_probabilities == (0.7, 0.2, 0.1)


def test_direct_allocation_does_not_fit_when_the_entire_pool_can_be_scored():
    from compose_v4.control.parent_edit_search import prepare_query_batch
    from compose_v4.control.program_task import ProgramTask

    search = optimizer()
    search.config = replace(search.config, score_direction="maximize")
    task = ProgramTask("fixture", search.oracle_protocol, "pmo")

    def forbidden():
        pytest.fail("no model fit is needed to select every candidate")

    batch = prepare_query_batch(
        search,
        task,
        count=2,
        seed=1,
        model_factory=forbidden,
        diagnostic=True,
    )
    assert len(batch["candidates"]) == 2
    assert batch["selection"]["mode"] == "direct"
    assert batch["selection"]["model_sha256"] is None
    assert len(batch["proposal_pool"]["candidates"]) == 2


def test_proposal_cache_reuses_work_without_dropping_unqueried_candidates():
    from compose_v4.control.program_work_cache import ProgramWorkCache

    uncached, cached = optimizer(), optimizer()
    cached.config = replace(cached.config, proposal_cache_entries=128)
    cached.work_cache = ProgramWorkCache(128)
    one, two = uncached.propose_batch(eligible), cached.propose_batch(eligible)
    assert one["candidates"] == two["candidates"]
    labels = [
        {**r, "score": None, "failure": "cache_miss_not_evaluated"}
        for r in outcomes(cached, two, -1)
    ]
    cached.observe_batch(two["batch_id"], labels)
    restored = ProgramOptimizer.restore(cached.snapshot())
    assert not restored.work_cache.entries
    # Caches do not change the random stream or suppress unqueried endpoints.
    assert (
        cached.propose_batch(eligible)["candidates"]
        == restored.propose_batch(eligible)["candidates"]
    )


def test_proposal_cache_is_bounded_and_addresses_exact_inputs():
    from compose_v4.control.program_work_cache import ProgramWorkCache

    cache, evaluated = ProgramWorkCache(2), []

    def compute():
        evaluated.append(1)
        return len(evaluated)

    assert cache.get("binding", "exact-state-a", compute) == 1
    assert cache.get("binding", "exact-state-a", compute) == 1
    assert cache.get("binding", "exact-state-b", compute) == 2
    assert cache.get("execution", "exact-state-a", compute) == 3
    assert len(cache.entries) == 2
    assert cache.get("binding", "exact-state-a", compute) == 4


def test_untried_mutations_preserve_conditional_history_across_resume():
    search = optimizer()
    search.config = replace(
        search.config,
        mutation_sampling="untried",
        double_mutation_probability=0,
        mutation_context_limit=2,
    )
    entry = next(iter(search.entries.values()))
    choices = []
    for _ in range(5):
        _, program, binding, detail = search._mutate(entry)
        choices.append((program.program_id, binding))
        assert detail["mutations"][0]["sampling"] == "untried_conditional_choice"
    assert len(set(choices)) == 5
    assert len(search.mutation_choices_seen) == 1
    restored = ProgramOptimizer.restore(search.snapshot())
    left, right = search._mutate(entry), restored._mutate(entry)
    assert left[1:] == right[1:]


def test_score_blind_ablation_and_exhaustion_preserve_endpoint_exploration():
    search = optimizer()
    batch = search.propose_batch(eligible)
    search.observe_batch(batch["batch_id"], outcomes(search, batch, -5))
    search.duplicate_counts.clear()  # Equal exhaustion for the initial uniform check.
    keys, ranked = search.selection()
    search.config = replace(search.config, parent_allocation="score_blind")
    same_keys, uniform = search.selection()
    assert same_keys == keys
    assert np.allclose(uniform, np.full(len(keys), 1 / len(keys)))
    assert not np.allclose(ranked, uniform)
    search.duplicate_counts[keys[0]] = 10_000
    _, exhausted = search.selection()
    assert exhausted[0] >= search.config.exploration / len(keys)


def test_new_programs_record_constructor_and_measured_parent_sizes():
    search = optimizer()
    batch = search.propose_batch(eligible)
    for candidate in batch["candidates"]:
        size = candidate["provenance"]["program_size"]
        assert size["source_heavy_atoms"] == 2
        assert size["measured_parent_heavy_atoms"] == 9
        assert size["peak_heavy_atoms"] <= 40
        assert size["final_heavy_atoms"] - 9 == size["delta_from_measured_parent"]


def test_mutation_kind_never_draws_an_unavailable_contraction():
    from tests.test_edit_program import graph
    from tests.test_edit_program_graph import add_chain

    source = graph("CC")
    program, binding = add_chain(source, 0)
    _, trace = execute_program_graph(source, compile_program_graph(program), binding)
    search = ProgramOptimizer(
        ProgramSearchConfig(require_broad_runtime=False, double_mutation_probability=0),
        source_group="fixture",
        oracle_protocol="fixture:no-real-oracle",
    )
    key = search.add_measured_program(
        {
            "source_group": search.source_group,
            "oracle_protocol": search.oracle_protocol,
            "source_state": trace["states"][0],
            "program": program.payload(),
            "assignment": list(binding),
            "trace": trace,
            "endpoint": trace["endpoint"],
        },
        receipt_id="fixture:one-birth",
        score=-1,
    )
    for _ in range(20):
        _, _, _, metadata = search._mutate(search.entries[key])
        move = metadata["mutations"][0]
        assert move["kind"] in move["available_kinds"]
        assert "contract_segment" not in move["available_kinds"]
