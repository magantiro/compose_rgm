"""Within-parent contrasts retain missing parts; query locks retain full pools."""

import numpy as np
import pytest

from compose_v4.control.adaptive_program_optimizer import ProgramOptimizer
from compose_v4.control.edit_learning_data import (
    exploration_niches,
    mutation_contrast_panel,
    observed_interaction,
)
from compose_v4.control.program_mutation import parameter_choices
from compose_v4.control.program_task import ProgramTask
from tests.test_adaptive_program_optimizer import eligible, optimizer, outcomes


def test_contrast_never_assigns_rewards_to_unscored_or_ineligible_parts():
    search = optimizer()
    entry = next(iter(search.entries.values()))
    program = search._program(entry)
    panel = mutation_contrast_panel(
        entry,
        attachment=entry["assignment"],
        parameter_move="extend_segment",
        parameter_choice=parameter_choices(program, "extend_segment")[0],
        eligibility=eligible,
    )
    task = ProgramTask("fixture", search.oracle_protocol, "pmo")
    result = observed_interaction(panel, [], task=task)
    assert result["status"] == "incomplete" and result["interaction"] is None
    labels = [
        {
            "endpoint": panel["arms"]["joint"]["endpoint"],
            "score": 0.8,
            "oracle_protocol": search.oracle_protocol,
            "receipt_id": "fixture:joint",
        }
    ]
    assert observed_interaction(panel, labels, task=task)["interaction"] is None
    assert panel["arms"]["joint"]["status"] == "eligible"
    labels.append(
        {
            "endpoint": entry["endpoint"],
            "score": 0.3,
            "oracle_protocol": search.oracle_protocol,
            "receipt_id": "fixture:parent",
        }
    )
    measured = observed_interaction(panel, labels, task=task)
    assert measured["status"] == "observed"
    assert measured["interaction"] == pytest.approx(0)
    assert measured["distinct_endpoints"] == 2  # This degenerate fixture is explicit.


def test_query_subset_and_resume_preserve_unqueried_candidates_and_attempts():
    search = optimizer()
    original = search.propose_batch(eligible)
    ids = [original["candidates"][0]["candidate_id"]]
    with pytest.raises(ValueError, match="unknown candidate"):
        search.lock_query_subset(original["batch_id"], ["absent"], {"selected_ids": ["absent"]})
    query = search.lock_query_subset(
        original["batch_id"], ids, {"selected_ids": ids, "mode": "fixture"}
    )
    assert query["proposal_pool"]["candidates"] == original["candidates"]
    assert query["attempts"] == original["attempts"]
    assert len(query["candidates"]) == 1 and query["batch_id"] != original["batch_id"]
    clone = ProgramOptimizer.restore(search.snapshot())
    labels = outcomes(search, query, -2)
    search.observe_batch(query["batch_id"], labels)
    clone.observe_batch(query["batch_id"], labels)
    assert search.snapshot() == clone.snapshot()
    assert len(search.observations) == 2  # Only selected endpoint was queried.


def test_small_niches_retain_non_global_leaders_without_changing_incumbent():
    molecules = ["CCCC", "CCCCC", "CCCCCC", "c1ccccc1", "c1ccncc1", "c1ccccc1O"]
    utilities = [6, 5, 4, 3, 2, 1]
    niches = exploration_niches(molecules, utilities, max_niches=2, per_niche=1)
    assert niches["incumbent"] == "CCCC"
    assert len(niches["selected"]) == 2
    assert any(s != "CCCC" for group in niches["selected"] for s in group)
    assert niches == exploration_niches(molecules, utilities, max_niches=2, per_niche=1)
    assert niches["objective_unchanged"]


def test_niche_allocation_keeps_a_floor_on_all_saved_parents():
    from dataclasses import replace

    search = optimizer()
    batch = search.propose_batch(eligible)
    search.observe_batch(batch["batch_id"], outcomes(search, batch, -5))
    search.config = replace(search.config, parent_allocation="niche_score")
    keys, weights = search.selection()
    assert np.isclose(sum(weights), 1)
    assert np.all(weights >= search.config.exploration / len(keys))


def test_shared_pmo_query_adapter_locks_selection_before_scoring():
    from dataclasses import replace

    from compose_v4.control.parent_edit_search import prepare_query_batch

    search = optimizer()
    # Explicit synthetic PMO observations, not transplanted docking labels.
    for observation in search.observations.values():
        observation["score"] = 0.2
    for entry in search.entries.values():
        entry["static_score"] = 0.2
    search.config = replace(search.config, score_direction="maximize")
    task = ProgramTask("fixture_pmo", search.oracle_protocol, "pmo")
    query = prepare_query_batch(search, task, count=1, seed=4)
    assert len(query["candidates"]) == 1
    assert len(query["proposal_pool"]["candidates"]) == 2
    assert query["selection"]["task_id"] == task.task_id
    assert query["selection"]["model_sha256"] is None
    assert len(search.observations) == 1
