"""Shared program logic, unscored cold starts and exact recipient identity."""

from dataclasses import replace

import pytest

from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
from compose_v4.control.edit_program import extract_program
from compose_v4.control.program_transfer import initial_program_batch, shared_program_library
from tests.test_edit_program import graph, ring_and_carbonyl


def test_shared_library_deduplicates_without_reading_endpoint_or_task_scores():
    source, stages = ring_and_carbonyl()
    program, _ = extract_program(source, stages)
    row = {"program": program.payload(), "source_group": "original", "oracle_label": object()}
    library = shared_program_library([row, row, {**row, "source_group": "another"}])
    assert len(library) == 1
    assert library[0].source_groups == ("another", "original")
    assert stages[-1]["endpoint"] not in str(library[0].program.payload())


def test_cold_start_transfers_programs_and_leaves_new_target_scores_unobserved():
    source, stages = ring_and_carbonyl()
    program, _ = extract_program(source, stages)
    entries = shared_program_library([{"program": program.payload(), "source_group": "origin"}])
    config = ProgramSearchConfig(
        seed=871, attempts_per_batch=32, candidates_per_batch=2, require_broad_runtime=False
    )
    recipient = graph("CCCO")
    call = {
        "source_group": "recipient",
        "oracle_protocol": "pending:recipient",
        "eligibility": lambda row: {"smiles": row["smiles"], "oracle_eligible": True},
    }
    first = initial_program_batch(recipient, entries, config, **call)
    second = initial_program_batch(recipient, entries, config, **call)
    assert first["batch_id"] == second["batch_id"]
    assert len(first["candidates"]) == 2
    assert all(c["score"] is None and c["source_group"] == "recipient" for c in first["candidates"])
    assert first["new_oracle_calls"] == 0
    with pytest.raises(ValueError, match="authenticated broad"):
        initial_program_batch(
            recipient, entries, replace(config, require_broad_runtime=True), **call
        )


def test_cold_start_can_retrieve_an_unmutated_context_matched_program():
    source, stages = ring_and_carbonyl()
    program, _ = extract_program(source, stages)
    entries = shared_program_library([{"program": program.payload(), "source_group": "origin"}])
    config = ProgramSearchConfig(
        seed=871,
        attempts_per_batch=16,
        candidates_per_batch=1,
        require_broad_runtime=False,
        cold_start_retrieval_candidates=1,
    )
    batch = initial_program_batch(
        source,
        entries,
        config,
        source_group="recipient",
        oracle_protocol="pending:recipient",
        eligibility=lambda row: {"smiles": row["smiles"], "oracle_eligible": True},
    )
    assert len(batch["candidates"]) == 1
    candidate = batch["candidates"][0]
    assert candidate["program"] == program.payload()
    assert candidate["endpoint"] == stages[-1]["endpoint"]
    assert candidate["provenance"]["channel"] == "retrieval"
    assert candidate["provenance"]["metadata"]["mutations"] == []
    assert batch["direct_retrieval"] == {
        "candidate_target": 1,
        "candidates_admitted": 1,
        "priority_attempts": 1,
    }
