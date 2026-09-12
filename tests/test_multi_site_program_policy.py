"""Proposal arms retain exact traces and never evaluate partial task utility."""

import numpy as np
import pytest
from test_edit_program import graph
from test_edit_program_graph import add_chain

from compose_v4.control.edit_program_graph import combine_bound_programs
from compose_v4.control.edit_program_policy import (
    ProgramEntry,
    choose_channel,
    source_balanced_prior,
)
from compose_v4.control.multi_site_program_policy import MODES, generate_program_candidates


def test_three_modes_execute_without_task_target_and_report_actual_sites():
    source = graph("CCCCCC")
    left, right = add_chain(source, 0), add_chain(source, 5)
    program, _ = combine_bound_programs(source, (left, right))
    entries = (ProgramEntry(left[0], ("train_source",)),)
    joint = (ProgramEntry(program, ("train_source",)),)
    for mode in MODES:
        result = generate_program_candidates(
            source,
            entries,
            joint_entries=joint,
            mode=mode,
            seed=17,
            attempts=3,
            max_primitives=2,
            max_blocks=2,
        )
        assert result["attempts_completed"] == 3
        assert result["intermediate_task_evaluations"] == 0
        assert result["inference_target_supplied"] is False
        assert result["unique_endpoint_first_attempt"]
        completed = [row for row in result["rows"] if row["status"] == "complete"]
        assert completed
        assert all(r["receipt"]["primitive_edits"] == 2 for r in completed)
        if mode == "joint_multi_site":
            assert all(r["receipt"]["actual_changes"]["changed_site_count"] == 2 for r in completed)


def test_joint_model_is_required_and_budgets_reject_instead_of_returning_prefix():
    source = graph("CCCCCC")
    entry = ProgramEntry(add_chain(source, 0)[0], ("train_source",))
    with pytest.raises(ValueError, match="coordinated-program bank"):
        generate_program_candidates(source, (entry,), mode="joint_multi_site", seed=1)
    result = generate_program_candidates(
        source, (entry,), mode="uninterrupted_serial", seed=1, attempts=2, max_primitives=1
    )
    assert not result["unique_endpoint_first_attempt"]
    assert all(row["reason_code"] == "complete_program_budget" for row in result["rows"])


def test_source_balancing_and_broad_channel_are_explicit():
    source = graph("CCCCCC")
    one, two, three = (add_chain(source, 0, length)[0] for length in (1, 2, 3))
    entries = (ProgramEntry(one, ("a",)), ProgramEntry(two, ("a",)), ProgramEntry(three, ("b",)))
    assert np.allclose(source_balanced_prior(entries), (0.25, 0.25, 0.5))
    for probability in (0, 1, float("nan")):
        with pytest.raises(ValueError, match="positive probability"):
            choose_channel(np.random.default_rng(0), program_probability=probability)
    rng = np.random.default_rng(0)
    assert {choose_channel(rng) for _ in range(100)} == {"program", "reference"}


def test_serial_horizon_is_not_truncated_to_the_number_of_mutable_sites():
    source = graph("CCCCCC")
    entry = ProgramEntry(add_chain(source, 0)[0], ("train_source",))
    result = generate_program_candidates(
        source,
        (entry,),
        mode="uninterrupted_serial",
        seed=17,
        attempts=3,
        sites=2,
        max_blocks=5,
        max_primitives=5,
    )
    complete = [row for row in result["rows"] if row["status"] == "complete"]
    assert complete
    assert all(row["receipt"]["primitive_edits"] == 5 for row in complete)
