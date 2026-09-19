from __future__ import annotations

from pathlib import Path

from compose_v4.benchmark.fragment_constrained import (
    FragmentPrompt,
    FragmentTask,
    check_fragment_constraint,
    load_genmol_prompts,
)
from compose_v4.benchmark.fragment_constrained_runner import (
    ProposalLimits,
    prompt_id,
    propose_prompt,
    run_smoke_panel,
)
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.rewrite.trace_shard import decode_state

ASSET = Path("data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv")


def _prompt(task: FragmentTask, *fragments: str) -> FragmentPrompt:
    return FragmentPrompt("fixture", "CCOc1ccccc1", task, tuple(fragments))


def test_generic_two_fragment_program_never_commits_disconnected_state() -> None:
    prompt = _prompt(
        FragmentTask.LINKER_DESIGN,
        "[1*]c1ccccc1",
        "[2*]N1CCCCC1",
    )
    proposal = propose_prompt(prompt, variant_index=1)

    assert proposal["status"] == "complete"
    assert proposal["source_fragment_index"] in {0, 1}
    assert proposal["program"]["primitive_edits"] <= 32
    assert proposal["program"]["dependencies"] == ((0, 1),)
    assert proposal["validation"]["fragment_constraint_satisfied"]
    assert all(
        row["all_locked_states_satisfied"]
        for row in proposal["validation"]["fragment_locks"]
    )
    states = [decode_state(row) for row in proposal["receipt"]["states"]]
    assert all(is_valid_state(state) for state in states)
    assert all(is_connected_or_null(state) for state in states)
    assert check_fragment_constraint(prompt, proposal["endpoint"]).satisfied


def test_original_reference_is_not_a_proposal_input() -> None:
    left = FragmentPrompt(
        "left",
        "CCOc1ccccc1",
        FragmentTask.MOTIF_EXTENSION,
        ("[1*]c1ccccc1",),
    )
    right = FragmentPrompt(
        "right",
        "CCNc1ccccc1",
        FragmentTask.MOTIF_EXTENSION,
        ("[1*]c1ccccc1",),
    )
    left_result = propose_prompt(left, variant_index=2)
    right_result = propose_prompt(right, variant_index=2)

    assert prompt_id(left) == prompt_id(right)
    assert left_result["status"] == right_result["status"] == "complete"
    assert left_result["endpoint"] == right_result["endpoint"]
    assert left_result["program"]["program_id"] == right_result["program"]["program_id"]
    assert not left_result["reference_original_used_for_proposal"]
    assert not right_result["reference_original_used_for_proposal"]


def test_every_task_label_has_a_complete_generic_fixture() -> None:
    prompts = (
        _prompt(FragmentTask.LINKER_DESIGN, "[1*]c1ccccc1", "[2*]N1CCCCC1"),
        _prompt(FragmentTask.SCAFFOLD_MORPHING, "[1*]c1ccccc1", "[2*]N1CCCCC1"),
        _prompt(FragmentTask.MOTIF_EXTENSION, "[1*]c1ccccc1"),
        _prompt(FragmentTask.SCAFFOLD_DECORATION, "[1*]c1cc([2*])ccc1"),
        _prompt(FragmentTask.SUPERSTRUCTURE_GENERATION, "c1ccccc1"),
    )
    results = [propose_prompt(prompt) for prompt in prompts]

    assert {result["task"] for result in results} == {
        task.value for task in FragmentTask
    }
    assert all(result["status"] == "complete" for result in results)
    assert all(result["oracle_calls"] == 0 for result in results)
    assert all(result["scoring_calls"] == 0 for result in results)


def test_multi_site_decoration_completes_every_declared_site() -> None:
    prompt = _prompt(
        FragmentTask.SCAFFOLD_DECORATION,
        "[1*]c1cc([2*])c([3*])cc1",
    )
    proposal = propose_prompt(prompt)

    assert proposal["status"] == "complete"
    assert len(proposal["program"]["blocks"]) == 3
    assert check_fragment_constraint(prompt, proposal["endpoint"]).satisfied


def test_bounded_runner_abstains_instead_of_relaxing_primitive_budget() -> None:
    prompt = _prompt(
        FragmentTask.LINKER_DESIGN,
        "[1*]c1ccccc1",
        "[2*]N1CCCCC1",
    )
    proposal = propose_prompt(
        prompt,
        limits=ProposalLimits(max_primitives=1),
    )

    assert proposal["status"] == "abstained"
    assert proposal["reason_code"] == "primitive_budget"
    assert proposal["oracle_calls"] == 0


def test_frozen_manifest_smoke_is_exactly_one_prompt_per_task() -> None:
    prompts = load_genmol_prompts(ASSET)
    smoke = run_smoke_panel(prompts)

    assert smoke["frozen_manifest_prompt_count"] == 50
    assert smoke["attempted_prompts"] == 5
    assert smoke["completed"] == 5
    assert smoke["abstained"] == 0
    assert smoke["oracle_calls"] == smoke["scoring_calls"] == 0
    assert [row["task"] for row in smoke["proposals"]] == [
        task.value for task in FragmentTask
    ]
