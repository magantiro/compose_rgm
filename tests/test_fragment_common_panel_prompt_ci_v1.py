"""Prompt-unit accounting for the completed common-panel comparisons."""

from analyze_fragment_common_panel_prompt_ci_v1 import analyze


def test_prompt_bootstrap_keeps_ten_structural_units() -> None:
    result = analyze()
    assert set(result["tasks"]) == {"motif_extension", "linker_design"}
    for task in result["tasks"].values():
        assert len(task["per_prompt"]) == 10
        for metric in task["estimates"].values():
            assert (
                metric["prompts_learned_higher"]
                + metric["prompts_equal"]
                + metric["prompts_uniform_higher"]
                == 10
            )
    assert (
        abs(
            result["tasks"]["motif_extension"]["estimates"]["quality"]["paired_prompt_mean"]
            - (42.63333333333333 - 37.2)
        )
        < 1e-9
    )
