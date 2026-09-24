"""Motif-focused development law and exact published-metric denominator."""

import copy

import numpy as np
import pytest
from run_fragment_motif_focused_dev_v1 import attempt_samples, load_contract, preflight

from compose_v4.benchmark.fragment_constrained import FragmentTask
from compose_v4.benchmark.fragment_motif_focused_programs import sample_motif_panel


def test_motif_metric_population_keeps_failed_attempt_and_all_eight_offers():
    rows = [
        {
            "drug": "fixture",
            "attempt_index": index,
            "panel": {
                "offered_count": 8,
                "offered": [{"draw": draw} for draw in range(8)],
                "selected_smiles": None if index == 0 else "CC",
            },
        }
        for index in range(20)
    ]
    samples = attempt_samples(rows, drug="fixture")
    assert len(samples) == 20 and samples[0] == ""
    with pytest.raises(ValueError, match="twenty ordered"):
        attempt_samples(rows[:-1], drug="fixture")
    rows[1]["panel"]["offered"].pop()
    with pytest.raises(ValueError, match="eight recorded offers"):
        attempt_samples(rows, drug="fixture")


def test_focused_motif_panel_uses_eight_coherent_offers_and_preserves_no_output(monkeypatch):
    import compose_v4.benchmark.fragment_motif_focused_programs as module

    def refuse(*_args):
        raise ValueError("fixture compiler refusal")

    monkeypatch.setattr(module, "propose_joint_region_completion", refuse)
    context = type(
        "Context", (), {"prompt": type("Prompt", (), {"task": FragmentTask.MOTIF_EXTENSION})()}
    )()
    rng = np.random.default_rng(19)
    before = copy.deepcopy(rng.bit_generator.state)
    result = sample_motif_panel(context, None, None, rng)
    assert result.selected is None and result.receipt["output_count"] == 0
    assert [entry["draw"] for entry in result.receipt["offered"]] == list(range(8))
    assert result.receipt["rng_state_before"] == before
    assert not result.receipt["qed_sa_guidance"]
    context.prompt.task = FragmentTask.SCAFFOLD_DECORATION
    with pytest.raises(ValueError, match="motif-extension"):
        sample_motif_panel(context, None, None, rng)


@pytest.mark.parametrize("task", ("motif_extension", "scaffold_decoration"))
def test_focused_development_contracts_pin_inputs_and_ten_prompts(task):
    contract, digest = load_contract(task)
    versions, prompts, _ = preflight(contract)
    assert len(digest) == 64
    assert len(prompts) == 10
    assert [prompt.drug_name for prompt in prompts] == contract["drugs"]
    assert versions == contract["versions"]
