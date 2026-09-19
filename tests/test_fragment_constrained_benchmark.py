from __future__ import annotations

import csv
from pathlib import Path

import pytest

from compose_v4.benchmark.fragment_constrained import (
    FragmentPrompt,
    FragmentTask,
    check_fragment_constraint,
    evaluate_prompt,
    load_genmol_prompts,
)

ASSET = Path("data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv")


def _prompt(task: FragmentTask, *fragments: str) -> FragmentPrompt:
    return FragmentPrompt("fixture", "Cc1ccccc1", task, tuple(fragments))


def test_released_asset_loads_all_ten_drugs_and_five_tasks() -> None:
    prompts = load_genmol_prompts(ASSET)
    assert len(prompts) == 50
    assert len({prompt.drug_name for prompt in prompts}) == 10
    assert {prompt.task for prompt in prompts} == set(FragmentTask)

    linker = {
        prompt.drug_name: prompt.fragments
        for prompt in prompts
        if prompt.task is FragmentTask.LINKER_DESIGN
    }
    morphing = {
        prompt.drug_name: prompt.fragments
        for prompt in prompts
        if prompt.task is FragmentTask.SCAFFOLD_MORPHING
    }
    assert linker == morphing
    assert all(len(fragments) == 2 for fragments in linker.values())


def test_every_released_reference_drug_satisfies_its_derived_constraint() -> None:
    prompts = load_genmol_prompts(ASSET)
    failures = [
        (prompt.drug_name, prompt.task.value)
        for prompt in prompts
        if not check_fragment_constraint(prompt, prompt.original_smiles).satisfied
    ]
    assert failures == []


def test_motif_requires_the_declared_attachment_site_to_be_extended() -> None:
    prompt = _prompt(FragmentTask.MOTIF_EXTENSION, "[1*]c1ccccc1")
    assert check_fragment_constraint(prompt, "Cc1ccccc1").satisfied
    result = check_fragment_constraint(prompt, "c1ccccc1")
    assert not result.satisfied
    assert result.reason == "missing_fragment_or_attachment"


def test_decoration_requires_every_attachment_site() -> None:
    prompt = _prompt(FragmentTask.SCAFFOLD_DECORATION, "[1*]c1cc([2*])ccc1")
    assert check_fragment_constraint(prompt, "Cc1cc(O)ccc1").satisfied
    assert not check_fragment_constraint(prompt, "Cc1ccccc1").satisfied


def test_linker_requires_nonoverlapping_completed_fragments() -> None:
    prompt = _prompt(
        FragmentTask.LINKER_DESIGN,
        "[1*]c1ccccc1",
        "[2*]N1CCCCC1",
    )
    assert check_fragment_constraint(prompt, "c1ccc(CCN2CCCCC2)cc1").satisfied
    assert not check_fragment_constraint(prompt, "Cc1ccccc1").satisfied
    assert (
        check_fragment_constraint(prompt, "c1ccccc1.CN1CCCCC1").reason == "disconnected"
    )


def test_superstructure_is_core_containment_without_invented_attachment_sites() -> None:
    prompt = _prompt(FragmentTask.SUPERSTRUCTURE_GENERATION, "c1ccccc1")
    assert check_fragment_constraint(prompt, "Cc1ccccc1").satisfied
    assert check_fragment_constraint(prompt, "c1ccccc1").satisfied
    assert not check_fragment_constraint(prompt, "C1CCCCC1").satisfied


def test_metrics_keep_chemical_and_fragment_validity_separate() -> None:
    prompt = _prompt(FragmentTask.MOTIF_EXTENSION, "[1*]c1ccccc1")
    samples = ("Cc1ccccc1", "Cc1ccccc1", "not_smiles", "c1ccccc1")
    metrics = evaluate_prompt(prompt, samples, expected_samples=4)
    assert metrics.chemical_validity == pytest.approx(0.75)
    assert metrics.constraint_validity == pytest.approx(2 / 3)
    assert metrics.benchmark_validity == pytest.approx(0.5)
    assert metrics.uniqueness == pytest.approx(0.5)
    assert metrics.unique_constraint_valid_count == 1
    assert metrics.diversity == 0.0
    assert 0.0 <= metrics.quality <= metrics.benchmark_validity
    assert 0.0 <= metrics.central_distance <= 1.0


def test_metric_denominator_cannot_silently_change() -> None:
    prompt = _prompt(FragmentTask.SUPERSTRUCTURE_GENERATION, "c1ccccc1")
    with pytest.raises(ValueError, match="expected exactly 100 samples"):
        evaluate_prompt(prompt, ["c1ccccc1"])


def test_loader_rejects_schema_drift(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    with bad.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("name", "smiles"))
        writer.writerow(("drug", "CC"))
    with pytest.raises(ValueError, match="unexpected fragment benchmark columns"):
        load_genmol_prompts(bad)
