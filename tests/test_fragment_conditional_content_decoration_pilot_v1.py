"""Focused contract and denominator checks for the matched decoration pilot."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import run_fragment_conditional_content_decoration_pilot_v1 as pilot
from run_fragment_conditional_content_decoration_pilot_v1 import (
    CATALOG,
    CONTENT,
    CONTRACT,
    MASS,
    PROMPTS,
    _attempt_samples,
    identity,
)

from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts
from compose_v4.benchmark.training_attachment_fragments import physical_sha256


def test_frozen_contract_and_train_only_inputs() -> None:
    envelope = json.loads(CONTRACT.read_text())
    assert identity(envelope["payload"]) == envelope["payload_sha256"]
    contract = envelope["payload"]
    prompts = [
        prompt
        for prompt in load_genmol_prompts(PROMPTS)
        if prompt.task is FragmentTask.SCAFFOLD_DECORATION
    ]
    assert len(prompts) == 10
    assert len({prompt.drug_name for prompt in prompts}) == 10
    assert [prompt.drug_name for prompt in prompts] == contract["drugs"]
    assert physical_sha256(PROMPTS) == contract["prompt_sha256"]
    assert physical_sha256(CATALOG) == contract["catalog_sha256"]
    assert physical_sha256(MASS) == contract["mass_prior_sha256"]
    assert physical_sha256(CONTENT) == contract["content_prior_sha256"]
    catalog, mass, content = (json.loads(path.read_text()) for path in (CATALOG, MASS, CONTENT))
    assert catalog["split"] == content["split"]
    assert mass["split"] == catalog["split"]
    assert content["benchmark_prompts_used_to_fit"] is False
    assert content["quality_labels_used"] is False
    assert contract["promotion_gate"]["minimum_quality_gain_points"] == 5.0


def test_attempt_denominator_and_eight_offer_receipts() -> None:
    attempts = [
        {
            "drug": "TEST",
            "attempt_index": index,
            "panel": {
                "offered_count": 8,
                "offered": [{"draw": draw} for draw in range(8)],
                "selected_smiles": "CC" if index % 2 == 0 else None,
            },
        }
        for index in range(10)
    ]
    samples = _attempt_samples(attempts, "TEST", 10)
    assert len(samples) == 10
    assert samples[0] == "CC"
    assert samples[1] != "CC"
    with pytest.raises(ValueError, match="ordered attempt denominator"):
        _attempt_samples(attempts[:-1], "TEST", 10)
    attempts[0]["panel"]["offered"].pop()
    with pytest.raises(ValueError, match="eight recorded offers"):
        _attempt_samples(attempts, "TEST", 10)


def test_completed_attempts_resume_without_redrawing(monkeypatch, tmp_path) -> None:
    prompt = SimpleNamespace(drug_name="TEST", task=FragmentTask.SCAFFOLD_DECORATION)
    context = SimpleNamespace(start_state=object())
    monkeypatch.setattr(pilot, "build_prompt_context", lambda _: context)
    monkeypatch.setattr(
        pilot,
        "ProgramConstraint",
        SimpleNamespace(from_context=lambda _: SimpleNamespace(lock=lambda _: object())),
    )
    monkeypatch.setattr(
        pilot,
        "official_prompt_metrics",
        lambda samples, *, expected_samples: {
            "validity": 0.0,
            "uniqueness": 0.0,
            "quality": 0.0,
            "diversity": 0.0,
        },
    )
    calls = []

    def sample(_, sampler, model, rng):
        calls.append(len(calls))
        before = rng.bit_generator.state
        rng.integers(0, 10)
        return SimpleNamespace(
            selected=None,
            receipt={
                "rng_state_before": before,
                "rng_state_after": rng.bit_generator.state,
                "offered_count": 8,
                "offered": [{"draw": draw, "status": "abstention"} for draw in range(8)],
                "output_count": 0,
                "selected_smiles": None,
            },
        )

    monkeypatch.setattr(pilot, "sample_pendant_panel", sample)
    first = pilot._run_arm_prompt(
        output=tmp_path,
        manifest_hash="test-manifest",
        prompt=prompt,
        arm="joint_mass",
        sampler=None,
        model=None,
        count=2,
    )
    second = pilot._run_arm_prompt(
        output=tmp_path,
        manifest_hash="test-manifest",
        prompt=prompt,
        arm="joint_mass",
        sampler=None,
        model=None,
        count=2,
    )
    assert first == second
    assert calls == [0, 1]
