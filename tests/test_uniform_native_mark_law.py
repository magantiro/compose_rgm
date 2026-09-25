from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from compose_v4.benchmark.fragment_attachment_control import AttachmentControlConfig
from compose_v4.benchmark.fragment_conditioned_sampler import (
    SamplerConfig,
    build_prompt_context,
    sample_completion,
)
from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts
from compose_v4.benchmark.uniform_native_mark_law import UniformNativeMarkLaw
from compose_v4.rewrite.kernel import de_novo_rewrite_system

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = Path("/Users/rmaganti/compose_fragment_ckpt/ringcore_a7546e2_best.pt")


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="documented RingCore fragment asset absent")
def test_uniform_mark_law_runs_in_verified_superstructure_sampler():
    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    model, _ = load_factorized_rollout_checkpoint(CHECKPOINT)
    adapter = UniformNativeMarkLaw(model)
    prompt = next(
        prompt
        for prompt in load_genmol_prompts(
            ROOT / "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"
        )
        if prompt.task is FragmentTask.SUPERSTRUCTURE_GENERATION
    )
    context = build_prompt_context(prompt)
    output = sample_completion(
        adapter,
        de_novo_rewrite_system(),
        context,
        np.random.default_rng(1),
        config=SamplerConfig(max_events=2, operational_horizon=16, mark_attempts_per_event=2),
        control=AttachmentControlConfig(
            enabled=True,
            condition_initial_locked_family=True,
            hard_lock_effective_chemistry=True,
        ),
    )
    assert output is not None
    assert output != ""


def test_uniform_mark_law_rejects_unsupported_ring_macro_configuration():
    class InvalidModel:
        enable_ring_grow_macro = True

    with pytest.raises(ValueError, match="primitive-ring"):
        UniformNativeMarkLaw(InvalidModel())
