"""Freeze the matched de novo molecular-prior scale comparison."""

from __future__ import annotations

from pathlib import Path

from compose_v4.experiments.recipe import load_tracelet_recipe

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_prior_scale_arms_differ_only_in_data_size_and_outputs():
    small = load_tracelet_recipe(ROOT / "recipes/denovo_frozen_cnof_prior_small_v1.json")[
        "arguments"
    ]
    large = load_tracelet_recipe(ROOT / "recipes/denovo_frozen_cnof_prior_large_v1.json")[
        "arguments"
    ]

    assert small["train_size"] == 50_000
    assert large["train_size"] == 216_149
    assert small["frozen_cnof_split_manifest"] == large["frozen_cnof_split_manifest"]
    assert small["model"] == large["model"] == "from_scratch"
    assert small["skip_rollouts"] is large["skip_rollouts"] is True
    assert small["quality_metrics"] is large["quality_metrics"] is False

    different = {key for key in small | large if small.get(key) != large.get(key)}
    assert different == {"train_size", "output", "checkpoint"}
