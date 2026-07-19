from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from compose_v4.data.cnof import load_cnof_corpus_split
from compose_v4.experiments.cnof_conditional import (
    build_path_records,
    conditional_metrics,
    fixed_conditional_examples,
    sample_factorized_ancestral,
    train_conditional_model,
)
from compose_v4.model.rate_model import FactorizedRateModel


def test_corpus_filter_split_and_short_conditional_training(tmp_path: Path) -> None:
    corpus = tmp_path / "molecules.smi"
    corpus.write_text(
        "\n".join(
            (
                "C",
                "N",
                "O",
                "F",
                "CC",
                "CO",
                "CN",
                "C=C",
                "C#N",
                "CCO",
                "CCN",
                "C1CC1",
                "CCCl",
                "C[N+](C)(C)C",
            )
        )
        + "\n"
    )
    split = load_cnof_corpus_split(
        corpus,
        train_size=8,
        validation_size=2,
        test_size=2,
        max_atoms=3,
        seed=3,
    )
    assert len(split.train) == 8
    assert not set(split.train) & set(split.validation)
    assert not set(split.train) & set(split.test)

    train = build_path_records(split.train, n_slots=3, seed=3)
    validation = build_path_records(split.validation, n_slots=3, seed=4)
    cache = {}
    examples = fixed_conditional_examples(
        validation,
        samples_per_record=3,
        seed=5,
        fiber_cache=cache,
    )
    torch.manual_seed(3)
    model = FactorizedRateModel(hidden_dim=16, message_passing_steps=1)
    initial = conditional_metrics(model, examples)
    history, final = train_conditional_model(
        model,
        train,
        examples,
        steps=20,
        batch_size=4,
        learning_rate=3e-3,
        seed=6,
        fiber_cache=cache,
    )
    # This is an integration smoke test, not a claim that twenty stochastic
    # updates on eight training molecules must improve a two-molecule holdout.
    assert len(history) == 20
    assert math.isfinite(initial["conditional_gm_loss"])
    assert math.isfinite(final["conditional_gm_loss"])
    assert all(math.isfinite(item["train_batch_loss"]) for item in history)

    rollout = sample_factorized_ancestral(
        model,
        rng=np.random.default_rng(7),
        n_slots=3,
        operational_horizon=0.5,
    )
    assert rollout.final_state.n_atoms == 3
    assert rollout.diagnostics is not None
    assert len(rollout.diagnostics.canonical_state_keys) == len(rollout.event_rules) + 1
    assert all(rollout.diagnostics.state_valid)
    assert all(rollout.diagnostics.state_connected_or_null)


def test_scaffold_split_has_no_cross_partition_scaffold_overlap(tmp_path: Path) -> None:
    corpus = tmp_path / "scaffolds.smi"
    corpus.write_text(
        "\n".join(
            (
                "C",
                "CC",
                "CCC",
                "c1ccccc1",
                "Cc1ccccc1",
                "Oc1ccccc1",
                "c1ccncc1",
                "Cc1ccncc1",
                "C1CCCCC1",
                "CC1CCCCC1",
                "C1CCCC1",
                "CC1CCCC1",
                "c1ncc[nH]1",
                "c1cncnc1",
                "C1CC1",
            )
        )
        + "\n"
    )
    split = load_cnof_corpus_split(
        corpus,
        train_size=8,
        validation_size=4,
        test_size=3,
        max_atoms=8,
        seed=11,
        split_strategy="scaffold",
    )
    assert sum(map(len, (split.train, split.validation, split.test))) == 15
    assert split.eligible_molecules == 15
    assert split.split_strategy == "scaffold"
    assert split.scaffold_overlap_count == 0
    assert not set(split.train) & set(split.validation)
    assert not set(split.train) & set(split.test)
