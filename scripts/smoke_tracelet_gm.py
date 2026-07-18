"""Reproducible from-scratch learning smoke gate for tracelet Generator Matching."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.experiments.tracelet_conditional import (
    build_tracelet_path_records,
    sample_tracelet_ancestral,
    sample_tracelet_conditional_batch,
    tracelet_conditional_metrics,
    train_tracelet_conditional_model,
)
from compose_v4.model.tracelet_rate_model import TraceletRateModel
from compose_v4.rewrite.kernel import canonical_state_key


TRAIN_SMILES = (
    "c1ccccc1",
    "c1ccncc1",
    "c1ccoc1",
    "C1CCCCC1",
    "C1CCCC1",
    "c1ccc2ccccc2c1",
    "C1CC2CCC1C2",
    "C1CCC2(CC1)CCCC2",
)
VALIDATION_SMILES = ("c1ncccc1", "C1=CC=CC=C1", "C1CCCCCCC1")


def main() -> None:
    seed = 20260714
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    train_records = build_tracelet_path_records(TRAIN_SMILES, n_slots=12)
    validation_records = build_tracelet_path_records(VALIDATION_SMILES, n_slots=12)
    fiber_cache = {}
    validation_examples = sample_tracelet_conditional_batch(
        validation_records,
        batch_size=18,
        rng=np.random.default_rng(7),
        fiber_cache=fiber_cache,
        late_time_fraction=0.5,
        operational_horizon=5.0,
    )
    model = TraceletRateModel(hidden_dim=24, message_passing_steps=2)
    initial = tracelet_conditional_metrics(model, validation_examples)
    history, best = train_tracelet_conditional_model(
        model,
        train_records,
        validation_examples,
        steps=60,
        batch_size=4,
        learning_rate=2e-3,
        seed=8,
        fiber_cache=fiber_cache,
        late_time_fraction=0.5,
        operational_horizon=5.0,
        evaluation_points=10,
    )
    rng = np.random.default_rng(9)
    rollouts = tuple(
        sample_tracelet_ancestral(
            model,
            rng=rng,
            n_slots=12,
            operational_horizon=5.0,
            time_step=0.2,
            max_events=20,
        )
        for _ in range(20)
    )
    rules = Counter(rule for rollout in rollouts for rule in rollout.event_rules)
    report = {
        "model": "from_scratch_tracelet_generator_matching",
        "uses_corpus_frequency_prior": False,
        "seed": seed,
        "initial": initial,
        "best": best,
        "last_history": history[-1],
        "rollouts": len(rollouts),
        "valid": sum(is_valid_state(item.final_state) for item in rollouts),
        "connected": sum(
            is_connected_or_null(item.final_state) for item in rollouts
        ),
        "nonnull": sum(item.final_state.n_real_atoms > 0 for item in rollouts),
        "unique": len(
            {canonical_state_key(item.final_state) for item in rollouts}
        ),
        "event_rules": dict(sorted(rules.items())),
        "mean_events": float(
            np.mean([len(item.event_times) for item in rollouts])
        ),
    }
    output = Path("results/tracelet_gm_smoke.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
