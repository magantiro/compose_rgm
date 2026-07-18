"""Controlled target-free diagnostics for a trained C/N/O/F checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from compose_v4.experiments.cnof_conditional import (
    corpus_rollout_metrics,
    sample_factorized_ancestral,
)
from compose_v4.experiments.corpus_marginal import CorpusMarginalRateModel
from compose_v4.experiments.prior_tilted import PriorTiltedRewriteRateModel
from compose_v4.model.rate_model import FactorizedRateModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cnof_gate_sampler_diagnostics.json"),
    )
    args = parser.parse_args()

    torch.set_num_threads(1)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    split = payload["split"]
    reference = (*split.train, *split.validation, *split.test)
    n_slots = int(payload["report"]["split"]["max_atoms"])

    trained, comparison, comparison_name = _models_from_checkpoint(payload)
    trained.load_state_dict(payload["model_state_dict"])
    trained.eval()

    configurations = (
        ("dt_0.20", trained, 7.0, 0.20),
        ("dt_0.10", trained, 7.0, 0.10),
        ("dt_0.05", trained, 7.0, 0.05),
    )
    evaluations = {}
    for index, (name, model, horizon, time_step) in enumerate(configurations):
        evaluations[name] = _evaluate(
            model,
            samples=args.samples,
            seed=int(payload["seed"]) + 100 + index,
            n_slots=n_slots,
            horizon=horizon,
            time_step=time_step,
            train_smiles=split.train,
            reference_smiles=reference,
        )

    comparison.eval()
    evaluations[comparison_name] = _evaluate(
        comparison,
        samples=args.samples,
        seed=int(payload["seed"]) + 200,
        n_slots=n_slots,
        horizon=7.0,
        time_step=0.10,
        train_smiles=split.train,
        reference_smiles=reference,
    )

    report = {
        "checkpoint": str(args.checkpoint),
        "samples_per_configuration": args.samples,
        "target_available": False,
        "evaluations": evaluations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def _models_from_checkpoint(payload):
    kwargs = {
        "hidden_dim": int(payload["hidden_dim"]),
        "message_passing_steps": int(payload["message_passing_steps"]),
        "use_rewrite_context": bool(payload.get("use_rewrite_context", False)),
        "use_topology_context": bool(payload.get("use_topology_context", False)),
    }
    model_type = payload.get("model_type", "factorized_rewrite_generator")
    torch.manual_seed(int(payload["seed"]))
    if model_type == "prior_tilted_rewrite_generator":
        table = payload.get("corpus_marginal_table")
        if table is None:
            raise ValueError("prior-tilted checkpoint is missing its corpus prior")
        prior = CorpusMarginalRateModel(table)
        trained = PriorTiltedRewriteRateModel(prior, **kwargs)
        comparison = PriorTiltedRewriteRateModel(prior, **kwargs)
        comparison_name = "corpus_prior_init_dt_0.10"
    elif model_type == "factorized_rewrite_generator":
        trained = FactorizedRateModel(**kwargs)
        comparison = FactorizedRateModel(**kwargs)
        comparison_name = "untrained_dt_0.10"
    else:
        raise ValueError(f"unsupported checkpoint model_type: {model_type}")
    return trained, comparison, comparison_name


def _evaluate(
    model,
    *,
    samples: int,
    seed: int,
    n_slots: int,
    horizon: float,
    time_step: float,
    train_smiles: tuple[str, ...],
    reference_smiles: tuple[str, ...],
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    fiber_cache = {}
    rate_cache = {}
    rollouts = tuple(
        sample_factorized_ancestral(
            model,
            rng=rng,
            n_slots=n_slots,
            operational_horizon=horizon,
            time_step=time_step,
            fiber_cache=fiber_cache,
            rate_cache=rate_cache,
        )
        for _ in range(samples)
    )
    metrics = corpus_rollout_metrics(
        rollouts,
        train_smiles=train_smiles,
        reference_smiles=reference_smiles,
    )
    metrics["operational_horizon"] = horizon
    metrics["time_step"] = time_step
    return metrics


if __name__ == "__main__":
    main()
