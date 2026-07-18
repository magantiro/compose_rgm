"""Cross-evaluate sequential and causal-teacher checkpoints on frozen batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from compose_v4.data.cnof import load_cnof_corpus_split
from compose_v4.experiments.tracelet_conditional import (
    build_tracelet_path_records,
    sample_tracelet_conditional_batch,
    tracelet_conditional_metrics,
)
from compose_v4.model.tracelet_rate_model import TraceletRateModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("smiles_file", type=Path)
    parser.add_argument("sequential_checkpoint", type=Path)
    parser.add_argument("causal_checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--max-atoms", type=int, default=16)
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--validation-size", type=int, default=48)
    parser.add_argument("--test-size", type=int, default=48)
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--late-time-fraction", type=float, default=0.5)
    parser.add_argument("--operational-horizon", type=float, default=7.0)
    parser.add_argument("--fast-split", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(args.torch_threads)
    sequential_payload = torch.load(
        args.sequential_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    causal_payload = torch.load(
        args.causal_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    ring_catalog = sequential_payload["ring_catalog"]
    if ring_catalog != causal_payload["ring_catalog"]:
        raise ValueError("checkpoints use different typed ring catalogs")

    split = load_cnof_corpus_split(
        args.smiles_file,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        max_atoms=args.max_atoms,
        seed=args.seed,
        scan_all=not args.fast_split,
    )
    validation_records = build_tracelet_path_records(
        split.validation,
        n_slots=args.max_atoms,
        typed_ring_payloads=True,
    )
    validation_records = tuple(
        record
        for record in validation_records
        if ring_catalog.supports_trace(record.path.trace)
    )
    fiber_cache = {}
    batch_arguments = dict(
        records=validation_records,
        batch_size=args.examples,
        fiber_cache=fiber_cache,
        late_time_fraction=args.late_time_fraction,
        operational_horizon=args.operational_horizon,
        ring_catalog=ring_catalog,
    )
    sequential_examples = sample_tracelet_conditional_batch(
        **batch_arguments,
        rng=np.random.default_rng(args.seed + 101),
        causal_teachers=False,
    )
    causal_examples = sample_tracelet_conditional_batch(
        **batch_arguments,
        rng=np.random.default_rng(args.seed + 102),
        causal_teachers=True,
        causal_cache={},
    )

    models = {
        "sequential": _load_model(sequential_payload),
        "causal_frontier": _load_model(causal_payload),
    }
    batches = {
        "sequential": sequential_examples,
        "causal_frontier": causal_examples,
    }
    cross_metrics = {
        model_name: {
            teacher_name: tracelet_conditional_metrics(model, examples)
            for teacher_name, examples in batches.items()
        }
        for model_name, model in models.items()
    }
    causal_support_sizes = [
        len(example.teacher_successor_rates) for example in causal_examples
    ]
    report = {
        "seed": args.seed,
        "examples_per_teacher": args.examples,
        "validation_records": len(validation_records),
        "fiber_cache_size": len(fiber_cache),
        "causal_teacher_support": {
            "mean": float(np.mean(causal_support_sizes)),
            "branched_fraction": float(
                np.mean(np.asarray(causal_support_sizes) > 1)
            ),
            "maximum": int(max(causal_support_sizes, default=0)),
        },
        "cross_metrics": cross_metrics,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def _load_model(payload: dict[str, object]) -> TraceletRateModel:
    model = TraceletRateModel(
        hidden_dim=int(payload["hidden_dim"]),
        message_passing_steps=int(payload["message_passing_steps"]),
        rate_factorization=str(payload["rate_factorization"]),
        use_aromatic_bond_view=payload["bond_representation"] == "aromatic",
        ring_catalog=payload["ring_catalog"],
    )
    model.load_state_dict(payload["state_dict"])
    return model.eval()


if __name__ == "__main__":
    main()
