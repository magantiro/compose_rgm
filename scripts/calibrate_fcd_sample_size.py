"""Calibrate finite-sample FCD on real-vs-real corpus splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from fcd_torch import FCD

from compose_v4.data.cnof import load_cnof_corpus_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("smiles_file", type=Path)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--max-atoms", type=int, default=16)
    parser.add_argument("--train-size", type=int, default=1067)
    parser.add_argument("--validation-size", type=int, default=133)
    parser.add_argument("--test-size", type=int, default=133)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=(133, 200, 500, 1000),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/fcd_sample_size_calibration.json"),
    )
    args = parser.parse_args()
    if args.replicates <= 0 or any(size <= 0 for size in args.sample_sizes):
        raise ValueError("replicates and sample sizes must be positive")

    split = load_cnof_corpus_split(
        args.smiles_file,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        max_atoms=args.max_atoms,
        seed=args.seed,
        scan_all=True,
    )
    rng = np.random.default_rng(args.seed)
    evaluator = FCD(device="cpu", n_jobs=1, batch_size=512)
    train = np.asarray(split.train, dtype=object)
    test = list(split.test)
    report: dict[str, object] = {
        "seed": args.seed,
        "train_size": len(split.train),
        "validation_size": len(split.validation),
        "test_size": len(split.test),
        "test_vs_same_test": float(evaluator(test, test)),
        "validation_vs_test": float(evaluator(list(split.validation), test)),
        "full_train_vs_test": float(evaluator(list(train), test)),
        "train_subsamples_vs_test": {},
    }
    subsamples = report["train_subsamples_vs_test"]
    assert isinstance(subsamples, dict)
    for size in args.sample_sizes:
        values = [
            float(
                evaluator(
                    list(rng.choice(train, size=size, replace=size > len(train))),
                    test,
                )
            )
            for _ in range(args.replicates)
        ]
        subsamples[str(size)] = {
            "values": values,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    rendered = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
