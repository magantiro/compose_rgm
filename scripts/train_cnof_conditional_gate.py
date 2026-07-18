"""Train and evaluate the first larger conditional-GM C/N/O/F gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from compose_v4.chem.molecular_graph import molecular_graph_to_smiles
from compose_v4.data.cnof import load_cnof_corpus_split
from compose_v4.eval.molecular_quality import molecular_quality_report
from compose_v4.experiments.cnof_conditional import (
    build_path_records,
    conditional_metrics,
    corpus_rollout_metrics,
    fixed_conditional_examples,
    sample_factorized_ancestral,
    train_conditional_model,
)
from compose_v4.experiments.corpus_marginal import fit_corpus_marginal_rate_model
from compose_v4.experiments.prior_tilted import PriorTiltedRewriteRateModel
from compose_v4.model.rate_model import FactorizedRateModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("smiles_file", type=Path)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--max-atoms", type=int, default=12)
    parser.add_argument("--train-size", type=int, default=96)
    parser.add_argument("--validation-size", type=int, default=16)
    parser.add_argument("--test-size", type=int, default=16)
    parser.add_argument(
        "--split-strategy",
        choices=("random", "scaffold"),
        default="random",
    )
    parser.add_argument("--scan-all", action="store_true")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--action-kl-weight", type=float, default=0.0)
    parser.add_argument("--hazard-tilt-weight", type=float, default=0.0)
    parser.add_argument("--validation-samples-per-record", type=int, default=4)
    parser.add_argument("--test-samples-per-record", type=int, default=8)
    parser.add_argument("--evaluation-points", type=int, default=20)
    parser.add_argument("--corpus-prior-tilt", action="store_true")
    parser.add_argument(
        "--defer-bond-orders",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--rewrite-context",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--topology-context",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--late-time-fraction", type=float, default=0.5)
    parser.add_argument("--operational-horizon", type=float, default=7.0)
    parser.add_argument("--rollout-samples", type=int, default=100)
    parser.add_argument("--baseline-samples", type=int, default=0)
    parser.add_argument("--quality-metrics", action="store_true")
    parser.add_argument("--include-fcd", action="store_true")
    parser.add_argument(
        "--evaluation-reference",
        choices=("test", "all"),
        default="test",
    )
    parser.add_argument("--output", type=Path, default=Path("results/cnof_gate.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/cnof_gate.pt"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    split = load_cnof_corpus_split(
        args.smiles_file,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        max_atoms=args.max_atoms,
        seed=args.seed,
        split_strategy=args.split_strategy,
        scan_all=args.scan_all,
    )
    train_records = build_path_records(
        split.train,
        n_slots=args.max_atoms,
        seed=args.seed,
        traces_per_molecule=2,
        defer_bond_orders=args.defer_bond_orders,
    )
    baseline = (
        fit_corpus_marginal_rate_model(train_records)
        if args.corpus_prior_tilt or args.baseline_samples > 0
        else None
    )
    validation_records = build_path_records(
        split.validation,
        n_slots=args.max_atoms,
        seed=args.seed + 1,
        defer_bond_orders=args.defer_bond_orders,
    )
    test_records = build_path_records(
        split.test,
        n_slots=args.max_atoms,
        seed=args.seed + 2,
        defer_bond_orders=args.defer_bond_orders,
    )
    fiber_cache = {}
    validation_examples = fixed_conditional_examples(
        validation_records,
        samples_per_record=args.validation_samples_per_record,
        seed=args.seed + 3,
        fiber_cache=fiber_cache,
        late_time_fraction=args.late_time_fraction,
        operational_horizon=args.operational_horizon,
    )
    test_examples = fixed_conditional_examples(
        test_records,
        samples_per_record=args.test_samples_per_record,
        seed=args.seed + 4,
        fiber_cache=fiber_cache,
        late_time_fraction=args.late_time_fraction,
        operational_horizon=args.operational_horizon,
    )
    if args.corpus_prior_tilt:
        if baseline is None:
            raise RuntimeError("corpus prior was not fitted")
        model = PriorTiltedRewriteRateModel(
            baseline,
            hidden_dim=args.hidden_dim,
            message_passing_steps=3,
            use_rewrite_context=args.rewrite_context,
            use_topology_context=args.topology_context,
        )
        model_type = "prior_tilted_rewrite_generator"
    else:
        model = FactorizedRateModel(
            hidden_dim=args.hidden_dim,
            message_passing_steps=3,
            use_rewrite_context=args.rewrite_context,
            use_topology_context=args.topology_context,
        )
        model_type = "factorized_rewrite_generator"
    initial_validation = conditional_metrics(model, validation_examples)
    history, final_validation = train_conditional_model(
        model,
        train_records,
        validation_examples,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        action_kl_weight=args.action_kl_weight,
        hazard_tilt_weight=args.hazard_tilt_weight,
        seed=args.seed + 5,
        fiber_cache=fiber_cache,
        late_time_fraction=args.late_time_fraction,
        operational_horizon=args.operational_horizon,
        evaluation_points=args.evaluation_points,
    )
    final_test = conditional_metrics(model, test_examples)

    rng = np.random.default_rng(args.seed + 6)
    rollout_fibers = {}
    rollout_rates = {}
    rollouts = tuple(
        sample_factorized_ancestral(
            model,
            rng=rng,
            n_slots=args.max_atoms,
            operational_horizon=args.operational_horizon,
            fiber_cache=rollout_fibers,
            rate_cache=rollout_rates,
        )
        for _ in range(args.rollout_samples)
    )
    all_reference = (*split.train, *split.validation, *split.test)
    evaluation_reference = (
        split.test if args.evaluation_reference == "test" else all_reference
    )
    learned_smiles = _rollout_smiles(rollouts)
    learned_rollout_metrics = corpus_rollout_metrics(
        rollouts,
        train_smiles=split.train,
        reference_smiles=evaluation_reference,
    )
    report = {
        "seed": args.seed,
        "split": {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
            "requested_train": args.train_size,
            "requested_validation": args.validation_size,
            "requested_test": args.test_size,
            "scanned_lines": split.scanned_lines,
            "eligible_molecules": split.eligible_molecules,
            "strategy": split.split_strategy,
            "scaffold_overlap_count": split.scaffold_overlap_count,
            "max_atoms": args.max_atoms,
        },
        "training": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "train_traces": len(train_records),
            "model_type": model_type,
            "defer_bond_orders": args.defer_bond_orders,
            "rewrite_context": args.rewrite_context,
            "topology_context": args.topology_context,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "action_kl_weight": args.action_kl_weight,
            "hazard_tilt_weight": args.hazard_tilt_weight,
            "late_time_fraction": args.late_time_fraction,
            "operational_horizon": args.operational_horizon,
            "validation_examples": len(validation_examples),
            "test_examples": len(test_examples),
            "evaluation_points": args.evaluation_points,
            "history": history,
        },
        "initial_validation": initial_validation,
        "final_validation": final_validation,
        "final_test": final_test,
        "rollout": learned_rollout_metrics,
        "generated_smiles": learned_smiles,
        "sampler": {
            "target_available": False,
            "operational_horizon": args.operational_horizon,
            "time_step": 0.1,
            "max_events": 32,
        },
        "evaluation_reference": args.evaluation_reference,
    }
    if args.quality_metrics:
        report["molecular_quality"] = molecular_quality_report(
            learned_smiles,
            reference_smiles=evaluation_reference,
            train_smiles=split.train,
            include_fcd=args.include_fcd,
            seed=args.seed + 7,
        )
    if args.baseline_samples > 0:
        if baseline is None:
            raise RuntimeError("corpus baseline was not fitted")
        baseline_rng = np.random.default_rng(args.seed + 8)
        baseline_fibers = {}
        baseline_rates = {}
        baseline_rollouts = tuple(
            sample_factorized_ancestral(
                baseline,
                rng=baseline_rng,
                n_slots=args.max_atoms,
                operational_horizon=args.operational_horizon,
                fiber_cache=baseline_fibers,
                rate_cache=baseline_rates,
            )
            for _ in range(args.baseline_samples)
        )
        baseline_smiles = _rollout_smiles(baseline_rollouts)
        report["corpus_marginal_baseline"] = {
            "time_bins": len(baseline.table.times),
            "mean_path_length": baseline.table.mean_path_length,
            "rollout": corpus_rollout_metrics(
                baseline_rollouts,
                train_smiles=split.train,
                reference_smiles=evaluation_reference,
            ),
            "generated_smiles": baseline_smiles,
        }
        if args.quality_metrics:
            report["corpus_marginal_baseline"]["molecular_quality"] = (
                molecular_quality_report(
                    baseline_smiles,
                    reference_smiles=evaluation_reference,
                    train_smiles=split.train,
                    include_fcd=args.include_fcd,
                    seed=args.seed + 9,
                )
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": model_type,
            "hidden_dim": args.hidden_dim,
            "message_passing_steps": 3,
            "use_rewrite_context": args.rewrite_context,
            "use_topology_context": args.topology_context,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "action_kl_weight": args.action_kl_weight,
            "hazard_tilt_weight": args.hazard_tilt_weight,
            "seed": args.seed,
            "split": split,
            "report": report,
            "corpus_marginal_table": baseline.table
            if args.corpus_prior_tilt and baseline is not None
            else None,
        },
        args.checkpoint,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _rollout_smiles(rollouts) -> tuple[str, ...]:
    smiles = []
    for rollout in rollouts:
        text = molecular_graph_to_smiles(rollout.final_state)
        if text is None:
            raise RuntimeError("valid rollout failed SMILES serialization")
        smiles.append(text)
    return tuple(smiles)


if __name__ == "__main__":
    main()
