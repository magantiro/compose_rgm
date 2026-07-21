#!/usr/bin/env python3
"""E1 unconditional generation metrics for a self-contained CNOF checkpoint.

Samples target-free ancestral rollouts from the base tracelet generator and
reports validity / uniqueness / novelty plus element / atom-count / bond-order /
ring (cycle-rank) marginals against the held-out CNOF reference split -- the
"the generator works" evidence for the paper. FCD-free (corpus_rollout_metrics);
FCD is intentionally NOT part of the unconditional story. The split is rebuilt
deterministically from the ZINC SMILES source with the base's training params
(defaults below), so the reference distribution matches the base's training set.

The smiles_file must be one SMILES per line (whitespace-delimited first field);
extract it from the ZINC CSV first (the raw quoted CSV does not parse).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compose_v4.chem.molecular_graph import molecular_graph_to_smiles
from compose_v4.data.cnof import load_cnof_corpus_split
from compose_v4.eval.molecular_quality import molecular_quality_report
from compose_v4.experiments.calibrated_rewrite_sampling import ThinnedRateCalibrationSampler
from compose_v4.experiments.cnof_conditional import corpus_rollout_metrics
from compose_v4.experiments.parallel_tracelet_sampling import sample_tracelet_ancestral_many


def _load_checkpoint():
    if __package__:
        from scripts.evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint
    else:
        from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint
    return load_factorized_rollout_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("smiles_file", type=Path, help="one SMILES per line (extract from ZINC CSV first)")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260717)
    p.add_argument("--max-atoms", type=int, default=40)
    p.add_argument("--train-size", type=int, default=50000)
    p.add_argument("--validation-size", type=int, default=2000)
    p.add_argument("--test-size", type=int, default=2000)
    p.add_argument("--operational-horizon", type=float, default=16.0)
    p.add_argument("--time-step", type=float, default=0.1)
    p.add_argument("--max-events", type=int, default=128)
    p.add_argument("--corpus-workers", type=int, default=0)
    p.add_argument("--rollout-workers", type=int, default=4)
    args = p.parse_args()

    import torch

    torch.set_num_threads(1)

    load_ckpt = _load_checkpoint()
    model, checkpoint = load_ckpt(str(args.checkpoint))
    model.eval()

    # No rate calibration -- the base is the unconditional generator as-trained
    # (B needs none). Zero adjustments = raw model rates.
    calibrated_sampler = ThinnedRateCalibrationSampler(
        model,
        family_log_rate_adjustments=(("atom_delete", 0.0),),
        small_ring_log_rate_adjustment=0.0,
        triple_bond_log_rate_adjustment=0.0,
    )
    source_prior = checkpoint["tree_source_prior"]
    if max(source_prior.sizes) > args.max_atoms:
        raise SystemExit("checkpoint source prior exceeds --max-atoms")

    split = load_cnof_corpus_split(
        args.smiles_file,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        max_atoms=args.max_atoms,
        seed=args.seed,
        scan_all=True,
        workers=args.corpus_workers,
    )
    print(
        json.dumps({"phase": "split_loaded", "train": len(split.train), "test": len(split.test)}),
        flush=True,
    )

    def _progress(done: int, total: int) -> None:
        if done % 50 == 0 or done == total:
            print(json.dumps({"phase": "sampling", "done": done, "total": total}), flush=True)

    rollouts = sample_tracelet_ancestral_many(
        calibrated_sampler,
        seed=args.seed + 4,
        samples=args.samples,
        workers=args.rollout_workers,
        n_slots=args.max_atoms,
        operational_horizon=args.operational_horizon,
        time_step=args.time_step,
        max_events=args.max_events,
        torch_threads_per_worker=1,
        progress_callback=_progress,
        source_prior=source_prior,
    )

    # Graph-structural diagnostics (supplementary only -- per-marginal TVs are an
    # internal diagnostic, NOT the field's reported currency; kept out of headline).
    graph_diagnostics = corpus_rollout_metrics(
        rollouts, train_smiles=split.train, reference_smiles=split.test
    )

    # Headline de novo report: V/U/N + physchem descriptor Wasserstein distances
    # (the recognized MOSES-style distribution signal). No FCD, no KL suite.
    generated_smiles = tuple(
        s for r in rollouts if (s := molecular_graph_to_smiles(r.final_state)) is not None
    )
    quality = molecular_quality_report(
        generated_smiles,
        reference_smiles=tuple(split.test),
        train_smiles=tuple(split.train),
        include_fcd=False,
    )

    # Plan's E1 sufficiency-gate criteria (paper1_compose_methods.html): pathwise
    # validity, diversity, faithful marginals, coverage, no topology pathology.
    metrics = {
        "samples": args.samples,
        "valid_fraction": graph_diagnostics.get("valid_fraction"),
        "unique_fraction": quality.get("unique_fraction"),
        "novel_to_train_fraction": quality.get("novel_to_train_fraction"),
        "population_diversity": graph_diagnostics.get("population_diversity"),
        "train_support_fraction": graph_diagnostics.get("train_support_fraction"),
        "full_reference_support_fraction": graph_diagnostics.get("full_reference_support_fraction"),
        "connected_fraction": graph_diagnostics.get("connected_or_null_fraction"),
        "descriptor_distributions": quality.get("descriptor_distributions"),
        "ring_systems": quality.get("ring_systems"),
        "operational_horizon": args.operational_horizon,
        "time_step": args.time_step,
        "checkpoint": str(args.checkpoint),
        "graph_structural_diagnostics_supplementary": graph_diagnostics,
    }
    args.output.write_text(json.dumps(metrics, indent=2, default=float))

    descriptor_w = {
        name: round(float(v.get("standardized_wasserstein", v.get("wasserstein_distance", 0.0))), 3)
        for name, v in (quality.get("descriptor_distributions") or {}).items()
    }
    headline = {
        "valid_fraction": metrics["valid_fraction"],
        "unique_fraction": metrics["unique_fraction"],
        "novel_to_train_fraction": metrics["novel_to_train_fraction"],
        "population_diversity": metrics["population_diversity"],
        "coverage_train_support": metrics["train_support_fraction"],
        "coverage_reference_support": metrics["full_reference_support_fraction"],
        "descriptor_standardized_wasserstein": descriptor_w,
    }
    print(json.dumps({"phase": "E1", **headline}, indent=2), flush=True)


if __name__ == "__main__":
    main()
