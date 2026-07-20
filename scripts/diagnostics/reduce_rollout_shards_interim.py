#!/usr/bin/env python3
"""Reduce every currently durable rollout shard into an interim metrics payload.

This reducer intentionally accepts an incomplete shard set.  Its output is
therefore descriptive and non-promotional; the declared shard count and every
missing index are recorded so it cannot be mistaken for the final evaluation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from compose_v4.chem.molecular_graph import molecular_graph_to_smiles
from compose_v4.data.cnof import _canonical_cnof_smiles
from compose_v4.eval.ring_taxonomy import ring_taxonomy_report
from compose_v4.experiments.cnof_conditional import corpus_rollout_metrics


def _read_matched_reference(path: Path, *, max_atoms: int = 40) -> tuple[str, ...]:
    accepted: dict[str, None] = {}
    for line in path.read_text().splitlines():
        fields = line.strip().split()
        if not fields:
            continue
        canonical = _canonical_cnof_smiles((fields[0], max_atoms))
        if canonical is not None:
            accepted.setdefault(canonical, None)
    if not accepted:
        raise ValueError("matched reference is empty")
    return tuple(accepted)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_directory", type=Path)
    parser.add_argument("reference_smiles", type=Path)
    parser.add_argument("--declared-shards", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = tuple(sorted(args.shard_directory.rglob("shard-*.pt")))
    if not paths:
        raise FileNotFoundError(f"no rollout shards under {args.shard_directory}")

    indexed: dict[int, tuple[object, ...]] = {}
    source_run_label: str | None = None
    checkpoint_name: str | None = None
    checkpoint_sha256: str | None = None
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"shard payload is not a dictionary: {path}")
        index = int(payload["shard_index"])
        if index in indexed:
            raise ValueError(f"duplicate shard index {index}")
        rollouts = payload.get("rollouts")
        if not isinstance(rollouts, (tuple, list)) or not rollouts:
            raise ValueError(f"shard has no rollouts: {path}")
        indexed[index] = tuple(rollouts)
        source_run_label = str(payload.get("source_run_label"))
        checkpoint_name = str(payload.get("checkpoint_name"))
        checkpoint_sha256 = str(payload.get("checkpoint_sha256"))

    completed_indices = tuple(sorted(indexed))
    missing_indices = tuple(
        index
        for index in range(args.declared_shards)
        if index not in indexed
    )
    rollouts = tuple(
        rollout
        for index in completed_indices
        for rollout in indexed[index]
    )
    generated_smiles = tuple(
        text
        for rollout in rollouts
        if (text := molecular_graph_to_smiles(rollout.final_state)) is not None
    )
    reference = _read_matched_reference(args.reference_smiles)
    event_counts = Counter(
        rule
        for rollout in rollouts
        for rule in rollout.event_rules
    )
    report: dict[str, object] = {
        "format": "compose_v4_incomplete_rollout_shard_reduction_v1",
        "decision_use": "posthoc_nonpromotional_incomplete_diagnostic",
        "promotion_authorized": False,
        "incomplete": bool(missing_indices),
        "declared_shards": args.declared_shards,
        "completed_shards": len(completed_indices),
        "completed_shard_indices": completed_indices,
        "missing_shard_indices": missing_indices,
        "source_run_label": source_run_label,
        "checkpoint_name": checkpoint_name,
        "checkpoint_sha256": checkpoint_sha256,
        "requested_samples": args.declared_shards,
        "merged_attempts": len(rollouts),
        "generated_nonnull_smiles": len(generated_smiles),
        "generated_smiles": generated_smiles,
        "rollout_event_counts": dict(sorted(event_counts.items())),
        "rollout": corpus_rollout_metrics(
            rollouts,
            train_smiles=(),
            reference_smiles=reference,
        ),
        "generated_ring_taxonomy": (
            ring_taxonomy_report(generated_smiles) if generated_smiles else None
        ),
        "limitations": {
            "missing_shards": len(missing_indices),
            "reference_kind": "matched neutral CNOF max40 heldout",
            "train_support_metrics_interpretable": False,
            "quality_fcd_computed": False,
            "note": (
                "This is an interim reduction over durably completed shards. "
                "Shard completion can be chemistry/runtime dependent, so the "
                "subset may be biased toward faster trajectories."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "completed_shards": len(completed_indices),
                "missing_shards": len(missing_indices),
                "merged_attempts": len(rollouts),
                "generated_nonnull_smiles": len(generated_smiles),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
