#!/usr/bin/env python3
"""Collect base-CTMC trajectories to train the V1 value twist.

For each lead, runs K ancestral rollouts from Lineage B (zero-calibration) to the
horizon, recording every intermediate as (canonical_smiles, qed, feasible), where
feasible = Tanimoto(lead) >= floor. The output feeds
`griddd_value_twist.py --trajectories`, whose reward-to-go labels train the twist
used by `griddd_value_guided_smc_controller.py --value-twist-checkpoint`.

Kept separate from the SMC controller so label collection (clean, un-resampled
ancestral paths) is decoupled from inference (resampled particle population).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import DataStructs

from compose_v4.chem.state import pad_molecular_graph
from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.model.time_convention import frozen_time
from compose_v4.rewrite.kernel import de_novo_rewrite_system


def _helpers():
    if __package__:
        from scripts.griddd_value_guided_smc_controller import (
            _canon,
            _fp,
            _load_base_sampler,
        )
    else:
        from griddd_value_guided_smc_controller import _canon, _fp, _load_base_sampler
    return _load_base_sampler, _fp, _canon


def rollout_trajectory(
    lead_state,
    *,
    sampler,
    qed_oracle,
    rewrite,
    lead_fp,
    similarity_minimum: float,
    rng: np.random.Generator,
    max_events: int,
    time_step: float,
    horizon: float,
    fp_fn,
    canon_fn,
) -> list[tuple[str, float, bool]]:
    state = lead_state
    time = 0.0
    steps: list[tuple[str, float, bool]] = []
    for _ in range(max_events):
        if time >= horizon:
            break
        mark = sampler.sample_rewrite_mark(state, frozen_time(time), rng)
        if mark.action is None:
            break
        try:
            successor = rewrite.apply(state, mark.rule_name, mark.action)
        except Exception:  # noqa: BLE001
            break
        key = canon_fn(successor)
        fingerprint = fp_fn(successor)
        if key is None or fingerprint is None:
            break
        feasible = (
            DataStructs.TanimotoSimilarity(lead_fp, fingerprint) >= similarity_minimum
        )
        steps.append((key, float(qed_oracle(successor)), bool(feasible)))
        state = successor
        time += time_step
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/private/tmp/lineage_b_checkpoint/checkpoint.best_so_far.pt"),
    )
    parser.add_argument("--leads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-leads", type=int, default=12)
    parser.add_argument("--rollouts-per-lead", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--similarity-minimum", type=float, default=0.40)
    parser.add_argument("--max-events", type=int, default=48)
    parser.add_argument("--time-step", type=float, default=0.1)
    parser.add_argument("--horizon", type=float, default=16.0)
    parser.add_argument("--n-slots", type=int, default=40)
    parser.add_argument("--atom-delete-log-rate-adjustment", type=float, default=0.0)
    args = parser.parse_args()

    load_base_sampler, fp_fn, canon_fn = _helpers()
    sampler, qed_oracle = load_base_sampler(
        str(args.checkpoint), args.atom_delete_log_rate_adjustment
    )
    rewrite = de_novo_rewrite_system()
    leads = json.loads(args.leads.read_text())[: args.max_leads]

    trajectories: list[list[tuple[str, float, bool]]] = []
    for index, smiles, _ in leads:
        try:
            state = pad_molecular_graph(smiles_to_molecular_graph(smiles), args.n_slots)
        except Exception:  # noqa: BLE001
            continue
        lead_fp = fp_fn(state)
        if lead_fp is None:
            continue
        for r in range(args.rollouts_per_lead):
            rng = np.random.default_rng(args.seed + 1000 * int(index) + r)
            steps = rollout_trajectory(
                state,
                sampler=sampler,
                qed_oracle=qed_oracle,
                rewrite=rewrite,
                lead_fp=lead_fp,
                similarity_minimum=args.similarity_minimum,
                rng=rng,
                max_events=args.max_events,
                time_step=args.time_step,
                horizon=args.horizon,
                fp_fn=fp_fn,
                canon_fn=canon_fn,
            )
            if steps:
                trajectories.append(steps)
        print(f"lead {index}: {sum(len(t) for t in trajectories)} states so far", flush=True)

    args.output.write_text(json.dumps(trajectories) + "\n")
    print(
        json.dumps(
            {
                "trajectories": len(trajectories),
                "states": sum(len(t) for t in trajectories),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
