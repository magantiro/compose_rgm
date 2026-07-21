#!/usr/bin/env python3
"""Reward-FT training driver (Rung 2 / the oracle-LIGHT fair-GrIDDD path).

Trains a policy (init from frozen Lineage B) toward the reward-tilted target via
Relative Trajectory Balance, using lead-conditioned trajectories: each rollout
starts from a TRAINING lead, the terminal reward is the objective (QED) if the
molecule stays in the hard similarity fiber, else a floor. After training, the
policy samples 20 candidates per TEST lead with NO oracle-in-the-loop -- matching
GrIDDD's ~20-candidate, oracle-light protocol (unlike the oracle-hungry SMC).

Split discipline: train on a training-lead set, evaluate on the held-out Jin test
leads. The oracle is used at TRAIN time (disclosed) and only to score the 20 test
candidates at eval -- never during test generation.

CPU-heavy (trajectory sampling); run when the cores are free.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import DataStructs

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.rewrite.kernel import de_novo_rewrite_system


def _imports():
    if __package__:
        from scripts.evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint
        from scripts.griddd_reward_finetune import RewardFineTuner, RTBConfig
        from scripts.griddd_value_guided_smc_controller import _canon, _fp, _load_base_sampler
    else:
        from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint
        from griddd_reward_finetune import RewardFineTuner, RTBConfig
        from griddd_value_guided_smc_controller import _canon, _fp, _load_base_sampler
    return load_factorized_rollout_checkpoint, RewardFineTuner, RTBConfig, _canon, _fp, _load_base_sampler


def sample_mark_trajectory(lead_state, *, policy, qed_oracle, rewrite, lead_fp, fp_fn,
                           similarity_minimum, rng, max_events, time_step, horizon, floor=0.0):
    """Roll out from a lead; record executed marks + terminal reward (feasible QED).

    Samples from the RAW policy model's sample_rewrite_mark so the sampling
    distribution matches the RTB scoring distribution (forward_mark_batch) exactly
    -- required for an unbiased trajectory-balance gradient. (B needs no
    calibration, so the raw model is the correct, consistent sampler.)
    """
    states, times, actions, rules, rates = [], [], [], [], []
    node, t = lead_state, 0.0
    reward = floor
    for _ in range(max_events):
        if t >= horizon:
            break
        mark = policy.sample_rewrite_mark(node, t, rng)
        if mark.action is None:
            break
        try:
            successor = rewrite.apply(node, mark.rule_name, mark.action)
        except Exception:  # noqa: BLE001
            break
        states.append(node); times.append(t); actions.append(mark.action)
        # jump_rates are unused by trajectory_log_prob (it reads
        # selected_mark_log_probability from forward_mark_batch), so a placeholder
        # is safe and avoids depending on the mark object's rate attribute.
        rules.append(mark.rule_name); rates.append(float(getattr(mark, "total_hazard", 0.0)))
        fingerprint = fp_fn(successor)
        if fingerprint is not None and DataStructs.TanimotoSimilarity(lead_fp, fingerprint) >= similarity_minimum:
            reward = max(reward, float(qed_oracle(successor)))
        node, t = successor, t + time_step
    if not states:
        return None
    return {"states": tuple(states), "times": tuple(times), "actions": tuple(actions),
            "rule_names": tuple(rules), "jump_rates": tuple(rates), "reward": reward}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("/private/tmp/lineage_b_checkpoint/checkpoint.best_so_far.pt"))
    p.add_argument("--train-leads", type=Path, required=True, help="JSON [[idx,smiles,val],...] TRAIN split")
    p.add_argument("--output", type=Path, required=True, help="fine-tuned policy .pt")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--trajectories-per-step", type=int, default=8)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--similarity-minimum", type=float, default=0.40)
    p.add_argument("--max-events", type=int, default=32)
    p.add_argument("--time-step", type=float, default=0.1)
    p.add_argument("--horizon", type=float, default=16.0)
    p.add_argument("--n-slots", type=int, default=40)
    p.add_argument("--atom-delete-log-rate-adjustment", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=20260726)
    args = p.parse_args()

    (load_ckpt, RewardFineTuner, RTBConfig, _canon, _fp, _load_base_sampler) = _imports()
    import torch

    # Policy (trainable) + frozen base share the sampler wrapper; policy is updated.
    sampler, qed_oracle = _load_base_sampler(str(args.checkpoint), args.atom_delete_log_rate_adjustment)
    policy = sampler.base_model  # the FactorizedTraceletRateModel inside the sampler
    frozen_base, _ = load_ckpt(str(args.checkpoint)); frozen_base.eval()
    tuner = RewardFineTuner(policy, frozen_base, RTBConfig(alpha=args.alpha, learning_rate=args.learning_rate))
    rewrite = de_novo_rewrite_system()

    leads = json.loads(args.train_leads.read_text())
    states = []
    for _idx, smiles, _v in leads:
        try:
            st = pad_molecular_graph(smiles_to_molecular_graph(smiles), args.n_slots)
            states.append((st, _fp(st)))
        except Exception:  # noqa: BLE001
            continue
    if not states:
        raise SystemExit("no usable training leads")

    rng = np.random.default_rng(args.seed)
    for step in range(args.steps):
        trajectories = []
        for _ in range(args.trajectories_per_step):
            lead_state, lead_fp = states[rng.integers(len(states))]
            if lead_fp is None:
                continue
            tau = sample_mark_trajectory(
                lead_state, policy=policy, qed_oracle=qed_oracle, rewrite=rewrite,
                lead_fp=lead_fp, fp_fn=_fp, similarity_minimum=args.similarity_minimum,
                rng=rng, max_events=args.max_events, time_step=args.time_step, horizon=args.horizon,
            )
            # the sampler's context cache is frozen-B; RTB scores under `policy` (grad).
            if tau is not None:
                trajectories.append(tau)
        if not trajectories:
            continue
        loss = tuner.train_step(trajectories)
        if step % 20 == 0:
            print(json.dumps({"step": step, "rtb_loss": loss, "log_z": float(tuner.log_z),
                              "trajectories": len(trajectories)}), flush=True)

    torch.save({"state_dict": policy.state_dict(), "reward_finetuned": True,
                "alpha": args.alpha}, args.output)
    print(json.dumps({"phase": "done", "output": str(args.output)}))


if __name__ == "__main__":
    main()
