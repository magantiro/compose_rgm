"""Train and FREEZE h_phi for the closed-loop test on the 24 development pairs.

Data size is chosen by the rule agreed before any curve existed: the smallest
subset where decision quality saturates. Measured over four seeds:

    2k    rescue 30.5%  harmful 21.5%  contrastive 52.3%
    5k    rescue 35.2%  harmful 17.2%  contrastive 59.3%
    10k   rescue 42.2%  harmful 13.8%  contrastive 65.9%
    all   rescue 44.5%  harmful 16.0%  contrastive 63.6%

10k is materially better than 5k on every metric; `all` is not better than 10k
(rescue slightly up, harmful slightly worse, both inside seed spread). So 10k.

A FOUR-SEED ENSEMBLE is used because precision dominates recall here -- the
population where greedy is already safe outnumbers the rescue population 4:1,
so variance reduction is worth far more than usual, and four small heads cost
nothing at inference.

TIE-BREAKING. Greedy's action is identified as max by (immediate similarity,
canonical key), matching Experiment C's controller. The collector's
`is_greedy_action` flag uses np.argmax (first index among ties) and disagrees on
3 of 415 validation states; using it would compare against a greedy policy that
was never run.

This script writes weights and the exact feature contract. It selects no
threshold: freezing the override margin is the single job of the 24.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from editing_v2_train_h_phi import (  # noqa: E402
    BUDGET_MAX,
    HPhi,
    decision_states,
    evaluate,
    featurise,
    load_embeddings,
    train,
)

SEEDS = (20260811, 101, 202, 303)
TARGET_LABELS = 10_000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    vectors, meta = load_embeddings(args.embeddings)
    train_payload = json.loads(args.train.read_text())
    valid_payload = json.loads(args.validation.read_text())
    target_of = {}
    for payload in (train_payload, valid_payload):
        for pair in payload["per_pair"]:
            rows = pair["label_rows"]
            if rows:
                target_of[rows[0].get("pair_id", "")] = pair["pair_id"].split(">>")[-1]

    train_states = decision_states(train_payload)
    valid_states = decision_states(valid_payload)
    vf, vr, vs, vi, vused = featurise(valid_states, vectors, target_of)

    # Same nested construction as the curve, same seed, so the frozen model is
    # trained on exactly the subset the 10k row reported.
    rng = np.random.default_rng(SEEDS[0])
    order = rng.permutation(len(train_states))
    subset, count = [], 0
    for position in order:
        if count >= TARGET_LABELS:
            break
        subset.append(train_states[position])
        count += len(train_states[position]["rows"])
    tf, tr, ts, ti, _ = featurise(subset, vectors, target_of)
    print(f"training subset: {len(subset):,} decision states / {tf.shape[0]:,} labels")

    args.out.mkdir(parents=True, exist_ok=True)
    members = []
    for seed in SEEDS:
        model = train(tf, tr, ts, args.epochs, seed)
        metrics = evaluate(model, vf, vr, vs, vi, vused)
        path = args.out / f"h_phi_seed{seed}.pt"
        torch.save({"state_dict": model.state_dict(),
                    "input_dim": int(tf.shape[1]), "hidden": 256}, path)
        members.append({"seed": seed, "path": path.name, **metrics})
        print(f"  seed {seed}: rescue {metrics['rescue_ranking']:.3f}  "
              f"harmful {metrics['harmful_override_rate']:.3f}  "
              f"contrastive {metrics['contrastive_ranking']:.3f}")

    digest = hashlib.sha256()
    for member in members:
        digest.update((args.out / member["path"]).read_bytes())

    contract = {
        "schema": "compose.editing_v2.h_phi_frozen",
        "status": "FROZEN_WEIGHTS_NO_THRESHOLD_SELECTED",
        "ensemble": "mean of the four member recovery logits; similarity likewise",
        "seeds": list(SEEDS),
        "training_labels": int(tf.shape[0]),
        "training_states": len(subset),
        "weights_sha256": digest.hexdigest(),
        "feature_contract": {
            "vector": "[e_y, e_z, e_y - e_z, e_y * e_z, sim(y,z), budget one-hot]",
            "input_dim": int(tf.shape[1]),
            "budget_max": BUDGET_MAX,
            "encoder": "frozen R_theta _encode_batch global_state",
            "canonical_slots": meta.get("canonical_slots"),
            "time_point": meta.get("time_point"),
            "note": ("Inference MUST pad to canonical_slots and use time_point; "
                     "the embedding is not padding-invariant, so a different "
                     "width silently shifts every feature."),
        },
        "greedy_tie_break": ("max by (immediate similarity, canonical key), "
                             "matching Experiment C's controller; the collector's "
                             "is_greedy_action flag uses np.argmax and disagrees "
                             "on 3 of 415 validation states"),
        "members": members,
    }
    (args.out / "H_PHI_FROZEN.json").write_text(
        json.dumps(contract, indent=2) + "\n")
    print(f"\n  weights sha256 {digest.hexdigest()[:16]}")
    print(f"  wrote {args.out}/H_PHI_FROZEN.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
