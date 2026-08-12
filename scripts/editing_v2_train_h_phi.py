"""Train h_phi and produce the nested 2k / 5k / 10k DECISION-ranking learning curve.

    h_phi(y, z, b)  ~  V_G(y, z, b)

y is the state AFTER a candidate action, z the target, b the budget remaining
from y. The controlled process is Markov, so the pre-action state is not needed:
the planner only ever evaluates V_G(y, z, b-1).

ARCHITECTURE -- deliberately the simplest thing that could work
--------------------------------------------------------------
    e_y, e_z  frozen R_theta encoder, shared weights, precomputed
    r = [e_y, e_z, e_y - e_z, e_y * e_z, sim(y,z), budget one-hot]
    -> 2-layer MLP -> recovery logit + terminal-similarity head

The signed difference makes it directional; sim(y,z) is included deliberately
because greedy already has it and the question is what FUTURE information adds
beyond that myopic signal. No cross-attention, no new GNN, no encoder
fine-tuning, no architecture search. If this fails we diagnose why; we do not
reach for capacity first.

LOSSES -- deliberately boring
-----------------------------
Pointwise BCE for recovery, plain MSE for terminal similarity. No pairwise
ranking loss, even though ranking is the evaluation target: if plain value
prediction yields the ranking, the result is stronger and simpler. A ranking
loss is only justified if a well-calibrated model systematically mis-orders
WITHIN states, which is diagnosable rather than assumed.

No class weighting: all-negative states are 53.7% of the data, well under the
preregistered 70% stop condition, so the imbalance does not warrant it.

SUBSETS -- whole decision states, never individual labels
---------------------------------------------------------
The unit being learned is "rank the candidates available at one decision", so a
candidate set is never split to hit a round number. Counts land near 2k/5k/10k
rather than exactly on them.

METRICS -- decision quality, not BCE
------------------------------------
    PRIMARY capability  rescue ranking: among validation states where greedy's
                        action loses recoverability and some candidate keeps it,
                        how often does h_phi pick a candidate that keeps it?
    PRIMARY safety      harmful-override rate: where greedy's action IS
                        recoverable, how often does h_phi pick one that is not?
    supporting          contrastive-state ranking accuracy, lexicographic value
                        regret vs the teacher, top-1 teacher agreement
    diagnostic only     BCE, similarity MAE

Capability and safety are the pair that matters: explicit rollout had a
policy-improvement guarantee, h_phi does not, so measuring only the upside of
overriding would hide the cost the learned approximation introduces.

Ranking is lexicographic on RAW predictions -- recovery score first, terminal
similarity as tiebreak. No probability cutoff is invented here; freezing how
much predicted advantage is required before overriding greedy is the single job
of the 24 development pairs, later.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

BUDGET_MAX = 8


def load_embeddings(directory: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    vectors: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {}
    for path in sorted(directory.rglob("*.npz")):
        payload = np.load(path, allow_pickle=True)
        for key, vector in zip(payload["keys"], payload["vectors"]):
            vectors[str(key)] = vector
        if "canonical_slots" in payload:
            meta["canonical_slots"] = int(payload["canonical_slots"][0])
            meta["time_point"] = float(payload["time_point"][0])
    if not vectors:
        raise SystemExit(f"no embeddings under {directory}")
    return vectors, meta


def decision_states(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Group labels into decision states, keeping each candidate set intact."""

    grouped: dict[tuple[str, int, str], list[dict]] = collections.defaultdict(list)
    for pair in payload["per_pair"]:
        for row in pair["label_rows"]:
            key = (row["decision_state"], int(row["remaining"]),
                   row.get("pair_id", ""))
            grouped[key].append(row)
    states = []
    for (state, remaining, pair_id), rows in grouped.items():
        states.append({"state": state, "remaining": remaining, "pair_id": pair_id,
                       "rows": rows,
                       "kind": rows[0].get("state_kind", "unknown")})
    return states


def featurise(states, vectors, target_of):
    """[e_y, e_z, e_y-e_z, e_y*e_z, sim, budget one-hot] per candidate."""

    features, recovery, similarity, index = [], [], [], []
    # Rows actually featurised, in order. evaluate() must index THESE, not the
    # block's full row list: a candidate skipped for a missing embedding would
    # otherwise shift every local index and silently misidentify greedy's action.
    used: list[dict] = []
    for position, block in enumerate(states):
        target = target_of.get(block["pair_id"])
        if target is None or target not in vectors:
            continue
        e_z = vectors[target]
        for row in block["rows"]:
            e_y = vectors.get(row["candidate"])
            if e_y is None:
                continue
            used.append(row)
            budget = np.zeros(BUDGET_MAX, dtype=np.float32)
            budget[min(int(row["remaining"]), BUDGET_MAX - 1)] = 1.0
            features.append(np.concatenate([
                e_y, e_z, e_y - e_z, e_y * e_z,
                np.array([row["immediate"]], dtype=np.float32), budget]))
            recovery.append(float(row["recovery"]))
            similarity.append(float(row["similarity"]))
            index.append(position)
    return (np.asarray(features, dtype=np.float32),
            np.asarray(recovery, dtype=np.float32),
            np.asarray(similarity, dtype=np.float32),
            np.asarray(index, dtype=np.int64), used)


class HPhi(nn.Module):
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.recovery = nn.Linear(hidden, 1)
        self.similarity = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.recovery(h).squeeze(-1), self.similarity(h).squeeze(-1)


def evaluate(model, features, recovery, similarity, index, used):
    """Decision-quality metrics. Ranking is lexicographic on RAW predictions."""

    model.eval()
    with torch.no_grad():
        logit, predicted_similarity = model(torch.from_numpy(features))
    logit = logit.numpy()
    predicted_similarity = predicted_similarity.numpy()

    rescue_total = rescue_hit = 0
    harmful_total = harmful_hit = 0
    contrastive_total = contrastive_hit = 0
    agree_total = agree_hit = 0
    regret: list[float] = []

    for position in np.unique(index):
        rows = np.flatnonzero(index == position)
        if rows.size < 2:
            continue
        truth = recovery[rows]
        here = [used[r] for r in rows]
        greedy_local = max(range(rows.size),
                           key=lambda i: (here[i]["immediate"], here[i]["candidate"]))
        # Lexicographic: recovery score first, similarity as tiebreak.
        order = sorted(range(rows.size),
                       key=lambda i: (logit[rows][i], predicted_similarity[rows][i]),
                       reverse=True)
        chosen = order[0]

        if truth.max() > 0:
            teacher_best = int(np.argmax(truth))
            agree_total += 1
            agree_hit += int(truth[chosen] == truth[teacher_best])

        if truth.min() < truth.max():
            contrastive_total += 1
            contrastive_hit += int(truth[chosen] > 0)

        greedy_recovers = truth[greedy_local] > 0
        some_recovers = truth.max() > 0
        if (not greedy_recovers) and some_recovers:
            rescue_total += 1
            rescue_hit += int(truth[chosen] > 0)
        if greedy_recovers:
            harmful_total += 1
            harmful_hit += int(truth[chosen] == 0)

        best_value = (truth.max(), similarity[rows].max())
        got = (truth[chosen], similarity[rows][chosen])
        regret.append((best_value[0] - got[0]) + 0.001 * (best_value[1] - got[1]))

    def rate(hit, total):
        return (hit / total) if total else None

    return {
        "rescue_ranking": rate(rescue_hit, rescue_total),
        "rescue_states": rescue_total,
        "harmful_override_rate": rate(harmful_hit, harmful_total),
        "harmful_states": harmful_total,
        "contrastive_ranking": rate(contrastive_hit, contrastive_total),
        "contrastive_states": contrastive_total,
        "teacher_top1_agreement": rate(agree_hit, agree_total),
        "value_regret": float(np.mean(regret)) if regret else None,
        "bce": float(nn.functional.binary_cross_entropy_with_logits(
            torch.from_numpy(logit), torch.from_numpy(recovery))),
        "similarity_mae": float(np.abs(predicted_similarity - similarity).mean()),
    }


def train(features, recovery, similarity, epochs, seed):
    torch.manual_seed(seed)
    model = HPhi(features.shape[1])
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.from_numpy(features)
    y_recovery = torch.from_numpy(recovery)
    y_similarity = torch.from_numpy(similarity)
    n = x.shape[0]
    for _ in range(epochs):
        model.train()
        order = torch.randperm(n)
        for start in range(0, n, 256):
            batch = order[start:start + 256]
            logit, predicted = model(x[batch])
            loss = (nn.functional.binary_cross_entropy_with_logits(
                        logit, y_recovery[batch])
                    + nn.functional.mse_loss(predicted, y_similarity[batch]))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()

    vectors, meta = load_embeddings(args.embeddings)
    print(f"{len(vectors):,} embeddings, dim "
          f"{len(next(iter(vectors.values())))}, {meta}")

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
    print(f"decision states: train {len(train_states):,}  "
          f"validation {len(valid_states):,}")

    vf, vr, vs, vi, vused = featurise(valid_states, vectors, target_of)
    print(f"validation labels featurised: {vf.shape[0]:,}")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(train_states))
    curve = []
    for budget in (2000, 5000, 10000, 10**9):
        subset, count = [], 0
        for position in order:
            if count >= budget:
                break
            subset.append(train_states[position])
            count += len(train_states[position]["rows"])
        tf, tr, ts, ti, _tused = featurise(subset, vectors, target_of)
        if tf.shape[0] == 0:
            continue
        model = train(tf, tr, ts, args.epochs, args.seed)
        metrics = evaluate(model, vf, vr, vs, vi, vused)
        label = "all" if budget > 10**8 else str(budget)
        row = {"requested": label, "states": len(subset), "labels": int(tf.shape[0]),
               **metrics}
        curve.append(row)
        print(f"\n  {label:>5}: {row['states']:,} states / {row['labels']:,} labels")
        print(f"    rescue ranking      {metrics['rescue_ranking']}"
              f"  (n={metrics['rescue_states']})")
        print(f"    harmful override    {metrics['harmful_override_rate']}"
              f"  (n={metrics['harmful_states']})")
        print(f"    contrastive ranking {metrics['contrastive_ranking']}"
              f"  (n={metrics['contrastive_states']})")
        print(f"    teacher top-1       {metrics['teacher_top1_agreement']}")
        print(f"    value regret        {metrics['value_regret']}")
        print(f"    [diagnostic] bce {metrics['bce']:.4f}  "
              f"sim MAE {metrics['similarity_mae']:.4f}")
        if label == "all" and curve and row["labels"] <= curve[-2]["labels"]:
            break

    args.out.write_text(json.dumps({
        "schema": "compose.editing_v2.h_phi_learning_curve",
        "architecture": "frozen R_theta encoder -> [e_y,e_z,e_y-e_z,e_y*e_z,sim,budget] -> 2-layer MLP -> {recovery, similarity}",
        "losses": "pointwise BCE + MSE; no ranking loss, no class weighting",
        "ranking": "lexicographic on raw predictions: recovery then similarity; no probability cutoff",
        "encoder_meta": meta,
        "curve": curve,
    }, indent=2) + "\n")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
