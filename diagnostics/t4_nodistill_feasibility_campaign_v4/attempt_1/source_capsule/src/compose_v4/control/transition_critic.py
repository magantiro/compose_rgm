"""A target-conditioned critic over whole transitions, not over local edit coordinates.

The proposal law fitted on `(site, mode)` coordinates is value-blind, and the measurement
is unambiguous: on held-out JAK2 it ranks top-decile constructions at median 156 and
bottom-decile at 11; on PARP1, 329 against 30. Weighting the likelihood by realized
docking quality does not repair it at any strength, and fitting on elite decisions alone
still leaves elite at 88 against poor at 35. Where you attach and how many atoms you add
simply do not determine whether a molecule is good.

So the object here is different in two ways. It scores the whole TRANSITION -- source
molecule, the executable program, and the RESULTING molecule -- rather than the
coordinates of an edit. And it is trained discriminatively against the alternatives that
were actually available, rather than by imitation, because imitating an archive whose
typical member is mediocre reproduces mediocrity faithfully.

Supervision is SIBLING-MATCHED. Comparing a child of a good parent against a child of a
bad one teaches the critic to recognise good parents, which it can already read off the
source score and which is useless for choosing among the moves available right now. Two
children of the SAME parent had the same opportunity, so a preference between them is a
statement about the transformation.

The model is Bradley-Terry with a linear score, so `P(a beats b) = sigmoid(w.(x_a - x_b))`
is convex in `w`: one optimum, no seed, no restarts.
"""

from __future__ import annotations

import numpy as np

SCHEMA_VERSION = "transition_critic_v1"


def pairwise_design(groups, features):
    """Every ordered within-group pair, as a difference vector and a label.

    `groups` maps a group key to the row indices sharing a parent; a pair is emitted only
    when the two members actually differ in outcome, since a tie carries no preference.
    """
    rows, labels = [], []
    for members, outcomes in groups:
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if outcomes[i] == outcomes[j]:
                    continue
                better, worse = (i, j) if outcomes[i] > outcomes[j] else (j, i)
                rows.append(features[members[better]] - features[members[worse]])
                labels.append(1.0)
                rows.append(features[members[worse]] - features[members[better]])
                labels.append(0.0)
    if not rows:
        return np.zeros((0, features.shape[1])), np.zeros(0)
    return np.asarray(rows), np.asarray(labels)


def fit_bradley_terry(design, labels, *, penalty: float = 1e-2, maxiter: int = 500):
    """Convex logistic on difference vectors. No intercept: a constant cancels in a difference."""
    from scipy.optimize import minimize

    if design.shape[0] == 0:
        raise ValueError("cannot fit a critic without comparable pairs")

    def objective(w):
        z = design @ w
        loss = float(np.mean(np.logaddexp(0.0, z) - labels * z)) + penalty * float(w @ w)
        probability = 1.0 / (1.0 + np.exp(-z))
        return loss, design.T @ (probability - labels) / len(labels) + 2 * penalty * w

    result = minimize(
        objective,
        np.zeros(design.shape[1]),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter},
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "weights": result.x,
        "penalty": penalty,
        "pairs": int(design.shape[0]),
        "converged": bool(result.success),
        "objective": float(result.fun),
    }


def concordance(model, groups, features) -> dict:
    """Within-group agreement: of the pairs that differ, how many does the critic order right?

    Reported per group and averaged, so a single large sibling group cannot carry the
    number, and always alongside the count of comparable pairs.
    """
    weights = model["weights"]
    correct = total = 0
    per_group = []
    for members, outcomes in groups:
        scores = features[members] @ weights
        hits = pairs = 0
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if outcomes[i] == outcomes[j]:
                    continue
                pairs += 1
                ordered = (scores[i] > scores[j]) == (outcomes[i] > outcomes[j])
                tied = scores[i] == scores[j]
                hits += 0.5 if tied else float(ordered)
        if pairs:
            per_group.append(hits / pairs)
            correct += hits
            total += pairs
    return {
        "pairs": total,
        "groups": len(per_group),
        "pooled": correct / total if total else float("nan"),
        "mean_per_group": float(np.mean(per_group)) if per_group else float("nan"),
    }


def top_k_of_best(model, groups, features, k_values=(1, 3, 5)) -> dict:
    """How often the critic's favourite sibling is the one that actually scored best.

    This is the decision the controller makes: among the moves available from here, pick
    one. A critic that orders pairs well but never puts the winner first does not help.
    """
    weights = model["weights"]
    hits = {k: [] for k in k_values}
    for members, outcomes in groups:
        if len(members) < 2:
            continue
        scores = features[members] @ weights
        order = np.argsort(-scores)
        best = int(np.argmax(outcomes))
        for k in k_values:
            hits[k].append(float(best in order[:k]))
    return {f"top{k}": float(np.mean(v)) if v else float("nan") for k, v in hits.items()} | {
        "groups": len(hits[k_values[0]])
    }
