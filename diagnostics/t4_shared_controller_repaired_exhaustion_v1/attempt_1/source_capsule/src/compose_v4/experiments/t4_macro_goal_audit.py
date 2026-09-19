"""How many distinct structural intents are there, and are they predictable?

The proposal law answers "which `(site, mode)` should I choose" and answers it well
enough to narrow 1,850 candidate pairs to a median rank of 31. That is a local question.
The route decomposition already said the global one is much smaller: a fifteen-primitive
teacher route is typically only a handful of coherent dependency regions, so a long
trajectory of micro-decisions corresponds to very few actual structural changes.

This module tests whether that smaller question is learnable, by treating each
constructive decision as a MACRO GOAL -- a canonical molecule-level delta independent of
the order the primitives happened to be emitted in -- and asking three things:

- how many distinct macro goals exist at each granularity;
- how often the same macro goal recurs across targets, which is what makes it a
  transferable intent rather than a per-protein accident;
- whether the correct macro goal is predictable from the SOURCE MOLECULE alone, under
  leave-one-target-out splits.

If the third holds, the controller decomposes as `p(g | G) q(a | G, g)`: choose the
structural intent, then let the proposal law realize it exactly. If it does not, the
hierarchy buys nothing and the flat law is the whole story.

Reads the frozen corpus. No compiler calls and no docking.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from compose_v4.chem.molecular_graph import IDX_TO_ELEMENT
from compose_v4.control.constructive_features import _adjacency, _ring_slots, occupied_slots

SCHEMA_VERSION = "t4_macro_goal_audit_v1"

MOLECULE_FEATURE_NAMES = (
    "heavy_atoms",
    "ring_atoms",
    "ring_fraction",
    "heteroatom_fraction",
    "nitrogen_fraction",
    "oxygen_fraction",
    "mean_degree",
    "terminal_fraction",
    "unsaturated_fraction",
    "free_valence_per_atom",
    "bias",
)


def bucket_created(created: int) -> str:
    if created == 0:
        return "0"
    if created == 1:
        return "1"
    if created <= 3:
        return "2-3"
    if created <= 6:
        return "4-6"
    return "7+"


def macro_goal(decision: dict, *, granularity: str) -> tuple:
    """A structural intent, independent of the order primitives were emitted.

    coarse  -- what KIND of change: grow, close a ring, open one, or rearrange.
    medium  -- adds how much, and using how many attachment points.
    fine    -- medium plus the realized program size, which distinguishes a compact
               decoration from an extended construction reaching the same atom count.
    """
    mode = decision["mode"]
    created, closes, opens = mode["created_atoms"], mode["closes_ring"], mode["opens_ring"]
    if granularity == "coarse":
        kind = "grow" if created else ("rearrange" if not (closes or opens) else "cyclize")
        return (kind, bool(closes), bool(opens))
    if granularity == "medium":
        return (bucket_created(created), min(int(mode["attachment_count"]), 3), bool(closes))
    if granularity == "fine":
        span = (
            "compact"
            if mode["primitive_count"] <= 4
            else ("extended" if mode["primitive_count"] <= 12 else "large")
        )
        return (
            bucket_created(created),
            min(int(mode["attachment_count"]), 3),
            bool(closes),
            bool(opens),
            span,
        )
    raise ValueError(f"unknown granularity {granularity!r}")


def molecule_features(state) -> np.ndarray:
    """Whole-molecule descriptors. Deliberately global: a macro goal is about the
    molecule, not about one atom, so nothing here is site-specific."""
    types = np.asarray(state["atom_types"])
    hydrogens = np.asarray(state["implicit_h_counts"])
    neighbours = _adjacency(state)
    rings = _ring_slots(state, neighbours)
    slots = occupied_slots(state)
    if not slots:
        return np.zeros(len(MOLECULE_FEATURE_NAMES))
    elements = [IDX_TO_ELEMENT.get(int(types[s]), "other") for s in slots]
    degrees = np.asarray([len(neighbours.get(s, [])) for s in slots], dtype=float)
    unsaturated = sum(1 for s in slots if any(order > 1 for _, order in neighbours.get(s, [])))
    n = len(slots)
    return np.asarray(
        [
            n / 40.0,
            len(rings) / 40.0,
            len(rings) / n,
            sum(1 for e in elements if e not in ("C", "null")) / n,
            elements.count("N") / n,
            elements.count("O") / n,
            degrees.mean() / 4.0,
            float((degrees <= 1).mean()),
            unsaturated / n,
            float(np.mean([hydrogens[s] for s in slots])) / 3.0,
            1.0,
        ],
        dtype=float,
    )


def census(decisions, *, granularity: str) -> dict:
    """How many macro goals exist, and how much mass sits on ones that recur."""
    goals = [macro_goal(d, granularity=granularity) for d in decisions]
    targets_of: dict[tuple, set] = {}
    for goal, decision in zip(goals, decisions):
        targets_of.setdefault(goal, set()).add(decision["target"])
    counts = Counter(goals)
    n_targets = len({d["target"] for d in decisions})
    shared = {
        k: sum(counts[g] for g, t in targets_of.items() if len(t) >= k) / len(goals)
        for k in range(2, n_targets + 1)
    }
    return {
        "granularity": granularity,
        "decisions": len(goals),
        "distinct_goals": len(counts),
        "goals_in_every_target": sum(1 for t in targets_of.values() if len(t) == n_targets),
        "mass_on_goals_shared_by_at_least": shared,
        "top": [
            {"goal": list(goal), "n": n, "targets": sorted(targets_of[goal])}
            for goal, n in counts.most_common(10)
        ],
    }


def predictability(decisions, *, granularity: str, penalty: float = 1e-2) -> dict:
    """Can the source molecule alone say which macro goal comes next?

    Multinomial logistic over the macro-goal menu, leave-one-target-out, graded against
    the two baselines that cost nothing: guessing uniformly, and always guessing the
    most frequent goal in the training folds. Beating uniform is trivial and beating the
    majority baseline is not -- a skewed goal distribution makes the majority rule
    strong, and a model that cannot beat it has learned the marginal, not the molecule.
    """
    from scipy.optimize import minimize

    goals = [macro_goal(d, granularity=granularity) for d in decisions]
    vocabulary = sorted(set(goals))
    index = {g: i for i, g in enumerate(vocabulary)}
    labels = np.asarray([index[g] for g in goals])
    features = np.asarray([molecule_features(d["state"]) for d in decisions])
    targets = np.asarray([d["target"] for d in decisions])
    n_classes, n_features = len(vocabulary), features.shape[1]

    def fit(design, y):
        def objective(flat):
            w = flat.reshape(n_classes, n_features)
            scores = design @ w.T
            shift = scores.max(axis=1, keepdims=True)
            exponent = np.exp(scores - shift)
            total = exponent.sum(axis=1, keepdims=True)
            loss = float(
                np.mean(shift.ravel() + np.log(total.ravel()) - scores[np.arange(len(y)), y])
            )
            probability = exponent / total
            probability[np.arange(len(y)), y] -= 1.0
            grad = probability.T @ design / len(y)
            reg = w.copy()
            reg[:, -1] = 0.0
            return loss + penalty * float((reg * reg).sum()), (grad + 2 * penalty * reg).ravel()

        result = minimize(
            objective,
            np.zeros(n_classes * n_features),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 400},
        )
        return result.x.reshape(n_classes, n_features)

    fitted, majority, uniform, sizes = [], [], [], []
    for target in sorted(set(targets)):
        held = targets == target
        if held.sum() < 30:
            continue
        mean, scale = features[~held].mean(axis=0), features[~held].std(axis=0)
        # A constant column is the intercept. Centring it sends it to zero and leaves the
        # model with no bias, so it cannot even represent the marginal over goals -- which
        # is exactly how this first read produced top-1 0.115 against a 0.351 majority
        # baseline, a result no correctly fitted model can give.
        constant = scale < 1e-12
        mean = np.where(constant, 0.0, mean)
        scale = np.where(constant, 1.0, scale)
        w = fit((features[~held] - mean) / scale, labels[~held])
        scores = ((features[held] - mean) / scale) @ w.T
        order = np.argsort(-scores, axis=1)
        truth = labels[held]
        fitted.append(
            {
                k: float(np.mean([(order[i, :k] == truth[i]).any() for i in range(len(truth))]))
                for k in (1, 3, 5)
            }
        )
        common = Counter(labels[~held]).most_common(5)
        ranked = [c for c, _ in common]
        majority.append({k: float(np.mean([t in ranked[:k] for t in truth])) for k in (1, 3, 5)})
        uniform.append({k: k / n_classes for k in (1, 3, 5)})
        sizes.append(int(held.sum()))

    if not sizes:
        return {"granularity": granularity, "gradeable": False}
    weight = np.asarray(sizes, float) / sum(sizes)

    def blend(arms):
        return {f"top{k}": float(np.dot(weight, [a[k] for a in arms])) for k in (1, 3, 5)}

    return {
        "granularity": granularity,
        "gradeable": True,
        "classes": n_classes,
        "decisions": int(sum(sizes)),
        "from_source_molecule": blend(fitted),
        "most_frequent_goal": blend(majority),
        "uniform": blend(uniform),
        "new_oracle_calls": 0,
    }
