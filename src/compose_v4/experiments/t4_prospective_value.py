"""Prospective labels and the honest baseline ladder for the future-value model.

The question this answers is narrow and falsifiable: from information available at the
moment a molecule was scored, can anything predict which lineage produces the next
improvement better than the signals a controller already has for free?

Those free signals are the baseline ladder, and they are strong. "The best molecule so
far is the best place to spend the next call" is a real policy, and the soft allocator
measured earlier on this branch already beats uniform by 5.5 to 14x. A future-value
model has to beat THOSE, not uniform.

The first label tried here -- best improvement over ALL descendants within `k` charged
calls -- is contaminated, and the contamination is worth stating because it is not
obvious. A max over descendants grows with the NUMBER of descendants, and the historical
search chose how many to generate. "Number of descendants actually generated" scores AUC
0.851 on that label by itself, which is not prediction, it is a readout of where the
behaviour policy spent its calls. Stratifying by it collapsed the soft allocator from
0.693 to 0.556-0.579, so roughly two thirds of that baseline's apparent skill was
sampling rather than chemistry.

`best_of_first_descendants` is the repair: the best improvement among the first `m`
children a state received, over states that received at least `m`. Fixed `m` makes the
label invariant to how generous the historical policy was with that lineage, and it is
also the question a budget actually poses -- spend `m` calls below this state, what do
you get?

Leakage is the whole risk here, so three rules are enforced structurally rather than
remembered. A descendant counts only if its charged-call index is strictly greater.
Every feature is computed from the prefix of the run that precedes the state. And folds
are cut by TARGET, so no protein's own runs inform its own score -- the earlier
hindsight allocator scored 26 to 206x precisely because post-champion decisions inherit
the champion's value, which is not a result, it is a definition.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from compose_v4.control.prospective_value import (
    HORIZONS,
    capture_at_fraction,
    fit_logistic,
    roc_auc,
    spearman,
    standardize,
)

SCHEMA_VERSION = "t4_prospective_value_v1"

FEATURE_NAMES = (
    "score",
    "improvement_over_parent",
    "gap_to_incumbent",
    "ancestral_primitives",
    "lineage_depth",
    "primitive_count",
    "changed_slot_count",
    "net_created",
    "ring_count_change",
    "input_endpoint_similarity",
    "parent_probability",
    "call_fraction",
    "siblings_scored_so_far",
    "lineage_scored_so_far",
    "lineage_best_improvement_so_far",
    "bias",
)

# What a controller already has without fitting anything. Each is a single column used
# directly as a ranking score; sign is chosen so that larger means "more promising".
BASELINES = {
    "current score": lambda f: -f["score"],
    "improvement over parent": lambda f: f["improvement_over_parent"],
    "lineage depth": lambda f: f["lineage_depth"],
    "ancestral edits": lambda f: f["ancestral_primitives"],
    "lineage improvement rate": lambda f: f["lineage_best_improvement_so_far"],
    "soft allocator weight": lambda f: f["parent_probability"],
}


def read_rows(path: Path) -> list[dict]:
    with gzip.open(Path(path), "rt") as handle:
        return [json.loads(line) for line in handle]


def indexed_runs(rows) -> list[dict]:
    """Only runs that recorded a charged-call index can carry a horizon label."""
    return [row for row in rows if row.get("query") is not None]


def descendants_within(rows) -> dict[str, dict[int, float]]:
    """For each state, the best improvement observed below it at each horizon.

    Walks each run's genealogy once, pushing every scored entry's improvement up to all
    of its ancestors, gated on the ancestor's own call index. Strictly forward: an
    entry whose call index does not exceed its ancestor's contributes nothing.
    """
    by_run = defaultdict(dict)
    for row in rows:
        by_run[row["run"]][row["entry_id"]] = row

    labels: dict[str, dict[int, float]] = {}
    for entries in by_run.values():
        for row in entries.values():
            labels[row["entry_id"]] = dict.fromkeys(HORIZONS, 0.0)
        for row in entries.values():
            # climb to the root, crediting every ancestor this entry improved on
            seen, node = set(), row
            while True:
                parent_id = node.get("genealogical_parent")
                if not parent_id or parent_id in seen or parent_id not in entries:
                    break
                seen.add(parent_id)
                parent = entries[parent_id]
                node = parent
                if parent.get("query") is None or row["query"] <= parent["query"]:
                    continue
                distance = row["query"] - parent["query"]
                gain = parent["score"] - row["score"]
                if gain <= 0:
                    continue
                bucket = labels[parent_id]
                for horizon in HORIZONS:
                    if distance <= horizon and gain > bucket[horizon]:
                        bucket[horizon] = gain
    return labels


def best_of_first_descendants(rows, *, width: int) -> dict[str, float | None]:
    """Best improvement among the first `width` descendants, or None if there were fewer.

    Budget-invariant by construction: every labelled state is credited with exactly
    `width` children, so a lineage the historical policy expanded fifty times cannot
    out-score one it expanded five times purely on draws. States with fewer than `width`
    descendants are labelled None and dropped rather than filled with a zero, which would
    reintroduce the same bias with the opposite sign.
    """
    by_run = defaultdict(dict)
    for row in rows:
        by_run[row["run"]][row["entry_id"]] = row

    children: dict[str, list[dict]] = defaultdict(list)
    for entries in by_run.values():
        for row in entries.values():
            node, seen = row, set()
            while True:
                parent_id = node.get("genealogical_parent")
                if not parent_id or parent_id in seen or parent_id not in entries:
                    break
                seen.add(parent_id)
                node = entries[parent_id]
                if node.get("query") is not None and row["query"] > node["query"]:
                    children[parent_id].append(row)

    labels: dict[str, float | None] = {}
    for row in rows:
        brood = sorted(children.get(row["entry_id"], []), key=lambda r: r["query"])
        if len(brood) < width:
            labels[row["entry_id"]] = None
            continue
        gains = [row["score"] - child["score"] for child in brood[:width]]
        labels[row["entry_id"]] = max(0.0, max(gains))
    return labels


def _lineage_chain(row, entries) -> list[dict]:
    chain, seen, node = [], set(), row
    while True:
        parent_id = node.get("genealogical_parent")
        if not parent_id or parent_id in seen or parent_id not in entries:
            break
        seen.add(parent_id)
        node = entries[parent_id]
        chain.append(node)
    return chain


def featurize(rows) -> tuple[np.ndarray, list[dict]]:
    """Features available at each state's own charged call, and nothing later."""
    by_run = defaultdict(dict)
    for row in rows:
        by_run[row["run"]][row["entry_id"]] = row

    matrix, kept = [], []
    for entries in by_run.values():
        ordered = sorted(entries.values(), key=lambda r: r["query"])
        budget = max(r["query"] for r in ordered)
        incumbent = float("inf")
        sibling_counts: dict[str, int] = defaultdict(int)
        lineage_counts: dict[str, int] = defaultdict(int)
        lineage_best: dict[str, float] = defaultdict(float)
        for row in ordered:
            chain = _lineage_chain(row, entries)
            parent_id = row.get("genealogical_parent") or ""
            root = chain[-1]["entry_id"] if chain else row["entry_id"]
            parent_score = row.get("parent_score")
            improvement = 0.0 if parent_score is None else parent_score - row["score"]
            matrix.append(
                [
                    row["score"],
                    improvement,
                    0.0 if incumbent == float("inf") else row["score"] - incumbent,
                    row.get("ancestral_primitives") or 0,
                    len(chain),
                    row.get("primitive_count") or 0,
                    row.get("changed_slot_count") or 0,
                    row.get("net_created") or 0,
                    row.get("ring_count_change") or 0,
                    row.get("input_endpoint_similarity") or 0.0,
                    row.get("parent_probability") or 0.0,
                    row["query"] / max(budget, 1),
                    sibling_counts[parent_id],
                    lineage_counts[root],
                    lineage_best[root],
                    1.0,
                ]
            )
            kept.append(row)
            # update the prefix state only AFTER this row is featurized
            sibling_counts[parent_id] += 1
            lineage_counts[root] += 1
            lineage_best[root] = max(lineage_best[root], improvement)
            incumbent = min(incumbent, row["score"])
    return np.asarray(matrix, dtype=float), kept


def evaluate(rows, *, horizon=None, width=None, penalties=(1e-1, 1e-2, 1e-3, 1e-4)) -> dict:
    """Leave-one-target-out grading of the model against the free baselines.

    Pass `width` for the budget-invariant label and `horizon` for the call-window one.
    The ridge is chosen inside the training folds by a further leave-one-target-out
    split, never on the held-out target, so the reported number is not a sweep maximum.
    """
    if (horizon is None) == (width is None):
        raise ValueError("choose exactly one label: a horizon in calls or a brood width")
    features, kept = featurize(rows)
    if width is not None:
        raw = best_of_first_descendants(kept, width=width)
        usable = np.asarray([raw[r["entry_id"]] is not None for r in kept])
        features, kept = features[usable], [r for r, u in zip(kept, usable) if u]
        magnitude = np.asarray([raw[r["entry_id"]] for r in kept], dtype=float)
    else:
        labels = descendants_within(kept)
        magnitude = np.asarray([labels[r["entry_id"]][horizon] for r in kept], dtype=float)
    improved = (magnitude > 0).astype(int)
    targets = np.asarray([r["target"] for r in kept])
    index = {name: i for i, name in enumerate(FEATURE_NAMES)}

    arms = {name: [] for name in (*BASELINES, "h_phi")}
    truth, realized, chosen_penalties = [], [], []
    for target in sorted(set(targets)):
        held = targets == target
        if held.sum() < 30 or improved[~held].sum() < 10:
            continue
        inner_targets = targets[~held]
        best, best_auc = penalties[0], -np.inf
        for candidate in penalties:
            inner = []
            for validation in sorted(set(inner_targets)):
                mask = inner_targets == validation
                if mask.sum() < 30 or improved[~held][~mask].sum() < 10:
                    continue
                a, b = standardize(features[~held][~mask], features[~held][mask])
                fitted = fit_logistic(a, improved[~held][~mask], penalty=candidate)
                inner.append(roc_auc(fitted.score(b), improved[~held][mask]))
            if inner and np.mean(inner) > best_auc:
                best, best_auc = candidate, float(np.mean(inner))
        train, test = standardize(features[~held], features[held])
        model = fit_logistic(train, improved[~held], penalty=best)
        arms["h_phi"].append(model.score(test))
        chosen_penalties.append(best)
        columns = {name: features[held][:, index[name]] for name in index}
        for name, rule in BASELINES.items():
            arms[name].append(rule(columns))
        truth.append(improved[held])
        realized.append(magnitude[held])

    if not truth:
        return {"horizon": horizon, "width": width, "gradeable": False}

    truth_all = np.concatenate(truth)
    realized_all = np.concatenate(realized)
    report = {}
    for name, pieces in arms.items():
        scores = np.concatenate(pieces)
        report[name] = {
            "auc": roc_auc(scores, truth_all),
            "spearman_vs_magnitude": spearman(scores, realized_all),
            "capture_top10pct": capture_at_fraction(scores, realized_all, 0.1),
        }
    return {
        "horizon": horizon,
        "width": width,
        "ridge_chosen_per_fold": chosen_penalties,
        "gradeable": True,
        "states": len(truth_all),
        "states_that_improved": int(truth_all.sum()),
        "base_rate": float(truth_all.mean()),
        "arms": report,
        "label_definition": (
            f"best observed improvement among the first {width} descendants"
            if width is not None
            else f"best observed descendant improvement within {horizon} charged calls "
            "(NOT budget-invariant: grows with how many descendants the policy generated)"
        ),
        "under_behaviour_policy": (
            "labels reflect what the historical search found, not what was reachable; "
            "this is a behaviour-policy future value, not a committor"
        ),
        "new_oracle_calls": 0,
    }
