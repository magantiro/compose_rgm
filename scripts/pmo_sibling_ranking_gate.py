"""Can a contextual model rank MACROS from the same parent by their actual PMO outcome?

This is the first hard gate on the contextual controller.  Within a sibling group the parent and
the search stage are fixed by construction, so a contrast between siblings identifies the macro's
own effect with no parent confounding -- the cleanest question the data can answer.

SPLITTING IS BY START LINEAGE, never at random.  Siblings are extremely related; a random split
would put a molecule's near-twin on the other side and report memorisation as generalisation.
Train on some initial-molecule lineages, test on entirely different ones.

BASELINES ARE THE POINT.  A model that beats nothing is not evidence.  Primitive length alone
already orders sibling pairs at ~71%, so that -- not random -- is the bar the contextual model has
to clear materially.
"""
from __future__ import annotations

import json
import pathlib
import sys
from itertools import combinations

import numpy as np

EDGES = "diagnostics/pmo_blind_traversal_v1/control_dataset_edges_v1.jsonl"
OUT = "diagnostics/pmo_sibling_ranking_v1/sibling_ranking_gate_v1.json"


def pairwise_accuracy(groups, score_of):
    """Fraction of sibling pairs ordered correctly; ties on either side are skipped."""
    right = total = 0
    for group in groups:
        for a, b in combinations(group, 2):
            if a["score"] == b["score"]:
                continue
            pa, pb = score_of(a), score_of(b)
            if pa == pb:
                continue
            total += 1
            right += (pa > pb) == (a["score"] > b["score"])
    return (right / total if total else float("nan")), total


def enrichment(groups, score_of, *, k=1):
    """Mean true score of the top-k picks, and the fraction of oracle headroom captured."""
    picked, best, mean = [], [], []
    for group in groups:
        order = sorted(group, key=lambda r: -score_of(r))[:k]
        picked.append(float(np.mean([r["score"] for r in order])))
        best.append(max(r["score"] for r in group))
        mean.append(float(np.mean([r["score"] for r in group])))
    picked, best, mean = np.array(picked), np.array(best), np.array(mean)
    headroom = best - mean
    captured = np.where(headroom > 0, (picked - mean) / np.maximum(headroom, 1e-12), 0.0)
    return {
        "mean_true_score_of_pick": float(picked.mean()),
        "mean_group_score": float(mean.mean()),
        "mean_group_best": float(best.mean()),
        "headroom_captured": float(captured.mean()),
    }


def spearman(groups, score_of):
    values = []
    for group in groups:
        if len(group) < 4:
            continue
        truth = np.argsort(np.argsort([r["score"] for r in group]))
        guess = np.argsort(np.argsort([score_of(r) for r in group]))
        if truth.std() == 0 or guess.std() == 0:
            continue
        values.append(float(np.corrcoef(truth, guess)[0, 1]))
    return float(np.mean(values)) if values else float("nan")


def main():
    from compose_v4.control.pmo_contextual_macro import ContextualMacroValue

    with open(EDGES) as handle:
        rows = [json.loads(line) for line in handle]
    starts = sorted({r["start"] for r in rows})
    if len(starts) < 2:
        print(f"need at least two start lineages, have {starts}")
        return 1
    holdout = starts[-1]
    train = [r for r in rows if r["start"] != holdout]
    test = [r for r in rows if r["start"] == holdout]
    by_group: dict[str, list] = {}
    for r in test:
        by_group.setdefault(r["sibling_group"], []).append(r)
    groups = [g for g in by_group.values() if len(g) >= 4]
    print(f"train {len(train)} edges ({len(starts) - 1} starts) | "
          f"test {len(test)} edges on '{holdout}' | sibling groups >=4: {len(groups)}")

    rng = np.random.default_rng(20260923)
    arms = {
        "random": lambda r: float(rng.random()),
        "primitive_length_shorter_is_better": lambda r: -float(r["primitives"]),
        "parent_score_only": lambda r: float(r["parent_score"]),
    }
    report = {
        "schema_version": "pmo_sibling_ranking_gate_v1",
        "evidence_role": "held_out_development_gate_on_beam_transitions",
        "new_charged_oracle_calls": 0,
        "holdout_start": holdout,
        "train_starts": [s for s in starts if s != holdout],
        "train_edges": len(train),
        "test_edges": len(test),
        "sibling_groups": len(groups),
        "arms": {},
    }
    for blocks in (("macro",), ("macro", "descriptor"), ("macro", "descriptor", "change"),
                   ("macro", "descriptor", "change", "fingerprint")):
        model = ContextualMacroValue(blocks=blocks)
        model.fit(train, [r["score"] for r in train])
        cache = {id(r): float(p) for r, p in zip(test, model.predict(test), strict=True)}
        arms["+".join(blocks)] = lambda r, c=cache: c[id(r)]

    for name, scorer in arms.items():
        accuracy, pairs = pairwise_accuracy(groups, scorer)
        entry = {
            "pairwise_accuracy": accuracy,
            "pairs": pairs,
            "mean_spearman": spearman(groups, scorer),
            **enrichment(groups, scorer, k=1),
            "top4": enrichment(groups, scorer, k=4),
        }
        report["arms"][name] = entry
        print(
            f"{name:44s} pair={accuracy:.3f} rho={entry['mean_spearman']:+.3f} "
            f"pick={entry['mean_true_score_of_pick']:.4f} "
            f"headroom={entry['headroom_captured']:+.1%}"
        )
    pathlib.Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=1)
    print("WROTE", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
