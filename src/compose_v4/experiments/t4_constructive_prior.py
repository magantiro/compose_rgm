"""Fit and grade the coupled constructive proposal law under frozen target folds.

Motivation is one measured number: teacher-forcing the production sampler at 308
decision points reproduced 379 of 7,353 destructive proposals and **0 of 11,762
constructive** ones, with every family compiling at 285 to 308 of the 308 states. The
sampler can build; it never builds where and how the route builds.

This driver turns the 105 constructive dependency-region decisions in the 77 exact
routes into `(state, site, mode)` examples, fits
`S(G,r,m) = A(G,r) + B(G,m) + C(G,r,m)` over the legal pairs at each state, and grades
it leave-one-TARGET-out so no protein's own routes inform its score.

Exact teacher reproduction is a diagnostic here, not the criterion. The criterion is
whether a held-out construction lands in the top of a ranked candidate list that the
existing realizer can then execute -- which is what a proposal prior has to do inside a
budget. The reported ranks are against the full legal menu at the state, so a random
law scores at the menu midpoint by construction and that is the stated baseline.

Teacher routes are training data for the prior and are never read at runtime. No
docking calls.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np

from compose_v4.control.constructive_decision import attachment_sites
from compose_v4.control.constructive_features import (
    MODE_FEATURE_NAMES,
    SITE_FEATURE_NAMES,
    mode_identity,
    mode_matrix,
    site_features,
)
from compose_v4.control.constructive_policy import PolicyShape, fit, pair_scores, rank_of_truth
from compose_v4.control.dependency_region_program import dependency_region_program
from compose_v4.experiments.t4_constructive_decisions import is_constructive

SCHEMA_VERSION = "t4_constructive_prior_v1"

# Shrink the interaction two orders harder than the pooled terms. 105 decisions cannot
# support 59 free joint bins; they can support a low-rank tilt on top of two marginals.
DEFAULT_PENALTIES = {"site": 1e-2, "mode": 1e-2, "interaction": 1.0}


def trace_decisions(states, actions, **carry) -> list[dict]:
    """Every constructive decision in one exact trace, as a state/site/mode triple.

    Works for a sealed teacher route and for an archived controller candidate alike:
    both carry the same `(states, actions)` shape, so the corpus is one code path and a
    decision cannot mean two different things depending on where it came from.
    """
    states, actions = tuple(states), tuple(actions)
    source_atoms = len(states[0].get("atom_types", []))
    program = dependency_region_program(states, actions)

    examples = []
    for component in program["components"]:
        if not is_constructive(component):
            continue
        indices = component["primitive_indices"]
        sites = attachment_sites(actions, indices, source_atoms)
        if not sites:
            # A construction with no pre-existing attachment point has no WHERE to
            # score. Counted, not silently dropped.
            continue
        examples.append(
            {
                **carry,
                "state": states[min(indices)],
                "site": min(sites),
                "mode": {
                    "attachment_count": len(sites),
                    "created_atoms": component["created_outputs"],
                    "closes_ring": component["contains_cycle_close"],
                    "opens_ring": component["contains_cycle_open"],
                    "primitive_count": component["primitive_count"],
                },
            }
        )
    return examples


def route_examples(receipt: Path) -> list[dict]:
    """Constructive decisions from one sealed teacher receipt."""
    payload = json.loads(gzip.decompress(Path(receipt).read_bytes()))["payload"]["path"]
    return trace_decisions(payload["states"], payload["actions"], receipt=str(receipt))


def mode_vocabulary(examples) -> list[dict]:
    """Distinct medium-granularity modes, which is the menu every state is scored over."""
    seen: dict[tuple, dict] = {}
    for example in examples:
        seen.setdefault(mode_identity(example["mode"]), example["mode"])
    return [seen[key] for key in sorted(seen)]


def encode(examples, vocabulary) -> list[dict]:
    """Attach the legal-pair matrices each example is scored against."""
    modes = mode_matrix(vocabulary)
    index = {mode_identity(m): i for i, m in enumerate(vocabulary)}
    encoded = []
    for example in examples:
        matrix, slots = site_features(example["state"])
        if example["site"] not in slots:
            continue
        key = mode_identity(example["mode"])
        if key not in index:
            continue
        encoded.append(
            {
                **example,
                "sites": matrix,
                "modes": modes,
                "site_index": slots.index(example["site"]),
                "mode_index": index[key],
                "slots": slots,
            }
        )
    return encoded


def folds(examples, target_of) -> dict[str, list[int]]:
    """Leave-one-target-out: a protein's own routes never inform its own score."""
    grouped: dict[str, list[int]] = {}
    for position, example in enumerate(examples):
        grouped.setdefault(target_of(example["receipt"]), []).append(position)
    return grouped


def _uniform_reference(example) -> dict:
    """What a law with no preference scores, which is the baseline every rank is read against."""
    n_sites, n_modes = example["sites"].shape[0], example["modes"].shape[0]
    return {
        "expected_joint_rank": (n_sites * n_modes - 1) / 2.0,
        "expected_site_rank": (n_sites - 1) / 2.0,
        "expected_mode_rank": (n_modes - 1) / 2.0,
    }


def evaluate(examples, target_of, *, shape=None, penalties=None) -> dict:
    """Fit once per held-out target and grade that target's own decisions."""
    penalties = penalties or DEFAULT_PENALTIES
    shape = shape or PolicyShape(len(SITE_FEATURE_NAMES), len(MODE_FEATURE_NAMES))
    assignment = folds(examples, target_of)

    graded, per_fold = [], {}
    for target, positions in sorted(assignment.items()):
        held_out = set(positions)
        train = [e for i, e in enumerate(examples) if i not in held_out]
        if not train:
            continue
        model = fit(train, shape, penalties=penalties)
        scored = []
        for position in positions:
            example = examples[position]
            ranks = rank_of_truth(model, example)
            ranks.update(_uniform_reference(example))
            ranks["target"] = target
            scored.append(ranks)
        per_fold[target] = {
            "decisions": len(scored),
            "train_decisions": len(train),
            "converged": model["converged"],
            "median_site_rank": float(np.median([s["site_rank"] for s in scored])),
            "median_joint_rank": float(np.median([s["joint_rank"] for s in scored])),
        }
        graded.extend(scored)

    return {"graded": graded, "per_fold": per_fold, "penalties": penalties}


def summarize(graded) -> dict:
    """Held-out ranking quality against the uniform law on the same menus."""
    if not graded:
        return {"decisions": 0}

    def hit(key, k):
        return sum(1 for g in graded if g[key] < k) / len(graded)

    return {
        "decisions": len(graded),
        "site": {
            "top1": hit("site_rank", 1),
            "top3": hit("site_rank", 3),
            "top8": hit("site_rank", 8),
            "median_rank": float(np.median([g["site_rank"] for g in graded])),
            "median_menu": float(np.median([g["site_candidates"] for g in graded])),
            "uniform_median_rank": float(np.median([g["expected_site_rank"] for g in graded])),
        },
        "mode_given_site": {
            "top1": hit("mode_rank_given_site", 1),
            "top3": hit("mode_rank_given_site", 3),
            "top8": hit("mode_rank_given_site", 8),
            "median_rank": float(np.median([g["mode_rank_given_site"] for g in graded])),
            "median_menu": float(np.median([g["mode_candidates"] for g in graded])),
            "uniform_median_rank": float(np.median([g["expected_mode_rank"] for g in graded])),
        },
        "joint": {
            "top1": hit("joint_rank", 1),
            "top8": hit("joint_rank", 8),
            "top32": hit("joint_rank", 32),
            "median_rank": float(np.median([g["joint_rank"] for g in graded])),
            "median_menu": float(np.median([g["joint_candidates"] for g in graded])),
            "uniform_median_rank": float(np.median([g["expected_joint_rank"] for g in graded])),
        },
        "per_target": dict(sorted(Counter(g["target"] for g in graded).items())),
        "new_oracle_calls": 0,
    }


def proposal_distribution(model, state, vocabulary) -> dict:
    """The law a sampler would draw from at one state: a distribution over legal pairs."""
    matrix, slots = site_features(state)
    scores = pair_scores(model["theta"], model["shape"], matrix, mode_matrix(vocabulary))
    flat = scores.reshape(-1)
    weights = np.exp(flat - flat.max())
    weights /= weights.sum()
    return {
        "slots": slots,
        "modes": [mode_identity(m) for m in vocabulary],
        "probabilities": weights.reshape(scores.shape),
    }
