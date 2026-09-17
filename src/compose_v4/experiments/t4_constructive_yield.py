"""Is the proposal law's preferred region reachable by the production compiler?

Ranking the teacher's construction inside a menu is necessary and not sufficient. The
law places the held-out teacher decision at a median joint rank of 31 out of 1,850, but
that is worth nothing operationally if the existing machinery can only realize pairs
ranked far lower: a prior that prefers what the compiler cannot build is a prior that
cannot be followed.

So this probes the OTHER direction. At each teacher state it draws proposals from the
unchanged generic machinery, reads off the `(site, mode)` pair each one actually landed
on, and reports where those realized pairs sit in the law's own ranking. Three numbers
decide it:

- where realizable pairs sit in the ranking, against the 1,850-pair menu;
- what share of the law's top-k is realizable at all, which is the hit rate a
  generate-and-filter controller would see;
- whether the teacher's own pair is ever realized, which is the 0-of-11,762 baseline
  restated at the level of decisions rather than exact action prefixes.

No compiler change, no vocabulary change, no docking. The law is fitted with the probed
state's target held out.
"""

from __future__ import annotations

import numpy as np

from compose_v4.control.constructive_features import mode_identity, mode_matrix, site_features
from compose_v4.control.constructive_policy import pair_scores
from compose_v4.control.dynamic_program_synthesis import GENERIC_MODULES, compile_generic_module
from compose_v4.experiments.t4_constructive_decisions import attachment_sites
from compose_v4.rewrite.trace_shard import decode_state

SCHEMA_VERSION = "t4_constructive_yield_v1"


def stage_decision(source_state, actions) -> dict | None:
    """The `(site, mode)` a realized proposal landed on, or None if it built nothing.

    Mirrors the corpus extractor's definition so a realized proposal and a teacher
    decision are described by the same quantities; otherwise the ranking would compare
    two different things and read as a miss.
    """
    if not actions:
        return None
    created = sum(1 for a in actions if a.get("executor_rule") in ("atom_insert",))
    closes = any(a.get("executor_rule") == "cycle_close" for a in actions)
    if not created and not closes:
        return None
    source_atoms = len(source_state.get("atom_types", []))
    sites = attachment_sites(actions, range(len(actions)), source_atoms)
    if not sites:
        return None
    return {
        "site": min(sites),
        "mode": {
            "attachment_count": len(sites),
            "created_atoms": created,
            "closes_ring": closes,
            "opens_ring": any(a.get("executor_rule") == "cycle_open" for a in actions),
            "primitive_count": len(actions),
        },
    }


def realized_pairs(source_state, rng, *, samples: int, families=GENERIC_MODULES) -> dict:
    """Draw from the unchanged machinery and collect the decisions it actually reaches.

    `source_state` is the receipt's serialised state; the compiler wants a decoded graph,
    so the decode happens here rather than at every call site.
    """
    decoded = decode_state(source_state)
    reached, constructive, compiled = [], 0, 0
    for family in families:
        for _ in range(samples):
            try:
                _, stage = compile_generic_module(decoded, rng, family)
            except ValueError:
                continue
            compiled += 1
            decision = stage_decision(source_state, stage["actions"])
            if decision is None:
                continue
            constructive += 1
            reached.append({**decision, "family": family})
    return {
        "compiled": compiled,
        "constructive": constructive,
        "constructive_rate": constructive / compiled if compiled else 0.0,
        "decisions": reached,
    }


def rank_realized(model, source_state, vocabulary, reached) -> dict:
    """Where the realized pairs sit in the law's ranking over the full legal menu."""
    matrix, slots = site_features(source_state)
    scores = pair_scores(model["theta"], model["shape"], matrix, mode_matrix(vocabulary))
    order = {mode_identity(m): i for i, m in enumerate(vocabulary)}
    flat = scores.reshape(-1)
    descending = np.argsort(-flat)
    rank_of = np.empty_like(descending)
    rank_of[descending] = np.arange(flat.size)

    ranks, unrepresentable = [], 0
    for decision in reached:
        key = mode_identity(decision["mode"])
        if decision["site"] not in slots or key not in order:
            unrepresentable += 1
            continue
        ranks.append(int(rank_of[slots.index(decision["site"]) * len(vocabulary) + order[key]]))
    return {
        "menu": int(flat.size),
        "realized": len(ranks),
        "outside_the_fitted_vocabulary": unrepresentable,
        "ranks": sorted(ranks),
        "median_rank": float(np.median(ranks)) if ranks else None,
        "best_rank": min(ranks) if ranks else None,
    }


def hit_rate_in_top_k(ranked: dict, k_values=(8, 32, 128, 512)) -> dict:
    """Share of realized proposals the law would keep if it kept only its top k.

    This is the number a generate-and-filter controller lives on: draw cheaply, keep the
    law's favourites, and this says how often a favourite is something the compiler can
    actually build.
    """
    ranks = np.asarray(ranked["ranks"])
    if ranks.size == 0:
        return dict.fromkeys(k_values, 0.0)
    return {k: float((ranks < k).mean()) for k in k_values}
