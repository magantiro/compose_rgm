"""Freeze a prospected pool into an immutable, score-blind candidate lock.

A pool from `constructive_composition.prospect` is eligible and unscored. Turning it
into charged calls needs a lock that is fixed before any score is observed, which is
what every prior T4 experiment in this repository does and what the milestone
requires.

Two rules do the real work here.

**Rediscovered endpoints are excluded from the charged set.** The recovered corpus
holds 35,895 charged observations; an endpoint already in it has a measured score, so
docking it again buys nothing. Those members are reported as free evidence about the
pool and their calls are spent on novel molecules instead.

**Selection is score-blind and declared before the pool exists.** Novel members are
ranked by program size, which is the measured value axis, under a diversity cap on
repeated family multisets so a budget is not spent on near-identical constructions.
No docking score, historical or otherwise, orders the charged set.
"""

from __future__ import annotations

from collections import Counter

from compose_v4.control.docking_value import identity
from compose_v4.experiments.t4_prospect_assessment import endpoint_sha256

SCHEMA_VERSION = "t4_prospect_candidate_lock_v1"


def select(pool, scored_endpoints, *, take: int, per_signature: int = 3) -> dict:
    """Score-blind selection of novel pool members, with a diversity cap.

    `scored_endpoints` is the set of endpoint digests already charged for this cell.
    Ranking is by primitive count descending, then by endpoint digest so the order is
    deterministic and independent of pool iteration order.
    """
    if take < 1 or per_signature < 1:
        raise ValueError("a lock needs a positive take and diversity cap")
    novel, rediscovered = [], []
    for row in pool:
        digest = endpoint_sha256(row["endpoint"])
        (rediscovered if digest in scored_endpoints else novel).append({**row, "digest": digest})
    ordered = sorted(novel, key=lambda row: (-row["primitives"], row["digest"]))
    taken, seen = [], Counter()
    for row in ordered:
        signature = tuple(sorted(row["families"]))
        if seen[signature] >= per_signature:
            continue
        seen[signature] += 1
        taken.append(row)
        if len(taken) == take:
            break
    return {
        "take": [
            {
                "endpoint": row["endpoint"],
                "endpoint_sha256": row["digest"],
                "primitives": row["primitives"],
                "families": row["families"],
                "changed_originals": row["changed_originals"],
                "created": row["created"],
                "properties": row["properties"],
            }
            for row in taken
        ],
        "pool_size": len(pool),
        "novel": len(novel),
        "rediscovered_excluded": len(rediscovered),
        "distinct_family_signatures": len(seen),
        "diversity_cap_per_signature": per_signature,
        "requested": take,
        "shortfall": max(0, take - len(taken)),
        "selection_rule": (
            "novel endpoints only, ranked by primitive count descending then by "
            "endpoint digest, capped per family signature; no docking score of any "
            "kind participates in this ordering"
        ),
    }


def build_lock(
    selections: dict,
    *,
    call_limit: int,
    authorization_ceiling: int,
    docking_seed: int,
    required_rdkit: str,
    oracle_protocols: dict,
    input_sha256: dict,
) -> dict:
    """Assemble the immutable lock. Raises rather than silently exceeding a budget."""
    charged = sum(len(entry["take"]) for entry in selections.values())
    if charged > call_limit:
        raise ValueError(f"lock would charge {charged} calls against a limit of {call_limit}")
    if call_limit > authorization_ceiling:
        raise ValueError("call limit exceeds the recorded authorization ceiling")
    missing = sorted(set(selections) - set(oracle_protocols))
    if missing:
        raise ValueError(f"no oracle protocol bound for: {', '.join(missing)}")
    body = {
        "schema_version": SCHEMA_VERSION,
        "cells": sorted(selections),
        "selections": {cell: selections[cell] for cell in sorted(selections)},
        "charged_calls": charged,
        "new_oracle_call_limit": call_limit,
        "authorization_ceiling": authorization_ceiling,
        "docking_seed": docking_seed,
        "required_rdkit": required_rdkit,
        "oracle_protocols": {cell: oracle_protocols[cell] for cell in sorted(selections)},
        "input_sha256": dict(sorted(input_sha256.items())),
        "automatic_retries": 0,
        "replacement_after_scoring": False,
        "interpretation": (
            "prospected eligible endpoints, never docked, frozen before any score is "
            "observed; bounded development evidence on answer-known cells, not a "
            "held-out benchmark result and not an IVG comparison"
        ),
    }
    return {**body, "lock_id": identity(body)}


def verify_lock(lock: dict) -> None:
    """Refuse a lock that was edited after sealing, or that breaks its own budget."""
    body = {k: v for k, v in lock.items() if k != "lock_id"}
    if lock.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"not a {SCHEMA_VERSION} lock")
    if identity(body) != lock.get("lock_id"):
        raise ValueError("candidate lock was modified after it was sealed")
    charged = sum(len(entry["take"]) for entry in lock["selections"].values())
    if charged != lock["charged_calls"] or charged > lock["new_oracle_call_limit"]:
        raise ValueError("candidate lock disagrees with its own charged-call accounting")
    digests = [
        row["endpoint_sha256"] for entry in lock["selections"].values() for row in entry["take"]
    ]
    if len(digests) != len(set(digests)):
        raise ValueError("candidate lock charges the same endpoint twice")
