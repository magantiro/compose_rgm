"""Round-synchronous, canonical-deduplicated scored donor memory.

This changes proposal geometry, not the executor or a frozen reference model.
The rank prior is an optimization heuristic, not calibrated uncertainty.
"""

from __future__ import annotations

import numpy as np

from compose_v4.control.archive_allocation import parent_distribution
from compose_v4.control.docking_value import identity
from compose_v4.control.molecular_search_codec import decode_search_state
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

RECIPE = {
    "schema_version": "evolving_donor_memory_v1",
    "donor_probability": 0.5,
    "donor_prior": "canonical_score_inverse_rank_with_uniform_exploration",
    "donor_exploration": 0.2,
    "cut_distribution": "uniform_oriented_single_bridge",
    "compiler_max_steps": 64,
    "compiler_max_expansions": 128,
    "updates": "only_own_requested_scored_products_after_round_completion",
}


def build_memory(initial: list[dict], archive: dict[str, dict], *, mode: str) -> dict:
    if mode not in ("fixed", "evolving"):
        raise ValueError(f"unknown donor memory mode: {mode}")
    rows = {r["smiles"]: r for r in initial}
    if not rows or len(rows) != len(initial):
        raise ValueError("initial donors must be nonempty and canonical-deduplicated")
    if mode == "evolving":
        for key, node in sorted(archive.items()):
            if key != node["smiles"]:
                raise ValueError("donor archive identity differs from its key")
            row = {
                "smiles": key,
                "score": node["score"],
                "state": node["node"]["graph"],
                "origin": node["id"],
            }
            if key in rows and rows[key]["score"] != row["score"]:
                raise ValueError("inconsistent canonical donor score")
            rows.setdefault(key, row)
    ordered = [rows[s] for s in sorted(rows)]
    for row in ordered:
        if canonical_state_key(decode_state(row["state"])) != row["smiles"]:
            raise ValueError("donor identity differs from exact stored graph")
    weights = parent_distribution(
        ordered, {r["smiles"]: r["score"] for r in ordered}, RECIPE["donor_exploration"]
    )
    body = {"recipe": RECIPE, "rows": ordered, "probabilities": weights.tolist()}
    return {**body, "memory_id": identity(body)}


def proposal_identity(memory_id: str) -> str:
    return identity({"recipe": RECIPE, "memory_id": memory_id})


def validate_memory(memory: dict, expected_id: str) -> None:
    if (
        memory["recipe"] != RECIPE
        or memory["memory_id"] != expected_id
        or identity({k: v for k, v in memory.items() if k != "memory_id"}) != expected_id
    ):
        raise ValueError("donor memory content identity mismatch")
    rows, weights = memory["rows"], np.asarray(memory["probabilities"])
    expected = parent_distribution(
        rows, {r["smiles"]: r["score"] for r in rows}, RECIPE["donor_exploration"]
    )
    if len({r["smiles"] for r in rows}) != len(rows) or not np.allclose(
        weights, expected, rtol=1e-14, atol=1e-15
    ):
        raise ValueError("donor memory prior or canonical census changed")


def checked_parent(node: dict) -> None:
    if canonical_state_key(decode_search_state(node["node"]).graph) != node["smiles"]:
        raise ValueError("parent lacks its recorded exact molecular state")
