"""Future versus immediate choice on one locked, sampled complete-program tree.

This adapter reuses the existing max-witness planner. Its reference is the
empirical four-draw proposal pool, NOT the full molecular reference kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compose_v4.control.frontier_search import FrontierSearch
from compose_v4.control.task_search import SearchRow


@dataclass(frozen=True)
class ProposalNode:
    key: str
    score: float
    children: tuple[str, ...] = ()


def choose_continuation(
    first: list[dict | None],
    second: list[list[dict | None]],
    *,
    seed: int,
    snapshot_id: str,
) -> dict:
    """Same generator/tree/acting uniform, differing only in value propagation."""
    if len(first) != 4 or len(second) != 4 or any(len(row) != 4 for row in second):
        raise ValueError("continuation comparison requires four-by-four locked draws")
    nodes = {"root": ProposalNode("root", 0, tuple(f"first/{i}" for i in range(4)))}
    for i, candidate in enumerate(first):
        if candidate is None and any(child is not None for child in second[i]):
            raise ValueError("failed first proposal cannot have successful continuations")
        key = f"first/{i}"
        nodes[key] = ProposalNode(
            key,
            0 if candidate is None else candidate["score"],
            tuple(f"{key}/{j}" for j in range(4)),
        )
        for j, child in enumerate(second[i]):
            subkey = f"{key}/{j}"
            nodes[subkey] = ProposalNode(subkey, 0 if child is None else child["score"])
    if any(not np.isfinite(n.score) or not 0 <= n.score <= 1 for n in nodes.values()):
        raise ValueError("continuation scores must be finite values in [0,1]")

    def reference(node):
        children = tuple(nodes[key] for key in node.children)
        probabilities = tuple(1 / len(children) for _ in children)
        return SearchRow(children, node.children, probabilities, probabilities, 0.1)

    decisions = {}
    uniform = float(np.random.default_rng(seed).random())
    for arm in ("immediate", "future"):
        planner = FrontierSearch(
            reference,
            lambda n: n.score,
            lambda n: not n.children,
            lambda n: n.key,
            reference_id=snapshot_id,
            snapshot_id=snapshot_id,
            seed=seed,
        )
        root = nodes["root"]
        planner.row(root)
        if arm == "future":
            # decision evaluates its witnessed children and backs their values up
            # through existing parent edges. No new scores are obtained here.
            for key in root.children:
                planner.decision(nodes[key])
        decision = planner.decision(root)
        index = min(
            int(np.searchsorted(np.cumsum(decision["probabilities"]), uniform, side="right")), 3
        )
        decisions[arm] = {
            **decision,
            "selected": index,
            "abstained": first[index] is None,
            "acting_uniform": uniform,
        }
    return {
        "schema_version": "continuation_choice_v1",
        "reference_semantics": "uniform_over_four_sampled_complete_program_draws",
        "decisions": decisions,
    }
