"""Goal-aligned imitation proposal, separate from the frozen COMPOSE reference.

This is an answer-informed reconstruction controller. It ranks all valid
difference-directed proposals, not the full molecular reference support. The
given persistent-slot goal correspondence is privileged development input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.control.option_continuation import exact_graph_key
from compose_v4.experiments.winner_paths import _actions
from compose_v4.rewrite.action_codec_v4 import encode_action
from compose_v4.rewrite.kernel import InvalidRewrite, canonical_state_key


@dataclass
class Candidate:
    graph: MolecularGraph
    pending: frozenset[int]
    mark: dict


def proposals(graph, goal, pending, system):
    """No teacher action lookup, future-state lookup or search-tree fallback."""
    rows, seen = [], set()
    source = canonical_state_key(graph)
    for family, action in _actions(graph, goal, pending):
        try:
            child = system.apply(graph, family, action)
        except InvalidRewrite:
            continue
        if not 1 <= child.n_real_atoms <= 40 or canonical_state_key(child) == source:
            continue
        remaining = pending - {action.v} if family == "atom_delete" else pending
        key = (exact_graph_key(child), remaining)
        if key in seen:
            continue
        seen.add(key)
        rows.append(Candidate(child, remaining, encode_action(family, action)))
    return rows


def graph_vector(graph):
    if len(graph.atom_types) != 48:
        raise ValueError("imitation requires exact 48-slot states")
    upper = np.triu_indices(48, 1)
    return np.concatenate(
        (
            graph.atom_types / 20,
            graph.formal_charges / 2,
            graph.implicit_h_counts / 4,
            graph.bonds[upper] / 3,
        )
    ).astype(np.float32)


def features(graph, goal, pending, candidates):
    before, target = graph_vector(graph), graph_vector(goal)
    mask = np.zeros(48, dtype=np.float32)
    mask[list(pending)] = 1
    return np.stack(
        [
            np.concatenate((before, target - before, graph_vector(c.graph) - before, mask))
            for c in candidates
        ]
    )


class ImitationRanker(nn.Module):
    def __init__(self, dimension: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, inputs):
        return self.net(inputs).squeeze(-1)


def rollout(source, goal, pending, system, model, max_steps=64):
    """Closed-loop learned decisions from actual states; no teacher forcing."""
    graph, actions, trace, visited = source, [], [], set()
    desired = canonical_state_key(goal)
    for step in range(max_steps + 1):
        smiles = canonical_state_key(graph)
        if smiles == desired:
            return {"status": "reconstructed", "actions": actions, "trace": trace, "smiles": smiles}
        key = (exact_graph_key(graph), pending)
        if key in visited or step == max_steps:
            return {
                "status": "cycle" if key in visited else "budget",
                "actions": actions,
                "trace": trace,
                "smiles": smiles,
            }
        visited.add(key)
        candidates = proposals(graph, goal, pending, system)
        if not candidates:
            return {
                "status": "no_valid_proposal",
                "actions": actions,
                "trace": trace,
                "smiles": smiles,
            }
        x = torch.from_numpy(features(graph, goal, pending, candidates))
        with torch.inference_mode():
            logits = model(x)
        selected = int(logits.argmax())
        child = candidates[selected]
        trace.append(
            {
                "step": step,
                "smiles": smiles,
                "candidate_count": len(candidates),
                "selected": selected,
                "logits": logits.tolist(),
            }
        )
        actions.append(child.mark)
        graph, pending = child.graph, child.pending
    raise AssertionError("unreachable")
