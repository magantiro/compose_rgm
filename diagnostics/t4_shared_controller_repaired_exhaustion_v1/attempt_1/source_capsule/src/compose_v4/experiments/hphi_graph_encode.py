"""Graph-only encoder input for `h_phi`. Skips the mark-space machinery.

WHY
---
Profiling put ~79% of an SMC particle transition in "encode". That path is
`_one_state_batch` -> `prepare_factorized_mark_batch`, 561 lines that build
admission masks, macro actions and ring-system deletes -- and then
`_encode_batch` reads **none of them**. It reads exactly eleven attributes, all
molecular-graph structure and time (enumerated by AST, every `batch.*` access):

    atom_types  formal_charges  implicit_h_counts  neural_bonds  times
    atom_topology  closure_topology  ring_system_topology
    property_condition_values  property_condition_mask  batch_size

So the encoder does not need a `FactorizedMarkBatch` at all -- it needs an
object with those attributes. This builds exactly that, from
`compute_topology_features` and the state's own arrays, and skips the seven
expensive call sites entirely.

WHAT THIS IS NOT
----------------
Not an approximation and not a reimplementation of the encoder. The same
`model._encode_batch` runs on the same tensor values; only the *construction*
of those tensors is cheaper. Qualification is therefore bitwise equality of the
embedding and of `h_phi`, run IN-PROCESS against the existing path -- comparing
two separate Modal runs would confound the change with cross-container float
nondeterminism, which is exactly what muddied the chemistry-cache parity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

__all__ = ["GraphOnlyBatch", "build_graph_only_batch", "encode_graph_only"]


@dataclass
class GraphOnlyBatch:
    """Duck-typed stand-in carrying only what `_encode_batch` reads.

    Deliberately NOT a FactorizedMarkBatch: that dataclass has 43 fields, 21 of
    them required and unread by the encoder, so constructing one would mean
    fabricating 21 tensors purely to satisfy a constructor.
    """

    atom_types: torch.Tensor
    formal_charges: torch.Tensor
    implicit_h_counts: torch.Tensor
    neural_bonds: torch.Tensor
    times: torch.Tensor
    atom_topology: torch.Tensor
    closure_topology: torch.Tensor
    ring_system_topology: torch.Tensor
    property_condition_values: torch.Tensor | None
    property_condition_mask: torch.Tensor | None
    batch_size: int


def build_graph_only_batch(states, times, *, use_aromatic_bond_view: bool = True):
    """Build encoder input for a batch of states. No mark-space work."""
    from compose_v4.model.factorized_tracelet_rate_model import (
        compute_topology_features, resonance_invariant_bond_classes,
    )

    at, fc, ih, nb, atop, ctop, rtop = [], [], [], [], [], [], []
    for st in states:
        a, c, r = compute_topology_features(st)
        atop.append(a); ctop.append(c); rtop.append(r)
        at.append(st.atom_types)
        fc.append(st.formal_charges)
        ih.append(st.implicit_h_counts)
        nb.append(resonance_invariant_bond_classes(st) if use_aromatic_bond_view
                  else st.bonds)

    def stack(xs):
        return torch.from_numpy(np.stack([np.asarray(x) for x in xs]))

    return GraphOnlyBatch(
        atom_types=stack(at), formal_charges=stack(fc),
        implicit_h_counts=stack(ih), neural_bonds=stack(nb),
        times=torch.tensor([float(t) for t in times]),
        atom_topology=stack(atop), closure_topology=stack(ctop),
        ring_system_topology=stack(rtop),
        property_condition_values=None, property_condition_mask=None,
        batch_size=len(states),
    )


def encode_graph_only(model, states, time: float) -> np.ndarray:
    """Graph-level embeddings via the cheap path. Same encoder, same values."""
    batch = build_graph_only_batch(states, [time] * len(states))
    with torch.no_grad():
        _node, g, _pair = model._encode_batch(batch)
    return g.detach().cpu().numpy()
