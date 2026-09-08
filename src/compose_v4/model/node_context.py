"""Generic opt-in CPU feature interface; application chemistry lives downstream."""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np
import torch

from compose_v4.chem.molecular_graph import MolecularGraph, is_element
from compose_v4.rewrite.scaffold_construction import ScaffoldContext


class NodeContextProvider(Protocol):
    """A deterministic, immutable, pickleable current-state-only feature recipe.

    No target/teacher, time, family label or outcome is passed to this interface.
    A recipe change MUST change schema. Implementations must not mutate inputs.
    This affects node encoding, never candidate support or executor semantics.
    A provider may set requires_aromatic_bond_view=True to reject raw Kekule
    collation when its chemistry depends on the resonance-invariant view.
    """

    @property
    def schema(self) -> str: ...

    @property
    def feature_dim(self) -> int: ...

    def __call__(
        self,
        state: MolecularGraph,
        scaffold_context: ScaffoldContext | None,
        neural_bonds: np.ndarray,
    ) -> np.ndarray: ...


def node_context_key(state: MolecularGraph, context: ScaffoldContext | None) -> str:
    """Bind prepared features to exact executable coordinates and supplied core."""
    digest = hashlib.sha256()
    digest.update(("none" if context is None else context.identity).encode())
    for name in ("atom_types", "formal_charges", "implicit_h_counts", "bonds"):
        value = np.asarray(getattr(state, name))
        digest.update(repr((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def prepare_node_context(provider, states, contexts, neural_bonds):
    if provider is None:
        return None, None, None
    if not isinstance(provider.schema, str) or not provider.schema:
        raise ValueError("node context provider requires a nonempty schema")
    if not isinstance(provider.feature_dim, int) or provider.feature_dim <= 0:
        raise ValueError("node context feature dimension must be a positive integer")
    rows, keys = [], []
    for index, (state, bonds) in enumerate(zip(states, neural_bonds, strict=True)):
        context = None if contexts is None else contexts[index]
        row = np.asarray(provider(state, context, bonds), dtype=np.float32)
        if row.shape != (state.n_atoms, provider.feature_dim):
            raise ValueError("node context features have the wrong shape")
        if not np.isfinite(row).all() or np.any(row[~is_element(state.atom_types)] != 0):
            raise ValueError("node context features must be finite and zero on inactive slots")
        rows.append(row)
        keys.append(node_context_key(state, context))
    return torch.from_numpy(np.stack(rows)), provider.schema, tuple(keys)
