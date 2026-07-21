"""Lipid region-aware rate model: a subclass of the shared factorized rate model.

Adds two lipid-native inductive biases on top of the frozen shared core, without
editing it beyond two behaviour-preserving extract-method hooks
(`_atom_node_features`, `_connected_atom_order_prior`):

1. a per-atom **region embedding** {head, linker, tail, other} in the encoder, so
   the network sees each atom's functional role;
2. a **region-conditioned** base-rate for connected atom insertion, so the
   element the model wants to add depends on the region (tails->carbon,
   linkers->O/N, heads->N) -- correcting the global element prior that would
   scatter heteroatoms into tails.

Region is a deterministic function of the current (partial) molecule
(`lipid_region_labels`), computed from the `MolecularGraph` states the batch
already carries -- so it works at training and during ancestral generation and
adds no new sampling degrees of freedom. The embedding is zero-initialized, so an
untrained model with no prior table is bit-for-bit the base model; the region
signal is learned as a residual (matching `corpus_residual_v1`).
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from torch import Tensor, nn

from compose_v4.chem.molecular_graph import (
    BOND_AROMATIC,
    BOND_DOUBLE,
    BOND_NULL,
    BOND_SINGLE,
    BOND_TRIPLE,
    IDX_TO_ELEMENT,
    MolecularGraph,
    NULL_IDX,
    contract_scars,
)
from compose_v4.lipids.region_labels import OTHER, lipid_region_labels
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedMarkBatch,
    FactorizedTraceletRateModel,
)

_N_REGIONS = 4
_BOND_TYPE = {
    BOND_SINGLE: Chem.BondType.SINGLE,
    BOND_DOUBLE: Chem.BondType.DOUBLE,
    BOND_TRIPLE: Chem.BondType.TRIPLE,
    BOND_AROMATIC: Chem.BondType.AROMATIC,
}


def _graph_to_mol(mg: MolecularGraph):
    """Order-preserving MolecularGraph -> (sanitized Mol, graph_idx->rdkit_idx).

    Mirrors chem.molecular_graph_to_smiles' build but returns the Mol (atoms in
    graph order) plus the index map, so region labels align with the batch tensors.
    Returns (None, {}) if the graph does not sanitize."""
    contracted = contract_scars(mg)
    if contracted is None:
        return None, {}
    mg = contracted
    rw = Chem.RWMol()
    n = mg.n_atoms
    is_aromatic = (mg.bonds == BOND_AROMATIC).any(axis=1)
    rd_idx: dict[int, int] = {}
    for i in range(n):
        elem = int(mg.atom_types[i])
        if elem == NULL_IDX:
            continue
        atom = Chem.Atom(IDX_TO_ELEMENT[elem])
        atom.SetFormalCharge(int(mg.formal_charges[i]))
        atom.SetNumExplicitHs(int(mg.implicit_h_counts[i]))
        atom.SetNoImplicit(True)
        if is_aromatic[i]:
            atom.SetIsAromatic(True)
        rd_idx[i] = rw.AddAtom(atom)
    for i in range(n):
        if i not in rd_idx:
            continue
        for j in range(i + 1, n):
            if j not in rd_idx:
                continue
            cls = int(mg.bonds[i, j])
            if cls == BOND_NULL:
                continue
            bt = _BOND_TYPE.get(cls)
            if bt is None:
                return None, {}
            rw.AddBond(rd_idx[i], rd_idx[j], bt)
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError):
        return None, {}
    return mol, rd_idx


def region_ids_for_state(mg: MolecularGraph) -> np.ndarray:
    """Per-atom region code aligned to the graph's atom order (OTHER for null /
    scar / unsanitizable). Length == mg.n_atoms (== batch n_slots)."""
    ids = np.full(int(mg.n_atoms), OTHER, dtype=np.int64)
    mol, rd_idx = _graph_to_mol(mg)
    if mol is None:
        return ids
    labels = lipid_region_labels(mol)  # rdkit order
    for graph_i, rd_i in rd_idx.items():
        if rd_i < len(labels):
            ids[graph_i] = labels[rd_i]
    return ids


class RegionAwareFactorizedTraceletRateModel(FactorizedTraceletRateModel):
    """Region-aware lipid variant of the shared factorized rate model."""

    def __init__(self, *args, region_prior_table=None, region_cache_size: int = 16384, **kwargs):
        super().__init__(*args, **kwargs)
        self.region_embedding = nn.Embedding(_N_REGIONS, self.hidden_dim)
        nn.init.zeros_(self.region_embedding.weight)  # untrained -> no region effect (residual)
        if region_prior_table is not None:
            table = torch.as_tensor(np.asarray(region_prior_table, dtype=np.float32))
            if tuple(table.shape) != (_N_REGIONS, 3, len("CNOF")):
                raise ValueError("region_prior_table must be [4, 3, 4] (region x order x C/N/O/F)")
            logp = torch.log(table.clamp_min(1e-6))
        else:
            logp = torch.zeros(_N_REGIONS, 3, 4)  # no region prior -> equals base
        self.register_buffer("region_conditioned_prior", logp, persistent=True)
        self._region_cache: OrderedDict[bytes, np.ndarray] = OrderedDict()
        self._region_cache_size = int(region_cache_size)

    @classmethod
    def prior_table_from_json(cls, path: Path) -> np.ndarray:
        """Load a [4,3,4] region x order x element prob table from build_region_conditioned_prior."""
        d = json.loads(Path(path).read_text())
        regions = d["axes"]["region"]  # order matches OTHER/HEAD/LINKER/TAIL codes
        per = d["connected_atom_order_prior"]["per_region"]
        return np.stack([np.asarray(per[r], dtype=np.float32) for r in regions])  # [4,3,4]

    def _state_region_ids(self, mg: MolecularGraph) -> np.ndarray:
        key = mg.atom_types.tobytes() + mg.bonds.tobytes()
        cached = self._region_cache.get(key)
        if cached is not None:
            self._region_cache.move_to_end(key)
            return cached
        ids = region_ids_for_state(mg)
        self._region_cache[key] = ids
        if len(self._region_cache) > self._region_cache_size:
            self._region_cache.popitem(last=False)
        return ids

    def _region_id_tensor(self, batch: FactorizedMarkBatch, device) -> Tensor:
        n_slots = batch.n_slots
        rows = np.zeros((batch.batch_size, n_slots), dtype=np.int64)
        for b, state in enumerate(batch.states):
            ids = self._state_region_ids(state)
            rows[b, : ids.shape[0]] = ids[:n_slots]
        return torch.from_numpy(rows).to(device)

    # --- overridden hooks (the only integration points) ---------------------
    def _atom_node_features(self, batch, atom_types, charges, hydrogens, real_float):
        base = super()._atom_node_features(batch, atom_types, charges, hydrogens, real_float)
        region = self._region_id_tensor(batch, base.device)
        return base + self.region_embedding(region) * real_float

    def _connected_atom_order_prior(self, batch):
        region = self._region_id_tensor(batch, self.region_conditioned_prior.device)
        return self.region_conditioned_prior[region]  # [batch, n_slots, order, type]
