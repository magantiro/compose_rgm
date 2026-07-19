"""Dense marked-rewrite CTMC parameterization for production training.

The exact successor-fiber model remains the semantic reference implementation.
This module implements the scalable parameterization: one batched graph
encoding, analytically masked rule matches, and a normalized distribution over
marks.  No successor molecule is materialized merely to evaluate a loss.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from math import sqrt
from typing import Any, MutableMapping

import networkx as nx
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from compose_v4.chem.aromaticity import resonance_invariant_bond_classes
from compose_v4.chem.graph_primitives import (
    FUSED_SIZE_BUCKETS,
    RING_SIZE_BUCKETS,
    compute_topology_features,
)
from compose_v4.chem.molecular_graph import (
    BOND_AROMATIC,
    BOND_CLASSES,
    BOND_CLASS_TO_H_CHANGE,
    H_COUNT_CLASSES,
    K,
    M,
    MAX_H_COUNT,
    MolecularGraph,
    NULL_IDX,
    SCAR_IDX,
    is_element,
)
from compose_v4.rewrite.factorized_fiber import CNOF_ATOM_TYPES, CNOF_VALENCE
from compose_v4.rewrite.operators import (
    AtomDelete,
    AtomInsert,
    AtomRestate,
    BondReorder,
    BondReroute,
)
from compose_v4.rewrite.ring_system_fiber import (
    build_semantic_ring_system_decoder,
    enumerate_executable_ring_grow_candidates,
    enumerate_ring_system_template_placements,
    enumerate_structured_ring_system_deletes,
    instantiate_semantic_ring_system_grow,
    matching_ring_system_template_indices,
    ring_system_grow_electronic_key,
    ring_system_electronic_template_support_mask,
    ring_system_placement,
    ring_system_placement_key,
    ring_system_placement_local_atom_type_mask,
    ring_system_template_local_support_mask,
    semantic_ring_categories_for_action,
    semantic_ring_next_category_mask,
    semantic_ring_prefix_is_completable,
    RingSystemPlacement,
    ExecutableRingGrowCandidate,
    SemanticRingSystemDecoder,
    structured_ring_system_electronic_aliases,
    structured_ring_system_template_aliases,
    structured_ring_system_templates,
)
from compose_v4.rewrite.tracelets import (
    CycleAttach,
    CycleInsert,
    RingSystemDelete,
    RingSystemGrow,
    RingSystemRestate,
    is_valid_ring_system_grow,
)
from compose_v4.rewrite.typed_ring_catalog import (
    TypedRingCatalog,
    attach_template,
    cycle_template,
)


MARK_RULE_NAMES = (
    "atom_insert",
    "atom_delete",
    "atom_restate",
    "bond_reorder",
    "bond_reroute",
    "cycle_insert",
    "cycle_attach",
    "ring_system_grow",
    "ring_system_delete",
    "ring_system_restate",
)
MARK_RULE_TO_INDEX = {name: index for index, name in enumerate(MARK_RULE_NAMES)}
_CNOF_TO_INDEX = {int(atom_type): index for index, atom_type in enumerate(CNOF_ATOM_TYPES)}
_ORDER_TO_INDEX = {1: 0, 2: 1, 3: 2}


StateCacheKey = tuple[bytes, bytes, bytes, bytes]


def molecular_state_cache_key(state: MolecularGraph) -> StateCacheKey:
    """Return an exact slot-aware key for chemistry-only feature reuse."""

    return (
        state.atom_types.tobytes(),
        state.formal_charges.tobytes(),
        state.implicit_h_counts.tobytes(),
        state.bonds.tobytes(),
    )


@dataclass(frozen=True)
class ChemistryStateFeatures:
    """Reusable application conditions that depend only on one molecular state."""

    atom_topology: np.ndarray
    closure_topology: np.ndarray
    ring_system_topology: np.ndarray
    atom_delete_mask: np.ndarray
    cycle_edge_mask: np.ndarray
    cyclic_pair_mask: np.ndarray
    graft_mask: np.ndarray
    graft_remove_neighbors: np.ndarray
    graft_successor_groups: np.ndarray
    neural_bonds: np.ndarray
    ring_delete_actions: tuple[RingSystemDelete, ...] | None


@dataclass(frozen=True)
class SparseBinaryRows:
    """CSR storage for a Boolean matrix with exact dense reconstruction."""

    indptr: Tensor
    indices: Tensor
    width: int

    def __post_init__(self) -> None:
        if self.indptr.ndim != 1 or self.indices.ndim != 1:
            raise ValueError("sparse Boolean rows require one-dimensional tensors")
        if self.indptr.dtype != torch.long or self.indices.dtype != torch.long:
            raise ValueError("sparse Boolean row indices must use torch.long")
        if self.width < 0 or len(self.indptr) == 0:
            raise ValueError("sparse Boolean rows have invalid dimensions")
        if int(self.indptr[0]) != 0 or int(self.indptr[-1]) != len(self.indices):
            raise ValueError("sparse Boolean row pointers are inconsistent")
        if bool((self.indptr[1:] < self.indptr[:-1]).any()):
            raise ValueError("sparse Boolean row pointers must be monotone")
        if len(self.indices) and (
            int(self.indices.min()) < 0 or int(self.indices.max()) >= self.width
        ):
            raise ValueError("sparse Boolean column index lies outside the matrix")

    @property
    def n_rows(self) -> int:
        return len(self.indptr) - 1

    @classmethod
    def from_index_rows(
        cls,
        rows: tuple[tuple[int, ...], ...],
        *,
        width: int,
    ) -> "SparseBinaryRows":
        if width < 0:
            raise ValueError("sparse Boolean width must be non-negative")
        flattened: list[int] = []
        pointers = [0]
        for row in rows:
            normalized = tuple(sorted(set(int(index) for index in row)))
            if normalized and (normalized[0] < 0 or normalized[-1] >= width):
                raise ValueError("sparse Boolean row contains an invalid index")
            flattened.extend(normalized)
            pointers.append(len(flattened))
        return cls(
            indptr=torch.tensor(pointers, dtype=torch.long),
            indices=torch.tensor(flattened, dtype=torch.long),
            width=int(width),
        )

    @classmethod
    def from_dense(cls, mask: Tensor) -> "SparseBinaryRows":
        if mask.ndim != 2:
            raise ValueError("dense Boolean support must be a matrix")
        cpu_mask = mask.detach().to(device="cpu", dtype=torch.bool)
        rows = tuple(
            tuple(int(index) for index in torch.nonzero(row, as_tuple=False).flatten())
            for row in cpu_mask
        )
        return cls.from_index_rows(rows, width=int(cpu_mask.shape[1]))

    def subrows(self, start: int, stop: int) -> "SparseBinaryRows":
        if not 0 <= start < stop <= self.n_rows:
            raise ValueError("invalid sparse Boolean row slice")
        lower = int(self.indptr[start])
        upper = int(self.indptr[stop])
        return SparseBinaryRows(
            indptr=self.indptr[start : stop + 1] - lower,
            indices=self.indices[lower:upper],
            width=self.width,
        )

    def index_rows(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(
                int(index)
                for index in self.indices[
                    int(self.indptr[row]) : int(self.indptr[row + 1])
                ]
            )
            for row in range(self.n_rows)
        )

    def to_dense(self, *, device: torch.device | None = None) -> Tensor:
        resolved_device = self.indices.device if device is None else device
        dense = torch.zeros(
            (self.n_rows, self.width),
            dtype=torch.bool,
            device=resolved_device,
        )
        if len(self.indices):
            counts = self.indptr[1:] - self.indptr[:-1]
            row_indices = torch.repeat_interleave(
                torch.arange(self.n_rows, dtype=torch.long),
                counts,
            ).to(device=resolved_device)
            dense[
                row_indices,
                self.indices.to(device=resolved_device),
            ] = True
        return dense

    def pin_memory(self) -> "SparseBinaryRows":
        return SparseBinaryRows(
            indptr=self.indptr.pin_memory(),
            indices=self.indices.pin_memory(),
            width=self.width,
        )


@dataclass(frozen=True)
class FactorizedMarkBatch:
    """CPU tensors and aligned teacher marks for one dense GM batch."""

    states: tuple[MolecularGraph, ...]
    atom_types: Tensor
    formal_charges: Tensor
    implicit_h_counts: Tensor
    bonds: Tensor
    neural_bonds: Tensor
    times: Tensor
    atom_topology: Tensor
    closure_topology: Tensor
    ring_system_topology: Tensor
    atom_delete_mask: Tensor
    cycle_edge_mask: Tensor
    cyclic_pair_mask: Tensor
    graft_mask: Tensor
    graft_remove_neighbors: Tensor
    graft_successor_groups: Tensor
    teacher_actions: tuple[Any | None, ...]
    teacher_rule_names: tuple[str | None, ...]
    teacher_rates: Tensor
    importance_weights: Tensor
    ring_restate_actions: tuple[tuple[RingSystemRestate, ...], ...]
    ring_grow_support_mask: Tensor | None = None
    ring_grow_support_sparse: SparseBinaryRows | None = None
    ring_delete_actions: tuple[tuple[RingSystemDelete, ...], ...] | None = None
    ring_grow_support_is_exact: Tensor | bool = False
    ring_grow_enablement_is_exact: Tensor | bool = False

    @property
    def batch_size(self) -> int:
        return int(self.atom_types.shape[0])

    @property
    def n_slots(self) -> int:
        return int(self.atom_types.shape[1])

    def dense_ring_grow_support(self, *, device: torch.device | None = None) -> Tensor | None:
        if self.ring_grow_support_mask is not None:
            return (
                self.ring_grow_support_mask
                if device is None
                else self.ring_grow_support_mask.to(device=device)
            )
        if self.ring_grow_support_sparse is None:
            return None
        if self.ring_grow_support_sparse.n_rows != self.batch_size:
            raise ValueError("sparse ring support has the wrong row count")
        return self.ring_grow_support_sparse.to_dense(device=device)

    def subbatch(self, start: int, stop: int) -> "FactorizedMarkBatch":
        """Return an aligned contiguous view for lossless streamed evaluation."""

        if not 0 <= start < stop <= self.batch_size:
            raise ValueError(
                f"invalid factorized subbatch bounds: {(start, stop)} for {self.batch_size}"
            )

        def tensor_slice(value: Tensor) -> Tensor:
            return value[start:stop]

        def flag_slice(value: Tensor | bool) -> Tensor | bool:
            return tensor_slice(value) if isinstance(value, Tensor) else value

        return FactorizedMarkBatch(
            states=self.states[start:stop],
            atom_types=tensor_slice(self.atom_types),
            formal_charges=tensor_slice(self.formal_charges),
            implicit_h_counts=tensor_slice(self.implicit_h_counts),
            bonds=tensor_slice(self.bonds),
            neural_bonds=tensor_slice(self.neural_bonds),
            times=tensor_slice(self.times),
            atom_topology=tensor_slice(self.atom_topology),
            closure_topology=tensor_slice(self.closure_topology),
            ring_system_topology=tensor_slice(self.ring_system_topology),
            atom_delete_mask=tensor_slice(self.atom_delete_mask),
            cycle_edge_mask=tensor_slice(self.cycle_edge_mask),
            cyclic_pair_mask=tensor_slice(self.cyclic_pair_mask),
            graft_mask=tensor_slice(self.graft_mask),
            graft_remove_neighbors=tensor_slice(self.graft_remove_neighbors),
            graft_successor_groups=tensor_slice(self.graft_successor_groups),
            teacher_actions=self.teacher_actions[start:stop],
            teacher_rule_names=self.teacher_rule_names[start:stop],
            teacher_rates=tensor_slice(self.teacher_rates),
            importance_weights=tensor_slice(self.importance_weights),
            ring_restate_actions=self.ring_restate_actions[start:stop],
            ring_grow_support_mask=(
                None
                if self.ring_grow_support_mask is None
                else tensor_slice(self.ring_grow_support_mask)
            ),
            ring_grow_support_sparse=(
                None
                if self.ring_grow_support_sparse is None
                else self.ring_grow_support_sparse.subrows(start, stop)
            ),
            ring_delete_actions=(
                None
                if self.ring_delete_actions is None
                else self.ring_delete_actions[start:stop]
            ),
            ring_grow_support_is_exact=flag_slice(self.ring_grow_support_is_exact),
            ring_grow_enablement_is_exact=flag_slice(
                self.ring_grow_enablement_is_exact
            ),
        )

    def to(self, device: torch.device, *, non_blocking: bool = False) -> "FactorizedMarkBatch":
        def move(value: Tensor) -> Tensor:
            return value.to(device=device, non_blocking=non_blocking)

        return FactorizedMarkBatch(
            states=self.states,
            atom_types=move(self.atom_types),
            formal_charges=move(self.formal_charges),
            implicit_h_counts=move(self.implicit_h_counts),
            bonds=move(self.bonds),
            neural_bonds=move(self.neural_bonds),
            times=move(self.times),
            atom_topology=move(self.atom_topology),
            closure_topology=move(self.closure_topology),
            ring_system_topology=move(self.ring_system_topology),
            atom_delete_mask=move(self.atom_delete_mask),
            cycle_edge_mask=move(self.cycle_edge_mask),
            cyclic_pair_mask=move(self.cyclic_pair_mask),
            graft_mask=move(self.graft_mask),
            graft_remove_neighbors=move(self.graft_remove_neighbors),
            graft_successor_groups=move(self.graft_successor_groups),
            teacher_actions=self.teacher_actions,
            teacher_rule_names=self.teacher_rule_names,
            teacher_rates=move(self.teacher_rates),
            importance_weights=move(self.importance_weights),
            ring_restate_actions=self.ring_restate_actions,
            ring_grow_support_mask=(
                None if self.ring_grow_support_mask is None else move(self.ring_grow_support_mask)
            ),
            # CSR support stays on CPU and is materialized directly on the
            # destination device only when the model builds its action table.
            ring_grow_support_sparse=self.ring_grow_support_sparse,
            ring_delete_actions=self.ring_delete_actions,
            # Exactness flags are control-flow metadata, not model inputs.
            # Keeping them on CPU avoids a CUDA-to-Python synchronization in
            # the per-row objective-aware support routing below.
            ring_grow_support_is_exact=self.ring_grow_support_is_exact,
            ring_grow_enablement_is_exact=self.ring_grow_enablement_is_exact,
        )

    def pin_memory(self) -> "FactorizedMarkBatch":
        def pin(value: Tensor) -> Tensor:
            return value.pin_memory()

        return FactorizedMarkBatch(
            states=self.states,
            atom_types=pin(self.atom_types),
            formal_charges=pin(self.formal_charges),
            implicit_h_counts=pin(self.implicit_h_counts),
            bonds=pin(self.bonds),
            neural_bonds=pin(self.neural_bonds),
            times=pin(self.times),
            atom_topology=pin(self.atom_topology),
            closure_topology=pin(self.closure_topology),
            ring_system_topology=pin(self.ring_system_topology),
            atom_delete_mask=pin(self.atom_delete_mask),
            cycle_edge_mask=pin(self.cycle_edge_mask),
            cyclic_pair_mask=pin(self.cyclic_pair_mask),
            graft_mask=pin(self.graft_mask),
            graft_remove_neighbors=pin(self.graft_remove_neighbors),
            graft_successor_groups=pin(self.graft_successor_groups),
            teacher_actions=self.teacher_actions,
            teacher_rule_names=self.teacher_rule_names,
            teacher_rates=pin(self.teacher_rates),
            importance_weights=pin(self.importance_weights),
            ring_restate_actions=self.ring_restate_actions,
            ring_grow_support_mask=(
                None if self.ring_grow_support_mask is None else pin(self.ring_grow_support_mask)
            ),
            ring_grow_support_sparse=(
                None
                if self.ring_grow_support_sparse is None
                else self.ring_grow_support_sparse.pin_memory()
            ),
            ring_delete_actions=self.ring_delete_actions,
            ring_grow_support_is_exact=(
                pin(self.ring_grow_support_is_exact)
                if isinstance(self.ring_grow_support_is_exact, Tensor)
                else self.ring_grow_support_is_exact
            ),
            ring_grow_enablement_is_exact=(
                pin(self.ring_grow_enablement_is_exact)
                if isinstance(self.ring_grow_enablement_is_exact, Tensor)
                else self.ring_grow_enablement_is_exact
            ),
        )


@dataclass(frozen=True)
class FactorizedMarkPrediction:
    total_hazard: Tensor
    selected_mark_log_probability: Tensor
    family_log_probabilities: Tensor
    enabled_families: Tensor

    @property
    def selected_mark_rate(self) -> Tensor:
        return self.total_hazard * self.selected_mark_log_probability.exp()


@dataclass(frozen=True)
class SampledRewriteMark:
    total_hazard: float
    rule_name: str
    action: Any


def prepare_factorized_mark_batch(
    states: tuple[MolecularGraph, ...],
    times: tuple[float, ...],
    teacher_actions: tuple[Any | None, ...],
    teacher_rule_names: tuple[str | None, ...],
    teacher_rates: tuple[float, ...],
    importance_weights: tuple[float, ...] | None = None,
    *,
    use_aromatic_bond_view: bool = True,
    ring_catalog: TypedRingCatalog | None = None,
    chemistry_feature_cache: MutableMapping[
        tuple[int, bool, StateCacheKey], ChemistryStateFeatures
    ]
    | None = None,
    chemistry_feature_cache_limit: int = 2048,
    compute_ring_grow_support: bool = True,
) -> FactorizedMarkBatch:
    """Collate valid states and cheap graph-theoretic application conditions."""

    count = len(states)
    if not count:
        raise ValueError("a factorized mark batch must be non-empty")
    lengths = {
        len(times),
        len(teacher_actions),
        len(teacher_rule_names),
        len(teacher_rates),
    }
    if lengths != {count}:
        raise ValueError("factorized mark batch fields have inconsistent lengths")
    n_slots = states[0].n_atoms
    if any(state.n_atoms != n_slots for state in states):
        raise ValueError("all factorized mark states must use the same slot count")
    weights = importance_weights or (1.0,) * count
    if len(weights) != count:
        raise ValueError("importance weights have the wrong length")
    if chemistry_feature_cache_limit <= 0:
        raise ValueError("chemistry feature cache limit must be positive")

    atom_topology = []
    closure_topology = []
    ring_system_topology = []
    delete_masks = []
    cycle_edge_masks = []
    cyclic_pair_masks = []
    graft_masks = []
    graft_remove_neighbors = []
    graft_successor_groups = []
    neural_bonds = []
    restate_actions = []
    ring_grow_support_masks = []
    ring_delete_actions = []
    ring_system_templates = (
        None
        if ring_catalog is None or not compute_ring_grow_support
        else structured_ring_system_templates(ring_catalog)
    )
    ring_system_template_aliases = (
        None
        if ring_catalog is None or not compute_ring_grow_support
        else structured_ring_system_template_aliases(ring_catalog)
    )
    for state in states:
        feature_key = (
            0 if ring_catalog is None else id(ring_catalog),
            bool(use_aromatic_bond_view),
            molecular_state_cache_key(state),
        )
        features = (
            None
            if chemistry_feature_cache is None
            else chemistry_feature_cache.get(feature_key)
        )
        if features is None:
            atom_topo, closure_topo, ring_system_topo = compute_topology_features(state)
            (
                delete_mask,
                cycle_edge_mask,
                cyclic_pair_mask,
                graft_mask,
                graft_remove_neighbor,
                graft_successor_group,
            ) = _graph_application_masks(state, atom_topo)
            features = ChemistryStateFeatures(
                atom_topology=atom_topo,
                closure_topology=closure_topo,
                ring_system_topology=ring_system_topo,
                atom_delete_mask=delete_mask,
                cycle_edge_mask=cycle_edge_mask,
                cyclic_pair_mask=cyclic_pair_mask,
                graft_mask=graft_mask,
                graft_remove_neighbors=graft_remove_neighbor,
                graft_successor_groups=graft_successor_group,
                neural_bonds=(
                    resonance_invariant_bond_classes(state)
                    if use_aromatic_bond_view
                    else state.bonds
                ),
                ring_delete_actions=(
                    None
                    if ring_catalog is None
                    else enumerate_structured_ring_system_deletes(state, ring_catalog)
                ),
            )
            if chemistry_feature_cache is not None:
                chemistry_feature_cache[feature_key] = features
                while len(chemistry_feature_cache) > chemistry_feature_cache_limit:
                    if isinstance(chemistry_feature_cache, OrderedDict):
                        chemistry_feature_cache.popitem(last=False)
                    else:
                        chemistry_feature_cache.pop(next(iter(chemistry_feature_cache)))
        elif isinstance(chemistry_feature_cache, OrderedDict):
            chemistry_feature_cache.move_to_end(feature_key)
        atom_topology.append(features.atom_topology)
        closure_topology.append(features.closure_topology)
        ring_system_topology.append(features.ring_system_topology)
        delete_masks.append(features.atom_delete_mask)
        cycle_edge_masks.append(features.cycle_edge_mask)
        cyclic_pair_masks.append(features.cyclic_pair_mask)
        graft_masks.append(features.graft_mask)
        graft_remove_neighbors.append(features.graft_remove_neighbors)
        graft_successor_groups.append(features.graft_successor_groups)
        neural_bonds.append(features.neural_bonds)
        # Atomic RingSystemGrow already commits the complete typed electronic
        # state in the production carbon-tree teachers.  Those paths contain
        # no independent RingSystemRestate marks, so enumerating an exhaustive
        # restatement fiber here would consume roughly half of collator time
        # for a family that is always masked out by the current objective.
        restate_actions.append(())
        if ring_catalog is not None and ring_system_templates is not None:
            ring_grow_support_masks.append(
                np.asarray(
                    ring_system_template_local_support_mask(
                        state,
                        ring_system_templates,
                        ring_system_template_aliases,
                    ),
                    dtype=np.bool_,
                )
            )
        if ring_catalog is not None:
            if features.ring_delete_actions is None:
                raise RuntimeError("ring-aware state features lack delete actions")
            ring_delete_actions.append(features.ring_delete_actions)

    return FactorizedMarkBatch(
        states=states,
        atom_types=torch.from_numpy(np.stack([state.atom_types for state in states])).long(),
        formal_charges=torch.from_numpy(
            np.stack([state.formal_charges for state in states])
        ).long(),
        implicit_h_counts=torch.from_numpy(
            np.stack([state.implicit_h_counts for state in states])
        ).long(),
        bonds=torch.from_numpy(np.stack([state.bonds for state in states])).long(),
        neural_bonds=torch.from_numpy(np.stack(neural_bonds)).long(),
        times=torch.tensor(times, dtype=torch.float32),
        atom_topology=torch.from_numpy(np.stack(atom_topology)).long(),
        closure_topology=torch.from_numpy(np.stack(closure_topology)).long(),
        ring_system_topology=torch.from_numpy(np.stack(ring_system_topology)).long(),
        atom_delete_mask=torch.from_numpy(np.stack(delete_masks)).bool(),
        cycle_edge_mask=torch.from_numpy(np.stack(cycle_edge_masks)).bool(),
        cyclic_pair_mask=torch.from_numpy(np.stack(cyclic_pair_masks)).bool(),
        graft_mask=torch.from_numpy(np.stack(graft_masks)).bool(),
        graft_remove_neighbors=torch.from_numpy(np.stack(graft_remove_neighbors)).long(),
        graft_successor_groups=torch.from_numpy(np.stack(graft_successor_groups)).long(),
        teacher_actions=teacher_actions,
        teacher_rule_names=teacher_rule_names,
        teacher_rates=torch.tensor(teacher_rates, dtype=torch.float32),
        importance_weights=torch.tensor(weights, dtype=torch.float32),
        ring_restate_actions=tuple(restate_actions),
        ring_grow_support_mask=(
            None
            if ring_catalog is None or not compute_ring_grow_support
            else torch.from_numpy(np.stack(ring_grow_support_masks)).bool()
        ),
        ring_delete_actions=(None if ring_catalog is None else tuple(ring_delete_actions)),
    )


def _graph_application_masks(
    state: MolecularGraph,
    atom_topology: np.ndarray,
    *,
    optimize_graft_canonicalization: bool = True,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    n_slots = state.n_atoms
    real = tuple(int(v) for v in np.flatnonzero(is_element(state.atom_types)))
    graph = nx.Graph()
    graph.add_nodes_from(real)
    graph.add_edges_from(
        (a, b)
        for offset, a in enumerate(real)
        for b in real[offset + 1 :]
        if int(state.bonds[a, b]) != 0
    )
    bridges = {frozenset((int(a), int(b))) for a, b in nx.bridges(graph)}
    cycle_edges = {frozenset((int(a), int(b))) for a, b in graph.edges()} - bridges
    cycle_edge_mask = np.zeros((n_slots, n_slots), dtype=np.bool_)
    for edge in cycle_edges:
        a, b = tuple(edge)
        cycle_edge_mask[a, b] = cycle_edge_mask[b, a] = True

    cyclic_pair_mask = np.zeros((n_slots, n_slots), dtype=np.bool_)
    for vertices in nx.biconnected_components(graph):
        block = tuple(sorted(int(v) for v in vertices))
        if graph.subgraph(block).number_of_edges() < len(block):
            continue
        for offset, a in enumerate(block):
            for b in block[offset + 1 :]:
                cyclic_pair_mask[a, b] = True

    articulation = set(nx.articulation_points(graph)) if len(real) > 1 else set()
    delete_mask = np.zeros(n_slots, dtype=np.bool_)
    for vertex in real:
        if int(atom_topology[vertex]) != 0 or vertex in articulation:
            continue
        allowed = True
        for neighbor in np.flatnonzero(state.bonds[vertex] != 0):
            order = int(state.bonds[vertex, neighbor])
            delta = int(BOND_CLASS_TO_H_CHANGE[order])
            if int(state.implicit_h_counts[neighbor]) + delta > MAX_H_COUNT:
                allowed = False
                break
        delete_mask[vertex] = allowed
    graft_mask = np.zeros((n_slots, n_slots), dtype=np.bool_)
    graft_remove_neighbor = np.full((n_slots, n_slots), -1, dtype=np.int64)
    graft_successor_group = np.full((n_slots, n_slots), -1, dtype=np.int64)
    all_single_tree = bool(
        real and nx.is_tree(graph) and all(int(state.bonds[a, b]) == 1 for a, b in graph.edges())
    )
    if all_single_tree:
        tree_adjacency = {
            vertex: {int(neighbor) for neighbor in graph.neighbors(vertex)}
            for vertex in real
        }
        first_hop = np.full((n_slots, n_slots), -1, dtype=np.int64)
        for moved in real:
            for neighbor in tree_adjacency[moved]:
                stack = [(neighbor, moved)]
                while stack:
                    vertex, parent = stack.pop()
                    first_hop[moved, vertex] = neighbor
                    stack.extend(
                        (child, vertex)
                        for child in tree_adjacency[vertex]
                        if child != parent
                    )
        tree_codebook: dict[tuple[Any, ...], int] = {}
        canonicalizer: _IncrementalColoredTreeCanonicalizer | None = None
        use_incremental_keys = False
        use_orbit_pruning = False
        if optimize_graft_canonicalization and len(real) >= 32:
            candidate_canonicalizer = _IncrementalColoredTreeCanonicalizer(
                tree_adjacency,
                state.atom_types,
                state.formal_charges,
                state.implicit_h_counts,
                codebook=tree_codebook,
            )
            vertex_orbit_count = candidate_canonicalizer.vertex_orbit_count()
            use_incremental_keys = 4 * vertex_orbit_count <= 3 * len(real)
            use_orbit_pruning = use_incremental_keys
            if use_incremental_keys:
                canonicalizer = candidate_canonicalizer
                current_key = canonicalizer.current_key()
            else:
                current_key = _canonical_colored_tree_key_from_adjacency(
                    tree_adjacency,
                    state.atom_types,
                    state.formal_charges,
                    state.implicit_h_counts,
                    codebook=tree_codebook,
                )
        else:
            current_key = _canonical_colored_tree_key_from_adjacency(
                tree_adjacency,
                state.atom_types,
                state.formal_charges,
                state.implicit_h_counts,
                codebook=tree_codebook,
            )
        successor_groups: dict[tuple[int, ...], int] = {}
        single_h_delta = int(BOND_CLASS_TO_H_CHANGE[1])
        candidates: list[tuple[int, int, int]] = []
        for moved in real:
            for target in real:
                if moved == target or target in tree_adjacency[moved]:
                    continue
                removed_neighbor = int(first_hop[moved, target])
                if removed_neighbor < 0:
                    raise RuntimeError("tree first-hop table is incomplete")
                if int(state.implicit_h_counts[target]) < 1:
                    continue
                if int(state.implicit_h_counts[removed_neighbor]) >= MAX_H_COUNT:
                    continue
                candidates.append((moved, target, removed_neighbor))

        candidate_groups: list[list[tuple[int, int, int]]]
        if use_orbit_pruning:
            if canonicalizer is None:
                raise RuntimeError("Graft orbit pruning lacks a canonicalizer")
            orbit_groups: OrderedDict[
                tuple[Any, ...], list[tuple[int, int, int]]
            ] = OrderedDict()
            for candidate in candidates:
                moved, target, _ = candidate
                orbit_groups.setdefault(
                    canonicalizer.ordered_pair_orbit_key(moved, target),
                    [],
                ).append(candidate)
            candidate_groups = list(orbit_groups.values())
        else:
            candidate_groups = [[candidate] for candidate in candidates]

        for aliases in candidate_groups:
            moved, target, removed_neighbor = aliases[0]
            if use_incremental_keys:
                if canonicalizer is None:
                    raise RuntimeError("incremental Graft keys lack a canonicalizer")
                successor_key = canonicalizer.graft_successor_key(
                    moved,
                    removed_neighbor,
                    target,
                    hydrogen_delta=single_h_delta,
                )
            else:
                working_hydrogens = state.implicit_h_counts.copy()
                tree_adjacency[moved].remove(removed_neighbor)
                tree_adjacency[removed_neighbor].remove(moved)
                tree_adjacency[moved].add(target)
                tree_adjacency[target].add(moved)
                working_hydrogens[removed_neighbor] += single_h_delta
                working_hydrogens[target] -= single_h_delta
                successor_key = _canonical_colored_tree_key_from_adjacency(
                    tree_adjacency,
                    state.atom_types,
                    state.formal_charges,
                    working_hydrogens,
                    codebook=tree_codebook,
                )
                tree_adjacency[moved].remove(target)
                tree_adjacency[target].remove(moved)
                tree_adjacency[moved].add(removed_neighbor)
                tree_adjacency[removed_neighbor].add(moved)
                # Slot permutations and automorphisms are gauge, not jumps of
                # the molecular CTMC.  Removing this group prevents a learned
                # high-rate sequence of visible events that leaves the
                # canonical molecule unchanged.
            if successor_key == current_key:
                continue
            group = successor_groups.setdefault(
                successor_key,
                len(successor_groups),
            )
            for moved, target, removed_neighbor in aliases:
                graft_mask[moved, target] = True
                graft_remove_neighbor[moved, target] = removed_neighbor
                graft_successor_group[moved, target] = group
    return (
        delete_mask,
        cycle_edge_mask,
        cyclic_pair_mask,
        graft_mask,
        graft_remove_neighbor,
        graft_successor_group,
    )


def _canonical_colored_tree_key_from_adjacency(
    adjacency: dict[int, set[int]],
    atom_types: np.ndarray,
    formal_charges: np.ndarray,
    implicit_h_counts: np.ndarray,
    *,
    codebook: dict[tuple[Any, ...], int] | None = None,
) -> tuple[int, ...]:
    """Return an exact isomorphism key for a node-colored molecular tree.

    Graft support is restricted to single-bond trees, so the AHU tree code is
    both exact and substantially cheaper than materializing and canonicalizing
    every candidate through RDKit.  The node color includes the full discrete
    atom state affected by a Graft.
    """

    if not adjacency:
        raise ValueError("colored-tree canonicalization requires a non-empty tree")
    if sum(len(neighbors) for neighbors in adjacency.values()) != 2 * (len(adjacency) - 1):
        raise ValueError("colored-tree canonicalization requires tree adjacency")

    remaining = set(adjacency)
    degree = {vertex: len(adjacency[vertex]) for vertex in remaining}
    leaves = sorted(vertex for vertex in remaining if degree[vertex] <= 1)
    while len(remaining) > 2:
        if not leaves:
            raise RuntimeError("tree-center peeling found no leaves")
        next_leaves: list[int] = []
        for leaf in leaves:
            if leaf not in remaining:
                continue
            remaining.remove(leaf)
            for neighbor in adjacency[leaf]:
                if neighbor not in remaining:
                    continue
                degree[neighbor] -= 1
                if degree[neighbor] == 1:
                    next_leaves.append(neighbor)
        leaves = sorted(set(next_leaves))
    centers = tuple(sorted(remaining))

    interned = {} if codebook is None else codebook

    def rooted_code(vertex: int, parent: int | None) -> int:
        children = sorted(
            rooted_code(neighbor, vertex)
            for neighbor in adjacency[vertex]
            if neighbor != parent
        )
        color = (
            int(atom_types[vertex]),
            int(formal_charges[vertex]),
            int(implicit_h_counts[vertex]),
        )
        descriptor = (*color, tuple(children))
        identifier = interned.get(descriptor)
        if identifier is None:
            identifier = len(interned) + 1
            interned[descriptor] = identifier
        return identifier

    if len(centers) == 1:
        return 1, rooted_code(centers[0], None)
    if len(centers) != 2:
        raise RuntimeError("a tree must have one or two centers")
    left, right = centers
    halves = sorted((rooted_code(left, right), rooted_code(right, left)))
    return 2, halves[0], halves[1]


class _IncrementalColoredTreeCanonicalizer:
    """Exact colored-tree keys with orbit pruning and branch reuse for Grafts.

    A Graft changes two edges and the hydrogen colors of two vertices.  The
    reference implementation rebuilt every rooted subtree for every labeled
    ``(moved, target)`` coordinate.  This helper memoizes the original directed
    branch descriptions and reuses every branch whose cut-side component does
    not contain a changed vertex.  It also gives an exact ordered-pair orbit
    key: the oriented moved-to-target path decorated by its off-path branches.
    """

    def __init__(
        self,
        adjacency: dict[int, set[int]],
        atom_types: np.ndarray,
        formal_charges: np.ndarray,
        implicit_h_counts: np.ndarray,
        *,
        codebook: dict[tuple[Any, ...], int],
    ) -> None:
        self.adjacency = {
            int(vertex): frozenset(int(neighbor) for neighbor in neighbors)
            for vertex, neighbors in adjacency.items()
        }
        self.colors = {
            int(vertex): (
                int(atom_types[vertex]),
                int(formal_charges[vertex]),
                int(implicit_h_counts[vertex]),
            )
            for vertex in adjacency
        }
        self.codebook = codebook
        self._base_descriptors: dict[tuple[int, int], tuple[Any, ...]] = {}
        self._base_identifiers: dict[tuple[int, int], int] = {}
        self._side_vertices: dict[tuple[int, int], frozenset[int]] = {}

    @staticmethod
    def _centers(adjacency: dict[int, frozenset[int]]) -> tuple[int, ...]:
        remaining = set(adjacency)
        degree = {vertex: len(adjacency[vertex]) for vertex in remaining}
        leaves = [vertex for vertex in remaining if degree[vertex] <= 1]
        while len(remaining) > 2:
            if not leaves:
                raise RuntimeError("tree-center peeling found no leaves")
            next_leaves: list[int] = []
            for leaf in leaves:
                if leaf not in remaining:
                    continue
                remaining.remove(leaf)
                for neighbor in adjacency[leaf]:
                    if neighbor not in remaining:
                        continue
                    degree[neighbor] -= 1
                    if degree[neighbor] == 1:
                        next_leaves.append(neighbor)
            leaves = next_leaves
        return tuple(sorted(remaining))

    def _base_descriptor(self, vertex: int, parent: int) -> tuple[Any, ...]:
        key = (int(vertex), int(parent))
        cached = self._base_descriptors.get(key)
        if cached is not None:
            return cached
        children = tuple(
            sorted(
                self._base_descriptor(neighbor, vertex)
                for neighbor in self.adjacency[vertex]
                if neighbor != parent
            )
        )
        descriptor = (*self.colors[vertex], children)
        self._base_descriptors[key] = descriptor
        return descriptor

    def _intern_descriptor(self, descriptor: tuple[Any, ...]) -> int:
        atom_type, charge, hydrogens, children = descriptor
        child_ids = tuple(
            sorted(self._intern_descriptor(child) for child in children)
        )
        compact = (int(atom_type), int(charge), int(hydrogens), child_ids)
        identifier = self.codebook.get(compact)
        if identifier is None:
            identifier = len(self.codebook) + 1
            self.codebook[compact] = identifier
        return identifier

    def _base_identifier(self, vertex: int, parent: int) -> int:
        key = (int(vertex), int(parent))
        cached = self._base_identifiers.get(key)
        if cached is None:
            cached = self._intern_descriptor(self._base_descriptor(vertex, parent))
            self._base_identifiers[key] = cached
        return cached

    def _key_from_descriptors(
        self,
        descriptors: tuple[tuple[Any, ...], ...],
    ) -> tuple[int, ...]:
        identifiers = tuple(sorted(self._intern_descriptor(item) for item in descriptors))
        if len(identifiers) == 1:
            return 1, identifiers[0]
        if len(identifiers) == 2:
            return 2, identifiers[0], identifiers[1]
        raise RuntimeError("a tree must have one or two centers")

    def current_key(self) -> tuple[int, ...]:
        centers = self._centers(self.adjacency)
        if len(centers) == 1:
            descriptors = (self._base_descriptor(centers[0], -1),)
        else:
            left, right = centers
            descriptors = (
                self._base_descriptor(left, right),
                self._base_descriptor(right, left),
            )
        return self._key_from_descriptors(descriptors)

    def vertex_orbit_count(self) -> int:
        """Return the exact number of color-preserving vertex orbits."""

        rooted_ids = [
            self._base_identifier(vertex, -1)
            for vertex in self.adjacency
        ]
        return len(set(rooted_ids))

    def _path(self, source: int, target: int) -> tuple[int, ...]:
        parents = {int(source): -1}
        stack = [int(source)]
        while stack:
            vertex = stack.pop()
            if vertex == target:
                break
            for neighbor in self.adjacency[vertex]:
                if neighbor in parents:
                    continue
                parents[neighbor] = vertex
                stack.append(neighbor)
        if target not in parents:
            raise RuntimeError("colored-tree pair has no connecting path")
        reverse_path = [int(target)]
        while reverse_path[-1] != source:
            reverse_path.append(parents[reverse_path[-1]])
        return tuple(reversed(reverse_path))

    def ordered_pair_orbit_key(self, moved: int, target: int) -> tuple[Any, ...]:
        """Return an exact automorphism-orbit key for an ordered vertex pair."""

        path = self._path(int(moved), int(target))
        decorated_path = []
        for index, vertex in enumerate(path):
            excluded = {
                path[index - 1] if index > 0 else -1,
                path[index + 1] if index + 1 < len(path) else -1,
            }
            off_path = tuple(
                sorted(
                    self._base_identifier(neighbor, vertex)
                    for neighbor in self.adjacency[vertex]
                    if neighbor not in excluded
                )
            )
            decorated_path.append((*self.colors[vertex], off_path))
        return tuple(decorated_path)

    def _original_side_vertices(self, vertex: int, parent: int) -> frozenset[int]:
        key = (int(vertex), int(parent))
        cached = self._side_vertices.get(key)
        if cached is not None:
            return cached
        seen = {int(parent)}
        stack = [int(vertex)]
        side: set[int] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            side.add(current)
            stack.extend(self.adjacency[current] - seen)
        result = frozenset(side)
        self._side_vertices[key] = result
        return result

    def graft_successor_key(
        self,
        moved: int,
        removed_neighbor: int,
        target: int,
        *,
        hydrogen_delta: int,
    ) -> tuple[int, ...]:
        moved = int(moved)
        removed_neighbor = int(removed_neighbor)
        target = int(target)
        if removed_neighbor not in self.adjacency[moved]:
            raise ValueError("Graft cut edge is absent from the source tree")
        if target in self.adjacency[moved] or target == moved:
            raise ValueError("Graft target must be a non-neighbor")

        successor = dict(self.adjacency)
        successor[moved] = (self.adjacency[moved] - {removed_neighbor}) | {target}
        successor[removed_neighbor] = self.adjacency[removed_neighbor] - {moved}
        successor[target] = self.adjacency[target] | {moved}
        affected = frozenset((moved, removed_neighbor, target))
        color_overrides = {
            removed_neighbor: (
                self.colors[removed_neighbor][0],
                self.colors[removed_neighbor][1],
                self.colors[removed_neighbor][2] + int(hydrogen_delta),
            ),
            target: (
                self.colors[target][0],
                self.colors[target][1],
                self.colors[target][2] - int(hydrogen_delta),
            ),
        }
        removed_edge = frozenset((moved, removed_neighbor))
        memo: dict[tuple[int, int], tuple[Any, ...]] = {}

        def rooted(vertex: int, parent: int) -> tuple[Any, ...]:
            key = (int(vertex), int(parent))
            cached = memo.get(key)
            if cached is not None:
                return cached
            original_edge = parent in self.adjacency[vertex]
            if (
                parent >= 0
                and original_edge
                and frozenset((vertex, parent)) != removed_edge
                and self._original_side_vertices(vertex, parent).isdisjoint(affected)
            ):
                descriptor = self._base_descriptor(vertex, parent)
                memo[key] = descriptor
                return descriptor
            children = tuple(
                sorted(
                    rooted(neighbor, vertex)
                    for neighbor in successor[vertex]
                    if neighbor != parent
                )
            )
            descriptor = (*color_overrides.get(vertex, self.colors[vertex]), children)
            memo[key] = descriptor
            return descriptor

        centers = self._centers(successor)
        if len(centers) == 1:
            descriptors = (rooted(centers[0], -1),)
        else:
            left, right = centers
            descriptors = (rooted(left, right), rooted(right, left))
        return self._key_from_descriptors(descriptors)


class FactorizedTraceletRateModel(nn.Module):
    """Batched hierarchical rates over stochastic rewrite-rule matches."""

    def __init__(
        self,
        ring_catalog: TypedRingCatalog,
        *,
        hidden_dim: int = 128,
        message_passing_steps: int = 4,
        mark_dim: int = 32,
        ring_electronic_mode: str = "factorized_local",
        ring_candidate_cache_limit: int = 32768,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or message_passing_steps <= 0 or mark_dim <= 0:
            raise ValueError("model dimensions and message-passing steps must be positive")
        if ring_candidate_cache_limit <= 0:
            raise ValueError("ring candidate cache limit must be positive")
        self.ring_catalog = ring_catalog
        self.hidden_dim = int(hidden_dim)
        self.message_passing_steps = int(message_passing_steps)
        self.mark_dim = int(mark_dim)
        if ring_electronic_mode not in {"factorized_local", "catalog_exact"}:
            raise ValueError("unknown ring electronic decoding mode")
        self.ring_electronic_mode = str(ring_electronic_mode)
        self._sampling_state_cache: OrderedDict[
            tuple[bytes, bytes, bytes, bytes],
            FactorizedMarkBatch,
        ] = OrderedDict()
        self._sampling_state_cache_limit = 4096
        self._ring_grow_support_cache: OrderedDict[
            tuple[bytes, bytes, bytes, bytes],
            tuple[bool, ...],
        ] = OrderedDict()
        self._ring_grow_enablement_certificate_cache: OrderedDict[
            tuple[bytes, bytes, bytes, bytes],
            tuple[bool, ...],
        ] = OrderedDict()
        self._ring_template_placement_group_cache: OrderedDict[
            tuple[tuple[bytes, bytes, bytes, bytes], int],
            tuple[tuple[RingSystemPlacement, ...], ...],
        ] = OrderedDict()
        self._ring_semantic_decoder_cache: OrderedDict[
            tuple[tuple[bytes, bytes, bytes, bytes], tuple],
            SemanticRingSystemDecoder,
        ] = OrderedDict()
        self._ring_delete_candidate_cache: OrderedDict[
            tuple[bytes, bytes, bytes, bytes],
            tuple[RingSystemDelete, ...],
        ] = OrderedDict()
        self._ring_paired_candidate_cache: OrderedDict[
            tuple[tuple[bytes, bytes, bytes, bytes], int],
            tuple[ExecutableRingGrowCandidate, ...],
        ] = OrderedDict()
        self._ring_candidate_cache_limit = int(ring_candidate_cache_limit)

        self.cycle_templates = tuple(ring_catalog.cycle_templates)
        self.attach_templates = tuple(ring_catalog.attach_templates)
        self.ring_system_templates = structured_ring_system_templates(ring_catalog)
        self.ring_system_template_aliases = structured_ring_system_template_aliases(ring_catalog)
        if len(self.ring_system_template_aliases) != len(self.ring_system_templates):
            raise RuntimeError("ring-system alias groups do not align with templates")
        self.ring_system_electronic_witness_aliases = (
            structured_ring_system_electronic_aliases(ring_catalog)
            if (
                self.ring_electronic_mode == "catalog_exact"
                or int(getattr(ring_catalog, "ring_system_electronic_alias_version", 0)) >= 1
            )
            else None
        )
        self.ring_system_electronic_aliases = (
            self.ring_system_electronic_witness_aliases
            if self.ring_electronic_mode == "catalog_exact"
            else None
        )
        if (
            self.ring_system_electronic_witness_aliases is not None
            and len(self.ring_system_electronic_witness_aliases)
            != len(self.ring_system_templates)
        ):
            raise RuntimeError("ring electronic witness aliases do not align with templates")
        raw_ring_counts = tuple(
            int(count) for count in getattr(ring_catalog, "ring_system_template_counts", ())
        )
        if len(raw_ring_counts) != len(ring_catalog.ring_system_templates):
            raw_ring_counts = tuple(1 for _ in ring_catalog.ring_system_templates)
        count_by_template = dict(zip(ring_catalog.ring_system_templates, raw_ring_counts))
        semantic_ring_counts = torch.tensor(
            tuple(
                1.0 + sum(count_by_template[alias] for alias in aliases)
                for aliases in self.ring_system_template_aliases
            ),
            dtype=torch.float32,
        )
        semantic_ring_log_prior = semantic_ring_counts.log()
        semantic_ring_log_prior -= torch.logsumexp(
            semantic_ring_log_prior,
            dim=0,
        )
        self.register_buffer(
            "ring_system_template_log_prior",
            semantic_ring_log_prior,
            persistent=False,
        )
        self._cycle_to_index = {
            template: index for index, template in enumerate(self.cycle_templates)
        }
        self._attach_to_index = {
            template: index for index, template in enumerate(self.attach_templates)
        }

        self.atom_embedding = nn.Embedding(M, hidden_dim)
        self.charge_embedding = nn.Embedding(K, hidden_dim)
        self.hydrogen_embedding = nn.Embedding(H_COUNT_CLASSES, hidden_dim)
        self.bond_embedding = nn.Embedding(BOND_CLASSES, hidden_dim)
        self.atom_ring_embedding = nn.Embedding(RING_SIZE_BUCKETS, hidden_dim)
        self.closure_ring_embedding = nn.Embedding(RING_SIZE_BUCKETS, hidden_dim)
        self.ring_system_embedding = nn.Embedding(FUSED_SIZE_BUCKETS, hidden_dim)
        self.empty_state = nn.Parameter(torch.zeros(hidden_dim))
        self.time_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.message_node = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.update_networks = nn.ModuleList(
            nn.Sequential(
                nn.Linear(3 * hidden_dim, 2 * hidden_dim),
                nn.SiLU(),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            for _ in range(message_passing_steps)
        )
        self.update_norms = nn.ModuleList(
            nn.LayerNorm(hidden_dim) for _ in range(message_passing_steps)
        )
        self.structure_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.global_project = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pair_project = nn.Sequential(
            nn.Linear(5 * hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

        self.total_hazard_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.family_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(MARK_RULE_NAMES)),
        )
        self.grow_root_head = nn.Linear(hidden_dim, len(CNOF_ATOM_TYPES))
        self.grow_query = nn.Linear(hidden_dim, mark_dim)
        self.grow_option = nn.Embedding(3 * len(CNOF_ATOM_TYPES), mark_dim)
        self.delete_head = nn.Linear(hidden_dim, 1)
        self.restate_head = nn.Linear(hidden_dim, len(CNOF_ATOM_TYPES))
        self.reorder_head = nn.Linear(hidden_dim, 3)
        self.graft_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.cycle_query = nn.Linear(hidden_dim, mark_dim)
        self.cycle_key = nn.Embedding(max(len(self.cycle_templates), 1), mark_dim)
        self.attach_query = nn.Linear(hidden_dim, mark_dim)
        self.attach_key = nn.Embedding(max(len(self.attach_templates), 1), mark_dim)
        self.ring_system_template_query = nn.Linear(hidden_dim, mark_dim)
        self.ring_system_template_key = nn.Embedding(
            max(len(self.ring_system_templates), 1),
            mark_dim,
        )
        self.ring_system_grow_head = nn.Sequential(
            nn.Linear(4 * hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, 1),
        )
        self.ring_system_atom_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(CNOF_ATOM_TYPES)),
        )
        self.ring_system_role_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.ring_system_delete_head = nn.Sequential(
            nn.Linear(4 * hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, 1),
        )
        self.restate_order_embedding = nn.Embedding(4, hidden_dim)
        self.ring_restate_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.register_buffer(
            "cnof_valences",
            torch.tensor([CNOF_VALENCE[item] for item in CNOF_ATOM_TYPES]),
            persistent=False,
        )
        self.register_buffer(
            "cycle_spans",
            torch.tensor([template.span for template in self.cycle_templates]),
            persistent=False,
        )
        self.register_buffer(
            "attach_spans",
            torch.tensor([template.span for template in self.attach_templates]),
            persistent=False,
        )
        self.register_buffer(
            "attach_required_h",
            torch.tensor(
                [
                    int(BOND_CLASS_TO_H_CHANGE[template.attachment_order])
                    for template in self.attach_templates
                ]
            ),
            persistent=False,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def clear_ring_candidate_caches(self) -> None:
        """Release state-dependent chemistry caches without changing rates."""

        self._ring_grow_support_cache.clear()
        self._ring_grow_enablement_certificate_cache.clear()
        self._ring_template_placement_group_cache.clear()
        self._ring_semantic_decoder_cache.clear()
        self._ring_delete_candidate_cache.clear()
        self._ring_paired_candidate_cache.clear()

    @staticmethod
    def _state_cache_key(
        state: MolecularGraph,
    ) -> tuple[bytes, bytes, bytes, bytes]:
        return molecular_state_cache_key(state)

    def _ring_grow_support(self, state: MolecularGraph) -> tuple[bool, ...]:
        key = self._state_cache_key(state)
        cached = self._ring_grow_support_cache.get(key)
        if cached is None:
            if self.ring_system_electronic_aliases is None:
                coarse_support = ring_system_template_local_support_mask(
                    state,
                    self.ring_system_templates,
                    self.ring_system_template_aliases,
                )
                support = np.zeros_like(coarse_support)
                witness_support = (
                    ring_system_electronic_template_support_mask(
                        state,
                        self.ring_system_templates,
                        self.ring_system_electronic_witness_aliases,
                    )
                    if self.ring_system_electronic_witness_aliases is not None
                    else np.zeros_like(coarse_support)
                )
                for template_index in np.flatnonzero(coarse_support):
                    template_index = int(template_index)
                    # Catalog assignments are positive witnesses only.  A
                    # verified candidate proves semantic support immediately;
                    # absence never removes the template because the catalog
                    # is not the generative vocabulary.  The exhaustive
                    # semantic decoder below remains the completeness fallback.
                    if witness_support[template_index] and self._ring_witness_candidates(
                        state,
                        template_index,
                    ):
                        support[template_index] = True
                        continue
                    placement_groups = self._ring_template_placement_groups(
                        state,
                        template_index,
                    )
                    support[template_index] = any(
                        semantic_ring_prefix_is_completable(
                            self._ring_semantic_decoder(state, group[0]),
                            (),
                        )
                        for group in placement_groups
                    )
            else:
                support = ring_system_electronic_template_support_mask(
                    state,
                    self.ring_system_templates,
                    self.ring_system_electronic_aliases,
                )
            cached = tuple(bool(item) for item in support)
            self._ring_grow_support_cache[key] = cached
            if len(self._ring_grow_support_cache) > self._ring_candidate_cache_limit:
                self._ring_grow_support_cache.popitem(last=False)
            certificate = [False] * len(cached)
            first_supported = next(
                (index for index, supported in enumerate(cached) if supported),
                None,
            )
            if first_supported is not None:
                certificate[first_supported] = True
            self._ring_grow_enablement_certificate_cache[key] = tuple(certificate)
            if (
                len(self._ring_grow_enablement_certificate_cache)
                > self._ring_candidate_cache_limit
            ):
                self._ring_grow_enablement_certificate_cache.popitem(last=False)
        else:
            self._ring_grow_support_cache.move_to_end(key)
        return cached

    def _ring_grow_enablement_certificate(
        self,
        state: MolecularGraph,
    ) -> tuple[bool, ...]:
        """Return one verified template iff the ring-grow family is enabled.

        Generator Matching needs the complete within-family partition only on
        rows whose teacher is ``ring_system_grow``.  Every other row needs the
        exact Boolean application condition for the family softmax, so this
        method stops at the first executor-supported template.  An all-false
        result is exhaustive and therefore also an exact disabled certificate.
        """

        key = self._state_cache_key(state)
        cached = self._ring_grow_enablement_certificate_cache.get(key)
        if cached is not None:
            self._ring_grow_enablement_certificate_cache.move_to_end(key)
            return cached
        full_support = self._ring_grow_support_cache.get(key)
        if full_support is not None:
            self._ring_grow_support_cache.move_to_end(key)
            certificate = [False] * len(full_support)
            first_supported = next(
                (index for index, supported in enumerate(full_support) if supported),
                None,
            )
            if first_supported is not None:
                certificate[first_supported] = True
            cached = tuple(certificate)
        elif self.ring_system_electronic_aliases is not None:
            support = ring_system_electronic_template_support_mask(
                state,
                self.ring_system_templates,
                self.ring_system_electronic_aliases,
            )
            certificate = np.zeros_like(support)
            supported = np.flatnonzero(support)
            if len(supported):
                certificate[int(supported[0])] = True
            cached = tuple(bool(item) for item in certificate)
        else:
            coarse_support = ring_system_template_local_support_mask(
                state,
                self.ring_system_templates,
                self.ring_system_template_aliases,
            )
            witness_support = (
                ring_system_electronic_template_support_mask(
                    state,
                    self.ring_system_templates,
                    self.ring_system_electronic_witness_aliases,
                )
                if self.ring_system_electronic_witness_aliases is not None
                else np.zeros_like(coarse_support)
            )
            certificate = np.zeros_like(coarse_support)
            witness_indices = np.flatnonzero(coarse_support & witness_support)
            for raw_index in witness_indices:
                template_index = int(raw_index)
                if self._ring_witness_candidates(state, template_index):
                    certificate[template_index] = True
                    break
            if not bool(certificate.any()):
                for raw_index in np.flatnonzero(coarse_support):
                    template_index = int(raw_index)
                    placement_groups = self._ring_template_placement_groups(
                        state,
                        template_index,
                    )
                    if any(
                        semantic_ring_prefix_is_completable(
                            self._ring_semantic_decoder(state, group[0]),
                            (),
                        )
                        for group in placement_groups
                    ):
                        certificate[template_index] = True
                        break
            cached = tuple(bool(item) for item in certificate)
            if not any(cached):
                # The certificate exhausted every coarse-supported template,
                # so the complete support is known to be empty as well.
                self._ring_grow_support_cache[key] = cached
                if len(self._ring_grow_support_cache) > self._ring_candidate_cache_limit:
                    self._ring_grow_support_cache.popitem(last=False)
        self._ring_grow_enablement_certificate_cache[key] = cached
        if (
            len(self._ring_grow_enablement_certificate_cache)
            > self._ring_candidate_cache_limit
        ):
            self._ring_grow_enablement_certificate_cache.popitem(last=False)
        return cached

    def _ring_template_placement_groups(
        self,
        state: MolecularGraph,
        template_index: int,
    ) -> tuple[tuple[RingSystemPlacement, ...], ...]:
        state_key = self._state_cache_key(state)
        key = (state_key, int(template_index))
        cached = self._ring_template_placement_group_cache.get(key)
        if cached is None:
            placements = {
                placement
                for template in self.ring_system_template_aliases[int(template_index)]
                for placement in enumerate_ring_system_template_placements(
                    state,
                    template,
                )
            }
            grouped: dict[tuple, list[RingSystemPlacement]] = {}
            for placement in placements:
                grouped.setdefault(ring_system_placement_key(placement), []).append(placement)
            cached = tuple(
                (
                    min(
                        group,
                        key=lambda placement: (
                            placement.system_atoms,
                            placement.topology_class,
                            placement.scaffold_bonds,
                            placement.bond_reorders,
                            placement.bond_insertions,
                            placement.aromatic_edges,
                        ),
                    ),
                )
                for _, group in sorted(grouped.items(), key=lambda item: item[0])
            )
            self._ring_template_placement_group_cache[key] = cached
            if (
                len(self._ring_template_placement_group_cache)
                > self._ring_candidate_cache_limit
            ):
                self._ring_template_placement_group_cache.popitem(last=False)
        else:
            self._ring_template_placement_group_cache.move_to_end(key)
        return cached

    def _ring_semantic_decoder(
        self,
        state: MolecularGraph,
        placement: RingSystemPlacement,
    ) -> SemanticRingSystemDecoder:
        state_key = self._state_cache_key(state)
        key = (state_key, ring_system_placement_key(placement))
        cached = self._ring_semantic_decoder_cache.get(key)
        if cached is None:
            cached = build_semantic_ring_system_decoder(state, placement)
            self._ring_semantic_decoder_cache[key] = cached
            if len(self._ring_semantic_decoder_cache) > self._ring_candidate_cache_limit:
                self._ring_semantic_decoder_cache.popitem(last=False)
        else:
            self._ring_semantic_decoder_cache.move_to_end(key)
        return cached

    def _ring_template_placements(
        self,
        state: MolecularGraph,
        template_index: int,
    ) -> tuple[RingSystemPlacement, ...]:
        """Return one representative per resonance-quotiented placement."""

        return tuple(
            group[0]
            for group in self._ring_template_placement_groups(state, template_index)
        )

    def _ring_witness_candidates(
        self,
        state: MolecularGraph,
        template_index: int,
    ) -> tuple[ExecutableRingGrowCandidate, ...]:
        """Return cached executor-verified catalog witnesses when available."""

        aliases = self.ring_system_electronic_witness_aliases
        if aliases is None:
            return ()
        state_key = self._state_cache_key(state)
        key = (state_key, int(template_index))
        cached = self._ring_paired_candidate_cache.get(key)
        if cached is None:
            cached = enumerate_executable_ring_grow_candidates(
                state,
                aliases[int(template_index)],
                stop_after_first=self.ring_electronic_mode != "catalog_exact",
            )
            self._ring_paired_candidate_cache[key] = cached
            if len(self._ring_paired_candidate_cache) > self._ring_candidate_cache_limit:
                self._ring_paired_candidate_cache.popitem(last=False)
        else:
            self._ring_paired_candidate_cache.move_to_end(key)
        return cached

    def _ring_paired_candidates(
        self,
        state: MolecularGraph,
        template_index: int,
    ) -> tuple[ExecutableRingGrowCandidate, ...]:
        """Return the exact catalog candidate table for catalog-exact scoring."""

        if self.ring_system_electronic_aliases is None:
            raise RuntimeError("paired ring candidates require catalog_exact mode")
        return self._ring_witness_candidates(state, template_index)

    def _ring_delete_candidates(
        self,
        state: MolecularGraph,
    ) -> tuple[RingSystemDelete, ...]:
        key = self._state_cache_key(state)
        cached = self._ring_delete_candidate_cache.get(key)
        if cached is None:
            cached = enumerate_structured_ring_system_deletes(
                state,
                self.ring_catalog,
            )
            self._ring_delete_candidate_cache[key] = cached
            if len(self._ring_delete_candidate_cache) > self._ring_candidate_cache_limit:
                self._ring_delete_candidate_cache.popitem(last=False)
        else:
            self._ring_delete_candidate_cache.move_to_end(key)
        return cached

    def _ring_grow_template_logits(
        self,
        batch: FactorizedMarkBatch,
        global_state: Tensor,
        *,
        require_exact_support: bool = True,
    ) -> tuple[Tensor, Tensor]:
        logits = _template_logits(
            self.ring_system_template_query(global_state),
            self.ring_system_template_key.weight[: len(self.ring_system_templates)],
        )
        logits = logits + self.ring_system_template_log_prior.unsqueeze(0)
        precomputed_support = batch.dense_ring_grow_support(device=self.device)
        if precomputed_support is not None:
            expected_shape = (batch.batch_size, len(self.ring_system_templates))
            if tuple(precomputed_support.shape) != expected_shape:
                raise ValueError(
                    "precomputed ring support has the wrong shape: "
                    f"{tuple(precomputed_support.shape)} != {expected_shape}"
                )
        if precomputed_support is None:
            # Training likelihoods require the exact executor-aware support.
            # Inference may defer this expensive refinement until the ring
            # family is actually selected; family probabilities use only the
            # Boolean enabled flag, not the within-family partition.
            mask = torch.tensor(
                tuple(self._ring_grow_support(state) for state in batch.states),
                dtype=torch.bool,
                device=self.device,
            )
        elif not require_exact_support:
            mask = precomputed_support
        else:
            def row_flags(value: Tensor | bool, name: str) -> Tensor:
                if isinstance(value, Tensor):
                    if tuple(value.shape) != (batch.batch_size,):
                        raise ValueError(
                            f"{name} has the wrong shape: {tuple(value.shape)} "
                            f"!= {(batch.batch_size,)}"
                        )
                    return value.detach().to(device="cpu", dtype=torch.bool)
                return torch.full(
                    (batch.batch_size,),
                    bool(value),
                    dtype=torch.bool,
                )

            full_rows = row_flags(
                batch.ring_grow_support_is_exact,
                "ring_grow_support_is_exact",
            )
            enablement_rows = full_rows | row_flags(
                batch.ring_grow_enablement_is_exact,
                "ring_grow_enablement_is_exact",
            )
            needs_full = torch.tensor(
                tuple(name == "ring_system_grow" for name in batch.teacher_rule_names),
                dtype=torch.bool,
            )
            usable_rows = torch.where(needs_full, full_rows, enablement_rows)
            if bool(usable_rows.all()):
                mask = precomputed_support
            else:
                rows = []
                for index, state in enumerate(batch.states):
                    if bool(usable_rows[index]):
                        rows.append(precomputed_support[index])
                    else:
                        rows.append(
                            torch.tensor(
                                self._ring_grow_support(state),
                                dtype=torch.bool,
                                device=self.device,
                            )
                        )
                mask = torch.stack(rows)
        return logits, mask

    def _ring_placement_logits(
        self,
        placements: tuple[RingSystemPlacement, ...],
        node: Tensor,
        global_state: Tensor,
        pair: Tensor,
    ) -> Tensor:
        rows = node.new_zeros((len(placements), 4 * self.hidden_dim))
        for index, placement in enumerate(placements):
            member_index = torch.tensor(
                placement.system_atoms,
                dtype=torch.long,
                device=self.device,
            )
            member_state = node.index_select(0, member_index).mean(dim=0)
            pair_state = torch.stack(
                tuple(pair[int(bond.a), int(bond.b)] for bond in placement.bond_insertions)
            ).mean(dim=0)
            aromatic = {
                frozenset((int(a), int(b))) for a, b in placement.aromatic_edges
            }
            content_pieces = [
                self.bond_embedding.weight[BOND_AROMATIC]
                for _ in placement.aromatic_edges
            ]
            content_pieces.extend(
                self.bond_embedding.weight[int(bond.order)]
                for bond in placement.bond_insertions
                if frozenset((int(bond.a), int(bond.b))) not in aromatic
            )
            content_pieces.extend(
                self.bond_embedding.weight[int(change.new_order)]
                for change in placement.bond_reorders
                if frozenset((int(change.a), int(change.b))) not in aromatic
            )
            content_state = torch.stack(content_pieces).mean(dim=0)
            rows[index] = torch.cat((member_state, pair_state, content_state, global_state))
        return self.ring_system_grow_head(rows).squeeze(-1)

    def _ring_atom_logits(
        self,
        state: MolecularGraph,
        placement: RingSystemPlacement,
        node: Tensor,
        global_state: Tensor,
        *,
        resonance_aliases: tuple[RingSystemPlacement, ...] | None = None,
    ) -> tuple[Tensor, Tensor]:
        member_index = torch.tensor(
            placement.system_atoms,
            dtype=torch.long,
            device=self.device,
        )
        logits = self.ring_system_atom_head(
            torch.cat(
                (
                    node.index_select(0, member_index),
                    global_state.expand(len(placement.system_atoms), -1),
                ),
                dim=-1,
            )
        )
        aliases = (placement,) if resonance_aliases is None else resonance_aliases
        if any(alias.system_atoms != placement.system_atoms for alias in aliases):
            raise ValueError("resonance aliases do not share placement members")
        # This four-way table is retained only for catalog_exact diagnostics.
        mask_array = ring_system_placement_local_atom_type_mask(state, placement)
        mask = torch.from_numpy(mask_array.copy()).to(self.device)
        return logits, mask

    def _ring_semantic_atom_logits(
        self,
        decoder: SemanticRingSystemDecoder,
        node: Tensor,
        global_state: Tensor,
    ) -> Tensor:
        member_index = torch.tensor(
            decoder.placement.system_atoms,
            dtype=torch.long,
            device=self.device,
        )
        features = torch.cat(
            (
                node.index_select(0, member_index),
                global_state.expand(decoder.span, -1),
            ),
            dim=-1,
        )
        type_logits = self.ring_system_atom_head(features)
        role_logits = self.ring_system_role_head(features)
        return (
            type_logits.unsqueeze(-1) + role_logits.unsqueeze(-2)
        ).reshape(decoder.span, 2 * len(CNOF_ATOM_TYPES))

    def _ring_semantic_sequence_log_probability(
        self,
        decoder: SemanticRingSystemDecoder,
        category_logits: Tensor,
        categories: tuple[int, ...],
    ) -> tuple[Tensor, Tensor]:
        score = category_logits.sum() * 0.0
        legal = score.new_tensor(True, dtype=torch.bool)
        if len(categories) != decoder.span:
            return score, score.new_tensor(False, dtype=torch.bool)
        prefix: tuple[int, ...] = ()
        for position, category in enumerate(categories):
            mask = torch.tensor(
                semantic_ring_next_category_mask(decoder, prefix),
                dtype=torch.bool,
                device=self.device,
            )
            if not bool(mask.any()) or not 0 <= int(category) < mask.numel():
                return score, score.new_tensor(False, dtype=torch.bool)
            legal = legal & mask[int(category)]
            score = score + torch.log_softmax(
                category_logits[position].masked_fill(~mask, float("-inf")),
                dim=-1,
            )[int(category)]
            prefix = (*prefix, int(category))
        return score, legal

    def _ring_supported_placement_tables(
        self,
        state: MolecularGraph,
        placement_groups: tuple[tuple[RingSystemPlacement, ...], ...],
        node: Tensor,
        global_state: Tensor,
        pair: Tensor,
    ) -> tuple[
        Tensor,
        tuple[tuple[SemanticRingSystemDecoder, Tensor], ...],
        Tensor,
    ]:
        """Build the identical semantic support table for loss and sampling."""

        placements = tuple(group[0] for group in placement_groups)
        placement_logits = self._ring_placement_logits(
            placements,
            node,
            global_state,
            pair,
        )
        decoders = tuple(
            self._ring_semantic_decoder(state, placement)
            for placement in placements
        )
        semantic_tables = tuple(
            (
                decoder,
                self._ring_semantic_atom_logits(
                    decoder,
                    node,
                    global_state,
                ),
            )
            for decoder in decoders
        )
        placement_mask = torch.tensor(
            tuple(
                semantic_ring_prefix_is_completable(decoder, ())
                for decoder in decoders
            ),
            dtype=torch.bool,
            device=self.device,
        )
        return placement_logits, semantic_tables, placement_mask

    def _ring_exact_candidate_log_probabilities(
        self,
        state: MolecularGraph,
        template_index: int,
        node: Tensor,
        global_state: Tensor,
        pair: Tensor,
    ) -> tuple[tuple[ExecutableRingGrowCandidate, ...], Tensor]:
        """Score the shared exact candidate table as a normalized hierarchy."""

        candidates = self._ring_paired_candidates(state, template_index)
        if not candidates:
            return candidates, node.new_empty((0,))
        placements = tuple(dict.fromkeys(candidate.placement for candidate in candidates))
        placement_to_index = {
            placement: index for index, placement in enumerate(placements)
        }
        placement_logits = self._ring_placement_logits(
            placements,
            node,
            global_state,
            pair,
        )
        placement_log_probabilities = torch.log_softmax(placement_logits, dim=0)
        atom_tables = tuple(
            self._ring_atom_logits(
                state,
                placement,
                node,
                global_state,
            )
            for placement in placements
        )
        label_scores = []
        candidate_placement_indices = []
        for candidate in candidates:
            placement_index = placement_to_index[candidate.placement]
            label_logits, label_mask = atom_tables[placement_index]
            selected = torch.tensor(
                tuple(_CNOF_TO_INDEX[int(atom_type)] for atom_type in candidate.atom_types),
                dtype=torch.long,
                device=self.device,
            )
            if not bool(label_mask.gather(-1, selected.unsqueeze(-1)).all()):
                raise RuntimeError("exact ring candidate violates the local label mask")
            label_scores.append(
                torch.log_softmax(
                    label_logits.masked_fill(~label_mask, float("-inf")),
                    dim=-1,
                )
                .gather(-1, selected.unsqueeze(-1))
                .sum()
            )
            candidate_placement_indices.append(placement_index)
        label_score_tensor = torch.stack(label_scores)
        label_normalizers = []
        for placement_index in range(len(placements)):
            indices = torch.tensor(
                tuple(
                    index
                    for index, candidate_placement_index in enumerate(
                        candidate_placement_indices
                    )
                    if candidate_placement_index == placement_index
                ),
                dtype=torch.long,
                device=self.device,
            )
            group_scores = label_score_tensor.index_select(0, indices)
            label_normalizers.append(torch.logsumexp(group_scores, dim=0))
        result = torch.stack(
            tuple(
                placement_log_probabilities[placement_index]
                + label_score_tensor[candidate_index]
                - label_normalizers[placement_index]
                for candidate_index, placement_index in enumerate(
                    candidate_placement_indices
                )
            )
        )
        return candidates, result

    def _ring_delete_action_logits(
        self,
        batch: FactorizedMarkBatch,
        node: Tensor,
        global_state: Tensor,
        pair: Tensor,
    ) -> tuple[Tensor, Tensor]:
        candidates = (
            tuple(self._ring_delete_candidates(state) for state in batch.states)
            if batch.ring_delete_actions is None
            else batch.ring_delete_actions
        )
        maximum = max((len(items) for items in candidates), default=0)
        width = max(maximum, 1)
        rows = node.new_zeros((batch.batch_size, width, 4 * self.hidden_dim))
        mask = torch.zeros(
            (batch.batch_size, width),
            dtype=torch.bool,
            device=self.device,
        )
        for batch_index, actions in enumerate(candidates):
            for action_index, action in enumerate(actions):
                member_slots = tuple(int(slot) for slot in action.system_atoms)
                member_index = torch.tensor(
                    member_slots,
                    dtype=torch.long,
                    device=self.device,
                )
                member_state = node[batch_index].index_select(0, member_index).mean(dim=0)
                cycle_bonds = action.bond_deletions
                pair_pieces = tuple(
                    pair[batch_index, int(bond.a), int(bond.b)] for bond in cycle_bonds
                )
                pair_state = torch.stack(pair_pieces).mean(dim=0)
                content_pieces = [
                    self.atom_embedding.weight[int(payload.atom_type)]
                    + self.charge_embedding.weight[int(payload.formal_charge) + 2]
                    + self.hydrogen_embedding.weight[int(payload.implicit_h_count)]
                    for payload in action.atom_payloads
                ]
                reorders = action.bond_reorders
                content_pieces.extend(
                    self.bond_embedding.weight[int(bond.order)] for bond in cycle_bonds
                )
                content_pieces.extend(
                    self.bond_embedding.weight[int(change.new_order)] for change in reorders
                )
                content_state = (
                    torch.stack(content_pieces).mean(dim=0)
                    if content_pieces
                    else node.new_zeros(self.hidden_dim)
                )
                rows[batch_index, action_index] = torch.cat(
                    (
                        member_state,
                        pair_state,
                        content_state,
                        global_state[batch_index],
                    )
                )
                mask[batch_index, action_index] = True
        return self.ring_system_delete_head(rows).squeeze(-1), mask

    def forward_mark_batch(self, batch: FactorizedMarkBatch) -> FactorizedMarkPrediction:
        if batch.atom_types.device != self.device:
            batch = batch.to(self.device)
        node, global_state, pair = self._encode_batch(batch)
        masks, logits, action_log_z = self._action_tables(batch, node, global_state, pair)
        enabled = torch.isfinite(action_log_z)
        has_legal_mark = enabled.any(dim=-1)
        masked_family_logits = self.family_head(global_state).masked_fill(
            ~enabled,
            float("-inf"),
        )
        family_logits = torch.where(
            has_legal_mark.unsqueeze(-1),
            masked_family_logits,
            torch.zeros_like(masked_family_logits),
        )
        family_log_prob = torch.log_softmax(family_logits, dim=-1)
        selected = self._selected_mark_log_probability(
            batch,
            node=node,
            global_state=global_state,
            pair=pair,
            masks=masks,
            logits=logits,
            action_log_z=action_log_z,
            family_log_prob=family_log_prob,
        )
        total_hazard = F.softplus(
            self.total_hazard_head(global_state).squeeze(-1)
        ) * has_legal_mark.to(global_state.dtype)
        return FactorizedMarkPrediction(
            total_hazard=total_hazard,
            selected_mark_log_probability=selected,
            family_log_probabilities=family_log_prob,
            enabled_families=enabled,
        )

    @torch.no_grad()
    def sample_rewrite_mark(
        self,
        state: MolecularGraph,
        time: float,
        rng: np.random.Generator,
    ) -> SampledRewriteMark:
        """Sample one legal rule match without constructing its successor fiber."""

        cache_key = self._state_cache_key(state)
        cached_batch = self._sampling_state_cache.get(cache_key)
        if cached_batch is None:
            cached_batch = prepare_factorized_mark_batch(
                (state,),
                (0.0,),
                (None,),
                (None,),
                (0.0,),
                use_aromatic_bond_view=True,
                ring_catalog=self.ring_catalog,
            )
            self._sampling_state_cache[cache_key] = cached_batch
            if len(self._sampling_state_cache) > self._sampling_state_cache_limit:
                self._sampling_state_cache.popitem(last=False)
        else:
            self._sampling_state_cache.move_to_end(cache_key)
        batch = replace(
            cached_batch,
            times=torch.tensor((float(time),), dtype=torch.float32),
        ).to(self.device)
        node, global_state, pair = self._encode_batch(batch)
        masks, logits, action_log_z = self._action_tables(
            batch,
            node,
            global_state,
            pair,
            require_exact_ring_support=False,
        )
        enabled = torch.isfinite(action_log_z[0]).clone()
        disabled_rule_names = frozenset(
            str(name) for name in getattr(self, "disabled_sampling_rule_names", ())
        )
        for rule_name in disabled_rule_names:
            family_index = MARK_RULE_TO_INDEX.get(rule_name)
            if family_index is not None:
                enabled[family_index] = False
        while bool(enabled.any()):
            family_logits = self.family_head(global_state)[0].masked_fill(
                ~enabled,
                float("-inf"),
            )
            family_probabilities = torch.softmax(family_logits, dim=-1).float().cpu().numpy()
            family_index = int(rng.choice(len(MARK_RULE_NAMES), p=family_probabilities))
            rule_name = MARK_RULE_NAMES[family_index]
            action = self._sample_action_from_family(
                rule_name,
                state,
                batch,
                node[0],
                global_state[0],
                pair[0],
                masks,
                logits,
                rng,
            )
            if action is None:
                enabled[family_index] = False
                continue
            hazard = float(F.softplus(self.total_hazard_head(global_state)[0, 0]))
            return SampledRewriteMark(hazard, rule_name, action)
        return SampledRewriteMark(0.0, "<TERMINAL>", None)

    def _sample_action_from_family(
        self,
        rule_name: str,
        state: MolecularGraph,
        batch: FactorizedMarkBatch,
        node: Tensor,
        global_state: Tensor,
        pair: Tensor,
        masks: dict[str, Tensor],
        logits: dict[str, Tensor],
        rng: np.random.Generator,
    ) -> Any:
        null_slots = tuple(int(v) for v in np.flatnonzero(state.atom_types == NULL_IDX))
        if rule_name == "atom_insert":
            group, coordinate = _sample_masked_group(
                (
                    (logits["grow_root"][0], masks["grow_root"][0]),
                    (logits["grow_connected"][0], masks["grow_connected"][0]),
                ),
                rng,
            )
            slot = null_slots[0]
            if group == 0:
                atom_index = coordinate[0]
                atom_type = int(CNOF_ATOM_TYPES[atom_index])
                return AtomInsert(
                    slot,
                    atom_type,
                    0,
                    int(CNOF_VALENCE[atom_type]),
                    (),
                )
            neighbor, order_index, atom_index = coordinate
            order = order_index + 1
            atom_type = int(CNOF_ATOM_TYPES[atom_index])
            return AtomInsert(
                slot,
                atom_type,
                0,
                int(CNOF_VALENCE[atom_type] - order),
                ((neighbor, order),),
            )
        if rule_name == "atom_delete":
            (vertex,) = _sample_masked_coordinate(
                logits["atom_delete"][0],
                masks["atom_delete"][0],
                rng,
            )
            return AtomDelete(vertex)
        if rule_name == "atom_restate":
            vertex, atom_index = _sample_masked_coordinate(
                logits["atom_restate"][0],
                masks["atom_restate"][0],
                rng,
            )
            atom_type = int(CNOF_ATOM_TYPES[atom_index])
            bond_valence = sum(
                int(BOND_CLASS_TO_H_CHANGE[int(order)]) for order in state.bonds[vertex]
            )
            return AtomRestate(
                vertex,
                atom_type,
                0,
                int(CNOF_VALENCE[atom_type] - bond_valence),
            )
        if rule_name == "bond_reorder":
            a, b, order_index = _sample_masked_coordinate(
                logits["bond_reorder"][0],
                masks["bond_reorder"][0],
                rng,
            )
            return BondReorder(a, b, order_index + 1)
        if rule_name == "bond_reroute":
            moved, target = _sample_masked_coordinate(
                logits["bond_reroute"][0],
                masks["bond_reroute"][0],
                rng,
            )
            removed_neighbor = int(batch.graft_remove_neighbors[0, moved, target])
            return BondReroute(
                a=moved,
                b=removed_neighbor,
                u=moved,
                v=target,
            )
        if rule_name == "cycle_insert":
            (template_index,) = _sample_masked_coordinate(
                logits["cycle_insert"][0],
                masks["cycle_insert"][0],
                rng,
            )
            template = self.cycle_templates[template_index]
            return template.instantiate(null_slots[: template.span])
        if rule_name == "cycle_attach":
            anchor, template_index = _sample_masked_coordinate(
                logits["cycle_attach"][0],
                masks["cycle_attach"][0],
                rng,
            )
            template = self.attach_templates[template_index]
            return template.instantiate(anchor, null_slots[: template.span])
        if rule_name == "ring_system_grow":
            exact_template_mask = torch.tensor(
                self._ring_grow_support(state),
                dtype=torch.bool,
                device=self.device,
            )
            if not bool(exact_template_mask.any()):
                return None
            (template_index,) = _sample_masked_coordinate(
                logits["ring_system_grow"][0],
                exact_template_mask,
                rng,
            )
            if self.ring_electronic_mode == "catalog_exact":
                candidates, candidate_log_probabilities = (
                    self._ring_exact_candidate_log_probabilities(
                        state,
                        template_index,
                        node,
                        global_state,
                        pair,
                    )
                )
                if not candidates:
                    raise RuntimeError("exact ring support advertised an empty candidate table")
                (candidate_index,) = _sample_masked_coordinate(
                    candidate_log_probabilities,
                    torch.ones_like(
                        candidate_log_probabilities,
                        dtype=torch.bool,
                    ),
                    rng,
                )
                return candidates[candidate_index].action

            placement_groups = self._ring_template_placement_groups(
                state,
                template_index,
            )
            placement_logits, semantic_tables, placement_mask = (
                self._ring_supported_placement_tables(
                    state,
                    placement_groups,
                    node,
                    global_state,
                    pair,
                )
            )
            if not bool(placement_mask.any()):
                raise RuntimeError("semantic ring support advertised an empty placement table")
            (placement_index,) = _sample_masked_coordinate(
                placement_logits,
                placement_mask,
                rng,
            )
            decoder, category_logits = semantic_tables[placement_index]
            prefix: tuple[int, ...] = ()
            for position in range(decoder.span):
                category_mask = torch.tensor(
                    semantic_ring_next_category_mask(decoder, prefix),
                    dtype=torch.bool,
                    device=self.device,
                )
                (category,) = _sample_masked_coordinate(
                    category_logits[position],
                    category_mask,
                    rng,
                )
                prefix = (*prefix, category)
            return instantiate_semantic_ring_system_grow(
                state,
                decoder,
                prefix,
            )
        if rule_name == "ring_system_delete":
            (action_index,) = _sample_masked_coordinate(
                logits["ring_system_delete"][0],
                masks["ring_system_delete"][0],
                rng,
            )
            return self._ring_delete_candidates(state)[action_index]
        if rule_name == "ring_system_restate":
            (action_index,) = _sample_masked_coordinate(
                logits["ring_system_restate"][0],
                masks["ring_system_restate"][0],
                rng,
            )
            return batch.ring_restate_actions[0][action_index]
        raise ValueError(f"unsupported sampled rule family: {rule_name}")

    def _encode_batch(
        self,
        batch: FactorizedMarkBatch,
    ) -> tuple[Tensor, Tensor, Tensor]:
        atom_types = batch.atom_types
        charges = batch.formal_charges + 2
        hydrogens = batch.implicit_h_counts
        bonds = batch.neural_bonds
        real = (atom_types != NULL_IDX) & (atom_types != SCAR_IDX)
        real_float = real.to(self.atom_embedding.weight.dtype).unsqueeze(-1)
        times = batch.times.to(self.atom_embedding.weight.dtype)
        time_features = torch.stack(
            (times, torch.sin(torch.pi * times), torch.cos(torch.pi * times)),
            dim=-1,
        )
        time_state = self.time_encoder(time_features)
        node = (
            self.atom_embedding(atom_types)
            + self.charge_embedding(charges)
            + self.hydrogen_embedding(hydrogens)
            + self.atom_ring_embedding(batch.atom_topology)
        ) * real_float
        edge_mask = (bonds != 0).unsqueeze(-1)
        edge_state = self.bond_embedding(bonds)
        batch_size, n_slots = atom_types.shape
        for update, norm in zip(self.update_networks, self.update_norms):
            sender = (
                self.message_node(node)
                .unsqueeze(1)
                .expand(
                    batch_size,
                    n_slots,
                    n_slots,
                    self.hidden_dim,
                )
            )
            messages = (sender + edge_state) * edge_mask
            aggregated = messages.sum(dim=2)
            tiled_time = time_state.unsqueeze(1).expand(-1, n_slots, -1)
            delta = update(torch.cat((node, aggregated, tiled_time), dim=-1))
            node = norm(node + delta) * real_float

        count = real_float.sum(dim=1).clamp_min(1.0)
        pooled = node.sum(dim=1) / count
        empty = ~real.any(dim=1)
        pooled = torch.where(empty.unsqueeze(-1), self.empty_state, pooled)
        n_real = real.sum(dim=1).to(pooled.dtype)
        edge_count = torch.triu(bonds != 0, diagonal=1).sum(dim=(1, 2)).to(pooled.dtype)
        cycle_rank = torch.clamp(edge_count - n_real + (n_real > 0), min=0.0)
        normalizer = float(max(n_slots, 1))
        structural = torch.stack(
            (
                n_real / normalizer,
                edge_count / normalizer,
                cycle_rank / normalizer,
                (hydrogens * real).sum(dim=1).to(pooled.dtype) / (4.0 * normalizer),
            ),
            dim=-1,
        )
        pooled = pooled + self.structure_encoder(structural)
        global_state = self.global_project(torch.cat((pooled, time_state), dim=-1))

        left = node.unsqueeze(2)
        right = node.unsqueeze(1)
        pair = self.pair_project(
            torch.cat(
                (
                    left + right,
                    torch.abs(left - right),
                    edge_state,
                    self.closure_ring_embedding(batch.closure_topology),
                    self.ring_system_embedding(batch.ring_system_topology),
                ),
                dim=-1,
            )
        )
        return node, global_state, pair

    def _action_tables(
        self,
        batch: FactorizedMarkBatch,
        node: Tensor,
        global_state: Tensor,
        pair: Tensor,
        *,
        require_exact_ring_support: bool = True,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], Tensor]:
        batch_size, n_slots = batch.atom_types.shape
        real = (batch.atom_types != NULL_IDX) & (batch.atom_types != SCAR_IDX)
        n_real = real.sum(dim=1)
        n_null = (batch.atom_types == NULL_IDX).sum(dim=1)
        hydrogens = batch.implicit_h_counts
        device = self.device
        order_values = torch.arange(1, 4, device=device)

        masks: dict[str, Tensor] = {}
        logits: dict[str, Tensor] = {}

        root_grow_logits = self.grow_root_head(global_state)
        root_grow_mask = (
            ((n_real == 0) & (n_null > 0)).unsqueeze(-1).expand(-1, len(CNOF_ATOM_TYPES))
        )
        grow_query = self.grow_query(node)
        grow_option = self.grow_option.weight.reshape(
            3,
            len(CNOF_ATOM_TYPES),
            self.mark_dim,
        )
        connected_grow_logits = torch.einsum(
            "bnr,otr->bnot",
            grow_query,
            grow_option,
        ) / sqrt(self.mark_dim)
        type_h = self.cnof_valences.view(1, 1, 1, -1) - order_values.view(1, 1, -1, 1)
        connected_grow_mask = (
            real.unsqueeze(-1).unsqueeze(-1)
            & (n_null > 0).view(-1, 1, 1, 1)
            & (hydrogens.unsqueeze(-1).unsqueeze(-1) >= order_values.view(1, 1, -1, 1))
            & (type_h >= 0)
            & (type_h <= MAX_H_COUNT)
        )
        masks["grow_root"] = root_grow_mask
        masks["grow_connected"] = connected_grow_mask
        logits["grow_root"] = root_grow_logits
        logits["grow_connected"] = connected_grow_logits
        atom_insert_z = torch.logsumexp(
            torch.stack(
                (
                    _masked_logsumexp(root_grow_logits, root_grow_mask),
                    _masked_logsumexp(connected_grow_logits, connected_grow_mask),
                ),
                dim=-1,
            ),
            dim=-1,
        )

        delete_logits = self.delete_head(node).squeeze(-1)
        delete_mask = batch.atom_delete_mask
        masks["atom_delete"] = delete_mask
        logits["atom_delete"] = delete_logits

        restate_logits = self.restate_head(node)
        bond_valence = torch.zeros_like(hydrogens)
        for order, delta in enumerate(BOND_CLASS_TO_H_CHANGE):
            if order == 0:
                continue
            bond_valence = bond_valence + (batch.bonds == order).sum(dim=-1) * int(delta)
        restate_h = self.cnof_valences.view(1, 1, -1) - bond_valence.unsqueeze(-1)
        restate_mask = (
            real.unsqueeze(-1)
            & (batch.atom_topology == 0).unsqueeze(-1)
            & (restate_h >= 0)
            & (restate_h <= MAX_H_COUNT)
            & (
                (batch.atom_types.unsqueeze(-1) != torch.tensor(CNOF_ATOM_TYPES, device=device))
                | (batch.implicit_h_counts.unsqueeze(-1) != restate_h)
                | (batch.formal_charges.unsqueeze(-1) != 0)
            )
        )
        masks["atom_restate"] = restate_mask
        logits["atom_restate"] = restate_logits

        reorder_logits = self.reorder_head(pair)
        old_order = batch.bonds.unsqueeze(-1)
        new_order = order_values.view(1, 1, 1, 3)
        delta = new_order - old_order
        upper = torch.triu(
            torch.ones((n_slots, n_slots), dtype=torch.bool, device=device), diagonal=1
        )
        reorder_mask = (
            upper.view(1, n_slots, n_slots, 1)
            & (old_order >= 1)
            & (old_order <= 3)
            & (new_order != old_order)
            & ~batch.cycle_edge_mask.unsqueeze(-1)
            & ((hydrogens.unsqueeze(2).unsqueeze(-1) - delta) >= 0)
            & ((hydrogens.unsqueeze(2).unsqueeze(-1) - delta) <= MAX_H_COUNT)
            & ((hydrogens.unsqueeze(1).unsqueeze(-1) - delta) >= 0)
            & ((hydrogens.unsqueeze(1).unsqueeze(-1) - delta) <= MAX_H_COUNT)
        )
        masks["bond_reorder"] = reorder_mask
        logits["bond_reorder"] = reorder_logits

        graft_features = torch.cat(
            (
                node.unsqueeze(2).expand(-1, -1, n_slots, -1),
                node.unsqueeze(1).expand(-1, n_slots, -1, -1),
                global_state[:, None, None, :].expand(-1, n_slots, n_slots, -1),
            ),
            dim=-1,
        )
        graft_logits = self.graft_head(graft_features).squeeze(-1)
        graft_mask = batch.graft_mask
        masks["bond_reroute"] = graft_mask
        logits["bond_reroute"] = graft_logits

        cycle_logits = _template_logits(
            self.cycle_query(global_state),
            self.cycle_key.weight[: len(self.cycle_templates)],
        )
        cycle_mask = (n_real == 0).unsqueeze(-1) & (
            n_null.unsqueeze(-1) >= self.cycle_spans.unsqueeze(0)
        )
        masks["cycle_insert"] = cycle_mask
        logits["cycle_insert"] = cycle_logits

        attach_logits = _template_logits(
            self.attach_query(node + global_state.unsqueeze(1)),
            self.attach_key.weight[: len(self.attach_templates)],
        )
        attach_mask = (
            real.unsqueeze(-1)
            & (n_null.view(-1, 1, 1) >= self.attach_spans.view(1, 1, -1))
            & (hydrogens.unsqueeze(-1) >= self.attach_required_h.view(1, 1, -1))
        )
        masks["cycle_attach"] = attach_mask
        logits["cycle_attach"] = attach_logits

        ring_grow_logits, ring_grow_mask = self._ring_grow_template_logits(
            batch,
            global_state,
            require_exact_support=require_exact_ring_support,
        )
        masks["ring_system_grow"] = ring_grow_mask
        logits["ring_system_grow"] = ring_grow_logits

        ring_delete_logits, ring_delete_mask = self._ring_delete_action_logits(
            batch,
            node,
            global_state,
            pair,
        )
        masks["ring_system_delete"] = ring_delete_mask
        logits["ring_system_delete"] = ring_delete_logits

        ring_restate_logits, ring_restate_mask = self._ring_restate_logits(
            batch,
            pair,
            global_state,
        )
        masks["ring_system_restate"] = ring_restate_mask
        logits["ring_system_restate"] = ring_restate_logits

        family_z = torch.stack(
            (
                atom_insert_z,
                _masked_logsumexp(delete_logits, delete_mask),
                _masked_logsumexp(restate_logits, restate_mask),
                _masked_logsumexp(reorder_logits, reorder_mask),
                _masked_logsumexp(graft_logits, graft_mask),
                _masked_logsumexp(cycle_logits, cycle_mask),
                _masked_logsumexp(attach_logits, attach_mask),
                _masked_logsumexp(ring_grow_logits, ring_grow_mask),
                _masked_logsumexp(ring_delete_logits, ring_delete_mask),
                _masked_logsumexp(ring_restate_logits, ring_restate_mask),
            ),
            dim=-1,
        )
        if family_z.shape != (batch_size, len(MARK_RULE_NAMES)):
            raise RuntimeError("factorized family partition has an invalid shape")
        return masks, logits, family_z

    def _ring_restate_logits(
        self,
        batch: FactorizedMarkBatch,
        pair: Tensor,
        global_state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        maximum = max((len(items) for items in batch.ring_restate_actions), default=0)
        width = max(maximum, 1)
        rows = pair.new_zeros((batch.batch_size, width, self.hidden_dim))
        mask = torch.zeros(
            (batch.batch_size, width),
            dtype=torch.bool,
            device=self.device,
        )
        for batch_index, actions in enumerate(batch.ring_restate_actions):
            for action_index, action in enumerate(actions):
                pieces = []
                for change in action.changes:
                    pieces.append(
                        pair[batch_index, int(change.a), int(change.b)]
                        + self.restate_order_embedding.weight[int(change.new_order)]
                    )
                rows[batch_index, action_index] = torch.stack(pieces).mean(dim=0)
                mask[batch_index, action_index] = True
        logits = self.ring_restate_head(
            torch.cat((rows, global_state.unsqueeze(1).expand(-1, width, -1)), dim=-1)
        ).squeeze(-1)
        return logits, mask

    def _selected_mark_log_probability(
        self,
        batch: FactorizedMarkBatch,
        *,
        node: Tensor,
        global_state: Tensor,
        pair: Tensor,
        masks: dict[str, Tensor],
        logits: dict[str, Tensor],
        action_log_z: Tensor,
        family_log_prob: Tensor,
    ) -> Tensor:
        selected = family_log_prob.new_zeros(batch.batch_size)
        for index, (rule_name, action) in enumerate(
            zip(batch.teacher_rule_names, batch.teacher_actions)
        ):
            if rule_name is None:
                continue
            if action is None:
                raise RuntimeError("nonterminal teacher is missing its rewrite mark")
            if rule_name not in MARK_RULE_TO_INDEX:
                raise ValueError(f"unsupported factorized teacher family: {rule_name}")
            family_index = MARK_RULE_TO_INDEX[rule_name]
            action_score, legal = self._teacher_action_score(
                index,
                action,
                rule_name,
                batch,
                node,
                global_state,
                pair,
                masks,
                logits,
            )
            selected[index] = (
                family_log_prob[index, family_index]
                + action_score
                - action_log_z[index, family_index]
            )
            # An illegal teacher produces ``-inf`` here and therefore a
            # non-finite loss without forcing one GPU synchronization per row.
            selected[index] = torch.where(
                legal,
                selected[index],
                selected[index].new_tensor(float("-inf")),
            )
        return selected

    def _teacher_action_score(
        self,
        batch_index: int,
        action: Any,
        rule_name: str,
        batch: FactorizedMarkBatch,
        node: Tensor,
        global_state: Tensor,
        pair: Tensor,
        masks: dict[str, Tensor],
        logits: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        if isinstance(action, AtomInsert):
            atom_index = _CNOF_TO_INDEX[int(action.atom_type)]
            if not action.neighbors:
                return (
                    logits["grow_root"][batch_index, atom_index],
                    masks["grow_root"][batch_index, atom_index],
                )
            if len(action.neighbors) != 1:
                raise ValueError("factorized grow supports one existing neighbor")
            neighbor, order = action.neighbors[0]
            order_index = _ORDER_TO_INDEX[int(order)]
            key = (batch_index, int(neighbor), order_index, atom_index)
            return logits["grow_connected"][key], masks["grow_connected"][key]
        if isinstance(action, AtomDelete):
            key = (batch_index, int(action.v))
            return logits["atom_delete"][key], masks["atom_delete"][key]
        if isinstance(action, AtomRestate):
            atom_index = _CNOF_TO_INDEX[int(action.atom_type)]
            key = (batch_index, int(action.v), atom_index)
            return logits["atom_restate"][key], masks["atom_restate"][key]
        if isinstance(action, BondReorder):
            a, b = sorted((int(action.a), int(action.b)))
            order_index = _ORDER_TO_INDEX[int(action.new_order)]
            key = (batch_index, a, b, order_index)
            return logits["bond_reorder"][key], masks["bond_reorder"][key]
        if isinstance(action, BondReroute):
            moved = int(action.u)
            target = int(action.v)
            removed = {int(action.a), int(action.b)} - {moved}
            if len(removed) != 1:
                raise ValueError("dense Graft teacher requires a root-free reroute")
            removed_neighbor = removed.pop()
            key = (batch_index, moved, target)
            legal = masks["bond_reroute"][key] & (
                batch.graft_remove_neighbors[key] == removed_neighbor
            )
            successor_group = batch.graft_successor_groups[key]
            group_mask = masks["bond_reroute"][batch_index] & (
                batch.graft_successor_groups[batch_index] == successor_group
            )
            legal = legal & (successor_group >= 0) & group_mask.any()
            # Generator Matching supervises the molecular successor, not one
            # arbitrary atom-slot presentation of the same Graft.  Summing
            # the marked rates is a log-sum-exp in the normalized logit table.
            grouped_score = torch.logsumexp(
                logits["bond_reroute"][batch_index]
                .masked_fill(~group_mask, float("-inf"))
                .reshape(-1),
                dim=0,
            )
            return grouped_score, legal
        if isinstance(action, CycleInsert):
            template_index = self._cycle_to_index[cycle_template(action)]
            key = (batch_index, template_index)
            return logits["cycle_insert"][key], masks["cycle_insert"][key]
        if isinstance(action, CycleAttach):
            template_index = self._attach_to_index[attach_template(action)]
            key = (batch_index, int(action.anchor), template_index)
            return logits["cycle_attach"][key], masks["cycle_attach"][key]
        if isinstance(action, RingSystemGrow):
            teacher_placement = ring_system_placement(action)
            teacher_placement_key = ring_system_placement_key(teacher_placement)
            if not is_valid_ring_system_grow(batch.states[batch_index], action):
                zero = logits["ring_system_grow"][batch_index].sum() * 0.0
                return zero, zero.new_tensor(False, dtype=torch.bool)
            template_indices = matching_ring_system_template_indices(
                action,
                self.ring_system_templates,
            )

            if self.ring_electronic_mode == "catalog_exact":
                teacher_key = ring_system_grow_electronic_key(action)
                exact_action_scores = []
                for template_index in template_indices:
                    key = (batch_index, template_index)
                    if not bool(masks["ring_system_grow"][key]):
                        continue
                    candidates, candidate_log_probabilities = (
                        self._ring_exact_candidate_log_probabilities(
                            batch.states[batch_index],
                            template_index,
                            node[batch_index],
                            global_state[batch_index],
                            pair[batch_index],
                        )
                    )
                    matching = tuple(
                        index
                        for index, candidate in enumerate(candidates)
                        if ring_system_grow_electronic_key(candidate.action) == teacher_key
                    )
                    if not matching:
                        continue
                    matching_indices = torch.tensor(
                        matching,
                        dtype=torch.long,
                        device=self.device,
                    )
                    exact_action_scores.append(
                        logits["ring_system_grow"][key]
                        + torch.logsumexp(
                            candidate_log_probabilities.index_select(
                                0,
                                matching_indices,
                            ),
                            dim=0,
                        )
                    )
                if not exact_action_scores:
                    zero = logits["ring_system_grow"][batch_index].sum() * 0.0
                    return zero, zero.new_tensor(False, dtype=torch.bool)
                return torch.logsumexp(
                    torch.stack(exact_action_scores),
                    dim=0,
                ), exact_action_scores[0].new_tensor(True, dtype=torch.bool)

            action_scores = []
            for template_index in template_indices:
                key = (batch_index, template_index)
                if not bool(masks["ring_system_grow"][key]):
                    continue
                placement_groups = self._ring_template_placement_groups(
                    batch.states[batch_index],
                    template_index,
                )
                placements = tuple(group[0] for group in placement_groups)
                matching_placements = tuple(
                    index
                    for index, candidate in enumerate(placements)
                    if ring_system_placement_key(candidate) == teacher_placement_key
                )
                if not matching_placements:
                    continue
                placement_logits, semantic_tables, placement_mask = (
                    self._ring_supported_placement_tables(
                        batch.states[batch_index],
                        placement_groups,
                        node[batch_index],
                        global_state[batch_index],
                        pair[batch_index],
                    )
                )
                if not bool(placement_mask.any()):
                    continue
                placement_log_z = torch.logsumexp(
                    placement_logits.masked_fill(~placement_mask, float("-inf")),
                    dim=0,
                )
                match_scores = []
                for placement_index in matching_placements:
                    if not bool(placement_mask[placement_index]):
                        continue
                    decoder, category_logits = semantic_tables[placement_index]
                    try:
                        categories = semantic_ring_categories_for_action(
                            decoder,
                            action,
                        )
                    except (KeyError, ValueError):
                        continue
                    label_score, labels_legal = (
                        self._ring_semantic_sequence_log_probability(
                            decoder,
                            category_logits,
                            categories,
                        )
                    )
                    if not bool(labels_legal):
                        continue
                    match_scores.append(
                        placement_logits[placement_index] - placement_log_z + label_score
                    )
                if match_scores:
                    action_scores.append(
                        logits["ring_system_grow"][key]
                        + torch.logsumexp(torch.stack(match_scores), dim=0)
                    )
            if not action_scores:
                zero = logits["ring_system_grow"][batch_index].sum() * 0.0
                return zero, zero.new_tensor(False, dtype=torch.bool)
            return torch.logsumexp(torch.stack(action_scores), dim=0), action_scores[0].new_tensor(
                True, dtype=torch.bool
            )
        if isinstance(action, RingSystemDelete):
            candidates = (
                self._ring_delete_candidates(batch.states[batch_index])
                if batch.ring_delete_actions is None
                else batch.ring_delete_actions[batch_index]
            )
            try:
                action_index = candidates.index(action)
            except ValueError as exc:
                raise RuntimeError(
                    "teacher ring-system delete is outside exact dynamic candidates"
                ) from exc
            key = (batch_index, action_index)
            return logits["ring_system_delete"][key], masks["ring_system_delete"][key]
        if isinstance(action, RingSystemRestate):
            candidates = batch.ring_restate_actions[batch_index]
            try:
                action_index = candidates.index(action)
            except ValueError as exc:
                raise RuntimeError(
                    "teacher ring restate is outside exact dynamic candidates"
                ) from exc
            key = (batch_index, action_index)
            return logits["ring_system_restate"][key], masks["ring_system_restate"][key]
        raise TypeError(f"unsupported factorized teacher action: {type(action).__name__}")

def _template_logits(query: Tensor, keys: Tensor) -> Tensor:
    if keys.shape[0] == 0:
        return query.new_empty((*query.shape[:-1], 0))
    return torch.einsum("...r,tr->...t", query, keys) / sqrt(query.shape[-1])


def _masked_logsumexp(logits: Tensor, mask: Tensor) -> Tensor:
    if logits.shape != mask.shape:
        raise ValueError(f"logit/mask shape mismatch: {logits.shape} != {mask.shape}")
    flat_logits = logits.flatten(start_dim=1)
    flat_mask = mask.flatten(start_dim=1)
    return torch.logsumexp(flat_logits.masked_fill(~flat_mask, float("-inf")), dim=-1)


def _sample_masked_coordinate(
    logits: Tensor,
    mask: Tensor,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    flat_mask = mask.reshape(-1)
    valid_indices = torch.nonzero(flat_mask, as_tuple=False).squeeze(-1)
    if valid_indices.numel() == 0:
        raise RuntimeError("cannot sample an empty factorized action table")
    valid_logits = logits.reshape(-1).index_select(0, valid_indices)
    probabilities = torch.softmax(valid_logits, dim=0).float().cpu().numpy()
    selected = int(valid_indices[int(rng.choice(len(valid_indices), p=probabilities))])
    return tuple(int(value) for value in np.unravel_index(selected, logits.shape))


def _sample_masked_group(
    groups: tuple[tuple[Tensor, Tensor], ...],
    rng: np.random.Generator,
) -> tuple[int, tuple[int, ...]]:
    valid_logits = []
    indices_by_group: list[Tensor] = []
    group_ids: list[int] = []
    for group_index, (group_logits, group_mask) in enumerate(groups):
        indices = torch.nonzero(group_mask.reshape(-1), as_tuple=False).squeeze(-1)
        if indices.numel() == 0:
            continue
        valid_logits.append(group_logits.reshape(-1).index_select(0, indices))
        indices_by_group.append(indices)
        group_ids.append(group_index)
    if not valid_logits:
        raise RuntimeError("cannot sample an empty factorized action family")
    probabilities = torch.softmax(torch.cat(valid_logits), dim=0).float().cpu().numpy()
    selected = int(rng.choice(len(probabilities), p=probabilities))
    offset = 0
    selected_group = 0
    for candidate_group, indices in enumerate(indices_by_group):
        next_offset = offset + int(indices.numel())
        if selected < next_offset:
            selected_group = candidate_group
            break
        offset = next_offset
    group_index = group_ids[selected_group]
    flat_index = int(indices_by_group[selected_group][selected - offset])
    shape = groups[group_index][0].shape
    coordinate = tuple(int(value) for value in np.unravel_index(flat_index, shape))
    return group_index, coordinate


def factorized_mark_bregman_loss(
    prediction: FactorizedMarkPrediction,
    batch: FactorizedMarkBatch,
) -> Tensor:
    """Poisson-KL Generator Matching loss in normalized marked-rate form."""

    teacher_rates = batch.teacher_rates.to(prediction.total_hazard.device)
    weights = batch.importance_weights.to(prediction.total_hazard.device)
    log_hazard = torch.log(prediction.total_hazard.clamp_min(1e-12))
    nonterminal = teacher_rates > 0
    teacher_term = teacher_rates * (log_hazard + prediction.selected_mark_log_probability)
    per_example = prediction.total_hazard - torch.where(
        nonterminal,
        teacher_term,
        torch.zeros_like(teacher_term),
    )
    return (per_example * weights).mean()


__all__ = [
    "FactorizedMarkBatch",
    "FactorizedMarkPrediction",
    "FactorizedTraceletRateModel",
    "MARK_RULE_NAMES",
    "SampledRewriteMark",
    "factorized_mark_bregman_loss",
    "prepare_factorized_mark_batch",
]
