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
from typing import Any, Mapping, MutableMapping

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
    AtomVocabulary,
    BOND_AROMATIC,
    BOND_CLASSES,
    BOND_CLASS_TO_H_CHANGE,
    CNOF_RING_ELEMENTS,
    CNOF_VOCABULARY,
    H_COUNT_CLASSES,
    K,
    M,
    MAX_H_COUNT,
    MolecularGraph,
    NULL_IDX,
    SCAR_IDX,
    is_element,
)
from compose_v4.rewrite.factorized_fiber import CNOF_ATOM_TYPES
from compose_v4.rewrite.operators import (
    AtomDelete,
    AtomInsert,
    AtomRestate,
    BondDelete,
    BondInsert,
    BondReorder,
    BondReroute,
)
from compose_v4.rewrite.ring_system_fiber import (
    build_semantic_ring_system_decoder,
    enumerate_executable_ring_grow_candidates,
    enumerate_ring_system_template_placements,
    enumerate_clean_ring_system_deletes,
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
from compose_v4.rewrite.factorized_fiber import pendant_graft_candidates
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
from compose_v4.rewrite.tracelet_fiber import enumerate_ring_system_restate_actions
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
# Compositional ring ops apply via the executor's bond_insert/bond_delete rules but are SELECTED by the model
# on the repurposed slots 5/6 (cycle_insert/cycle_attach) when enable_cycle_ops. This maps a teacher/sampled
# executor rule name back to its selecting family for the family-index lookup.
_CYCLE_OP_EXECUTOR_TO_FAMILY = {"bond_insert": "cycle_insert", "bond_delete": "cycle_attach"}
_ORDER_TO_INDEX = {1: 0, 2: 1, 3: 2}
# Families gated by a capability flag; everything else in MARK_RULE_NAMES is always production-enabled.
_CYCLE_OP_FAMILIES = ("cycle_insert", "cycle_attach")
_RING_GROW_MACRO_FAMILY = "ring_system_grow"


def production_enabled_families(
    *, enable_cycle_ops: bool, enable_ring_grow_macro: bool
) -> list[str]:
    """The operator families a run actually enables, DERIVED from the capability flags that gate them.

    Single source for every manifest/contract/gate that records a "production_enabled" set. Two families
    are flag-gated and mirror each other -- compositional ring addition (``cycle_insert``/``cycle_attach``,
    carrying cycle_close/cycle_open) versus the legacy whole-ring macro (``ring_system_grow``):

        RingCore-V1 (cycle ops on, grow off)  -> everything except ring_system_grow
        de-novo base B (cycle ops off, grow on) -> everything except the two cycle families

    Deriving this rather than hand-writing a name filter is deliberate. The scaled-manifest builder
    previously hard-coded the SECOND list; it was correct for base B and silently stale after RingCore
    inverted both flags. ``operator_subtype_supervision_gate`` consumes this set as the REQUIRED-supervision
    families, so a stale list demands supervision for a disabled family (spurious NO_GO) while allowing an
    enabled one to go unsupervised undetected -- both failure directions at once.

    Order follows MARK_RULE_NAMES so manifests compare byte-for-byte across builders.
    """
    disabled: set[str] = set()
    if not enable_cycle_ops:
        disabled.update(_CYCLE_OP_FAMILIES)
    if not enable_ring_grow_macro:
        disabled.add(_RING_GROW_MACRO_FAMILY)
    return [name for name in MARK_RULE_NAMES if name not in disabled]


StateCacheKey = tuple[bytes, bytes, bytes, bytes]


@dataclass(frozen=True)
class FactorizedMarkEmpiricalPriors:
    """Fixed corpus base measures underneath learned mark-logit residuals.

    Each table is a normalized log probability over the corresponding global
    mark categories.  Runtime application masks condition the base measure on
    chemistry exactly as they condition the learned residual logits.
    """

    root_atom_log_probabilities: tuple[float, ...]
    connected_atom_order_log_probabilities: tuple[tuple[float, ...], ...]
    atom_restate_log_probabilities: tuple[float, ...]
    bond_reorder_log_probabilities: tuple[float, ...]
    ring_electronic_log_probabilities: tuple[float, ...]
    root_atom_observations: int = 0
    connected_atom_observations: int = 0
    atom_restate_observations: int = 0
    bond_reorder_observations: int = 0
    ring_electronic_observations: int = 0

    def __post_init__(self) -> None:
        expected_shapes = {
            "root_atom_log_probabilities": (len(CNOF_ATOM_TYPES),),
            "connected_atom_order_log_probabilities": (
                3,
                len(CNOF_ATOM_TYPES),
            ),
            "atom_restate_log_probabilities": (len(CNOF_ATOM_TYPES),),
            "bond_reorder_log_probabilities": (3,),
            "ring_electronic_log_probabilities": (2 * len(CNOF_ATOM_TYPES),),
        }
        for name, expected_shape in expected_shapes.items():
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.shape != expected_shape:
                raise ValueError(
                    f"{name} has shape {values.shape}, expected {expected_shape}"
                )
            if not np.isfinite(values).all():
                raise ValueError(f"{name} must contain only finite values")
            if not np.isclose(np.logaddexp.reduce(values.reshape(-1)), 0.0, atol=1e-6):
                raise ValueError(f"{name} must be a normalized log probability table")
        for name in (
            "root_atom_observations",
            "connected_atom_observations",
            "atom_restate_observations",
            "bond_reorder_observations",
            "ring_electronic_observations",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "root_atom_log_probabilities": list(self.root_atom_log_probabilities),
            "connected_atom_order_log_probabilities": [
                list(row) for row in self.connected_atom_order_log_probabilities
            ],
            "atom_restate_log_probabilities": list(
                self.atom_restate_log_probabilities
            ),
            "bond_reorder_log_probabilities": list(
                self.bond_reorder_log_probabilities
            ),
            "ring_electronic_log_probabilities": list(
                self.ring_electronic_log_probabilities
            ),
            "root_atom_observations": int(self.root_atom_observations),
            "connected_atom_observations": int(self.connected_atom_observations),
            "atom_restate_observations": int(self.atom_restate_observations),
            "bond_reorder_observations": int(self.bond_reorder_observations),
            "ring_electronic_observations": int(self.ring_electronic_observations),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> "FactorizedMarkEmpiricalPriors":
        def vector(name: str) -> tuple[float, ...]:
            raw = payload.get(name)
            if not isinstance(raw, (list, tuple)):
                raise ValueError(f"empirical prior lacks {name}")
            return tuple(float(value) for value in raw)

        raw_connected = payload.get("connected_atom_order_log_probabilities")
        if not isinstance(raw_connected, (list, tuple)):
            raise ValueError(
                "empirical prior lacks connected_atom_order_log_probabilities"
            )
        return cls(
            root_atom_log_probabilities=vector("root_atom_log_probabilities"),
            connected_atom_order_log_probabilities=tuple(
                tuple(float(value) for value in row)
                for row in raw_connected
            ),
            atom_restate_log_probabilities=vector(
                "atom_restate_log_probabilities"
            ),
            bond_reorder_log_probabilities=vector(
                "bond_reorder_log_probabilities"
            ),
            ring_electronic_log_probabilities=vector(
                "ring_electronic_log_probabilities"
            ),
            root_atom_observations=int(payload.get("root_atom_observations", 0)),
            connected_atom_observations=int(
                payload.get("connected_atom_observations", 0)
            ),
            atom_restate_observations=int(
                payload.get("atom_restate_observations", 0)
            ),
            bond_reorder_observations=int(
                payload.get("bond_reorder_observations", 0)
            ),
            ring_electronic_observations=int(
                payload.get("ring_electronic_observations", 0)
            ),
        )


def _ring_topology_group_key(template: Any) -> tuple[str, tuple[int, ...]]:
    """Group a semantic template by target topology, not electronics."""

    graph = nx.Graph()
    graph.add_nodes_from(range(int(template.span)))
    graph.add_edges_from(
        (int(left), int(right)) for left, right, _ in template.target_bonds
    )
    cycle_sizes = tuple(
        sorted(len(cycle) for cycle in nx.minimum_cycle_basis(graph))
    )
    return str(template.topology_class), cycle_sizes


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
class RingTeacherPlacementCertificate:
    """Exact chemistry-only support for one placed teacher ring system."""

    placement_key: Any
    supported: bool
    categories: tuple[int, ...] | None = None
    next_category_masks: tuple[tuple[bool, ...], ...] | None = None


@dataclass(frozen=True)
class RingTeacherTemplateCertificate:
    """Placement support aligned to one catalog template."""

    template_index: int
    placements: tuple[RingTeacherPlacementCertificate, ...]


@dataclass(frozen=True)
class RingTeacherSemanticCertificate:
    """Executor-verified teacher certificate computed outside GPU forward."""

    action_is_valid: bool
    templates: tuple[RingTeacherTemplateCertificate, ...]


@dataclass(frozen=True)
class OperatorCapabilities:
    """Immutable editing-operator enumeration capabilities -- the single source of which editing families a
    batch builder must enumerate so its dynamic candidate set matches the teacher support. Threading this ONE
    object through every batch-building path (training loader, eval/validation/test builder) enforces one
    shared representability contract: a teacher generated under editing support is always inside the batch's
    exact candidates. Derive it from the active model (``model.operator_capabilities``) or the edit config;
    never let a batch builder fall back to silent de-novo defaults."""

    compute_ring_grow_support: bool = True
    compute_ring_restates: bool = False
    compute_cyclic_graft: bool = False
    compute_ring_opening: bool = False

    @classmethod
    def de_novo(cls) -> OperatorCapabilities:
        """The de-novo (base-B) capabilities: no editing families, legacy grow macro on."""
        return cls()

    def fingerprint(self) -> str:
        import hashlib

        payload = (
            f"{int(self.compute_ring_grow_support)}{int(self.compute_ring_restates)}"
            f"{int(self.compute_cyclic_graft)}{int(self.compute_ring_opening)}"
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


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
    ring_topology_local_support_log_mass: Tensor | None = None
    ring_teacher_semantic_certificates: tuple[
        RingTeacherSemanticCertificate | None, ...
    ] | None = None
    property_condition_values: Tensor | None = None
    property_condition_mask: Tensor | None = None

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
            ring_topology_local_support_log_mass=(
                None
                if self.ring_topology_local_support_log_mass is None
                else tensor_slice(self.ring_topology_local_support_log_mass)
            ),
            ring_teacher_semantic_certificates=(
                None
                if self.ring_teacher_semantic_certificates is None
                else self.ring_teacher_semantic_certificates[start:stop]
            ),
            property_condition_values=(
                None
                if self.property_condition_values is None
                else tensor_slice(self.property_condition_values)
            ),
            property_condition_mask=(
                None
                if self.property_condition_mask is None
                else tensor_slice(self.property_condition_mask)
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
            ring_topology_local_support_log_mass=(
                None
                if self.ring_topology_local_support_log_mass is None
                else move(self.ring_topology_local_support_log_mass)
            ),
            ring_teacher_semantic_certificates=(
                self.ring_teacher_semantic_certificates
            ),
            property_condition_values=(
                None
                if self.property_condition_values is None
                else move(self.property_condition_values)
            ),
            property_condition_mask=(
                None
                if self.property_condition_mask is None
                else move(self.property_condition_mask)
            ),
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
            ring_topology_local_support_log_mass=(
                None
                if self.ring_topology_local_support_log_mass is None
                else pin(self.ring_topology_local_support_log_mass)
            ),
            ring_teacher_semantic_certificates=(
                self.ring_teacher_semantic_certificates
            ),
            property_condition_values=(
                None
                if self.property_condition_values is None
                else pin(self.property_condition_values)
            ),
            property_condition_mask=(
                None
                if self.property_condition_mask is None
                else pin(self.property_condition_mask)
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


def _legacy_prequotient_graft_tables(
    state: MolecularGraph,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the root-free Graft support used by pre-quotient checkpoints.

    Legacy models normalized their Graft operand logits over every valid
    colored-tree reroute, including automorphism aliases whose canonical
    successor equals the source.  This compact reconstruction is used only to
    thin those virtual jumps during legacy inference; production training uses
    the canonical-successor support compiled in ``ChemistryStateFeatures``.
    """

    n_slots = state.n_atoms
    mask = np.zeros((n_slots, n_slots), dtype=np.bool_)
    removed_neighbors = np.full((n_slots, n_slots), -1, dtype=np.int64)
    real = tuple(int(v) for v in np.flatnonzero(is_element(state.atom_types)))
    graph = nx.Graph()
    graph.add_nodes_from(real)
    graph.add_edges_from(
        (int(a), int(b))
        for offset, a in enumerate(real)
        for b in real[offset + 1 :]
        if int(state.bonds[a, b]) != 0
    )
    if not real or not nx.is_tree(graph) or any(
        int(state.bonds[a, b]) != 1 for a, b in graph.edges()
    ):
        return mask, removed_neighbors

    for moved in real:
        for target in real:
            if moved == target or graph.has_edge(moved, target):
                continue
            path = nx.shortest_path(graph, moved, target)
            removed_neighbor = int(path[1])
            if int(state.implicit_h_counts[target]) < 1:
                continue
            if int(state.implicit_h_counts[removed_neighbor]) >= MAX_H_COUNT:
                continue
            mask[moved, target] = True
            removed_neighbors[moved, target] = removed_neighbor
    return mask, removed_neighbors


def _masked_family_logits(
    raw_family_logits: Tensor,
    action_log_z: Tensor,
    enabled: Tensor,
    *,
    rate_factorization: str,
) -> Tensor:
    """Return finite family scores for a normalized marked rewrite rate.

    ``hierarchical`` learns a categorical family distribution independently of
    the family's executable matches.  ``superposed`` instead adds the exact
    within-family log partition, which is equivalent to globally normalizing
    the scores of all executable canonical rewrite matches.
    """

    if raw_family_logits.shape != action_log_z.shape or enabled.shape != action_log_z.shape:
        raise ValueError("family logits, partitions, and support must have equal shapes")
    if rate_factorization == "superposed":
        raw_family_logits = raw_family_logits + torch.where(
            enabled,
            action_log_z,
            torch.zeros_like(action_log_z),
        )
    elif rate_factorization not in {"hierarchical", "quotient_energy"}:
        raise ValueError("unknown marked-rate factorization")
    masked = raw_family_logits.masked_fill(~enabled, float("-inf"))
    has_legal_mark = enabled.any(dim=-1)
    return torch.where(
        has_legal_mark.unsqueeze(-1),
        masked,
        torch.zeros_like(masked),
    )


def _hierarchical_ring_template_logits(
    template_logits: Tensor,
    support: Tensor,
    topology_group_logits: Tensor,
    topology_group_members: tuple[tuple[int, ...], ...],
) -> Tensor:
    """Compose topology-group and within-group template distributions.

    The flat ring head has one independently learned key per complete ring
    template.  This factorization adds a shared decision over
    ``(topology_class, cycle_sizes)`` groups while retaining the existing
    template logits as conditional residuals within the selected group.  Each
    group is normalized only over templates legal in the current state, so a
    group's probability is not accidentally multiplied by its catalog width.
    """

    if template_logits.ndim != 2 or support.shape != template_logits.shape:
        raise ValueError("ring template logits and support must be aligned matrices")
    if topology_group_logits.ndim != 2:
        raise ValueError("ring topology-group logits must be a matrix")
    if topology_group_logits.shape[0] != template_logits.shape[0]:
        raise ValueError("ring template and topology-group batches must align")
    if topology_group_logits.shape[1] != len(topology_group_members):
        raise ValueError("ring topology-group logits have the wrong width")

    result = template_logits.new_full(template_logits.shape, float("-inf"))
    for group_index, raw_members in enumerate(topology_group_members):
        if not raw_members:
            continue
        members = torch.tensor(
            raw_members,
            dtype=torch.long,
            device=template_logits.device,
        )
        member_logits = template_logits.index_select(1, members)
        member_support = support.index_select(1, members)
        masked = member_logits.masked_fill(~member_support, float("-inf"))
        enabled = member_support.any(dim=1)
        normalizer = torch.logsumexp(masked, dim=1)
        safe_normalizer = torch.where(
            enabled,
            normalizer,
            torch.zeros_like(normalizer),
        )
        conditional = masked - safe_normalizer.unsqueeze(1)
        combined = conditional + topology_group_logits[:, group_index].unsqueeze(1)
        combined = combined.masked_fill(~member_support, float("-inf"))
        result.index_copy_(1, members, combined)
    return result


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
        tuple[int, bool, bool, bool, StateCacheKey], ChemistryStateFeatures
    ]
    | None = None,
    chemistry_feature_cache_limit: int = 2048,
    compute_ring_grow_support: bool = True,
    compute_ring_restates: bool = False,
    compute_cyclic_graft: bool = False,
    compute_ring_opening: bool = False,
    property_condition_values: tuple[tuple[float, ...], ...] | None = None,
    property_condition_mask: tuple[tuple[bool, ...], ...] | None = None,
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
    if (property_condition_values is None) != (property_condition_mask is None):
        raise ValueError("property condition values and mask must be provided together")
    if property_condition_values is not None:
        if len(property_condition_values) != count or len(property_condition_mask) != count:
            raise ValueError("property condition rows do not align with the batch")
        widths = {len(row) for row in property_condition_values}
        mask_widths = {len(row) for row in property_condition_mask}
        if len(widths) != 1 or widths != mask_widths or next(iter(widths)) <= 0:
            raise ValueError("property condition rows must have one positive shared width")
        values_array = np.asarray(property_condition_values, dtype=np.float32)
        mask_array = np.asarray(property_condition_mask, dtype=np.bool_)
        if not np.isfinite(values_array[mask_array]).all():
            raise ValueError("observed property conditions must be finite")
        values_array = np.where(mask_array, values_array, 0.0)
    else:
        values_array = None
        mask_array = None

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
    # Built once (not per state) so the restate validator does not rebuild the executor each call.
    restate_system = de_novo_rewrite_system() if compute_ring_restates else None
    for state in states:
        feature_key = (
            0 if ring_catalog is None else id(ring_catalog),
            bool(use_aromatic_bond_view),
            bool(compute_cyclic_graft),
            bool(compute_ring_opening),
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
            ) = _graph_application_masks(
                state, atom_topo, compute_cyclic_graft=compute_cyclic_graft
            )
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
                    else enumerate_clean_ring_system_deletes(state, ring_catalog)
                    if compute_ring_opening
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
        # ring_system_restate is dead by default: carbon-tree teachers carry no restate marks, and
        # enumerating restates ~doubles collator time. The corrupted-prior editing model turns it on
        # via compute_ring_restates, so aromatize<->de-aromatize become proposable at inference and
        # scoreable as restate teacher marks (both directions; see enumerate_ring_system_restate_actions).
        restate_actions.append(
            enumerate_ring_system_restate_actions(state, system=restate_system)
            if compute_ring_restates
            else ()
        )
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
        property_condition_values=(
            None if values_array is None else torch.from_numpy(values_array)
        ),
        property_condition_mask=(
            None if mask_array is None else torch.from_numpy(mask_array)
        ),
    )


def _relocated_pendant_state(
    state: MolecularGraph,
    moved: int,
    removed_neighbor: int,
    target: int,
    single_h_delta: int,
) -> MolecularGraph:
    """Successor of relocating the pendant fragment rooted at ``moved`` across a single bond: cut the
    ``(moved, removed_neighbor)`` bridge and add ``(moved, target)``. ``moved``'s freed and consumed H
    cancel at order 1; ``removed_neighbor`` regains the freed H and ``target`` spends one for the new
    bond -- matching the executor's ``BondReroute`` bookkeeping exactly."""
    bonds = state.bonds.copy()
    bonds[moved, removed_neighbor] = bonds[removed_neighbor, moved] = 0
    bonds[moved, target] = bonds[target, moved] = 1
    hydrogens = state.implicit_h_counts.copy()
    hydrogens[removed_neighbor] += single_h_delta
    hydrogens[target] -= single_h_delta
    return MolecularGraph(state.atom_types, state.formal_charges, hydrogens, bonds)


def _graph_application_masks(
    state: MolecularGraph,
    atom_topology: np.ndarray,
    *,
    optimize_graft_canonicalization: bool = True,
    compute_cyclic_graft: bool = False,
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
    elif compute_cyclic_graft:
        # B's tree graft is undefined on a cyclic (or non-single-bond) molecule: its colored-tree
        # canonicalizer peels leaves and asserts a tree. Editing still wants graft, so enumerate the
        # well-defined subset -- relocate a PENDANT tree fragment across a SINGLE-bond bridge while the
        # ring core stays fixed. ``moved`` is the bridge endpoint on the acyclic side, so
        # BondReroute(a=moved, b=removed_neighbor, u=moved, v=target) matches the sampler and teacher
        # exactly; the successor quotient uses the general canonical key (the tree one crashes on cycles).
        cyclic_successor_groups: dict[Any, int] = {}
        single_h_delta = int(BOND_CLASS_TO_H_CHANGE[1])
        source_key = canonical_state_key(state)
        for moved, removed_neighbor, target in pendant_graft_candidates(state):
            successor = _relocated_pendant_state(
                state, moved, removed_neighbor, target, single_h_delta
            )
            successor_key = canonical_state_key(successor)
            if successor_key == source_key:
                continue  # a graft to a symmetric position is gauge, not a jump
            group = cyclic_successor_groups.setdefault(
                successor_key, len(cyclic_successor_groups)
            )
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
        rate_factorization: str = "hierarchical",
        ring_candidate_cache_limit: int = 32768,
        property_condition_dim: int = 0,
        empirical_mark_prior_mode: str = "none",
        empirical_mark_priors: FactorizedMarkEmpiricalPriors | None = None,
        ring_family_mass_mode: str = "boolean",
        ring_template_factorization: str = "flat",
        enable_ring_restates: bool = False,
        enable_cyclic_graft: bool = False,
        enable_heteroatom_scan: bool = False,
        enable_ring_opening: bool = False,
        enable_cycle_ops: bool = False,
        enable_ring_grow_macro: bool = True,
        atom_vocabulary: AtomVocabulary | None = None,
        ring_atom_elements: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or message_passing_steps <= 0 or mark_dim <= 0:
            raise ValueError("model dimensions and message-passing steps must be positive")
        if ring_candidate_cache_limit <= 0:
            raise ValueError("ring candidate cache limit must be positive")
        if property_condition_dim < 0:
            raise ValueError("property condition dimension must be non-negative")
        self.ring_catalog = ring_catalog
        # Editing capabilities B lacks: aromatize<->de-aromatize restates and cyclic pendant graft.
        # Off for de-novo B (byte-identical); the corrupted-prior fine-tune (B-edit) turns them on.
        self.enable_ring_restates = bool(enable_ring_restates)
        self.enable_cyclic_graft = bool(enable_cyclic_graft)
        # ungate atom_restate on ring atoms -> heteroatom scanning (pyridine<->benzene); off for de-novo B
        self.enable_heteroatom_scan = bool(enable_heteroatom_scan)
        # swap the carbon-izing structured ring_system_delete for the decoration-preserving clean delete
        # (ring OPENING that keeps heteroatoms; its inverse re-cyclizes); off for de-novo B, on for B-edit
        self.enable_ring_opening = bool(enable_ring_opening)
        # Support-complete COMPOSITIONAL ring closure/opening (cycle_close = bond_insert between two existing
        # atoms; cycle_open = bond_delete of a non-bridge cycle edge). Defines ring-generation support; the
        # ring_system_* macros are an acceleration cache. Off for de-novo B + current B-edit (byte-identical:
        # the cycle heads below are absent), on for the ring-support B-edit fine-tune.
        self.enable_cycle_ops = bool(enable_cycle_ops)
        # Legacy whole-ring-system GROW macro (adds a full ring in one event; B's de-novo ring-adding path).
        # Default True = byte-identical to B / current B-edit. RING_CORE_V1 sets this False so ring ADDITION
        # is purely COMPOSITIONAL (cycle_close): the grow family is masked dead (all-zero template mask ->
        # -inf family logit -> zero rate, never sampled, never a teacher target). The grow head params are
        # retained (byte-identical shape) so a warm-start still loads them; they are simply never activated.
        # Unlike the other enable_* flags this gates an EXISTING capability, hence the True default.
        self.enable_ring_grow_macro = bool(enable_ring_grow_macro)
        # Guard G1 (fail-loud): compositional cycle ops REPLACE the legacy grow macro; they must never both be
        # live, which would give a two-ring-addition-mechanism model outside the RingCore-V1 contract (ring
        # addition would be attributable to both cycle_close and grow). A forgetful launch that passes
        # --cycle-op-mix without --disable-ring-grow-macro is rejected here at construction (every caller --
        # trainer, loader, reward-FT -- hits this single source). Combining the two is a future RING_HYBRID_V2
        # decision, gated by enable_ring_macros, not this flag pair.
        if self.enable_cycle_ops and self.enable_ring_grow_macro:
            raise ValueError(
                "enable_cycle_ops and enable_ring_grow_macro are mutually exclusive: compositional cycle "
                "ops replace the legacy ring_system_grow macro (RingCore-V1). Pass "
                "enable_ring_grow_macro=False when enabling cycle ops (launch: --disable-ring-grow-macro)."
            )
        # Atom-type prediction vocabulary shared by the heads, candidate masks, sampling, and teacher-
        # scoring, so head width and the fiber it is scored against never drift. Default CNOF_VOCABULARY
        # (4 classes) is byte-identical to the historical model; ORGANIC_VOCABULARY (15 (element, valence)
        # classes) covers the drug-like subset with every valence-state its own class (sink-free restate).
        self.atom_vocabulary = atom_vocabulary if atom_vocabulary is not None else CNOF_VOCABULARY
        # Ring-atom element vocabulary (element x role); CNOF by default (byte-identical), +S/P for the
        # organic model so ring_system_grow can build heteroaromatic rings (thiophene, thiazole, phosphole).
        self.ring_atom_elements = (
            tuple(int(e) for e in ring_atom_elements)
            if ring_atom_elements is not None
            else CNOF_RING_ELEMENTS
        )
        self._ring_element_to_index = {int(e): i for i, e in enumerate(self.ring_atom_elements)}
        if tuple(self.ring_atom_elements) != tuple(CNOF_RING_ELEMENTS):
            # The organic ring head is wired, but the ring-GENERATION electronic model (Huckel pi-electron
            # decoder + cached category masks) is still CNOF -- de-novo heteroaromatic BUILDING is a
            # scoped follow-up. Editing existing heteroaromatic rings (read + destroy + restate + graft)
            # is fully organic and uses the default CNOF ring head, so guard the un-wired combination.
            raise NotImplementedError(
                "ring_atom_elements beyond CNOF requires the organic ring-generation electronic model "
                "(de-novo heteroaromatic ring building), which is not wired; the organic editing model "
                "uses the default CNOF ring head (leads' heteroaromatic rings are read and edited, not "
                "built de-novo)."
            )
        self.hidden_dim = int(hidden_dim)
        self.message_passing_steps = int(message_passing_steps)
        self.mark_dim = int(mark_dim)
        self.property_condition_dim = int(property_condition_dim)
        if ring_electronic_mode not in {
            "factorized_local",
            "factorized_contextual",
            "catalog_exact",
        }:
            raise ValueError("unknown ring electronic decoding mode")
        self.ring_electronic_mode = str(ring_electronic_mode)
        if rate_factorization not in {
            "hierarchical",
            "quotient_energy",
            "superposed",
        }:
            raise ValueError("unknown marked-rate factorization")
        self.rate_factorization = str(rate_factorization)
        if empirical_mark_prior_mode not in {"none", "corpus_residual_v1"}:
            raise ValueError("unknown empirical mark prior mode")
        if empirical_mark_prior_mode == "none" and empirical_mark_priors is not None:
            raise ValueError("empirical priors require corpus_residual_v1 mode")
        if empirical_mark_prior_mode != "none" and empirical_mark_priors is None:
            raise ValueError("corpus_residual_v1 mode requires empirical priors")
        if ring_family_mass_mode not in {
            "boolean",
            "catalog_topology_local_support",
        }:
            raise ValueError("unknown ring family mass mode")
        if ring_template_factorization not in {
            "flat",
            "topology_cycle_hierarchical",
        }:
            raise ValueError("unknown ring template factorization")
        self.empirical_mark_prior_mode = str(empirical_mark_prior_mode)
        self.empirical_mark_priors = empirical_mark_priors
        self.ring_family_mass_mode = str(ring_family_mass_mode)
        self.ring_template_factorization = str(ring_template_factorization)
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
        self._ring_topology_local_support_log_mass_cache: OrderedDict[
            tuple[bytes, bytes, bytes, bytes],
            float,
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
        topology_group_indices: dict[tuple[str, tuple[int, ...]], int] = {}
        topology_group_members: list[list[int]] = []
        for template_index, template in enumerate(self.ring_system_templates):
            group_key = _ring_topology_group_key(template)
            group_index = topology_group_indices.get(group_key)
            if group_index is None:
                group_index = len(topology_group_members)
                topology_group_indices[group_key] = group_index
                topology_group_members.append([])
            topology_group_members[group_index].append(template_index)
        self.ring_topology_group_keys = tuple(topology_group_indices)
        self.ring_topology_group_members = tuple(
            tuple(members) for members in topology_group_members
        )
        topology_group_counts = (
            torch.stack(
                tuple(
                    semantic_ring_counts[
                        torch.tensor(members, dtype=torch.long)
                    ].sum()
                    for members in self.ring_topology_group_members
                )
            )
            if self.ring_topology_group_members
            else torch.empty(0, dtype=torch.float32)
        )
        topology_group_log_prior = topology_group_counts.log()
        if len(topology_group_log_prior):
            topology_group_log_prior -= torch.logsumexp(
                topology_group_log_prior,
                dim=0,
            )
        self._ring_topology_group_log_prior_cpu = tuple(
            float(value) for value in topology_group_log_prior
        )
        self.register_buffer(
            "ring_topology_group_log_prior",
            topology_group_log_prior,
            persistent=False,
        )

        if empirical_mark_priors is not None and len(self.atom_vocabulary) != len(CNOF_VOCABULARY):
            # Empirical mark priors are CNOF-shaped (validated against len(CNOF_ATOM_TYPES)); they would
            # silently shape-mismatch the wider organic heads. The organic model uses zero priors.
            raise NotImplementedError(
                "empirical mark priors are CNOF-shaped and unsupported with a non-CNOF atom_vocabulary; "
                "construct the organic model with empirical_mark_priors=None"
            )
        if empirical_mark_priors is None:
            root_atom_log_prior = torch.zeros(len(self.atom_vocabulary))
            connected_atom_order_log_prior = torch.zeros(
                3,
                len(self.atom_vocabulary),
            )
            atom_restate_log_prior = torch.zeros(len(self.atom_vocabulary))
            bond_reorder_log_prior = torch.zeros(3)
            # ring path is (element x role) over the ring-atom element vocabulary (CNOF or +S/P).
            ring_electronic_log_prior = torch.zeros(2 * len(self.ring_atom_elements))
        else:
            root_atom_log_prior = torch.tensor(
                empirical_mark_priors.root_atom_log_probabilities,
                dtype=torch.float32,
            )
            connected_atom_order_log_prior = torch.tensor(
                empirical_mark_priors.connected_atom_order_log_probabilities,
                dtype=torch.float32,
            )
            atom_restate_log_prior = torch.tensor(
                empirical_mark_priors.atom_restate_log_probabilities,
                dtype=torch.float32,
            )
            bond_reorder_log_prior = torch.tensor(
                empirical_mark_priors.bond_reorder_log_probabilities,
                dtype=torch.float32,
            )
            ring_electronic_log_prior = torch.tensor(
                empirical_mark_priors.ring_electronic_log_probabilities,
                dtype=torch.float32,
            )
        for name, values in (
            ("root_atom_log_prior", root_atom_log_prior),
            ("connected_atom_order_log_prior", connected_atom_order_log_prior),
            ("atom_restate_log_prior", atom_restate_log_prior),
            ("bond_reorder_log_prior", bond_reorder_log_prior),
            ("ring_electronic_log_prior", ring_electronic_log_prior),
        ):
            self.register_buffer(name, values, persistent=False)
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
        self.property_condition_encoder: nn.Sequential | None = None
        if self.property_condition_dim:
            self.property_condition_encoder = nn.Sequential(
                nn.Linear(2 * self.property_condition_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            # A conditioned model can be warm-started from an unconditional
            # checkpoint without changing its initial generator.  Training
            # then learns a residual conditional context, including the
            # classifier-free all-missing condition.
            nn.init.zeros_(self.property_condition_encoder[-1].weight)
            nn.init.zeros_(self.property_condition_encoder[-1].bias)
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
        self.grow_root_head = nn.Linear(hidden_dim, len(self.atom_vocabulary))
        self.grow_query = nn.Linear(hidden_dim, mark_dim)
        self.grow_option = nn.Embedding(3 * len(self.atom_vocabulary), mark_dim)
        self.delete_head = nn.Linear(hidden_dim, 1)
        self.restate_head = nn.Linear(hidden_dim, len(self.atom_vocabulary))
        self.reorder_head = nn.Linear(hidden_dim, 3)
        self.graft_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Compositional ring-op heads over PAIR features (added only when enabled -> byte-identical when off).
        # cycle_close: a ring-closing bond over a nonbonded same-component pair x bond order (1/2/3);
        # cycle_open: remove a non-bridge cycle edge (per-edge score). Support-complete; template-free.
        if self.enable_cycle_ops:
            self.cycle_close_head = nn.Linear(hidden_dim, 3)
            self.cycle_open_head = nn.Linear(hidden_dim, 1)

        self.cycle_query = nn.Linear(hidden_dim, mark_dim)
        self.cycle_key = nn.Embedding(max(len(self.cycle_templates), 1), mark_dim)
        self.attach_query = nn.Linear(hidden_dim, mark_dim)
        self.attach_key = nn.Embedding(max(len(self.attach_templates), 1), mark_dim)
        self.ring_system_template_query = nn.Linear(hidden_dim, mark_dim)
        self.ring_system_template_key = nn.Embedding(
            max(len(self.ring_system_templates), 1),
            mark_dim,
        )
        if self.ring_template_factorization == "topology_cycle_hierarchical":
            self.ring_topology_group_head = nn.Linear(
                hidden_dim,
                max(len(self.ring_topology_group_keys), 1),
            )
            nn.init.zeros_(self.ring_topology_group_head.weight)
            nn.init.zeros_(self.ring_topology_group_head.bias)
        else:
            self.ring_topology_group_head = None
        self.ring_system_grow_head = nn.Sequential(
            nn.Linear(4 * hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, 1),
        )
        self.ring_system_atom_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            # ring atom prediction is (element x role) over the ring-atom element vocabulary
            nn.Linear(hidden_dim, len(self.ring_atom_elements)),
        )
        self.ring_system_role_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        if self.ring_electronic_mode == "factorized_contextual":
            ring_category_count = 2 * len(self.ring_atom_elements)
            # A complete ring installation is one atomic rewrite event, but
            # its electronic mark is decoded as a finite sequence. These
            # zero-started pair potentials learn joint ring composition and
            # bonded atom-role correlations (for example N--N or O--O)
            # instead of treating every installed ring atom independently.
            self.ring_system_global_category_pair = nn.Parameter(
                torch.zeros(ring_category_count, ring_category_count)
            )
            self.ring_system_adjacent_category_pair = nn.Parameter(
                torch.zeros(ring_category_count, ring_category_count)
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
            torch.tensor([valence for _, valence in self.atom_vocabulary.classes]),
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

    @property
    def operator_capabilities(self) -> OperatorCapabilities:
        """The editing-family enumeration capabilities this model was configured with -- the single source a
        batch builder must use so its dynamic candidate set matches the teacher support (one shared
        representability contract)."""
        return OperatorCapabilities(
            compute_ring_grow_support=self.enable_ring_grow_macro,
            compute_ring_restates=self.enable_ring_restates,
            compute_cyclic_graft=self.enable_cyclic_graft,
            compute_ring_opening=self.enable_ring_opening,
        )

    def clear_ring_candidate_caches(self) -> None:
        """Release state-dependent chemistry caches without changing rates."""

        self._ring_grow_support_cache.clear()
        self._ring_grow_enablement_certificate_cache.clear()
        self._ring_topology_local_support_log_mass_cache.clear()
        self._ring_template_placement_group_cache.clear()
        self._ring_semantic_decoder_cache.clear()
        self._ring_delete_candidate_cache.clear()
        self._ring_paired_candidate_cache.clear()

    @staticmethod
    def _state_cache_key(
        state: MolecularGraph,
    ) -> tuple[bytes, bytes, bytes, bytes]:
        return molecular_state_cache_key(state)

    def _ring_topology_local_support_log_mass(
        self,
        state: MolecularGraph,
    ) -> float:
        """Return catalog topology mass admitted by the fast local tree DP.

        This is deliberately a topology proposal mass, not an electronic
        executor-support claim.  The existing exact/certificate mask remains
        authoritative for whether the ring family is executable.
        """

        key = self._state_cache_key(state)
        cached = self._ring_topology_local_support_log_mass_cache.get(key)
        if cached is not None:
            self._ring_topology_local_support_log_mass_cache.move_to_end(key)
            return cached
        local_support = ring_system_template_local_support_mask(
            state,
            self.ring_system_templates,
            self.ring_system_template_aliases,
        )
        enabled_groups = tuple(
            any(bool(local_support[index]) for index in members)
            for members in self.ring_topology_group_members
        )
        if not any(enabled_groups):
            value = float("-inf")
        else:
            value = float(
                np.logaddexp.reduce(
                    np.asarray(
                        tuple(
                            log_prior
                            for log_prior, enabled
                            in zip(
                                self._ring_topology_group_log_prior_cpu,
                                enabled_groups,
                            )
                            if enabled
                        ),
                        dtype=np.float64,
                    )
                )
            )
        self._ring_topology_local_support_log_mass_cache[key] = value
        if (
            len(self._ring_topology_local_support_log_mass_cache)
            > self._ring_candidate_cache_limit
        ):
            self._ring_topology_local_support_log_mass_cache.popitem(last=False)
        return value

    def _family_base_logits(
        self,
        batch: FactorizedMarkBatch,
        global_state: Tensor,
    ) -> Tensor:
        logits = self.family_head(global_state)
        if self.ring_family_mass_mode == "boolean":
            return logits
        if batch.ring_topology_local_support_log_mass is None:
            ring_mass = torch.tensor(
                tuple(
                    self._ring_topology_local_support_log_mass(state)
                    for state in batch.states
                ),
                dtype=logits.dtype,
                device=logits.device,
            )
        else:
            ring_mass = batch.ring_topology_local_support_log_mass.to(
                device=logits.device,
                dtype=logits.dtype,
            )
            if tuple(ring_mass.shape) != (batch.batch_size,):
                raise ValueError("ring topology support mass has the wrong shape")
        logits = logits.clone()
        logits[:, MARK_RULE_TO_INDEX["ring_system_grow"]] += ring_mass
        return logits

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
                unresolved: list[int] = []
                for template_index in np.flatnonzero(coarse_support):
                    template_index = int(template_index)
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
                    if not support[template_index]:
                        unresolved.append(template_index)
                # The semantic decoder is the complete factorized path for
                # almost every template.  Fixed catalog assignments remain
                # positive executor-verified witnesses for rare resonance
                # lowerings that its canonical matching does not yet expose.
                # Checking that exact fallback only after semantic failure is
                # the same logical union as the former witness-first order,
                # but avoids enumerating hundreds of redundant RDKit-validated
                # aliases on ordinary rows.
                if unresolved and self.ring_system_electronic_witness_aliases is not None:
                    for template_index in unresolved:
                        if witness_support[template_index] and self._ring_witness_candidates(
                            state,
                            template_index,
                        ):
                            support[template_index] = True
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
            unresolved: list[int] = []
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
                unresolved.append(template_index)
            if not bool(certificate.any()):
                for template_index in (
                    unresolved
                    if self.ring_system_electronic_witness_aliases is not None
                    else ()
                ):
                    if witness_support[template_index] and self._ring_witness_candidates(
                        state,
                        template_index,
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
            cached = (
                enumerate_clean_ring_system_deletes(state, self.ring_catalog)
                if self.enable_ring_opening
                else enumerate_structured_ring_system_deletes(state, self.ring_catalog)
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
        if not self.enable_ring_grow_macro:
            # RING_CORE_V1: legacy whole-ring grow macro disabled -> mask the family dead (all-False support
            # -> family logsumexp = -inf -> zero rate, never sampled/taught). Shape matches the live path.
            shape = (batch.batch_size, len(self.ring_system_templates))
            return (
                torch.zeros(shape, device=self.device),
                torch.zeros(shape, dtype=torch.bool, device=self.device),
            )
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
        if self.ring_template_factorization == "topology_cycle_hierarchical":
            if self.ring_topology_group_head is None:
                raise RuntimeError("hierarchical ring topology head is unavailable")
            topology_group_logits = self.ring_topology_group_head(global_state)[
                :, : len(self.ring_topology_group_keys)
            ]
            topology_group_logits = (
                topology_group_logits
                + self.ring_topology_group_log_prior.unsqueeze(0)
            )
            logits = _hierarchical_ring_template_logits(
                logits,
                mask,
                topology_group_logits,
                self.ring_topology_group_members,
            )
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
        residual_logits = (
            type_logits.unsqueeze(-1) + role_logits.unsqueeze(-2)
        ).reshape(decoder.span, 2 * len(self.ring_atom_elements))
        return residual_logits + self.ring_electronic_log_prior.unsqueeze(0)

    def _ring_semantic_sequence_log_probability(
        self,
        decoder: SemanticRingSystemDecoder,
        category_logits: Tensor,
        categories: tuple[int, ...],
        *,
        next_category_masks: tuple[tuple[bool, ...], ...] | None = None,
    ) -> tuple[Tensor, Tensor]:
        score = category_logits.sum() * 0.0
        legal = score.new_tensor(True, dtype=torch.bool)
        if len(categories) != decoder.span:
            return score, score.new_tensor(False, dtype=torch.bool)
        if next_category_masks is not None and len(next_category_masks) != decoder.span:
            return score, score.new_tensor(False, dtype=torch.bool)
        prefix: tuple[int, ...] = ()
        for position, category in enumerate(categories):
            contextual_logits = self._ring_semantic_contextual_logits(
                decoder,
                category_logits,
                position=position,
                prefix=prefix,
            )
            mask = torch.tensor(
                (
                    semantic_ring_next_category_mask(decoder, prefix)
                    if next_category_masks is None
                    else next_category_masks[position]
                ),
                dtype=torch.bool,
                device=self.device,
            )
            if not bool(mask.any()) or not 0 <= int(category) < mask.numel():
                return score, score.new_tensor(False, dtype=torch.bool)
            legal = legal & mask[int(category)]
            score = score + torch.log_softmax(
                contextual_logits.masked_fill(~mask, float("-inf")),
                dim=-1,
            )[int(category)]
            prefix = (*prefix, int(category))
        return score, legal

    def _ring_semantic_contextual_logits(
        self,
        decoder: SemanticRingSystemDecoder,
        category_logits: Tensor,
        *,
        position: int,
        prefix: tuple[int, ...],
    ) -> Tensor:
        """Add exact-prefix joint electronic potentials to one ring position.

        The global term learns ring-composition correlations; the adjacent
        term learns correlations only across bonds in the installed ring
        system.  Symmetrization prevents an arbitrary atom-slot decoding order
        from defining different pair energies for the same two categories.
        """

        if not 0 <= position < decoder.span:
            raise ValueError("ring semantic position is outside the decoder")
        if len(prefix) != position:
            raise ValueError("ring semantic prefix does not match the position")
        base = category_logits[position]
        if self.ring_electronic_mode != "factorized_contextual" or not prefix:
            return base
        category_count = 2 * len(self.ring_atom_elements)
        if base.shape != (category_count,):
            raise ValueError("ring semantic category table has the wrong shape")
        if any(not 0 <= int(value) < category_count for value in prefix):
            raise ValueError("ring semantic prefix contains an invalid category")

        global_pair = 0.5 * (
            self.ring_system_global_category_pair
            + self.ring_system_global_category_pair.transpose(0, 1)
        )
        adjacent_pair = 0.5 * (
            self.ring_system_adjacent_category_pair
            + self.ring_system_adjacent_category_pair.transpose(0, 1)
        )
        residual = base.new_zeros((category_count,))
        target_slot = int(decoder.placement.system_atoms[position])
        internal_edges = {
            frozenset((int(item.a), int(item.b)))
            for items in (
                decoder.placement.scaffold_bonds,
                decoder.placement.bond_insertions,
            )
            for item in items
        }
        internal_edges.update(
            frozenset((int(item.a), int(item.b)))
            for item in decoder.placement.bond_reorders
        )
        for previous_position, previous_category in enumerate(prefix):
            previous_category = int(previous_category)
            residual = residual + global_pair[previous_category]
            previous_slot = int(
                decoder.placement.system_atoms[previous_position]
            )
            if frozenset((previous_slot, target_slot)) in internal_edges:
                residual = residual + adjacent_pair[previous_category]
        return base + residual

    def ring_teacher_semantic_certificate(
        self,
        state: MolecularGraph,
        action: RingSystemGrow,
    ) -> RingTeacherSemanticCertificate:
        """Compile the exact teacher-only semantic DP outside neural forward.

        The masks depend only on the valid molecular state, the catalog, and
        the teacher action.  Computing them in DataLoader/CPU compilation
        workers preserves the identical normalized mark probability while
        preventing recursive NetworkX executor checks from serializing the
        A100 training loop.
        """

        action_is_valid = bool(is_valid_ring_system_grow(state, action))
        if not action_is_valid:
            return RingTeacherSemanticCertificate(False, ())
        teacher_key = ring_system_placement_key(ring_system_placement(action))
        template_certificates = []
        for template_index in matching_ring_system_template_indices(
            action,
            self.ring_system_templates,
        ):
            placement_groups = self._ring_template_placement_groups(
                state,
                int(template_index),
            )
            placement_certificates = []
            for group in placement_groups:
                placement = group[0]
                decoder = self._ring_semantic_decoder(state, placement)
                supported = bool(semantic_ring_prefix_is_completable(decoder, ()))
                categories = None
                masks = None
                if supported and ring_system_placement_key(placement) == teacher_key:
                    try:
                        categories = semantic_ring_categories_for_action(decoder, action)
                    except (KeyError, ValueError):
                        categories = None
                    if categories is not None:
                        prefix: tuple[int, ...] = ()
                        rows = []
                        for category in categories:
                            rows.append(semantic_ring_next_category_mask(decoder, prefix))
                            prefix = (*prefix, int(category))
                        masks = tuple(rows)
                placement_certificates.append(
                    RingTeacherPlacementCertificate(
                        placement_key=ring_system_placement_key(placement),
                        supported=supported,
                        categories=categories,
                        next_category_masks=masks,
                    )
                )
            template_certificates.append(
                RingTeacherTemplateCertificate(
                    template_index=int(template_index),
                    placements=tuple(placement_certificates),
                )
            )
        return RingTeacherSemanticCertificate(
            action_is_valid=True,
            templates=tuple(template_certificates),
        )

    def _ring_supported_placement_tables(
        self,
        state: MolecularGraph,
        placement_groups: tuple[tuple[RingSystemPlacement, ...], ...],
        node: Tensor,
        global_state: Tensor,
        pair: Tensor,
        *,
        certificate: RingTeacherTemplateCertificate | None = None,
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
        if certificate is None:
            placement_support = tuple(
                semantic_ring_prefix_is_completable(decoder, ()) for decoder in decoders
            )
        else:
            if len(certificate.placements) != len(placements):
                raise RuntimeError("ring teacher certificate placement count changed")
            observed_keys = tuple(ring_system_placement_key(item) for item in placements)
            expected_keys = tuple(item.placement_key for item in certificate.placements)
            if observed_keys != expected_keys:
                raise RuntimeError("ring teacher certificate placement order changed")
            placement_support = tuple(
                bool(item.supported) for item in certificate.placements
            )
        placement_mask = torch.tensor(
            placement_support,
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
                tuple(self._ring_element_to_index[int(atom_type)] for atom_type in candidate.atom_types),
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
        family_logits = _masked_family_logits(
            self._family_base_logits(batch, global_state),
            action_log_z,
            enabled,
            rate_factorization=self.rate_factorization,
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

        return self.sample_rewrite_mark_conditioned(
            state,
            time,
            rng,
            property_values=None,
            property_mask=None,
        )

    @torch.no_grad()
    def sample_rewrite_mark_conditioned(
        self,
        state: MolecularGraph,
        time: float,
        rng: np.random.Generator,
        *,
        property_values: tuple[float, ...] | None,
        property_mask: tuple[bool, ...] | None = None,
    ) -> SampledRewriteMark:
        """Sample one legal mark under an optional standardized target vector."""

        if (property_values is None) != (property_mask is None):
            raise ValueError("property values and mask must be provided together")
        if property_values is not None:
            if self.property_condition_dim == 0:
                raise ValueError("unconditional model cannot accept property targets")
            if len(property_values) != self.property_condition_dim or len(
                property_mask
            ) != self.property_condition_dim:
                raise ValueError("property target has the wrong dimension")
            values = torch.tensor((property_values,), dtype=torch.float32)
            mask = torch.tensor((property_mask,), dtype=torch.bool)
        else:
            values = None
            mask = None

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
                compute_ring_grow_support=self.enable_ring_grow_macro,
                compute_ring_restates=self.enable_ring_restates,
                compute_cyclic_graft=self.enable_cyclic_graft,
                compute_ring_opening=self.enable_ring_opening,
            )
            self._sampling_state_cache[cache_key] = cached_batch
            if len(self._sampling_state_cache) > self._sampling_state_cache_limit:
                self._sampling_state_cache.popitem(last=False)
        else:
            self._sampling_state_cache.move_to_end(cache_key)
        batch = replace(
            cached_batch,
            times=torch.tensor((float(time),), dtype=torch.float32),
            property_condition_values=values,
            property_condition_mask=mask,
        ).to(self.device)
        node, global_state, pair = self._encode_batch(batch)
        masks, logits, action_log_z = self._action_tables(
            batch,
            node,
            global_state,
            pair,
            require_exact_ring_support=False,
        )
        nonself_graft_mask = masks["bond_reroute"][0].clone()
        legacy_virtual_grafts = bool(
            getattr(self, "virtualize_legacy_self_grafts", False)
        )
        if legacy_virtual_grafts:
            raw_mask_array, raw_removed_array = _legacy_prequotient_graft_tables(
                state
            )
            raw_mask = torch.from_numpy(raw_mask_array).to(self.device)
            raw_removed = torch.from_numpy(raw_removed_array).to(self.device)
            masks = dict(masks)
            masks["bond_reroute"] = raw_mask.unsqueeze(0)
            batch = replace(
                batch,
                graft_remove_neighbors=raw_removed.unsqueeze(0),
            )
            action_log_z = action_log_z.clone()
            action_log_z[0, MARK_RULE_TO_INDEX["bond_reroute"]] = (
                _masked_logsumexp(
                    logits["bond_reroute"][0].unsqueeze(0),
                    raw_mask.unsqueeze(0),
                )[0]
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
            family_logits = _masked_family_logits(
                self._family_base_logits(batch, global_state)[0],
                action_log_z[0],
                enabled,
                rate_factorization=self.rate_factorization,
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
            if legacy_virtual_grafts and rule_name == "bond_reroute":
                moved = int(action.u)
                target = int(action.v)
                if not bool(nonself_graft_mask[moved, target]):
                    return SampledRewriteMark(
                        hazard,
                        "<VIRTUAL_GRAFT>",
                        None,
                    )
            # cycle ops: the model family is slot 5/6 (cycle_insert/cycle_attach), but the executor rule is
            # keyed on the action type -> apply via bond_insert/bond_delete.
            applied_rule = rule_name
            if isinstance(action, BondInsert):
                applied_rule = "bond_insert"
            elif isinstance(action, BondDelete):
                applied_rule = "bond_delete"
            return SampledRewriteMark(hazard, applied_rule, action)
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
                atom_type = int(self.atom_vocabulary.element_of(atom_index))
                return AtomInsert(
                    slot,
                    atom_type,
                    0,
                    int(self.cnof_valences[atom_index]),
                    (),
                )
            neighbor, order_index, atom_index = coordinate
            order = order_index + 1
            atom_type = int(self.atom_vocabulary.element_of(atom_index))
            return AtomInsert(
                slot,
                atom_type,
                0,
                int(self.cnof_valences[atom_index]) - order,
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
            atom_type = int(self.atom_vocabulary.element_of(atom_index))
            bond_valence = sum(
                int(BOND_CLASS_TO_H_CHANGE[int(order)]) for order in state.bonds[vertex]
            )
            return AtomRestate(
                vertex,
                atom_type,
                0,
                int(self.cnof_valences[atom_index]) - bond_valence,
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
            if self.enable_cycle_ops:  # cycle_close: sample (a, b, order) -> BondInsert
                a, b, order_index = _sample_masked_coordinate(
                    logits["cycle_insert"][0], masks["cycle_insert"][0], rng
                )
                return BondInsert(a, b, order_index + 1)
            (template_index,) = _sample_masked_coordinate(
                logits["cycle_insert"][0],
                masks["cycle_insert"][0],
                rng,
            )
            template = self.cycle_templates[template_index]
            return template.instantiate(null_slots[: template.span])
        if rule_name == "cycle_attach":
            if self.enable_cycle_ops:  # cycle_open: sample (a, b) -> BondDelete
                a, b = _sample_masked_coordinate(
                    logits["cycle_attach"][0], masks["cycle_attach"][0], rng
                )
                return BondDelete(a, b)
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
            excluded_template_indices = tuple(
                int(index)
                for index in getattr(
                    self,
                    "excluded_sampling_ring_template_indices",
                    (),
                )
            )
            if excluded_template_indices:
                if min(excluded_template_indices) < 0 or max(
                    excluded_template_indices
                ) >= len(self.ring_system_templates):
                    raise ValueError("excluded sampling ring-template index is invalid")
                exact_template_mask[
                    torch.tensor(
                        excluded_template_indices,
                        dtype=torch.long,
                        device=self.device,
                    )
                ] = False
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
                contextual_logits = self._ring_semantic_contextual_logits(
                    decoder,
                    category_logits,
                    position=position,
                    prefix=prefix,
                )
                category_mask = torch.tensor(
                    semantic_ring_next_category_mask(decoder, prefix),
                    dtype=torch.bool,
                    device=self.device,
                )
                (category,) = _sample_masked_coordinate(
                    contextual_logits,
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
        if self.property_condition_dim:
            values = batch.property_condition_values
            mask = batch.property_condition_mask
            if values is None or mask is None:
                values = torch.zeros(
                    (batch.batch_size, self.property_condition_dim),
                    dtype=time_state.dtype,
                    device=time_state.device,
                )
                mask = torch.zeros_like(values, dtype=torch.bool)
            if tuple(values.shape) != (batch.batch_size, self.property_condition_dim):
                raise ValueError("property condition values have the wrong shape")
            if tuple(mask.shape) != tuple(values.shape):
                raise ValueError("property condition mask has the wrong shape")
            values = values.to(device=time_state.device, dtype=time_state.dtype)
            mask = mask.to(device=time_state.device, dtype=torch.bool)
            if not bool(torch.isfinite(values[mask]).all()):
                raise ValueError("observed property conditions must be finite")
            condition_input = torch.cat(
                (torch.where(mask, values, torch.zeros_like(values)), mask.to(values.dtype)),
                dim=-1,
            )
            assert self.property_condition_encoder is not None
            context_state = time_state + self.property_condition_encoder(condition_input)
        else:
            if (
                batch.property_condition_values is not None
                or batch.property_condition_mask is not None
            ):
                raise ValueError("unconditional model received property conditions")
            context_state = time_state
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
            tiled_time = context_state.unsqueeze(1).expand(-1, n_slots, -1)
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
        global_state = self.global_project(torch.cat((pooled, context_state), dim=-1))

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

        root_grow_logits = (
            self.grow_root_head(global_state)
            + self.root_atom_log_prior.unsqueeze(0)
        )
        # A valence class is not automatically a legal isolated atom.  The
        # persistent graph representation caps implicit H at MAX_H_COUNT, so
        # hypervalent neutral classes such as S(VI), P(V) and I(V) cannot be
        # instantiated from the null state with zero heavy-atom bonds.  Keep
        # them available for connected insertion/restatement when the heavy
        # bond sum brings the implied H count into range.
        root_atom_mask = (
            (self.cnof_valences >= 0)
            & (self.cnof_valences <= MAX_H_COUNT)
        )
        root_grow_mask = (
            ((n_real == 0) & (n_null > 0)).unsqueeze(-1)
            & root_atom_mask.unsqueeze(0)
        )
        grow_query = self.grow_query(node)
        grow_option = self.grow_option.weight.reshape(
            3,
            len(self.atom_vocabulary),
            self.mark_dim,
        )
        connected_grow_logits = torch.einsum(
            "bnr,otr->bnot",
            grow_query,
            grow_option,
        ) / sqrt(self.mark_dim)
        connected_grow_logits = (
            connected_grow_logits
            + self.connected_atom_order_log_prior.unsqueeze(0).unsqueeze(0)
        )
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

        restate_logits = (
            self.restate_head(node)
            + self.atom_restate_log_prior.unsqueeze(0).unsqueeze(0)
        )
        bond_valence = torch.zeros_like(hydrogens)
        for order, delta in enumerate(BOND_CLASS_TO_H_CHANGE):
            if order == 0:
                continue
            bond_valence = bond_valence + (batch.bonds == order).sum(dim=-1) * int(delta)
        restate_h = self.cnof_valences.view(1, 1, -1) - bond_valence.unsqueeze(-1)
        # Peripheral by default; the editing model ungates ring atoms so heteroatom scanning
        # (e.g. pyridine<->benzene) is a scoreable restate. The valence + no-op checks below still gate it.
        restate_site = real if self.enable_heteroatom_scan else (real & (batch.atom_topology == 0))
        restate_mask = (
            restate_site.unsqueeze(-1)
            & (restate_h >= 0)
            & (restate_h <= MAX_H_COUNT)
            & (
                (batch.atom_types.unsqueeze(-1) != torch.tensor(self.atom_vocabulary.element_index, device=device))
                | (batch.implicit_h_counts.unsqueeze(-1) != restate_h)
                | (batch.formal_charges.unsqueeze(-1) != 0)
            )
        )
        masks["atom_restate"] = restate_mask
        logits["atom_restate"] = restate_logits

        reorder_logits = (
            self.reorder_head(pair)
            + self.bond_reorder_log_prior.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )
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

        if self.enable_cycle_ops:
            # COMPOSITIONAL ring support: override the dead slots 5/6 with cycle_close/cycle_open. Same
            # pair layout as bond_reorder, so family_z + the dicts + the per-family logsumexp all apply.
            # cycle_close (slot "cycle_insert" -> executor bond_insert): add a ring-closing bond over a
            # nonbonded pair x order (H decreases by the order, so both atoms need H >= order).
            order_col = order_values.view(1, 1, 1, 3)
            old_order = batch.bonds.unsqueeze(-1)
            cycle_logits = self.cycle_close_head(pair)
            cycle_mask = (
                upper.view(1, n_slots, n_slots, 1)
                & (old_order == 0)
                & (hydrogens.unsqueeze(2).unsqueeze(-1) >= order_col)
                & (hydrogens.unsqueeze(1).unsqueeze(-1) >= order_col)
            )
            # cycle_open (slot "cycle_attach" -> executor bond_delete): remove a non-bridge cycle edge (H
            # increases by the removed order, so both atoms' resulting H must stay <= MAX_H_COUNT).
            attach_logits = self.cycle_open_head(pair).squeeze(-1)
            edge_order = batch.bonds
            attach_mask = (
                upper
                & batch.cycle_edge_mask
                & ((hydrogens.unsqueeze(2) + edge_order) <= MAX_H_COUNT)
                & ((hydrogens.unsqueeze(1) + edge_order) <= MAX_H_COUNT)
            )
        else:
            cycle_logits = _template_logits(
                self.cycle_query(global_state),
                self.cycle_key.weight[: len(self.cycle_templates)],
            )
            cycle_mask = (n_real == 0).unsqueeze(-1) & (
                n_null.unsqueeze(-1) >= self.cycle_spans.unsqueeze(0)
            )
            attach_logits = _template_logits(
                self.attach_query(node + global_state.unsqueeze(1)),
                self.attach_key.weight[: len(self.attach_templates)],
            )
            attach_mask = (
                real.unsqueeze(-1)
                & (n_null.view(-1, 1, 1) >= self.attach_spans.view(1, 1, -1))
                & (hydrogens.unsqueeze(-1) >= self.attach_required_h.view(1, 1, -1))
            )
        masks["cycle_insert"] = cycle_mask
        logits["cycle_insert"] = cycle_logits
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
            family_name = rule_name
            if self.enable_cycle_ops and rule_name in _CYCLE_OP_EXECUTOR_TO_FAMILY:
                family_name = _CYCLE_OP_EXECUTOR_TO_FAMILY[rule_name]
            if family_name not in MARK_RULE_TO_INDEX:
                raise ValueError(f"unsupported factorized teacher family: {rule_name}")
            family_index = MARK_RULE_TO_INDEX[family_name]
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
            atom_index = self.atom_vocabulary.class_index(
                int(action.atom_type),
                sum(int(BOND_CLASS_TO_H_CHANGE[int(o)]) for _, o in action.neighbors),
                int(action.implicit_h_count),
                formal_charge=int(action.formal_charge),
            )
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
            bond_valence = sum(
                int(BOND_CLASS_TO_H_CHANGE[int(o)])
                for o in batch.bonds[batch_index, int(action.v)]
            )
            atom_index = self.atom_vocabulary.class_index(
                int(action.atom_type), bond_valence, int(action.implicit_h_count),
                formal_charge=int(action.formal_charge),
            )
            key = (batch_index, int(action.v), atom_index)
            return logits["atom_restate"][key], masks["atom_restate"][key]
        if isinstance(action, BondReorder):
            a, b = sorted((int(action.a), int(action.b)))
            order_index = _ORDER_TO_INDEX[int(action.new_order)]
            key = (batch_index, a, b, order_index)
            return logits["bond_reorder"][key], masks["bond_reorder"][key]
        if isinstance(action, BondInsert):
            # cycle_close teacher: scored against slot 5 (holds cycle_close when enable_cycle_ops). Pair x
            # order, per-coordinate like bond_reorder (symmetric closures sum correctly, no successor group).
            a, b = sorted((int(action.a), int(action.b)))
            order_index = _ORDER_TO_INDEX[int(action.order)]
            key = (batch_index, a, b, order_index)
            return logits["cycle_insert"][key], masks["cycle_insert"][key]
        if isinstance(action, BondDelete):
            # cycle_open teacher: scored against slot 6 (holds cycle_open when enable_cycle_ops). Per-edge.
            a, b = sorted((int(action.a), int(action.b)))
            key = (batch_index, a, b)
            return logits["cycle_attach"][key], masks["cycle_attach"][key]
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
            certificates = getattr(
                batch,
                "ring_teacher_semantic_certificates",
                None,
            )
            teacher_certificate = (
                None if certificates is None else certificates[batch_index]
            )
            action_is_valid = (
                bool(is_valid_ring_system_grow(batch.states[batch_index], action))
                if teacher_certificate is None
                else bool(teacher_certificate.action_is_valid)
            )
            if not action_is_valid:
                zero = logits["ring_system_grow"][batch_index].sum() * 0.0
                return zero, zero.new_tensor(False, dtype=torch.bool)
            template_indices = (
                matching_ring_system_template_indices(
                    action,
                    self.ring_system_templates,
                )
                if teacher_certificate is None
                else tuple(
                    item.template_index for item in teacher_certificate.templates
                )
            )
            template_certificates = (
                {}
                if teacher_certificate is None
                else {
                    item.template_index: item for item in teacher_certificate.templates
                }
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
                template_certificate = template_certificates.get(template_index)
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
                        certificate=template_certificate,
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
                    placement_certificate = (
                        None
                        if template_certificate is None
                        else template_certificate.placements[placement_index]
                    )
                    if placement_certificate is None:
                        try:
                            categories = semantic_ring_categories_for_action(
                                decoder,
                                action,
                            )
                        except (KeyError, ValueError):
                            continue
                        next_category_masks = None
                    else:
                        categories = placement_certificate.categories
                        next_category_masks = (
                            placement_certificate.next_category_masks
                        )
                        if categories is None or next_category_masks is None:
                            continue
                    label_score, labels_legal = (
                        self._ring_semantic_sequence_log_probability(
                            decoder,
                            category_logits,
                            categories,
                            next_category_masks=next_category_masks,
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
    "FactorizedMarkEmpiricalPriors",
    "FactorizedMarkPrediction",
    "FactorizedTraceletRateModel",
    "MARK_RULE_NAMES",
    "SampledRewriteMark",
    "factorized_mark_bregman_loss",
    "prepare_factorized_mark_batch",
]
