"""Permutation-equivariant marked rewrite-rate model for the first base gate."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin

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
    BOND_CLASSES,
    H_COUNT_CLASSES,
    K,
    M,
    MolecularGraph,
    NULL_IDX,
    SCAR_IDX,
)
from compose_v4.rewrite.fiber import (
    ActionFiberSpec,
    MarkedTransition,
    enumerate_action_fiber,
    group_transition_indices_by_successor,
)
from compose_v4.rewrite.factorized_fiber import (
    FactorizedFiber,
    enumerate_factorized_cnof_fiber,
)
from compose_v4.rewrite.operators import (
    AtomDelete,
    AtomInsert,
    AtomRestate,
    BondDelete,
    BondInsert,
    BondReorder,
)


RULE_NAMES = (
    "atom_insert",
    "atom_delete",
    "atom_restate",
    "bond_insert",
    "bond_delete",
    "bond_reorder",
)
RULE_TO_INDEX = {name: index for index, name in enumerate(RULE_NAMES)}


@dataclass(frozen=True)
class FiberRatePrediction:
    transitions: tuple[MarkedTransition, ...]
    marked_rates: Tensor
    grouped_indices: dict[str, tuple[int, ...]]
    action_kl_to_prior: Tensor | None = None
    squared_log_hazard_tilt: Tensor | None = None

    @property
    def total_hazard(self) -> Tensor:
        return self.marked_rates.sum()

    def successor_rate(self, successor_key: str) -> Tensor:
        indices = self.grouped_indices.get(successor_key, ())
        if not indices:
            return self.marked_rates.new_zeros(())
        index = torch.as_tensor(
            indices,
            dtype=torch.long,
            device=self.marked_rates.device,
        )
        return self.marked_rates.index_select(0, index).sum()

    def successor_rate_dict(self) -> dict[str, Tensor]:
        return {key: self.successor_rate(key) for key in self.grouped_indices}


class WholeGraphRateModel(nn.Module):
    """Score executable rewrites from the current whole graph and time only.

    The model never receives a target molecule, trace index, construction root,
    or fragment identity. Marked padding-slot aliases share a score by design;
    their rates are summed at complete successors by ``FiberRatePrediction``.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        message_passing_steps: int = 3,
        *,
        use_rewrite_context: bool = False,
        use_topology_context: bool = False,
        use_aromatic_bond_view: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or message_passing_steps <= 0:
            raise ValueError("hidden_dim and message_passing_steps must be positive")
        self.hidden_dim = hidden_dim
        self.message_passing_steps = message_passing_steps
        self.use_rewrite_context = use_rewrite_context
        self.use_topology_context = use_topology_context
        self.use_aromatic_bond_view = bool(use_aromatic_bond_view)
        self._topology_cache: dict[
            tuple[bytes, bytes],
            tuple[np.ndarray, np.ndarray, np.ndarray],
        ] = {}
        self._aromatic_bond_cache: dict[
            tuple[bytes, bytes, bytes, bytes],
            np.ndarray,
        ] = {}

        self.atom_embedding = nn.Embedding(M, hidden_dim)
        self.charge_embedding = nn.Embedding(K, hidden_dim)
        self.hydrogen_embedding = nn.Embedding(H_COUNT_CLASSES, hidden_dim)
        self.bond_embedding = nn.Embedding(BOND_CLASSES, hidden_dim)
        self.rule_embedding = nn.Embedding(len(RULE_NAMES), hidden_dim)
        self.empty_state = nn.Parameter(torch.zeros(hidden_dim))
        if use_rewrite_context:
            self.path_distance_embedding = nn.Embedding(34, hidden_dim)
        else:
            self.path_distance_embedding = None
        if use_rewrite_context or use_topology_context:
            self.structure_encoder = nn.Sequential(
                nn.Linear(4, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.structure_encoder = None
        if use_topology_context:
            self.atom_ring_embedding = nn.Embedding(RING_SIZE_BUCKETS, hidden_dim)
            self.closure_ring_embedding = nn.Embedding(RING_SIZE_BUCKETS, hidden_dim)
            self.ring_system_embedding = nn.Embedding(FUSED_SIZE_BUCKETS, hidden_dim)
        else:
            self.atom_ring_embedding = None
            self.closure_ring_embedding = None
            self.ring_system_embedding = None

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
        self.pair_project = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.global_project = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rate_head = nn.Sequential(
            nn.Linear(5 * hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_rate_bias = nn.Parameter(torch.tensor(-2.0))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def predict_fiber(
        self,
        state: MolecularGraph,
        time: float,
        *,
        transitions: tuple[MarkedTransition, ...] | None = None,
        spec: ActionFiberSpec | None = None,
    ) -> FiberRatePrediction:
        marked = transitions
        if marked is None:
            marked = enumerate_action_fiber(state, spec=spec)
        rates = self.forward(state, time, marked)
        return FiberRatePrediction(
            transitions=marked,
            marked_rates=rates,
            grouped_indices=group_transition_indices_by_successor(marked),
        )

    def forward(
        self,
        state: MolecularGraph,
        time: float,
        transitions: tuple[MarkedTransition, ...],
    ) -> Tensor:
        if not 0.0 <= float(time) <= 1.0:
            raise ValueError("normalized model time must lie in [0, 1]")
        if not transitions:
            return torch.empty(0, device=self.device)

        node_states, global_state, time_state, bonds = self._encode_graph(state, time)
        topology = self._topology_features(state)
        action_rows = self._encode_action_rows(
            transitions,
            state,
            node_states,
            global_state,
            time_state,
            bonds,
            topology,
        )
        logits = self.rate_head(torch.stack(action_rows)).squeeze(-1)
        return F.softplus(logits + self.log_rate_bias)

    def action_logits(
        self,
        state: MolecularGraph,
        time: float,
        transitions: tuple[MarkedTransition, ...],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return action logits plus encoded global/time states."""

        node_states, global_state, time_state, bonds = self._encode_graph(state, time)
        topology = self._topology_features(state)
        action_rows = self._encode_action_rows(
            transitions,
            state,
            node_states,
            global_state,
            time_state,
            bonds,
            topology,
        )
        if action_rows:
            logits = self.rate_head(torch.stack(action_rows)).squeeze(-1)
        else:
            logits = torch.empty(0, device=self.device)
        return logits, global_state, time_state

    def _encode_action_rows(
        self,
        transitions: tuple[MarkedTransition, ...],
        state: MolecularGraph,
        node_states: Tensor,
        global_state: Tensor,
        time_state: Tensor,
        bonds: Tensor,
        topology: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    ) -> list[Tensor]:
        """Encode an action fiber; subclasses may share repeated substructure."""

        return [
            self._encode_action(
                transition,
                state,
                node_states,
                global_state,
                time_state,
                bonds,
                topology,
            )
            for transition in transitions
        ]

    def _encode_graph(
        self,
        state: MolecularGraph,
        time: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        atom_types = torch.as_tensor(
            state.atom_types,
            dtype=torch.long,
            device=self.device,
        )
        charges = torch.as_tensor(
            state.formal_charges + 2,
            dtype=torch.long,
            device=self.device,
        )
        hydrogens = torch.as_tensor(
            state.implicit_h_counts,
            dtype=torch.long,
            device=self.device,
        )
        neural_bonds = self._neural_bond_classes(state)
        bonds = torch.as_tensor(neural_bonds, dtype=torch.long, device=self.device)
        real = (atom_types != NULL_IDX) & (atom_types != SCAR_IDX)
        real_float = real.to(dtype=self.atom_embedding.weight.dtype).unsqueeze(-1)
        time_features = torch.tensor(
            [float(time), sin(pi * float(time)), cos(pi * float(time))],
            dtype=self.atom_embedding.weight.dtype,
            device=self.device,
        )
        time_state = self.time_encoder(time_features)

        node_states = (
            self.atom_embedding(atom_types)
            + self.charge_embedding(charges)
            + self.hydrogen_embedding(hydrogens)
        ) * real_float
        edge_mask = (bonds != 0).unsqueeze(-1)
        edge_states = self.bond_embedding(bonds)
        for update, norm in zip(self.update_networks, self.update_norms):
            sender = self.message_node(node_states).unsqueeze(0).expand(
                state.n_atoms,
                -1,
                -1,
            )
            messages = (sender + edge_states) * edge_mask
            aggregated = messages.sum(dim=1)
            tiled_time = time_state.unsqueeze(0).expand(state.n_atoms, -1)
            delta = update(torch.cat((node_states, aggregated, tiled_time), dim=-1))
            node_states = norm(node_states + delta) * real_float

        count = real_float.sum().clamp_min(1.0)
        pooled = node_states.sum(dim=0) / count
        if not bool(real.any()):
            pooled = self.empty_state
        if self.structure_encoder is not None:
            n_real = real.sum().to(dtype=pooled.dtype)
            edge_count = torch.triu(bonds != 0, diagonal=1).sum().to(dtype=pooled.dtype)
            cycle_rank = torch.clamp(edge_count - n_real + (n_real > 0), min=0.0)
            normalizer = float(max(state.n_atoms, 1))
            structural_features = torch.stack(
                (
                    n_real / normalizer,
                    edge_count / normalizer,
                    cycle_rank / normalizer,
                    hydrogens[real].to(dtype=pooled.dtype).sum()
                    / max(4.0 * normalizer, 1.0),
                )
            )
            pooled = pooled + self.structure_encoder(structural_features)
        global_state = self.global_project(torch.cat((pooled, time_state), dim=-1))
        return node_states, global_state, time_state, bonds

    def _neural_bond_classes(self, state: MolecularGraph) -> np.ndarray:
        if not self.use_aromatic_bond_view:
            return state.bonds
        key = (
            state.atom_types.tobytes(),
            state.formal_charges.tobytes(),
            state.implicit_h_counts.tobytes(),
            state.bonds.tobytes(),
        )
        cached = self._aromatic_bond_cache.get(key)
        if cached is None:
            cached = resonance_invariant_bond_classes(state)
            if len(self._aromatic_bond_cache) >= 20_000:
                self._aromatic_bond_cache.clear()
            self._aromatic_bond_cache[key] = cached
        return cached

    def _encode_action(
        self,
        transition: MarkedTransition,
        state: MolecularGraph,
        node_states: Tensor,
        global_state: Tensor,
        time_state: Tensor,
        bonds: Tensor,
        topology: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> Tensor:
        action = transition.action
        local = node_states.new_zeros(self.hidden_dim)
        target = node_states.new_zeros(self.hidden_dim)

        if isinstance(action, AtomInsert):
            target = self._atom_state_embedding(
                action.atom_type,
                action.formal_charge,
                action.implicit_h_count,
            )
            for neighbor, order in action.neighbors:
                local = local + node_states[int(neighbor)]
                target = target + self.bond_embedding.weight[int(order)]
                if topology is not None:
                    atom_topology, _, _ = topology
                    local = local + self.atom_ring_embedding.weight[
                        int(atom_topology[int(neighbor)])
                    ]
        elif isinstance(action, AtomDelete):
            local = node_states[int(action.v)]
            if topology is not None:
                atom_topology, _, _ = topology
                local = local + self.atom_ring_embedding.weight[
                    int(atom_topology[int(action.v)])
                ]
        elif isinstance(action, AtomRestate):
            local = node_states[int(action.v)]
            target = self._atom_state_embedding(
                action.atom_type,
                action.formal_charge,
                action.implicit_h_count,
            )
            if topology is not None:
                atom_topology, _, _ = topology
                local = local + self.atom_ring_embedding.weight[
                    int(atom_topology[int(action.v)])
                ]
        elif isinstance(action, (BondInsert, BondDelete, BondReorder)):
            a = int(action.a)
            b = int(action.b)
            pair = torch.cat(
                (node_states[a] + node_states[b], torch.abs(node_states[a] - node_states[b]))
            )
            local = self.pair_project(pair) + self.bond_embedding(bonds[a, b])
            if isinstance(action, BondInsert):
                order = int(action.order)
            elif isinstance(action, BondReorder):
                order = int(action.new_order)
            else:
                order = 0
            target = self.bond_embedding.weight[order]
            if topology is not None:
                atom_topology, closure_topology, ring_system_topology = topology
                local = local + self.atom_ring_embedding.weight[
                    int(atom_topology[a])
                ]
                local = local + self.atom_ring_embedding.weight[
                    int(atom_topology[b])
                ]
                target = target + self.closure_ring_embedding.weight[
                    int(closure_topology[a, b])
                ]
                target = target + self.ring_system_embedding.weight[
                    int(ring_system_topology[a, b])
                ]
            if self.path_distance_embedding is not None:
                path_distance = min(_graph_distance(state, a, b), 33)
                target = target + self.path_distance_embedding.weight[path_distance]
        else:
            raise TypeError(f"unsupported rewrite action: {type(action).__name__}")

        rule_index = RULE_TO_INDEX[transition.rule_name]
        rule = self.rule_embedding.weight[rule_index]
        return torch.cat((global_state, local, target, rule, time_state), dim=-1)

    def _topology_features(
        self,
        state: MolecularGraph,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if not self.use_topology_context:
            return None
        key = (state.atom_types.tobytes(), (state.bonds != 0).tobytes())
        cached = self._topology_cache.get(key)
        if cached is None:
            cached = compute_topology_features(state)
            if len(self._topology_cache) >= 50_000:
                self._topology_cache.clear()
            self._topology_cache[key] = cached
        return cached

    def _atom_state_embedding(
        self,
        atom_type: int,
        formal_charge: int,
        implicit_h_count: int,
    ) -> Tensor:
        return (
            self.atom_embedding.weight[int(atom_type)]
            + self.charge_embedding.weight[int(formal_charge) + 2]
            + self.hydrogen_embedding.weight[int(implicit_h_count)]
        )


class FactorizedRateModel(WholeGraphRateModel):
    """Normalize total hazard, rule family, and operands hierarchically."""

    def __init__(
        self,
        hidden_dim: int = 64,
        message_passing_steps: int = 3,
        *,
        use_rewrite_context: bool = False,
        use_topology_context: bool = False,
    ) -> None:
        super().__init__(
            hidden_dim,
            message_passing_steps,
            use_rewrite_context=use_rewrite_context,
            use_topology_context=use_topology_context,
        )
        self.log_rate_bias.requires_grad_(False)
        self.total_hazard_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.family_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(RULE_NAMES)),
        )

    def predict_factorized_fiber(
        self,
        state: MolecularGraph,
        time: float,
        *,
        fiber: FactorizedFiber | None = None,
    ) -> FiberRatePrediction:
        legal_fiber = fiber or enumerate_factorized_cnof_fiber(state)
        transitions = legal_fiber.transitions
        if not transitions:
            return FiberRatePrediction(
                transitions=(),
                marked_rates=torch.empty(0, device=self.device),
                grouped_indices={},
            )

        action_logits, global_state, time_state = self.action_logits(
            state,
            time,
            transitions,
        )
        context = torch.cat((global_state, time_state), dim=-1)
        total_hazard = F.softplus(self.total_hazard_head(context).squeeze(-1))
        raw_family_logits = self.family_head(context)
        enabled = torch.tensor(
            [bool(legal_fiber.by_family.get(name)) for name in RULE_NAMES],
            dtype=torch.bool,
            device=self.device,
        )
        family_logits = raw_family_logits.masked_fill(~enabled, float("-inf"))
        family_probabilities = torch.softmax(family_logits, dim=0)

        rates = torch.empty_like(action_logits)
        offset = 0
        for family_index, family_name in enumerate(RULE_NAMES):
            count = len(legal_fiber.by_family.get(family_name, ()))
            if count == 0:
                continue
            family_slice = slice(offset, offset + count)
            operand_probabilities = torch.softmax(action_logits[family_slice], dim=0)
            rates[family_slice] = (
                total_hazard
                * family_probabilities[family_index]
                * operand_probabilities
            )
            offset += count
        if offset != len(transitions):
            raise RuntimeError("factorized transition order does not match rule families")
        return FiberRatePrediction(
            transitions=transitions,
            marked_rates=rates,
            grouped_indices=group_transition_indices_by_successor(transitions),
        )


def _graph_distance(state: MolecularGraph, source: int, target: int) -> int:
    """Shortest heavy-atom path length, capped by the caller for embedding."""

    if source == target:
        return 0
    seen = {source}
    frontier = [source]
    distance = 0
    while frontier:
        distance += 1
        next_frontier = []
        for vertex in frontier:
            for neighbor_index in np.flatnonzero(state.bonds[vertex] != 0):
                neighbor_index = int(neighbor_index)
                if neighbor_index == target:
                    return distance
                if neighbor_index not in seen:
                    seen.add(neighbor_index)
                    next_frontier.append(neighbor_index)
        frontier = next_frontier
    return 33
