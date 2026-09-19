"""Hierarchical marked-rate model over micro rewrites and ring tracelets."""

from __future__ import annotations

from contextlib import contextmanager
from math import log
from typing import cast
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.model.rate_model import (
    FiberRatePrediction,
    RULE_NAMES as MICRO_RULE_NAMES,
    WholeGraphRateModel,
)
from compose_v4.rewrite.fiber import MarkedTransition, group_transition_indices_by_successor
from compose_v4.rewrite.ring_junctions import (
    RING_JUNCTION_KINDS,
    RING_JUNCTION_TO_INDEX,
    RingJunctionDescriptor,
    describe_ring_transaction,
)
from compose_v4.rewrite.tracelet_fiber import (
    TRACELET_RULE_FAMILIES,
    TRACELET_TRANSPORT_RULE_FAMILIES,
    TraceletFiber,
    enumerate_tracelet_cnof_fiber,
)
from compose_v4.rewrite.operators import (
    AtomDelete,
    AtomInsert,
    AtomRestate,
    BondDelete,
    BondInsert,
    BondReorder,
    BondReroute,
)
from compose_v4.rewrite.tracelets import (
    CycleAttach,
    CycleInsert,
    RingEarInsert,
    RingSystemRestate,
)
from compose_v4.rewrite.typed_ring_catalog import TypedRingCatalog


TRACELET_RULE_NAMES = TRACELET_RULE_FAMILIES
TRACELET_RULE_TO_INDEX = {
    name: index for index, name in enumerate(TRACELET_RULE_NAMES)
}
_MICRO_RULE_SET = frozenset(MICRO_RULE_NAMES)


class TraceletRateModel(WholeGraphRateModel):
    """Normalize total hazard, rule family, and tracelet operands.

    Variable-length macro marks are encoded by permutation-invariant pooling,
    with explicit span and macro-kind embeddings. The model sees the current
    complete molecule and a legal executable action, never the target molecule
    or the compiler trace.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        message_passing_steps: int = 3,
        *,
        use_rewrite_context: bool = True,
        use_topology_context: bool = True,
        max_macro_span: int = 64,
        rate_factorization: str = "hierarchical",
        use_aromatic_bond_view: bool = False,
        ring_catalog: TypedRingCatalog | None = None,
        enable_bond_reroute: bool = False,
    ) -> None:
        super().__init__(
            hidden_dim,
            message_passing_steps,
            use_rewrite_context=use_rewrite_context,
            use_topology_context=use_topology_context,
            use_aromatic_bond_view=use_aromatic_bond_view,
        )
        if max_macro_span < 3:
            raise ValueError("max_macro_span must be at least three")
        if rate_factorization not in {
            "hierarchical",
            "quotient_energy",
            "superposed",
        }:
            raise ValueError(
                "rate_factorization must be 'hierarchical', 'quotient_energy', "
                "or 'superposed'"
            )
        self.max_macro_span = int(max_macro_span)
        self.rate_factorization = rate_factorization
        self.ring_catalog = ring_catalog
        self.enable_bond_reroute = bool(enable_bond_reroute)
        self.rule_names = (
            TRACELET_TRANSPORT_RULE_FAMILIES
            if self.enable_bond_reroute
            else TRACELET_RULE_NAMES
        )
        self.rule_to_index = {
            name: index for index, name in enumerate(self.rule_names)
        }
        # Transport mode appends its new family, preserving every old index.
        self.rule_embedding = nn.Embedding(len(self.rule_names), hidden_dim)
        self.macro_span_embedding = nn.Embedding(max_macro_span + 1, hidden_dim)
        self.macro_kind_embedding = nn.Embedding(5, hidden_dim)
        self.ring_junction_embedding = nn.Embedding(
            len(RING_JUNCTION_KINDS), hidden_dim
        )
        self.ring_size_embedding = nn.Embedding(max_macro_span + 1, hidden_dim)
        self.macro_self_update = nn.Linear(hidden_dim, hidden_dim)
        self.macro_neighbor_update = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.log_rate_bias.requires_grad_(False)
        self.total_hazard_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.family_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(self.rule_names)),
        )
        self._shared_tracelet_encoding_cache: dict[
            tuple[object, ...], Tensor
        ] | None = None

    @contextmanager
    def shared_tracelet_encoding_cache(self):
        """Share state-independent template encodings within one loss/metric pass."""

        previous = self._shared_tracelet_encoding_cache
        self._shared_tracelet_encoding_cache = {}
        try:
            yield
        finally:
            self._shared_tracelet_encoding_cache = previous

    def predict_tracelet_fiber(
        self,
        state: MolecularGraph,
        time: float,
        *,
        fiber: TraceletFiber | None = None,
    ) -> FiberRatePrediction:
        legal_fiber = fiber or enumerate_tracelet_cnof_fiber(
            state,
            ring_catalog=self.ring_catalog,
            allow_bond_reroute=self.enable_bond_reroute,
        )
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
            [bool(legal_fiber.by_family.get(name)) for name in self.rule_names],
            dtype=torch.bool,
            device=self.device,
        )
        if self.rate_factorization == "quotient_energy":
            return self._predict_quotient_energy(
                legal_fiber=legal_fiber,
                transitions=transitions,
                action_logits=action_logits,
                total_hazard=total_hazard,
                raw_family_logits=raw_family_logits,
                enabled=enabled,
            )
        if self.rate_factorization == "superposed":
            return self._predict_superposed(
                legal_fiber=legal_fiber,
                transitions=transitions,
                action_logits=action_logits,
                raw_family_logits=raw_family_logits,
            )

        family_logits = raw_family_logits.masked_fill(~enabled, float("-inf"))
        family_probabilities = torch.softmax(family_logits, dim=0)

        rates = torch.empty_like(action_logits)
        offset = 0
        for family_index, family_name in enumerate(self.rule_names):
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
            raise RuntimeError("tracelet transition order does not match rule families")
        return FiberRatePrediction(
            transitions=transitions,
            marked_rates=rates,
            grouped_indices=group_transition_indices_by_successor(transitions),
        )

    def _predict_superposed(
        self,
        *,
        legal_fiber: TraceletFiber,
        transitions: tuple[MarkedTransition, ...],
        action_logits: Tensor,
        raw_family_logits: Tensor,
    ) -> FiberRatePrediction:
        """Add independently normalized valid rewrite generators.

        Every enabled family receives its own non-negative hazard instead of
        competing for a fixed softmax budget.  Candidate evidence uses a
        log-mean-exp over distinct chemical successors, preserving invariance
        to padding, match, and resonance aliases.  The resulting family
        generators are conservative individually and therefore remain a valid
        generator under summation.
        """

        family_rate_rows = []
        offset = 0
        for family_index, family_name in enumerate(self.rule_names):
            family_transitions = legal_fiber.by_family.get(family_name, ())
            count = len(family_transitions)
            if count == 0:
                continue
            local_logits = action_logits[offset : offset + count]
            successor_indices: dict[str, list[int]] = {}
            for local_index, transition in enumerate(family_transitions):
                successor_indices.setdefault(transition.successor_key, []).append(
                    local_index
                )
            groups = tuple(successor_indices.values())
            successor_scores = torch.stack(
                [local_logits[indices].max() for indices in groups]
            )
            family_evidence = (
                torch.logsumexp(successor_scores, dim=0) - log(len(groups))
            )
            family_hazard = F.softplus(
                raw_family_logits[family_index] + family_evidence
            )
            successor_probabilities = torch.softmax(successor_scores, dim=0)
            local_rates: list[Tensor | None] = [None] * count
            for probability, local_indices in zip(
                successor_probabilities,
                groups,
            ):
                alias_rate = family_hazard * probability / len(local_indices)
                for local_index in local_indices:
                    local_rates[local_index] = alias_rate
            if any(rate is None for rate in local_rates):
                raise RuntimeError("superposed family did not assign every rate")
            family_rate_rows.append(torch.stack(local_rates))
            offset += count
        if offset != len(transitions):
            raise RuntimeError("tracelet transition order does not match rule families")
        rates = torch.cat(family_rate_rows)
        return FiberRatePrediction(
            transitions=transitions,
            marked_rates=rates,
            grouped_indices=group_transition_indices_by_successor(transitions),
        )

    def _predict_quotient_energy(
        self,
        *,
        legal_fiber: TraceletFiber,
        transitions: tuple[MarkedTransition, ...],
        action_logits: Tensor,
        total_hazard: Tensor,
        raw_family_logits: Tensor,
        enabled: Tensor,
    ) -> FiberRatePrediction:
        """Normalize over chemical successors, not representation aliases.

        For each family, marks reaching the same canonical chemical state form
        one quotient class.  Its energy is the maximum score among its legal
        derivations, which is invariant to duplicating a padding or resonance
        alias.  A log-mean-exp over distinct successors supplies candidate
        evidence to the family gate without rewarding large fibers merely for
        having more operands.  Formally, for family ``f`` and successor ``y``,

            p(f, y | x, t) proportional to
                exp(b_f(x,t) + s_f,y(x,t)) / |Y_f(x)|.

        Rates are split uniformly among alias marks only so the existing
        marked-action sampler remains executable; they re-aggregate exactly at
        the semantic successor.
        """

        family_rows = []
        combined_family_logits = []
        offset = 0
        for family_index, family_name in enumerate(self.rule_names):
            family_transitions = legal_fiber.by_family.get(family_name, ())
            count = len(family_transitions)
            if count == 0:
                family_rows.append(None)
                combined_family_logits.append(raw_family_logits[family_index])
                continue
            local_logits = action_logits[offset : offset + count]
            successor_indices: dict[str, list[int]] = {}
            for local_index, transition in enumerate(family_transitions):
                successor_indices.setdefault(transition.successor_key, []).append(
                    local_index
                )
            groups = tuple(successor_indices.values())
            successor_scores = torch.stack(
                [local_logits[indices].max() for indices in groups]
            )
            family_evidence = (
                torch.logsumexp(successor_scores, dim=0) - log(len(groups))
            )
            family_rows.append((offset, groups, successor_scores))
            combined_family_logits.append(
                raw_family_logits[family_index] + family_evidence
            )
            offset += count
        if offset != len(transitions):
            raise RuntimeError("tracelet transition order does not match rule families")

        family_logits = torch.stack(combined_family_logits).masked_fill(
            ~enabled,
            float("-inf"),
        )
        family_probabilities = torch.softmax(family_logits, dim=0)
        family_rate_rows = []
        for family_index, row in enumerate(family_rows):
            if row is None:
                continue
            family_offset, groups, successor_scores = row
            successor_probabilities = torch.softmax(successor_scores, dim=0)
            count = sum(len(local_indices) for local_indices in groups)
            local_rates: list[Tensor | None] = [None] * count
            for successor_probability, local_indices in zip(
                successor_probabilities,
                groups,
            ):
                alias_rate = (
                    total_hazard
                    * family_probabilities[family_index]
                    * successor_probability
                    / len(local_indices)
                )
                for local_index in local_indices:
                    local_rates[local_index] = alias_rate
            if any(value is None for value in local_rates):
                raise RuntimeError("quotient family did not assign every marked rate")
            family_rate_rows.append(torch.stack(local_rates))
        rates = torch.cat(family_rate_rows)
        return FiberRatePrediction(
            transitions=transitions,
            marked_rates=rates,
            grouped_indices=group_transition_indices_by_successor(transitions),
        )

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
        """Batch coordinated ring marks and share repeated template encodings."""

        tracelet_cache: dict[tuple[object, ...], object] = {}
        rows: list[Tensor | None] = [None] * len(transitions)
        micro_positions = [
            index
            for index, transition in enumerate(transitions)
            if transition.rule_name in _MICRO_RULE_SET
        ]
        ring_positions = [
            index
            for index, transition in enumerate(transitions)
            if isinstance(
                transition.action,
                (CycleInsert, CycleAttach, RingEarInsert),
            )
        ]
        batched_positions = set((*micro_positions, *ring_positions))
        for index, transition in enumerate(transitions):
            if index in batched_positions:
                continue
            rows[index] = self._encode_action(
                transition,
                state,
                node_states,
                global_state,
                time_state,
                bonds,
                topology,
                tracelet_cache=tracelet_cache,
            )
        if micro_positions:
            micro_rows = self._encode_micro_action_rows(
                tuple(transitions[index] for index in micro_positions),
                state,
                node_states,
                global_state,
                time_state,
                bonds,
                topology,
            )
            for position, row in zip(micro_positions, micro_rows):
                rows[position] = row
        if ring_positions:
            ring_rows = self._encode_ring_action_rows(
                tuple(transitions[index] for index in ring_positions),
                state,
                node_states,
                global_state,
                time_state,
                topology,
                tracelet_cache,
            )
            for position, row in zip(ring_positions, ring_rows):
                rows[position] = row
        if any(row is None for row in rows):
            raise RuntimeError("tracelet action batching left an unencoded row")
        return [cast(Tensor, row) for row in rows]

    def _encode_micro_action_rows(
        self,
        transitions: tuple[MarkedTransition, ...],
        state: MolecularGraph,
        node_states: Tensor,
        global_state: Tensor,
        time_state: Tensor,
        bonds: Tensor,
        topology: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    ) -> Tensor:
        """Encode a complete micro-action fiber with a fixed number of kernels.

        The old scalar loop launched several tiny GPU kernels for every legal
        atom/bond operand.  A 24-atom state can expose hundreds of operands, so
        Python and launch latency dominated H100 training.  This routine keeps
        the exact action features but batches each operand kind.
        """

        count = len(transitions)
        local = node_states.new_zeros((count, self.hidden_dim))
        target = node_states.new_zeros((count, self.hidden_dim))

        inserts: list[tuple[int, AtomInsert]] = []
        deletes: list[tuple[int, AtomDelete]] = []
        restates: list[tuple[int, AtomRestate]] = []
        bonds_actions: list[
            tuple[int, BondInsert | BondDelete | BondReorder]
        ] = []
        for row, transition in enumerate(transitions):
            action = transition.action
            if isinstance(action, AtomInsert):
                inserts.append((row, action))
            elif isinstance(action, AtomDelete):
                deletes.append((row, action))
            elif isinstance(action, AtomRestate):
                restates.append((row, action))
            elif isinstance(action, (BondInsert, BondDelete, BondReorder)):
                bonds_actions.append((row, action))
            else:
                raise TypeError(f"unsupported micro action: {type(action).__name__}")

        def indices(values: list[int] | tuple[int, ...]) -> Tensor:
            return torch.as_tensor(values, dtype=torch.long, device=self.device)

        def payload_rows(actions: list[AtomInsert | AtomRestate]) -> Tensor:
            return (
                self.atom_embedding(indices([int(action.atom_type) for action in actions]))
                + self.charge_embedding(
                    indices([int(action.formal_charge) + 2 for action in actions])
                )
                + self.hydrogen_embedding(
                    indices([int(action.implicit_h_count) for action in actions])
                )
            )

        if inserts:
            row_index = indices([row for row, _ in inserts])
            actions = [action for _, action in inserts]
            target = target.index_copy(0, row_index, payload_rows(actions))
            neighbor_rows = []
            neighbor_vertices = []
            neighbor_orders = []
            for row, action in inserts:
                for neighbor, order in action.neighbors:
                    neighbor_rows.append(row)
                    neighbor_vertices.append(int(neighbor))
                    neighbor_orders.append(int(order))
            if neighbor_rows:
                neighbor_row_index = indices(neighbor_rows)
                neighbor_index = indices(neighbor_vertices)
                local_values = node_states.index_select(0, neighbor_index)
                if topology is not None:
                    atom_topology, _, _ = topology
                    local_values = local_values + self.atom_ring_embedding(
                        indices(
                            [int(atom_topology[v]) for v in neighbor_vertices]
                        )
                    )
                local = local.index_add(0, neighbor_row_index, local_values)
                target = target.index_add(
                    0,
                    neighbor_row_index,
                    self.bond_embedding(indices(neighbor_orders)),
                )

        if deletes:
            row_index = indices([row for row, _ in deletes])
            vertices = [int(action.v) for _, action in deletes]
            values = node_states.index_select(0, indices(vertices))
            if topology is not None:
                atom_topology, _, _ = topology
                values = values + self.atom_ring_embedding(
                    indices([int(atom_topology[v]) for v in vertices])
                )
            local = local.index_copy(0, row_index, values)

        if restates:
            row_index = indices([row for row, _ in restates])
            actions = [action for _, action in restates]
            vertices = [int(action.v) for action in actions]
            values = node_states.index_select(0, indices(vertices))
            if topology is not None:
                atom_topology, _, _ = topology
                values = values + self.atom_ring_embedding(
                    indices([int(atom_topology[v]) for v in vertices])
                )
            local = local.index_copy(0, row_index, values)
            target = target.index_copy(0, row_index, payload_rows(actions))

        if bonds_actions:
            row_index = indices([row for row, _ in bonds_actions])
            actions = [action for _, action in bonds_actions]
            left_vertices = [int(action.a) for action in actions]
            right_vertices = [int(action.b) for action in actions]
            left_index = indices(left_vertices)
            right_index = indices(right_vertices)
            left = node_states.index_select(0, left_index)
            right = node_states.index_select(0, right_index)
            values = self.pair_project(
                torch.cat((left + right, torch.abs(left - right)), dim=-1)
            ) + self.bond_embedding(bonds[left_index, right_index])
            target_orders = [
                int(action.order)
                if isinstance(action, BondInsert)
                else (
                    int(action.new_order)
                    if isinstance(action, BondReorder)
                    else 0
                )
                for action in actions
            ]
            target_values = self.bond_embedding(indices(target_orders))
            if topology is not None:
                atom_topology, closure_topology, ring_system_topology = topology
                values = values + self.atom_ring_embedding(
                    indices([int(atom_topology[v]) for v in left_vertices])
                )
                values = values + self.atom_ring_embedding(
                    indices([int(atom_topology[v]) for v in right_vertices])
                )
                target_values = target_values + self.closure_ring_embedding(
                    indices(
                        [
                            int(closure_topology[a, b])
                            for a, b in zip(left_vertices, right_vertices)
                        ]
                    )
                )
                target_values = target_values + self.ring_system_embedding(
                    indices(
                        [
                            int(ring_system_topology[a, b])
                            for a, b in zip(left_vertices, right_vertices)
                        ]
                    )
                )
            if self.path_distance_embedding is not None:
                target_values = target_values + self.path_distance_embedding(
                    indices(
                        [
                            min(_graph_distance(state, a, b), 33)
                            for a, b in zip(left_vertices, right_vertices)
                        ]
                    )
                )
            local = local.index_copy(0, row_index, values)
            target = target.index_copy(0, row_index, target_values)

        rule = self.rule_embedding(
            indices([self.rule_to_index[item.rule_name] for item in transitions])
        )
        return torch.cat(
            (
                global_state.unsqueeze(0).expand(count, -1),
                local,
                target,
                rule,
                time_state.unsqueeze(0).expand(count, -1),
            ),
            dim=-1,
        )

    def _encode_ring_action_rows(
        self,
        transitions: tuple[MarkedTransition, ...],
        state: MolecularGraph,
        node_states: Tensor,
        global_state: Tensor,
        time_state: Tensor,
        topology: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
        tracelet_cache: dict[tuple[object, ...], object],
    ) -> Tensor:
        actions = tuple(transition.action for transition in transitions)
        count = len(actions)
        local = node_states.new_zeros((count, self.hidden_dim))
        target_extra = node_states.new_zeros((count, self.hidden_dim))
        template_rows = []
        macro_kinds = []
        spans = []
        junction_kinds = []
        ring_sizes = []
        attach_rows: list[int] = []
        attach_anchors: list[int] = []
        attach_orders: list[int] = []
        same_ear_rows: list[int] = []
        same_ear_anchors: list[int] = []
        distinct_ear_rows: list[int] = []
        distinct_ear_a: list[int] = []
        distinct_ear_b: list[int] = []
        distinct_ear_distances: list[int] = []

        for row, action in enumerate(actions):
            if isinstance(action, CycleInsert):
                cyclic = True
                macro_kinds.append(0)
                junction_kinds.append(RING_JUNCTION_TO_INDEX["root"])
                ring_sizes.append(len(action.atoms))
            elif isinstance(action, CycleAttach):
                cyclic = True
                macro_kinds.append(4)
                junction_kinds.append(RING_JUNCTION_TO_INDEX["attached"])
                ring_sizes.append(len(action.atoms))
                attach_rows.append(row)
                attach_anchors.append(int(action.anchor))
                attach_orders.append(int(action.attachment_order))
            else:
                if not isinstance(action, RingEarInsert):
                    raise TypeError(f"unsupported ring action: {type(action).__name__}")
                cyclic = False
                a, b = int(action.a), int(action.b)
                if a == b:
                    macro_kinds.append(1)
                    junction_kinds.append(RING_JUNCTION_TO_INDEX["spiro"])
                    ring_sizes.append(len(action.atoms) + 1)
                    same_ear_rows.append(row)
                    same_ear_anchors.append(a)
                else:
                    macro_kinds.append(2)
                    kind = "fused" if int(state.bonds[a, b]) != 0 else "bridged"
                    distance = min(_graph_distance(state, a, b), 33)
                    junction_kinds.append(RING_JUNCTION_TO_INDEX[kind])
                    ring_sizes.append(len(action.atoms) + 1 + distance)
                    distinct_ear_rows.append(row)
                    distinct_ear_a.append(a)
                    distinct_ear_b.append(b)
                    distinct_ear_distances.append(distance)
            spans.append(len(action.atoms))
            template_rows.append(
                self._cached_typed_tracelet_embedding(
                    action.atoms,
                    action.bond_orders,
                    cyclic=cyclic,
                    cache=tracelet_cache,
                )
            )

        def indices(values: list[int]) -> Tensor:
            return torch.as_tensor(values, dtype=torch.long, device=self.device)

        if attach_rows:
            row_index = indices(attach_rows)
            anchor_index = indices(attach_anchors)
            values = node_states.index_select(0, anchor_index)
            if topology is not None:
                atom_topology, _, _ = topology
                values = values + self.atom_ring_embedding(
                    indices([int(atom_topology[v]) for v in attach_anchors])
                )
            local = local.index_copy(0, row_index, values)
            target_extra = target_extra.index_copy(
                0,
                row_index,
                self.bond_embedding(indices(attach_orders)),
            )
        if same_ear_rows:
            row_index = indices(same_ear_rows)
            anchor_index = indices(same_ear_anchors)
            values = node_states.index_select(0, anchor_index)
            if topology is not None:
                atom_topology, _, _ = topology
                values = values + self.atom_ring_embedding(
                    indices([int(atom_topology[v]) for v in same_ear_anchors])
                )
            local = local.index_copy(0, row_index, values)
        if distinct_ear_rows:
            row_index = indices(distinct_ear_rows)
            a_index = indices(distinct_ear_a)
            b_index = indices(distinct_ear_b)
            left = node_states.index_select(0, a_index)
            right = node_states.index_select(0, b_index)
            values = self.pair_project(
                torch.cat((left + right, torch.abs(left - right)), dim=-1)
            )
            extra = node_states.new_zeros((len(distinct_ear_rows), self.hidden_dim))
            if topology is not None:
                atom_topology, closure_topology, ring_system_topology = topology
                values = values + self.atom_ring_embedding(
                    indices([int(atom_topology[v]) for v in distinct_ear_a])
                )
                values = values + self.atom_ring_embedding(
                    indices([int(atom_topology[v]) for v in distinct_ear_b])
                )
                extra = extra + self.closure_ring_embedding(
                    indices(
                        [
                            int(closure_topology[a, b])
                            for a, b in zip(distinct_ear_a, distinct_ear_b)
                        ]
                    )
                )
                extra = extra + self.ring_system_embedding(
                    indices(
                        [
                            int(ring_system_topology[a, b])
                            for a, b in zip(distinct_ear_a, distinct_ear_b)
                        ]
                    )
                )
            if self.path_distance_embedding is not None:
                extra = extra + self.path_distance_embedding(
                    indices(distinct_ear_distances)
                )
            local = local.index_copy(0, row_index, values)
            target_extra = target_extra.index_copy(0, row_index, extra)

        target = (
            torch.stack(template_rows)
            + self.macro_kind_embedding(indices(macro_kinds))
            + self.macro_span_embedding(
                indices([min(value, self.max_macro_span) for value in spans])
            )
            + target_extra
            + self.ring_junction_embedding(indices(junction_kinds))
            + self.ring_size_embedding(
                indices([min(value, self.max_macro_span) for value in ring_sizes])
            )
        )
        rule = self.rule_embedding(
            indices([self.rule_to_index[item.rule_name] for item in transitions])
        )
        return torch.cat(
            (
                global_state.unsqueeze(0).expand(count, -1),
                local,
                target,
                rule,
                time_state.unsqueeze(0).expand(count, -1),
            ),
            dim=-1,
        )

    def _encode_action(
        self,
        transition: MarkedTransition,
        state: MolecularGraph,
        node_states: Tensor,
        global_state: Tensor,
        time_state: Tensor,
        bonds: Tensor,
        topology: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        *,
        tracelet_cache: dict[tuple[object, ...], object] | None = None,
    ) -> Tensor:
        if transition.rule_name in _MICRO_RULE_SET:
            return super()._encode_action(
                transition,
                state,
                node_states,
                global_state,
                time_state,
                bonds,
                topology,
            )

        action = transition.action
        local = node_states.new_zeros(self.hidden_dim)
        target = node_states.new_zeros(self.hidden_dim)
        if isinstance(action, BondReroute):
            old_a, old_b = int(action.a), int(action.b)
            new_u, new_v = int(action.u), int(action.v)
            old_pair = torch.cat(
                (
                    node_states[old_a] + node_states[old_b],
                    torch.abs(node_states[old_a] - node_states[old_b]),
                )
            )
            new_pair = torch.cat(
                (
                    node_states[new_u] + node_states[new_v],
                    torch.abs(node_states[new_u] - node_states[new_v]),
                )
            )
            local = self.pair_project(old_pair) + self.bond_embedding(
                bonds[old_a, old_b]
            )
            target = self.pair_project(new_pair) + self.bond_embedding.weight[
                int(action.new_order)
            ]
            if self.path_distance_embedding is not None:
                distance_key = ("distance", min(new_u, new_v), max(new_u, new_v))
                distance = (
                    tracelet_cache.get(distance_key)
                    if tracelet_cache is not None
                    else None
                )
                if distance is None:
                    distance = min(_graph_distance(state, new_u, new_v), 33)
                    if tracelet_cache is not None:
                        tracelet_cache[distance_key] = distance
                target = target + self.path_distance_embedding.weight[
                    cast(int, distance)
                ]
            if topology is not None:
                atom_topology, _, _ = topology
                buckets = torch.as_tensor(
                    [
                        int(atom_topology[old_a]),
                        int(atom_topology[old_b]),
                        int(atom_topology[new_u]),
                        int(atom_topology[new_v]),
                    ],
                    dtype=torch.long,
                    device=self.device,
                )
                local = local + self.atom_ring_embedding(buckets[:2]).mean(dim=0)
                target = target + self.atom_ring_embedding(buckets[2:]).mean(dim=0)
        elif isinstance(action, CycleInsert):
            target = (
                self.macro_kind_embedding.weight[0]
                + self._span_embedding(len(action.atoms))
                + self._cached_typed_tracelet_embedding(
                    action.atoms,
                    action.bond_orders,
                    cyclic=True,
                    cache=tracelet_cache,
                )
            )
        elif isinstance(action, CycleAttach):
            anchor = int(action.anchor)
            local = node_states[anchor]
            target = (
                self.macro_kind_embedding.weight[4]
                + self._span_embedding(len(action.atoms))
                + self._cached_typed_tracelet_embedding(
                    action.atoms,
                    action.bond_orders,
                    cyclic=True,
                    cache=tracelet_cache,
                )
                + self.bond_embedding.weight[int(action.attachment_order)]
            )
            if topology is not None:
                atom_topology, _, _ = topology
                local = local + self.atom_ring_embedding.weight[
                    int(atom_topology[anchor])
                ]
        elif isinstance(action, RingEarInsert):
            a, b = int(action.a), int(action.b)
            if a == b:
                local = node_states[a]
                kind = 1
            else:
                pair = torch.cat(
                    (
                        node_states[a] + node_states[b],
                        torch.abs(node_states[a] - node_states[b]),
                    )
                )
                local = self.pair_project(pair)
                kind = 2
            target = (
                self.macro_kind_embedding.weight[kind]
                + self._span_embedding(len(action.atoms))
                + self._cached_typed_tracelet_embedding(
                    action.atoms,
                    action.bond_orders,
                    cyclic=False,
                    cache=tracelet_cache,
                )
            )
            if topology is not None:
                atom_topology, closure_topology, ring_system_topology = topology
                local = local + self.atom_ring_embedding.weight[
                    int(atom_topology[a])
                ]
                if a != b:
                    local = local + self.atom_ring_embedding.weight[
                        int(atom_topology[b])
                    ]
                    target = target + self.closure_ring_embedding.weight[
                        int(closure_topology[a, b])
                    ]
                    target = target + self.ring_system_embedding.weight[
                        int(ring_system_topology[a, b])
                    ]
            if self.path_distance_embedding is not None and a != b:
                distance_key = ("distance", min(a, b), max(a, b))
                distance = (
                    tracelet_cache.get(distance_key)
                    if tracelet_cache is not None
                    else None
                )
                if distance is None:
                    distance = min(_graph_distance(state, a, b), 33)
                    if tracelet_cache is not None:
                        tracelet_cache[distance_key] = distance
                target = target + self.path_distance_embedding.weight[
                    cast(int, distance)
                ]
        elif isinstance(action, RingSystemRestate):
            vertices = sorted(
                {int(v) for change in action.changes for v in (change.a, change.b)}
            )
            if vertices:
                index = torch.as_tensor(vertices, dtype=torch.long, device=self.device)
                local = node_states.index_select(0, index).mean(dim=0)
            old_orders = tuple(
                int(bonds[change.a, change.b].item()) for change in action.changes
            )
            new_orders = tuple(int(change.new_order) for change in action.changes)
            local = local + self._mean_bond_embedding(old_orders)
            target = (
                self.macro_kind_embedding.weight[3]
                + self._span_embedding(len(action.changes))
                + self._mean_bond_embedding(new_orders)
            )
            if topology is not None and vertices:
                atom_topology, _, _ = topology
                buckets = torch.as_tensor(
                    [int(atom_topology[v]) for v in vertices],
                    dtype=torch.long,
                    device=self.device,
                )
                local = local + self.atom_ring_embedding(buckets).mean(dim=0)
        else:
            raise TypeError(f"unsupported tracelet action: {type(action).__name__}")

        if isinstance(action, (CycleInsert, CycleAttach, RingEarInsert)):
            junction_key = self._junction_cache_key(action)
            junction = (
                tracelet_cache.get(junction_key)
                if tracelet_cache is not None
                else None
            )
            if junction is None:
                junction = describe_ring_transaction(state, action)
                if tracelet_cache is not None:
                    tracelet_cache[junction_key] = junction
            junction = cast(RingJunctionDescriptor, junction)
            target = (
                target
                + self.ring_junction_embedding.weight[
                    RING_JUNCTION_TO_INDEX[junction.kind]
                ]
                + self.ring_size_embedding.weight[
                    min(junction.nominal_ring_size, self.max_macro_span)
                ]
            )

        rule = self.rule_embedding.weight[self.rule_to_index[transition.rule_name]]
        return torch.cat((global_state, local, target, rule, time_state), dim=-1)

    def _cached_typed_tracelet_embedding(
        self,
        atoms,
        orders,
        *,
        cyclic: bool,
        cache: dict[tuple[object, ...], object] | None,
    ) -> Tensor:
        atom_states = tuple(
            (
                int(atom.atom_type),
                int(atom.formal_charge),
                int(atom.implicit_h_count),
            )
            for atom in atoms
        )
        key = ("typed_tracelet", cyclic, atom_states, tuple(int(v) for v in orders))
        cached = cache.get(key) if cache is not None else None
        if cached is None and self._shared_tracelet_encoding_cache is not None:
            cached = self._shared_tracelet_encoding_cache.get(key)
        if cached is None:
            cached = self._typed_tracelet_embedding(atoms, orders, cyclic=cyclic)
            if cache is not None:
                cache[key] = cached
            if self._shared_tracelet_encoding_cache is not None:
                self._shared_tracelet_encoding_cache[key] = cached
        return cast(Tensor, cached)

    @staticmethod
    def _junction_cache_key(
        action: CycleInsert | CycleAttach | RingEarInsert,
    ) -> tuple[object, ...]:
        if isinstance(action, CycleInsert):
            return ("junction", "root", len(action.atoms))
        if isinstance(action, CycleAttach):
            return ("junction", "attached", int(action.anchor), len(action.atoms))
        return (
            "junction",
            "ear",
            min(int(action.a), int(action.b)),
            max(int(action.a), int(action.b)),
            len(action.atoms),
        )

    def _span_embedding(self, span: int) -> Tensor:
        return self.macro_span_embedding.weight[min(int(span), self.max_macro_span)]

    def _mean_payload_embedding(self, atoms) -> Tensor:
        rows = [
            self._atom_state_embedding(
                atom.atom_type,
                atom.formal_charge,
                atom.implicit_h_count,
            )
            for atom in atoms
        ]
        if not rows:
            return self.empty_state.new_zeros(self.hidden_dim)
        return torch.stack(rows).mean(dim=0)

    def _typed_tracelet_embedding(self, atoms, orders, *, cyclic: bool) -> Tensor:
        """Encode atom/bond arrangement, not only its empirical histogram."""

        atoms = tuple(atoms)
        orders = tuple(int(order) for order in orders)
        if not atoms:
            return self.empty_state.new_zeros(self.hidden_dim)
        states = torch.stack(
            [
                self._atom_state_embedding(
                    atom.atom_type,
                    atom.formal_charge,
                    atom.implicit_h_count,
                )
                for atom in atoms
            ]
        )
        edges: list[tuple[int, int, int]] = []
        if cyclic:
            if len(orders) != len(atoms):
                raise ValueError("cycle tracelet has inconsistent bond dimensions")
            edges = [
                (index, (index + 1) % len(atoms), orders[index])
                for index in range(len(atoms))
            ]
            boundary_orders: tuple[int, ...] = ()
        else:
            if len(orders) != len(atoms) + 1:
                raise ValueError("ear tracelet has inconsistent bond dimensions")
            edges = [
                (index, index + 1, orders[index + 1])
                for index in range(len(atoms) - 1)
            ]
            boundary_orders = (orders[0], orders[-1])
        for _ in range(2):
            messages = torch.zeros_like(states)
            for left, right, order in edges:
                bond = self.bond_embedding.weight[int(order)]
                messages[left] = messages[left] + states[right] + bond
                messages[right] = messages[right] + states[left] + bond
            if boundary_orders:
                messages[0] = (
                    messages[0]
                    + self.bond_embedding.weight[int(boundary_orders[0])]
                )
                messages[-1] = (
                    messages[-1]
                    + self.bond_embedding.weight[int(boundary_orders[-1])]
                )
            states = F.silu(
                self.macro_self_update(states)
                + self.macro_neighbor_update(messages)
            )
        return states.mean(dim=0)

    def _mean_bond_embedding(self, orders) -> Tensor:
        values = tuple(int(order) for order in orders)
        if not values:
            return self.empty_state.new_zeros(self.hidden_dim)
        index = torch.as_tensor(values, dtype=torch.long, device=self.device)
        return self.bond_embedding(index).mean(dim=0)


def _graph_distance(state: MolecularGraph, source: int, target: int) -> int:
    if source == target:
        return 0
    seen = {int(source)}
    frontier = [int(source)]
    distance = 0
    while frontier:
        distance += 1
        next_frontier = []
        for vertex in frontier:
            for neighbor in np.flatnonzero(state.bonds[vertex] != 0):
                neighbor = int(neighbor)
                if neighbor == target:
                    return distance
                if neighbor not in seen:
                    seen.add(neighbor)
                    next_frontier.append(neighbor)
        frontier = next_frontier
    return 33
