"""Rule registry and validity-closed rewrite execution."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from compose_v4.chem.molecular_graph import MolecularGraph, molecular_graph_to_smiles
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.data.charge_policy import charge_policy_preserved
from compose_v4.rewrite import operators as ops
from compose_v4.rewrite import tracelets

Validator = Callable[[MolecularGraph, Any], bool]
Executor = Callable[[MolecularGraph, Any], MolecularGraph]
Constraint = Callable[[MolecularGraph, Any, MolecularGraph], bool]


class InvalidRewrite(ValueError):
    """Raised when a rule instance cannot legally commit."""


@dataclass(frozen=True)
class RewriteRule:
    name: str
    action_type: type
    validate: Validator
    execute: Executor


@dataclass
class AggregatedSuccessor:
    state: MolecularGraph
    rate: float
    provenance: list[tuple[str, Any]]


class RewriteSystem:
    """Executable typed rewrite system with optional hard-condition hooks."""

    def __init__(
        self,
        rules: Iterable[RewriteRule],
        constraints: Iterable[Constraint] = (),
    ) -> None:
        self.rules = {rule.name: rule for rule in rules}
        if len(self.rules) == 0:
            raise ValueError("a rewrite system needs at least one rule")
        self.constraints = tuple(constraints)

    def apply(
        self, state: MolecularGraph, rule_name: str, action: Any
    ) -> MolecularGraph:
        try:
            rule = self.rules[rule_name]
        except KeyError as exc:
            raise InvalidRewrite(f"unknown rewrite rule: {rule_name}") from exc
        if not isinstance(action, rule.action_type):
            raise InvalidRewrite(
                f"{rule_name} expects {rule.action_type.__name__}, "
                f"received {type(action).__name__}"
            )
        if not is_valid_state(state):
            raise InvalidRewrite("source state is invalid")
        if not rule.validate(state, action):
            raise InvalidRewrite(f"invalid {rule_name} instance: {action!r}")
        successor = rule.execute(state, action)
        if not is_valid_state(successor):
            raise InvalidRewrite(f"{rule_name} violated the validity invariant")
        if not all(
            constraint(state, action, successor) for constraint in self.constraints
        ):
            raise InvalidRewrite(f"{rule_name} violates a hard condition")
        return successor

    def aggregate_successor_rates(
        self,
        state: MolecularGraph,
        marked_rates: Iterable[tuple[str, Any, float]],
    ) -> dict[str, AggregatedSuccessor]:
        """Sum rates for distinct marked actions that reach the same successor."""

        aggregated: dict[str, AggregatedSuccessor] = {}
        for rule_name, action, rate in marked_rates:
            rate = float(rate)
            if rate < 0:
                raise ValueError("jump rates must be non-negative")
            if rate == 0:
                continue
            successor = self.apply(state, rule_name, action)
            key = canonical_state_key(successor)
            if key not in aggregated:
                aggregated[key] = AggregatedSuccessor(successor, 0.0, [])
            aggregated[key].rate += rate
            aggregated[key].provenance.append((rule_name, action))
        return aggregated


def canonical_state_key(state: MolecularGraph) -> str:
    if state.n_real_atoms == 0:
        return "<NULL>"
    smiles = molecular_graph_to_smiles(state)
    if smiles is None:
        raise InvalidRewrite("valid state did not admit a canonical SMILES key")
    return smiles


def default_rewrite_system(constraints: Iterable[Constraint] = ()) -> RewriteSystem:
    return RewriteSystem(
        rules=(
            RewriteRule(
                "atom_insert",
                ops.AtomInsert,
                ops.is_valid_atom_insert,
                ops.apply_atom_insert,
            ),
            RewriteRule(
                "atom_delete",
                ops.AtomDelete,
                ops.is_valid_atom_delete,
                ops.apply_atom_delete,
            ),
            RewriteRule(
                "atom_restate",
                ops.AtomRestate,
                ops.is_valid_atom_restate,
                ops.apply_atom_restate,
            ),
            RewriteRule(
                "bond_insert",
                ops.BondInsert,
                ops.is_valid_bond_insert,
                ops.apply_bond_insert,
            ),
            RewriteRule(
                "bond_delete",
                ops.BondDelete,
                ops.is_valid_bond_delete,
                ops.apply_bond_delete,
            ),
            RewriteRule(
                "bond_reorder",
                ops.BondReorder,
                ops.is_valid_bond_reorder,
                ops.apply_bond_reorder,
            ),
            RewriteRule(
                "bond_reroute",
                ops.BondReroute,
                ops.is_valid_bond_reroute,
                ops.apply_bond_reroute,
            ),
            RewriteRule(
                "cycle_insert",
                tracelets.CycleInsert,
                tracelets.is_valid_cycle_insert,
                tracelets.apply_cycle_insert,
            ),
            RewriteRule(
                "cycle_delete",
                tracelets.CycleDelete,
                tracelets.is_valid_cycle_delete,
                tracelets.apply_cycle_delete,
            ),
            RewriteRule(
                "cycle_attach",
                tracelets.CycleAttach,
                tracelets.is_valid_cycle_attach,
                tracelets.apply_cycle_attach,
            ),
            RewriteRule(
                "cycle_detach",
                tracelets.CycleDetach,
                tracelets.is_valid_cycle_detach,
                tracelets.apply_cycle_detach,
            ),
            RewriteRule(
                "ring_ear_insert",
                tracelets.RingEarInsert,
                tracelets.is_valid_ring_ear_insert,
                tracelets.apply_ring_ear_insert,
            ),
            RewriteRule(
                "ring_ear_delete",
                tracelets.RingEarDelete,
                tracelets.is_valid_ring_ear_delete,
                tracelets.apply_ring_ear_delete,
            ),
            RewriteRule(
                "ring_system_restate",
                tracelets.RingSystemRestate,
                tracelets.is_valid_ring_system_restate,
                tracelets.apply_ring_system_restate,
            ),
            RewriteRule(
                "ring_system_grow",
                tracelets.RingSystemGrow,
                tracelets.is_valid_ring_system_grow,
                tracelets.apply_ring_system_grow,
            ),
            RewriteRule(
                "ring_system_delete",
                tracelets.RingSystemDelete,
                tracelets.is_valid_ring_system_delete,
                tracelets.apply_ring_system_delete,
            ),
        ),
        constraints=constraints,
    )


def connected_successor_constraint(
    _source: MolecularGraph,
    _action: Any,
    successor: MolecularGraph,
) -> bool:
    """Hard condition for a single connected molecule (plus formal null)."""

    return is_connected_or_null(successor)


def editing_charge_policy_constraint(
    source: MolecularGraph,
    _action: Any,
    successor: MolecularGraph,
) -> bool:
    """Enforce the frozen charge-preserving Editing-V2 transition policy."""

    return charge_policy_preserved(source, successor)


def de_novo_rewrite_system(
    constraints: Iterable[Constraint] = (),
) -> RewriteSystem:
    """Default runtime for de novo generation with connected visible states."""

    return default_rewrite_system(
        constraints=(connected_successor_constraint, *tuple(constraints))
    )


def editing_v2_rewrite_system(
    constraints: Iterable[Constraint] = (),
) -> RewriteSystem:
    """Editing-V2 runtime with explicit semantic cycle opening.

    The legacy and de-novo runtimes intentionally retain their historical rule
    registry. This separate constructor prevents the new action identity from
    silently expanding an old process or unconditional checkpoint.
    """

    legacy = default_rewrite_system()
    return RewriteSystem(
        rules=(
            *legacy.rules.values(),
            RewriteRule(
                "cycle_close",
                ops.CycleCloseEdge,
                ops.is_valid_cycle_close_edge,
                ops.apply_cycle_close_edge,
            ),
            RewriteRule(
                "cycle_open",
                ops.CycleOpenEdge,
                ops.is_valid_cycle_open_edge,
                ops.apply_cycle_open_edge,
            ),
        ),
        constraints=(connected_successor_constraint, *tuple(constraints)),
    )


def editing_v2_semantic_cycle_rewrite_system(
    constraints: Iterable[Constraint] = (),
) -> RewriteSystem:
    """Editing-V2 runtime whose public cycle actions are fully semantic.

    The V1 constructor above is retained for the already frozen semantic-open
    evidence. This V2 process excludes raw bond insertion and deletion from
    its rule registry so callers cannot accidentally mix the legacy micro
    ontology with ``cycle_close`` and ``cycle_open``.
    """

    legacy = default_rewrite_system()
    retained = tuple(
        rule
        for name, rule in legacy.rules.items()
        if name not in {"bond_insert", "bond_delete"}
    )
    return RewriteSystem(
        rules=(
            *retained,
            RewriteRule(
                "cycle_close",
                ops.CycleCloseEdge,
                ops.is_valid_cycle_close_edge,
                ops.apply_cycle_close_edge,
            ),
            RewriteRule(
                "cycle_open",
                ops.CycleOpenEdge,
                ops.is_valid_cycle_open_edge,
                ops.apply_cycle_open_edge,
            ),
        ),
        constraints=(
            connected_successor_constraint,
            editing_charge_policy_constraint,
            *tuple(constraints),
        ),
    )


def editing_v2_semantic_rewrite_system(
    constraints: Iterable[Constraint] = (),
) -> RewriteSystem:
    """Complete frozen Active8 Editing-V2 public runtime.

    Historical raw atom restatement and bond insertion/deletion remain
    available through their original runtimes. This process is an exact
    allowlist, not a filtered legacy registry, so disabled macros and internal
    tracelet rules cannot silently enter the declared Editing-V2 support.
    """

    return RewriteSystem(
        rules=(
            RewriteRule(
                "atom_insert",
                ops.AtomInsert,
                ops.is_valid_editing_v2_atom_insert,
                ops.apply_atom_insert,
            ),
            RewriteRule(
                "atom_delete",
                ops.AtomDelete,
                ops.is_valid_atom_delete,
                ops.apply_atom_delete,
            ),
            RewriteRule(
                "atom_restate_semantic",
                ops.SemanticAtomRestate,
                ops.is_valid_semantic_atom_restate,
                ops.apply_semantic_atom_restate,
            ),
            RewriteRule(
                "bond_reorder",
                ops.BondReorder,
                ops.is_valid_bond_reorder,
                ops.apply_bond_reorder,
            ),
            RewriteRule(
                "bond_reroute",
                ops.BondReroute,
                ops.is_valid_bond_reroute,
                ops.apply_bond_reroute,
            ),
            RewriteRule(
                "cycle_close",
                ops.CycleCloseEdge,
                ops.is_valid_cycle_close_edge,
                ops.apply_cycle_close_edge,
            ),
            RewriteRule(
                "cycle_open",
                ops.CycleOpenEdge,
                ops.is_valid_cycle_open_edge,
                ops.apply_cycle_open_edge,
            ),
            RewriteRule(
                "ring_system_restate",
                tracelets.RingSystemRestate,
                tracelets.is_valid_ring_system_restate,
                tracelets.apply_ring_system_restate,
            ),
        ),
        constraints=(
            connected_successor_constraint,
            editing_charge_policy_constraint,
            *tuple(constraints),
        ),
    )
