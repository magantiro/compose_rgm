"""Source-preserving construction teachers; not a learned conditional sampler.

The context is source-only, bound to persistent executable slots. Target atom
correspondence is used only by the teacher compiler. No executed state is
canonicalized or reparsed to recover that correspondence. Existing primitives
and validity guards remain the chemical authority.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from compose_v4.chem.molecular_graph import (
    BOND_AROMATIC,
    BOND_CLASS_TO_H_CHANGE,
    MAX_H_COUNT,
    NULL_IDX,
    MolecularGraph,
    is_element,
)
from compose_v4.chem.state import (
    empty_molecular_graph,
    is_connected_or_null,
    is_valid_state,
    pad_molecular_graph,
)
from compose_v4.rewrite.compiler import TraceCompilationError, compile_null_to_target
from compose_v4.rewrite.kernel import InvalidRewrite, RewriteSystem, de_novo_rewrite_system
from compose_v4.rewrite.operators import AtomInsert, BondInsert
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace


def _real_slots(state: MolecularGraph) -> tuple[int, ...]:
    return tuple(int(v) for v in np.flatnonzero(is_element(state.atom_types)))


def _validate_graph(state: MolecularGraph) -> None:
    if not is_valid_state(state) or not is_connected_or_null(state):
        raise TraceCompilationError("construction requires a valid connected or null graph")
    if np.any((state.atom_types != NULL_IDX) & ~is_element(state.atom_types)):
        raise TraceCompilationError("construction supports real atoms and NULL slots only")
    if np.any(state.bonds == BOND_AROMATIC):
        raise TraceCompilationError("construction requires the executable Kekule bond view")


def _same_state(left: MolecularGraph, right: MolecularGraph) -> bool:
    return all(np.array_equal(getattr(left, name), getattr(right, name)) for name in (
        "atom_types", "formal_charges", "implicit_h_counts", "bonds"))


@dataclass(frozen=True)
class ScaffoldContext:
    """Immutable source chemistry plus allowed attachment sites, without a target.

    All supplied heavy atoms and their induced bonds are protected. Hydrogens
    may be consumed/recovered at allowed attachment sites by executable edits.
    This is a representation-specific context identity, not a canonical molecule
    key. A cache must bind it *and* its current executable state. The node flags
    here are an interface for future model integration, not a trained feature.
    """

    n_slots: int
    atoms: tuple[tuple[int, int, int, int], ...]  # slot, element, charge, source H
    bonds: tuple[tuple[int, int, int], ...]  # nonzero induced edges
    attachment_slots: tuple[int, ...]

    @classmethod
    def from_source(
        cls, source: MolecularGraph, attachment_slots: Sequence[int] = (),
    ) -> ScaffoldContext:
        _validate_graph(source)
        slots = _real_slots(source)
        attachments = tuple(attachment_slots)
        if any(not isinstance(v, (int, np.integer)) for v in attachments):
            raise ValueError("attachment slots must be integers")
        if len(set(attachments)) != len(attachments) or not set(attachments) <= set(slots):
            raise ValueError("attachment slots must be distinct supplied atoms")
        if any(source.implicit_h_counts[v] <= 0 for v in attachments):
            raise ValueError("an attachment site must have source hydrogen capacity")
        return cls(
            source.n_atoms,
            tuple((v, int(source.atom_types[v]), int(source.formal_charges[v]),
                   int(source.implicit_h_counts[v])) for v in slots),
            tuple((v, u, int(source.bonds[v, u])) for v in slots for u in slots
                  if u > v and source.bonds[v, u]),
            tuple(sorted(int(v) for v in attachments)),
        )

    @property
    def protected_slots(self) -> tuple[int, ...]:
        return tuple(atom[0] for atom in self.atoms)

    @property
    def identity(self) -> str:
        payload = {"format": "preserved_scaffold_v1", **asdict(self)}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def node_flags(self) -> np.ndarray:
        """Columns: supplied-core membership, allowed attachment membership."""
        flags = np.zeros((self.n_slots, 2), dtype=np.float32)
        flags[list(self.protected_slots), 0] = 1
        flags[list(self.attachment_slots), 1] = 1
        return flags

    def state_cache_key(self, state: MolecularGraph) -> tuple:
        """Bind exact executable state AND condition, never endpoint identity alone."""
        if not self.accepts(state):
            raise ValueError("state violates its supplied scaffold context")
        return (self.identity, *(getattr(state, name).tobytes() for name in (
            "atom_types", "formal_charges", "implicit_h_counts", "bonds")))

    def permuted(self, old_to_new: Sequence[int]) -> ScaffoldContext:
        """Carry the condition through an explicit bijection of *all* slots."""
        p = tuple(old_to_new)
        if (len(p) != self.n_slots or
                any(not isinstance(v, (int, np.integer)) for v in p) or
                set(p) != set(range(self.n_slots))):
            raise ValueError("old_to_new must be a complete slot permutation")
        return ScaffoldContext(
            self.n_slots,
            tuple(sorted((int(p[v]), element, charge, h) for v, element, charge, h in self.atoms)),
            tuple(sorted((min(int(p[a]), int(p[b])), max(int(p[a]), int(p[b])), order)
                         for a, b, order in self.bonds)),
            tuple(sorted(int(p[v]) for v in self.attachment_slots)),
        )

    def accepts(self, state: MolecularGraph) -> bool:
        """Check the context invariant; executor validity is a separate guard."""
        if state.n_atoms != self.n_slots:
            return False
        protected = set(self.protected_slots)
        attachments = set(self.attachment_slots)
        actual_bonds = tuple((a, b, int(state.bonds[a, b])) for a in sorted(protected)
                             for b in sorted(protected) if b > a and state.bonds[a, b])
        if actual_bonds != self.bonds:
            return False
        for v, element, charge, source_h in self.atoms:
            if state.atom_types[v] != element or state.formal_charges[v] != charge:
                return False
            exterior = [int(u) for u in np.flatnonzero(state.bonds[v]) if u not in protected]
            if v not in attachments and (exterior or state.implicit_h_counts[v] != source_h):
                return False
            consumed = sum(int(BOND_CLASS_TO_H_CHANGE[state.bonds[v, u]]) for u in exterior)
            if int(state.implicit_h_counts[v]) + consumed != source_h:
                return False
        return True

    def constraint(self, before: MolecularGraph, action: object, after: MolecularGraph) -> bool:
        return self.accepts(before) and self.accepts(after)

    def rewrite_system(self, base: RewriteSystem | None = None) -> RewriteSystem:
        """Retain caller rules/constraints and add connectivity plus this core."""
        runtime = base or de_novo_rewrite_system()
        return RewriteSystem(runtime.rules.values(), (*runtime.constraints,
            lambda before, action, after: is_connected_or_null(after), self.constraint))


@dataclass(frozen=True)
class ScaffoldConstruction:
    trace: RewriteTrace
    context: ScaffoldContext
    target_to_execution: tuple[tuple[int, int], ...]  # teacher-only


def _align_scaffold_target(
    source: MolecularGraph,
    target: MolecularGraph,
    source_to_target: Mapping[int, int],
    attachment_slots: Sequence[int],
) -> tuple[ScaffoldContext, MolecularGraph, dict[int, int]]:
    _validate_graph(target)
    context = ScaffoldContext.from_source(source, attachment_slots)
    source_slots, target_slots = set(context.protected_slots), set(_real_slots(target))
    mapping = dict(source_to_target)
    if any(not isinstance(v, (int, np.integer)) for v in (*mapping, *mapping.values())):
        raise TraceCompilationError("correspondence must contain integer slots")
    if set(mapping) != source_slots or len(set(mapping.values())) != len(mapping):
        raise TraceCompilationError("correspondence must bijectively cover every supplied atom")
    if not set(mapping.values()) <= target_slots:
        raise TraceCompilationError("correspondence refers to absent target atoms")
    if len(target_slots) > source.n_atoms:
        raise TraceCompilationError("target exceeds source slot capacity")
    target_to_slot = {int(t): int(s) for s, t in mapping.items()}
    free = set(range(source.n_atoms)) - source_slots
    for t in sorted(target_slots - set(target_to_slot)):
        slot = t if t in free else min(free)
        target_to_slot[t] = slot
        free.remove(slot)
    aligned = empty_molecular_graph(source.n_atoms)
    for t, slot in target_to_slot.items():
        aligned.atom_types[slot] = target.atom_types[t]
        aligned.formal_charges[slot] = target.formal_charges[t]
        aligned.implicit_h_counts[slot] = target.implicit_h_counts[t]
        for u, other in target_to_slot.items():
            aligned.bonds[slot, other] = target.bonds[t, u]
    if not context.accepts(aligned):
        raise TraceCompilationError("target disagrees with exact supplied chemistry or attachment permission")
    return context, aligned, target_to_slot


def compile_scaffold_to_target_tracelets(
    source: MolecularGraph,
    target: MolecularGraph,
    source_to_target: Mapping[int, int],
    *,
    attachment_slots: Sequence[int] = (),
    system: RewriteSystem | None = None,
    typed_ring_payloads: bool = False,
) -> ScaffoldConstruction:
    """Complete a ring-closed source using the existing block-aware tracelets.

    This keeps the legacy model-family meanings: cycle_attach grows a pendant
    ring, rather than opening a bond. Empty sources use the unchanged empty
    compiler. The typed diagnostic commits target atom/bond labels together;
    the default retains carbon carriers followed by chemical restatement.
    Exact learned-candidate support still needs catalog qualification.
    """
    from compose_v4.rewrite.tracelet_compiler import _compile_null_to_target_tracelets

    context, aligned, mapping = _align_scaffold_target(
        source, target, source_to_target, attachment_slots)
    trace = _compile_null_to_target_tracelets(
        aligned, system=context.rewrite_system(system), typed_ring_payloads=typed_ring_payloads,
        preserved_source=source)
    trace = RewriteTrace(trace.source, trace.target, trace.steps, {
        **trace.metadata, "compiler": "preserved_scaffold_block_tracelets_v1",
        "typed_ring_payloads": typed_ring_payloads,
        "context_identity": context.identity,
    })
    return ScaffoldConstruction(trace, context, tuple(sorted(mapping.items())))


def compile_scaffold_to_target(
    source: MolecularGraph,
    target: MolecularGraph,
    source_to_target: Mapping[int, int],
    *,
    attachment_slots: Sequence[int] = (),
    system: RewriteSystem | None = None,
) -> ScaffoldConstruction:
    """Grow a connected supplied core by typed births and chord insertions.

    Source slots never move; target relocation is explicit and teacher-only.
    Exact Kekule disagreement fails without SMILES repair. This primitive
    baseline is not a learned-candidate or universal reachability guarantee.
    Empty contexts retain the existing micro compiler, not the tracelet recipe.
    """
    context, aligned, target_to_slot = _align_scaffold_target(
        source, target, source_to_target, attachment_slots)
    source_slots = set(context.protected_slots)
    # Independent copies prevent a caller's later array mutation changing replay.
    initial = pad_molecular_graph(source, source.n_atoms)
    runtime = context.rewrite_system(system)
    if not source_slots:
        trace = compile_null_to_target(aligned, system=runtime)
    else:
        state = initial
        built = set(source_slots)
        pending = set(_real_slots(aligned)) - built
        steps = []
        try:
            while pending:
                frontier = [(v, u) for v in sorted(pending) for u in sorted(built)
                            if aligned.bonds[v, u]]
                if not frontier:
                    raise TraceCompilationError("no connected extension from supplied core")
                # Prefer larger attachment order to avoid excessive transient H
                # at high-valence atoms. This is deterministic, not target input.
                v, anchor = min(frontier, key=lambda edge: (
                    -int(aligned.bonds[edge]), edge))
                order = int(aligned.bonds[v, anchor])
                h = int(aligned.implicit_h_counts[v]) + sum(
                    int(BOND_CLASS_TO_H_CHANGE[aligned.bonds[v, u]])
                    for u in np.flatnonzero(aligned.bonds[v]) if u != anchor)
                if h > MAX_H_COUNT:
                    raise TraceCompilationError(f"slot {v} requires transient H={h}")
                step = RewriteStep("atom_insert", AtomInsert(
                    v, int(aligned.atom_types[v]), int(aligned.formal_charges[v]), h,
                    ((anchor, order),)))
                state = runtime.apply(state, step.rule_name, step.action)
                steps.append(step)
                # Close only target chords to atoms already present. Intermediates
                # are complete valid molecules, never disconnected fragments.
                for u in sorted(built):
                    if aligned.bonds[v, u] and not state.bonds[v, u]:
                        step = RewriteStep("bond_insert", BondInsert(v, u, int(aligned.bonds[v, u])))
                        state = runtime.apply(state, step.rule_name, step.action)
                        steps.append(step)
                built.add(v)
                pending.remove(v)
        except InvalidRewrite as exc:
            raise TraceCompilationError(f"scaffold extension failed: {exc}") from exc
        if not _same_state(state, aligned):
            raise TraceCompilationError("scaffold extension did not reach the exact target")
        trace = RewriteTrace(initial, aligned, tuple(steps), {
            "compiler": "preserved_scaffold_typed_growth_v1",
            "context_identity": context.identity,
            "source_heavy_atoms": len(source_slots),
            "atom_steps": sum(s.rule_name == "atom_insert" for s in steps),
            "chord_steps": sum(s.rule_name == "bond_insert" for s in steps),
        })
    return ScaffoldConstruction(trace, context, tuple(sorted(target_to_slot.items())))
