"""Corrupted-molecule source prior: native-GM teacher traces over B's REAL editing vocabulary.

We swap the de-novo source prior (carbon trees) for lightly-CORRUPTED real molecules -- a source-prior
swap in the SAME rewrite CTMC, nothing else (no edit-flow / bridge / diffusion machinery). The
corruption samples the edits B can apply to a REAL, complete molecule; the model then learns the exact
inverses, so it can EDIT real leads with B's own vocabulary.

Vocabulary implemented here (ring-count-preserving; both directions exact, no fiber search):
  - atom_delete / atom_insert -- peripheral add/remove. (atom_insert carries its bond; there is no
    trainable bond_insert/bond_delete mark family, so they are excluded.)
  - atom_restate -- bioisostere, INCLUDING RING-ATOM heteroatom scanning (pyridine<->benzene, N<->C);
    ring-preserving. B gates atom_restate to PERIPHERAL atoms; the model ungates ring atoms for the edit
    regime via enable_heteroatom_scan, so heteroatom scanning is a scoreable edit.
  - bond_reorder -- bond-order change on NON-aromatic bonds (the executor freezes aromatic bonds).
  - ring_system_restate -- aromatize <-> DE-aromatize (coordinated ring bond orders; e.g. pyridine<->
    piperidine). Both directions per pair: trim de-aromatizes a real ring, grow re-aromatizes it. B
    ships aromatize-only and it is dead at inference, so the model wires it on via compute_ring_restates.
  - bond_reroute (Graft) -- relocate a PENDANT tree fragment across a single-bond bridge (the common
    "walk a substituent to a new position" edit); the ring core is fixed. B gates graft to carbon trees,
    so the model enumerates the cyclic subset via compute_cyclic_graft (pendant_graft_candidates, shared
    so the teacher graft always lands in the mask it is scored against).
  - ring_system_delete (clean) -- decoration-PRESERVING ring OPENING (de-cyclize): remove a ring's
    closing bonds keeping every atom's element (enumerate_clean_ring_system_deletes), so heteroatoms
    survive -- the editing upgrade of B's carbon-izing structured delete. Taught TRIM-ONLY (de-cyclize a
    real lead's ring), across saturated/aromatic rings of any size and fused/spiro/bridged systems; needs
    the ring catalog (threaded in), a no-op without it.

Ring-CLOSING (re-cyclize) stays retained from B's de-novo ring_system_grow. The clean delete's exact
inverse IS a ring_system_grow (inverse_ring_system_delete, scaffold-fixed) but on a HETEROATOM scaffold,
which B's carbon-scaffold grow vocabulary cannot score -- so that direction is not emitted. B-edit thus
reaches B's whole ring vocabulary (grow retained + clean delete + restate), delete upgraded to keep
decoration.

EXCLUDED: the carbon-izing structured ring_system_delete (enumerate_structured_ring_system_deletes) --
it retypes every ring atom->C, destroying decoration; the clean delete replaces it for the edit regime.

make_edit_pair returns BOTH directions exactly: TRIM (source = the real molecule; recorded forward
edits) and GROW (source = the corrupted molecule; per-step inverse_step), each replaying through valid
connected molecules to its target -- identical in form to the carbon-tree teacher traces.
"""

from __future__ import annotations

from itertools import combinations

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from compose_v4.chem.molecular_graph import (CNOF_VOCABULARY, MolecularGraph,
                                             is_element, molecular_graph_to_smiles)
from compose_v4.rewrite.factorized_fiber import (_factorized_candidates,
                                                 enumerate_pendant_graft_actions)
from compose_v4.rewrite.kernel import RewriteSystem, canonical_state_key
from compose_v4.rewrite.ring_system_fiber import enumerate_clean_ring_system_deletes
from compose_v4.rewrite.tracelet_fiber import enumerate_ring_system_restate_actions
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace, inverse_step

# Micro corruption ops (from _candidate_actions), each with an exact inverse in MARK_RULE_NAMES.
# Family-level weights (split across a family's candidate instances below): peripheral + bioisostere
# dominant, bond-order common.
_MICRO_WEIGHTS = {
    "atom_delete": 4.0,    # peripheral atom+bond trim      -> inverse atom_insert
    "atom_restate": 4.0,   # bioisostere (heteroatom<->C…)  -> inverse atom_restate
    "bond_reorder": 2.0,   # bond-order change              -> inverse bond_reorder
}
_RESTATE_WEIGHT = 2.0      # ring_system_restate: aromatize<->de-aromatize -> inverse ring_system_restate
_GRAFT_WEIGHT = 2.0        # bond_reroute: relocate a pendant tree             -> inverse bond_reroute
_RING_DELETE_WEIGHT = 2.0  # ring_system_delete (clean): open a ring keeping decoration -> inverse re-grows
_FAMILY_WEIGHTS = {
    **_MICRO_WEIGHTS,
    "ring_system_restate": _RESTATE_WEIGHT,
    "bond_reroute": _GRAFT_WEIGHT,
    "ring_system_delete": _RING_DELETE_WEIGHT,
}
# Families allowed to CHANGE the ring count (ring-topology edits); every other family preserves it.
_RING_TOPOLOGY_FAMILIES = ("ring_system_delete",)
FORBIDDEN_FAMILIES = ("bond_insert", "bond_delete")  # not trainable mark families -- never emit


def _mol(node: MolecularGraph):
    return Chem.MolFromSmiles(molecular_graph_to_smiles(node) or "")


def _cycle_edges(node: MolecularGraph):
    """Return the cycle (non-bridge) edges in slot indices. B's ``bond_reorder`` dense mask excludes them
    (a single ring-bond reorder is outside the mask; ring bond orders are ``ring_system_restate``'s job),
    so the corruption must not emit one. (``atom_restate`` DOES now reach ring atoms -- heteroatom
    scanning -- so ring *atoms* are no longer excluded, only ring *bonds* for reorder.)"""
    real = [int(v) for v in np.flatnonzero(is_element(node.atom_types))]
    graph = nx.Graph()
    graph.add_nodes_from(real)
    graph.add_edges_from(
        (a, b) for a, b in combinations(real, 2) if int(node.bonds[a, b]) != 0
    )
    return {frozenset(e) for e in graph.edges()} - {frozenset(e) for e in nx.bridges(graph)}


def _charged_atoms(node: MolecularGraph):
    """Real atoms carrying a nonzero formal charge. The dense edit heads represent only NEUTRAL
    (element, valence) classes, so the corruption PROTECTS charged atoms: no edit may touch a charged
    atom's element/charge or its bond environment (which would emit a teacher mark the heads cannot
    score -- e.g. an inverse ``atom_restate`` reconstructing an aromatic N+, or a ``ring_system_restate``
    over a charged ring). Charged centers (nitro, ammonium, carboxylate, N-oxide) are kept fixed -- a
    protected-atom scope, chemically sensible for (QED, similarity) editing, not a limitation."""
    real = is_element(node.atom_types)
    return tuple(int(v) for v in np.flatnonzero(real & (node.formal_charges != 0)))


def _touches_charged(node: MolecularGraph, succ: MolecularGraph, charged) -> bool:
    """True if ``succ`` changed any charged atom's element, charge, H, or bonds (slot-stable states)."""
    return any(
        int(succ.atom_types[c]) != int(node.atom_types[c])
        or int(succ.formal_charges[c]) != int(node.formal_charges[c])
        or int(succ.implicit_h_counts[c]) != int(node.implicit_h_counts[c])
        or not np.array_equal(succ.bonds[c], node.bonds[c])
        for c in charged
    )


def _net_charge(node: MolecularGraph) -> int:
    """Total formal charge over real atoms."""
    real = is_element(node.atom_types)
    return int(node.formal_charges[real].sum())


def corrupt_to_source(
    target: MolecularGraph,
    depth: int,
    *,
    system: RewriteSystem,
    rng,
    restate_prob: float = 1.0,
    catalog=None,
    vocabulary=CNOF_VOCABULARY,
):
    """Walk ``target`` a few legal micro edits to a nearby valid ``source`` over B's real-molecule
    vocabulary. Returns ``(steps, states)`` (``states[i]`` is the pre-step state of ``steps[i]``). Every
    step is RING-COUNT-PRESERVING (peripheral add/remove, bioisostere incl. ring-atom heteroatom scan,
    non-ring bond-order); ring TOPOLOGY (add/remove a whole ring) is left to B's ring_system family."""
    steps: list[tuple[str, object]] = []
    states = [target]
    node = target
    for _ in range(depth):
        m = _mol(node)
        if m is None:
            break
        base_rings = Descriptors.RingCount(m)
        base_charge = _net_charge(node)
        cur_key = canonical_state_key(node)
        cycle_edges = _cycle_edges(node)
        charged = _charged_atoms(node)
        by_family: dict[str, list] = {}
        # Model-consistent micro fiber (canonical derived h-counts, not fiber._candidate_actions).
        # atom_restate reaches RING atoms too (heteroatom scanning; the model ungates it via
        # enable_heteroatom_scan). bond_reorder stays OFF ring bonds -- a single ring-bond reorder is
        # outside B's dense mask (ring bond orders are ring_system_restate's coordinated job).
        for rule_name, action in _factorized_candidates(
            node, allow_bond_reroute=False, vocabulary=vocabulary
        ):
            if rule_name not in _MICRO_WEIGHTS:
                continue
            if rule_name == "bond_reorder" and frozenset((int(action.a), int(action.b))) in cycle_edges:
                continue
            by_family.setdefault(rule_name, []).append(action)
        if rng.random() < restate_prob:  # aromatize<->de-aromatize offered occasionally (freq + cost)
            restates = enumerate_ring_system_restate_actions(node, system=system)
            if restates:
                by_family["ring_system_restate"] = list(restates)
        graft_actions = enumerate_pendant_graft_actions(node)  # relocate a pendant tree (bond_reroute)
        if graft_actions:
            by_family["bond_reroute"] = list(graft_actions)
        if catalog is not None:  # clean ring OPENING: de-cyclize keeping decoration (inverse re-grows)
            ring_deletes = enumerate_clean_ring_system_deletes(node, catalog)
            if ring_deletes:
                by_family["ring_system_delete"] = list(ring_deletes)
        if not by_family:
            break
        # Draw a family by its weight, then try that family's instances until one is a valid,
        # ring-count-preserving, non-duplicate move -- so the realized mix tracks the family weights
        # instead of over-picking whichever family happens to be most reliably valid.
        families = list(by_family)
        fam_weight = np.asarray(
            [_FAMILY_WEIGHTS[name] for name in families], dtype=float
        )
        fam_weight /= fam_weight.sum()
        moved = False
        for fi in rng.choice(len(families), size=len(families), replace=False, p=fam_weight):
            rule_name = families[int(fi)]
            actions = by_family[rule_name]
            for ai in rng.permutation(len(actions)):
                action = actions[int(ai)]
                try:
                    succ = system.apply(node, rule_name, action)
                except Exception:  # noqa: BLE001
                    continue
                ssmi = molecular_graph_to_smiles(succ)
                sm = Chem.MolFromSmiles(ssmi or "")
                if sm is None or "." in (ssmi or "") or canonical_state_key(succ) == cur_key:
                    continue
                if (  # micro/bioisostere/graft preserve ring count; ring_system_delete may drop one
                    rule_name not in _RING_TOPOLOGY_FAMILIES
                    and Descriptors.RingCount(sm) != base_rings
                ):
                    continue
                if charged and _touches_charged(node, succ, charged):
                    continue  # protect charged atoms -- the neutral-class heads cannot score the mark
                if _net_charge(succ) != base_charge:
                    # protect charged MOTIFS: no edit may neutralize/introduce/shift the net formal charge
                    # (e.g. deleting a neutral carbonyl O of a carboxylate re-derives O- -> OH). Universal so
                    # a neutral molecule can never silently GAIN a charge either.
                    continue
                node = succ
                steps.append((rule_name, action))
                states.append(node)
                moved = True
                break
            if moved:
                break
        if not moved:
            break
    return steps, states


def make_edit_pair(
    target: MolecularGraph,
    depth: int,
    *,
    system: RewriteSystem,
    rng,
    restate_prob: float = 1.0,
    catalog=None,
    vocabulary=CNOF_VOCABULARY,
) -> tuple[RewriteTrace | None, RewriteTrace | None]:
    """Return ``(trim_trace, grow_trace)`` (either may be ``None``), built exactly and without search.
    TRIM: source = the real target, steps = the recorded forward edits. GROW: source = the corrupted
    molecule, steps = the exact per-step inverses (``inverse_step``), verified to replay to the real
    target -- so the model learns to add atoms, substitute (bioisostere), change bond order, and flip
    ring aromaticity (aromatize<->de-aromatize)."""
    steps, states = corrupt_to_source(
        target, depth, system=system, rng=rng, restate_prob=restate_prob,
        catalog=catalog, vocabulary=vocabulary,
    )
    if len(states) < 2:
        return None, None
    trim = RewriteTrace(
        source=states[0],
        target=states[-1],
        steps=tuple(RewriteStep(rule_name, action) for rule_name, action in steps),
        metadata={"prior": "corrupted_source_trim", "n_steps": len(steps)},
    )
    grow: RewriteTrace | None = None
    if any(rule_name == "ring_system_delete" for rule_name, _ in steps):
        # Ring OPENING is taught TRIM-ONLY (de-cyclize from the real lead). Its exact inverse is a
        # ring_system_grow on a HETEROATOM scaffold (the opened ring keeps its N/O), which B's
        # carbon-scaffold grow vocabulary cannot score (-> inf teacher-rate). Ring-CLOSING stays
        # retained from B's de-novo grow, so B-edit reaches parity with B's ring vocabulary.
        return trim, None
    try:
        grow_steps = tuple(
            inverse_step(states[i], RewriteStep(rule_name, action))
            for i, (rule_name, action) in reversed(list(enumerate(steps)))
        )
        state = states[-1]
        for step in grow_steps:  # verify the exact inverse replays to the real target
            state = system.apply(state, step.rule_name, step.action)
        if canonical_state_key(state) == canonical_state_key(states[0]):
            grow = RewriteTrace(
                source=states[-1],
                target=states[0],
                steps=grow_steps,
                metadata={"prior": "corrupted_source_grow", "n_steps": len(steps)},
            )
    except Exception:  # noqa: BLE001
        grow = None
    return trim, grow
