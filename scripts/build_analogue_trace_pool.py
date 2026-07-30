#!/usr/bin/env python3
"""A2.1 analogue trace-pool builder -- mine one-cut MMP pairs and compile each into a verified,
trainer-consumable rewrite trace (both directions), with per-pair curriculum diagnostics.

Method (per pair sharing a constant core via one acyclic single-bond cut, acyclic variable fragment):
    First prefer a one-step atom_restate, bond_reorder, or pendant bond_reroute already exposed by the
    RingCore-V1 editing support, but only after executing it through the production runtime and proving
    its canonical successor is B. Multiple same-family marks in one canonical-successor fiber are retained
    as alias multiplicity: a deterministic exact representative defines the persistent trace while the
    successor objective aggregates the complete fiber. Cross-family ambiguity is not resolved arbitrarily.
    Otherwise:
        A --peel R_a leaf-to-root (atom_delete)--> core_a   (A's padded-slot layout)
        core_a --graft R_b (atom_insert)--> B, where each R_b insert is the exact inverse (inverse_step,
            so h/charge are exact) of B's own peel, remapped into A's layout through a typed core graph
            isomorphism core_b -> core_a with fresh free slots for the R_b atoms.
Both directions are compiled INDEPENDENTLY (A->B and B->A) so neither carries database-ordering bias,
and each is verified STRONGER than endpoint replay: every intermediate valid + connected, exact
endpoint canonical isomorphism vs target, forward AND inverse replay (round-trip source ->pi target
->pi^-1 source). Only fully verified traces enter the pool.

Trainer-consumable record shape (mirrors experiments.corrupted_source_prior.PathRecord): each pool
record stores the EXACT source SMILES -- our MolecularGraph preserves RDKit atom order, so
pad_molecular_graph(smiles_to_molecular_graph(source_smiles), n_slots) reproduces the exact padded
slot layout the step indices reference -- plus the operator steps and n_slots. rewrite_trace_from_
record replays the steps to rebuild the RewriteTrace; wrap it as PathRecord(target_key,
TraceProgressCTMC(trace)) to feed the GM trainer. --self-check re-executes a sample of the written
pool through the real executor to prove consumability.

Outputs (under --out-dir): analogue_trace_pool.jsonl (verified traces), analogue_pair_diagnostics.
jsonl (one line per mined pair incl. failures -- feeds compiler-success-by-type), and
analogue_trace_pool_summary.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdmolops

from compose_v4.data.charge_policy import (
    charge_policy_preserved as _charge_policy_preserved,
)
from compose_v4.chem.molecular_graph import (
    NULL_IDX,
    ORGANIC_VOCABULARY,
    MolecularGraphError,
    is_element,
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
from compose_v4.chem.state import is_connected_or_null, is_valid_state, pad_molecular_graph
from compose_v4.experiments.analogue_prior import ANALOGUE_SUPPORT_CONTRACT
from compose_v4.rewrite.action_codec import canonical_json, encode_action
from compose_v4.rewrite.factorized_fiber import (
    _factorized_candidates,
    enumerate_pendant_graft_actions,
)
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
from compose_v4.rewrite.operators import (
    AtomDelete,
    AtomInsert,
    AtomRestate,
    BondReorder,
    BondReroute,
)
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace, execute_trace, inverse_step

RDLogger.DisableLog("rdApp.*")

_ROOT = Path(__file__).resolve().parents[1]
CORPUS = _ROOT / "results/tree_fcd_transfer_stage1_factorized_v1/guacamol_heldout_val_5000_seed0.smiles"
OUT_DIR = _ROOT / "diagnostics" / "composition"
SYSTEM = de_novo_rewrite_system()
# ---- molecular graph <-> networkx (typed) for the peel + core isomorphism ----
def _state_graph(state) -> nx.Graph:
    real = [int(v) for v in np.flatnonzero(is_element(state.atom_types))]
    graph = nx.Graph()
    for v in real:
        graph.add_node(v, t=int(state.atom_types[v]), c=int(state.formal_charges[v]),
                       h=int(state.implicit_h_counts[v]))
    for i in range(len(real)):
        for j in range(i + 1, len(real)):
            a, b = real[i], real[j]
            order = int(state.bonds[a, b])
            if order != 0:
                graph.add_edge(a, b, o=order)
    return graph


# ---- one-cut MMP fragmentation (one acyclic single bond, small acyclic variable fragment) ----
def iter_one_cut_transformations(smi: str, *, max_variable_atoms: int = 8):
    """Yield (constant_core_smiles, variable_atom_indices) for every single acyclic-bond cut whose
    smaller fragment is acyclic and within the variable-size cap."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
            continue
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        editable = Chem.RWMol(mol)
        editable.RemoveBond(a, b)
        frags = rdmolops.GetMolFrags(editable)
        if len(frags) != 2:
            continue
        first, second = set(frags[0]), set(frags[1])
        core, variable = (first, second) if len(first) >= len(second) else (second, first)
        if any(mol.GetAtomWithIdx(i).IsInRing() for i in variable):
            continue
        if not (1 <= len(variable) <= max_variable_atoms):
            continue
        core_smi = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(core), canonical=True)
        yield core_smi, frozenset(int(i) for i in variable)


def mine_one_cut_pairs(smiles, *, max_variable_atoms: int = 8):
    """Return a deduped list of (smi_a, smi_b) unordered pairs sharing a one-cut constant core."""
    core_index: dict[str, dict[str, frozenset]] = defaultdict(dict)
    for smi in smiles:
        for core_smi, variable in iter_one_cut_transformations(
                smi, max_variable_atoms=max_variable_atoms):
            core_index[core_smi].setdefault(smi, variable)
    pairs, seen = [], set()
    for members in core_index.values():
        items = list(members.keys())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                sa, sb = items[i], items[j]
                key = frozenset((sa, sb))
                if sa != sb and key not in seen:
                    seen.add(key)
                    pairs.append((sa, sb))
    return pairs


# ---- the A2.1 compiler: exact direct edit, else peel R + inverted-peel graft ----
def peel_delete(state, variable_atoms):
    """Delete variable_atoms leaf-to-root (atom_delete on the current degree-1 atom). Returns
    (steps, states) with states[0]=state and states[-1]=core, or (None, None) on no valid order."""
    steps, states = [], [state]
    remaining = {int(v) for v in variable_atoms}
    node = state
    while remaining:
        graph = _state_graph(node)
        leaf = next((v for v in remaining if graph.degree(v) == 1), None)
        if leaf is None:
            return None, None
        try:
            successor = SYSTEM.apply(node, "atom_delete", AtomDelete(leaf))
        except Exception:  # noqa: BLE001
            return None, None
        if not (is_valid_state(successor) and is_connected_or_null(successor)):
            return None, None
        steps.append(RewriteStep("atom_delete", AtomDelete(leaf)))
        states.append(successor)
        node = successor
        remaining.discard(leaf)
    return steps, states


def core_isomorphism(core_b, core_a):
    """Return {core_b slot -> core_a slot} for the typed (element/charge/h + bond-order) core graph
    isomorphism, or None."""
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        _state_graph(core_b), _state_graph(core_a),
        node_match=lambda x, y: x["t"] == y["t"] and x["c"] == y["c"] and x["h"] == y["h"],
        edge_match=lambda x, y: x["o"] == y["o"],
    )
    for mapping in matcher.isomorphisms_iter():
        return {int(k): int(v) for k, v in mapping.items()}
    return None


def remap_inserts(inserts, sigma, core_a):
    """Remap B's inverted peel (atom_insert steps, target-R atoms outward) into A's core layout,
    allocating fresh free slots and rewriting every neighbour through sigma."""
    free = [
        int(v)
        for v in np.flatnonzero(core_a.atom_types == NULL_IDX)
    ]
    sigma = dict(sigma)
    out = []
    for next_free, step in enumerate(inserts):
        action = step.action
        if next_free >= len(free):
            return None
        new_slot = free[next_free]
        sigma[int(action.slot)] = new_slot
        new_neighbors = []
        for neighbor, order in action.neighbors:
            if int(neighbor) not in sigma:
                return None
            new_neighbors.append((sigma[int(neighbor)], int(order)))
        out.append(RewriteStep("atom_insert", AtomInsert(
            slot=new_slot, atom_type=int(action.atom_type),
            formal_charge=int(action.formal_charge),
            implicit_h_count=int(action.implicit_h_count), neighbors=tuple(new_neighbors))))
    return out


def _cycle_edges(state) -> set[frozenset[int]]:
    graph = _state_graph(state)
    return {
        frozenset((int(a), int(b))) for a, b in graph.edges()
    } - {
        frozenset((int(a), int(b))) for a, b in nx.bridges(graph)
    }


def _touches_charged_atom(state, rule_name, action) -> bool:
    if rule_name == "atom_restate":
        vertices = (int(action.v),)
    elif rule_name == "bond_reorder":
        vertices = (int(action.a), int(action.b))
    elif rule_name == "bond_reroute":
        vertices = (
            int(action.a),
            int(action.b),
            int(action.u),
            int(action.v),
        )
    else:
        raise ValueError(f"not a direct analogue operator: {rule_name!r}")
    return any(int(state.formal_charges[vertex]) != 0 for vertex in vertices)


def _ringcore_v1_direct_candidates(state):
    """Yield direct actions represented by the frozen editing support contract.

    Atom labels and bond changes come from the same factorized support
    enumerators used to construct production masks, never from an inferred
    source/target atom map.  This contract requires broad-organic vocabulary,
    heteroatom scanning, neutral charge preservation, non-ring bond reorder,
    and the production pendant-tree graft subset.
    """

    cycle_edges = _cycle_edges(state)
    for rule_name, action in _factorized_candidates(
        state,
        allow_bond_reroute=False,
        vocabulary=ORGANIC_VOCABULARY,
    ):
        if rule_name == "atom_restate":
            if (
                int(action.formal_charge) == 0
                and not _touches_charged_atom(state, rule_name, action)
            ):
                yield rule_name, action
        elif rule_name == "bond_reorder":
            edge = frozenset((int(action.a), int(action.b)))
            old_order = int(state.bonds[int(action.a), int(action.b)])
            if (
                edge not in cycle_edges
                and old_order in (1, 2, 3)
                and not _touches_charged_atom(state, rule_name, action)
            ):
                yield rule_name, action
    for action in enumerate_pendant_graft_actions(state):
        if not _touches_charged_atom(state, "bond_reroute", action):
            yield "bond_reroute", action


def _ringcore_v1_action_is_supported(state, step: RewriteStep) -> bool:
    """Require exact membership in the declared support-only action fiber."""

    if step.rule_name == "bond_reroute":
        return (
            not _touches_charged_atom(state, step.rule_name, step.action)
            and step.action in enumerate_pendant_graft_actions(state)
        )
    if step.rule_name not in {
        "atom_insert",
        "atom_delete",
        "atom_restate",
        "bond_reorder",
    }:
        return False
    for rule_name, action in _factorized_candidates(
        state,
        allow_bond_reroute=False,
        vocabulary=ORGANIC_VOCABULARY,
    ):
        if rule_name != step.rule_name or action != step.action:
            continue
        if rule_name == "atom_restate" and (
            int(action.formal_charge) != 0
            or _touches_charged_atom(state, rule_name, action)
        ):
            return False
        return not (
            rule_name == "bond_reorder"
            and (
                frozenset((int(action.a), int(action.b)))
                in _cycle_edges(state)
                or _touches_charged_atom(state, rule_name, action)
            )
        )
    return False


def _action_sort_key(step: RewriteStep) -> str:
    return canonical_json(encode_action(step.rule_name, step.action))


def _local_environment_invariant(state) -> tuple[tuple, ...]:
    """Cheap necessary invariant for canonical molecular equality.

    This is deliberately only a rejection filter.  Every real atom contributes
    its element, charge, hydrogens, and sorted labeled one-hop environment.
    Isomorphic molecular graphs must agree, but agreement still proceeds to
    the authoritative canonical identity.
    """

    real = tuple(int(v) for v in np.flatnonzero(is_element(state.atom_types)))
    return tuple(
        sorted(
            (
                int(state.atom_types[v]),
                int(state.formal_charges[v]),
                int(state.implicit_h_counts[v]),
                tuple(
                    sorted(
                        (
                            int(state.atom_types[u]),
                            int(state.formal_charges[u]),
                            int(state.bonds[v, u]),
                        )
                        for u in real
                        if int(state.bonds[v, u]) != 0
                    )
                ),
            )
            for v in real
        )
    )


def _executor_verified_direct_path(source, target):
    """Return one production-represented canonical-successor path, or reject.

    Canonical endpoint equality proves the molecular transformation. Multiple
    marks from one family are within-fiber aliases under successor supervision;
    their deterministic minimum supplies the exact persistent trace. A
    cross-family match remains unresolved because choosing its teacher family
    would change family-route diagnostics.
    """

    source_key = canonical_state_key(source)
    target_key = canonical_state_key(target)
    if source_key == target_key:
        return None, "self_target", 0
    if source.n_real_atoms != target.n_real_atoms:
        return None, "direct_cardinality_mismatch", 0
    target_invariant = _local_environment_invariant(target)
    matches = []
    for rule_name, action in _ringcore_v1_direct_candidates(source):
        try:
            successor = SYSTEM.apply(source, rule_name, action)
        except Exception:  # noqa: BLE001, S112
            continue
        if _local_environment_invariant(successor) != target_invariant:
            continue
        if canonical_state_key(successor) == target_key:
            matches.append(RewriteStep(rule_name, action))
    matched_families = {step.rule_name for step in matches}
    if len(matched_families) == 1:
        step = min(matches, key=_action_sort_key)
        return (step,), f"direct_{step.rule_name}", len(matches)
    if matches:
        return None, "ambiguous_direct_family", len(matches)
    return None, "no_direct_action", 0


def _cut_descriptors(smi_a, smi_b, core_smi, r_a, r_b) -> dict:
    mol_a = Chem.MolFromSmiles(smi_a)
    mol_b = Chem.MolFromSmiles(smi_b)
    source_var, target_var = len(r_a), len(r_b)
    variable_ring_atoms = (sum(mol_a.GetAtomWithIdx(i).IsInRing() for i in r_a)
                           + sum(mol_b.GetAtomWithIdx(i).IsInRing() for i in r_b))
    return {
        "constant_core_smiles": core_smi,
        "constant_core_heavy": int(mol_a.GetNumAtoms() - source_var),
        "source_variable_heavy": int(source_var),
        "target_variable_heavy": int(target_var),
        "variable_ring_atoms": int(variable_ring_atoms),
        "heavy_delta": int(target_var - source_var),
        "attachment_count": 1,
        "pair_type": _pair_type(source_var, target_var),
    }


def _pair_type(source_var: int, target_var: int) -> str:
    if source_var == target_var == 1:
        return "atom_swap"
    if source_var == target_var:
        return "isosteric"
    return "grow" if target_var > source_var else "shrink"


def compile_one_cut(smi_a, smi_b, *, n_slots: int, max_variable_atoms: int = 8):
    """Compile the A->B one-cut MMP edit path. Return (steps_tuple_or_None, reason, cut_meta_or_None);
    cut_meta is populated whenever a shared core matched (so failures keep their curriculum stats)."""
    try:
        a = pad_molecular_graph(smiles_to_molecular_graph(smi_a), n_slots)
        b = pad_molecular_graph(smiles_to_molecular_graph(smi_b), n_slots)
    except (MolecularGraphError, ValueError):
        return None, "unsupported_element", None
    if int(a.formal_charges[is_element(a.atom_types)].sum()) != int(
        b.formal_charges[is_element(b.atom_types)].sum()
    ):
        return None, "charge_change_unsupported", None
    direct_steps, direct_status, direct_match_count = _executor_verified_direct_path(a, b)
    cuts_b: dict[str, frozenset] = {}
    for core_smi, variable in iter_one_cut_transformations(
            smi_b, max_variable_atoms=max_variable_atoms):
        cuts_b.setdefault(core_smi, variable)
    for core_smi, r_a in iter_one_cut_transformations(smi_a, max_variable_atoms=max_variable_atoms):
        if core_smi not in cuts_b:
            continue
        r_b = cuts_b[core_smi]
        meta = _cut_descriptors(smi_a, smi_b, core_smi, r_a, r_b)
        del_a, states_a = peel_delete(a, r_a)
        del_b, states_b = peel_delete(b, r_b)
        if del_a is None or del_b is None:
            return None, "peel_fail", meta
        core_a, core_b = states_a[-1], states_b[-1]
        if canonical_state_key(core_a) != canonical_state_key(core_b):
            return None, "core_mismatch", meta
        sigma = core_isomorphism(core_b, core_a)
        if sigma is None:
            return None, "iso_fail", meta
        if direct_steps is not None:
            return direct_steps, f"compiled_{direct_status}", {
                **meta,
                "compiler_path_class": direct_status,
                "direct_path_status": direct_status,
                "direct_path_match_count": direct_match_count,
            }
        inserts_b = [inverse_step(states_b[i], del_b[i]) for i in reversed(range(len(del_b)))]
        grow = remap_inserts(inserts_b, sigma, core_a)
        if grow is None:
            return None, "remap_fail", meta
        return tuple(del_a) + tuple(grow), "compiled_delete_insert_fallback", {
            **meta,
            "compiler_path_class": "delete_insert_fallback",
            "direct_path_status": direct_status,
            "direct_path_match_count": direct_match_count,
        }
    return None, "no_matching_core", None


def verify_trace(source_smi, target_smi, steps, *, n_slots: int):
    """Return (status, states). status=='ok' requires every intermediate valid + connected, exact
    endpoint canonical isomorphism vs target, exact production-support membership, charge-policy
    preservation, and a persistent-slot-exact inverse round-trip back to the source."""
    source = pad_molecular_graph(smiles_to_molecular_graph(source_smi), n_slots)
    target = pad_molecular_graph(smiles_to_molecular_graph(target_smi), n_slots)
    try:
        _final, states = execute_trace(source, steps, system=SYSTEM, return_states=True)
    except Exception as exc:  # noqa: BLE001
        return f"exec_fail:{type(exc).__name__}", None
    for state in states:
        if not (is_valid_state(state) and is_connected_or_null(state)):
            return "invalid_intermediate", None
    for step, before, after in zip(
        steps,
        states[:-1],
        states[1:],
        strict=True,
    ):
        if not _ringcore_v1_action_is_supported(before, step):
            return "teacher_outside_ringcore_v1_support", None
        if not _charge_policy_preserved(before, after):
            return "charge_policy_violation", None
    if canonical_state_key(states[-1]) != canonical_state_key(target):
        return "endpoint_mismatch", None
    walk = states[-1]
    try:
        for i in reversed(range(len(steps))):
            inverse = inverse_step(states[i], steps[i])
            walk = SYSTEM.apply(walk, inverse.rule_name, inverse.action)
    except Exception:  # noqa: BLE001
        return "roundtrip_exec_fail", None
    if persistent_slot_state_sha256(walk) != persistent_slot_state_sha256(states[0]):
        return "persistent_slot_roundtrip_mismatch", None
    return "ok", states


# ---- (de)serialization: the pool is trainer-consumable via rewrite_trace_from_record ----
def _step_to_dict(step) -> dict:
    action = step.action
    if step.rule_name == "atom_delete":
        return {"rule": "atom_delete", "v": int(action.v)}
    if step.rule_name == "atom_insert":
        return {"rule": "atom_insert", "slot": int(action.slot), "atom_type": int(action.atom_type),
                "formal_charge": int(action.formal_charge),
                "implicit_h_count": int(action.implicit_h_count),
                "neighbors": [[int(n), int(o)] for n, o in action.neighbors]}
    if step.rule_name == "atom_restate":
        return {
            "rule": "atom_restate",
            "v": int(action.v),
            "atom_type": int(action.atom_type),
            "formal_charge": int(action.formal_charge),
            "implicit_h_count": int(action.implicit_h_count),
        }
    if step.rule_name == "bond_reorder":
        return {
            "rule": "bond_reorder",
            "a": int(action.a),
            "b": int(action.b),
            "new_order": int(action.new_order),
        }
    if step.rule_name == "bond_reroute":
        return {
            "rule": "bond_reroute",
            "a": int(action.a),
            "b": int(action.b),
            "u": int(action.u),
            "v": int(action.v),
            "new_order": int(action.new_order),
        }
    raise ValueError(f"unsupported analogue-trace rule {step.rule_name!r}")


def _dict_to_step(entry: dict):
    if entry["rule"] == "atom_delete":
        return RewriteStep("atom_delete", AtomDelete(int(entry["v"])))
    if entry["rule"] == "atom_insert":
        return RewriteStep("atom_insert", AtomInsert(
            slot=int(entry["slot"]), atom_type=int(entry["atom_type"]),
            formal_charge=int(entry["formal_charge"]),
            implicit_h_count=int(entry["implicit_h_count"]),
            neighbors=tuple((int(n), int(o)) for n, o in entry["neighbors"])))
    if entry["rule"] == "atom_restate":
        return RewriteStep(
            "atom_restate",
            AtomRestate(
                v=int(entry["v"]),
                atom_type=int(entry["atom_type"]),
                formal_charge=int(entry["formal_charge"]),
                implicit_h_count=int(entry["implicit_h_count"]),
            ),
        )
    if entry["rule"] == "bond_reorder":
        return RewriteStep(
            "bond_reorder",
            BondReorder(
                a=int(entry["a"]),
                b=int(entry["b"]),
                new_order=int(entry["new_order"]),
            ),
        )
    if entry["rule"] == "bond_reroute":
        return RewriteStep(
            "bond_reroute",
            BondReroute(
                a=int(entry["a"]),
                b=int(entry["b"]),
                u=int(entry["u"]),
                v=int(entry["v"]),
                new_order=int(entry["new_order"]),
            ),
        )
    raise ValueError(f"unknown rule {entry['rule']!r}")


def rewrite_trace_from_record(record: dict) -> RewriteTrace:
    """Rebuild the RewriteTrace a pool record encodes (the GM-trainer entry point). The source is
    rebuilt from its exact SMILES (deterministic atom order) padded to the recorded n_slots; the
    target is the executed endpoint, so trace.target is array-identical to compile time."""
    metadata = dict(record.get("metadata", {}))
    support_contract = metadata.get("support_contract")
    if support_contract != ANALOGUE_SUPPORT_CONTRACT:
        raise ValueError(
            "analogue row is not bound to the current editing-support contract: "
            f"expected {ANALOGUE_SUPPORT_CONTRACT!r}, got {support_contract!r}"
        )
    source = pad_molecular_graph(smiles_to_molecular_graph(record["source_smiles"]),
                                 int(record["n_slots"]))
    steps = tuple(_dict_to_step(entry) for entry in record["steps"])
    target = execute_trace(source, steps, system=SYSTEM)
    return RewriteTrace(
        source=source,
        target=target,
        steps=steps,
        metadata=metadata,
    )


def _compile_verify(source_smi, target_smi, *, n_slots, max_variable_atoms) -> dict:
    steps, reason, meta = compile_one_cut(source_smi, target_smi, n_slots=n_slots,
                                          max_variable_atoms=max_variable_atoms)
    if steps is None:
        return {"outcome": reason, "path_length": None, "operator_histogram": {},
                "meta": meta, "steps": None, "states": None}
    status, states = verify_trace(source_smi, target_smi, steps, n_slots=n_slots)
    if status != "ok":
        return {"outcome": status, "path_length": None, "operator_histogram": {},
                "meta": meta, "steps": steps, "states": None}
    return {"outcome": "ok", "path_length": len(steps),
            "operator_histogram": dict(Counter(step.rule_name for step in steps)),
            "meta": meta, "steps": steps, "states": states}


def _pool_record(pair_index, direction, source_smi, result, n_slots) -> dict:
    steps, states = result["steps"], result["states"]
    meta = result["meta"] or {}
    executed_target = states[-1]
    return {
        "pair_index": pair_index,
        "direction": direction,
        "source_smiles": source_smi,
        "target_smiles": molecular_graph_to_smiles(executed_target),
        "target_key": canonical_state_key(executed_target),
        "n_slots": int(n_slots),
        "path_length": len(steps),
        "operator_histogram": result["operator_histogram"],
        "steps": [_step_to_dict(step) for step in steps],
        "metadata": {"prior": "analogue_mmp_one_cut", "direction": direction,
                     "n_steps": len(steps), "pair_type": meta.get("pair_type"),
                     "support_contract": ANALOGUE_SUPPORT_CONTRACT,
                     "compiler_path_class": meta.get("compiler_path_class"),
                     "direct_path_status": meta.get("direct_path_status"),
                     "direct_path_match_count": meta.get("direct_path_match_count")},
        "diagnostics": {
            "pair_type": meta.get("pair_type"),
            "support_contract": ANALOGUE_SUPPORT_CONTRACT,
            "compiler_path_class": meta.get("compiler_path_class"),
            "direct_path_status": meta.get("direct_path_status"),
            "direct_path_match_count": meta.get("direct_path_match_count"),
            "attachment_count": meta.get("attachment_count"),
            "constant_core_heavy": meta.get("constant_core_heavy"),
            "source_variable_heavy": meta.get("source_variable_heavy"),
            "target_variable_heavy": meta.get("target_variable_heavy"),
            "variable_ring_atoms": meta.get("variable_ring_atoms"),
            "heavy_delta": meta.get("heavy_delta"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--limit", type=int, default=5000, help="max corpus molecules to mine")
    parser.add_argument("--max-pairs", type=int, default=None, help="cap mined pairs compiled")
    parser.add_argument("--n-slots", type=int, default=48)
    parser.add_argument("--max-variable-atoms", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--self-check", type=int, default=8,
                        help="re-execute this many written pool records to prove consumability")
    args = parser.parse_args()

    raw = []
    with open(args.corpus) as handle:
        for line in handle:
            token = line.strip().split()[0] if line.strip() else ""
            if token:
                raw.append(token)
    raw = raw[:args.limit]

    supported, skipped = [], 0
    for smi in raw:
        try:
            pad_molecular_graph(smiles_to_molecular_graph(smi), args.n_slots)
        except (MolecularGraphError, ValueError):
            skipped += 1
            continue
        supported.append(smi)
    print(f"corpus: {len(raw)} read | {len(supported)} supported | {skipped} skipped "
          f"(unsupported element / > {args.n_slots} slots)", flush=True)

    pairs = mine_one_cut_pairs(supported, max_variable_atoms=args.max_variable_atoms)
    if args.max_pairs is not None:
        pairs = pairs[:args.max_pairs]
    print(f"mined one-cut MMP pairs: {len(pairs)}", flush=True)

    pool_records, pair_diagnostics = [], []
    forward_outcomes, reverse_outcomes = Counter(), Counter()
    for index, (sa, sb) in enumerate(pairs):
        forward = _compile_verify(sa, sb, n_slots=args.n_slots,
                                  max_variable_atoms=args.max_variable_atoms)
        reverse = _compile_verify(sb, sa, n_slots=args.n_slots,
                                  max_variable_atoms=args.max_variable_atoms)
        forward_outcomes[forward["outcome"]] += 1
        reverse_outcomes[reverse["outcome"]] += 1
        meta = forward["meta"] or reverse["meta"] or {}
        pair_diagnostics.append({
            "pair_index": index, "smiles_a": sa, "smiles_b": sb,
            "pair_type": meta.get("pair_type"),
            "attachment_count": meta.get("attachment_count"),
            "constant_core_heavy": meta.get("constant_core_heavy"),
            "source_variable_heavy": meta.get("source_variable_heavy"),
            "target_variable_heavy": meta.get("target_variable_heavy"),
            "variable_ring_atoms": meta.get("variable_ring_atoms"),
            "heavy_delta": meta.get("heavy_delta"),
            "forward_outcome": forward["outcome"], "forward_path_length": forward["path_length"],
            "forward_operator_histogram": forward["operator_histogram"],
            "forward_compiler_path_class": (forward["meta"] or {}).get("compiler_path_class"),
            "forward_direct_path_status": (forward["meta"] or {}).get("direct_path_status"),
            "forward_direct_path_match_count": (forward["meta"] or {}).get(
                "direct_path_match_count"
            ),
            "reverse_outcome": reverse["outcome"], "reverse_path_length": reverse["path_length"],
            "reverse_operator_histogram": reverse["operator_histogram"],
            "reverse_compiler_path_class": (reverse["meta"] or {}).get("compiler_path_class"),
            "reverse_direct_path_status": (reverse["meta"] or {}).get("direct_path_status"),
            "reverse_direct_path_match_count": (reverse["meta"] or {}).get(
                "direct_path_match_count"
            ),
            "both_directions_ok": forward["outcome"] == "ok" and reverse["outcome"] == "ok",
        })
        for direction, source_smi, result in (("A->B", sa, forward), ("B->A", sb, reverse)):
            if result["outcome"] == "ok":
                pool_records.append(_pool_record(index, direction, source_smi, result, args.n_slots))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pool_path = args.out_dir / "analogue_trace_pool.jsonl"
    diag_path = args.out_dir / "analogue_pair_diagnostics.jsonl"
    summary_path = args.out_dir / "analogue_trace_pool_summary.json"
    with open(pool_path, "w") as handle:
        handle.writelines(json.dumps(record) + "\n" for record in pool_records)
    with open(diag_path, "w") as handle:
        handle.writelines(
            json.dumps(record) + "\n" for record in pair_diagnostics
        )

    checked, check_fail = 0, 0
    for record in pool_records[:max(args.self_check, 0)]:
        trace = rewrite_trace_from_record(record)
        endpoint = execute_trace(trace.source, trace.steps, system=SYSTEM)
        checked += 1
        if not (canonical_state_key(endpoint) == record["target_key"]
                and is_valid_state(endpoint) and is_connected_or_null(endpoint)):
            check_fail += 1

    lengths = [record["path_length"] for record in pool_records]
    operator_totals = Counter()
    compiler_path_totals = Counter()
    direct_status_totals = Counter()
    direct_match_count_totals = Counter()
    for record in pool_records:
        operator_totals.update(record["operator_histogram"])
        metadata = record["metadata"]
        compiler_path_totals.update(
            (str(metadata.get("compiler_path_class") or "missing"),)
        )
        direct_status_totals.update(
            (str(metadata.get("direct_path_status") or "missing"),)
        )
        direct_match_count_totals.update(
            (str(metadata.get("direct_path_match_count")),)
        )
    both_ok = sum(1 for record in pair_diagnostics if record["both_directions_ok"])
    denom = max(len(pairs), 1)
    summary = {
        "experiment": "analogue_trace_pool_a2_1",
        "corpus": str(args.corpus),
        "corpus_supported": len(supported),
        "corpus_skipped": skipped,
        "mined_pairs": len(pairs),
        "forward_outcomes": dict(forward_outcomes.most_common()),
        "reverse_outcomes": dict(reverse_outcomes.most_common()),
        "forward_success_rate": forward_outcomes["ok"] / denom,
        "reverse_success_rate": reverse_outcomes["ok"] / denom,
        "both_direction_success_rate": both_ok / denom,
        "pool_records": len(pool_records),
        "support_contract": ANALOGUE_SUPPORT_CONTRACT,
        "operator_totals": dict(operator_totals.most_common()),
        "compiler_path_totals": dict(compiler_path_totals.most_common()),
        "direct_path_status_totals": dict(direct_status_totals.most_common()),
        "direct_path_match_count_totals": dict(
            direct_match_count_totals.most_common()
        ),
        "path_length_min": min(lengths) if lengths else None,
        "path_length_median": float(np.median(lengths)) if lengths else None,
        "path_length_mean": float(np.mean(lengths)) if lengths else None,
        "path_length_max": max(lengths) if lengths else None,
        "self_check_records": checked,
        "self_check_failures": check_fail,
        "outputs": {"trace_pool": str(pool_path), "pair_diagnostics": str(diag_path),
                    "summary": str(summary_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n=== A2.1 compile+verify over {len(pairs)} one-cut MMP pairs ===", flush=True)
    print(f"forward outcomes: {summary['forward_outcomes']}", flush=True)
    print(f"forward ok {forward_outcomes['ok']}/{len(pairs)} "
          f"({100 * summary['forward_success_rate']:.1f}%) | reverse ok {reverse_outcomes['ok']}"
          f"/{len(pairs)} ({100 * summary['reverse_success_rate']:.1f}%) | both {both_ok}/"
          f"{len(pairs)} ({100 * summary['both_direction_success_rate']:.1f}%)", flush=True)
    if lengths:
        print(f"pool: {len(pool_records)} verified traces | path length min="
              f"{summary['path_length_min']} median={summary['path_length_median']:.0f} "
              f"mean={summary['path_length_mean']:.1f} max={summary['path_length_max']}", flush=True)
        print(f"operator totals: {summary['operator_totals']}", flush=True)
    print(f"self-check: {checked - check_fail}/{checked} pool records re-executed to their target",
          flush=True)
    print(f"wrote {pool_path}\n      {diag_path}\n      {summary_path}", flush=True)


if __name__ == "__main__":
    main()
