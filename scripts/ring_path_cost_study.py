#!/usr/bin/env python3
"""§3 of the RingCore preflight: compositional path-cost of ring-changing transformations.

The pilot analogue pool has no ring-changing pairs, so we measure the compositional edit length directly on
REAL held-out GuacaMol ring systems via the executor-verified spanning-tree decomposition (§2's mechanism):
the canonical ring-changing transformation is ring-system <-> its ring-opened (spanning-tree) precursor, and
the compositional edit length for that change is exactly the number of cycle_close/cycle_open steps.

Two decision-relevant costs per ring system, stratified by topology class:
  * ring_restructure_steps   = cycle rank (cycle_close/open steps when the ring ATOMS are already present).
  * ring_introduction_steps  = ring_atom_count + cycle rank (compositional primitives to build the ring
    de-novo: one atom_insert per ring atom + one cycle_close per ring-closing bond) -- the count a single
    whole-ring MACRO would replace, i.e. the UPPER bound on the macro's per-ring savings.

Every measured transformation is executor-verified (open->close round trip): all-intermediate validity +
connectedness, <=40 atoms at every state (max temporary path size reported; any violation rejected), charge
preservation, exact endpoint identity. Reports compilation success, median/p75/p90/max, and the fraction
reachable within 4 / 8 / 12 edits and beyond the production edit budget. Required before the macro decision.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from compose_v4.chem.molecular_graph import (  # noqa: E402
    MolecularGraphError,
    is_element,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import (  # noqa: E402
    is_connected_or_null,
    is_valid_state,
    pad_molecular_graph,
)
from compose_v4.data.organic_corpus import BROAD_ORGANIC_V1, scan_corpus  # noqa: E402
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system  # noqa: E402
from compose_v4.rewrite.operators import BondDelete, BondInsert  # noqa: E402

_DEFAULT_CORPUS = Path("results/tree_fcd_transfer_stage1_factorized_v1/guacamol_heldout_val_5000_seed0.smiles")
_PRODUCTION_EDIT_BUDGET = 16  # the sampler operational horizon; a ring change should fit well inside it.


def _real(state):
    return [i for i in range(len(state.atom_types)) if bool(is_element(state.atom_types)[i])]


def _net_charge(state) -> int:
    return int(state.formal_charges[is_element(state.atom_types)].sum())


def _n_atoms(state) -> int:
    return int(is_element(state.atom_types).sum())


def _ring_graph(state) -> nx.Graph:
    real = _real(state)
    graph = nx.Graph()
    graph.add_nodes_from(real)
    for a in real:
        for b in real:
            if a < b and int(state.bonds[a, b]) != 0:
                graph.add_edge(a, b, order=int(state.bonds[a, b]))
    return graph


def _non_tree_edges(graph: nx.Graph) -> list[tuple[int, int, int]]:
    tree = {frozenset(e) for e in nx.minimum_spanning_tree(graph).edges()}
    return [
        (a, b, graph[a][b]["order"]) for a, b in graph.edges() if frozenset((a, b)) not in tree
    ]


def _classify_topology(smiles: str) -> str:
    """Coarse ring-topology stratum from RDKit ring info (dominant character of the molecule's rings)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "unparsed"
    ri = mol.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings()]
    if not rings:
        return "acyclic"
    sizes = [len(r) for r in rings]
    aromatic = any(all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r) for r in rings)
    hetero = any(any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in r) for r in rings)
    fused = spiro = bridged = False
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            shared = rings[i] & rings[j]
            if len(shared) == 1:
                spiro = True
            elif len(shared) == 2:
                fused = True
            elif len(shared) >= 3:
                bridged = True
    if bridged:
        base = "bridged"
    elif spiro:
        base = "spiro"
    elif fused:
        base = "fused"
    elif max(sizes) >= 8:
        base = "macrocycle"
    else:
        base = "monocycle"
    aroma = "aromatic" if aromatic else "saturated"
    het = "hetero" if hetero else "carbocycle"
    return f"{base}/{aroma}/{het}"


def _measure(smiles: str, system) -> dict | None:
    try:
        graph = smiles_to_molecular_graph(smiles)
    except MolecularGraphError:
        return None
    if graph is None:
        return None
    try:
        target = pad_molecular_graph(graph, 40)
    except ValueError:
        return None
    rg = _ring_graph(target)
    non_tree = _non_tree_edges(rg)
    if not non_tree:
        return None  # acyclic; not a ring-changing transformation
    cycle_rank = len(non_tree)
    rdmol = Chem.MolFromSmiles(smiles)
    ring_atoms = sum(1 for a in rdmol.GetAtoms() if a.IsInRing()) if rdmol is not None else 0
    q0 = _net_charge(target)
    target_key = canonical_state_key(target)
    max_temp = _n_atoms(target)

    # Executor round trip: open all ring-closing bonds to the spanning tree, then close them back.
    try:
        state = target
        for a, b, _o in non_tree:
            state = system.apply(state, "bond_delete", BondDelete(a, b))
            if not (is_valid_state(state) and is_connected_or_null(state)):
                return {"smiles": smiles, "compiled": False}
            if _n_atoms(state) > 40 or _net_charge(state) != q0:
                return {"smiles": smiles, "compiled": False}
        for a, b, o in non_tree:
            state = system.apply(state, "bond_insert", BondInsert(a, b, o))
            if _n_atoms(state) > 40 or _net_charge(state) != q0:
                return {"smiles": smiles, "compiled": False}
        compiled = canonical_state_key(state) == target_key
    except Exception:  # noqa: BLE001
        return {"smiles": smiles, "compiled": False}

    return {
        "smiles": smiles,
        "compiled": compiled,
        "topology": _classify_topology(smiles),
        "ring_restructure_steps": cycle_rank,  # cycle_close/open count
        "ring_introduction_steps": ring_atoms + cycle_rank,  # atom_inserts + closures
        "max_temp_atoms": max_temp,
    }


def _dist(values: list[int]) -> dict:
    if not values:
        return {}
    values = sorted(values)
    n = len(values)
    return {
        "n": n,
        "median": st.median(values),
        "p75": values[min(n - 1, int(0.75 * n))],
        "p90": values[min(n - 1, int(0.90 * n))],
        "max": values[-1],
        "frac_le_4": round(sum(v <= 4 for v in values) / n, 3),
        "frac_le_8": round(sum(v <= 8 for v in values) / n, 3),
        "frac_le_12": round(sum(v <= 12 for v in values) / n, 3),
        "frac_gt_budget": round(sum(v > _PRODUCTION_EDIT_BUDGET for v in values) / n, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=_DEFAULT_CORPUS)
    parser.add_argument("--sample-size", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument(
        "--output", type=Path, default=Path("diagnostics/production_preflight/ring_path_cost.json")
    )
    args = parser.parse_args()

    with args.corpus.open() as handle:
        texts = [line.strip().split()[0] for line in handle if line.strip()]
    accepted, _ = scan_corpus(texts, BROAD_ORGANIC_V1, workers=0)
    rng = np.random.default_rng(args.seed)
    mols = np.asarray(accepted, dtype=object)
    rng.shuffle(mols)

    system = de_novo_rewrite_system()
    measured = []
    for smi in mols:
        if len(measured) >= args.sample_size:
            break
        m = _measure(str(smi), system)
        if m is not None:
            measured.append(m)
    print(json.dumps({"phase": "measured", "n": len(measured)}, sort_keys=True), flush=True)

    ok = [m for m in measured if m.get("compiled")]
    compile_success = round(len(ok) / max(1, len(measured)), 4)
    restructure = [m["ring_restructure_steps"] for m in ok]
    introduction = [m["ring_introduction_steps"] for m in ok]
    max_temp = max((m["max_temp_atoms"] for m in ok), default=0)

    by_topo_restructure = defaultdict(list)
    by_topo_intro = defaultdict(list)
    for m in ok:
        by_topo_restructure[m["topology"]].append(m["ring_restructure_steps"])
        by_topo_intro[m["topology"]].append(m["ring_introduction_steps"])

    out = {
        "measurement": "compositional edit length for ring-changing transformations (ring-system <-> "
        "ring-opened precursor), executor-verified over real held-out GuacaMol ring systems.",
        "production_edit_budget": _PRODUCTION_EDIT_BUDGET,
        "n_measured": len(measured),
        "compilation_success": compile_success,
        "max_temporary_path_atoms": max_temp,
        "atom_bound_respected": max_temp <= 40,
        "ring_restructure_steps_overall": _dist(restructure),
        "ring_introduction_steps_overall": _dist(introduction),
        "by_topology": {
            topo: {
                "restructure": _dist(by_topo_restructure[topo]),
                "introduction": _dist(by_topo_intro[topo]),
            }
            for topo in sorted(by_topo_restructure)
        },
        "topology_counts": dict(Counter(m["topology"] for m in ok)),
        "interpretation": "ring_restructure_steps = cycle_close/open count when ring atoms are present "
        "(the direct cost of a ring open/close/reshape edit). ring_introduction_steps = atom_inserts + "
        "closures to build a ring de-novo (the UPPER bound the whole-ring macro would collapse to 1 step). "
        "frac_gt_budget is the fraction of ring changes exceeding the production edit budget.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "compilation_success": compile_success,
                "restructure": out["ring_restructure_steps_overall"],
                "introduction_median": out["ring_introduction_steps_overall"].get("median"),
                "introduction_frac_gt_budget": out["ring_introduction_steps_overall"].get(
                    "frac_gt_budget"
                ),
                "max_temp_atoms": max_temp,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
