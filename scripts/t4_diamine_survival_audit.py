"""Why do generated diamine-ring endpoints die before reaching the feasible set?

38 of 2,375 base programs from the JAK2 root produce an endpoint carrying a diamine ring,
and roughly 1 in 1,610 FEASIBLE endpoints has one. That gap is the sharpest fact in the
basin analysis, and it separates two very different problems: too little proposal mass, or
proposals that cannot survive the constraint corridor.

So every diamine endpoint is classified by the first gate it fails, and -- because
"contains a diamine ring" is far too coarse -- by whether the ring incorporates the
SEMANTIC ROLE that matters. The strong basin needs the root's ester oxygen (slot 1) to
become a nitrogen that is itself in the new ring. A piperazine acquired somewhere else on
the molecule is a different event that happens to match the same SMARTS.

Ring membership is computed on the GRAPH, not on a SMILES round trip, so the slot identity
is exact rather than inferred from atom ordering. Costs no oracle calls.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, DataStructs, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
import os
import sys

from rdkit.Chem import RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

from compose_v4.chem.molecular_graph import molecular_graph_to_smiles, smiles_to_molecular_graph
from compose_v4.chem.state import is_element, pad_molecular_graph
from compose_v4.control.dynamic_program_synthesis import synthesize_dynamic_program
from compose_v4.control.structural_subgoal import (
    attachment_bindings,
    extract_structural_goal,
    instantiate_goal,
)
from compose_v4.gates.med_chem_gate import validity_reasons

JAK2_ROOT = "COC(=O)CC2Nc1ccccc1c3ccnc4[nH]cc2c34"
ESTER_OXYGEN = 1
NITROGEN = 3
DELTA, QED_MIN, SA_MAX = 0.6, 0.6, 4.0

ESTER = Chem.MolFromSmarts("[CX4][OX2][CX3](=O)[CH2]")
AMIDE = Chem.MolFromSmarts("[NX3][CX3](=O)[CH2]")
NCN = Chem.MolFromSmarts("[NX3;R][CX4;R][NX3;R]")
NCCN = Chem.MolFromSmarts("[NX3;R][CX4;R][CX4;R][NX3;R]")


def graph_rings(graph):
    """Cycles of the product graph, as sets of slots. Exact; no SMILES round trip."""
    real = [int(i) for i in np.flatnonzero(is_element(graph.atom_types))]
    g = nx.Graph()
    g.add_nodes_from(real)
    for a in real:
        for b in real:
            if a < b and int(graph.bonds[a, b]):
                g.add_edge(a, b)
    return [set(cycle) for cycle in nx.cycle_basis(g)], g


def audit_endpoint(graph, smiles, generator, seed_fp):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    similarity = DataStructs.TanimotoSimilarity(seed_fp, generator.GetFingerprint(molecule))
    quality, access = QED.qed(molecule), sascorer.calculateScore(molecule)
    cycles, g = graph_rings(graph)
    real = [int(i) for i in np.flatnonzero(is_element(graph.atom_types))]

    # is the ESTER ROLE itself a ring nitrogen with a second nitrogen in its ring?
    role_is_nitrogen = (
        ESTER_OXYGEN in real and int(graph.atom_types[ESTER_OXYGEN]) == NITROGEN
    )
    role_in_ring = any(ESTER_OXYGEN in cycle for cycle in cycles)
    role_diamine = False
    role_ring_size = None
    for cycle in cycles:
        if ESTER_OXYGEN not in cycle:
            continue
        role_ring_size = len(cycle)
        nitrogens = [s for s in cycle if int(graph.atom_types[s]) == NITROGEN]
        if role_is_nitrogen and len(nitrogens) >= 2:
            others = [s for s in nitrogens if s != ESTER_OXYGEN]
            if any(nx.shortest_path_length(g, ESTER_OXYGEN, o) <= 2 for o in others):
                role_diamine = True
    # first gate that refuses it
    reasons = validity_reasons(smiles)
    if "." in smiles:
        first = "disconnected"
    elif reasons:
        first = f"structural:{reasons[0].split(':')[0]}"
    elif similarity < DELTA:
        first = "similarity"
    elif quality < QED_MIN:
        first = "qed"
    elif access > SA_MAX:
        first = "sa"
    elif molecule.GetNumHeavyAtoms() > 40:
        first = "size"
    else:
        first = "PASSES"
    return {
        "smiles": smiles, "similarity": similarity, "qed": quality, "sa": access,
        "similarity_margin": similarity - DELTA, "qed_margin": quality - QED_MIN,
        "sa_margin": SA_MAX - access, "heavy": molecule.GetNumHeavyAtoms(),
        "first_gate_failed": first,
        "ester_retained": bool(molecule.HasSubstructMatch(ESTER)),
        "amide_linkage": bool(molecule.HasSubstructMatch(AMIDE)),
        "ncn": bool(molecule.HasSubstructMatch(NCN)),
        "nccn": bool(molecule.HasSubstructMatch(NCCN)),
        "ester_role_is_nitrogen": role_is_nitrogen,
        "ester_role_in_ring": role_in_ring,
        "ester_role_in_diamine_ring": role_diamine,
        "ester_role_ring_size": role_ring_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=300)
    parser.add_argument("--seed", type=int, default=9100)
    parser.add_argument("--out", required=True)
    options = parser.parse_args()

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    seed_fp = generator.GetFingerprint(Chem.MolFromSmiles(JAK2_ROOT))
    rng = np.random.default_rng(options.seed)
    source = pad_molecular_graph(smiles_to_molecular_graph(JAK2_ROOT), 48)

    found, tally = [], Counter()
    for _ in range(options.draws):
        try:
            _, _, _, trace, receipt = synthesize_dynamic_program(source, rng, max_modules=3)
            goal, _, _ = extract_structural_goal(
                tuple(trace["states"]), tuple(trace["actions"])
            )
        except (ValueError, KeyError, IndexError, TypeError):
            continue
        tally["programs"] += 1
        modules = [m.get("name") for m in receipt.get("modules", []) if isinstance(m, dict)]
        rules = [a.get("executor_rule") for a in trace["actions"]]
        for rule in rules:
            tally[f"rule:{rule}"] += 1
        for module in modules:
            tally[f"module:{module}"] += 1

        bindings, ok = [], True
        for subgoal in goal.subgoals:
            census = attachment_bindings(subgoal, source)
            if not census.assignments:
                ok = False
                break
            bindings.append(census.assignments[0])
        if not ok:
            continue
        try:
            built, _ = instantiate_goal(source, goal, tuple(bindings))
            endpoint = molecular_graph_to_smiles(built)
        except (ValueError, KeyError, IndexError, TypeError):
            continue
        if not endpoint:
            continue
        molecule = Chem.MolFromSmiles(endpoint)
        if molecule is None:
            continue
        if not (molecule.HasSubstructMatch(NCN) or molecule.HasSubstructMatch(NCCN)):
            continue
        record = audit_endpoint(built, endpoint, generator, seed_fp)
        if record is None:
            continue
        record["modules"] = modules
        record["rules"] = rules
        record["touches_ester_oxygen"] = ESTER_OXYGEN in {
            int(s) for a in bindings for s in a
        }
        found.append(record)

    Path(options.out).write_text(json.dumps(
        {"diamine_endpoints": found, "tally": dict(tally), "draws": options.draws,
         "seed": options.seed}, indent=1))


if __name__ == "__main__":
    main()
