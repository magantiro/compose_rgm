"""Counterfactual decomposition of where the v1 T4 proposal loses a feasible witness.

DIAGNOSIS ONLY. Zero oracle calls, zero docking calls, nothing launched. Reads the
pinned contracts and the pinned proposal library; writes nothing under src/, configs/
or modal_apps/.

The question this answers: for each delta=0.6 cell whose round one produced ZERO
eligible candidates, a program-free probe already proved an eligible endpoint exists.
Where in v1's proposal distribution does that endpoint lose its probability mass?

The axes are the controller's own structural decisions:

    R          which retained region the module binds
    mode       prune / replace / grow family
    scale      how many atoms the module moves
    H          replacement topology
    alpha      attachment
    D          dependency structure across regions
    stopping   how many modules the program carries

Method. Every `substituent_delete` and `segment_replace` module in v1 begins at
`dynamic_program_synthesis._pendant_fragments`, which enumerates (fragment, anchor)
pairs obtained by cutting ONE non-ring bond such that one side disconnects and holds
at most MAX_SEGMENT_LENGTH atoms. `_delete_pendant_fragment` then draws from that set
by `rng.permutation`, i.e. UNIFORMLY, with no feasibility conditioning. So the region
law for a prune module is exactly uniform over that enumerated set, and the probability
the controller binds the region a witness needs is a counting quantity, not a black box.

This module measures:

  * the uncapped bridge-cut census per source (every single-cut prune the geometry
    admits, with its endpoint's exact gate verdict),
  * which of those endpoints are ELIGIBLE, and the fragment size each one needs,
  * the capped vocabulary the controller can actually draw (size <= MAX_SEGMENT_LENGTH),
  * the resulting uniform region mass on the productive region, and
  * the module composition depth a >cap witness requires.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import QED, DataStructs, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.dynamic_program_synthesis import (
    MAX_SEGMENT_LENGTH,
    _pendant_fragments,
)
from compose_v4.gates.med_chem_gate import is_valid as structurally_valid

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "t4_bridge_cut_census_v1"
QED_MIN, SA_MAX = 0.6, 4.0
REPRESENTABLE_HEAVY_ATOMS = 40
PROPOSAL_SLOTS = 48

FAILED = ("braf_0", "braf_1", "fa7_0", "fa7_2", "5ht1b_2")
CONTROLS = ("braf_2", "fa7_1", "5ht1b_0", "5ht1b_1")

_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


# ---- Gate, verbatim in thresholds with the audit ----


def gate(source_fp, source_smiles: str, smiles: str) -> dict:
    """Exact task gate decomposition for one endpoint. Free; no oracle involved."""
    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    if molecule is None:
        return {"stage": "chemically_valid", "verdict": False, "fragmented": True}
    if len(Chem.GetMolFrags(molecule)) != 1:
        return {"stage": "chemically_valid", "verdict": False, "fragmented": True}
    canonical = Chem.MolToSmiles(molecule)
    heavy = molecule.GetNumHeavyAtoms()
    similarity = DataStructs.TanimotoSimilarity(
        source_fp, _GENERATOR.GetFingerprint(molecule)
    )
    quality = QED.qed(molecule)
    access = sascorer.calculateScore(molecule)
    row = {
        "smiles": canonical,
        "heavy": heavy,
        "similarity": round(similarity, 5),
        "qed": round(quality, 5),
        "sa": round(access, 5),
        "fragmented": False,
        "capacity_ok": heavy <= REPRESENTABLE_HEAVY_ATOMS,
        "sim_ok": similarity >= 0.6,
        "qed_ok": quality >= QED_MIN,
        "sa_ok": access <= SA_MAX,
        "med_chem_ok": bool(structurally_valid(canonical)),
        "same_as_source": canonical == source_smiles,
    }
    if not row["capacity_ok"]:
        stage = "capacity_valid"
    elif not row["sim_ok"]:
        stage = "similarity"
    elif not row["qed_ok"]:
        stage = "qed"
    elif not row["sa_ok"]:
        stage = "sa"
    elif not row["med_chem_ok"]:
        stage = "med_chem_valid"
    else:
        stage = "eligible"
    row["stage"] = stage
    row["verdict"] = stage == "eligible"
    return row


# ---- The region vocabulary the controller actually draws from ----


def bridge_cut_census(smiles: str) -> dict:
    """Every single-non-ring-bond prune, uncapped, with its exact gate verdict.

    `_pendant_fragments` is the production enumerator and is called here at an
    uncapped `maximum`, so the census is a SUPERSET of the controller's vocabulary
    and the cap can be read off as a filter rather than assumed.
    """
    graph = pad_molecular_graph(smiles_to_molecular_graph(smiles), PROPOSAL_SLOTS)
    source_mol = Chem.MolFromSmiles(smiles)
    source_canonical = Chem.MolToSmiles(source_mol)
    source_fp = _GENERATOR.GetFingerprint(source_mol)
    heavy = source_mol.GetNumHeavyAtoms()

    uncapped = _pendant_fragments(graph, maximum=heavy)
    capped = _pendant_fragments(graph, maximum=MAX_SEGMENT_LENGTH)
    capped_keys = {(fragment, anchor) for fragment, anchor in capped}

    # Slot order must agree with RDKit atom order for the deletion to mean what the
    # fragment says. Verified per source rather than assumed.
    order_ok = _slot_order_matches(graph, source_mol)

    rows = []
    for fragment, anchor in uncapped:
        endpoint = _delete_atoms(source_mol, fragment)
        verdict = gate(source_fp, source_canonical, endpoint)
        rows.append(
            {
                "fragment": list(fragment),
                "size": len(fragment),
                "anchor": int(anchor),
                "within_vocabulary": (fragment, anchor) in capped_keys,
                **{
                    key: verdict[key]
                    for key in (
                        "smiles",
                        "stage",
                        "verdict",
                        "similarity",
                        "qed",
                        "sa",
                        "heavy",
                    )
                    if key in verdict
                },
            }
        )
    eligible = [row for row in rows if row.get("verdict")]
    eligible_in_vocabulary = [row for row in eligible if row["within_vocabulary"]]
    return {
        "smiles": smiles,
        "heavy_atoms": heavy,
        "slot_order_matches_rdkit": order_ok,
        "max_segment_length": MAX_SEGMENT_LENGTH,
        "cut_census": {
            "uncapped_fragments": len(uncapped),
            "vocabulary_fragments": len(capped),
            "eligible_single_cut": len(eligible),
            "eligible_single_cut_within_vocabulary": len(eligible_in_vocabulary),
        },
        "eligible_sizes": sorted({row["size"] for row in eligible}),
        "minimum_eligible_cut_size": min((row["size"] for row in eligible), default=None),
        "uniform_region_mass_on_eligible": (
            round(len(eligible_in_vocabulary) / len(capped), 6) if capped else None
        ),
        "eligible_rows": sorted(
            eligible, key=lambda row: (-row["qed"], row["size"])
        )[:12],
        "rows": rows,
    }


def _slot_order_matches(graph, molecule) -> bool:
    """True when padded slot i is RDKit atom i for every real atom."""
    from compose_v4.chem.molecular_graph import IDX_TO_ELEMENT, is_element

    real = [int(i) for i in np.flatnonzero(is_element(graph.atom_types))]
    if len(real) != molecule.GetNumHeavyAtoms():
        return False
    for position, slot in enumerate(real):
        element = IDX_TO_ELEMENT[int(graph.atom_types[slot])]
        if molecule.GetAtomWithIdx(position).GetSymbol() != element:
            return False
    return True


def _delete_atoms(molecule, fragment) -> str:
    editable = Chem.RWMol(molecule)
    for index in sorted(fragment, reverse=True):
        editable.RemoveAtom(int(index))
    try:
        product = editable.GetMol()
        Chem.SanitizeMol(product)
        return Chem.MolToSmiles(product)
    except (ValueError, RuntimeError):
        return ""


def main() -> None:
    contracts = {}
    for protein in ("braf", "fa7", "5ht1b", "jak2", "parp1"):
        path = ROOT / f"configs/t4_held_target_distilled_{protein}_d06_250.json"
        contracts[protein] = json.loads(path.read_text())["payload"]

    cells = {}
    for protein, contract in contracts.items():
        for row in contract["cells"]:
            cells[row["cell"]] = {"protein": protein, "smiles": row["smiles"]}

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "DIAGNOSTIC_EVIDENCE_ONLY_ZERO_ORACLE_CALLS",
        "oracle_calls": 0,
        "failed_cells": list(FAILED),
        "control_cells": list(CONTROLS),
        "cells": {},
    }
    for name in sorted(cells):
        census = bridge_cut_census(cells[name]["smiles"])
        census["protein"] = cells[name]["protein"]
        census["role"] = (
            "failed" if name in FAILED else "control" if name in CONTROLS else "other"
        )
        payload["cells"][name] = census
        print(
            f"{name:10} {census['role']:8} heavy={census['heavy_atoms']:3} "
            f"vocab={census['cut_census']['vocabulary_fragments']:4} "
            f"uncapped={census['cut_census']['uncapped_fragments']:4} "
            f"elig={census['cut_census']['eligible_single_cut']:3} "
            f"elig_in_vocab={census['cut_census']['eligible_single_cut_within_vocabulary']:3} "
            f"min_elig_size={census['minimum_eligible_cut_size']} "
            f"order_ok={census['slot_order_matches_rdkit']}",
            flush=True,
        )

    destination = ROOT / "diagnostics/t4_bridge_cut_census_v1.json"
    destination.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
