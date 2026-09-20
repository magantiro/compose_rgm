"""Per-family single-edit census for each T4 cell. DIAGNOSIS ONLY, zero oracle calls.

The `substituent_delete` / `segment_replace` prune families are covered by
`t4_prune_closure_probe.py`. This module covers the OTHER half of v1's module
vocabulary -- the five `current_state_edits` families that `_local_module` reaches:

    atom_restate_semantic, bond_reroute, cycle_close, cycle_open, ring_system_restate

`current_state_program` selects a family, enumerates ALL its legal actions on the exact
current state, and draws ONE uniformly. So the conditional law inside a family is
uniform over the enumeration, and `conditional_action_probability = 1/len(actions)` is
the controller's own reported mass. This module enumerates every action, executes it
through the production executor, gates the endpoint, and reports how much of that
uniform mass lands inside the feasible region.

Together the two probes cover every family that can produce a depth-1 endpoint.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import QED, DataStructs, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

from compose_v4.chem.molecular_graph import (
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.current_state_edits import ENUMERATORS
from compose_v4.experiments.whole_ring_plan import execute_program
from compose_v4.gates.med_chem_gate import is_valid as structurally_valid
from compose_v4.rewrite.action_codec_v4 import encode_action

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "t4_single_edit_family_census_v1"
QED_MIN, SA_MAX = 0.6, 4.0
REPRESENTABLE_HEAVY_ATOMS = 40
PROPOSAL_SLOTS = 48

FAILED = ("braf_0", "braf_1", "fa7_0", "fa7_2", "5ht1b_2")
CONTROLS = ("braf_2", "fa7_1", "5ht1b_0", "5ht1b_1")

_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _evaluate(smiles: str, source_fp, source_canonical: str) -> dict | None:
    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        return None
    canonical = Chem.MolToSmiles(molecule)
    heavy = molecule.GetNumHeavyAtoms()
    similarity = DataStructs.TanimotoSimilarity(
        source_fp, _GENERATOR.GetFingerprint(molecule)
    )
    quality = QED.qed(molecule)
    access = sascorer.calculateScore(molecule)
    eligible = (
        heavy <= REPRESENTABLE_HEAVY_ATOMS
        and similarity >= 0.6
        and quality >= QED_MIN
        and access <= SA_MAX
        and bool(structurally_valid(canonical))
        and canonical != source_canonical
    )
    return {
        "smiles": canonical,
        "similarity": round(similarity, 5),
        "qed": round(quality, 5),
        "sa": round(access, 5),
        "heavy": heavy,
        "eligible": bool(eligible),
        "sim_ok": similarity >= 0.6,
        "qed_ok": quality >= QED_MIN,
        "sa_ok": access <= SA_MAX,
    }


def census(smiles: str) -> dict:
    source = pad_molecular_graph(smiles_to_molecular_graph(smiles), PROPOSAL_SLOTS)
    source_mol = Chem.MolFromSmiles(smiles)
    source_canonical = Chem.MolToSmiles(source_mol)
    source_fp = _GENERATOR.GetFingerprint(source_mol)
    net_charge = Chem.GetFormalCharge(source_mol)

    families = {}
    for family, enumerate_actions in ENUMERATORS.items():
        try:
            actions = enumerate_actions(source)
        except (ValueError, KeyError, IndexError, TypeError, RuntimeError):
            actions = ()
        rows, failures = [], Counter()
        for action in actions:
            try:
                product, _ = execute_program(source, [encode_action(family, action)])
                endpoint = molecular_graph_to_smiles(product)
            except (ValueError, KeyError, IndexError, TypeError, RuntimeError) as error:
                failures[type(error).__name__] += 1
                continue
            row = _evaluate(endpoint, source_fp, source_canonical)
            if row is not None:
                rows.append(row)
        eligible = [row for row in rows if row["eligible"]]
        charge_changing = [
            row
            for row in rows
            if (mol := Chem.MolFromSmiles(row["smiles"])) is not None
            and Chem.GetFormalCharge(mol) != net_charge
        ]
        families[family] = {
            "enumerated_actions": len(actions),
            "executed": len(rows),
            "execution_failures": dict(failures),
            "eligible": len(eligible),
            "uniform_eligible_mass": (
                round(len(eligible) / len(actions), 6) if actions else None
            ),
            "charge_changing_endpoints": len(charge_changing),
            "best_eligible": sorted(eligible, key=lambda row: -row["qed"])[:4],
            "best_qed_among_similarity_passing": sorted(
                [row for row in rows if row["sim_ok"]], key=lambda row: -row["qed"]
            )[:3],
        }
    return {
        "smiles": smiles,
        "heavy_atoms": source_mol.GetNumHeavyAtoms(),
        "net_formal_charge": net_charge,
        "families": families,
        "total_enumerated": sum(f["enumerated_actions"] for f in families.values()),
        "total_eligible": sum(f["eligible"] for f in families.values()),
    }


def main() -> None:
    cells = {}
    for protein in ("braf", "fa7", "5ht1b", "jak2", "parp1"):
        path = ROOT / f"configs/t4_held_target_distilled_{protein}_d06_250.json"
        for row in json.loads(path.read_text())["payload"]["cells"]:
            cells[row["cell"]] = {"protein": protein, "smiles": row["smiles"]}

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "DIAGNOSTIC_EVIDENCE_ONLY_ZERO_ORACLE_CALLS",
        "oracle_calls": 0,
        "cells": {},
    }
    for name in sorted(set(FAILED) | set(CONTROLS)):
        result = census(cells[name]["smiles"])
        result["role"] = "failed" if name in FAILED else "control"
        result["protein"] = cells[name]["protein"]
        payload["cells"][name] = result
        detail = " ".join(
            f"{family.split('_')[0][:5]}={data['enumerated_actions']}/{data['eligible']}"
            for family, data in result["families"].items()
        )
        print(
            f"{name:10} {result['role']:8} q={result['net_formal_charge']:+d} "
            f"total={result['total_enumerated']:5} elig={result['total_eligible']:3}  {detail}",
            flush=True,
        )

    destination = ROOT / "diagnostics/t4_single_edit_family_census_v1.json"
    destination.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
