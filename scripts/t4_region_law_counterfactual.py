"""Counterfactual arms over v1's prune region law. DIAGNOSIS ONLY, zero oracle calls.

`t4_prune_closure_probe.py` established that every eligible endpoint reachable by
pruning IS inside v1's support, and that the outcome of a cell tracks the MASS RANK
that support assigns it. This module asks which single decision, changed, moves that
rank -- and which does not.

Arms, all on the same exact closure so they are comparable state for state:

    v1              uniform over bounded pendant cuts, cap = MAX_SEGMENT_LENGTH (8)
    wider_scale     uniform, cap raised to 16 -- the SCALE axis alone
    similarity_conditioned
                    cap 8, uniform over the cuts whose immediate child still passes
                    the similarity gate; every other cut keeps a floor so support is
                    preserved -- the REGION axis, conditioned on one bit of already-free
                    information
    both            cap 16 and similarity-conditioned

Nothing here is tuned: the conditioning is the task's own similarity threshold, which
the controller can already evaluate for free at proposal time, and the floor is a fixed
support-preserving epsilon. The point is attribution, not a controller.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import QED, DataStructs, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

from compose_v4.control.dynamic_program_synthesis import MAX_SEGMENT_LENGTH
from compose_v4.gates.med_chem_gate import is_valid as structurally_valid

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "t4_region_law_counterfactual_v1"
QED_MIN, SA_MAX = 0.6, 4.0
REPRESENTABLE_HEAVY_ATOMS = 40
HORIZON = 3
SUPPORT_FLOOR = 0.05

FAILED = ("braf_0", "braf_1", "fa7_0", "fa7_2", "5ht1b_2")
CONTROLS = ("braf_2", "fa7_1", "5ht1b_0", "5ht1b_1")

ARMS = {
    "v1": {"cap": MAX_SEGMENT_LENGTH, "conditioned": False},
    "wider_scale": {"cap": 16, "conditioned": False},
    "similarity_conditioned": {"cap": MAX_SEGMENT_LENGTH, "conditioned": True},
    "both": {"cap": 16, "conditioned": True},
}

_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _submol(molecule, kept):
    editable = Chem.RWMol(molecule)
    for index in sorted(set(range(molecule.GetNumAtoms())) - set(kept), reverse=True):
        editable.RemoveAtom(int(index))
    try:
        product = editable.GetMol()
        Chem.SanitizeMol(product)
    except (ValueError, RuntimeError):
        return None
    return product


def _cuts(molecule, kept, cap):
    index_of = {atom: position for position, atom in enumerate(sorted(kept))}
    back = {position: atom for atom, position in index_of.items()}
    current = _submol(molecule, kept)
    if current is None:
        return []
    adjacency = {i: set() for i in range(current.GetNumAtoms())}
    for bond in current.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[a].add(b)
        adjacency[b].add(a)
    total = current.GetNumAtoms()
    found = set()
    for bond in current.GetBonds():
        if bond.IsInRing():
            continue
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        seen, stack = {a}, [a]
        while stack:
            at = stack.pop()
            for neighbour in adjacency[at]:
                if (at, neighbour) in ((a, b), (b, a)) or neighbour in seen:
                    continue
                seen.add(neighbour)
                stack.append(neighbour)
        for side in (seen, set(range(total)) - seen):
            if 1 <= len(side) <= cap and len(side) < total:
                found.add(frozenset(back[i] for i in side))
    return sorted(found, key=lambda f: (len(f), sorted(f)))


def _properties(molecule, source_fp, source_canonical):
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        return None
    canonical = Chem.MolToSmiles(molecule)
    heavy = molecule.GetNumHeavyAtoms()
    similarity = DataStructs.TanimotoSimilarity(
        source_fp, _GENERATOR.GetFingerprint(molecule)
    )
    quality = QED.qed(molecule)
    access = sascorer.calculateScore(molecule)
    return {
        "smiles": canonical,
        "similarity": similarity,
        "qed": quality,
        "sa": access,
        "heavy": heavy,
        "sim_ok": similarity >= 0.6,
        "eligible": bool(
            heavy <= REPRESENTABLE_HEAVY_ATOMS
            and similarity >= 0.6
            and quality >= QED_MIN
            and access <= SA_MAX
            and structurally_valid(canonical)
            and canonical != source_canonical
        ),
    }


def run_arm(source, source_fp, source_canonical, cap, conditioned, cache):
    universe = frozenset(range(source.GetNumAtoms()))
    visited = {universe: 0.0}
    visited[universe] = 1.0
    frontier = {universe: 1.0}
    for _ in range(HORIZON):
        nxt = {}
        for kept, mass in frontier.items():
            cuts = _cuts(source, kept, cap)
            if not cuts:
                continue
            children = []
            for fragment in cuts:
                child = kept - fragment
                if not child:
                    continue
                if child not in cache:
                    cache[child] = _properties(
                        _submol(source, child), source_fp, source_canonical
                    )
                children.append((child, cache[child]))
            if not children:
                continue
            if conditioned:
                weights = [
                    1.0 if (row is not None and row["sim_ok"]) else SUPPORT_FLOOR
                    for _, row in children
                ]
            else:
                weights = [1.0] * len(children)
            total = sum(weights)
            for (child, _), weight in zip(children, weights):
                contribution = mass * weight / total
                visited[child] = visited.get(child, 0.0) + contribution
                nxt[child] = nxt.get(child, 0.0) + contribution
        frontier = nxt
        if not frontier:
            break

    rows = []
    for kept, mass in visited.items():
        if kept == universe:
            continue
        row = cache.get(kept)
        if row is None:
            continue
        rows.append({**row, "path_mass": mass})
    ranked = sorted(rows, key=lambda row: -row["path_mass"])
    for position, row in enumerate(ranked, start=1):
        row["mass_rank"] = position
    eligible = sorted(
        [row for row in rows if row["eligible"]], key=lambda row: -row["path_mass"]
    )
    total_mass = sum(row["path_mass"] for row in rows) or 1.0
    return {
        "cap": cap,
        "conditioned": conditioned,
        "closure_states": len(rows),
        "eligible_states": len(eligible),
        "eligible_mass_share": round(
            sum(row["path_mass"] for row in eligible) / total_mass, 8
        ),
        "best_eligible_rank": eligible[0]["mass_rank"] if eligible else None,
        "best_eligible_mass": round(eligible[0]["path_mass"], 8) if eligible else None,
        "eligible_ranks": [row["mass_rank"] for row in eligible][:8],
        "best_eligible_smiles": eligible[0]["smiles"] if eligible else None,
    }


def main() -> None:
    cells = {}
    for protein in ("braf", "fa7", "5ht1b", "jak2", "parp1"):
        path = ROOT / f"configs/t4_held_target_distilled_{protein}_d06_250.json"
        for row in json.loads(path.read_text())["payload"]["cells"]:
            cells[row["cell"]] = row["smiles"]

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "DIAGNOSTIC_EVIDENCE_ONLY_ZERO_ORACLE_CALLS",
        "oracle_calls": 0,
        "horizon": HORIZON,
        "support_floor": SUPPORT_FLOOR,
        "arms": ARMS,
        "cells": {},
    }
    for name in sorted(set(FAILED) | set(CONTROLS)):
        smiles = cells[name]
        source = Chem.MolFromSmiles(smiles)
        source_fp = _GENERATOR.GetFingerprint(source)
        source_canonical = Chem.MolToSmiles(source)
        cache: dict = {}
        result = {
            "role": "failed" if name in FAILED else "control",
            "smiles": smiles,
            "arms": {},
        }
        for arm, settings in ARMS.items():
            result["arms"][arm] = run_arm(
                source,
                source_fp,
                source_canonical,
                settings["cap"],
                settings["conditioned"],
                cache,
            )
        payload["cells"][name] = result
        line = " | ".join(
            f"{arm}: rank={data['best_eligible_rank']}/{data['closure_states']} "
            f"mass={data['eligible_mass_share']:.4f}"
            for arm, data in result["arms"].items()
        )
        print(f"{name:10} {result['role']:8} {line}", flush=True)

    destination = ROOT / "diagnostics/t4_region_law_counterfactual_v1.json"
    destination.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
