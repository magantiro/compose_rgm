"""Exact depth<=3 composed-prune closure for each T4 cell. DIAGNOSIS ONLY.

Zero oracle calls. Nothing launched. Writes only under diagnostics/.

v1's `shallow` lane runs `synthesize_dynamic_program(..., max_modules=horizon)` with
horizon=3. A pure-prune program is a sequence of at most three `substituent_delete`
(or the delete half of `segment_replace`) modules, each of which draws UNIFORMLY from
`_pendant_fragments(current, maximum=MAX_SEGMENT_LENGTH)`.

That makes the reachable endpoint set of the prune family exactly enumerable: BFS over
"delete one bounded pendant fragment", to depth 3. State is the retained atom subset of
the original molecule, which is faithful because a pendant deletion never alters the
bonds among the atoms it retains -- the cycle_open prologue inside
`_delete_pendant_fragment` only opens bonds INSIDE the fragment being removed.

What this yields that a sample cannot: the exact support, the exact uniform probability
of each leaf under the controller's own region law, and therefore the RANK and MASS a
known-feasible witness would carry if the region law were the only thing standing
between the controller and it.
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
SCHEMA_VERSION = "t4_prune_closure_probe_v1"
QED_MIN, SA_MAX = 0.6, 4.0
REPRESENTABLE_HEAVY_ATOMS = 40
HORIZON = 3

FAILED = ("braf_0", "braf_1", "fa7_0", "fa7_2", "5ht1b_2")
CONTROLS = ("braf_2", "fa7_1", "5ht1b_0", "5ht1b_1")

_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _submol(molecule, kept: frozenset[int]) -> Chem.Mol | None:
    editable = Chem.RWMol(molecule)
    for index in sorted(set(range(molecule.GetNumAtoms())) - kept, reverse=True):
        editable.RemoveAtom(int(index))
    try:
        product = editable.GetMol()
        Chem.SanitizeMol(product)
    except (ValueError, RuntimeError):
        return None
    return product


def _bounded_pendant_cuts(molecule, kept: frozenset[int]) -> list[frozenset[int]]:
    """Fragments removable by one bounded pendant deletion, in ORIGINAL indices.

    Mirrors `_pendant_fragments`: cut one bond whose removal disconnects the graph,
    keep the side holding at most MAX_SEGMENT_LENGTH atoms and leaving a non-empty
    remainder.
    """
    index_of = {atom: position for position, atom in enumerate(sorted(kept))}
    back = {position: atom for atom, position in index_of.items()}
    current = _submol(molecule, kept)
    if current is None:
        return []
    adjacency: dict[int, set[int]] = {i: set() for i in range(current.GetNumAtoms())}
    for bond in current.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[a].add(b)
        adjacency[b].add(a)
    total = current.GetNumAtoms()
    found: set[frozenset[int]] = set()
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
        left = seen
        right = set(range(total)) - left
        for side in (left, right):
            if 1 <= len(side) <= MAX_SEGMENT_LENGTH and len(side) < total:
                found.add(frozenset(back[i] for i in side))
    return sorted(found, key=lambda f: (len(f), sorted(f)))


def _evaluate(molecule, source_fp, source_canonical: str) -> dict | None:
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
        "heavy": heavy,
        "similarity": round(similarity, 5),
        "qed": round(quality, 5),
        "sa": round(access, 5),
        "eligible": bool(eligible),
        "sim_ok": similarity >= 0.6,
        "qed_ok": quality >= QED_MIN,
    }


def closure(smiles: str, horizon: int = HORIZON) -> dict:
    source = Chem.MolFromSmiles(smiles)
    source_canonical = Chem.MolToSmiles(source)
    source_fp = _GENERATOR.GetFingerprint(source)
    universe = frozenset(range(source.GetNumAtoms()))

    # frontier maps retained-set -> (depth, uniform path probability under the
    # controller's own region law: product of 1/|cuts| at each step)
    frontier = {universe: (0, 1.0)}
    visited: dict[frozenset[int], tuple[int, float]] = {universe: (0, 1.0)}
    per_depth = []
    for depth in range(1, horizon + 1):
        nxt: dict[frozenset[int], tuple[int, float]] = {}
        for kept, (_, mass) in frontier.items():
            cuts = _bounded_pendant_cuts(source, kept)
            if not cuts:
                continue
            step = 1.0 / len(cuts)
            for fragment in cuts:
                child = kept - fragment
                if not child:
                    continue
                contribution = mass * step
                if child in visited:
                    seen_depth, seen_mass = visited[child]
                    visited[child] = (seen_depth, seen_mass + contribution)
                    if child in nxt:
                        nxt[child] = (depth, nxt[child][1] + contribution)
                    continue
                visited[child] = (depth, contribution)
                nxt[child] = (depth, contribution)
        per_depth.append({"depth": depth, "new_states": len(nxt)})
        frontier = nxt
        if not frontier:
            break

    rows = []
    for kept, (depth, mass) in visited.items():
        if kept == universe:
            continue
        row = _evaluate(_submol(source, kept), source_fp, source_canonical)
        if row is None:
            continue
        row["depth"] = depth
        row["path_mass"] = mass
        row["removed"] = source.GetNumAtoms() - len(kept)
        rows.append(row)

    # Rank every reachable endpoint by the controller's own path mass, so a known
    # feasible endpoint can be reported as "rank R of N" rather than merely present.
    ranked = sorted(rows, key=lambda row: -row["path_mass"])
    for position, row in enumerate(ranked, start=1):
        row["mass_rank"] = position
        row["mass_percentile"] = round(position / len(ranked), 5) if ranked else None

    eligible = [row for row in rows if row["eligible"]]
    eligible.sort(key=lambda row: -row["path_mass"])
    by_qed = sorted(rows, key=lambda row: -row["qed"])
    sim_pass = [row for row in rows if row["sim_ok"]]
    total_mass = sum(row["path_mass"] for row in rows) or 1.0
    return {
        "smiles": smiles,
        "heavy_atoms": source.GetNumAtoms(),
        "horizon": horizon,
        "max_segment_length": MAX_SEGMENT_LENGTH,
        "per_depth": per_depth,
        "closure_states": len(rows),
        "similarity_passing": len(sim_pass),
        "eligible_states": len(eligible),
        "eligible_path_mass": round(sum(row["path_mass"] for row in eligible), 8),
        "eligible_mass_share": round(
            sum(row["path_mass"] for row in eligible) / total_mass, 8
        ),
        "best_eligible": eligible[:6],
        "best_qed_in_closure": by_qed[:4],
        "best_qed_among_similarity_passing": sorted(
            sim_pass, key=lambda row: -row["qed"]
        )[:4],
        "max_removed": max((row["removed"] for row in rows), default=0),
        "eligible_mass_ranks": [row["mass_rank"] for row in eligible],
        "best_eligible_rank": eligible[0]["mass_rank"] if eligible else None,
        "mass_of_top_ranked_state": round(ranked[0]["path_mass"], 8) if ranked else None,
    }


def main() -> None:
    cells = {}
    for protein in ("braf", "fa7", "5ht1b", "jak2", "parp1"):
        path = ROOT / f"configs/t4_held_target_distilled_{protein}_d06_250.json"
        for row in json.loads(path.read_text())["payload"]["cells"]:
            cells[row["cell"]] = {"protein": protein, "smiles": row["smiles"]}

    selected = [name for name in cells if name in set(FAILED) | set(CONTROLS)]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "DIAGNOSTIC_EVIDENCE_ONLY_ZERO_ORACLE_CALLS",
        "oracle_calls": 0,
        "cells": {},
    }
    for name in sorted(selected):
        result = closure(cells[name]["smiles"])
        result["role"] = "failed" if name in FAILED else "control"
        result["protein"] = cells[name]["protein"]
        payload["cells"][name] = result
        print(
            f"{name:10} {result['role']:8} states={result['closure_states']:5} "
            f"simpass={result['similarity_passing']:4} "
            f"elig={result['eligible_states']:4} "
            f"elig_mass={result['eligible_mass_share']:.6f} "
            f"max_removed={result['max_removed']:3}",
            flush=True,
        )

    destination = ROOT / "diagnostics/t4_prune_closure_probe_v1.json"
    destination.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
