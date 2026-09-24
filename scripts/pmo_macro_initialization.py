"""Build a prescreen initialization bank that includes context-preserving macro products.

WHY.  The frozen prescreen bank is the best ZINC molecules by the task's own oracle.  Those
are good molecules, but they are molecules the corpus already contained.  MEASURED on two
tasks, a macro transition into a molecule the corpus does NOT contain can put ordinary
COMPOSE search somewhere it makes progress when the incumbent basin is dead.  This puts a
few such products in the population from call 0, where they are charged like any other seed.

INFORMATION REGIME -- the reason both sides are mined from the prescreen table.  The
substituent library comes from the top of the frozen ZINC prescreen ranking, and the sources
the macro is applied to come from the same ranking.  Both are DECLARED PREPROCESSING, the
same regime as GenMol's released PMO vocabulary, which decomposes oracle-scored ZINC into
fragments and recombines them.  Deliberately NOT used: the leading molecule of a previous
COMPOSE run on this task.  That is carried-over same-task search history, it is not part of
the declared prescreen, and building initialization from it would make the arm
uninterpretable as a no-history result even though every call is still charged.

No reference structure, no target SMILES and no similarity to one enters anywhere.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import pathlib

import numpy as np
from rdkit import Chem, RDLogger

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.attachment_compatibility import admissible, build_attachment_table
from compose_v4.control.contextual_region_replace import substituent_replacements
from compose_v4.control.docking_value import identity

RDLogger.DisableLog("rdApp.*")
PRESCREEN = pathlib.Path("diagnostics/pmo_prescreen_v1")
CANONICAL = PRESCREEN / "zinc250k_canonical_v1.csv"


def mine_substituents(scores_csv, canonical, *, top_n, min_support):
    """Decorated one-attachment fragments of the highest-scoring prescreen molecules.

    DECORATED, not bare skeletons: a bare library can only offer the ring, and the measured
    consequence is that the correct decoration is stripped from whatever it replaces.
    """
    rows = []
    with open(scores_csv) as handle:
        for row in csv.DictReader(handle):
            rows.append((float(row["oracle_score"]), int(row["source_row_id"])))
    rows.sort(key=lambda r: -r[0])
    library = collections.defaultdict(lambda: {"n": 0, "sum": 0.0})
    for score, row_id in rows[:top_n]:
        mol = Chem.MolFromSmiles(canonical[row_id])
        if mol is None:
            continue
        for bond in mol.GetBonds():
            if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
                continue
            try:
                parts = Chem.GetMolFrags(
                    Chem.FragmentOnBonds(mol, [bond.GetIdx()], dummyLabels=[(1, 1)]),
                    asMols=True, sanitizeFrags=True)
            except (ValueError, RuntimeError):
                continue
            if len(parts) != 2:
                continue
            for part in parts:
                heavy = sum(1 for a in part.GetAtoms() if a.GetAtomicNum() not in (0, 1))
                if not 2 <= heavy <= 16:
                    continue
                entry = library[Chem.MolToSmiles(part)]
                entry["n"] += 1
                entry["sum"] += score
    mined = [(smiles, value["sum"] / value["n"], value["n"])
             for smiles, value in library.items() if value["n"] >= min_support]
    mined.sort(key=lambda row: -row[1])
    return mined


def encode(smiles):
    graph = pad_molecular_graph(smiles_to_molecular_graph(smiles), 48)
    bonds = np.asarray(graph.bonds)
    return {"n_slots": int(np.asarray(graph.atom_types).shape[0]),
            "atom_types": [int(x) for x in graph.atom_types],
            "formal_charges": [int(x) for x in graph.formal_charges],
            "implicit_h_counts": [int(x) for x in graph.implicit_h_counts],
            "bonds": [[int(i), int(j), int(bonds[i, j])]
                      for i in range(bonds.shape[0])
                      for j in range(i + 1, bonds.shape[1]) if bonds[i, j] > 0]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--macro-seats", type=int, default=8)
    parser.add_argument("--sources", type=int, default=6)
    parser.add_argument("--library-top", type=int, default=400)
    parser.add_argument("--top-n", type=int, default=20000)
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--attachment-molecules", type=int, default=60000)
    parser.add_argument("--attachment-min-support", type=int, default=2)
    args = parser.parse_args()

    canonical = {}
    with CANONICAL.open() as handle:
        for row in csv.DictReader(handle):
            canonical[int(row["source_row_id"])] = row["canonical_smiles"]

    # A macro protects everything outside its cut, but the one bond it CREATES is
    # unconstrained. Measured here: an unfiltered bank put 4 disulfones in 8 macro seats.
    # The table is mined from molecular structure only -- no oracle, no task, no reference.
    table = build_attachment_table(
        [canonical[k] for k in sorted(canonical)[:args.attachment_molecules]])
    print(f"attachment table: {table.molecules:,} molecules, "
          f"{len(table.counts):,} distinct bond environments")

    frozen = json.loads(
        (PRESCREEN / "initialization" / f"{args.task}.json").read_text())
    seats = len(frozen["candidates"])
    if not 0 < args.macro_seats < seats:
        raise ValueError(f"macro seats must leave some frozen seats of {seats}")

    mined = mine_substituents(
        pathlib.Path(args.scores_dir) / f"{args.task}.csv", canonical,
        top_n=args.top_n, min_support=args.min_support)
    payloads = [(smiles, {"library_mean": mean, "library_n": n})
                for smiles, mean, n in mined[:args.library_top]]
    print(f"mined {len(mined):,} decorated substituents; using top {len(payloads)}")

    # The macro is applied to PRESCREEN molecules, never to a previous run's leader.
    keep = [row["endpoint"] for row in frozen["candidates"][:seats - args.macro_seats]]
    products, seen, refused = [], set(keep), 0
    for source in [row["endpoint"] for row in frozen["candidates"][:args.sources]]:
        for proposal in substituent_replacements(source, payloads, limit=200):
            if proposal.endpoint in seen:
                continue
            seen.add(proposal.endpoint)
            if not admissible(source, proposal.endpoint, table,
                              min_support=args.attachment_min_support):
                refused += 1
                continue
            products.append(proposal)
    print(f"{len(products)} distinct executor-valid macro products from "
          f"{args.sources} prescreen sources "
          f"({refused} refused on attachment compatibility)")

    # Diversity, not library rank: a bank of near-identical products wastes seats, and the
    # thiothixene arms measured two near-identical starts behaving identically (dead) while
    # a structurally distinct sibling carried the whole gain.
    from rdkit.Chem import AllChem, DataStructs
    chosen, fingerprints = [], []
    ranked = sorted(products, key=lambda p: -p.detail.get("library_mean", 0.0))
    for proposal in ranked:
        mol = Chem.MolFromSmiles(proposal.endpoint)
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        if fingerprints and max(DataStructs.BulkTanimotoSimilarity(
                fingerprint, fingerprints)) > 0.6:
            continue
        chosen.append(proposal)
        fingerprints.append(fingerprint)
        if len(chosen) >= args.macro_seats:
            break
    if len(chosen) < args.macro_seats:
        raise ValueError(f"only {len(chosen)} diverse macro products for "
                         f"{args.macro_seats} seats")

    endpoints = keep + [p.endpoint for p in chosen]
    body = {
        "accounting": "all initialization scores count against each run's oracle budget",
        "count": seats, "available_unique": len(set(endpoints)),
        "rule": (f"frozen prescreen top {seats - args.macro_seats} plus "
                 f"{args.macro_seats} diverse one-cut decorated macro products built from "
                 f"prescreen sources and a prescreen-mined substituent library; no "
                 f"reference structure and no prior-run molecule enters"),
        "task_independent": False,
        "candidates": [{"endpoint": smiles, "source_id": f"macro_init:{i}",
                        "state": encode(smiles)} for i, smiles in enumerate(endpoints)],
        "source_sha256": hashlib.sha256(
            "".join(endpoints).encode()).hexdigest(), "seed": 20260924,
    }
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.task}.json").write_text(
        json.dumps({**body, "lock_sha256": identity(body)}, indent=1) + "\n")
    for i, smiles in enumerate(endpoints):
        print(f"  {'frozen' if i < seats - args.macro_seats else 'MACRO ':6s} {smiles}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
