"""Emit one continuation-tournament arm per macro candidate, for one PMO task.

WHY A TOURNAMENT AND NOT A SCREEN.  MEASURED on five matched thiothixene arms: within ONE
operator family, three candidates started 0.5118 / 0.5118 / 0.5000 and continued +0.0000 /
+0.0000 / +0.0966 -- the arm with the LOWEST immediate score is the best arm, and 2-of-2
reference ring-system recovery does not order them either.  So a macro cannot be chosen by
the reward it hands back; it has to be chosen by what ordinary COMPOSE search REACHES from
where it lands.  This script builds the arms; each is then run for a fixed charged budget
and compared.

RUNG LENGTH IS MEASURED, NOT GUESSED.  Those same five arms reported at charged calls 33 /
49 / 64, and the final ordering was already correct at **33** -- the winner reached its
final 0.5966 at the first checkpoint while both dead arms were flat.  33 is therefore the
cheapest rung that has been shown to separate, and it is what `--budget` defaults to.

THE NO-MACRO ARM IS MANDATORY AND IS NOT SPECIAL-CASED: it is the incumbent leader emitted
as an ordinary arm.  A macro has to beat staying put on the same measurement.

INFORMATION REGIME.  Payloads and regions are mined from the frozen ZINC prescreen table by
the task's own oracle -- DECLARED PREPROCESSING, the regime GenMol's released PMO vocabulary
also uses.  The source molecule is the task's own charged leader, which is counted feedback
from its own run.  No reference structure, no target SMILES, no uncounted evaluation.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import itertools
import json
import pathlib

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.attachment_compatibility import admissible, build_attachment_table
from compose_v4.control.contextual_region_replace import (
    FAMILIES,
    region_excisions,
    region_replacements,
    substituent_replacements,
)
from compose_v4.control.docking_value import identity
from compose_v4.experiments import pmo_population_v1 as production

RDLogger.DisableLog("rdApp.*")
PRESCREEN = pathlib.Path("diagnostics/pmo_prescreen_v1")
CANONICAL = PRESCREEN / "zinc250k_canonical_v1.csv"
#: Measured on five matched thiothixene arms: the final ordering is correct at this many
#: charged calls, so a longer rung buys nothing for a first-round screen.
DEFAULT_RUNG_CALLS = 33


def _ranked_prescreen(task, scores_dir, canonical, top_n):
    rows = []
    with (pathlib.Path(scores_dir) / f"{task}.csv").open() as handle:
        for row in csv.DictReader(handle):
            rows.append((float(row["oracle_score"]), int(row["source_row_id"])))
    rows.sort(key=lambda r: -r[0])
    return [(score, canonical[rid]) for score, rid in rows[:top_n]]


def mine_payloads(ranked, *, min_support, max_heavy=16):
    """One-attachment DECORATED fragments, and two-attachment ring REGIONS."""
    ones: dict = collections.defaultdict(lambda: {"n": 0, "sum": 0.0})
    twos: dict = collections.defaultdict(lambda: {"n": 0, "sum": 0.0})
    for score, smiles in ranked:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumHeavyAtoms() > 40:
            continue
        cuts = [b.GetIdx() for b in mol.GetBonds()
                if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing()]
        for bond in cuts:
            try:
                parts = Chem.GetMolFrags(
                    Chem.FragmentOnBonds(mol, [bond], dummyLabels=[(1, 1)]),
                    asMols=True, sanitizeFrags=True)
            except (ValueError, RuntimeError):
                continue
            if len(parts) != 2:
                continue
            for part in parts:
                heavy = sum(1 for a in part.GetAtoms() if a.GetAtomicNum() not in (0, 1))
                if 2 <= heavy <= max_heavy:
                    entry = ones[Chem.MolToSmiles(part)]
                    entry["n"] += 1
                    entry["sum"] += score
        if not 2 <= len(cuts) <= 14:
            continue
        for first, second in itertools.combinations(cuts, 2):
            try:
                parts = Chem.GetMolFrags(
                    Chem.FragmentOnBonds(mol, [first, second],
                                         dummyLabels=[(1, 1), (2, 2)]),
                    asMols=True, sanitizeFrags=True)
            except (ValueError, RuntimeError):
                continue
            if len(parts) != 3:
                continue
            for part in parts:
                marks = sorted(a.GetIsotope() for a in part.GetAtoms()
                               if a.GetAtomicNum() == 0)
                if marks != [1, 2]:
                    continue
                heavy = sum(1 for a in part.GetAtoms() if a.GetAtomicNum() not in (0, 1))
                # A two-cut region is only worth carrying if it brings RING content: the
                # measured requirement is that 80.6% of productive transitions install a
                # ring, which a linear chain cannot supply.
                if 5 <= heavy <= 24 and part.GetRingInfo().NumRings():
                    entry = twos[Chem.MolToSmiles(part)]
                    entry["n"] += 1
                    entry["sum"] += score
    def _rank(table):
        rows = [(s, v["sum"] / v["n"], v["n"]) for s, v in table.items()
                if v["n"] >= min_support]
        rows.sort(key=lambda r: -r[1])
        return rows
    return _rank(ones), _rank(twos)


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


def write_arm(smiles, task, label, out_root):
    body = {
        "accounting": "all initialization scores count against each run's oracle budget",
        "count": production.INIT_COUNT, "available_unique": 1,
        "rule": f"macro rollout tournament arm: {label}",
        "task_independent": False,
        "candidates": [{"endpoint": smiles, "source_id": f"rollout:{label}:{i}",
                        "state": encode(smiles)}
                       for i in range(production.INIT_COUNT)],
        "source_sha256": hashlib.sha256(smiles.encode()).hexdigest(), "seed": 20260924,
    }
    folder = pathlib.Path(out_root) / f"init_{label}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{task}.json").write_text(
        json.dumps({**body, "lock_sha256": identity(body)}, indent=1) + "\n")
    return str(folder)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--leader", required=True, help="the task's own charged best")
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--arms", type=int, default=7, help="macro arms, plus no_macro")
    parser.add_argument("--top-n", type=int, default=20000)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--library-top", type=int, default=300)
    # DEFAULT NON-BINDING ON PURPOSE. Diversity-filtering the library was my first fix
    # for the census missing the winning payload class, and it MEASURED WORSE: at
    # min_support=2 a plain top-300-by-mean offers 11 of the 172 winning-class regions
    # and a diverse-300 offers 7. The real lever was min_support (3 -> 2), which takes
    # the library from 19,115 regions with 22 winning-class and NONE in the top 300, to
    # 45,452 with 172 and 11 offered. Kept as a knob, off by default.
    parser.add_argument("--library-diversity", type=float, default=1.0,
                        help="max Tanimoto between two payloads offered; 1.0 = no filter")
    parser.add_argument("--budget", type=int, default=DEFAULT_RUNG_CALLS)
    parser.add_argument("--attachment-molecules", type=int, default=60000)
    parser.add_argument("--attachment-min-support", type=int, default=2)
    parser.add_argument("--diversity", type=float, default=0.6)
    args = parser.parse_args()

    canonical = {}
    with CANONICAL.open() as handle:
        for row in csv.DictReader(handle):
            canonical[int(row["source_row_id"])] = row["canonical_smiles"]

    table = build_attachment_table(
        [canonical[k] for k in sorted(canonical)[:args.attachment_molecules]])
    ranked = _ranked_prescreen(args.task, args.scores_dir, canonical, args.top_n)
    ones, twos = mine_payloads(ranked, min_support=args.min_support)
    print(f"{args.task}: mined {len(ones):,} one-cut payloads, "
          f"{len(twos):,} two-cut ring regions")

    leader = args.leader

    def diverse_library(rows, limit, threshold):
        """Top-N by library mean surfaces COMMON payloads, not rare structural classes.

        MEASURED on thiothixene, and the first reading was wrong. At min_support=3 the
        winning payload class (>=3 rings with an exocyclic ring alkene) is 22 of 19,115
        regions with best library-mean rank #387, so a top-300 cut offered ZERO and the
        census beat the incumbent on 0 of 9 arms. That looked like a ranking problem. It
        was a SUPPORT-THRESHOLD problem: at min_support=2 the library holds 172 of that
        class and a plain top-300-by-mean already offers 11, while this diversity filter
        offers only 7. The filter is therefore OFF by default and kept only as a knob.
        """
        kept, fingerprints = [], []
        for smiles, mean, n in rows:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
            if fingerprints and max(DataStructs.BulkTanimotoSimilarity(
                    fingerprint, fingerprints)) > threshold:
                continue
            kept.append((smiles, {"library_mean": mean, "library_n": n}))
            fingerprints.append(fingerprint)
            if len(kept) >= limit:
                break
        return kept

    one_lib = diverse_library(ones, args.library_top, args.library_diversity)
    two_lib = diverse_library(twos, args.library_top, args.library_diversity)
    print(f"  library after diversity: {len(one_lib)} one-cut, {len(two_lib)} two-cut")
    families: dict = {}
    families["substituent_replace"] = substituent_replacements(
        leader, one_lib, limit=400)
    families["region_replace"] = region_replacements(leader, two_lib, limit=400)
    families["region_excise"] = region_excisions(leader, min_removed=1, max_removed=24)

    # The one unconstrained degree of freedom a context-preserving macro still has is the
    # bond it CREATES; an unfiltered bank put disulfones in half its seats.
    pool, refused = [], 0
    for family, proposals in families.items():
        for proposal in proposals:
            if not admissible(leader, proposal.endpoint, table,
                              min_support=args.attachment_min_support):
                refused += 1
                continue
            pool.append(proposal)
        print(f"  {family:22s} {len(proposals):5d} produced")
    print(f"  {refused} refused on attachment compatibility; {len(pool)} admissible")

    # STRATIFY BY FAMILY. Ranking the pooled candidates by library mean put all seven
    # thiothixene arms in ONE family, which would leave the census matrix with empty
    # columns and answer nothing about which family helps. Take a quota per family, then
    # backfill from whatever is left, and PRINT the realized strata.
    # Diversity within a stratum, not library rank: two near-identical arms waste a rung,
    # measured on the thiothixene Cl/F siblings which scored identically and were both dead.
    by_family: dict = collections.defaultdict(list)
    for proposal in pool:
        by_family[proposal.family].append(proposal)
    for proposals in by_family.values():
        proposals.sort(key=lambda p: -p.detail.get("library_mean", 0.0))

    chosen, fingerprints = [], []

    def _take(proposal):
        mol = Chem.MolFromSmiles(proposal.endpoint)
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        if fingerprints and max(DataStructs.BulkTanimotoSimilarity(
                fingerprint, fingerprints)) > args.diversity:
            return False
        chosen.append(proposal)
        fingerprints.append(fingerprint)
        return True

    present = [f for f in FAMILIES if by_family.get(f)]
    quota = max(1, args.arms // max(1, len(present)))
    for family in present:
        taken = 0
        for proposal in by_family[family]:
            if len(chosen) >= args.arms or taken >= quota:
                break
            if _take(proposal):
                taken += 1
    for proposal in pool:  # backfill, still diversity-filtered
        if len(chosen) >= args.arms:
            break
        if proposal not in chosen:
            _take(proposal)
    realized = collections.Counter(p.family for p in chosen)
    print(f"  realized strata: {dict(realized)} (quota {quota} over {present})")

    out_root = pathlib.Path(args.out)
    arms = [{"label": f"{args.task}__no_macro", "family": "no_macro",
             "endpoint": leader, "bank": write_arm(
                 leader, args.task, f"{args.task}__no_macro", out_root)}]
    for index, proposal in enumerate(chosen):
        label = f"{args.task}__{proposal.family}_{index}"
        arms.append({"label": label, "family": proposal.family,
                     "endpoint": proposal.endpoint,
                     "removed_atoms": proposal.removed_atoms,
                     "installed_atoms": proposal.installed_atoms,
                     "library_mean": proposal.detail.get("library_mean"),
                     "bank": write_arm(proposal.endpoint, args.task, label, out_root)})
    plan = {"task": args.task, "leader": leader, "budget_per_arm": args.budget,
            "rung_length_is_measured": (
                "33 charged calls reproduced the final ordering of five matched "
                "thiothixene arms at their first checkpoint"),
            "arms": arms, "charged_calls_total": args.budget * len(arms)}
    (out_root / f"plan_{args.task}.json").write_text(json.dumps(plan, indent=1) + "\n")
    for arm in arms:
        print(f"  ARM {arm['family']:22s} {arm['endpoint']}")
    print(f"  -> {len(arms)} arms x {args.budget} = {plan['charged_calls_total']} charged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
