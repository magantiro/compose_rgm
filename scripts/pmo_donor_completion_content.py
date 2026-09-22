#!/usr/bin/env python
"""POINT 7 -- can donor chemistry SUPPLY the completion content a required macro
edit needs?

This is deliberately NOT the question "does donor transport work as a search
strategy" (that was already answered: the lane is not even in the scored import
closure).  The question here is narrower and different: the census says the
productive transitions install a connected region that is ring-bearing, decorated
and around 6-20 heavy atoms.  If a macro controller declared "replacement of
scale S at this site", is the donor bank's own supply of pendant components
shaped like the regions that are actually required?

The comparison is between distributions, both computed by the same instrument:
  REQUIRED   the installed regions of the productive-basin census
  SUPPLIED   every pendant component obtainable from the task-independent init
             bank via the production ``pendant_cuts``
  CURRENT    what the running controller's ``_grow_actions`` completion can build
             (a linear single-bonded C/N/O chain of at most 8 atoms)

ZERO ORACLE CALLS.  The init bank is task-independent and objective-blind, so
this measurement carries no answer-known material on the SUPPLIED side.
"""

from __future__ import annotations

import collections
import json
import os
import sys

import networkx as nx
import numpy as np
from rdkit import Chem, RDLogger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
RDLogger.DisableLog("rdApp.*")

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph  # noqa: E402
from compose_v4.chem.state import is_element, pad_molecular_graph  # noqa: E402
from compose_v4.control.donor_program import pendant_cuts  # noqa: E402

OUT = "diagnostics/pmo_global_delta_census_v1"
PMO_SLOTS = 48  # PMO states are 48 slots; see the region-law / capacity note


def component_profile(mol: Chem.Mol, atoms: tuple[int, ...]) -> dict:
    rings = mol.GetRingInfo()
    graph = nx.Graph()
    graph.add_nodes_from(atoms)
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in atoms and b in atoms:
            graph.add_edge(a, b)
    is_path = graph.number_of_edges() == len(atoms) - 1 and all(
        graph.degree(n) <= 2 for n in graph.nodes
    )
    all_single = all(
        mol.GetBondBetweenAtoms(u, v).GetBondType() == Chem.BondType.SINGLE
        for u, v in graph.edges
    )
    only_cno = all(mol.GetAtomWithIdx(a).GetSymbol() in ("C", "N", "O") for a in atoms)
    return {
        "size": len(atoms),
        "has_ring": any(rings.NumAtomRings(a) > 0 for a in atoms),
        "heteroatoms": sum(1 for a in atoms if mol.GetAtomWithIdx(a).GetSymbol() != "C"),
        "branch_points": sum(1 for n in graph.nodes if graph.degree(n) >= 3),
        "expressible_as_grow_chain": bool(
            is_path and all_single and only_cno and len(atoms) <= 8
        ),
    }


def supplied_components() -> list[dict]:
    bank = json.load(open("docs/PMO_INIT_BANK.json"))["smiles"]
    profiles = []
    skipped = 0
    for smiles in bank:
        mol = Chem.MolFromSmiles(smiles)
        graph = smiles_to_molecular_graph(smiles)
        if mol is None or graph is None:
            skipped += 1
            continue
        # PMO proposal states are 48 slots; a tight graph is the wrong object here.
        padded = pad_molecular_graph(graph, PMO_SLOTS)
        real = [int(i) for i in np.flatnonzero(is_element(padded.atom_types))]
        if len(real) != mol.GetNumAtoms():
            skipped += 1
            continue
        for cut in pendant_cuts(padded):
            profiles.append(component_profile(mol, tuple(cut.component)))
    print(f"[strata] init-bank molecules used {len(bank) - skipped} of {len(bank)}"
          f" (skipped {skipped})")
    return profiles


def required_regions() -> dict[str, list[dict]]:
    atlas = json.load(open(os.path.join(OUT, "productive_basin_atlas_v1.json")))["records"]
    out: dict[str, list[dict]] = collections.defaultdict(list)
    for row in atlas:
        delta = row.get("delta") or {}
        if delta.get("status") != "ok" or not delta["installed_region_sizes"]:
            continue
        out[row["pool"]].append(
            {
                "size": delta["installed_region_sizes"][0],
                "has_ring": delta["installed_region_has_ring"],
                "heteroatoms": delta["installed_heteroatoms"],
                "branch_points": delta["installed_branch_points"],
                "expressible_as_grow_chain": delta["installed_expressible_as_grow_chain"],
            }
        )
    return out


def describe(name: str, rows: list[dict]) -> dict:
    n = len(rows)
    sizes = sorted(r["size"] for r in rows)
    summary = {
        "n": n,
        "size_median": sizes[n // 2],
        "size_p25": sizes[n // 4],
        "size_p75": sizes[3 * n // 4],
        "size_max": sizes[-1],
        "frac_size_ge_6": round(sum(1 for s in sizes if s >= 6) / n, 4),
        "frac_size_ge_13": round(sum(1 for s in sizes if s >= 13) / n, 4),
        "frac_has_ring": round(sum(1 for r in rows if r["has_ring"]) / n, 4),
        "frac_branched": round(sum(1 for r in rows if r["branch_points"] > 0) / n, 4),
        "frac_expressible_as_grow_chain": round(
            sum(1 for r in rows if r["expressible_as_grow_chain"]) / n, 4
        ),
        "heteroatoms_median": sorted(r["heteroatoms"] for r in rows)[n // 2],
    }
    print(
        f"{name:44s} n={n:6d}  size med {summary['size_median']:3d} "
        f"(p25 {summary['size_p25']}, p75 {summary['size_p75']}, max {summary['size_max']})"
        f"  ring {summary['frac_has_ring'] * 100:5.1f}%"
        f"  >=6 {summary['frac_size_ge_6'] * 100:5.1f}%"
        f"  >=13 {summary['frac_size_ge_13'] * 100:5.1f}%"
        f"  growchain {summary['frac_expressible_as_grow_chain'] * 100:5.1f}%"
    )
    return summary


def main() -> None:
    print("=== INSTALLED-REGION PROFILE: required vs supplied vs currently buildable ===\n")
    report = {}
    for pool, rows in sorted(required_regions().items()):
        if len(rows) >= 8:
            report[f"REQUIRED::{pool}"] = describe(f"REQUIRED  {pool}", rows)
    supplied = supplied_components()
    report["SUPPLIED::init_bank_pendant_cuts"] = describe(
        "SUPPLIED  init-bank pendant components", supplied
    )

    # The current completion channel, stated from the code rather than sampled.
    report["CURRENT::grow_actions"] = {
        "description": (
            "dynamic_program_synthesis._grow_actions builds a linear single-bonded "
            "chain of elements drawn uniformly from ('C','N','O'); segment_replace "
            "draws its length uniformly from 1..min(MAX_SEGMENT_LENGTH=8, 40-n). "
            "It can never install a ring, a branch, or an element outside C/N/O."
        ),
        "size_max": 8,
        "frac_has_ring": 0.0,
        "frac_branched": 0.0,
        "frac_expressible_as_grow_chain": 1.0,
    }
    print(
        "CURRENT   segment_replace completion            "
        "linear single-bonded C/N/O chain, length uniform on 1..8, "
        "ring 0.0%  branch 0.0%"
    )

    payload = {
        "schema_version": "pmo_donor_completion_content_v1",
        "information_regime": "DEVELOPMENT_INFORMED_DIAGNOSTIC",
        "information_regime_statement": (
            "The REQUIRED side is answer-known development evidence. The SUPPLIED side is "
            "the task-independent objective-blind init bank and carries no answer "
            "information. Nothing here may become a prior or library for a scored run."
        ),
        "new_oracle_calls": 0,
        "question": (
            "Could donor chemistry supply the completion content a required generic macro "
            "edit needs -- not whether donor transport works as a search strategy."
        ),
        "profiles": report,
    }
    with open(os.path.join(OUT, "donor_completion_content_v1.json"), "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    print(f"\nwrote {OUT}/donor_completion_content_v1.json")


if __name__ == "__main__":
    main()
