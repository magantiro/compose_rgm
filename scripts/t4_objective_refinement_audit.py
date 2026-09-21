"""Measure terminal free-objective refinement on stored T4 endpoint shards.

WHAT QUESTION THIS ANSWERS
--------------------------
A T4 cell that terminated with ``candidate_exhaustion`` produced zero eligible
endpoints.  This driver asks, per cell and without a single oracle call:

  1. does the CONTRACT comparator (``>=`` / ``<=``) admit any endpoint that a
     boundary-exclusive comparator would refuse?  That is the cheapest possible
     rescue and has to be ruled out before any policy work;
  2. of the endpoints closest to eligible, how many are lifted over the bound by
     ONE legal rewrite from the production vocabulary?

PANEL SELECTION
---------------
By the endpoint's own constraint-violation scalar ``v`` from
``t4_endpoint_selection.calculate_properties`` -- the same "select by existing
violation, never by a successor result or winner" rule ``t4_repair_neighbors``
uses.  No target, cell or seed identity enters the selection, and crucially
neither does WHICH bound the endpoint fails: on braf_1 four of five rescues are
similarity failures repaired jointly with QED, which a QED-only panel misses.

SLOT SEMANTICS
--------------
Refinement sources are padded to ``PRODUCTION_MAX_ATOMS`` (40) by
``t4_objective_refinement``.  40 is deliberate rather than the 48 the T4
proposal lanes use: ``Fiber`` refuses anything above
``REPRESENTABLE_HEAVY_ATOMS`` (40), so slots beyond 40 can only produce
successors the gate always rejects.  The tight-vs-padded census is recorded per
cell so an operator-coverage claim here is never made from a tight graph.

USAGE
-----
    python3 scripts/t4_objective_refinement_audit.py \
        --shards ~/compose_t4_support_audit_shards/cells \
        --out diagnostics/t4_objective_refinement_v1.json \
        --cell fa7_0=0.6 --panel 25
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from rdkit import Chem, RDLogger

from compose_v4.chem.molecular_graph import MolecularGraphError, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.generic_legal_action_policy import enumerate_legal_successors
from compose_v4.experiments.t4_fiber_campaign import QED_MIN, SA_MAX, Fiber
from compose_v4.experiments.t4_objective_refinement import (
    PRODUCTION_MAX_ATOMS,
    SCHEMA_VERSION,
    VIOLATION_RANKED_PANEL,
    free_objective_near_miss,
    properties_for_fiber,
    refine_endpoint,
    select_refinement_panel,
)

RDLogger.DisableLog("rdApp.*")

SCHEMA = "t4_objective_refinement_audit_v1"


def _distinct_scored(shard: dict, fiber: Fiber) -> list[dict]:
    """Every distinct, representable endpoint of the shard with its properties."""

    rows, seen = [], set()
    for record in shard["endpoints"]:
        smiles = record["smiles"]
        if not smiles or "." in smiles or smiles in seen:
            continue
        seen.add(smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumHeavyAtoms() > 40:
            continue
        props = properties_for_fiber(mol, fiber)
        if props is None:
            continue
        rows.append(
            {
                "smiles": Chem.MolToSmiles(mol),
                "lane": record.get("lane"),
                **props,
                "eligible": fiber.check(smiles) is not None,
            }
        )
    return rows


def comparator_check(rows: list[dict], delta: float) -> dict:
    """Does the contract comparator admit anything a strict one refuses?"""

    inclusive = [r for r in rows if r["sim"] >= delta and r["qed"] >= QED_MIN and r["sa"] <= SA_MAX]
    strict = [r for r in rows if r["sim"] >= delta and r["qed"] > QED_MIN and r["sa"] < SA_MAX]
    return {
        "eligible_inclusive_contract": len(inclusive),
        "eligible_strict_legacy": len(strict),
        "comparator_only_gain": len(inclusive) - len(strict),
        "exact_on_qed_bound": sum(1 for r in rows if r["qed"] == QED_MIN),
        "exact_on_sa_bound": sum(1 for r in rows if r["sa"] == SA_MAX),
        "exact_on_similarity_bound": sum(1 for r in rows if r["sim"] == delta),
    }


def slot_census(smiles: str) -> dict:
    """Tight-vs-padded legal-family census; a tight graph cannot express birth."""

    try:
        tight = enumerate_legal_successors(smiles_to_molecular_graph(smiles))
        padded = enumerate_legal_successors(
            pad_molecular_graph(smiles_to_molecular_graph(smiles), PRODUCTION_MAX_ATOMS)
        )
    except MolecularGraphError:
        return {"representable": False}
    tight_census = collections.Counter(s.rule for s in tight)
    padded_census = collections.Counter(s.rule for s in padded)
    return {
        "representable": True,
        "tight_marks": len(tight),
        "padded_marks": len(padded),
        "tight_atom_insert": tight_census.get("atom_insert", 0),
        "padded_atom_insert": padded_census.get("atom_insert", 0),
        "tight_would_forbid_atom_birth": tight_census.get("atom_insert", 0) == 0,
        "padded_census": dict(padded_census),
    }


def failing_bounds(props: dict, delta: float) -> list[str]:
    out = []
    if props["sim"] < delta:
        out.append("similarity")
    if props["qed"] < QED_MIN:
        out.append("qed")
    if props["sa"] > SA_MAX:
        out.append("sa")
    return out


def audit_cell(shard_path: Path, delta: float, panel: int) -> dict:
    shard = json.loads(shard_path.read_text())
    seed = shard["source_smiles"]
    fiber = Fiber(seed, delta)
    rows = _distinct_scored(shard, fiber)

    panel_rows = select_refinement_panel(rows, panel)
    similarity_pass = [r for r in rows if r["sim"] >= delta]
    best = max(similarity_pass, key=lambda r: r["qed"]) if similarity_pass else None

    refined, oracle_calls = [], 0
    for row in panel_rows:
        report = refine_endpoint(seed, delta, row["smiles"])
        oracle_calls += report["oracle_calls"]
        best_row = (
            max(report["eligible"], key=lambda r: r["qed"]) if report["eligible"] else None
        )
        refined.append(
            {
                "endpoint": row["smiles"],
                "v": row["v"],
                "failing_bounds": failing_bounds(row, delta),
                "qed": row["qed"],
                "similarity": row["sim"],
                "sa": row["sa"],
                "trigger_fired": bool(free_objective_near_miss(row, delta)),
                "source_representable": report["source_representable"],
                "successors_scored": report["successors_scored"],
                "eligible_count": report["eligible_count"],
                "best_eligible": best_row,
            }
        )

    lifted = [r for r in refined if r["eligible_count"] > 0]
    return {
        "cell": shard["cell"],
        "delta": delta,
        "observed_status": shard.get("observed_status"),
        "seed_smiles": seed,
        "distinct_endpoints_scored": len(rows),
        "similarity_pass": len(similarity_pass),
        "eligible_as_generated": sum(1 for r in rows if r["eligible"]),
        "best_qed_among_similarity_passing": None if best is None else best["qed"],
        "best_qed_margin": None if best is None else best["qed"] - QED_MIN,
        "comparator": comparator_check(rows, delta),
        "slot_semantics": slot_census(seed),
        "panel_size": len(refined),
        "panel_trigger_fired": sum(1 for r in refined if r["trigger_fired"]),
        "panel_unrepresentable": sum(1 for r in refined if not r["source_representable"]),
        "endpoints_lifted_to_eligible": len(lifted),
        "eligible_produced": sum(r["eligible_count"] for r in refined),
        "oracle_calls": oracle_calls,
        "panel": refined,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--cell",
        action="append",
        required=True,
        metavar="CELL=DELTA",
        help="repeatable; delta comes from the cell's own contract, never a default",
    )
    parser.add_argument("--panel", type=int, default=25)
    args = parser.parse_args()

    cells = []
    for spec in args.cell:
        name, _, delta = spec.partition("=")
        if not delta:
            raise SystemExit(f"--cell needs an explicit delta: {spec!r}")
        result = audit_cell(args.shards / f"{name}.json", float(delta), args.panel)
        cells.append(result)
        print(
            f"{result['cell']:9s} delta={result['delta']} "
            f"panel={result['panel_size']:3d} "
            f"lifted={result['endpoints_lifted_to_eligible']:3d} "
            f"eligible_produced={result['eligible_produced']:4d} "
            f"comparator_gain={result['comparator']['comparator_only_gain']}",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "refinement_schema_version": SCHEMA_VERSION,
                "panel_selection": VIOLATION_RANKED_PANEL,
                "oracle_calls": sum(c["oracle_calls"] for c in cells),
                "cells": cells,
            },
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
