"""Does the region law improve the DELETE HALF of a non-prune witness?

ZERO ORACLE CALLS.

WHY THIS EXISTS
---------------
``t4_bridge_region_law_gate.py`` measures the pure-prune closure: states reached
by region excision alone.  Three of the five exhausted delta=0.6 cells have
census witnesses in that closure.  Two do not -- their witnesses are recorded
under the ``replace`` / ``grow`` / ``remodel`` proposal modes, which excise a
region and then GROW onto the retained anchor.  Reporting "no eligible endpoint"
for those cells would be true of the prune closure and misleading about the law,
because ``segment_replace`` routes its delete half through exactly the same
``_delete_pendant_fragment`` draw.

WHAT IS MEASURED
----------------
For each census witness ``W`` and each region ``R`` in the source's uncapped
support, realize the child ``C = source - R`` and ask whether ``C`` is a
substructure of ``W``.  If it is, ``R`` is a VIABLE DELETE HALF for ``W``: the
remainder of the program is growth onto the retained anchor.  The reported
number is the best draw rank such an ``R`` gets under each arm.

A substructure match is a NECESSARY condition on the delete half, not a proof
that the grow half is reachable; the grow vocabulary is a separate question this
script does not touch.  Read a rank here as "the region the witness needs is /
is not buried", never as "the witness is reachable".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rdkit import Chem, RDLogger

from compose_v4.chem.molecular_graph import (
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.bridge_region_law import (
    BridgeRegionLaw,
    FreeFeasibilityGate,
    RegionRealizationError,
    excise_region,
    free_gate_margin_law,
)
from compose_v4.control.dynamic_program_synthesis import MAX_SEGMENT_LENGTH
from compose_v4.data.charge_policy import audit_charge_policy_transition

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "t4_region_prefix_coverage_v1"
PROPOSAL_SLOTS = 48
WITNESS_CENSUS = Path(
    "/Users/rmaganti/compose_t4_rescue_prep/diagnostics/t4_rescue_witness_census_v1.json"
)


def _rank_regions(source, law):
    """Regions in descending draw probability, with that probability attached."""

    regions = law.regions(source)
    if not regions:
        return []
    weights = law.weights(source, regions)
    total = float(weights.sum())
    ordered = sorted(
        zip(regions, (float(w) / total for w in weights)),
        key=lambda row: -row[1],
    )
    return [
        {"region": region, "probability": probability, "rank": position}
        for position, (region, probability) in enumerate(ordered, start=1)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="diagnostics/t4_region_prefix_coverage_v1.json")
    args = parser.parse_args()

    census = json.loads(WITNESS_CENSUS.read_text())
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "DIAGNOSTIC_EVIDENCE_ONLY_ZERO_ORACLE_CALLS",
        "oracle_calls": 0,
        "measures": (
            "best single-draw rank of a region whose child is a substructure of the "
            "witness; a necessary condition on the delete half only"
        ),
        "witness_source": str(WITNESS_CENSUS),
        "cells": {},
    }

    for entry in census["cells"].values():
        if float(entry["delta"]) != 0.6:
            continue
        name = entry["cell"]
        smiles = entry["source_smiles"]
        source = pad_molecular_graph(smiles_to_molecular_graph(smiles), PROPOSAL_SLOTS)
        gate = FreeFeasibilityGate(delta=float(entry["delta"]))
        arms = {
            "v1": BridgeRegionLaw(maximum=MAX_SEGMENT_LENGTH, margin=None),
            "cap_only": BridgeRegionLaw(maximum=None, margin=None),
            "conditioned_only": free_gate_margin_law(
                gate, smiles, maximum=MAX_SEGMENT_LENGTH
            ),
            "repair": free_gate_margin_law(gate, smiles, maximum=None),
        }
        ranked = {arm: _rank_regions(source, law) for arm, law in arms.items()}

        # Realize each region once; reuse across arms and witnesses.
        realized: dict[tuple, dict] = {}
        for rows in ranked.values():
            for row in rows:
                key = (row["region"].fragment, row["region"].anchor)
                if key in realized:
                    continue
                try:
                    child = excise_region(source, row["region"])
                except RegionRealizationError:
                    realized[key] = {"child": None, "executable": False}
                    continue
                child_smiles = molecular_graph_to_smiles(child)
                realized[key] = {
                    "child": Chem.MolFromSmiles(child_smiles) if child_smiles else None,
                    "smiles": child_smiles,
                    "executable": audit_charge_policy_transition(
                        source, child
                    ).preserved,
                }

        record = {
            "cell": name,
            "source_smiles": smiles,
            "source_net_formal_charge": int(source.formal_charges.sum()),
            "witnesses_listed": len(entry["witnesses"]),
            "pool_total": entry["clean_eligible_total_in_pool"],
            "listing_complete": entry["witness_listing_is_complete"],
            "witnesses": [],
        }
        for witness in entry["witnesses"]:
            mol = Chem.MolFromSmiles(witness["witness_smiles"])
            best = {}
            for arm, rows in ranked.items():
                hit = None
                for row in rows:
                    key = (row["region"].fragment, row["region"].anchor)
                    state = realized[key]
                    if not state["executable"] or state["child"] is None:
                        continue
                    if mol.HasSubstructMatch(state["child"]):
                        hit = {
                            "rank": row["rank"],
                            "probability": round(row["probability"], 8),
                            "region_size": row["region"].size,
                            "retained_child": state["smiles"],
                        }
                        break
                best[arm] = hit
            record["witnesses"].append(
                {
                    "witness_smiles": witness["witness_smiles"],
                    "proposal_mode": witness["proposal_mode"],
                    "heavy_delta": witness["heavy_delta"],
                    "net_charge_delta": witness["net_charge_delta"],
                    "best_delete_half": best,
                }
            )
        covered = {
            arm: sum(1 for row in record["witnesses"] if row["best_delete_half"][arm])
            for arm in arms
        }
        record["witnesses_with_a_viable_delete_half"] = covered
        record["best_rank"] = {
            arm: min(
                (
                    row["best_delete_half"][arm]["rank"]
                    for row in record["witnesses"]
                    if row["best_delete_half"][arm]
                ),
                default=None,
            )
            for arm in arms
        }
        payload["cells"][name] = record
        print(
            f"{name:9} covered={covered}  best_rank={record['best_rank']}",
            flush=True,
        )

    destination = ROOT / args.out
    destination.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
