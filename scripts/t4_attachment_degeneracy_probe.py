"""Address degeneracy of v1's attachment key. DIAGNOSIS ONLY, zero oracle calls.

`t4_fiber_campaign.expand` binds every subgoal with `census.assignments[0]` -- the
FIRST row `structural_subgoal.attachment_bindings` enumerates, never a sampled or
scored one. That enumeration matches source slots to subgoal roles by
`edit_program.atom_signature` (element, formal charge, implicit H, degree) plus
`edit_program.environment` (that signature plus the sorted multiset of one-hop
(bond order, neighbour signature) pairs). The address is therefore RADIUS 1.

Consequence, and the thing this module measures: a structural goal is address-free, so
whenever several atoms share a radius-1 class the goal can bind to any of them, and v1
always takes the lowest-slot member. If the region a witness needs is anchored on an
atom that is not the lowest-slot member of its own class, then synthesizing the right
structural decision is not sufficient -- the realization lands somewhere else.

Reported per cell: the class-multiplicity histogram, and for every eligible single-cut
prune the anchor's class size and whether that anchor is the class representative
`assignments[0]` would select.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from compose_v4.chem.molecular_graph import is_element, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.edit_program import environment

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "t4_attachment_degeneracy_probe_v1"
PROPOSAL_SLOTS = 48

FAILED = ("braf_0", "braf_1", "fa7_0", "fa7_2", "5ht1b_2")
CONTROLS = ("braf_2", "fa7_1", "5ht1b_0", "5ht1b_1")


def probe(smiles: str, eligible_rows: list[dict]) -> dict:
    graph = pad_molecular_graph(smiles_to_molecular_graph(smiles), PROPOSAL_SLOTS)
    real = [int(i) for i in np.flatnonzero(is_element(graph.atom_types))]
    classes: dict[str, list[int]] = defaultdict(list)
    for slot in real:
        classes[environment(graph, slot)].append(slot)

    sizes = Counter(len(members) for members in classes.values())
    degenerate = sum(1 for slot in real if len(classes[environment(graph, slot)]) > 1)

    anchors = []
    for row in eligible_rows:
        anchor = int(row["anchor"])
        members = classes[environment(graph, anchor)]
        anchors.append(
            {
                "anchor": anchor,
                "fragment_size": row["size"],
                "qed": row.get("qed"),
                "similarity": row.get("similarity"),
                "class_size": len(members),
                "is_class_representative": anchor == min(members),
                "representative_slot": min(members),
            }
        )
    return {
        "heavy_atoms": len(real),
        "distinct_radius1_classes": len(classes),
        "class_size_histogram": {str(k): v for k, v in sorted(sizes.items())},
        "atoms_in_degenerate_class": degenerate,
        "degenerate_fraction": round(degenerate / len(real), 5) if real else None,
        "largest_class_size": max((len(m) for m in classes.values()), default=0),
        "eligible_cut_anchors": anchors,
        "eligible_anchors_that_are_representative": sum(
            1 for row in anchors if row["is_class_representative"]
        ),
    }


def main() -> None:
    census = json.loads(
        (ROOT / "diagnostics/t4_bridge_cut_census_v1.json").read_text()
    )["cells"]

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "DIAGNOSTIC_EVIDENCE_ONLY_ZERO_ORACLE_CALLS",
        "oracle_calls": 0,
        "attachment_key_radius": 1,
        "binding_selection": "census.assignments[0] (lowest-slot consistent match)",
        "cells": {},
    }
    for name in sorted(set(FAILED) | set(CONTROLS)):
        cell = census[name]
        eligible = [row for row in cell["rows"] if row.get("verdict")]
        result = probe(cell["smiles"], eligible)
        result["role"] = "failed" if name in FAILED else "control"
        payload["cells"][name] = result
        print(
            f"{name:10} {result['role']:8} atoms={result['heavy_atoms']:3} "
            f"classes={result['distinct_radius1_classes']:3} "
            f"degenerate={result['degenerate_fraction']:.3f} "
            f"largest_class={result['largest_class_size']:2} "
            f"elig_anchors={len(result['eligible_cut_anchors'])} "
            f"representative={result['eligible_anchors_that_are_representative']}",
            flush=True,
        )

    destination = ROOT / "diagnostics/t4_attachment_degeneracy_probe_v1.json"
    destination.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
