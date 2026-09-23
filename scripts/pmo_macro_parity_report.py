"""Does the PMO region option expose the same COMPOSE repertoire the rich machinery has?

The region replacement is meant to constrain WHERE COMPOSE acts, not WHAT it may do there.  If the
rebuild stage silently offers a reduced vocabulary, then a failed macro-boundary test is ambiguous:
we cannot tell a bad search from a capability that was never exposed.  This report closes that
ambiguity BEFORE the Celecoxib diagnostic runs.

It measures applicability by EXECUTION against the production compilers -- never by reading a
source list -- on the fixed Celecoxib macro-boundary states, both on the raw state and on the
CONTRACTED state left after a region excision, which is where a rebuild actually happens.

Zero oracle calls.  No target, no witness, no answer-known constant enters any decision here.
"""
from __future__ import annotations

import json
import sys
from collections import Counter

import numpy as np

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.current_state_edits import ENUMERATORS
from compose_v4.control.dynamic_program_synthesis import (
    GENERIC_MODULES,
    _delete_pendant_fragment,
    compile_generic_module,
)
from compose_v4.control.ring_program import default_ring_options
from compose_v4.control.scale_balanced_region_law import ScaleBalancedRegionLaw

SEGMENTS = "diagnostics/pmo_macro_horizon_v1/macro_segments_v1.json"
OUT = "diagnostics/pmo_macro_parity_v1/macro_parity_report_v1.json"
TRIALS = 24


def _applicable(graph, family, trials=TRIALS, seed=0):
    """Can this family execute on this exact state?  Measured, with the real compiler."""
    rng = np.random.default_rng(seed)
    reasons = Counter()
    for _ in range(trials):
        try:
            compile_generic_module(graph, rng, family)
        except (ValueError, RuntimeError) as error:
            reasons[str(error)[:70]] += 1
            continue
        return True, dict(reasons)
    return False, dict(reasons)


def main():
    with open(SEGMENTS) as handle:
        segments = json.load(handle)
    states, seen = [], set()
    for row in segments:
        if row["source_smiles"] not in seen:
            seen.add(row["source_smiles"])
            states.append(row)

    report = {
        "schema_version": "pmo_macro_parity_v1",
        "evidence_role": "zero_oracle_capability_parity_measurement",
        "new_charged_oracle_calls": 0,
        "question": (
            "Does the PMO region-replacement rebuild stage reach the same COMPOSE action "
            "repertoire the shallow synthesizer already has, at the selected region?"
        ),
        "registered_vocabulary": {
            "generic_modules": list(GENERIC_MODULES),
            "generic_module_count": len(GENERIC_MODULES),
            "local_edit_families": sorted(ENUMERATORS),
            "ring_options": list(default_ring_options()),
            "ring_option_count": len(default_ring_options()),
        },
        "states": [],
    }

    law = ScaleBalancedRegionLaw()
    for index, row in enumerate(states):
        graph = pad_molecular_graph(smiles_to_molecular_graph(row["source_smiles"]), 48)
        entry = {
            "route": row["route"],
            "seg": row["seg"],
            "source_smiles": row["source_smiles"],
            "heavy_atoms": int(graph.n_real_atoms),
            "applicable_on_state": {},
            "applicable_on_contracted": {},
            "refusals_on_state": {},
        }
        for family in GENERIC_MODULES:
            ok, reasons = _applicable(graph, family, seed=20260923 + index)
            entry["applicable_on_state"][family] = ok
            if not ok:
                entry["refusals_on_state"][family] = reasons

        # The rebuild actually happens on the CONTRACTED graph left after an excision, so
        # measure applicability there too -- that is the state the option must serve.
        rng = np.random.default_rng(20260923 + index)
        try:
            _acts, contracted, _anchor, path = _delete_pendant_fragment(graph, rng, law=law)
        except (ValueError, RuntimeError) as error:
            entry["contraction_failed"] = str(error)[:120]
        else:
            entry["excised_atoms"] = len(path)
            for family in GENERIC_MODULES:
                ok, _reasons = _applicable(contracted, family, seed=20260923 + index)
                entry["applicable_on_contracted"][family] = ok
        report["states"].append(entry)
        applicable = sum(entry["applicable_on_state"].values())
        contracted_ok = sum(entry["applicable_on_contracted"].values())
        print(
            f"{row['route'][:16]:16s} s{row['seg']} heavy={entry['heavy_atoms']:3d} "
            f"| applicable on state {applicable:2d}/{len(GENERIC_MODULES)} "
            f"| on contracted {contracted_ok:2d}/{len(GENERIC_MODULES)}",
            flush=True,
        )

    # Aggregate: a family applicable ANYWHERE is a capability PMO's rebuild stage must not omit.
    anywhere = {
        family: any(s["applicable_on_contracted"].get(family) for s in report["states"])
        for family in GENERIC_MODULES
    }
    report["applicable_somewhere_on_a_contracted_state"] = anywhere
    report["families_applicable_but_not_exposed_by_v1_option"] = sorted(
        family
        for family, ok in anywhere.items()
        if ok and family not in {"segment_grow", "append_ring", "fuse_ring"}
    )
    import pathlib

    pathlib.Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=1)
    print("\nfamilies applicable on some contracted state but NOT exposed by the v1 rebuild stage:")
    for family in report["families_applicable_but_not_exposed_by_v1_option"]:
        print("   ", family)
    print("WROTE", OUT)


if __name__ == "__main__":
    sys.exit(main())
