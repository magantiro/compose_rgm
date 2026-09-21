"""Are the discarded attachment placements DUPLICATES, or reachable-but-ineligible?

WHY THIS ONE NUMBER MATTERS
---------------------------
`attachment_bindings` enumerates up to 128 exact bindings per subgoal and the
production gating loop historically used `assignments[0]`.  Measured, 24.8% of
real fa7_0 subgoals carry more than one binding, so placements ARE discarded --
yet widening the choice to four bindings produced byte-identical eligible counts
on three of three cells.  Two very different things explain that, and they imply
opposite scoping statements:

  DUPLICATES ....... the alternative bindings rebuild the SAME molecule (or fail
                     to instantiate), so `assignments[0]` was never a real
                     choice. Placement is a DEAD axis here: there is nothing to
                     exploit and the 24.8% is an artifact of how bindings are
                     counted, not of what they produce.

  INELIGIBLE ....... the alternative bindings build genuinely DIFFERENT molecules
                     that the gate then refuses. Placement is a REAL but
                     BADLY-EXPLOITED axis: the support is there and the current
                     law puts it in the wrong place.

HOW IT IS MEASURED
------------------
The production `expand` is run twice, identically seeded, differing only in
`max_bindings_per_subgoal`. Its `Fiber` is WRAPPED so every endpoint offered to
the gate is recorded -- `expand` itself returns only survivors, so the interior
is invisible from its return value. The discriminator is then simply:

    endpoints offered at k>1  MINUS  endpoints offered at k=1

If that difference is empty, the extra placements are duplicates or unbuildable.
If it is non-empty while eligible stays flat, they are distinct and refused --
and the report says which gate refused them.

The wrapped gate delegates every decision to the real one and its decomposition
is cross-checked against that verdict on every call, so `gate_disagreements`
must be 0 or the attribution here is void. ZERO ORACLE CALLS.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from compose_v4.control.completion_law_contract import (
    CONTRACT_LANE,
    FREE_GATE_MARGIN_V1,
    resolve_completion_law,
)
from compose_v4.control.region_law_contract import resolve_region_law
from compose_v4.experiments.t4_fiber_campaign import Fiber, expand
from compose_v4.experiments.t4_matched_pilot import unseal

from t4_fa7_0_reachability_probe import _FunnelFiber

SCHEMA_VERSION = "t4_fa7_0_placement_discriminator_v1"


def _first_refusal(row: dict) -> str:
    """Which gate refused this endpoint, in the production order."""

    if not row.get("parsed"):
        return "unparseable_or_over_heavy_ceiling"
    if not row["pass_similarity"]:
        return "similarity"
    if not row["pass_qed"]:
        return "qed"
    if not row["pass_sa"]:
        return "sa"
    if not row["pass_structural"]:
        return "structural"
    return "eligible"


def _arm(payload, cell, *, draws, max_bindings, max_combos, seed_offset) -> dict:
    shaped = json.loads(json.dumps(payload))
    shaped["proposal"][CONTRACT_LANE]["completion_law"] = FREE_GATE_MARGIN_V1
    region_law = resolve_region_law(
        shaped, delta=shaped["delta"], reference_smiles=cell["smiles"]
    )
    completion_law = resolve_completion_law(
        shaped, delta=shaped["delta"], reference_smiles=cell["smiles"]
    )
    probe = _FunnelFiber(
        Fiber(cell["smiles"], payload["delta"], support=payload["support"])
    )
    records = expand(
        cell["smiles"],
        0.0,
        probe,
        np.random.default_rng(int(cell["controller_seed"]) + seed_offset),
        draws=draws,
        multi_region=True,
        horizon=shaped["proposal"][CONTRACT_LANE]["horizon"],
        proposal_lane=CONTRACT_LANE,
        region_law=region_law,
        completion_law=completion_law,
        max_bindings_per_subgoal=max_bindings,
        max_binding_combinations=max_combos,
    )
    return {
        "max_bindings_per_subgoal": max_bindings,
        "max_binding_combinations": max_combos,
        "gate_calls": probe.calls,
        "gate_disagreements": len(probe.disagreements),
        "distinct_endpoints_offered": len(probe.rows),
        "eligible": len(records),
        "rows": probe.rows,
    }


def run(contract: Path, *, cell_name: str, draws: int, wide: int, combos: int, seed_offset: int) -> dict:
    payload = unseal(contract)
    cell = next(c for c in payload["cells"] if c["cell"] == cell_name)
    narrow = _arm(payload, cell, draws=draws, max_bindings=1, max_combos=4, seed_offset=seed_offset)
    broad = _arm(payload, cell, draws=draws, max_bindings=wide, max_combos=combos, seed_offset=seed_offset)

    added = set(broad["rows"]) - set(narrow["rows"])
    lost = set(narrow["rows"]) - set(broad["rows"])
    refusals = Counter(_first_refusal(broad["rows"][key]) for key in added)
    added_eligible = [k for k in added if broad["rows"][k].get("eligible")]

    if not added:
        verdict = "DUPLICATES_OR_UNBUILDABLE"
        reading = (
            "The wider binding choice offered the gate NO molecule the narrow "
            "choice did not already offer. assignments[0] was not a real choice "
            "on these states: placement is a dead axis at this budget."
        )
    elif not added_eligible:
        verdict = "DISTINCT_BUT_INELIGIBLE"
        reading = (
            "The wider binding choice built genuinely different molecules and the "
            "gate refused every one. Placement is a real axis that the current "
            "law exploits badly, and the binding gate that refuses them is named "
            "in added_first_refusal."
        )
    else:
        verdict = "DISTINCT_AND_SOME_ELIGIBLE"
        reading = (
            "The wider binding choice produced eligible endpoints the narrow one "
            "missed. This CONTRADICTS the earlier flat eligible counts and must be "
            "reconciled before anything is concluded."
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "contract": str(contract),
        "cell": cell_name,
        "delta": payload["delta"],
        "support": payload["support"],
        "draws": draws,
        "verdict": verdict,
        "reading": reading,
        "narrow": {k: v for k, v in narrow.items() if k != "rows"},
        "broad": {k: v for k, v in broad.items() if k != "rows"},
        "endpoints_added_by_wider_binding": len(added),
        "endpoints_lost_by_wider_binding": len(lost),
        "added_that_are_eligible": len(added_eligible),
        "added_first_refusal": dict(refusals.most_common()),
        "added_examples": [
            {
                "smiles": key,
                "similarity": round(broad["rows"][key].get("similarity", -1), 6),
                "qed": round(broad["rows"][key].get("qed", -1), 6),
                "sa": round(broad["rows"][key].get("sa", -1), 4),
                "first_refusal": _first_refusal(broad["rows"][key]),
            }
            for key in list(added)[:10]
        ],
        "oracle_calls": 0,
        "docking_calls": 0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contract", required=True, type=Path)
    p.add_argument("--cell", default="fa7_0")
    p.add_argument("--draws", type=int, default=240)
    p.add_argument("--wide", type=int, default=4)
    p.add_argument("--combinations", type=int, default=8)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    r = run(a.contract, cell_name=a.cell, draws=a.draws, wide=a.wide, combos=a.combinations, seed_offset=a.seed_offset)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(r, indent=1, sort_keys=True))
    print(f"{r['cell']}  VERDICT {r['verdict']}")
    print(
        f"  narrow(k=1): offered {r['narrow']['distinct_endpoints_offered']}  "
        f"eligible {r['narrow']['eligible']}  disagree {r['narrow']['gate_disagreements']}"
    )
    print(
        f"  broad(k={a.wide}):  offered {r['broad']['distinct_endpoints_offered']}  "
        f"eligible {r['broad']['eligible']}  disagree {r['broad']['gate_disagreements']}"
    )
    print(
        f"  ADDED {r['endpoints_added_by_wider_binding']}  LOST {r['endpoints_lost_by_wider_binding']}  "
        f"added_eligible {r['added_that_are_eligible']}"
    )
    print(f"  added first refusal: {r['added_first_refusal']}")
    for e in r["added_examples"][:6]:
        print(f"    +{e['first_refusal']:>12} sim={e['similarity']:.4f} qed={e['qed']:.4f} {e['smiles']}")


if __name__ == "__main__":
    main()
