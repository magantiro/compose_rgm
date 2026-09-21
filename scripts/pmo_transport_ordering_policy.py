"""The measured ordering policy, and the residual that sizes the protected floor.

Zero oracle calls, zero program execution.

For every declared-target pair, every ordering of every available alignment is built and
checked, admissible ones are compared on trajectory shape, and one is selected. Three
things are reported:

  1. what the POLICY picks, against what a fixed prune-first default would have picked;
  2. the RESIDUAL -- transports with no dip-free admissible ordering, which are the ones
     that genuinely need budget protected from score-based selection;
  3. the WIDTH that protection must cover, because protecting a single crossing is a
     different and stronger guarantee than protecting an arbitrary number of rounds.

Admissibility -- every intermediate valid and connected, and the final molecule IS the
target -- is a hard filter rather than a term in the score. An ordering with a better
trajectory that lands somewhere else is not a candidate. This matters concretely:
interleaving removes the dip on 9 of 10 dipping transports but preserves the endpoint on
only 6 of 10, so scoring shape without the filter would select orderings that never
arrive.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from compose_v4.control.pmo_transport_correspondence import correspondences
from compose_v4.control.pmo_transport_staging import (
    DEFAULT_MAX_PRIMITIVES,
    candidate_orderings,
    select_ordering,
)

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def best_start(bank, target, *, top):
    mol = Chem.MolFromSmiles(target)
    fingerprint = _GEN.GetFingerprint(mol)
    scored = []
    for smiles in bank:
        candidate = Chem.MolFromSmiles(smiles)
        if candidate is not None:
            scored.append(
                (DataStructs.TanimotoSimilarity(fingerprint, _GEN.GetFingerprint(candidate)),
                 smiles)
            )
    scored.sort(reverse=True)
    return [smiles for _, smiles in scored[:top]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--alignments", type=int, default=8)
    parser.add_argument("--max-primitives", type=int, default=DEFAULT_MAX_PRIMITIVES)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    audit = json.loads(
        (ROOT / "diagnostics/pmo_discovery_v1/goal_specification_audit_v1.json").read_text()
    )
    bank = json.loads((ROOT / "docs/PMO_INIT_BANK.json").read_text())
    if isinstance(bank, dict):
        bank = bank.get("smiles") or bank.get("molecules")

    rows = []
    for entry in audit["rows"]:
        declared = [d for d in entry.get("declared", []) if d["kind"] == "smiles"]
        if not declared or entry["goal_kind"] == "declared_smarts":
            continue
        target = declared[0]["value"]
        for source in best_start(bank, target, top=args.starts):
            found = correspondences(source, target, top_k=args.alignments)
            if not found:
                continue
            selection = select_ordering(found, max_primitives=args.max_primitives)
            every = candidate_orderings(found, max_primitives=args.max_primitives)
            default = next(
                (c for c in every if c.alignment == 0 and not c.interleave), None
            )
            chosen = selection.chosen
            rows.append({
                "task": entry["task"],
                "source": source,
                "alignments_available": len(found),
                "candidates": len(every),
                "admissible": sum(1 for c in every if c.admissible),
                "default_admissible": bool(default and default.admissible),
                "default_dips": None if default is None else default.dips,
                "selected": None if chosen is None else chosen.payload(),
                "selected_interleaved": None if chosen is None else chosen.interleave,
                "has_dip_free_ordering": selection.has_dip_free_ordering,
                "protected_rounds_required": selection.protected_rounds_required,
                "no_admissible_ordering": chosen is None,
            })

    total = len(rows)
    unstageable = [r for r in rows if r["no_admissible_ordering"]]
    stageable = [r for r in rows if not r["no_admissible_ordering"]]
    residual = [r for r in stageable if not r["has_dip_free_ordering"]]
    widths = [r["protected_rounds_required"] for r in residual]
    default_dipping = [r for r in stageable if r["default_dips"]]
    fixed_by_policy = [r for r in default_dipping if r["has_dip_free_ordering"]]

    print(f"transports: {total}")
    print(f"  no admissible ordering at all:      {len(unstageable)}")
    print(f"  stageable:                          {len(stageable)}")
    print(f"    prune-first default would dip:    {len(default_dipping)}")
    print(f"    POLICY finds a dip-free ordering: {len(fixed_by_policy)}/{len(default_dipping)}")
    print(f"  RESIDUAL (no dip-free ordering):    {len(residual)}/{len(stageable)}"
          f"  = {len(residual)/max(1,len(stageable)):.1%}")
    if widths:
        print(f"    protected rounds needed: median {statistics.median(widths)}, "
              f"max {max(widths)}")
    print(f"  policy selects interleaved on:      "
          f"{sum(1 for r in stageable if r['selected_interleaved'])}/{len(stageable)}")
    verdict = (
        "NO_TRANSPORT_IS_STAGEABLE"
        if not stageable
        else "POLICY_REMOVES_EVERY_DIP_NO_FLOOR_NEEDED"
        if not residual
        else "FLOOR_IS_CHEAP_INSURANCE"
        if len(residual) <= 0.25 * len(stageable)
        else "FLOOR_IS_THE_MECHANISM"
    )
    print(f"VERDICT: {verdict}")
    report = {
        "schema_version": "pmo_transport_ordering_policy_v1",
        "oracle_calls": 0,
        "programs_executed": 0,
        "transports": total,
        "no_admissible_ordering": len(unstageable),
        "stageable": len(stageable),
        "default_dipping": len(default_dipping),
        "dips_removed_by_policy": len(fixed_by_policy),
        "residual_needing_protection": len(residual),
        "protected_rounds_median": statistics.median(widths) if widths else 0,
        "protected_rounds_max": max(widths) if widths else 0,
        "residual_tasks": dict(collections.Counter(r["task"] for r in residual)),
        "verdict": verdict,
        "rows": rows,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
