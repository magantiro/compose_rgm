"""Carve and seal the held-out retargeting confirmation panel. 40 sources.

Selection order is fixed and every step is outcome-independent:

  1. start from the held-out matched reserve;
  2. exclude the CLAIM-BEARING MOLECULE REGISTRY -- every reserve molecule
     already touched by a claim-bearing evaluation;
  3. apply x_0-level applicability criteria only, identical to the ones the
     development cohort used;
  4. shuffle under a fixed seed and take the first 40;
  5. assert zero overlap against every other panel;
  6. commit the ordered canonical list and its digest.

WHY THE REGISTRY COMES BEFORE SELECTION. The sealed 67-pair panel lost two
pairs to an endpoint-overlap discovered during a pre-outcome audit, because the
dev/eval split had been pair-disjoint rather than endpoint-disjoint. Excluding
after the fact means discovering the problem after the fact. The registry is
applied as a filter here, and the zero-overlap assertions are a check on the
filter rather than the mechanism itself.

WHAT IS DELIBERATELY NOT AN ELIGIBILITY CRITERION. Nothing about what happens
during the prefixes. In particular a source is NOT required to show "good
headroom after three edits" -- that would condition the panel on the very thing
the controller produces, and would guarantee the confirmation reproduces the
development result by construction. Eligibility reads the starting molecule and
formal task applicability, nothing else.

No source may be replaced later because its prefix behaves awkwardly or its
outcome is inconvenient.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from pathlib import Path

#: Identical to the development cohort's rule. Reused verbatim, not re-derived.
MIN_HEAVY, MAX_HEAVY = 18, 38
QED_FLOOR = 0.6
LOGP_BOX = (1.0, 4.0)
CANONICAL_SLOTS = 48
SEED = 20260815
PANEL_SIZE = 40


def claim_bearing_registry(repo: Path) -> dict[str, list[str]]:
    """Every reserve molecule already used in a claim-bearing evaluation."""
    reg: dict[str, list[str]] = {}
    seal = json.loads((repo / "diagnostics/editing_v2_controller_panel_seal.json").read_text())
    reg["exact_target_sealed"] = sorted({p["source"] for p in seal["sealed"]}
                                        | {p["target"] for p in seal["sealed"]})
    reg["exact_target_development"] = sorted({p["source"] for p in seal["development"]}
                                             | {p["target"] for p in seal["development"]})
    cohort = json.loads((repo / "diagnostics/retarget_calibration_cohort.json").read_text())
    reg["retargeting_development"] = sorted(s["source"] for s in cohort["sources"])
    return reg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, QED

    RDLogger.DisableLog("rdApp.*")
    reserve = json.load(gzip.open(
        args.repo / "diagnostics/editing_v2_matched_validation_reserve_ids.json.gz", "rt"))
    pool = sorted(reserve["reserve_source_keys"])
    training = set(reserve["training_source_keys"])

    registry = claim_bearing_registry(args.repo)
    excluded = {m for group in registry.values() for m in group}
    print(f"held-out reserve pool           {len(pool):,}")
    for name, group in registry.items():
        hit = len(set(group) & set(pool))
        print(f"  registry {name:<28} {len(group):>4} molecules, {hit:>3} in the reserve")
    print(f"claim-bearing registry total    {len(excluded)} molecules")

    # STEP 2 before STEP 3: registry first, then applicability.
    candidates = [s for s in pool if s not in excluded]
    print(f"after registry exclusion        {len(candidates):,}")

    random.Random(SEED).shuffle(candidates)
    chosen, scanned, rejected = [], 0, {"unparseable": 0, "size_band": 0,
                                        "already_developable": 0}
    for smiles in candidates:
        if len(chosen) >= PANEL_SIZE:
            break
        scanned += 1
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            rejected["unparseable"] += 1
            continue
        if not (MIN_HEAVY <= mol.GetNumHeavyAtoms() <= MAX_HEAVY):
            rejected["size_band"] += 1
            continue
        try:
            qed, logp = QED.qed(mol), Crippen.MolLogP(mol)
        except Exception:  # noqa: BLE001
            rejected["unparseable"] += 1
            continue
        if qed >= QED_FLOOR and LOGP_BOX[0] <= logp <= LOGP_BOX[1]:
            rejected["already_developable"] += 1
            continue
        chosen.append({"index": len(chosen), "source": smiles,
                       "slots": CANONICAL_SLOTS,
                       "heavy_atoms": int(mol.GetNumHeavyAtoms()),
                       "qed": round(float(qed), 4), "clogp": round(float(logp), 4)})

    if len(chosen) < PANEL_SIZE:
        raise SystemExit(f"only {len(chosen)} eligible sources found")

    keys = [c["source"] for c in chosen]
    # STEP 5. Assertions check the filter; they are not the mechanism.
    overlaps = {
        "training_universe": len(set(keys) & training),
        "retargeting_development": len(set(keys) & set(registry["retargeting_development"])),
        "exact_target_sealed": len(set(keys) & set(registry["exact_target_sealed"])),
        "exact_target_development": len(set(keys) & set(registry["exact_target_development"])),
    }
    print("\nzero-overlap assertions")
    for name, n in overlaps.items():
        print(f"  {name:<30} {n}")
        if n:
            raise SystemExit(f"PANEL REJECTED: {n} overlap with {name}")
    if len(set(keys)) != len(keys):
        raise SystemExit("PANEL REJECTED: duplicate sources")

    digest = hashlib.sha256(json.dumps(keys, sort_keys=True).encode()).hexdigest()
    print(f"\nscanned {scanned} candidates -> {len(chosen)} sources")
    for reason, count in rejected.items():
        print(f"  rejected {reason}: {count}")
    print(f"panel sha256 {digest[:16]}")

    args.out.write_text(json.dumps({
        "schema": "compose.retarget.heldout_confirmation_panel",
        "status": "SEALED_BEFORE_ANY_PREFIX_WAS_GENERATED",
        "panel_size": PANEL_SIZE, "seed": SEED,
        "pool": "held-out matched validation reserve",
        "selection_order": ["registry exclusion", "x_0 applicability", "seeded shuffle"],
        "eligibility": {"heavy_atoms": [MIN_HEAVY, MAX_HEAVY],
                        "qed_floor": QED_FLOOR, "logp_box": list(LOGP_BOX),
                        "rule": "exclude sources already inside the developability region",
                        "identical_to": "the development cohort's rule, reused verbatim"},
        "not_an_eligibility_criterion": (
            "anything about prefix behaviour, including post-prefix headroom. "
            "Conditioning on that would guarantee the confirmation reproduces "
            "the development result by construction."),
        "claim_bearing_registry": {k: len(v) for k, v in registry.items()},
        "zero_overlap_verified": overlaps,
        "panel_sha256": digest,
        "scanned": scanned, "rejected": rejected,
        "no_replacement_rule": ("No source may be replaced because its prefix "
                                "behaves awkwardly or its outcome is inconvenient."),
        "sources": chosen,
    }, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
