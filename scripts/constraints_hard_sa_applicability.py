"""SA-constraint census — the MODEL-FREE half.

Measures the externally defined predicate `SA(y) <= tau` on held-in SOURCES using
only RDKit. Needs **no** `R_theta` checkpoint, **no** successor kernel, **no**
oracle and **no** Gate-0 chain, so it runs locally on CPU while the fiber half
stays blocked (`SA_CENSUS_PROTOCOL.md` section 8).

Thresholds are CDD's published values, fixed in `SA_CENSUS_PROTOCOL.md` and
committed BEFORE this script was run. They are not re-derived here and must not
be widened.

What this CAN establish
-----------------------
Source applicability: what fraction of held-in sources already satisfy
`SA(x_0) <= tau`, and therefore how many are eligible at all. Also the SA
distribution, which says how much slack a source has before it violates.

What it CANNOT establish
------------------------
Anything about the successor fiber -- violation prevalence under the process,
retained support, mask-empty rate, headroom, or recoverability. Those need the
kernel. **Do not read this as the census verdict.**

Usage
-----
    python3 scripts/constraints_hard_sa_applicability.py \
        --cohort diagnostics/retarget_calibration_cohort.json \
        --pool   diagnostics/editing_v2_matched_validation_reserve_ids.json.gz \
        --out    diagnostics/constraints_hard_sa_applicability.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics as st
from collections import Counter
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Contrib.SA_Score import sascorer

RDLogger.DisableLog("rdApp.*")

ARTIFACT_STATUS = "SMOKE_HELD_IN"

# CDD's published thresholds, section 5.2, verbatim. Primary is 3.0 -- their
# Figure 4 caption defines the headline novelty metric at "no violation
# (tau <= 3.0)". Fixed in SA_CENSUS_PROTOCOL.md and committed before this ran.
TAUS: tuple[float, ...] = (3.0, 3.5, 4.0, 4.5)
PRIMARY_TAU = 3.0

# Reused verbatim from Lane 2. Not minted here.
HEAVY_ATOM_BAND = (18, 38)

PROVENANCE = {
    "thresholds": "CDD (NeurIPS 2025, arXiv:2503.09790) section 5.2, verbatim",
    "primary_tau": "CDD Figure 4 caption defines novelty at 'no violation (tau <= 3.0)'",
    "HEAVY_ATOM_BAND": "Lane 2 pathwise_constraints.HEAVY_ATOM_BAND, reused verbatim",
    "scorer": "rdkit.Contrib.SA_Score.sascorer (Ertl & Schuffenhauer), as already used by molecular_quality.py",
}


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    quantile = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]  # noqa: E731
    return {
        "n": len(ordered),
        "mean": round(st.mean(ordered), 4),
        "median": round(st.median(ordered), 4),
        "p10": round(quantile(0.10), 4),
        "p90": round(quantile(0.90), 4),
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
    }


def census(smiles_list: list[str], *, label: str) -> dict[str, Any]:
    scores: list[float] = []
    heavy: list[int] = []
    unparseable = 0
    per_tau: dict[str, dict[str, Any]] = {}
    # Slack = tau - SA(x_0): how far the source is from violating. A source with
    # tiny slack is one edit from infeasibility; a source with large slack may
    # never be pushed over. Both matter for whether the constraint can bite.
    slack: dict[str, list[float]] = {str(t): [] for t in TAUS}

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            unparseable += 1
            continue
        score = float(sascorer.calculateScore(mol))
        scores.append(score)
        heavy.append(mol.GetNumHeavyAtoms())
        for tau in TAUS:
            slack[str(tau)].append(round(tau - score, 4))

    in_band = [
        h for h in heavy if HEAVY_ATOM_BAND[0] <= h <= HEAVY_ATOM_BAND[1]
    ]

    for tau in TAUS:
        satisfying = [s for s in scores if s <= tau]
        # Eligibility = satisfies the predicate AND sits in the frozen size band.
        eligible = sum(
            1
            for s, h in zip(scores, heavy, strict=True)
            if s <= tau and HEAVY_ATOM_BAND[0] <= h <= HEAVY_ATOM_BAND[1]
        )
        per_tau[str(tau)] = {
            "sources_satisfying_predicate": len(satisfying),
            "fraction_satisfying_predicate": (
                round(len(satisfying) / len(scores), 4) if scores else 0.0
            ),
            "eligible_with_size_band": eligible,
            "eligible_fraction": (
                round(eligible / len(scores), 4) if scores else 0.0
            ),
            "slack_tau_minus_sa": summarize(slack[str(tau)]),
        }

    return {
        "label": label,
        "sources": len(smiles_list),
        "parsed": len(scores),
        "unparseable": unparseable,
        "sa_distribution": summarize(scores),
        "heavy_atoms_in_frozen_band": len(in_band),
        "per_tau": per_tau,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--pool", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cohort_payload = json.loads(args.cohort.read_text())
    cohort_smiles = [str(r["source"]) for r in cohort_payload["sources"]]
    blocks = [census(cohort_smiles, label="held_in_developability_cohort")]

    if args.pool:
        import gzip

        with gzip.open(args.pool, "rt") as handle:
            pool = json.load(handle)
        # Structurally refuse to look at the held-out reserve.
        pool.pop("reserve_source_keys", None)
        blocks.append(census(pool["training_source_keys"], label="held_in_pool"))

    result = {
        "schema": "compose.constraints_hard.sa_applicability",
        "schema_version": 1,
        "status": ARTIFACT_STATUS,
        "held_out_opened": False,
        "scope": (
            "MODEL-FREE ONLY. Source applicability for SA(x_0) <= tau. "
            "Establishes NOTHING about the successor fiber: not violation "
            "prevalence, not retained support, not mask-empty rate, not headroom."
        ),
        "predicate": "SA(y) <= tau",
        "taus": list(TAUS),
        "primary_tau": PRIMARY_TAU,
        "tau_rule_fixed_before_measurement": True,
        "tau_rule_document": "docs/workstreams/constraints-hard/SA_CENSUS_PROTOCOL.md",
        "provenance": PROVENANCE,
        "rdkit_version": Chem.rdBase.rdkitVersion,
        "rdkit_production_pin": "2024.03.5",
        "rdkit_pin_matches_production": Chem.rdBase.rdkitVersion == "2024.03.5",
        "rdkit_pin_warning": (
            "CDD pins no RDKit version; SA shifts with the fragment-contribution "
            "tables, so a tau boundary is not version-portable. Any claim-bearing "
            "run must use the pinned shared evaluator."
        ),
        "cohort_path": str(args.cohort),
        "cohort_file_sha256": hashlib.sha256(args.cohort.read_bytes()).hexdigest(),
        "pool_path": str(args.pool) if args.pool else None,
        "blocks": blocks,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")

    for block in blocks:
        print(f"\n=== {block['label']} (n={block['parsed']}) ===")
        print("  SA:", json.dumps(block["sa_distribution"]))
        for tau in TAUS:
            row = block["per_tau"][str(tau)]
            mark = "  <-- PRIMARY" if tau == PRIMARY_TAU else ""
            print(
                f"  tau={tau}: satisfying={row['fraction_satisfying_predicate']:.4f}"
                f"  eligible={row['eligible_fraction']:.4f}"
                f"  median_slack={row['slack_tau_minus_sa'].get('median')}{mark}"
            )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
