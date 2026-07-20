#!/usr/bin/env python3
"""Operational nominator behaviour on REAL Michael-addition lipids (BEAE's class).

The leave-library-out / positive-control experiments answer "can the oracle rank
Michael-addition lipids" (held-out: weak; with data: 0.633). This answers the
complementary operational question: when the *deployed* OracleNominator sees real
Michael-addition lipids, how many does it ADMIT and rank vs ABSTAIN (novel head ->
active-learning; off-domain)? Uses the LiON RM_Michael_addition_branched library
(BEAE's exact chemistry class), with reductive-amination as a reference library.

Ranking here is in-sample (the A549 filtering bundle was trained on this screen) --
a wiring check, not held-out validation; the held-out picture is the leave-library
+ positive-control diagnostic. This script's signal is the admit/abstain breakdown.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src \
        python3 scripts/qualify_nominator_michael.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy.stats import spearmanr

from compose_v4.oracles.nominate import OracleNominator

RDLogger.DisableLog("rdApp.*")
REPO_ROOT = Path(__file__).resolve().parents[1]
LIBRARIES = {"michael_addition": "RM_Michael_addition_branched",
             "reductive_amination": "IR_Reductive_amination"}


def _library_report(nom: OracleNominator, lipids: "pd.Series") -> dict:
    decisions: Counter = Counter()
    admitted_scores, admitted_truth = [], []
    for smi, potency in lipids.items():
        r = nom.nominate(smi, head_pka=None)
        decisions[r["decision"]] += 1
        if r["ranked"]:
            admitted_scores.append(r["filtering_score"])
            admitted_truth.append(float(potency))
    n = int(lipids.shape[0])
    in_sample = (round(float(spearmanr(admitted_truth, admitted_scores).statistic), 3)
                 if len(admitted_scores) >= 5 else None)
    return {
        "unique_lipids": n,
        "decision_breakdown": dict(decisions),
        "admission_rate": round(decisions["rank"] / n, 3),
        "abstain_novel_head_rate": round(decisions.get("abstain_novel_head", 0) / n, 3),
        "abstain_off_domain_rate": round(decisions.get("abstain_off_domain", 0) / n, 3),
        "in_sample_ranking_spearman": in_sample,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lion", type=Path, default=Path(
        "/Users/rmaganti/Desktop/thesis_projects_ML/diffusion_project/lipid_diffusion/external/LNP_ML/data/all_data.csv"))
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "diagnostics/nominator_michael_operational.json")
    args = parser.parse_args()

    nom = OracleNominator.load(REPO_ROOT, ranking_domain_id="a549")
    df = pd.read_csv(args.lion, low_memory=False)
    a = df[df["Model_type"].astype(str).eq("A549")].copy()
    a = a[a["smiles"].notna() & a["unnormalized_delivery"].notna() & a["Library_ID"].notna()]
    a["canon"] = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(str(s)) else None
                  for s in a["smiles"]]
    a = a[a["canon"].notna()].copy()

    per_library = {}
    for label, lib_id in LIBRARIES.items():
        sub = a[a["Library_ID"].eq(lib_id)]
        if sub.empty:
            continue
        lipids = sub.groupby("canon")["unnormalized_delivery"].mean()
        per_library[label] = {"library_id": lib_id, **_library_report(nom, lipids)}

    michael = per_library.get("michael_addition", {})
    out = {
        "format": "compose_nominator_michael_operational_v1",
        "question": "When the deployed nominator sees real Michael-addition lipids, how many are admitted & ranked vs abstain?",
        "per_library": per_library,
        "interpretation": (
            f"{michael.get('admission_rate', 0):.0%} of real Michael-addition lipids (BEAE's "
            f"chemistry class) are admitted & ranked by the deployed AD gate; "
            f"{michael.get('abstain_novel_head_rate', 0):.0%} abstain on a novel head -> "
            "these define the size of the active-learning calibration round. Operational "
            "bridge from the transfer diagnostics to the deployed nomination engine. Ranking "
            "shown is in-sample wiring; held-out ranking = leave-library + positive-control."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    for label, r in per_library.items():
        print(f"[{label}] n={r['unique_lipids']}  admit={r['admission_rate']:.1%}  "
              f"abstain_novel_head={r['abstain_novel_head_rate']:.1%}  "
              f"abstain_off_domain={r['abstain_off_domain_rate']:.1%}  "
              f"in-sample rank={r['in_sample_ranking_spearman']}")
        print(f"    decisions: {r['decision_breakdown']}")
    print(f"written: {args.output.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
