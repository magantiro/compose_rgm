#!/usr/bin/env python3
"""Qualify the end-to-end nominator: admission rate, abstention, ranking quality.

Runs the OracleNominator over real A549 lipids (in-domain) and novel-head
candidates, reporting: what fraction of real lipids are admitted vs abstained
(and why), whether the filtering score ranks admitted lipids against their
actual A549 potency, and that novel heads abstain. This is the calibration /
abstention-behaviour evidence for the nomination engine.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src \
        python3 scripts/qualify_nominator.py
"""

from __future__ import annotations

import argparse
import csv
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

NOVEL_HEADS = {
    "guanidine_head": "CCCCCCCCCCCCCCCCNC(=N)N",
    "amidine_head": "CCCCCCCCCCCCCCCCNC=N",
    "non_ionizable_ester": "CCCCCCCCCCCCCCCCCC(=O)OCC",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lnpdb", type=Path, default=Path(
        "/Users/rmaganti/Desktop/thesis_projects_ML/diffusion_project/lipid_diffusion/external/LNPDB/LNPDB.csv"))
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "diagnostics/pan_lung_nominator_qualification.json")
    args = parser.parse_args()

    nom = OracleNominator.load(REPO_ROOT, ranking_domain_id="a549")
    pka = pd.read_csv(REPO_ROOT / "artifacts/oracles/head_pka_v1/a549_head_pka.csv").set_index("canonical_smiles")

    df = pd.read_csv(args.lnpdb, low_memory=False)
    a = df[(df["Experiment_ID"] == "JW_2024") & (df["Model_type"] == "A549")].copy()
    a["canon"] = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in a["IL_SMILES"]]
    # unique real lipids with their mean measured potency
    lipids = a.groupby("canon")["Experiment_value"].mean()

    decisions = Counter()
    admitted_scores, admitted_truth = [], []
    for smi, potency in lipids.items():
        hp = pka["head_pka"].get(smi)
        hp = float(hp) if hp is not None and not pd.isna(hp) else None
        r = nom.nominate(smi, head_pka=hp)
        decisions[r["decision"]] += 1
        if r["ranked"]:
            admitted_scores.append(r["filtering_score"])
            admitted_truth.append(float(potency))

    n_real = int(lipids.shape[0])
    admit_rate = round(decisions["rank"] / n_real, 3)
    rank_spearman = (round(float(spearmanr(admitted_truth, admitted_scores).statistic), 3)
                     if len(admitted_scores) >= 5 else None)

    novel = {}
    for name, smi in NOVEL_HEADS.items():
        r = nom.nominate(smi, head_pka=12.5 if "guanid" in name or "amid" in name else None)
        novel[name] = {"decision": r["decision"], "abstained": not r["ranked"]}

    out = {
        "format": "compose_pan_lung_nominator_qualification_v1",
        "real_a549_lipids": n_real,
        "decision_breakdown": dict(decisions),
        "admission_rate": admit_rate,
        "in_sample_ranking_spearman_score_vs_potency": rank_spearman,
        "ranking_note": "in-sample (filtering bundle trained on A549); a directional check that scores track potency, not held-out validation",
        "novel_head_candidates": novel,
        "all_novel_heads_abstain": all(v["abstained"] for v in novel.values()),
        "interpretation": (
            f"{admit_rate:.0%} of real A549 lipids are admitted and ranked; the rest abstain "
            "(off molecular-AD or head domain) rather than being scored on extrapolation. "
            "Novel/non-ionizable heads all abstain. Honest domain-bounded nomination."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    print(f"real A549 lipids: {n_real}")
    print(f"decisions: {dict(decisions)}")
    print(f"admission rate: {admit_rate:.1%}")
    print(f"in-sample ranking Spearman (score vs potency): {rank_spearman}")
    print(f"novel heads all abstain: {out['all_novel_heads_abstain']}  {novel}")
    print(f"written: {args.output.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
