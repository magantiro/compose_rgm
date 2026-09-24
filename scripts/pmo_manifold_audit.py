"""Zero-oracle chemistry/manifold audit of the top-N molecules a PMO run actually found.

WHY.  A weak task can be weak for three very different reasons, and they call for three
different interventions.  The benchmark score alone cannot tell them apart:

  off_manifold_exploitation : the run left drug-like space and is gaming a predictor.
                              MEASURED signature: top molecules at very low QED and high
                              SA, score ANTI-correlated with QED, exotic/hypervalent atoms.
                              Intervention: a construction prior, not more search.
  on_manifold_stagnation    : molecules are plausible but the frontier stopped moving.
                              Intervention: macros / basin transport.
  late_discovery            : the frontier is still climbing at the end of the budget.
                              Intervention: memory, i.e. make the same discoveries earlier.

NOTHING HERE TOUCHES THE OBJECTIVE.  QED, SA and alerts are computed for DIAGNOSIS only and
are never fed back into selection; doing so would change the task.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

RDLogger.DisableLog("rdApp.*")
#: Elements a drug-like molecule essentially never contains, used only as a manifold flag.
EXOTIC = {"B", "Si", "Se", "Te", "As"}
#: Substructures that mark adversarial chemistry rather than a real lead.
BAD_SMARTS = {
    "peroxide": "[OX2][OX2]",
    "hydrazine_chain": "[NX3][NX3][NX3]",
    "hypervalent_S": "[#16;v4,v6;!$([#16](=[OX1])(=[OX1]))]",
    "hypervalent_I": "[#53;v3,v5,v7]",
    "hypervalent_P": "[#15;v5;!$([#15](=[OX1]))]",
}
_PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in BAD_SMARTS.items()}


def profile(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    flags = [k for k, p in _PATTERNS.items() if p is not None and mol.HasSubstructMatch(p)]
    exotic = sorted({a.GetSymbol() for a in mol.GetAtoms()} & EXOTIC)
    try:
        qed = QED.qed(mol)
        alerts = QED.properties(mol).ALERTS
    except (ValueError, RuntimeError):
        return None
    return {"qed": qed, "sa": sascorer.calculateScore(mol), "alerts": int(alerts),
            "heavy": mol.GetNumHeavyAtoms(), "flags": flags, "exotic": exotic}


def classify(rows, trajectory):
    """Three outcomes, decided on measured quantities rather than on the score alone."""
    qed = np.array([r["qed"] for r in rows])
    sa = np.array([r["sa"] for r in rows])
    score = np.array([r["score"] for r in rows])
    bad = np.mean([bool(r["flags"] or r["exotic"]) for r in rows])
    corr = float(np.corrcoef(score, qed)[0, 1]) if len(rows) > 2 and qed.std() > 0 else 0.0
    # Still climbing? compare the last quarter of the charged trajectory to the previous one.
    climbing = None
    if len(trajectory) >= 8:
        q = len(trajectory) // 4
        climbing = float(np.mean(trajectory[-q:]) - np.mean(trajectory[-2 * q:-q]))
    off = (np.median(qed) < 0.25 and np.median(sa) > 4.5) or bad > 0.5 or corr < -0.3
    if off:
        label = "off_manifold_exploitation"
    elif climbing is not None and climbing > 0.005:
        label = "late_discovery"
    else:
        label = "on_manifold_stagnation"
    return {"label": label, "median_qed": float(np.median(qed)),
            "median_sa": float(np.median(sa)), "median_alerts": float(np.median(
                [r["alerts"] for r in rows])),
            "fraction_flagged": float(bad), "corr_score_qed": corr,
            "recent_frontier_gain": climbing}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledgers", required=True, help="dir of <task>.jsonl charged ledgers")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    report = {}
    for path in sorted(pathlib.Path(args.ledgers).glob("*.jsonl")):
        task = path.stem
        scored = []
        for line in path.read_text().strip().split("\n"):
            if not line:
                continue
            row = json.loads(line)
            if row.get("score") is None or not row.get("endpoint"):
                continue
            scored.append((float(row["score"]), row["endpoint"]))
        if len(scored) < 20:
            continue
        running, best = [], -1.0
        for s, _ in scored:
            best = max(best, s)
            running.append(best)
        top = sorted(scored, key=lambda r: -r[0])[:args.top]
        rows = []
        for s, smi in top:
            p = profile(smi)
            if p:
                rows.append({**p, "score": s, "smiles": smi})
        if len(rows) < 10:
            continue
        verdict = classify(rows, running)
        report[task] = {**verdict, "n_scored": len(scored), "best": max(s for s, _ in scored),
                        "worst_offender": max(rows, key=lambda r: r["score"])}
        print(f"{task:26s} {verdict['label']:26s} QED {verdict['median_qed']:.3f} "
              f"SA {verdict['median_sa']:.2f} flagged {verdict['fraction_flagged']:.2f} "
              f"r(score,QED) {verdict['corr_score_qed']:+.2f}", flush=True)
    pathlib.Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print(f"\n{len(report)} tasks audited -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
