#!/usr/bin/env python3
"""Read the matched PMO A/B and report it in the shape the goal asks for.

Reports best-vs-calls, top-10 AUC at the actual budget, the frontier trajectory, unique
scored molecules, proposal throughput, provenance, and the chemistry of what each arm
actually scored.  gsk3b is reported with its predeclared caveat attached, never silently.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import QED, RDConfig

RDLogger.DisableLog("rdApp.*")
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

TASKS = ("perindopril_mpo", "celecoxib_rediscovery", "gsk3b")
CLEAN = ("perindopril_mpo", "celecoxib_rediscovery")
CAVEAT = (
    "gsk3b's score is NOT interpretable as search quality: its fingerprint-RandomForest "
    "oracle rewards off-manifold chemistry (r(score,QED) = -0.433, r(score,SA) = +0.684). "
    "Predeclared before the run; promotion evidence comes from the other two tasks."
)


def _chem(smis: list[str]) -> dict:
    q, a, h = [], [], []
    for s in smis:
        m = Chem.MolFromSmiles(s) if s else None
        if m is None:
            continue
        q.append(QED.qed(m))
        a.append(sascorer.calculateScore(m))
        h.append(m.GetNumHeavyAtoms())
    if not q:
        return {}
    return {
        "median_qed": round(st.median(q), 4),
        "median_sa": round(st.median(a), 4),
        "median_heavy": round(st.median(h), 1),
        "pct_qed_ge_0_6": round(100 * sum(x >= 0.6 for x in q) / len(q), 1),
        "pct_sa_le_4": round(100 * sum(x <= 4 for x in a) / len(a), 1),
    }


def read_arm(root: Path, task: str) -> dict | None:
    result = root / task / "result.json"
    if not result.exists():
        return None
    res = json.loads(result.read_text())
    rounds = sorted((root / task / "campaign").glob("round_*/complete.json"))
    traj, obs = [], {}
    for p in rounds:
        d = json.loads(p.read_text())
        s = d["summary"]
        traj.append({
            "round": s.get("round"),
            "calls": s.get("oracle_calls_including_initialization"),
            "auc_so_far": s.get("pmo_top10_auc_so_far"),
            "proposal_seconds": s.get("proposal_seconds"),
            "proposal_attempts": s.get("proposal_attempts"),
        })
        obs = d["snapshot"].get("observations", obs)
    scores = sorted((v["score"] for v in obs.values() if v.get("score") is not None), reverse=True)
    channels = {}
    last = json.loads(rounds[-1].read_text()) if rounds else {}
    for entry in (last.get("snapshot", {}).get("entries") or {}).values():
        ch = (entry.get("provenance") or {}).get("planner_channel", "?")
        channels[ch] = channels.get(ch, 0) + 1
    return {
        "best": res.get("best_score"),
        "auc": res.get("auc_top10_at_budget") or res.get("auc_at_250"),
        "charged": res.get("charged_calls"),
        "unique_scored": len({v.get("endpoint") for v in obs.values() if v.get("endpoint")}),
        "top10_mean": round(st.fmean(scores[:10]), 4) if len(scores) >= 10 else None,
        "trajectory": traj,
        "channel_entries": channels,
        "chemistry_of_scored": _chem([v.get("endpoint") for v in obs.values()]),
        "proposal_seconds_total": round(sum(t["proposal_seconds"] or 0 for t in traj), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-root", required=True, type=Path)
    ap.add_argument("--b-root", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    report = {"schema_version": "pmo_ab_comparison_v1", "gsk3b_caveat": CAVEAT, "tasks": {}}
    for task in TASKS:
        a, b = read_arm(args.a_root, task), read_arm(args.b_root, task)
        if a is None or b is None:
            report["tasks"][task] = {"status": "INCOMPLETE",
                                     "A_present": a is not None, "B_present": b is not None}
            continue
        report["tasks"][task] = {
            "A_baseline": a, "B_memory": b,
            "delta_best": None if None in (a["best"], b["best"]) else round(b["best"] - a["best"], 4),
            "delta_auc": None if None in (a["auc"], b["auc"]) else round(b["auc"] - a["auc"], 5),
            "counts_for_promotion": task in CLEAN,
        }
    done = [t for t, v in report["tasks"].items()
            if v.get("counts_for_promotion") and "delta_auc" in v and v["delta_auc"] is not None]
    report["verdict_basis"] = (
        f"clean tasks complete: {done}. Promotion requires the mechanism to be ACTIVE and search to "
        "IMPROVE on these; gsk3b is excluded by its predeclared caveat.")
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
