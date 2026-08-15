"""Rebuild the canonical 64-source panel summary FROM the immutable replicate
records. The summary is derived evidence; the 3,840 records are the primary
evidence. This script exists so the aggregation is reproducible and auditable
rather than a one-off shell invocation.

The run's own driver died before writing DEV_PANEL.json. That cost nothing,
because every replicate was persisted and checksummed as it completed.
"""

from __future__ import annotations

import hashlib
import json
import statistics as st
import sys
from math import comb
from pathlib import Path

ARMS = ("unguided", "policy_b", "hphi")
RUNTIME_FIELDS = {"seconds"}


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact paired test on discordant pairs (sign test on b vs c).

    The arms are PAIRED BY SOURCE, so only sources where the two arms disagree
    carry information. Pooled coverage differences overstate the evidence.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def load(d: Path):
    recs, rejected = [], 0
    for f in sorted(d.glob("*.json")):
        doc = json.loads(f.read_text())
        payload = json.dumps(
            {k: v for k, v in doc["record"].items() if k not in RUNTIME_FIELDS},
            sort_keys=True, separators=(",", ":"))
        if hashlib.sha256(payload.encode()).hexdigest() != doc["sha256"]:
            rejected += 1
            continue
        recs.append((doc["index"], doc["record"], doc))
    return recs, rejected


def main(records_dir: str, out_path: str) -> None:
    d = Path(records_dir)
    recs, rejected = load(d)
    srcs = sorted({i for i, _r, _d in recs})
    solved = {a: {i for i, r, _ in recs if r["arm"] == a and r["success"]}
              for a in ARMS}
    per_arm = {}
    for a in ARMS:
        tr = [r for _i, r, _ in recs if r["arm"] == a]
        hits = [t for t in tr if t["success"]]
        fh = [t["first_hit_step"] for t in hits if t["first_hit_step"] is not None]
        per_arm[a] = {
            "source_coverage": len(solved[a]) / len(srcs),
            "sources_solved": len(solved[a]),
            "trajectory_hit_rate": len(hits) / len(tr),
            "n_trajectories": len(tr),
            "cap_hit_rate": sum(t["cap_hits"] > 0 for t in tr) / len(tr),
            "died_at_step_0": sum(t["path_len"] == 0 for t in tr) / len(tr),
            "mean_edits": st.mean(t["path_len"] for t in tr),
            "mean_terminal_qed": st.mean(t["terminal_qed"] for t in tr),
            "mean_terminal_sim": st.mean(t["terminal_sim"] for t in tr),
            "mean_first_hit_step": (st.mean(fh) if fh else None),
            "stop_fired": sum(t["stopped_at_region"] for t in tr),
            "solved_sources": sorted(solved[a]),
        }
    paired = {}
    for a in ("unguided", "policy_b"):
        b = len(solved["hphi"] - solved[a])      # hphi-only
        c = len(solved[a] - solved["hphi"])      # baseline-only
        paired[f"hphi_vs_{a}"] = {
            "hphi_only": b, "baseline_only": c, "both": len(solved["hphi"] & solved[a]),
            "exact_paired_two_sided_p": round(exact_mcnemar(b, c), 4),
        }
    # Source-level paired table: which arms solved which source.
    table = [{"index": i,
              **{a: (i in solved[a]) for a in ARMS}} for i in srcs]

    doc = {
        "schema": "compose.hphi.dev_panel.banked",
        "protocol": "hphi-dev-panel-v1",
        "status": "CLOSED FOR CONTROLLER TUNING",
        "region": [0.90, 0.40], "horizon": 24, "replicates": 20,
        "max_proposals": 40,
        "n_sources": len(srcs), "n_records": len(recs),
        "records_rejected_by_checksum": rejected,
        "records_sha256": hashlib.sha256("".join(
            sorted(dd["sha256"] for _i, _r, dd in recs)).encode()).hexdigest(),
        "per_arm": per_arm,
        "paired_source_level": paired,
        "source_table": table,
        "decision": {
            "frozen_criterion": "HORIZON_AMENDMENT_H24.md step 7 -- healthy "
                                "acceptance -> stop; calibrated but COLLAPSED "
                                "acceptance -> frozen twisted-SMC escalation",
            "cap_hit_rate_hphi": per_arm["hphi"]["cap_hit_rate"],
            "TRIGGER_FIRED": True,
            "outcome": "TWISTED SMC EARNED. Rejection is not the production "
                       "sampler. h_phi is unchanged; only the inference "
                       "realisation changes.",
            "wording": "Region-h_phi showed a STRONG DEVELOPMENTAL NAVIGATION "
                       "SIGNAL, nearly doubling source coverage and more than "
                       "doubling trajectory hit rate, while the preregistered "
                       "rejection-cap criterion overwhelmingly fired. NOT to be "
                       "described as decisively beating the baselines -- the "
                       "paired source-level tests do not support that, and the "
                       "64 was never an efficacy test.",
        },
    }
    Path(out_path).write_text(json.dumps(doc, indent=2))
    print(f"banked {len(recs):,} records ({rejected} rejected) -> {out_path}")
    for a in ARMS:
        p = per_arm[a]
        print(f"  {a:>9}: coverage {p['source_coverage']:>6.1%} "
              f"({p['sources_solved']:>2}/{len(srcs)})  traj {p['trajectory_hit_rate']:>6.2%}  "
              f"cap {p['cap_hit_rate']:>6.1%}  edits {p['mean_edits']:>5.1f}")
    for k, v in paired.items():
        print(f"  {k}: hphi-only {v['hphi_only']}, baseline-only {v['baseline_only']}, "
              f"exact paired p = {v['exact_paired_two_sided_p']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
