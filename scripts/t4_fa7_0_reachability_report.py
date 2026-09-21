"""Reduce the fa7_0 reachability shards to a single verdict.

Reads every shard written by `scripts/t4_fa7_0_reachability_probe.py`, refuses to
emit a verdict from a degraded input set, and reports the one thing the panel
needs: did ANY production proposal configuration reach an eligible endpoint for
`fa7_0` at delta=0.6, and if not, which stage refused it.

WHY THE GUARDS
--------------
A verdict branch reachable with an empty or partial input set is a landmine: it
can be right for the wrong reason.  So this reducer raises rather than reporting
when there are no shards, when any shard carries a gate disagreement (which
voids that shard's attribution), or when the positive CONTROL arms produced no
eligible endpoint -- because "0 everywhere" is otherwise indistinguishable from
a broken harness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA_VERSION = "t4_fa7_0_reachability_report_v1"


def reduce_shards(shard_dir: Path, *, cell: str = "fa7_0") -> dict:
    shards = sorted(shard_dir.glob("*.json"))
    if not shards:
        raise ValueError(f"no shards under {shard_dir}; refusing to emit a verdict")
    rows = []
    for path in shards:
        row = json.loads(path.read_text())
        row["arm"] = path.stem
        rows.append(row)

    bad = [r["arm"] for r in rows if r["funnel"]["gate_disagreements"] != 0]
    if bad:
        raise ValueError(f"gate decomposition disagreed with the live gate in {bad}")

    target = [r for r in rows if r["cell"] == cell]
    controls = [r for r in rows if r["cell"] != cell]
    if not target:
        raise ValueError(f"no shard for {cell}")
    if not controls:
        raise ValueError("no control arm; a negative cannot be distinguished from a broken harness")
    control_eligible = sum(r["eligible"] for r in controls)
    if control_eligible == 0:
        raise ValueError(
            "every control arm returned zero eligible endpoints -- the harness is "
            "not demonstrated to be able to find one, so the target's zero means nothing"
        )

    total_draws = sum(r["draws"] for r in target)
    total_eligible = sum(r["eligible"] for r in target)
    total_checked = sum(r["funnel"]["distinct_endpoints_checked"] for r in target)
    total_sim = sum(r["funnel"]["pass_similarity"] for r in target)
    total_qed = sum(r["funnel"]["pass_qed"] for r in target)
    total_both = sum(r["funnel"]["pass_similarity_and_qed"] for r in target)

    best_qed_given_sim = max(
        (
            r["funnel"]["best_qed_among_similarity_passing"]
            for r in target
            if r["funnel"]["best_qed_among_similarity_passing"] is not None
        ),
        default=None,
    )
    best_sim_given_qed = max(
        (
            r["funnel"]["best_similarity_among_qed_passing"]
            for r in target
            if r["funnel"]["best_similarity_among_qed_passing"] is not None
        ),
        default=None,
    )

    if total_eligible > 0:
        verdict = "REACHABLE"
        killing_stage = None
    elif total_checked == 0:
        verdict = "NOT_REACHABLE"
        killing_stage = "region_draw_or_program_synthesis: the path constructed no endpoint at all"
    elif total_sim == 0:
        verdict = "NOT_REACHABLE"
        killing_stage = "replacement_construction: nothing built stayed inside the similarity ball"
    elif total_both == 0:
        verdict = "NOT_REACHABLE"
        killing_stage = (
            "eligibility_intersection: endpoints pass similarity and pass QED "
            "separately, but no single endpoint passes both"
        )
    else:
        verdict = "NOT_REACHABLE"
        killing_stage = "structural_or_sa_gate: similarity AND QED were met together but the endpoint was still refused"

    return {
        "schema_version": SCHEMA_VERSION,
        "cell": cell,
        "verdict": verdict,
        "killing_stage": killing_stage,
        "oracle_calls": 0,
        "docking_calls": 0,
        "arms": len(target),
        "total_draws": total_draws,
        "total_eligible": total_eligible,
        "eligible_per_draw": total_eligible / total_draws if total_draws else 0.0,
        "distinct_endpoints_checked": total_checked,
        "pass_similarity": total_sim,
        "pass_qed": total_qed,
        "pass_similarity_and_qed": total_both,
        "best_qed_among_similarity_passing": best_qed_given_sim,
        "best_similarity_among_qed_passing": best_sim_given_qed,
        "control_arms": [
            {
                "arm": r["arm"],
                "cell": r["cell"],
                "eligible": r["eligible"],
                "draws": r["draws"],
                "eligible_per_draw": round(r["eligible_per_draw"], 5),
            }
            for r in controls
        ],
        "target_arms": [
            {
                "arm": r["arm"],
                "lane": r["lane"],
                "region_law": r["region_law"],
                "horizon": r["horizon"],
                "draws": r["draws"],
                "seed": r["seed"],
                "eligible": r["eligible"],
                "checked": r["funnel"]["distinct_endpoints_checked"],
                "pass_similarity": r["funnel"]["pass_similarity"],
                "pass_qed": r["funnel"]["pass_qed"],
                "pass_similarity_and_qed": r["funnel"]["pass_similarity_and_qed"],
                "best_qed_among_similarity_passing": r["funnel"][
                    "best_qed_among_similarity_passing"
                ],
                "max_heavy_excision": r["funnel"]["max_heavy_excision"],
            }
            for r in sorted(target, key=lambda r: r["arm"])
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", required=True, type=Path)
    parser.add_argument("--cell", default="fa7_0")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = reduce_shards(args.shards, cell=args.cell)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"VERDICT {report['verdict']}  stage={report['killing_stage']}")
    print(
        f"  {report['arms']} arms, {report['total_draws']} draws, "
        f"{report['distinct_endpoints_checked']} distinct endpoints checked"
    )
    print(
        f"  simPass={report['pass_similarity']} qedPass={report['pass_qed']} "
        f"BOTH={report['pass_similarity_and_qed']}"
    )
    for row in report["target_arms"]:
        print(
            f"   {row['arm']:>22} lane={row['lane']:<21} law={str(row['region_law']):<19} "
            f"h={row['horizon']} elig={row['eligible']:<3} chk={row['checked']:<6} "
            f"sim={row['pass_similarity']:<5} qed={row['pass_qed']:<5} both={row['pass_similarity_and_qed']}"
        )
    for row in report["control_arms"]:
        print(f"   CONTROL {row['arm']:>18} {row['cell']} eligible={row['eligible']}/{row['draws']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
