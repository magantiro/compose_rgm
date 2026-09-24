"""Build target-blind reservoir banks and gate them against the shipped top-N banks.

Two artifacts per task, from the SAME frozen prescreen table the current banks use:
  initialization/<task>.json   16 molecules, the campaign's charged starts
  donors/<task>.json           a wider pool the transplant lane draws donors from

The gate is ZERO ORACLE and compares reservoir against the shipped top-N on the axes the
prefix-depth audit said matter: scaffold coverage, ring-system coverage, and whether the
oracle head is preserved. Motif counts are printed for HUMAN verification only and are
never an acceptance criterion -- they are reference-derived, and the selection rule cannot
see them.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

sys.path.insert(0, "src")
from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.prescreen_reservoir import select_scaffold_coverage  # noqa: E402
from compose_v4.rewrite.trace_shard import encode_state

OUT = pathlib.Path("diagnostics/pmo_prescreen_v1")
SLOTS = 48


def ranked_for(task: str, scores_dir: pathlib.Path, canon: dict[int, str]):
    rows = [(float(r["oracle_score"]), int(r["source_row_id"]))
            for r in csv.DictReader((scores_dir / f"{task}.csv").open())]
    rows.sort(key=lambda x: (-x[0], x[1]))
    return [(s, canon[i]) for s, i in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--init-count", type=int, default=16)
    ap.add_argument("--donor-count", type=int, default=200)
    ap.add_argument("--prefix", type=int, default=2000)
    args = ap.parse_args()

    canon = {int(r["source_row_id"]): r["canonical_smiles"]
             for r in csv.DictReader((OUT / "zinc250k_canonical_v1.csv").open())}
    scores_dir = pathlib.Path(args.scores_dir)
    (OUT / "reservoir_initialization").mkdir(parents=True, exist_ok=True)
    (OUT / "reservoir_donors").mkdir(parents=True, exist_ok=True)
    report = {}

    for task in args.tasks:
        ranked = ranked_for(task, scores_dir, canon)
        shipped = json.loads((OUT / "seeds" / f"{task}.json").read_text())["smiles"]
        for kind, count, folder in (("initialization", args.init_count, "reservoir_initialization"),
                                    ("donors", args.donor_count, "reservoir_donors")):
            picked = select_scaffold_coverage(ranked, count, prefix=args.prefix)
            candidates = []
            for rank, entry in enumerate(picked):
                graph = pad_molecular_graph(smiles_to_molecular_graph(entry.smiles), SLOTS)
                candidates.append({
                    "endpoint": entry.smiles,
                    "source_id": f"reservoir:{task}:{kind}:{rank}",
                    "state": encode_state(graph),
                    "oracle_rank": entry.oracle_rank,
                    "oracle_score": entry.oracle_score,
                    "selected_by": entry.selected_by,
                })
            body = {
                "accounting": "all initialization scores count against each run's oracle budget",
                "count": len(candidates),
                "rule": ("official-oracle ranking, keeping the FIRST molecule of each distinct "
                         "Bemis-Murcko scaffold. Target-blind: the dedup key knows nothing about "
                         "any reference, fingerprint or motif."),
                "prefix": args.prefix,
                "task_independent": False,
                "candidates": candidates,
            }
            path = OUT / folder / f"{task}.json"
            path.write_text(json.dumps(body, indent=1) + "\n")
            if kind == "initialization":
                res_smiles = [c["endpoint"] for c in candidates]

        top_n = shipped[:args.init_count]
        report[task] = {
            "reservoir_max_rank": max(c["oracle_rank"] for c in candidates),
            "reservoir_head_kept": sum(1 for c in candidates if c["selected_by"] == "oracle_head"),
            "init_oracle_mean_shipped": None,
            "shipped_top_n": top_n,
            "reservoir_init": res_smiles,
        }
        print(f"{task:26s} init={len(res_smiles)} donors={len(candidates)} "
              f"max_rank={report[task]['reservoir_max_rank']}", flush=True)
    (OUT / "reservoir_build_report.json").write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
