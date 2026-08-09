"""Census the Active8 corpus before compiling more of it.

WHY CENSUS FIRST
----------------
328 Active8 tasks exist on disk and only 34 are in ``configs/process_v2_prep_subset.json``, so
roughly 299k accepted real-lane rows are available for the cost of compute alone.  Compiling all
of them is the obvious move and probably right, but the raw count is not the question.  The
compiled corpus is already 58% insert/delete (``real_endpoint_multistep_path`` and
``observed_local_analogue`` are BOTH two-family lanes), and three families -- cycle_insert,
cycle_attach, ring_system_restate -- have zero real support anywhere in the Active8 output.

So the decision needs the marginal quantities, not the totals:

  * which families do the uncompiled rows actually add?
  * how many of their source molecules and Bemis-Murcko scaffolds are NEW, rather than more
    transitions from molecules already in the corpus?
  * how much new canonical-successor diversity?
  * would compiling them multiply the existing imbalance rather than repair it?

More data yes; more of exactly the same imbalance, no.

Writes a diagnostics JSON. Carries no authority and selects nothing.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
from pathlib import Path


def _scaffold(smiles: str) -> str | None:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except Exception:  # noqa: BLE001 - a scaffold failure must not abort a census
        return None


def census(active8_root: Path, compiled_tasks: set[str], *, scaffold_cap: int = 0) -> dict:
    """``scaffold_cap = 0`` means uncapped.

    A capped scaffold count is a FLOOR, not a measurement, and must not be used
    to choose a subsampling ratio.  The rows-per-scaffold distribution is
    reported alongside because a diversity-aware rule ("new scaffold first, then
    cap repeats") needs to know how concentrated a lane is -- a lane whose
    median scaffold carries 2 rows and one whose median carries 200 need very
    different caps for the same row budget.
    """

    tasks = sorted(os.listdir(active8_root / "tasks"))
    fam = collections.defaultdict(collections.Counter)          # (lane, state) -> family counts
    sources = collections.defaultdict(set)                      # (lane, state) -> source keys
    successors = collections.defaultdict(set)
    rows_seen = 0

    for task in tasks:
        receipt = active8_root / "tasks" / task / "RECEIPT.json"
        stream = active8_root / "tasks" / task / "transitions.jsonl.gz"
        if not (receipt.exists() and stream.exists()):
            continue
        lane = json.loads(receipt.read_text()).get("data_lane", "?")
        state = "compiled" if task in compiled_tasks else "uncompiled"
        for line in gzip.open(stream, "rt"):
            record = json.loads(line)
            evidence = record["candidate_evidence"]
            if evidence.get("exclusion_reason") is not None:
                continue
            rows_seen += 1
            key = (lane, state)
            fam[key][record["model_family"]] += 1
            sources[key].add(evidence["source_canonical_key"])
            successors[key].add(evidence["canonical_successor_key"])

    compiled_sources: set[str] = set()
    for (lane, state), keys in sources.items():
        if state == "compiled":
            compiled_sources |= keys

    # Scaffolds are the honest novelty measure: many "new" sources are analogues
    # of molecules already present, and add far less than their count suggests.
    cache: dict[str, str | None] = {}

    def scaffold_of(smiles: str) -> str | None:
        if smiles not in cache:
            cache[smiles] = _scaffold(smiles)
        return cache[smiles]

    def limited(keys):
        ordered = sorted(keys)
        return ordered if scaffold_cap <= 0 else ordered[:scaffold_cap]

    compiled_scaffolds = {s for s in (scaffold_of(k) for k in limited(compiled_sources)) if s}

    per_lane_scaffolds: dict[str, dict] = {}
    new_scaffolds: set[str] = set()
    for (lane, state), keys in sources.items():
        if state != "uncompiled":
            continue
        counts: collections.Counter = collections.Counter()
        for key in limited(keys):
            sc = scaffold_of(key)
            if sc:
                counts[sc] += 1
                if sc not in compiled_scaffolds:
                    new_scaffolds.add(sc)
        fresh = [s for s in counts if s not in compiled_scaffolds]
        sizes = sorted(counts.values())
        per_lane_scaffolds[lane] = {
            "unique_scaffolds": len(counts),
            "new_vs_compiled": len(fresh),
            "sources_per_scaffold_median": (sizes[len(sizes) // 2] if sizes else 0),
            "sources_per_scaffold_p90": (sizes[int(len(sizes) * 0.9)] if sizes else 0),
            "sources_per_scaffold_max": (sizes[-1] if sizes else 0),
            # what a "one source per scaffold" cap would actually retain
            "sources_retained_at_cap_1": len(counts),
            "sources_retained_at_cap_4": int(sum(min(v, 4) for v in counts.values())),
        }

    lanes = sorted({lane for lane, _s in fam})
    report = {
        "schema": "compose.editing_v2.corpus_census",
        "schema_version": 1,
        "status": "CENSUS_EVIDENCE_ONLY_NO_AUTHORITY",
        "tasks_on_disk": len(tasks),
        "tasks_compiled": len(compiled_tasks & set(tasks)),
        "accepted_rows_scanned": rows_seen,
        "scaffold_cap": scaffold_cap,
        "scaffold_count_is_exact": scaffold_cap <= 0,
        "compiled_scaffolds": len(compiled_scaffolds),
        "new_scaffolds_from_uncompiled": len(new_scaffolds),
        "scaffolds_by_lane": per_lane_scaffolds,
        "lanes": [],
    }
    for lane in lanes:
        entry = {"lane": lane}
        for state in ("compiled", "uncompiled"):
            k = (lane, state)
            src = sources.get(k, set())
            entry[state] = {
                "rows": int(sum(fam[k].values())),
                "families": dict(fam[k].most_common()),
                "unique_sources": len(src),
                "unique_successors": len(successors.get(k, set())),
                "sources_not_already_compiled": len(src - compiled_sources),
            }
        report["lanes"].append(entry)

    total_new_fam: collections.Counter = collections.Counter()
    for (lane, state), counter in fam.items():
        if state == "uncompiled":
            total_new_fam.update(counter)
    report["uncompiled_family_totals"] = dict(total_new_fam.most_common())
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--active8-root", required=True)
    ap.add_argument("--prep-subset", default="configs/process_v2_prep_subset.json")
    ap.add_argument("--scaffold-cap", type=int, default=0,
                    help="0 = uncapped; a capped count is a FLOOR and must not be "
                         "used to choose a subsampling ratio")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    plan = json.loads(Path(args.prep_subset).read_text())
    compiled = {c["task_identity_sha256"] for c in plan["chunks"]}
    report = census(Path(args.active8_root), compiled, scaffold_cap=args.scaffold_cap)

    print(f"tasks {report['tasks_compiled']}/{report['tasks_on_disk']} compiled, "
          f"{report['accepted_rows_scanned']:,} accepted rows scanned\n")
    print(f"{'lane':38s} {'state':11s} {'rows':>8} {'uniq src':>9} {'NEW src':>8} {'uniq succ':>10}")
    for entry in report["lanes"]:
        for state in ("compiled", "uncompiled"):
            d = entry[state]
            print(f"  {entry['lane']:36s} {state:11s} {d['rows']:>8,} "
                  f"{d['unique_sources']:>9,} {d['sources_not_already_compiled']:>8,} "
                  f"{d['unique_successors']:>10,}")
    print(f"\nfamilies the uncompiled rows would add: {report['uncompiled_family_totals']}")
    exact = "EXACT" if report["scaffold_count_is_exact"] else f"FLOOR (cap {report['scaffold_cap']:,})"
    print(f"\nnew Bemis-Murcko scaffolds vs compiled: {report['new_scaffolds_from_uncompiled']:,} "
          f"[{exact}]   compiled scaffolds {report['compiled_scaffolds']:,}")
    print(f"\n{'lane':38s} {'scaffolds':>10} {'new':>8} {'med':>5} {'p90':>5} {'max':>6} {'@cap4':>8}")
    for lane, s in sorted(report["scaffolds_by_lane"].items(),
                          key=lambda kv: -kv[1]["new_vs_compiled"]):
        print(f"  {lane:36s} {s['unique_scaffolds']:>10,} {s['new_vs_compiled']:>8,} "
              f"{s['sources_per_scaffold_median']:>5} {s['sources_per_scaffold_p90']:>5} "
              f"{s['sources_per_scaffold_max']:>6,} {s['sources_retained_at_cap_4']:>8,}")

    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
