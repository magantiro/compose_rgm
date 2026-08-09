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


def census(active8_root: Path, compiled_tasks: set[str], *, scaffold_cap: int = 60000) -> dict:
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
    uncompiled_sources: set[str] = set()
    for (lane, state), keys in sources.items():
        if state == "uncompiled":
            uncompiled_sources |= keys
    sample = sorted(compiled_sources)[:scaffold_cap]
    compiled_scaffolds = {s for s in (_scaffold(k) for k in sample) if s}
    new_scaffolds = set()
    for key in sorted(uncompiled_sources)[:scaffold_cap]:
        sc = _scaffold(key)
        if sc and sc not in compiled_scaffolds:
            new_scaffolds.add(sc)

    lanes = sorted({lane for lane, _s in fam})
    report = {
        "schema": "compose.editing_v2.corpus_census",
        "schema_version": 1,
        "status": "CENSUS_EVIDENCE_ONLY_NO_AUTHORITY",
        "tasks_on_disk": len(tasks),
        "tasks_compiled": len(compiled_tasks & set(tasks)),
        "accepted_rows_scanned": rows_seen,
        "compiled_scaffolds_sampled": len(compiled_scaffolds),
        "new_scaffolds_from_uncompiled": len(new_scaffolds),
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
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    plan = json.loads(Path(args.prep_subset).read_text())
    compiled = {c["task_identity_sha256"] for c in plan["chunks"]}
    report = census(Path(args.active8_root), compiled)

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
    print(f"new Bemis-Murcko scaffolds vs compiled: {report['new_scaffolds_from_uncompiled']:,} "
          f"(compiled sampled {report['compiled_scaffolds_sampled']:,})")

    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
