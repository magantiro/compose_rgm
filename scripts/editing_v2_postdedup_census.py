"""Freeze every V2 headline AFTER global canonical (x, y) deduplication.

WHY RECOUNT
-----------
Every figure quoted while designing V2 came from the PRE-dedup manifest, and
45.1% of those records are duplicate canonical pairs.  Since almost all the
duplication sits in the REAL multistep lane (63.6% there, 0.0% in both small
real lanes, 10.7% synthetic), removing it necessarily RAISES the synthetic
share.  The 10.8% figure quoted earlier is stale, and a training sampler or a
paper table built on it would be describing a corpus that does not exist.

This also checks something the per-lane numbers cannot: whether the same
canonical (x, y) appears in more than one LANE.  Per-lane dedup would miss that,
and cross-lane duplicates would quietly double a transition's weight.

Emits a frozen hash over the final counts so a later table can be checked
against the corpus it claims to describe.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--active8-root", required=True)
    ap.add_argument("--manifest", default="diagnostics/editing_v2_v2_selection_manifest.json")
    ap.add_argument("--compiled-census", default="diagnostics/editing_v2_corpus_census.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.active8_root)
    manifest = json.loads(Path(args.manifest).read_text())
    selection = manifest["selection"]
    lane_of_task = {}
    for lane, tasks in selection.items():
        for task in tasks:
            lane_of_task[task] = lane
    wanted = {task: set(sources) for lane, tasks in selection.items()
              for task, sources in tasks.items()}

    seen_global: set[tuple[str, str]] = set()
    by_lane_pairs = collections.defaultdict(set)
    by_lane_family: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    by_lane_sources = collections.defaultdict(set)
    successors_per_source = collections.defaultdict(set)
    cells: set[str] = set()
    cross_lane = 0
    raw = 0

    for task in sorted(wanted):
        stream = root / "tasks" / task / "transitions.jsonl.gz"
        if not stream.exists():
            continue
        lane = lane_of_task[task]
        for line in gzip.open(stream, "rt"):
            record = json.loads(line)
            evidence = record["candidate_evidence"]
            if evidence.get("exclusion_reason") is not None:
                continue
            source = evidence["source_canonical_key"]
            if source not in wanted[task]:
                continue
            raw += 1
            pair = (source, evidence["canonical_successor_key"])
            if pair in seen_global:
                if pair not in by_lane_pairs[lane]:
                    cross_lane += 1          # same transition, different lane
                continue
            seen_global.add(pair)
            by_lane_pairs[lane].add(pair)
            by_lane_family[lane][record["model_family"]] += 1
            by_lane_sources[lane].add(source)
            successors_per_source[source].add(pair[1])
            cells.add(record.get("capability_cell_id", "?"))

    synthetic_lane = "reversible_synthetic_walk"
    new_total = sum(len(v) for v in by_lane_pairs.values())
    new_synth = len(by_lane_pairs.get(synthetic_lane, ()))
    families: collections.Counter = collections.Counter()
    for counter in by_lane_family.values():
        families.update(counter)

    existing_real = existing_synth = 0
    census_path = Path(args.compiled_census)
    if census_path.exists():
        for entry in json.loads(census_path.read_text())["lanes"]:
            rows = entry["compiled"]["rows"]
            if entry["lane"] == synthetic_lane:
                existing_synth += rows
            else:
                existing_real += rows

    multi = sum(1 for v in successors_per_source.values() if len(v) > 1)
    report = {
        "schema": "compose.editing_v2.postdedup_census",
        "schema_version": 1,
        "status": "CENSUS_EVIDENCE_ONLY_NO_AUTHORITY",
        "selection_sha256": manifest["selection_sha256"],
        "raw_records_scanned": raw,
        "rows_after_dedup_total": new_total,
        "duplicate_fraction": round(1 - new_total / max(raw, 1), 4),
        "cross_lane_duplicate_pairs": cross_lane,
        "rows_by_lane": {k: len(v) for k, v in sorted(by_lane_pairs.items())},
        "family_totals": dict(families.most_common()),
        "distinct_sources": len(successors_per_source),
        "distinct_canonical_pairs": len(seen_global),
        "multi_successor_sources": multi,
        "multi_successor_fraction": round(multi / max(len(successors_per_source), 1), 4),
        "mean_successors_per_source": round(
            sum(len(v) for v in successors_per_source.values())
            / max(len(successors_per_source), 1), 3),
        "capability_cells": len(cells),
        "new_synthetic_rows": new_synth,
        "new_synthetic_share": round(new_synth / max(new_total, 1), 4),
        "existing_compiled_real_rows": existing_real,
        "existing_compiled_synthetic_rows": existing_synth,
        "combined_synthetic_share": round(
            (new_synth + existing_synth) / max(new_total + existing_real + existing_synth, 1), 4),
    }
    body = json.dumps({k: v for k, v in report.items() if k != "frozen_sha256"},
                      sort_keys=True).encode()
    report["frozen_sha256"] = hashlib.sha256(body).hexdigest()

    print(f"raw records scanned          {report['raw_records_scanned']:,}")
    print(f"after canonical (x,y) dedup  {report['rows_after_dedup_total']:,} "
          f"({100 * report['duplicate_fraction']:.1f}% duplicates removed)")
    print(f"cross-lane duplicate pairs   {report['cross_lane_duplicate_pairs']:,}")
    print(f"\n{'lane':38s} {'rows':>9}")
    for lane, n in report["rows_by_lane"].items():
        print(f"  {lane:36s} {n:>9,}")
    print(f"\nfamilies: {report['family_totals']}")
    print(f"\ndistinct sources             {report['distinct_sources']:,}")
    print(f"multi-successor sources      {report['multi_successor_sources']:,} "
          f"({100 * report['multi_successor_fraction']:.1f}%)")
    print(f"mean successors per source   {report['mean_successors_per_source']}")
    print(f"capability cells             {report['capability_cells']}")
    print(f"\nNEW synthetic share          {100 * report['new_synthetic_share']:.1f}%   "
          f"(pre-dedup estimate was 10.8%)")
    print(f"COMBINED synthetic share     {100 * report['combined_synthetic_share']:.1f}%   "
          f"(incl. {report['existing_compiled_synthetic_rows']:,} already-compiled synthetic)")
    print(f"\nfrozen sha256 {report['frozen_sha256']}")

    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
