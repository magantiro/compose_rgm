"""Does the V2 selection preserve MULTI-SUCCESSOR structure, or flatten it?

WHY THIS GATE EXISTS
--------------------
The manifest keeps one `(task, source)` pair per Murcko scaffold, which cuts the
multistep lane from ~160k rows to 52,182.  Removing near-identical sources on a
shared scaffold is exactly what we want.  Collapsing a source that has SEVERAL
distinct plausible successors down to one arbitrary compiler choice is not:

    bad redundancy    50 near-identical sources on one scaffold, all showing
                      essentially the same insert/delete pattern
    useful multiplicity  one exact source x with distinct canonical successors
                      y1, y2, ... -- the supervision that teaches P(y|x) is a
                      DISTRIBUTION rather than a single memorised answer

The second kind is scientifically load-bearing here.  A recurring finding is
that teacher top-1 agreement is intrinsically ambiguous because several edits
are equally valid; a corpus retaining multi-successor states is what lets that
ambiguity be measured and trained on directly instead of assumed.

WHAT IS COMPARED
----------------
The selected set against the full cap-1 candidate pool, on: unique exact source
states, distinct canonical successors per source, the fraction of sources with
more than one successor, and the total (source, successor) pairs retained.

If multiplicity survives, 109k selected rows beat 218k unselected ones.  If the
selection flattens it, the fix is local -- keep one/few sources per scaffold but
retain up to K distinct successor classes per selected source -- and does not
touch the rest of the V2 design.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import pickle
import statistics
from pathlib import Path

LANE = "real_endpoint_multistep_path"


def scan_successors(active8_root: Path, compiled: set[str], cache_path: Path) -> dict:
    fingerprint = hashlib.sha256(
        json.dumps({"root": str(active8_root), "compiled": sorted(compiled), "v": 2},
                   sort_keys=True).encode()
    ).hexdigest()[:16]
    if cache_path.exists():
        blob = pickle.loads(cache_path.read_bytes())
        if blob.get("fingerprint") == fingerprint:
            print(f"cache hit ({cache_path.name})", flush=True)
            return blob

    per_task_source = collections.defaultdict(set)     # (task, source) -> successors
    per_source = collections.defaultdict(set)          # source -> successors (all tasks)
    for task in sorted(os.listdir(active8_root / "tasks")):
        receipt = active8_root / "tasks" / task / "RECEIPT.json"
        stream = active8_root / "tasks" / task / "transitions.jsonl.gz"
        if not (receipt.exists() and stream.exists()):
            continue
        if json.loads(receipt.read_text()).get("data_lane") != LANE:
            continue
        if task in compiled:
            continue
        for line in gzip.open(stream, "rt"):
            record = json.loads(line)
            evidence = record["candidate_evidence"]
            if evidence.get("exclusion_reason") is not None:
                continue
            source = evidence["source_canonical_key"]
            successor = evidence["canonical_successor_key"]
            per_task_source[(task, source)].add(successor)
            per_source[source].add(successor)

    blob = {
        "fingerprint": fingerprint,
        "per_task_source": {k: len(v) for k, v in per_task_source.items()},
        "per_source": {k: len(v) for k, v in per_source.items()},
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(pickle.dumps(blob))
    print(f"successor scan cached -> {cache_path}", flush=True)
    return blob


def summarize(label: str, counts: list[int]) -> dict:
    if not counts:
        return {"label": label, "sources": 0}
    multi = sum(1 for c in counts if c > 1)
    return {
        "label": label,
        "sources": len(counts),
        "successor_pairs": int(sum(counts)),
        "mean_successors": round(statistics.mean(counts), 3),
        "median_successors": statistics.median(counts),
        "fraction_multi_successor": round(multi / len(counts), 4),
        "max_successors": max(counts),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--active8-root", required=True)
    ap.add_argument("--prep-subset", default="configs/process_v2_prep_subset.json")
    ap.add_argument("--manifest", default="diagnostics/editing_v2_v2_selection_manifest.json")
    ap.add_argument("--cache", default="diagnostics/.v2_successor_cache.pkl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    plan = json.loads(Path(args.prep_subset).read_text())
    compiled = {c["task_identity_sha256"] for c in plan["chunks"]}
    blob = scan_successors(Path(args.active8_root), compiled, Path(args.cache))
    per_task_source = blob["per_task_source"]
    per_source = blob["per_source"]

    manifest = json.loads(Path(args.manifest).read_text())
    selected = manifest["selection"].get(LANE, {})
    sel_pairs = [(t, s) for t, sources in selected.items() for s in sources]

    pool_counts = list(per_task_source.values())
    sel_counts = [per_task_source.get(p, 0) for p in sel_pairs]
    # what the selected SOURCES would offer if every task were compiled
    sel_sources = {s for _t, s in sel_pairs}
    sel_all_tasks = [per_source[s] for s in sel_sources if s in per_source]

    rows = [
        summarize("candidate pool (all task,source)", pool_counts),
        summarize("SELECTED (task,source)", sel_counts),
        summarize("selected sources, all tasks", sel_all_tasks),
    ]
    print(f"\n{'set':34s} {'sources':>9} {'pairs':>9} {'mean':>7} {'med':>5} "
          f"{'%multi':>8} {'max':>5}")
    for r in rows:
        print(f"  {r['label']:32s} {r['sources']:>9,} {r.get('successor_pairs', 0):>9,} "
              f"{r.get('mean_successors', 0):>7.3f} {r.get('median_successors', 0):>5} "
              f"{100 * r.get('fraction_multi_successor', 0):>7.1f}% {r.get('max_successors', 0):>5}")

    pool = rows[0]
    sel = rows[1]
    retained = sel["successor_pairs"] / max(pool["successor_pairs"], 1)
    multi_ratio = (sel["fraction_multi_successor"]
                   / max(pool["fraction_multi_successor"], 1e-9))
    print(f"\nsuccessor pairs retained: {100 * retained:.1f}% of the pool")
    print(f"multi-successor rate vs pool: {multi_ratio:.2f}x "
          f"({100 * sel['fraction_multi_successor']:.1f}% vs "
          f"{100 * pool['fraction_multi_successor']:.1f}%)")
    verdict = "PRESERVED" if multi_ratio >= 0.95 else "FLATTENED"
    print(f"VERDICT: multi-successor structure {verdict}")
    if verdict == "FLATTENED":
        print("  fix locally: keep one/few sources per scaffold but retain up to K")
        print("  distinct successor classes per selected source; do not touch the mixture")

    Path(args.out).write_text(json.dumps(
        {"schema": "compose.editing_v2.multisuccessor_audit", "schema_version": 1,
         "status": "AUDIT_EVIDENCE_ONLY_NO_AUTHORITY", "lane": LANE,
         "verdict": verdict, "successor_pairs_retained_fraction": retained,
         "multi_successor_rate_ratio": multi_ratio, "sets": rows},
        indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
