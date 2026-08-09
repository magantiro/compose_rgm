"""Census the physically compiled library, as the basis for a new sampling law.

The frozen V2 manifest is being retired as a sampling law: only 38,435 of its
107,872 pairs physically exist, and realizing it would compile 69,437 more while
leaving 63,881 already-valid compiled pairs unused. Its incremental logic assumed
a backbone that turned out to be a different selection entirely.

What survives is the SCIENCE behind it, and a new law has to be written against
the corpus that exists. This produces the evidence that law needs:

  * composition by lane, family, capability cell and role
  * real vs synthetic provenance (the synthetic walk is a capability
    regularizer, not the main corpus)
  * multi-successor structure -- how many sources carry more than one
    successor, which is the supervision the objective actually consumes
  * duplicate accounting, since a canonical (x, y) duplicate must not create
    artificial probability mass

Two numbers must never be conflated downstream, so both are reported: the
AVAILABLE composition here, and the REALIZED training coefficients after
sampling. This file is the former only.

Attribution is single-pass. The corpora carry ``source_state_sha256`` directly,
so the library is first keyed by ``(state, successor)`` -- no translation
needed -- and each stream record that matches contributes its lane, family,
cell and role. Pairs are then collapsed CANONICALLY, because the state hash
covers the padded slot arrangement and inflates 1.31x in the multistep lane.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path

SYNTHETIC_LANE = "reversible_synthetic_walk"


def library_state_pairs(root: Path) -> set[str]:
    """``state\\tsuccessor`` for every published entry in a local corpus tree."""

    found: set[str] = set()
    chunks = Path(root) / "chunks"
    if not chunks.is_dir():
        return found
    for task in sorted(chunks.iterdir()):
        if not task.is_dir():
            continue
        for slice_dir in sorted(task.iterdir()):
            payload = slice_dir / "ENTRIES.json"
            if not (slice_dir / "RECEIPT.json").exists() or not payload.exists():
                continue
            for entry in json.loads(payload.read_text())["entries"]:
                found.add(
                    f'{entry["source_state_sha256"]}\t{entry["successor_canonical_key"]}'
                )
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active8-root", required=True)
    parser.add_argument("--corpus-root", action="append", default=[])
    parser.add_argument("--pairs-file", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    state_pairs: set[str] = set()
    for root in args.corpus_root:
        found = library_state_pairs(Path(root))
        print(f"corpus {root}: {len(found):,} state-keyed pairs")
        state_pairs |= found
    for path in args.pairs_file:
        found = set(json.loads(Path(path).read_text()))
        print(f"pairs {path}: {len(found):,} state-keyed pairs")
        state_pairs |= found
    print(f"library: {len(state_pairs):,} state-keyed pairs\n")

    canonical: dict[str, dict] = {}
    successors_per_source: dict[str, set[str]] = collections.defaultdict(set)
    matched = 0
    tasks_root = Path(args.active8_root) / "tasks"
    for task_dir in sorted(tasks_root.iterdir()):
        stream = task_dir / "transitions.jsonl.gz"
        if not stream.exists():
            continue
        with gzip.open(stream, "rt") as handle:
            for line in handle:
                record = json.loads(line)
                evidence = record["candidate_evidence"]
                if evidence.get("exclusion_reason") is not None:
                    continue
                key = (
                    f'{evidence["source_state_sha256"]}\t'
                    f'{evidence["canonical_successor_key"]}'
                )
                if key not in state_pairs:
                    continue
                matched += 1
                source = str(evidence["source_canonical_key"])
                successor = str(evidence["canonical_successor_key"])
                digest = hashlib.blake2b(
                    f"{source}\t{successor}".encode(), digest_size=16
                ).hexdigest()
                successors_per_source[source].add(successor)
                if digest in canonical:
                    continue
                canonical[digest] = {
                    "lane": str(record["data_lane"]),
                    "family": str(record["model_family"]),
                    "cell": str(record.get("capability_cell_id", "?")),
                    "role": str(record.get("partition_role", "?")),
                    "source": source,
                }

    rows = list(canonical.values())
    total = len(rows)
    by_role: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_role[row["role"]].append(row)

    def _compose(subset: list[dict]) -> dict:
        lanes = collections.Counter(r["lane"] for r in subset)
        synthetic = sum(1 for r in subset if r["lane"] == SYNTHETIC_LANE)
        sources = {r["source"] for r in subset}
        multi = sum(1 for s in sources if len(successors_per_source[s]) > 1)
        return {
            "pairs": len(subset),
            "by_lane": dict(lanes.most_common()),
            "by_family": dict(collections.Counter(r["family"] for r in subset).most_common()),
            "by_capability_cell": dict(
                collections.Counter(r["cell"] for r in subset).most_common()
            ),
            "synthetic_pairs": synthetic,
            "real_pairs": len(subset) - synthetic,
            "synthetic_share": round(synthetic / max(len(subset), 1), 4),
            "distinct_sources": len(sources),
            "multi_successor_sources": multi,
            "multi_successor_fraction": round(multi / max(len(sources), 1), 4),
            "capability_cells": len({r["cell"] for r in subset}),
        }

    report = {
        "schema": "compose.editing_v2.available_library_census",
        "schema_version": 1,
        "status": "CENSUS_EVIDENCE_ONLY_NO_AUTHORITY",
        "state_keyed_pairs": len(state_pairs),
        "canonical_pairs": total,
        "duplicate_pairs_collapsed": len(state_pairs) - total,
        "stream_records_matched": matched,
        "overall": _compose(rows),
        "by_role": {role: _compose(subset) for role, subset in sorted(by_role.items())},
    }
    body = json.dumps(report, sort_keys=True).encode()
    report["frozen_sha256"] = hashlib.sha256(body).hexdigest()
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"canonical pairs {total:,} "
          f"({report['duplicate_pairs_collapsed']:,} state-keyed duplicates collapsed)\n")
    for role, subset in sorted(by_role.items()):
        c = _compose(subset)
        print(f"{role:24} {c['pairs']:>7,}  synthetic {100*c['synthetic_share']:5.1f}%  "
              f"cells {c['capability_cells']:>2}  multi-succ {100*c['multi_successor_fraction']:5.1f}%")
    train = _compose(by_role.get("train", []))
    print(f"\nTRAIN families:")
    for family, count in train["by_family"].items():
        print(f"  {family:22} {count:>7,}  {100*count/max(train['pairs'],1):5.2f}%")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
