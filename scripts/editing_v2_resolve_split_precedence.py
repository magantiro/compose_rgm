"""Make the four partitions mutually source-disjoint by a fixed precedence.

Roles are assigned per TASK in the source cache, and that does not make the
partitions source-disjoint: measured, five of six role pairs share canonical
source molecules (final_test x train 82, controller_validation x final_test 10,
train x validation 10, controller_validation x train 6, final_test x validation
6).

Excluding shared sources from TRAIN alone is not enough. It cleans training
contamination but leaves final_test overlapping controller_validation and
validation, so "no canonical source occurs in more than one partition" would
still be false -- and that sentence is the one an evaluation claim rests on.

PRECEDENCE
----------
    final_test > controller_validation > validation > train

Held-out roles outrank train because a sealed evaluation set is worth more than
a handful of training rows, and final_test outranks the other held-out roles
because it is the one that must stay pristine. A source appearing in several
roles is kept ONLY in its highest-ranked role and excluded from the rest.

The counts are tiny, so this costs almost nothing and buys an exact statement.
Nothing is recompiled and nothing is deleted: the library keeps every row, and
these exclusions bind the samplers that draw from it.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import itertools
import json
from pathlib import Path

#: Highest priority first.
PRECEDENCE = ("final_test", "controller_validation", "validation", "train")


def sources_by_role(active8_root: Path) -> dict[str, set[str]]:
    by_role: dict[str, set[str]] = collections.defaultdict(set)
    for task_dir in sorted((Path(active8_root) / "tasks").iterdir()):
        stream = task_dir / "transitions.jsonl.gz"
        if not stream.exists():
            continue
        with gzip.open(stream, "rt") as handle:
            for line in handle:
                record = json.loads(line)
                evidence = record["candidate_evidence"]
                if evidence.get("exclusion_reason") is not None:
                    continue
                by_role[str(record.get("partition_role", "?"))].add(
                    str(evidence["source_canonical_key"])
                )
    return dict(by_role)


def resolve(by_role: dict[str, set[str]]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Return (kept, excluded) per role under PRECEDENCE."""

    claimed: set[str] = set()
    kept: dict[str, set[str]] = {}
    excluded: dict[str, set[str]] = {}
    for role in PRECEDENCE:
        present = by_role.get(role, set())
        keep = present - claimed
        kept[role] = keep
        excluded[role] = present & claimed
        claimed |= keep
    for role in by_role:
        if role not in kept:  # a role outside the precedence list must not be silent
            raise SystemExit(f"role {role!r} is not in PRECEDENCE; refusing to guess")
    return kept, excluded


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active8-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    by_role = sources_by_role(Path(args.active8_root))
    kept, excluded = resolve(by_role)

    print(f"{'role':24} {'before':>9} {'kept':>9} {'excluded':>9}")
    for role in PRECEDENCE:
        print(f"  {role:22} {len(by_role.get(role, set())):>9,} "
              f"{len(kept[role]):>9,} {len(excluded[role]):>9,}")

    residual = {
        f"{a}|{b}": len(kept[a] & kept[b])
        for a, b in itertools.combinations(PRECEDENCE, 2)
    }
    leaks = {k: v for k, v in residual.items() if v}
    print(f"\nresidual intersections after resolution: "
          f"{'NONE - partitions are source-disjoint' if not leaks else leaks}")

    report = {
        "schema": "compose.editing_v2.split_precedence_resolution",
        "schema_version": 1,
        "status": "RESOLUTION_EVIDENCE_ONLY_NO_AUTHORITY",
        "precedence": list(PRECEDENCE),
        "sources_before": {r: len(by_role.get(r, set())) for r in PRECEDENCE},
        "sources_kept": {r: len(kept[r]) for r in PRECEDENCE},
        "sources_excluded": {r: len(excluded[r]) for r in PRECEDENCE},
        "residual_pairwise_intersections": residual,
        "mutually_source_disjoint": not leaks,
        "excluded_source_keys": {r: sorted(excluded[r]) for r in PRECEDENCE},
    }
    body = json.dumps({k: v for k, v in report.items() if k != "frozen_sha256"}, sort_keys=True)
    report["frozen_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nfrozen {report['frozen_sha256'][:16]}\nwrote {args.out}")
    return 0 if not leaks else 1


if __name__ == "__main__":
    raise SystemExit(main())
