"""Carve controller-validation ENDPOINT-DISJOINT from teacher training.

WHY PAIR-DISJOINT IS NOT ENOUGH
-------------------------------
h_phi(x, z, b) conditions on the target z. So a split that merely keeps pairs
apart still leaks whenever

    training pair    A -> Z
    validation pair  B -> Z

because the model has already been taught the value landscape around Z. The
sealed 67 will demand generalisation to NEW target molecules, so held-in
validation has to ask the same question.

    no source or target canonical key in validation may appear as an
    endpoint of ANY training pair

CONSTRUCTION
------------
Molecules are nodes, nominated source-target pairs are edges, and whole
CONNECTED COMPONENTS are assigned to one side. That makes endpoint disjointness
structural rather than enforced: two pairs sharing any endpoint are in the same
component by definition and cannot be split.

One-cut MMP graphs are hub-prone -- a popular constant core links many
molecules -- so a giant component is expected. It is reported rather than worked
around, and the audit fails loudly if disjointness is ever violated.

The split is made BEFORE generation, so no label is produced without already
knowing which side it belongs to.

THE FOUR SETS, AND THEIR DISTINCT JOBS
--------------------------------------
    held-in train        teach h_phi
    held-in validation   choose dataset size / checkpoint / override threshold
    24 development       freeze the conservative override rule
    67 sealed            opened exactly once, after everything is frozen
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def commitment(rows: list[dict[str, Any]]) -> str:
    items = sorted(f"{r['source']}>>{r['target']}@{r['steps']}" for r in rows)
    return hashlib.sha256("\n".join(items).encode()).hexdigest()


def components(rows: list[dict[str, Any]]) -> dict[int, list[int]]:
    """Connected components of the molecule graph induced by the pairs."""

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for row in rows:
        union(row["source"], row["target"])

    grouped: dict[str, list[int]] = collections.defaultdict(list)
    for index, row in enumerate(rows):
        grouped[find(row["source"])].append(index)
    return {i: members for i, members in enumerate(grouped.values())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--validation-pairs", type=int, default=400,
                        help="size of the endpoint-disjoint validation POOL; only "
                             "a subset of it is ever teacher-labelled")
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()

    rows = json.loads(args.pairs.read_text())["transformations"]
    print(f"{len(rows)} held-in pairs "
          f"{dict(sorted(collections.Counter(r['steps'] for r in rows).items()))}")

    groups = components(rows)
    sizes = sorted((len(m) for m in groups.values()), reverse=True)
    print(f"  {len(groups)} connected components; largest {sizes[:5]}, "
          f"singletons {sum(1 for s in sizes if s == 1)}")
    largest = sizes[0] / len(rows) if rows else 0
    print(f"  giant component holds {largest:.1%} of pairs")

    # Fill validation from smaller components first: the giant component is
    # unsplittable without breaking endpoint disjointness, so it stays in train.
    rng = random.Random(args.seed)
    order = sorted(groups.values(), key=lambda m: (len(m), rng.random()))
    validation_index: set[int] = set()
    for members in order:
        if len(validation_index) >= args.validation_pairs:
            break
        if len(members) + len(validation_index) > args.validation_pairs * 1.3:
            continue
        validation_index.update(members)

    validation = [rows[i] for i in sorted(validation_index)]
    train = [rows[i] for i in range(len(rows)) if i not in validation_index]

    # AUDIT. Endpoint disjointness is the whole point, so it is asserted rather
    # than assumed -- in both directions.
    train_endpoints = {r["source"] for r in train} | {r["target"] for r in train}
    validation_endpoints = ({r["source"] for r in validation}
                            | {r["target"] for r in validation})
    overlap = train_endpoints & validation_endpoints
    print(f"  endpoint overlap: {len(overlap)}")
    if overlap:
        raise SystemExit(
            f"{len(overlap)} canonical molecules appear as endpoints on BOTH "
            f"sides; refusing to write a leaking carve")

    payload = {
        "schema": "compose.editing_v2.teacher_validation_carve",
        "status": "ENDPOINT_DISJOINT_CARVED_BEFORE_GENERATION",
        "rationale": (
            "h_phi conditions on the target z, so a merely pair-disjoint split "
            "leaks when training A->Z and validation B->Z share Z. Whole "
            "connected components of the molecule graph are assigned to one "
            "side, making endpoint disjointness structural."),
        "roles": {
            "held_in_train": "teach h_phi",
            "held_in_validation": "choose dataset size, checkpoint, threshold",
            "development_24": "freeze the conservative override rule",
            "sealed_67": "opened exactly once, after everything is frozen",
        },
        "seed": args.seed,
        "components": {
            "count": len(groups),
            "largest_pairs": sizes[0] if sizes else 0,
            "giant_share": largest,
            "singletons": sum(1 for s in sizes if s == 1),
        },
        "audit": {"endpoint_overlap": len(overlap),
                  "train_endpoints": len(train_endpoints),
                  "validation_endpoints": len(validation_endpoints)},
        "counts": {"train": len(train), "validation": len(validation)},
        "commitment": {"train_sha256": commitment(train),
                       "validation_sha256": commitment(validation)},
        "by_steps": {
            "train": dict(sorted(collections.Counter(
                r["steps"] for r in train).items())),
            "validation": dict(sorted(collections.Counter(
                r["steps"] for r in validation).items())),
        },
        "labelling_note": (
            "The validation POOL is deliberately larger than what gets labelled. "
            "Label ~50-100 validation pairs initially; that already yields "
            "hundreds of decision states, and the pool can be extended later "
            "without touching the sealed 67."),
        "train": train,
        "validation": validation,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"  train {len(train)} {payload['by_steps']['train']}")
    print(f"  validation {len(validation)} {payload['by_steps']['validation']}")
    print(f"  train sha256      {payload['commitment']['train_sha256'][:16]}")
    print(f"  validation sha256 {payload['commitment']['validation_sha256'][:16]}")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
