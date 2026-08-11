"""Freeze the teacher-generation cohort BEFORE any label exists.

WHY
---
If the number of pairs processed could depend on how many labels earlier pairs
produced, then pair membership is a function of observed teacher outcomes. Even
when that adaptation is benign, it is indefensible on paper: a reviewer cannot
distinguish "we processed more pairs because yield was low" from "we processed
more pairs until the data looked how we wanted".

So the cohort is chosen deterministically, hashed, and committed first. Every
pair in it is labelled, whatever it yields.

    pair selection  INDEPENDENT OF  observed teacher outcomes

Deliberately oversized. The union collector produced 16-78 labels per pair
across twelve verification pairs -- a wide spread around ~45 -- so a cohort
sized only to the central estimate would land under target often enough to
create pressure to extend it. Sizing well above target removes that pressure
entirely, at a cost of a few dollars.

Selection is stratified across verified path length and deterministic given the
seed. It consults nothing about labels, because none exist yet.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path
from typing import Any

#: Measured on the twelve union-collector verification pairs.
LABELS_PER_PAIR_ESTIMATE = 44.75


def commitment(rows: list[dict[str, Any]]) -> str:
    items = sorted(f"{r['source']}>>{r['target']}@{r['steps']}" for r in rows)
    return hashlib.sha256("\n".join(items).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carve", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--train-pairs", type=int, default=290)
    parser.add_argument("--validation-pairs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()

    carve = json.loads(args.carve.read_text())
    if carve["audit"]["endpoint_overlap"] != 0:
        raise SystemExit("the carve reports endpoint overlap; refusing to freeze")

    def stratified(rows: list[dict[str, Any]], want: int, seed: int):
        rng = random.Random(seed)
        lengths = sorted({r["steps"] for r in rows})
        per = max(1, want // max(len(lengths), 1))
        picked: list[dict[str, Any]] = []
        for length in lengths:
            band = sorted([r for r in rows if r["steps"] == length],
                          key=lambda r: (r["source"], r["target"]))
            rng.shuffle(band)
            picked.extend(band[:per])
        # Top up deterministically if a band was short, so the cohort size is
        # exact rather than dependent on how the bands happened to fill.
        if len(picked) < want:
            chosen = {(r["source"], r["target"]) for r in picked}
            rest = sorted([r for r in rows
                           if (r["source"], r["target"]) not in chosen],
                          key=lambda r: (r["source"], r["target"]))
            rng.shuffle(rest)
            picked.extend(rest[: want - len(picked)])
        return picked[:want]

    train = stratified(carve["train"], args.train_pairs, args.seed)
    validation = stratified(carve["validation"], args.validation_pairs,
                            args.seed + 1)

    # The carve guaranteed endpoint disjointness for the pools; re-assert it for
    # the frozen cohorts, since these are what actually get labelled.
    train_endpoints = {r["source"] for r in train} | {r["target"] for r in train}
    validation_endpoints = ({r["source"] for r in validation}
                            | {r["target"] for r in validation})
    overlap = train_endpoints & validation_endpoints
    if overlap:
        raise SystemExit(f"{len(overlap)} endpoints shared between the frozen "
                         f"cohorts; refusing to freeze")

    expected = len(train) * LABELS_PER_PAIR_ESTIMATE
    payload = {
        "schema": "compose.editing_v2.teacher_cohort_freeze",
        "status": "FROZEN_BEFORE_ANY_TEACHER_LABEL_EXISTS",
        "rule": (
            "Every pair in this cohort is labelled, whatever it yields. Pair "
            "membership is never revised in response to observed label counts, "
            "so pair selection is independent of teacher outcomes."),
        "sizing_note": (
            f"Deliberately oversized. Measured yield was 16-78 labels per pair "
            f"(mean {LABELS_PER_PAIR_ESTIMATE}), so a cohort sized to the central "
            f"estimate would often land under target and create pressure to "
            f"extend it. {len(train)} pairs project to ~{expected:,.0f} labels "
            f"against a 10,000 target."),
        "seed": args.seed,
        "counts": {"train": len(train), "validation": len(validation)},
        "expected_train_labels": round(expected),
        "commitment": {"train_sha256": commitment(train),
                       "validation_sha256": commitment(validation)},
        "audit": {"endpoint_overlap": 0,
                  "train_endpoints": len(train_endpoints),
                  "validation_endpoints": len(validation_endpoints)},
        "by_steps": {
            "train": dict(sorted(collections.Counter(
                r["steps"] for r in train).items())),
            "validation": dict(sorted(collections.Counter(
                r["steps"] for r in validation).items())),
        },
        "train": train,
        "validation": validation,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"FROZEN COHORT")
    print(f"  train      {len(train):4d} pairs {payload['by_steps']['train']}")
    print(f"  validation {len(validation):4d} pairs "
          f"{payload['by_steps']['validation']}")
    print(f"  projected train labels ~{expected:,.0f} against a 10,000 target")
    print(f"  train sha256      {payload['commitment']['train_sha256'][:16]}")
    print(f"  validation sha256 {payload['commitment']['validation_sha256'][:16]}")
    print(f"  endpoint overlap between cohorts: 0")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
