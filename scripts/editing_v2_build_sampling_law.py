"""Derive the training sampling law over the physical library.

    compiled library  !=  training distribution

The library is what chemistry exists; the law is the probability measure the
model is trained under. They are deliberately different, and both must be
reported -- the available composition, and the realized coefficients.

WHY A NEW LAW
-------------
The frozen V2 manifest is retired as a sampling law: only 38,435 of its 107,872
pairs physically exist, and its incremental logic assumed a backbone that was a
different selection entirely. What survives is the SCIENCE it encoded, restated
here as explicit constraints over the corpus that actually exists:

  * real chemistry dominates; the synthetic walk is a capability regularizer,
    not the corpus
  * no family disappears from training
  * rare, load-bearing operators get enough exposure
  * multi-successor supervision is preserved
  * canonical (x, y) duplicates never create artificial probability mass
  * held-out sources are never drawn

THE SYNTHETIC TENSION, RESOLVED BY THE LAW RATHER THAN THE COMPILE
------------------------------------------------------------------
``ring_system_restate`` exists ONLY in the synthetic walk lane, so closing that
capability hole necessarily made the LIBRARY 44.6% synthetic. Training at 44.6%
synthetic would contradict "real chemistry dominates". The law therefore holds
synthetic to a target share while still meeting every family floor, by drawing
synthetic rows preferentially where they are the only source of a family and
sparingly elsewhere. Compile broadly, train deliberately.

HOW THE ALLOCATION WORKS
------------------------
1. Family targets start proportional to availability, then are clamped to
   [floor, cap] and renormalized. The clamp is what stops insert/delete
   dominance and what keeps a rare family from vanishing.
2. Within a family, real rows are preferred; synthetic fills only the shortfall.
3. If the synthetic draw implied by (2) exceeds the global synthetic cap, the
   excess is trimmed from the families with the most real headroom first --
   never from a family whose only supply is synthetic, since that would silently
   reintroduce the capability hole this corpus was just repaired to close.

Weights are per stratum (family, provenance), uniform within a stratum. Uniform
within stratum keeps multi-successor structure intact: every successor of a
source is an independent row and none is upweighted for having siblings.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path

SYNTHETIC_LANE = "reversible_synthetic_walk"


def load_library(corpus_roots: list[str], pairs_files: list[str]) -> set[str]:
    state_pairs: set[str] = set()
    for root in corpus_roots:
        chunks = Path(root) / "chunks"
        if not chunks.is_dir():
            continue
        for task in sorted(chunks.iterdir()):
            if not task.is_dir():
                continue
            for slice_dir in sorted(task.iterdir()):
                payload = slice_dir / "ENTRIES.json"
                if not (slice_dir / "RECEIPT.json").exists() or not payload.exists():
                    continue
                for entry in json.loads(payload.read_text())["entries"]:
                    state_pairs.add(
                        f'{entry["source_state_sha256"]}\t{entry["successor_canonical_key"]}'
                    )
    for path in pairs_files:
        state_pairs |= set(json.loads(Path(path).read_text()))
    return state_pairs


def attribute(active8_root: Path, state_pairs: set[str], excluded: set[str]) -> list[dict]:
    """One record per canonical pair, with the facts the law allocates over."""

    seen: set[str] = set()
    rows: list[dict] = []
    for task_dir in sorted((Path(active8_root) / "tasks").iterdir()):
        stream = task_dir / "transitions.jsonl.gz"
        if not stream.exists():
            continue
        with gzip.open(stream, "rt") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("partition_role") != "train":
                    continue
                evidence = record["candidate_evidence"]
                if evidence.get("exclusion_reason") is not None:
                    continue
                key = (
                    f'{evidence["source_state_sha256"]}\t'
                    f'{evidence["canonical_successor_key"]}'
                )
                if key not in state_pairs:
                    continue
                source = str(evidence["source_canonical_key"])
                # A source that appears in any held-out partition is never drawn.
                if source in excluded:
                    continue
                digest = hashlib.blake2b(
                    f'{source}\t{evidence["canonical_successor_key"]}'.encode(),
                    digest_size=16,
                ).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                rows.append({
                    "family": str(record["model_family"]),
                    "synthetic": str(record["data_lane"]) == SYNTHETIC_LANE,
                    "cell": str(record.get("capability_cell_id", "?")),
                    "source": source,
                })
    return rows


def allocate(rows: list[dict], *, floor: float, cap: float, synthetic_target: float,
             max_oversample: float) -> dict:
    supply: dict[tuple[str, bool], int] = collections.Counter(
        (r["family"], r["synthetic"]) for r in rows
    )
    families = sorted({r["family"] for r in rows})
    total = len(rows)

    # 1. family targets: availability, clamped, renormalized
    raw = {f: sum(v for (fam, _), v in supply.items() if fam == f) / total for f in families}
    clamped = {f: min(max(raw[f], floor), cap) for f in families}
    scale = sum(clamped.values())
    target = {f: clamped[f] / scale for f in families}

    # 2. prefer real within each family, OVERSAMPLING real up to
    #    max_oversample before falling back to synthetic.
    #
    #    Capping the real draw at its available share was wrong: it sent a
    #    family to synthetic merely because its real rows were fewer than its
    #    target, when repeating those real rows is exactly what an oversample
    #    factor is for. That bug made the law a no-op -- realized synthetic came
    #    out at 44.6%, identical to the library.
    #
    #    The bound matters: unbounded oversampling would train on a handful of
    #    real rows repeated hundreds of times, which is not "real chemistry
    #    dominates", it is memorization wearing its label.
    draw: dict[tuple[str, bool], float] = {}
    for family in families:
        want = target[family]
        real_supply = supply.get((family, False), 0) / total
        real = min(want, real_supply * max_oversample)
        draw[(family, False)] = real
        draw[(family, True)] = want - real

    # 3. trim synthetic overshoot from families with the most REAL headroom,
    #    never from a family whose only supply is synthetic
    synthetic = sum(v for (_, syn), v in draw.items() if syn)
    if synthetic > synthetic_target:
        excess = synthetic - synthetic_target
        headroom = sorted(
            (
                (supply.get((f, False), 0) / total - draw[(f, False)], f)
                for f in families
                if supply.get((f, False), 0) > 0 and draw[(f, True)] > 0
            ),
            reverse=True,
        )
        for room, family in headroom:
            if excess <= 0:
                break
            move = min(excess, draw[(family, True)], max(room, 0.0))
            draw[(family, True)] -= move
            draw[(family, False)] += move
            excess -= move
    return {"supply": supply, "target": target, "draw": draw, "total": total}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active8-root", required=True)
    parser.add_argument("--corpus-root", action="append", default=[])
    parser.add_argument("--pairs-file", action="append", default=[])
    parser.add_argument("--precedence", default="diagnostics/editing_v2_split_precedence_resolution.json")
    parser.add_argument("--family-floor", type=float, default=0.05)
    parser.add_argument("--family-cap", type=float, default=0.22)
    parser.add_argument("--synthetic-target", type=float, default=0.18)
    parser.add_argument("--max-oversample", type=float, default=3.0,
                        help="How many times a real row may be repeated before a "
                             "family falls back to synthetic supply.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    precedence = json.loads(Path(args.precedence).read_text())
    excluded = set(precedence["excluded_source_keys"].get("train", []))
    print(f"held-out source exclusions applied to train: {len(excluded):,}")

    state_pairs = load_library(args.corpus_root, args.pairs_file)
    print(f"library state-keyed pairs: {len(state_pairs):,}")
    rows = attribute(Path(args.active8_root), state_pairs, excluded)
    print(f"eligible canonical train pairs: {len(rows):,}\n")

    plan = allocate(
        rows,
        floor=args.family_floor,
        cap=args.family_cap,
        synthetic_target=args.synthetic_target,
        max_oversample=args.max_oversample,
    )
    supply, draw, total = plan["supply"], plan["draw"], plan["total"]

    strata = []
    for (family, synthetic), share in sorted(draw.items()):
        count = supply.get((family, synthetic), 0)
        if share <= 0 or count == 0:
            continue
        strata.append({
            "family": family,
            "provenance": "synthetic" if synthetic else "real",
            "available_pairs": count,
            "draw_share": round(share, 6),
            "per_row_weight": round(share / count, 12),
            "oversample_factor": round((share * total) / count, 4),
        })

    realized_family = collections.Counter()
    for s in strata:
        realized_family[s["family"]] += s["draw_share"]
    realized_synth = sum(s["draw_share"] for s in strata if s["provenance"] == "synthetic")
    available_synth = sum(1 for r in rows if r["synthetic"]) / max(total, 1)

    report = {
        "schema": "compose.editing_v2.training_sampling_law",
        "schema_version": 1,
        "status": "SAMPLING_LAW_EVIDENCE_ONLY_NO_AUTHORITY",
        "constraints": {
            "family_floor": args.family_floor,
            "family_cap": args.family_cap,
            "synthetic_target": args.synthetic_target,
        },
        "held_out_source_exclusions": len(excluded),
        "eligible_canonical_train_pairs": total,
        "available_composition": {
            "synthetic_share": round(available_synth, 4),
            "by_family": {
                f: round(sum(v for (fam, _), v in supply.items() if fam == f) / total, 4)
                for f in sorted({s["family"] for s in strata})
            },
        },
        "synthetic_floor_from_supply": round(
            sum(v for (f, syn), v in draw.items()
                if syn and supply.get((f, False), 0) == 0), 4),
        "synthetic_target_met": bool(realized_synth <= args.synthetic_target + 1e-9),
        "realized_coefficients": {
            "synthetic_share": round(realized_synth, 4),
            "by_family": {f: round(v, 4) for f, v in sorted(realized_family.items())},
        },
        "strata": strata,
    }
    body = json.dumps({k: v for k, v in report.items() if k != "frozen_sha256"}, sort_keys=True)
    report["frozen_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"{'family':22} {'avail':>8} {'realized':>9}   {'real/synth draw':>20}")
    for family in sorted(realized_family):
        av = report["available_composition"]["by_family"][family]
        rl = report["realized_coefficients"]["by_family"][family]
        r = next((s["draw_share"] for s in strata
                  if s["family"] == family and s["provenance"] == "real"), 0.0)
        y = next((s["draw_share"] for s in strata
                  if s["family"] == family and s["provenance"] == "synthetic"), 0.0)
        print(f"  {family:20} {100*av:7.2f}% {100*rl:8.2f}%   {100*r:8.2f}% / {100*y:6.2f}%")
    print(f"\nsynthetic: available {100*available_synth:.1f}%  ->  realized "
          f"{100*realized_synth:.1f}%  (target {100*args.synthetic_target:.0f}%)")
    print(f"frozen {report['frozen_sha256'][:16]}\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
