#!/usr/bin/env python
"""Realize the post-split sampling law as a deterministic training sequence.

    law v2  ->  ordered entry-ID multiset  +  recounted coefficient proof

WHY A MULTISET, NOT AN ELIGIBILITY LIST
---------------------------------------
The law carries nonuniform coefficients: real chemistry oversampled up to 3x
where a family is short, synthetic drawn only where it is the sole supply. A
preparation step that emitted "these rows are eligible" and let the trainer
sample uniformly would discard the entire law while looking correct -- same
rows, same count, different probability measure. A stratum drawn at twice its
available share therefore appears twice in the sequence, and the realized
coefficients are RECOUNTED from the finished sequence rather than restated from
the law that produced it.

WHY NO SHARD INDEX
------------------
The original manifest recorded (root, task, slice) per entry because the corpus
was read from 4,764 slice files. Training now reads the row-addressable packed
store by entry id, and the corpus is one consolidated file, so a physical index
would be a second map to keep in sync with nothing reading it.

WHY LIBRARY-DERIVED
-------------------
Same reason as the law: the local Active8 diverges from the volume's, and
family / cell / lane are already carried by the objects the training loader
reads. ``realize`` is imported from the original builder, so the capability-cell
floor and its within-family compensation are the same code, not a restatement.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
SYNTHETIC_LANE = "reversible_synthetic_walk"


def _load_realize():
    spec = importlib.util.spec_from_file_location(
        "_editing_v2_manifest", HERE / "editing_v2_build_prepared_manifest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.realize


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--law", type=Path, required=True)
    parser.add_argument("--train-library", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--reserve-ids", type=Path, required=True)
    parser.add_argument("--epoch-size", type=int, default=0,
                        help="0 means one pass' worth: the eligible row count.")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--cell-floor", type=int, default=256)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--index-out", type=Path, required=True)
    args = parser.parse_args()

    realize = _load_realize()
    law = json.loads(args.law.read_text())
    with gzip.open(args.reserve_ids, "rt") as handle:
        training_ids = set(json.load(handle)["training_entry_ids"])
    with gzip.open(args.provenance, "rt") as handle:
        lane_by_entry = json.load(handle)["lane_by_entry_id"]

    eligible: list[tuple[str, dict]] = []
    seen: set[str] = set()
    with gzip.open(args.train_library, "rt") as handle:
        for line in handle:
            entry = json.loads(line)
            entry_id = str(entry["p50_entry_sha256"])
            if entry_id in seen or entry_id not in training_ids:
                continue
            seen.add(entry_id)
            lane = lane_by_entry.get(entry_id)
            if lane is None:
                continue
            eligible.append((entry_id, {
                "family": str(entry["model_family"]),
                "synthetic": lane == SYNTHETIC_LANE,
                "cell": str(entry["capability_cell_id"]),
            }))
    if len(eligible) != len(training_ids):
        raise SystemExit(
            f"{len(training_ids):,} training ids but {len(eligible):,} eligible "
            "rows; the manifest must cover the split exactly")
    epoch_size = args.epoch_size or len(eligible)

    strata = {(s["family"], s["provenance"] == "synthetic"): s["draw_share"]
              for s in law["strata"]}
    sequence = realize(eligible, strata, epoch_size=epoch_size, seed=args.seed,
                       cell_floor=args.cell_floor)

    # RECOUNT from the finished sequence. The law states intent; this measures
    # what the sequence will actually deliver, including whatever the
    # capability-cell floor moved.
    record_of = dict(eligible)
    family_counts: collections.Counter = collections.Counter()
    synthetic_draws = 0
    for entry_id in sequence:
        record = record_of[entry_id]
        family_counts[record["family"]] += 1
        synthetic_draws += bool(record["synthetic"])
    drawn = collections.Counter(sequence)
    realized = {f: c / len(sequence) for f, c in family_counts.items()}
    intended = law["realized_coefficients"]["by_family"]
    drift = {f: round(realized[f] - intended.get(f, 0.0), 6) for f in sorted(realized)}

    cells: dict[str, list[str]] = collections.defaultdict(list)
    for entry_id, record in eligible:
        cells[record["cell"]].append(entry_id)
    starved = sorted(
        cell for cell, pool in cells.items()
        if not any(drawn[entry_id] for entry_id in pool))
    unused = sum(1 for entry_id, _ in eligible if not drawn[entry_id])

    # WHICH rows the law never draws, not just how many. The sequence is fixed,
    # so a row absent from it is excluded permanently rather than merely unseen
    # this epoch -- and a bare count reads as rounding when it is not. MEASURED:
    # the synthetic strata of atom_delete, atom_insert, atom_restate and
    # bond_reroute are drawn at or near ZERO, because those families' real
    # supply already meets their target. Evaluating on those strata measures a
    # capability the law deliberately declines to train.
    never: collections.Counter = collections.Counter()
    available: collections.Counter = collections.Counter()
    for entry_id, record in eligible:
        key = f"{record['family']}|{'synthetic' if record['synthetic'] else 'real'}"
        available[key] += 1
        if not drawn[entry_id]:
            never[key] += 1
    never_drawn_by_stratum = {
        key: {"available": available[key], "never_drawn": never[key],
              "never_drawn_fraction": round(never[key] / available[key], 4)}
        for key in sorted(available) if never[key]
    }
    zero_mass_strata = sorted(
        key for key in available if never[key] == available[key])

    gates = {
        "every_family_within_0p005_of_law": all(abs(v) <= 0.005 for v in drift.values()),
        "no_capability_cell_starved": not starved,
        "synthetic_share_within_0p005_of_law": abs(
            synthetic_draws / len(sequence)
            - law["realized_coefficients"]["synthetic_share"]) <= 0.005,
        "sequence_covers_only_training_ids": set(sequence) <= training_ids,
        "law_bound_to_this_split": bool(
            law.get("matched_validation_reserve_source_digest")),
    }

    manifest = {
        "schema": "compose.editing_v2.prepared_training_manifest",
        "schema_version": 2,
        "status": "PREPARED_SEQUENCE_EVIDENCE_ONLY_NO_AUTHORITY",
        "law_frozen_sha256": law["frozen_sha256"],
        "matched_validation_reserve_source_digest":
            law.get("matched_validation_reserve_source_digest"),
        "seed": args.seed,
        "epoch_size": epoch_size,
        "cell_floor": args.cell_floor,
        "eligible_rows": len(eligible),
        "sequence_length": len(sequence),
        "distinct_entries_drawn": len(drawn),
        "eligible_rows_never_drawn": unused,
        "never_drawn_by_stratum": never_drawn_by_stratum,
        "zero_mass_strata": zero_mass_strata,
        "zero_mass_note": (
            "These strata receive NO training mass under this law, because "
            "their family's real supply already meets its target. The "
            "sequence is fixed, so this is permanent exclusion, not slow "
            "exposure -- more epochs will not reach them. Any evaluation "
            "view that includes them measures a capability the law "
            "deliberately declines to train, and must weight them by the "
            "LAW's draw share rather than the library's composition."),
        "recounted_coefficients": {
            "synthetic_share": round(synthetic_draws / len(sequence), 6),
            "by_family": {f: round(v, 6) for f, v in sorted(realized.items())},
        },
        "drift_from_law": drift,
        "starved_capability_cells": starved,
        "gates": gates,
        "sequence_sha256": hashlib.sha256("\n".join(sequence).encode()).hexdigest(),
    }
    body = json.dumps({k: v for k, v in manifest.items() if k != "frozen_sha256"},
                      sort_keys=True)
    manifest["frozen_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    args.index_out.write_text(json.dumps({"sequence": sequence}))

    print(f"eligible {len(eligible):,}  sequence {len(sequence):,}  "
          f"distinct drawn {len(drawn):,}  never drawn {unused:,}")
    print(f"\n{'family':22} {'law':>9} {'recounted':>10} {'drift':>9}")
    for family in sorted(realized):
        print(f"  {family:20} {100*intended.get(family, 0):8.2f}% "
              f"{100*realized[family]:9.2f}% {100*drift[family]:+8.3f}%")
    print(f"\nsynthetic: law {100*law['realized_coefficients']['synthetic_share']:.1f}%"
          f"  ->  recounted {100*synthetic_draws/len(sequence):.1f}%")
    for name, passed in sorted(gates.items()):
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"\nfrozen {manifest['frozen_sha256'][:16]}\n"
          f"wrote {args.out} and {args.index_out}")
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
