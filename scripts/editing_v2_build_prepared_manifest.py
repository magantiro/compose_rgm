"""Realize the frozen sampling law as a deterministic training manifest.

    frozen law  ->  ordered entry-ID sequence  +  shard index  +  coefficient proof

WHY A MANIFEST AND NOT A SUBSET
-------------------------------
The frozen law is NOT "these 144,870 pairs are eligible". It carries nonuniform
realized coefficients -- real chemistry oversampled up to 3x where a family is
short, synthetic drawn only where it is the sole supply. A preparation step that
emits an eligibility list and lets the trainer sample uniformly would silently
discard the entire law while looking correct: same rows, same count, different
probability measure.

So this emits a deterministic MULTISET. A stratum drawn at twice its available
share appears twice as often in the sequence. The realized coefficients are then
recomputable by counting the sequence, which is exactly what the gate does
before any GPU time is spent.

SAMPLING IS GLOBAL; SHARDING IS STORAGE
---------------------------------------
The entry sequence is produced as if the whole corpus were resident, then a
shard index records where each row physically lives. Loading groups a
minibatch's IDs by shard purely to avoid re-reading files.

Sharding must never become the first sampling unit. Drawing a shard and then a
row inside it distorts the global law unless shard probabilities are derived
perfectly from their total sampling mass, and that derivation is easy to get
subtly wrong. Keeping the law strictly upstream of storage makes the question
not arise.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import random
from pathlib import Path

from compose_v4.data.durable_path import require_durable_path

SYNTHETIC_LANE = "reversible_synthetic_walk"


def index_library(corpus_roots: list[Path]) -> dict[str, dict]:
    """Map every compiled entry id to its physical shard location.

    The location is (root, task, slice) rather than a byte offset because a
    slice is the unit the corpus is published in, so it is also the natural
    unit to read back.
    """

    index: dict[str, dict] = {}
    for root_position, root in enumerate(corpus_roots):
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
                    index[str(entry["p50_entry_sha256"])] = {
                        "root": root_position,
                        "task": task.name,
                        "slice": slice_dir.name,
                        "state": str(entry["source_state_sha256"]),
                        "successor": str(entry["successor_canonical_key"]),
                        "family": str(entry["model_family"]),
                        "cell": str(entry.get("capability_cell_id", "?")),
                    }
    return index


def attribute_sources(active8_root: Path, index: dict[str, dict]) -> int:
    """Attach the canonical source SMILES and lane to each indexed entry.

    Compiled entries carry ``source_state_sha256`` only, and both the split
    exclusions and the synthetic/real distinction are keyed on the canonical
    source, so the stream is the only place the two can be joined.
    """

    by_state: dict[str, list[dict]] = collections.defaultdict(list)
    for record in index.values():
        by_state[record["state"]].append(record)
    resolved = 0
    for task_dir in sorted((Path(active8_root) / "tasks").iterdir()):
        stream = task_dir / "transitions.jsonl.gz"
        if not stream.exists():
            continue
        with gzip.open(stream, "rt") as handle:
            for line in handle:
                record = json.loads(line)
                evidence = record["candidate_evidence"]
                targets = by_state.get(str(evidence["source_state_sha256"]))
                if not targets:
                    continue
                successor = str(evidence["canonical_successor_key"])
                for target in targets:
                    if target.get("source") or target["successor"] != successor:
                        continue
                    target["source"] = str(evidence["source_canonical_key"])
                    target["synthetic"] = str(record["data_lane"]) == SYNTHETIC_LANE
                    resolved += 1
    return resolved


def realize(
    eligible: list[tuple[str, dict]],
    strata: dict[tuple[str, bool], float],
    *,
    epoch_size: int,
    seed: int,
    cell_floor: int = 256,
) -> list[str]:
    """Expand the law's stratum shares into a deterministic ID sequence.

    CAPABILITY-CELL PROTECTION
    --------------------------
    A family-level law can extinguish a capability cell outright. MEASURED:
    ``atom_restate:valence_state_change`` has 55 pairs, all in the synthetic
    lane, and the law draws atom_restate 21.35% real / 0.00% synthetic because
    real supply already meets the family target -- so that declared capability
    received ZERO training mass while every family coefficient looked perfect
    and the drift was +0.00000.

    The rule encoded here is not "special-case those 55 rows" but: no declared
    capability cell with available support may receive zero mass. Any cell
    below ``min(available, cell_floor)`` is topped up, and the same number of
    draws is removed from the LARGEST stratum OF THE SAME FAMILY, so family
    coefficients are preserved exactly and only the provenance split moves.
    """

    by_stratum: dict[tuple[str, bool], list[str]] = collections.defaultdict(list)
    for entry_id, record in eligible:
        by_stratum[(record["family"], bool(record["synthetic"]))].append(entry_id)
    rng = random.Random(seed)
    sequence: list[str] = []
    for stratum in sorted(by_stratum, key=lambda s: (s[0], s[1])):
        share = strata.get(stratum, 0.0)
        if share <= 0:
            continue
        draws = int(round(share * epoch_size))
        pool = sorted(by_stratum[stratum])
        if not pool:
            continue
        # Repetition is how an oversampled stratum reaches its share. Whole
        # passes first, then a seeded sample for the remainder, so the multiset
        # is reproducible from (seed, epoch_size) alone.
        whole, remainder = divmod(draws, len(pool))
        sequence.extend(pool * whole)
        if remainder:
            sequence.extend(rng.sample(pool, remainder))
    # ---- capability-cell floors, compensated within the family ----
    by_cell: dict[str, list[str]] = collections.defaultdict(list)
    family_of: dict[str, str] = {}
    stratum_of: dict[str, tuple[str, bool]] = {}
    for entry_id, record in eligible:
        by_cell[record["cell"]].append(entry_id)
        family_of[entry_id] = record["family"]
        stratum_of[entry_id] = (record["family"], bool(record["synthetic"]))
    drawn = collections.Counter(sequence)
    for cell in sorted(by_cell):
        pool = sorted(by_cell[cell])
        present = sum(1 for entry_id in pool if drawn[entry_id])
        target = min(len(pool), cell_floor)
        if present >= target:
            continue
        needed = [entry_id for entry_id in pool if not drawn[entry_id]][: target - present]
        if not needed:
            continue
        family = family_of[needed[0]]
        # Remove an equal number of draws from the same family's largest
        # stratum so the family coefficient does not move.
        family_counts: collections.Counter = collections.Counter()
        for entry_id in sequence:
            if family_of.get(entry_id) == family:
                family_counts[stratum_of[entry_id]] += 1
        if not family_counts:
            continue
        donor = family_counts.most_common(1)[0][0]
        removable = [
            position
            for position, entry_id in enumerate(sequence)
            if stratum_of.get(entry_id) == donor
        ][: len(needed)]
        for position in reversed(removable):
            sequence.pop(position)
        sequence.extend(needed)
    rng.shuffle(sequence)
    return sequence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--active8-root", required=True)
    parser.add_argument("--corpus-root", action="append", default=[])
    parser.add_argument("--epoch-size", type=int, default=0,
                        help="0 = one draw per eligible pair")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--cell-floor", type=int, default=256,
                        help="Minimum draws per capability cell with support.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--index-out", default="")
    args = parser.parse_args()

    freeze = json.loads(Path(args.freeze).read_text())
    active8 = require_durable_path(args.active8_root, role="active8 root")
    roots = [require_durable_path(r, role="corpus root") for r in args.corpus_root]

    index = index_library(roots)
    print(f"indexed entries: {len(index):,}")
    resolved = attribute_sources(active8, index)
    print(f"attributed: {resolved:,}")

    excluded = set(
        json.loads(Path(args.freeze).read_text())["split"].get("excluded_train_sources", [])
    )
    if not excluded:
        # The freeze stores counts; the key list lives in the precedence artifact.
        precedence = Path(args.freeze).parent / "editing_v2_split_precedence_resolution.json"
        excluded = set(json.loads(precedence.read_text())["excluded_source_keys"]["train"])
    print(f"held-out source exclusions: {len(excluded):,}")

    # ONE ENTRY PER CANONICAL PAIR.
    #
    # A compiled entry is finer-grained than a canonical (x, y) pair: several
    # entries can carry the same pair. Sampling entries directly would give such
    # a pair extra probability mass purely because it was compiled more than
    # once, which is exactly the artificial mass the corpus was deduplicated to
    # remove. Measured here: 151,078 compiled entries over 144,886 canonical
    # pairs.
    #
    # The representative is the lexicographically smallest entry id, so the
    # choice is deterministic and independent of filesystem or root ordering.
    by_pair: dict[tuple[str, str], tuple[str, dict]] = {}
    for entry_id, record in sorted(index.items()):
        if not record.get("source"):
            continue
        key = (record["source"], record["successor"])
        if key not in by_pair:
            by_pair[key] = (entry_id, record)
    print(f"canonical pairs in library: {len(by_pair):,}")

    eligible = [
        (entry_id, record)
        for (source, _), (entry_id, record) in sorted(by_pair.items())
        if source not in excluded
    ]
    print(f"eligible after exclusions: {len(eligible):,}")

    strata = {
        (s["family"], s["provenance"] == "synthetic"): float(s["draw_share"])
        for s in json.loads(
            (Path(args.freeze).parent / "editing_v2_training_sampling_law.json").read_text()
        )["strata"]
    }
    epoch = args.epoch_size or len(eligible)
    sequence = realize(eligible, strata, epoch_size=epoch, seed=args.seed,
                       cell_floor=args.cell_floor)

    # ---- coefficient proof: the law must survive translation ----
    lookup = dict(eligible)
    realized_family: collections.Counter = collections.Counter()
    realized_prov = collections.Counter()
    realized_cell: collections.Counter = collections.Counter()
    for entry_id in sequence:
        record = lookup[entry_id]
        realized_family[record["family"]] += 1
        realized_prov["synthetic" if record["synthetic"] else "real"] += 1
        realized_cell[record["cell"]] += 1
    n = len(sequence)
    expected_family: collections.Counter = collections.Counter()
    for (family, _), share in strata.items():
        expected_family[family] += share
    drift = {
        f: round(realized_family[f] / n - expected_family[f], 5)
        for f in sorted(expected_family)
    }
    worst = max(abs(v) for v in drift.values()) if drift else 0.0
    synthetic_realized = realized_prov["synthetic"] / n
    synthetic_expected = sum(v for (_, syn), v in strata.items() if syn)

    manifest_body = {
        "schema": "compose.editing_v2.prepared_training_manifest",
        "schema_version": 1,
        "library_sha256": freeze["library_sha256"],
        "split_sha256": freeze["split_sha256"],
        "sampling_law_sha256": freeze["sampling_law_sha256"],
        "seed": args.seed,
        "epoch_size": n,
        "eligible_entries": len(eligible),
        "excluded_sources": len(excluded),
        "cell_floor": args.cell_floor,
        "capability_cell_realized": {c: realized_cell[c] for c in sorted(realized_cell)},
        "coefficient_proof": {
            "family_expected": {f: round(expected_family[f], 5) for f in sorted(expected_family)},
            "family_realized": {f: round(realized_family[f] / n, 5) for f in sorted(realized_family)},
            "family_drift": drift,
            "worst_family_drift": worst,
            "synthetic_expected": round(synthetic_expected, 5),
            "synthetic_realized": round(synthetic_realized, 5),
            "capability_cells": len(realized_cell),
        },
    }
    manifest_body["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest_body, sort_keys=True).encode()
    ).hexdigest()

    Path(args.out).write_text(json.dumps(manifest_body, indent=2, sort_keys=True) + "\n")
    if args.index_out:
        Path(args.index_out).write_text(json.dumps({
            "manifest_sha256": manifest_body["manifest_sha256"],
            "sequence": sequence,
            "shards": {
                entry_id: [lookup[entry_id]["root"], lookup[entry_id]["task"],
                           lookup[entry_id]["slice"]]
                for entry_id in set(sequence)
            },
        }))

    print(f"\nepoch entries {n:,}  cells {len(realized_cell)}")
    print(f"synthetic expected {100*synthetic_expected:.2f}%  realized "
          f"{100*synthetic_realized:.2f}%")
    print(f"worst family drift {worst:+.5f}")
    print(f"manifest {manifest_body['manifest_sha256'][:16]}\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
