"""Reconcile the frozen sampling manifest against the physical compiled library.

WHY THIS AND NOT A COUNT
------------------------
Comparing "107,165 physical rows" with "107,872 recipe rows" and concluding we
are 707 short is the wrong arithmetic twice over. The recipe's figure is
POST-DEDUP, the physical figure was raw rows, and the physical rows contain
2,687 duplicate canonical pairs -- so the true physical count is 104,478
distinct pairs. Worse, two sets of the same size can still disagree completely;
only a set difference says what is actually absent.

    M = canonical (x, y) pairs the frozen manifest requires
    A = canonical (x, y) pairs physically compiled and on disk

    M \\ A  is the work that remains
    A \\ M  is compiled chemistry the recipe does not consume

A row is keyed by ``(source_canonical_key, canonical_successor_key)`` -- SMILES
on both sides. It is tempting to key on ``source_state_sha256`` instead, since
the stream and every compiled entry carry that field verbatim and the join then
needs no translation at all. That is WRONG, and quietly so: the state hash is
over the padded slot arrangement, so one molecule occupying different slots
hashes differently. MEASURED: identical in the two small real lanes (1.00x) but
**1.31x inflation in real_endpoint_multistep_path**, the lane that dominates the
corpus. Keying on it reported 145,995 required pairs where the frozen census,
keying canonically, reported 107,872.

Compiled entries carry only the state hash, so a ``state -> canonical`` map is
built from the streams (which carry both) and used to translate the library
side. A compiled entry whose state hash is absent from that map is counted and
reported rather than dropped.

Duplicates are collapsed on BOTH sides before differencing, because the
standing decision is that a canonical (x, y) duplicate must not create
artificial training probability mass.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path

SYNTHETIC_LANE = "reversible_synthetic_walk"


def pair_digest(source_state_sha256: str, successor_key: str) -> str:
    return hashlib.blake2b(
        f"{source_state_sha256}\t{successor_key}".encode(), digest_size=16
    ).hexdigest()


def available_from_corpus(root: Path, state_to_canonical: dict[str, str]) -> tuple[set[str], int]:
    """Pairs physically present in a local compiled corpus tree, keyed canonically.

    Returns the pair set and the count of entries whose source state hash had no
    canonical translation -- those are reported, never silently dropped.
    """

    present: set[str] = set()
    untranslated = 0
    chunks = Path(root) / "chunks"
    if not chunks.is_dir():
        return present, untranslated
    for task in sorted(chunks.iterdir()):
        if not task.is_dir():
            continue
        for slice_dir in sorted(task.iterdir()):
            entries = slice_dir / "ENTRIES.json"
            if not (slice_dir / "RECEIPT.json").exists() or not entries.exists():
                continue
            for entry in json.loads(entries.read_text())["entries"]:
                canonical = state_to_canonical.get(str(entry["source_state_sha256"]))
                if canonical is None:
                    untranslated += 1
                    continue
                present.add(
                    pair_digest(canonical, str(entry["successor_canonical_key"]))
                )
    return present, untranslated


def required_from_manifest(
    manifest: dict, active8_root: Path
) -> tuple[dict[str, dict], dict[str, str]]:
    """Pairs the frozen manifest requires, with the facts needed to break them down.

    Deduplicated globally and first-lane-wins, matching how the frozen census
    counted, so a pair shared by two lanes is one requirement rather than two.
    """

    required: dict[str, dict] = {}
    state_to_canonical: dict[str, str] = {}
    # The map must cover EVERY task on disk, not just the manifest's. The
    # compiled library spans lanes the V2 manifest never selects (notably
    # observed_local_analogue), and a compiled entry whose state cannot be
    # translated is dropped from A -- which overstates what is missing. First
    # measured run dropped 41,547 of train_65k's 70,301 entries exactly this way.
    lane_of_task = {
        task: lane
        for lane, tasks in manifest["selection"].items()
        for task in tasks
    }
    selected_of_task = {
        task: set(sources)
        for tasks in manifest["selection"].values()
        for task, sources in tasks.items()
    }
    tasks_root = Path(active8_root) / "tasks"
    for task_dir in sorted(tasks_root.iterdir()) if tasks_root.is_dir() else []:
        task = task_dir.name
        lane = lane_of_task.get(task)
        wanted = selected_of_task.get(task, set())
        for stream in [task_dir / "transitions.jsonl.gz"]:
            if not stream.exists():
                continue
            with gzip.open(stream, "rt") as handle:
                for line in handle:
                    record = json.loads(line)
                    evidence = record["candidate_evidence"]
                    if evidence.get("exclusion_reason") is not None:
                        continue
                    state_to_canonical[str(evidence["source_state_sha256"])] = str(
                        evidence["source_canonical_key"]
                    )
                    if lane is None or evidence["source_canonical_key"] not in wanted:
                        continue
                    digest = pair_digest(
                        str(evidence["source_canonical_key"]),
                        str(evidence["canonical_successor_key"]),
                    )
                    if digest in required:
                        continue
                    required[digest] = {
                        "lane": lane,
                        "task": task,
                        "family": record["model_family"],
                        "cell": record.get("capability_cell_id", "?"),
                        "role": record.get("partition_role", "?"),
                        "source": evidence["source_canonical_key"],
                    }
    return required, state_to_canonical


def _breakdown(rows: list[dict]) -> dict:
    lanes = collections.Counter(r["lane"] for r in rows)
    families = collections.Counter(r["family"] for r in rows)
    cells = collections.Counter(r["cell"] for r in rows)
    roles = collections.Counter(r["role"] for r in rows)
    synthetic = sum(1 for r in rows if r["lane"] == SYNTHETIC_LANE)
    return {
        "rows": len(rows),
        "by_lane": dict(lanes.most_common()),
        "by_family": dict(families.most_common()),
        "by_capability_cell": dict(cells.most_common()),
        "by_role": dict(roles),
        "synthetic_rows": synthetic,
        "real_rows": len(rows) - synthetic,
        "distinct_sources": len({r["source"] for r in rows}),
        "distinct_tasks": len({r["task"] for r in rows}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active8-root", required=True)
    parser.add_argument("--manifest", default="diagnostics/editing_v2_v2_selection_manifest.json")
    parser.add_argument("--corpus-root", action="append", default=[],
                        help="Local compiled corpus tree. Repeatable.")
    parser.add_argument("--pairs-file", action="append", default=[],
                        help="JSON list of 'source_state_sha256\\tsuccessor_canonical_key' "
                             "for a corpus not on this disk. Translated canonically "
                             "here, so unlike a digest list it can be re-keyed.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--missing-sources-out", default="",
                        help="Write the source keys of M\\A, grouped by task, ready "
                             "to drive a source-selective compile.")
    args = parser.parse_args()

    # The manifest scan comes FIRST because it builds the state -> canonical map
    # the library side needs. Every corpus must be on this disk: a precomputed
    # digest list cannot be re-keyed, and mixing keys silently reports a full
    # corpus as missing.
    manifest = json.loads(Path(args.manifest).read_text())
    required, state_to_canonical = required_from_manifest(manifest, Path(args.active8_root))
    print(f"REQUIRED by manifest: {len(required):,} distinct canonical pairs")
    print(f"state->canonical map: {len(state_to_canonical):,} states\n")

    available: set[str] = set()
    untranslated_total = 0
    for root in args.corpus_root:
        found, untranslated = available_from_corpus(Path(root), state_to_canonical)
        untranslated_total += untranslated
        print(f"corpus {root}: {len(found):,} distinct pairs"
              + (f"  ({untranslated:,} entries had no canonical translation)" if untranslated else ""))
        available |= found
    for path in args.pairs_file:
        raw = json.loads(Path(path).read_text())
        found = set()
        missing_translation = 0
        for item in raw:
            state, _, successor = str(item).partition("\t")
            canonical = state_to_canonical.get(state)
            if canonical is None:
                missing_translation += 1
                continue
            found.add(pair_digest(canonical, successor))
        untranslated_total += missing_translation
        print(f"pairs {path}: {len(found):,} distinct pairs"
              + (f"  ({missing_translation:,} untranslated)" if missing_translation else ""))
        available |= found
    print(f"AVAILABLE union: {len(available):,} distinct canonical pairs\n")

    missing = [v for k, v in required.items() if k not in available]
    unused = len(available - set(required))

    report = {
        "schema": "compose.editing_v2.manifest_library_reconciliation",
        "schema_version": 1,
        "status": "RECONCILIATION_EVIDENCE_ONLY_NO_AUTHORITY",
        "required_pairs": len(required),
        "available_pairs": len(available),
        "missing_pairs": len(missing),
        "available_not_required_pairs": unused,
        "entries_without_canonical_translation": untranslated_total,
        "missing": _breakdown(missing),
    }
    body = json.dumps(report, sort_keys=True).encode()
    report["frozen_sha256"] = hashlib.sha256(body).hexdigest()
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"MISSING (M \\ A): {len(missing):,} pairs")
    print(f"UNUSED  (A \\ M): {unused:,} pairs of compiled chemistry the recipe skips\n")
    if missing:
        b = report["missing"]
        print(f"  synthetic {b['synthetic_rows']:,}  real {b['real_rows']:,}  "
              f"sources {b['distinct_sources']:,}  tasks {b['distinct_tasks']}")
        print(f"  by lane   {b['by_lane']}")
        print(f"  by family {b['by_family']}")
        print(f"  by role   {b['by_role']}")

    if args.missing_sources_out:
        by_task: dict[str, set[str]] = collections.defaultdict(set)
        for row in missing:
            by_task[row["task"]].add(row["source"])
        Path(args.missing_sources_out).write_text(
            json.dumps(
                [{"task_identity_sha256": t, "sources": sorted(s)}
                 for t, s in sorted(by_task.items())],
                indent=2,
            ) + "\n"
        )
        print(f"\nwrote missing sources for {len(by_task)} tasks -> "
              f"{args.missing_sources_out}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
