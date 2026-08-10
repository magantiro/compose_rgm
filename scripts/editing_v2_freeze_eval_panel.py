#!/usr/bin/env python
"""Freeze the held-out evaluation panel for the final R_theta run.

The panel is drawn EXCLUSIVELY from the frozen held-out validation role. It is
never cut from the training library, however convenient that would be: a panel
sharing sources with training cannot support a held-out claim, and the failure
is invisible in every metric it produces.

WHY THE WHOLE ROLE, NOT A SAMPLE
--------------------------------
Checkpoint selection needs teacher-successor NLL and minimum probability
(``_criterion``), and both come from ``forward_teacher_successor_batch``, which
reads only the teacher fiber -- a median of ONE alias. Scoring all 14,140
held-out entries is therefore cheap, so there is no sampling decision to
defend and every entry in the thin capability cells is retained. Rank and top-1
would need full successor partitions, which the chunk library does not store;
they are diagnostics, not selection inputs, and are out of scope here.

THE PANEL IS NOT DISTRIBUTED LIKE TRAINING
------------------------------------------
MEASURED: cycle_insert is 36.33% of the panel against 6.50% of the training
draw, and cycle_attach 26.32% against 4.52%. A plain panel mean is therefore
NOT an estimate of held-out performance under the sampling law. This artifact
carries the training composition as deployment weights so both can be
reported, and refuses to imply that either alone is "the" number.

THIN CELLS ARE DECLARED, NOT SILENTLY GATED
-------------------------------------------
MEASURED: three capability cells hold 2, 4 and 8 entries. A per-cell mean over
2 examples is noise, and a checkpoint gate on it would fire or not fire at
random. Those cells are flagged so a gate can exclude or widen them
deliberately rather than trusting them by default.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compose_v4.data.corpus_training_library import load_corpus_training_library  # noqa: E402
from compose_v4.data.durable_path import require_durable_path  # noqa: E402

PRECEDENCE = ("final_test", "controller_validation", "validation", "train")
PANEL_ROLE = "validation"
#: Below this, a per-cell mean is not a usable gate signal.
THIN_CELL_ENTRIES = 30


def sources_by_role(active8_root: Path) -> dict[str, set[str]]:
    by_role: dict[str, set[str]] = collections.defaultdict(set)
    for task_dir in sorted((active8_root / "tasks").iterdir()):
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


def resolve(by_role: dict[str, set[str]]) -> dict[str, set[str]]:
    """Keep each source only in its highest-ranked role."""

    kept: dict[str, set[str]] = {}
    for index, role in enumerate(PRECEDENCE):
        claimed: set[str] = set()
        for higher in PRECEDENCE[:index]:
            claimed |= by_role.get(higher, set())
        kept[role] = by_role.get(role, set()) - claimed
    return kept


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active8-root", required=True)
    parser.add_argument("--panel-root", required=True,
                        help="chunk root holding the compiled held-out entries")
    parser.add_argument("--train-root", action="append", default=[],
                        help="training chunk roots, for the disjointness gate")
    parser.add_argument("--manifest", required=True,
                        help="prepared training manifest, for deployment weights")
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ids-out", default="")
    args = parser.parse_args()

    active8 = require_durable_path(Path(args.active8_root), role="active8 root")
    panel_root = require_durable_path(Path(args.panel_root), role="panel chunk root")
    train_roots = [require_durable_path(Path(r), role="training chunk root")
                   for r in args.train_root]
    manifest = json.loads(Path(args.manifest).read_text())
    freeze = json.loads(Path(args.freeze).read_text())

    resolution = json.loads(Path(args.freeze).with_name(
        "editing_v2_split_precedence_resolution.json"
    ).read_text())
    # The compiled held-out chunks predate the precedence resolution, so they
    # still contain sources that a HIGHER-ranked role claims. MEASURED: 3 of
    # them belong to final_test. Leaving those in the validation panel would
    # contaminate the final test set, and nothing downstream would notice.
    excluded_panel = set(resolution["excluded_source_keys"][PANEL_ROLE])
    excluded_train = set(resolution["excluded_source_keys"]["train"])
    panel = load_corpus_training_library(
        [panel_root], excluded_sources=excluded_panel, verify_state_roundtrip=True
    )
    print(f"panel entries dropped by {PANEL_ROLE} precedence exclusions: "
          f"{panel.excluded_entry_count:,}")
    train = load_corpus_training_library(
        train_roots, excluded_sources=excluded_train, verify_state_roundtrip=False
    )

    kept = resolve(sources_by_role(active8))
    panel_sources = {entry.source_key for entry in panel.entries}
    train_sources = {entry.source_key for entry in train.entries}
    panel_ids = {entry.entry_id for entry in panel.entries}
    train_ids = {entry.entry_id for entry in train.entries}

    families = collections.Counter(entry.model_family for entry in panel.entries)
    cells = collections.Counter(entry.capability_cell_id for entry in panel.entries)
    train_families = collections.Counter(entry.model_family for entry in train.entries)
    train_cells = collections.Counter(entry.capability_cell_id for entry in train.entries)

    # ---- predeclared gates; the artifact is not emitted unless all hold ----
    misrouted = {
        role: len(panel_sources & kept.get(role, set()))
        for role in PRECEDENCE
        if role != PANEL_ROLE
    }
    gates = {
        "panel_nonempty": len(panel) > 0,
        "every_source_in_validation_role":
            panel_sources.issubset(kept.get(PANEL_ROLE, set())),
        "no_source_in_another_role": not any(misrouted.values()),
        "zero_source_overlap_with_train": not (panel_sources & train_sources),
        "zero_entry_overlap_with_train": not (panel_ids & train_ids),
        "all_training_families_present":
            set(train_families).issubset(set(families)),
        "all_training_cells_present":
            set(train_cells).issubset(set(cells)),
        "manifest_binds_frozen_law":
            manifest["sampling_law_sha256"] == freeze["sampling_law_sha256"],
    }

    print(f"panel entries      {len(panel):,}   sources {len(panel_sources):,}")
    print(f"training entries   {len(train):,}   sources {len(train_sources):,}")
    print()
    for name, ok in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if any(misrouted.values()):
        print(f"\n  misrouted sources by role: "
              f"{ {r: n for r, n in misrouted.items() if n} }")

    thin = {cell: count for cell, count in cells.items() if count < THIN_CELL_ENTRIES}
    if thin:
        print(f"\nthin cells (< {THIN_CELL_ENTRIES} entries; a per-cell gate here is noise):")
        for cell, count in sorted(thin.items(), key=lambda kv: kv[1]):
            print(f"  {count:5,}  {cell}")

    if not all(gates.values()):
        print("\nREFUSING to emit: a predeclared gate failed.")
        return 1

    total = len(panel)
    train_total = len(train)
    body = {
        "schema": "compose.editing_v2.eval_panel",
        "schema_version": 1,
        "status": "FROZEN_EVAL_PANEL",
        "role": PANEL_ROLE,
        "entry_count": total,
        "source_count": len(panel_sources),
        "entries_dropped_by_precedence": panel.excluded_entry_count,
        "gates": gates,
        "library_sha256": freeze["library_sha256"],
        "split_sha256": freeze["split_sha256"],
        "sampling_law_sha256": freeze["sampling_law_sha256"],
        "training_manifest_sha256": manifest["manifest_sha256"],
        "panel_composition": {
            "family_counts": dict(sorted(families.items())),
            "family_share": {f: round(c / total, 5) for f, c in sorted(families.items())},
            "cell_counts": dict(sorted(cells.items())),
        },
        # The panel is NOT distributed like the training draw, so a plain mean
        # over it answers a different question than performance under the law.
        "deployment_weights": {
            "note": "training-draw shares; use to reweight panel means so they "
                    "estimate held-out performance UNDER THE SAMPLING LAW",
            "family_share": {f: round(c / train_total, 5)
                             for f, c in sorted(train_families.items())},
            "cell_share": {c: round(n / train_total, 5)
                           for c, n in sorted(train_cells.items())},
        },
        "thin_cells": {
            "threshold_entries": THIN_CELL_ENTRIES,
            "cells": dict(sorted(thin.items(), key=lambda kv: kv[1])),
            "note": "per-cell means over these are not usable gate signals",
        },
    }
    body["panel_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()
    ).hexdigest()
    Path(args.out).write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(f"\npanel_sha256 {body['panel_sha256']}")
    print(f"wrote {args.out}")

    if args.ids_out:
        Path(args.ids_out).write_text(json.dumps({
            "panel_sha256": body["panel_sha256"],
            "entry_ids": sorted(panel_ids),
        }))
        print(f"wrote {args.ids_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
