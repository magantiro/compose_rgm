#!/usr/bin/env python
"""Collapse the per-slice ENTRIES.json files into one gzipped JSON-lines file.

WHY
---
The training path reads ten fields per entry, spread across 4,764 compiled
slice files. On a local SSD that costs ~100 s and does not matter. On a network
volume, per-object latency dominates and the same read becomes the longest part
of container startup -- paid again on every container, and on a GPU container
paid at GPU prices.

One object of ~80 MB gzipped replaces 4,764 of them. ``load_corpus_training_
library`` prefers this file when it is present and falls back to the slice walk
otherwise, so nothing else changes.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
The collated tensors. Those live in the packed store, which is a different
problem with a different fix: fixed-size records for random row access. This
file carries only what the loader reads -- teacher fibers, encoded states and
the handful of metadata fields -- and drops ``successor_partition``, which the
training path never touches.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compose_v4.data.corpus_training_library import (  # noqa: E402
    CONSOLIDATED_FIELDS,
    CONSOLIDATED_FILENAME,
    ENTRIES_FILENAME,
)
from compose_v4.data.durable_path import require_durable_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-reapable", action="store_true")
    args = parser.parse_args()

    roots = [
        Path(r) if args.allow_reapable else require_durable_path(Path(r), role="chunk root")
        for r in args.root
    ]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / CONSOLIDATED_FILENAME

    files: list[Path] = []
    for root in roots:
        files.extend(sorted(root.rglob(ENTRIES_FILENAME)))
    if not files:
        raise SystemExit("no ENTRIES.json found under any root; refusing to write")
    print(f"slice files     {len(files):,}")

    started = time.perf_counter()
    written = 0
    digest = hashlib.sha256()
    with gzip.open(target, "wt") as sink:
        for position, path in enumerate(files):
            for entry in json.loads(path.read_text())["entries"]:
                missing = [f for f in CONSOLIDATED_FIELDS if f not in entry]
                if missing:
                    raise SystemExit(f"{path}: entry is missing {missing}")
                line = json.dumps(
                    {field: entry[field] for field in CONSOLIDATED_FIELDS},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                sink.write(line + "\n")
                digest.update(line.encode())
                written += 1
            if (position + 1) % 1000 == 0 or position + 1 == len(files):
                print(f"  {position + 1:5,}/{len(files):,}  {written:8,} entries  "
                      f"{time.perf_counter() - started:5.0f}s")

    size = target.stat().st_size
    print()
    print(f"entries         {written:,}")
    print(f"{CONSOLIDATED_FILENAME:15} {size / 1e6:.1f} MB gzipped")
    print(f"entries_sha256  {digest.hexdigest()}")
    print(f"wall            {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
