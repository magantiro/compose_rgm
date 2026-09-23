"""Check whether each sha256 pin still addresses the bytes it was written against.

A pin is a reproducibility link: it is how a published number gets back to the
code that produced it. This repository has three recorded instances of a pin
that addressed nothing and READ AS VERIFIED rather than failing loudly, which is
the worst case, because a chain verifier that cannot resolve an edge tends to
downgrade the finding to a warning instead of a failure.

This audit takes only the unambiguous shape: a mapping entry whose KEY is a
repo-relative path and whose VALUE is a 64-hex digest. For each such entry the
file's current bytes are hashed and compared.

Three outcomes, and the distinction between the last two is the whole point:

``resolves``
    Current bytes hash to the pinned value. The link is live.

``absent``
    The pinned path is not in the repository at all. The pin addresses nothing.

``stale``
    The path exists and hashes to something else.

A ``stale`` result is NOT automatically a defect. Most pins in this tree record
the state of a file at the moment a run was launched, and the file has moved on
since under normal development; that is exactly what an immutable launch record
is supposed to do, and re-pointing such a pin would forge the record. The
signals worth acting on are ``absent`` pins, and ``stale`` pins inside artifacts
that are supposed to describe the CURRENT tree rather than a past launch.

Because ``stale`` is expected, the report also records, per pinning artifact, how
many of its pins resolve. An artifact at 0/N resolving is a launch record; an
artifact at (N-1)/N is far more interesting, because one file moved out from
under a contract that otherwise still describes the tree.

Usage::

    python3 tools/repo_pin_resolution_audit.py --out diagnostics/repo_hygiene/x.json
    python3 tools/repo_pin_resolution_audit.py --out x.json --artifact configs/foo.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys
from collections import defaultdict
from typing import Any

HEX64 = re.compile(r"^[0-9a-f]{64}$")
SCAN_DIRECTORIES = ("configs", "diagnostics", "modal_apps", "tools", "scripts", "recipes", "docs")
PATH_SUFFIXES = (".py", ".json", ".md", ".txt", ".yaml", ".yml", ".jsonl", ".pt", ".csv", ".smiles")


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _iter_mappings(node: Any):
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            yield current
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)


def collect_pins(root: pathlib.Path) -> list[tuple[str, str, str]]:
    """Return ``(artifact, pinned_path, pinned_digest)`` for the unambiguous shape."""
    pins: list[tuple[str, str, str]] = []
    for directory in SCAN_DIRECTORIES:
        base = root / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.json")):
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, UnicodeDecodeError):
                continue
            artifact = str(path.relative_to(root))
            for mapping in _iter_mappings(document):
                for key, value in mapping.items():
                    if (
                        isinstance(key, str)
                        and isinstance(value, str)
                        and HEX64.match(value)
                        and key.endswith(PATH_SUFFIXES)
                        and not key.startswith("/")
                    ):
                        pins.append((artifact, key, value))
    return pins


def build(root: pathlib.Path) -> dict[str, Any]:
    tracked = set(
        subprocess.run(
            ["git", "ls-files"], cwd=str(root), capture_output=True, text=True, check=True
        ).stdout.split("\n")
    ) - {""}

    digests: dict[str, str] = {}

    def _digest(path: pathlib.Path) -> str | None:
        key = str(path)
        if key in digests:
            return digests[key]
        if not path.is_file():
            return None
        value = hashlib.sha256(path.read_bytes()).hexdigest()
        digests[key] = value
        return value

    def resolve(artifact: str, relative: str) -> tuple[str | None, str]:
        """Hash ``relative`` under each convention this tree actually uses.

        Pinned paths are written under two conventions: repo-root-relative in
        launch contracts, and relative to the artifact's OWN directory in
        per-run receipts (``attempt_1/case_0/failure.json``, ``../../src/...``).
        Treating every pin as repo-relative reports the second kind as absent,
        which would manufacture a large false finding about broken provenance.
        """
        candidates = (
            ("repo_relative", root / relative),
            ("artifact_relative", (root / artifact).parent / relative),
        )
        for rule, path in candidates:
            try:
                resolved = path.resolve()
            except OSError:
                continue
            digest = _digest(resolved)
            if digest is not None:
                return digest, rule
        return None, "unresolved"

    pins = collect_pins(root)
    per_artifact: dict[str, dict[str, int]] = defaultdict(
        lambda: {"resolves": 0, "stale": 0, "absent": 0}
    )
    absent: list[dict[str, str]] = []
    counts = {"resolves": 0, "stale": 0, "absent": 0}
    rules: dict[str, int] = defaultdict(int)

    for artifact, relative, pinned in pins:
        current, rule = resolve(artifact, relative)
        rules[rule] += 1
        if current is None:
            outcome = "absent"
            absent.append(
                {
                    "artifact": artifact,
                    "pinned_path": relative,
                    "pinned_sha256": pinned,
                    "tracked": relative in tracked,
                }
            )
        elif current == pinned:
            outcome = "resolves"
        else:
            outcome = "stale"
        counts[outcome] += 1
        per_artifact[artifact][outcome] += 1

    # An artifact whose pins ALMOST all resolve is the interesting case: it still
    # describes the current tree except for the file that moved out from under it.
    near_miss = []
    for artifact, tally in per_artifact.items():
        total = sum(tally.values())
        broken = tally["stale"] + tally["absent"]
        if total >= 3 and tally["resolves"] >= 1 and 1 <= broken <= max(2, total // 5):
            near_miss.append(
                {
                    "artifact": artifact,
                    "total_pins": total,
                    "resolves": tally["resolves"],
                    "stale": tally["stale"],
                    "absent": tally["absent"],
                }
            )
    near_miss.sort(key=lambda row: (-row["resolves"], row["artifact"]))

    fully_live = sorted(
        a for a, t in per_artifact.items() if t["resolves"] >= 1 and not t["stale"] and not t["absent"]
    )

    absent_paths = sorted({row["pinned_path"] for row in absent})
    return {
        "schema_version": 1,
        "baseline_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(root), capture_output=True, text=True, check=True
        ).stdout.strip(),
        "interpretation": (
            "A stale pin is usually an immutable launch record and re-pointing it would forge "
            "that record. Act on absent pins, and on artifacts that are near-miss."
        ),
        "resolution_rule_counts": dict(sorted(rules.items())),
        "pin_count": len(pins),
        "distinct_pinned_paths": len({p for _, p, _ in pins}),
        "pinning_artifact_count": len(per_artifact),
        "counts": counts,
        "absent_pin_count": counts["absent"],
        "absent_distinct_paths": absent_paths,
        "absent_examples": absent[:60],
        "fully_live_artifact_count": len(fully_live),
        "near_miss_artifact_count": len(near_miss),
        "near_miss_artifacts": near_miss[:60],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = repo_root()
    report = build(root)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, list)}, indent=2))
    print(f"absent distinct paths: {len(report['absent_distinct_paths'])}")
    for path in report["absent_distinct_paths"][:25]:
        print(f"  ABSENT {path}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
