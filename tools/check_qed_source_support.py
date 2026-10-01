"""Fail closed if QED sources change benchmark metrics on graph conversion."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import rdkit

from compose_v4.experiments.qed_source_support import audit_qed_source

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = ROOT / "data/jin/qed_test.txt"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-count", type=int, default=800)
    parser.add_argument("--max-active-atoms", type=int, default=40)
    args = parser.parse_args()

    sources = args.sources.resolve()
    if not sources.is_file():
        parser.error(f"missing source file: {sources}")
    observed_sha256 = sha256(sources)
    if observed_sha256 != args.expected_sha256:
        parser.error(
            f"source hash mismatch for {sources}: {observed_sha256}, "
            f"expected {args.expected_sha256}"
        )
    rows = [line.strip() for line in sources.read_text().splitlines() if line.strip()]
    if len(rows) != args.expected_count:
        parser.error(f"{sources}: expected {args.expected_count} sources, found {len(rows)}")

    audited = [audit_qed_source(row, max_active_atoms=args.max_active_atoms) for row in rows]
    reasons = Counter(row.reason for row in audited if row.reason is not None)
    result = {
        "schema_version": "qed_source_support_v1",
        "source_path": str(sources),
        "source_sha256": observed_sha256,
        "code_revision": code_revision(),
        "code_sha256": {
            "support": sha256(ROOT / "src/compose_v4/experiments/qed_source_support.py"),
            "representation": sha256(ROOT / "src/compose_v4/chem/molecular_graph.py"),
            "script": sha256(Path(__file__)),
        },
        "configuration": {
            "expected_count": args.expected_count,
            "max_active_atoms": args.max_active_atoms,
            "fingerprint_radius": 2,
            "fingerprint_bits": 2048,
            "qed_tolerance": 1e-12,
        },
        "software": {"rdkit": rdkit.__version__},
        "counts": {
            "sources": len(rows),
            "supported": sum(row.supported for row in audited),
            "benchmark_equivalent": sum(row.benchmark_equivalent for row in audited),
            "isomeric_identity_retained": sum(row.isomeric_identity_retained for row in audited),
            "reasons": dict(sorted(reasons.items())),
        },
    }
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if all(row.supported and row.benchmark_equivalent for row in audited) else 1


if __name__ == "__main__":
    raise SystemExit(main())
