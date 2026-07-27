#!/usr/bin/env python3
"""Committed pre-Modal-launch gate.

Replaces the never-committed ``scripts/sanity_check.py`` (0 commits in history) that CLAUDE.md still
referenced. Runs the REAL required checks before any Modal launch and prints every provenance hash, so a
launch can never proceed from a dirty tree, an unexpected commit, or an unverified corpus. Exits nonzero
on any failure.

Checks:
- code working tree clean (src/ scripts/ modal_apps/ tests/ have no uncommitted changes);
- HEAD is the expected commit/tag (if given);
- ruff clean on src/ (kept strict);
- full test suite green (unless --skip-tests, only for local iteration -- never for a launch);
- the required corpus file exists on the Modal volume;
- prints commit + standardization + operator-registry + compiler hashes + the corpus id (the corpus
  SHA-256 and ring-catalog fingerprint are computed on Modal where the corpus/catalog live and pinned
  into the scaled manifest at mining time).

Usage:
    PYTHONPATH=src python scripts/prelaunch_gate.py --corpus guacamol_subset_500000_seed0.smiles \
        [--expected-commit <sha-or-tag>] [--volume guacamol] [--skip-tests]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_edit_data_manifest import _hash_sources  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
_CODE_PREFIXES = ("src/", "scripts/", "modal_apps/", "tests/")


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout.strip()


def _code_tree_clean() -> tuple[bool, list[str]]:
    dirty = [
        line for line in _git("status", "--porcelain").splitlines()
        if line[3:].startswith(_CODE_PREFIXES)
    ]
    return (not dirty, dirty)


def _corpus_on_volume(corpus: str, volume: str) -> bool | None:
    try:
        out = subprocess.run(["modal", "volume", "ls", volume],
                             capture_output=True, text=True, timeout=90)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return corpus in out.stdout if out.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="guacamol_subset_500000_seed0.smiles")
    parser.add_argument("--volume", default="guacamol")
    parser.add_argument("--expected-commit", default=None)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    commit = _git("rev-parse", "--short", "HEAD")

    clean, dirty = _code_tree_clean()
    if not clean:
        failures.append(f"code working tree not clean: {dirty}")
    if args.expected_commit and _git("rev-parse", args.expected_commit) != _git("rev-parse", "HEAD"):
        failures.append(f"HEAD {commit} != expected {args.expected_commit}")

    ruff = subprocess.run(["ruff", "check", "src/"], cwd=REPO, capture_output=True, text=True)
    if ruff.returncode != 0:
        failures.append("ruff check src/ failed")

    if not args.skip_tests:
        env = {"KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "1", "PYTHONPATH": "src"}
        tests = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                               cwd=REPO, capture_output=True, text=True,
                               env={**__import__("os").environ, **env})
        if tests.returncode != 0:
            failures.append("pytest tests/ failed")
        print("test-suite:", tests.stdout.strip().splitlines()[-1] if tests.stdout else "(no output)")

    corpus_ok = _corpus_on_volume(args.corpus, args.volume)
    if corpus_ok is None:
        failures.append(f"could not verify corpus on Modal volume '{args.volume}' (auth/network?)")
    elif not corpus_ok:
        failures.append(f"corpus {args.corpus} not found on Modal volume '{args.volume}'")

    print("\n=== provenance ===")
    print(f"  commit:               {commit}")
    print(f"  standardization_hash: {_hash_sources(['src/compose_v4/chem/molecular_graph.py', 'src/compose_v4/chem/state.py'])}")
    print(f"  operator_registry:    {_hash_sources(['src/compose_v4/rewrite/operators.py', 'src/compose_v4/rewrite/kernel.py', 'src/compose_v4/rewrite/tracelets.py', 'src/compose_v4/rewrite/factorized_fiber.py'])}")
    print(f"  compiler(corruption): {_hash_sources(['src/compose_v4/rewrite/source_corruption.py'])}")
    print(f"  compiler(mmp):        {_hash_sources(['scripts/build_analogue_trace_pool.py'])}")
    print(f"  corpus:               {args.corpus} on volume '{args.volume}' (SHA-256 + ring-catalog fingerprint pinned at mining time)")

    if failures:
        print("\nPRELAUNCH GATE: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nPRELAUNCH GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
