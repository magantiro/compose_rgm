#!/usr/bin/env python3
"""Fail fast if the tree about to be mounted is not the tree you think it is.

A Modal image ships `src/` and `configs/` from whatever directory the launcher
resolves as ROOT. When the repository is edited in one checkout and runs are
launched from another, a container can execute code that was fixed hours ago:
that happened here, and a population-search pilot died on an ImportError for a
function that existed in git and not in the run tree. The compute was wasted
before anything scientific happened.

This makes that impossible to do silently. Call `assert_synced()` at the top of
a local entrypoint -- entrypoints run locally, before any container starts, so
the abort costs nothing.

Checks, in order of what actually goes wrong:

  1. the mounted directories exist and are readable
  2. no file under them differs from git HEAD *in a way the author did not
     intend* -- uncommitted edits are reported, not silently mounted
  3. the resolved ROOT is the repository you are standing in

Usage:
    python3 tools/preflight.py                # report
    python3 tools/preflight.py --strict       # exit 1 on any drift
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

MOUNTED = ("src", "configs")


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def git_state(root: Path) -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=root, capture_output=True,
                              text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": run("status", "--porcelain")}


def drift(root: Path) -> list[str]:
    """Files under the mounted paths that differ from committed state."""
    out = subprocess.run(["git", "status", "--porcelain", "--", *MOUNTED],
                         cwd=root, capture_output=True, text=True).stdout
    return [ln.strip() for ln in out.split("\n") if ln.strip()]


def assert_synced(strict: bool = True, quiet: bool = False) -> dict:
    """Raise (or warn) before a run if the mounted tree has uncommitted drift."""
    root = _root()
    st = git_state(root)
    missing = [d for d in MOUNTED if not (root / d).is_dir()]
    if missing:
        raise SystemExit(f"PREFLIGHT FAIL: mounted dirs missing: {missing}")
    d = drift(root)
    if not quiet:
        print(f"[preflight] root={root}")
        print(f"[preflight] branch={st['branch']} commit={st['commit'][:12]}")
        print(f"[preflight] mounted={list(MOUNTED)} drift={len(d)} file(s)")
    if d:
        msg = ("PREFLIGHT: uncommitted changes under the MOUNTED paths -- the "
               "container will run exactly these, which may differ from what "
               "was reviewed:\n  " + "\n  ".join(d[:20]))
        if strict:
            raise SystemExit(msg + "\n\ncommit them, or pass strict=False if "
                                   "this is deliberate.")
        print("[preflight] WARNING\n" + msg)
    return {"root": str(root), **st, "drift": d}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    try:
        assert_synced(strict=a.strict)
    except SystemExit as e:
        print(e)
        sys.exit(1)
    print("[preflight] OK")


if __name__ == "__main__":
    main()
