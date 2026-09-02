#!/usr/bin/env python3
"""Reference-graph audit: which files can be moved or archived SAFELY.

The 2026-08-19 reorganisation moved nothing, on the grounds that moving a
referenced file breaks the tree. That was right, but it left the repository
unorganised. The way out is not to move things more bravely -- it is to know,
per file, whether anything points at it.

A file is REFERENCED if its path or module name appears in any tracked source,
config, doc or notebook. Modal apps additionally reference paths through
`add_local_file(ROOT / "...")`, which is how a checkpoint under `docs/` can be
load-bearing even though nothing imports it. That case is checked explicitly,
because missing it is exactly how a "safe" cleanup breaks a run.

Usage:
    python3 tools/repo_audit.py                 # summary
    python3 tools/repo_audit.py --unreferenced  # movable candidates
    python3 tools/repo_audit.py --check PATH    # who references one path
"""
from __future__ import annotations

import argparse
import re
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".json", ".toml", ".yaml", ".yml", ".cfg",
                 ".txt", ".ipynb", ".sh"}
SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules"}


def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                         text=True).stdout.split("\n")
    return [ROOT / p for p in out if p]


def scan_text(paths: list[Path]) -> str:
    chunks = []
    for p in paths:
        if p.suffix.lower() not in TEXT_SUFFIXES or not p.exists():
            continue
        try:
            chunks.append(p.read_text(errors="ignore"))
        except Exception:
            continue
    return "\n".join(chunks)


def mention_index(paths: list[Path]) -> tuple[set, set]:
    """Index every path-like and module-like token that appears in the tree.

    One pass over the corpus building sets, instead of one substring scan per
    file over the whole corpus -- the latter is quadratic and does not finish
    on a repository this size.
    """
    PATHY = re.compile(r"[A-Za-z0-9_./-]+\.[A-Za-z0-9]{1,6}")
    DOTTED = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{2,}")
    names: set[str] = set()
    dotted: set[str] = set()
    for p in paths:
        if p.suffix.lower() not in TEXT_SUFFIXES or not p.exists():
            continue
        try:
            t = p.read_text(errors="ignore")
        except Exception:
            continue
        self_rel = str(p.relative_to(ROOT))
        for m in PATHY.findall(t):
            if m != self_rel and Path(m).name != p.name:
                names.add(m)
                names.add(Path(m).name)
        for m in DOTTED.findall(t):
            dotted.add(m)
            dotted.add(m.split(".")[-1])
    return names, dotted


def is_referenced(rel: str, names: set, dotted: set) -> bool:
    p = Path(rel)
    if rel in names or p.name in names:
        return True
    if rel.endswith(".py"):
        if rel[:-3].replace("/", ".") in dotted or p.stem in dotted:
            return True
    return False


def classify(rel: str, referenced: bool) -> str:
    """What a file IS, which decides whether "unreferenced" means anything.

    A Modal app is invoked as `modal run modal_apps/x.py::entry`; nothing
    imports it, and pytest discovers tests by convention rather than by
    reference. Treating either as dead because no file points at it is the
    fastest way to delete working code, so entrypoints and tests are named
    explicitly and never fall into ORPHAN.
    """
    # Conventional files are load-bearing by convention, not by reference.
    if rel in {".gitignore", "LICENSE", "README.md", "pyproject.toml",
               "CLAUDE.md", "AGENTS.md"} or Path(rel).name == "LICENSE":
        return "CONVENTION"
    top = rel.split("/")[0]
    if top in {"diagnostics", "results", "artifacts", "upload"}:
        return "OUTPUT"
    if top.startswith("paper") or top == "docs":
        return "DOC"
    if top == "tests":
        return "TEST"
    if top == "third_party":
        return "VENDORED"
    if rel.endswith(".py"):
        try:
            t = (ROOT / rel).read_text(errors="ignore")
        except Exception:
            t = ""
        if "local_entrypoint" in t or "__main__" in t or "argparse" in t:
            return "ENTRYPOINT"
        return "LIBRARY" if referenced else "ORPHAN"
    return "DATA" if referenced else "ORPHAN"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unreferenced", action="store_true")
    ap.add_argument("--classify", action="store_true")
    ap.add_argument("--check")
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    files = tracked_files()
    names, dotted = mention_index(files)

    if args.check:
        rel = args.check
        hits = []
        for p in files:
            if p.suffix.lower() not in TEXT_SUFFIXES or not p.exists():
                continue
            try:
                t = p.read_text(errors="ignore")
            except Exception:
                continue
            if rel in t or Path(rel).name in t:
                hits.append(p.relative_to(ROOT))
        print(f"{rel}: {len(hits)} referencing file(s)")
        for h in hits[: args.top]:
            print("  ", h)
        return

    by_dir: dict[str, list[int]] = defaultdict(list)
    unref: list[str] = []
    for p in files:
        rel = str(p.relative_to(ROOT))
        ok = is_referenced(rel, names, dotted)
        by_dir[rel.split("/")[0]].append(1 if ok else 0)
        if not ok:
            unref.append(rel)

    if args.classify:
        from collections import Counter
        kinds = Counter()
        orphans = []
        for p in files:
            rel = str(p.relative_to(ROOT))
            k = classify(rel, is_referenced(rel, names, dotted))
            kinds[k] += 1
            if k == "ORPHAN":
                orphans.append(rel)
        for k, n in kinds.most_common():
            print(f"{k:<12} {n}")
        print(f"\nORPHAN candidates (no inbound reference, not an entrypoint, "
              f"test, doc or output): {len(orphans)}")
        for rel in sorted(orphans)[: args.top]:
            print("  ", rel)
        return

    if args.unreferenced:
        print(f"{len(unref)} tracked files with no inbound reference\n")
        for rel in sorted(unref)[: args.top]:
            print("  ", rel)
        return

    print(f"tracked files: {len(files)}   unreferenced: {len(unref)}\n")
    print(f"{'directory':<28} {'files':>6} {'unreferenced':>13}")
    for d in sorted(by_dir, key=lambda d: -len(by_dir[d])):
        v = by_dir[d]
        print(f"{d:<28} {len(v):>6} {sum(1 for x in v if x == 0):>13}")


if __name__ == "__main__":
    main()
