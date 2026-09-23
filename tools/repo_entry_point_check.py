"""Prove every repository entry point still resolves and still imports.

An importable symbol table is not the whole capability surface. A Modal app that
cannot be launched is a lost capability even when every library symbol it used to
call still exists, and this repository's entry points are spread across three
directories with three different import conventions:

``modal_apps/``
    Cloud app definitions. Several compute ``ROOT = parents[1]`` and mount
    ``src/`` and ``configs/`` explicitly, so they are position-sensitive.

``tools/``
    Repository tooling and durable launchers.

``scripts/``
    Local experiment drivers. These rely on ``scripts`` being on ``PYTHONPATH``
    in addition to ``src``.

Each file is imported in its OWN subprocess. That is deliberate and costs wall
clock: a single shared process would let one entry point's import-time side
effect, ``sys.exit`` or hang destroy the measurement for every file after it, and
an entry point that hangs is exactly the kind of thing this check should report
rather than be killed by.

IMPORTING IS NOT SIDE-EFFECT FREE, AND THIS TOOL LEARNED THAT THE HARD WAY.
215 of this repository's entry points perform a module-level write with no
``if __name__ == "__main__"`` guard, so importing them RUNS them. A first run of
this check executed ``scripts/hphi_valid128_read.py``, which rewrote
``docs/VALID128_K8_RESULT.json`` and silently dropped its comparator block, panel
description and revision note: committed scientific record, destroyed by a
measurement that was supposed to observe. Nothing in the check noticed.

So the scan now brackets itself with ``git status --porcelain``. If the tree was
CLEAN beforehand, any path the scan touched is restored and reported. If the tree
was already dirty, nothing is touched, because telling someone else's
uncommitted work apart from a probe's side effect is not something this tool can
do, and guessing would be worse than reporting. Either way the mutated paths
appear in the report rather than being absorbed silently.

The comparison rule is a SUPERSET check on the set of entry points whose import
succeeds. An entry point that already fails at baseline is recorded with its
error so the failure is attributable and cannot later be mistaken for a
regression this pass introduced.

Usage::

    python3 tools/repo_entry_point_check.py --out diagnostics/repo_hygiene/x.json
    python3 tools/repo_entry_point_check.py --compare BASELINE.json --out AFTER.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

ENTRY_POINT_DIRECTORIES = ("modal_apps", "tools", "scripts")

# Bounded because several agents share this laptop; this repository's own
# measurement is that oversubscription took a build from 5,507 entries/h to 250.
MAX_CONCURRENT_SUBPROCESSES = 3
PER_FILE_TIMEOUT_SECONDS = 90

_IMPORT_PROBE = r"""
import importlib.util, sys, pathlib
path = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("_entry_point_probe_" + path.stem, path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
"""


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def discover_entry_points(root: pathlib.Path) -> list[str]:
    found: list[str] = []
    for directory in ENTRY_POINT_DIRECTORIES:
        base = root / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            found.append(str(path.relative_to(root)))
    return sorted(found)


def _probe(root: pathlib.Path, relative: str, interpreter: str) -> tuple[str, dict[str, Any]]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(root / "src"), str(root / "scripts")])
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["OMP_NUM_THREADS"] = "1"
    # Refuse to let a probe charge an oracle call or reach the network.
    env["COMPOSE_ENTRY_POINT_IMPORT_PROBE"] = "1"
    try:
        completed = subprocess.run(
            [interpreter, "-c", _IMPORT_PROBE, str(root / relative)],
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
            timeout=PER_FILE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return relative, {"import_ok": False, "error": "TimeoutExpired: import did not finish"}
    if completed.returncode == 0:
        return relative, {"import_ok": True, "error": None}
    tail = [line for line in completed.stderr.strip().splitlines() if line.strip()]
    return relative, {
        "import_ok": False,
        "error": (tail[-1] if tail else f"exit {completed.returncode}")[:240],
    }


def _worktree_state(root: pathlib.Path) -> set[str]:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(root), capture_output=True, text=True, check=False,
    )
    return {line for line in completed.stdout.splitlines() if line.strip()}


def _restore(root: pathlib.Path, entries: set[str]) -> list[str]:
    """Undo what the scan wrote. Only ever called when the tree started clean."""
    restored: list[str] = []
    for entry in sorted(entries):
        # Porcelain v1 is a fixed two-character status followed by a space, so the
        # path begins at index 3. Splitting on the first space instead yields
        # "M docs/..." for an unstaged modification, because such a line STARTS
        # with a space; the resulting git checkout then fails silently and the
        # file is reported restored while still being damaged.
        path = entry[3:].strip().strip('"')
        if not path:
            continue
        if entry.startswith("??"):
            target = root / path
            try:
                if target.is_file():
                    target.unlink()
                    restored.append(f"removed {path}")
            except OSError:
                restored.append(f"COULD NOT REMOVE {path}")
        else:
            subprocess.run(
                ["git", "checkout", "--", path], cwd=str(root), capture_output=True, check=False
            )
            restored.append(f"reverted {path}")
    return restored


def _assert_restored(root: pathlib.Path, before: set[str]) -> None:
    """A restore that reports success while leaving damage is the worst outcome."""
    residue = _worktree_state(root) - before
    if residue:
        raise RuntimeError(
            "entry-point scan could not restore what it touched: "
            + "; ".join(sorted(residue))
            + ". Restore by hand before trusting any result from this run."
        )


def build_report(root: pathlib.Path, interpreter: str) -> dict[str, Any]:
    before = _worktree_state(root)
    started_clean = not before
    entry_points = discover_entry_points(root)
    results: dict[str, Any] = {}
    with concurrent.futures.ThreadPoolExecutor(MAX_CONCURRENT_SUBPROCESSES) as pool:
        futures = [pool.submit(_probe, root, name, interpreter) for name in entry_points]
        for done in concurrent.futures.as_completed(futures):
            name, outcome = done.result()
            results[name] = outcome

    ok = sorted(n for n, r in results.items() if r["import_ok"])
    per_directory = {}
    for directory in ENTRY_POINT_DIRECTORIES:
        members = [n for n in results if n.startswith(directory + "/")]
        per_directory[directory] = {
            "total": len(members),
            "import_ok": sum(1 for n in members if results[n]["import_ok"]),
        }
    after = _worktree_state(root)
    touched = sorted(after - before)
    restored: list[str] = []
    if touched and started_clean:
        restored = _restore(root, set(touched))
    residue = sorted(_worktree_state(root) - before)

    return {
        "schema_version": 1,
        "interpreter": interpreter,
        "entry_point_count": len(results),
        "import_ok_count": len(ok),
        "per_directory": per_directory,
        "side_effects": {
            "note": (
                "Importing an entry point RUNS it when the module has no __main__ guard. "
                "215 entry points here perform a module-level write. Paths the scan touched "
                "are listed, and restored only when the tree started clean."
            ),
            "worktree_started_clean": started_clean,
            "paths_touched_by_the_scan": touched,
            "restored": restored,
            "unrestored_residue": residue,
            "clean_after": not residue,
        },
        "entry_points": {name: results[name] for name in sorted(results)},
    }


def compare(baseline: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_ok = {n for n, r in baseline["entry_points"].items() if r["import_ok"]}
    after_ok = {n for n, r in after["entry_points"].items() if r["import_ok"]}
    before_all = set(baseline["entry_points"])
    after_all = set(after["entry_points"])
    lost_ok = sorted(before_ok - after_ok)
    vanished = sorted(before_all - after_all)
    return {
        "baseline_entry_points": len(before_all),
        "after_entry_points": len(after_all),
        "baseline_import_ok": len(before_ok),
        "after_import_ok": len(after_ok),
        "regressed_entry_points": lost_ok,
        "regressed_count": len(lost_ok),
        "entry_points_no_longer_present": vanished,
        "newly_importable_count": len(after_ok - before_ok),
        "verdict": (
            "ENTRY_POINTS_PRESERVED" if not lost_ok and not vanished else "ENTRY_POINTS_REGRESSED"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--compare")
    parser.add_argument("--interpreter", default=sys.executable)
    args = parser.parse_args()

    root = repo_root()
    report = build_report(root, args.interpreter)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"entry_points={report['entry_point_count']} import_ok={report['import_ok_count']} "
        f"per_directory={report['per_directory']} -> {out}"
    )
    side = report["side_effects"]
    if side["paths_touched_by_the_scan"]:
        print(f"SIDE EFFECTS: the scan touched {len(side['paths_touched_by_the_scan'])} path(s)")
        for entry in side["paths_touched_by_the_scan"][:20]:
            print(f"   {entry}")
        for entry in side["restored"][:20]:
            print(f"   {entry}")
        if side["unrestored_residue"]:
            print("   UNRESTORED -- the tree was already dirty, so nothing was touched:")
            for entry in side["unrestored_residue"][:20]:
                print(f"     {entry}")

    if args.compare:
        baseline = json.loads(pathlib.Path(args.compare).read_text(encoding="utf-8"))
        diff = compare(baseline, report)
        diff_path = out.with_name(out.stem + "_vs_baseline.json")
        diff_path.write_text(json.dumps(diff, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(diff, indent=2)[:4000])
        print(f"-> {diff_path}")
        return 0 if diff["verdict"] == "ENTRY_POINTS_PRESERVED" else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
