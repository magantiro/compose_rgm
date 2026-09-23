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


def build_report(root: pathlib.Path, interpreter: str) -> dict[str, Any]:
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
    return {
        "schema_version": 1,
        "interpreter": interpreter,
        "entry_point_count": len(results),
        "import_ok_count": len(ok),
        "per_directory": per_directory,
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
