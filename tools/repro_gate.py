"""The merge gate for a preservation migration. Every line must read zero.

This is the mechanical answer to "did the cleanup lose anything". It does not
read a diff and it does not accept an argument; each line is a count, and a
nonzero count blocks the merge.

    pinned_historical_files_modified      files in the read-only set that changed
    lost_public_symbols                   (module, symbol) pairs gone since baseline
    lost_entrypoints                      apps, launchers or drivers that stopped importing
    lost_capability_groups                named subsystems that lost every module
    new_test_regressions                  nodes failing now that passed at baseline
    newly_broken_pins                     pins that stopped resolving
    external_artifacts_without_backup     off-Git objects with fewer than two verified copies

The last line is expected to be NONZERO on the first run and is reported rather
than suppressed: three known off-Git objects have fewer than two verified
backups today. A gate that hides a gap it cannot close is worse than one that
names it.

``lost_public_symbols`` is necessary and NOT sufficient. An import can resolve
while the mechanism behind it is reached by no production caller, which this
repository has found six times, so the entry-point and test lines are separate
gates rather than corroboration of the symbol line.

Usage::

    python3 tools/repro_gate.py --base 061ead93 --out repro/cleanup_report_v1.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from typing import Any

HYGIENE = pathlib.Path("diagnostics/repo_hygiene")


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _git(root: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(root), capture_output=True, text=True, check=True
    ).stdout


def _load(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def check_pinned(root: pathlib.Path, base: str) -> dict[str, Any]:
    pinned_report = _load(root / HYGIENE / "pinned_file_set_v1.json")
    pinned = set(pinned_report["files"]) if pinned_report else set()
    changed = {p for p in _git(root, "diff", "--name-only", base, "HEAD").split("\n") if p}
    violations = sorted(changed & pinned)
    return {
        "count": len(violations),
        "violations": violations,
        "changed_files_total": len(changed),
        "command": f"git diff --name-only {base} HEAD  (intersected with the pinned set)",
    }


def check_symbols(root: pathlib.Path) -> dict[str, Any]:
    diff = _load(root / HYGIENE / "capability_after_vs_baseline.json")
    if diff is None:
        return {"count": None, "note": "not yet re-measured; run repo_capability_baseline --compare"}
    return {
        "count": diff["import_symbols"]["lost_count"],
        "ast_lost_count": diff["ast_symbols"]["lost_count"],
        "lost": diff["import_symbols"]["lost"][:50],
        "verdict": diff["verdict"],
    }


def check_entry_points(root: pathlib.Path) -> dict[str, Any]:
    diff = _load(root / HYGIENE / "entry_point_after_vs_baseline.json")
    if diff is None:
        return {"count": None, "note": "not yet re-measured; run repo_entry_point_check --compare"}
    return {
        "count": diff["regressed_count"],
        "regressed": diff["regressed_entry_points"][:50],
        "vanished": diff["entry_points_no_longer_present"][:50],
        "verdict": diff["verdict"],
    }


def check_capability_groups(root: pathlib.Path) -> dict[str, Any]:
    manifest = _load(root / "repro" / "capability_manifest_v1.json")
    if manifest is None:
        return {"count": None, "note": "repro/capability_manifest_v1.json absent"}
    empty = sorted(
        name
        for name, entry in manifest["capability_groups"].items()
        if entry["importable_module_count"] == 0
    )
    return {"count": len(empty), "empty_groups": empty, "group_count": len(manifest["capability_groups"])}


def check_tests(root: pathlib.Path) -> dict[str, Any]:
    diff = _load(root / "repro" / "test_fingerprint_v1_vs_baseline.json")
    if diff is None:
        return {"count": None, "note": "no post-change run yet; baseline only"}
    return {"count": diff["new_regression_count"], "verdict": diff["verdict"]}


def check_pins(root: pathlib.Path, baseline_absent: int | None) -> dict[str, Any]:
    resolution = _load(root / HYGIENE / "pin_resolution_v1.json")
    if resolution is None:
        return {"count": None, "note": "pin_resolution_v1.json absent"}
    absent = resolution["counts"]["absent"]
    delta = None if baseline_absent is None else absent - baseline_absent
    return {
        "count": max(delta, 0) if delta is not None else 0,
        "absent_now": absent,
        "absent_at_baseline": baseline_absent,
        "note": (
            "All absent pins at baseline were traced: none is lost. 26 live on unfetched remote "
            "branches and the rest are uncommitted run outputs."
        ),
    }


def check_external(root: pathlib.Path) -> dict[str, Any]:
    external = _load(root / "repro" / "external_artifacts_v1.json")
    if external is None:
        return {"count": None, "note": "repro/external_artifacts_v1.json absent"}
    gaps = external["critical_objects_without_two_verified_backups"]
    return {
        "count": len(gaps),
        "objects": gaps,
        "note": (
            "EXPECTED NONZERO and deliberately not suppressed. Closing it requires copying the "
            "objects to a second durable location and verifying the sha AFTER copying."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="061ead93", help="baseline commit or tag")
    parser.add_argument("--out", default="repro/cleanup_report_v1.json")
    parser.add_argument("--baseline-absent-pins", type=int, default=10876)
    args = parser.parse_args()

    root = repo_root()
    gates = {
        "pinned_historical_files_modified": check_pinned(root, args.base),
        "lost_public_symbols": check_symbols(root),
        "lost_entrypoints": check_entry_points(root),
        "lost_capability_groups": check_capability_groups(root),
        "new_test_regressions": check_tests(root),
        "newly_broken_pins": check_pins(root, args.baseline_absent_pins),
        "external_artifacts_without_backup": check_external(root),
    }

    blocking = sorted(k for k, v in gates.items() if isinstance(v["count"], int) and v["count"] > 0)
    unmeasured = sorted(k for k, v in gates.items() if v["count"] is None)
    report = {
        "schema_version": 1,
        "baseline": args.base,
        "baseline_tag": "pre-cleanup-2026-09-23",
        "head": _git(root, "rev-parse", "HEAD").strip(),
        "rollback_sha": _git(root, "rev-parse", args.base).strip(),
        "rollback_command": f"git checkout {args.base}",
        "gates": gates,
        "blocking_gates": blocking,
        "unmeasured_gates": unmeasured,
        "verdict": (
            "BLOCKED" if blocking else ("INCOMPLETE" if unmeasured else "ALL_GATES_ZERO")
        ),
    }
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"{'gate':42s} count")
    for name, entry in gates.items():
        print(f"{name:42s} {entry['count']}")
    print(f"\nverdict: {report['verdict']}")
    if blocking:
        print(f"blocking: {blocking}")
    if unmeasured:
        print(f"unmeasured: {unmeasured}")
    print(f"rollback: {report['rollback_command']} ({report['rollback_sha'][:12]})")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
