"""Check every experiment task manifest against the files on disk.

    python3 tools/verify_experiment_inputs.py
    python3 tools/verify_experiment_inputs.py --task pmo

Each ``experiments/<task>/manifest.json`` lists the inputs a task depends on.
This confirms they are present and reports whether their bytes still match what
the manifest recorded.

TWO PIN STRENGTHS, because they mean different things
-----------------------------------------------------
``strict``
    A frozen artifact or a fixed input.  A mismatch is a FAILURE: the thing the
    recorded results were produced from is not the thing on disk.

``informational``
    Live code that is expected to move.  A mismatch is reported as DRIFT and is
    not a failure.  Recording it is still worth doing -- it dates the manifest
    and tells you which parts of the runtime have changed since.

A file that is MISSING is a failure at either strength: an absent input cannot
be verified, and silently continuing is how a manifest stops describing
anything.  Assets the manifest itself declares missing (``missing_assets``) are
reported separately and are never failures -- they are documented gaps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = "experiments"


def sha256_16(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()[:16]


def manifests(root: Path, only: str | None) -> list[Path]:
    found = sorted((root / EXPERIMENTS).glob("*/manifest.json"))
    if only:
        found = [p for p in found if p.parent.name == only]
    return found


def check(manifest_path: Path, root: Path) -> tuple[int, int, int]:
    payload = json.loads(manifest_path.read_text())
    task = payload.get("task", manifest_path.parent.name)
    print(f"\n  {task}  ({manifest_path.relative_to(root)})")

    reported = payload.get("reported_result") or {}
    print(f"    reported result: {reported.get('status')}")
    if reported.get("reproduce"):
        print(f"    reproduce with:  {reported['reproduce']}")

    failures = drift = ok = 0
    for item in payload.get("inputs") or []:
        path = root / item["path"]
        strength = item.get("pin", "informational")
        if not path.exists():
            print(f"    MISSING   {item['path']}  ({item.get('role','')})")
            failures += 1
            continue
        actual = sha256_16(path)
        expected = item.get("sha256_16")
        if expected is None or actual == expected:
            ok += 1
            continue
        if strength == "strict":
            print(f"    CHANGED   {item['path']}")
            print(f"              strict pin {expected} but file is {actual}")
            print(f"              role: {item.get('role','')}")
            failures += 1
        else:
            print(f"    drift     {item['path']}  {expected} -> {actual}")
            drift += 1

    missing = payload.get("missing_assets") or []
    for asset in missing:
        print(f"    absent by design: {asset.get('what')}")
        print(f"              not needed for: {asset.get('not_needed_for')}")

    print(f"    -> {ok} verified, {drift} drifted (informational), "
          f"{failures} FAILED")
    return ok, drift, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify experiment task manifests against files on disk."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--task", help="restrict to one task directory name")
    args = parser.parse_args(argv)

    root = args.repo_root.resolve()
    found = manifests(root, args.task)
    if not found:
        where = f"experiments/{args.task}/" if args.task else "experiments/*/"
        print(f"FAIL: no manifest.json under {where}", file=sys.stderr)
        return 2

    print("=" * 78)
    print("EXPERIMENT INPUT VERIFICATION")
    print("=" * 78)

    total_ok = total_drift = total_fail = 0
    for manifest_path in found:
        ok, drift, failures = check(manifest_path, root)
        total_ok += ok
        total_drift += drift
        total_fail += failures

    print("\n" + "=" * 78)
    print(f"{total_ok} verified, {total_drift} drifted (informational), "
          f"{total_fail} FAILED")
    if total_fail:
        print("A strict pin moved or an input is missing -- the recorded results "
              "were not produced from what is on disk now.")
    print("=" * 78)
    return 1 if total_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
