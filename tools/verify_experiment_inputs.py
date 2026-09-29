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


#: Schemas this tool understands. An unrecognized schema is REJECTED rather than
#: skipped: a verifier that returns success on a manifest it cannot read is
#: worse than no verifier, because the success is quoted.
SUPPORTED_SCHEMAS = frozenset({"experiment_task_manifest_v1"})

#: Accepted hash field names and their exact hex length. A truncated hash is
#: allowed only where the field NAME says it is truncated.
HASH_FIELDS = {"sha256": 64, "sha256_16": 16}


def sha256_hex(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def declared_hash(item: dict) -> tuple[str, str] | str:
    """Return ``(field, value)`` or an error string.

    A missing or malformed hash is an ERROR, never a pass. The original version
    of this tool treated `expected is None` as "verified", so an input with no
    hash at all counted toward a green result.
    """
    present = [f for f in HASH_FIELDS if f in item]
    if not present:
        return (f"declares no hash (expected one of {sorted(HASH_FIELDS)})")
    if len(present) > 1:
        return f"declares several hashes {present}; exactly one is required"
    field = present[0]
    value = item[field]
    width = HASH_FIELDS[field]
    if not isinstance(value, str) or len(value) != width:
        return f"{field} must be {width} hex characters, got {value!r}"
    if any(c not in "0123456789abcdef" for c in value.lower()):
        return f"{field} is not hexadecimal: {value!r}"
    return field, value.lower()


def manifests(root: Path, only: str | None) -> list[Path]:
    found = sorted((root / EXPERIMENTS).glob("*/manifest.json"))
    if only:
        found = [p for p in found if p.parent.name == only]
    return found


def check(manifest_path: Path, root: Path) -> tuple[int, int, int]:
    try:
        payload = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        print(f"\n  {manifest_path.relative_to(root)}")
        print(f"    REJECTED  unreadable manifest: {error}")
        return 0, 0, 1

    task = payload.get("task", manifest_path.parent.name)
    print(f"\n  {task}  ({manifest_path.relative_to(root)})")

    schema = payload.get("schema_version")
    if schema not in SUPPORTED_SCHEMAS:
        print(f"    REJECTED  schema_version={schema!r} is not supported by this "
              f"tool.")
        print(f"              supported: {sorted(SUPPORTED_SCHEMAS)}")
        print("              Refusing to report success on a manifest whose "
              "shape is unknown;")
        print("              a green result here would mean nothing was checked.")
        return 0, 0, 1

    reported = payload.get("reported_result") or {}
    print(f"    reported result: {reported.get('status')}")
    if reported.get("reproduce"):
        print(f"    reproduce with:  {reported['reproduce']}")

    inputs = payload.get("inputs") or []
    if not inputs:
        print("    REJECTED  the manifest declares no inputs; there is nothing "
              "to verify,")
        print("              and '0 verified' must not read as a pass.")
        return 0, 0, 1

    failures = drift = ok = 0
    for item in inputs:
        if "path" not in item:
            print(f"    REJECTED  an input entry has no 'path': {item!r}")
            failures += 1
            continue
        path = root / item["path"]
        strength = item.get("pin", "informational")

        resolved = declared_hash(item)
        if isinstance(resolved, str):
            print(f"    REJECTED  {item['path']}: {resolved}")
            failures += 1
            continue
        field, expected = resolved

        if not path.exists():
            print(f"    MISSING   {item['path']}  ({item.get('role','')})")
            failures += 1
            continue

        full = sha256_hex(path)
        if full is None:
            print(f"    UNREADABLE {item['path']}")
            failures += 1
            continue
        actual = full if field == "sha256" else full[:16]

        if actual == expected:
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
    if total_ok == 0:
        print("NOTHING WAS VERIFIED. That is a failure, not a pass -- a run that "
              "checks\nzero inputs cannot tell you the inputs are intact.")
        return 1
    if total_fail:
        print("A strict pin moved or an input is missing -- the recorded results "
              "were not produced from what is on disk now.")
    print("=" * 78)
    return 1 if total_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
