"""Inventory what git does NOT protect, and hash it so a backup can be verified.

WHY THIS EXISTS
---------------
A clean ``git status`` and a pushed branch protect TRACKED files and nothing
else.  They say nothing about ignored checkpoints, downloaded oracle assets, or
campaign outputs sitting on one disk.  This repo has already lost a worktree to a
``/private/tmp`` reap and has had a 70,301-entry corpus exist in exactly one
place; in both cases the git state looked perfectly healthy.

    python3 tools/preservation_inventory.py                    # report
    python3 tools/preservation_inventory.py --write-manifest   # report + manifest

The manifest is written to ``diagnostics/preservation_inventory_v1.json`` and is
meant to be COMMITTED.  A report printed to a terminal, or a file in a temp
directory, protects nothing -- which is the whole point of the exercise.

THE THREE CLASSES
-----------------
``tracked``
    In git.  Replication is then a question about remotes, which this script
    also answers for the current branch.

``regenerable``
    Not in git, but reconstructible by a RECORDED recipe, which is printed with
    the class.  A file only belongs here if the recipe is named and the inputs
    it needs are themselves tracked.  Claiming something is regenerable without
    naming how is how assets get lost.

``at_risk``
    Not in git and no recipe.  These are hashed individually.  If it is not in
    a backup, it exists once.

WHAT THE HASHES ARE FOR
-----------------------
Not integrity of the repo -- git already does that for tracked files.  They let
you verify that a COPY somewhere else is the same bytes, which is the only thing
that makes a backup a backup.  Verify one with ``--verify <dir>``.

WHAT THIS DOES NOT DO
---------------------
It never deletes, moves or uploads anything.  It reads and reports.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = Path("diagnostics/preservation_inventory_v1.json")

# Never walked: build caches and virtualenvs, none of which are assets.
SKIP_DIRS = {
    ".git", ".uv-cache", ".uv-tools", ".venv", "__pycache__",
    ".pytest_cache", ".ruff_cache", "node_modules",
}

# (glob, class-name, why, recipe).  Order matters: first match wins.
# A file is only 'regenerable' when the recipe is NAMED and its inputs are
# tracked.  Anything else falls through to at_risk.
REGENERABLE_RULES: tuple[tuple[str, str, str, str], ...] = (
    (
        "diagnostics/*/source_capsule/*",
        "source_capsule",
        "Verbatim copy of every file a run read, 12-17 MB per attempt.",
        "python3 tools/rebuild_source_capsule.py <source_capsule_manifest*.json> --out <dir>",
    ),
    (
        "diagnostics/*/source_capsule_v*/*",
        "source_capsule",
        "Verbatim copy of every file a run read, 12-17 MB per attempt.",
        "python3 tools/rebuild_source_capsule.py <source_capsule_manifest*.json> --out <dir>",
    ),
    (
        "diagnostics/*/campaign/round_*/complete.json",
        "cumulative_round_snapshot",
        ("Each round's snapshot CONTAINS every earlier round, so a K-round "
         "campaign stores itself K times."),
        ("Not regenerated: the load-bearing records (oracle/query_*/result.json "
         "receipts, manifests, progress reports) are committed beside it."),
    ),
    (
        "diagnostics/*/campaign/round_*/pending.json",
        "cumulative_round_snapshot",
        "Un-charged batch snapshot, same cumulative shape as complete.json.",
        "Not regenerated: no charged call is recorded only here.",
    ),
    (
        "oracle/*.pkl",
        "upstream_oracle_asset",
        "Upstream PyTDC predictor pickles (drd2/gsk3b/jnk3), ~71 MB.",
        ("PyTDC downloads these on first use into the directory "
         "AssetPinnedOracle pins around every call."),
    ),
    (
        "diagnostics/pmo_ivg_oracle_parity/ivg_oracle_assets/*",
        "upstream_oracle_asset",
        "Comparator oracle assets kept for the parity gate only.",
        "Re-fetched by the parity gate; not an input to any COMPOSE result.",
    ),
    (
        "diagnostics/*/dryrun/*",
        "zero_oracle_gate_workspace",
        ("Campaign state from a gate that charged no oracle calls. Its own "
         "committed report is the artifact worth keeping; the workspace is "
         "rebuilt from scratch on every run."),
        ("Re-run the gate. NOTE its ledger is shaped like a scored run but its "
         "scores are NOT production scores -- tools/reproduce_pmo_tables.py "
         "excludes it on that evidence."),
    ),
    (
        "*.log",
        "run_log",
        "Worker stdout/stderr from a completed run.",
        "Not regenerated; the run's own artifacts carry its results.",
    ),
)

HASH_CHUNK = 1 << 20


# ---- Git ----


def git_tracked(root: Path) -> set[str] | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            capture_output=True, check=True, timeout=180,
        ).stdout.decode()
    except (OSError, subprocess.SubprocessError):
        return None
    return {p for p in out.split("\0") if p}


def git_branch_replication(root: Path) -> dict:
    """Is the current branch on a remote, at this exact commit?

    ``for-each-ref refs/remotes`` enumerates LOCAL tracking refs, which exist
    only after a fetch created them, so a pushed branch nobody has fetched reads
    as absent.  ``ls-remote`` asks the remote.
    """
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True, check=True, timeout=180,
            ).stdout.decode().strip()
        except (OSError, subprocess.SubprocessError):
            return None

    head = run(["rev-parse", "HEAD"])
    branch = run(["rev-parse", "--abbrev-ref", "HEAD"])
    if not head or not branch:
        return {"status": "unknown", "reason": "not a git repository"}
    remote = run(["ls-remote", "--heads", "origin", branch])
    if remote is None:
        return {"branch": branch, "head": head, "status": "unreachable",
                "reason": "could not query origin"}
    if not remote:
        return {"branch": branch, "head": head, "status": "ABSENT_ON_REMOTE",
                "reason": f"origin has no branch {branch}"}
    remote_sha = remote.split()[0]
    if remote_sha == head:
        return {"branch": branch, "head": head, "remote_head": remote_sha,
                "status": "replicated"}
    behind = run(["rev-list", "--count", f"{remote_sha}..{head}"])
    return {"branch": branch, "head": head, "remote_head": remote_sha,
            "status": "AHEAD_OF_REMOTE",
            "unpushed_commits": int(behind) if behind and behind.isdigit() else None}


# ---- Classification ----


def git_dirty(root: Path) -> set[str]:
    """Tracked paths whose WORKING-TREE bytes are not what git stores.

    `git ls-files` lists tracked PATHS, which is not the same as saying the
    current content is preserved. A modified-but-uncommitted file is tracked and
    its present bytes exist nowhere else. Treating it as protected is how an
    edit gets lost between a clean-looking status and a reap.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "-z",
             "--untracked-files=no"],
            capture_output=True, check=True, timeout=180,
        ).stdout.decode()
    except (OSError, subprocess.SubprocessError):
        return set()
    dirty: set[str] = set()
    for record in out.split("\0"):
        if len(record) > 3:
            dirty.add(record[3:])
    return dirty


def recovery_dependency_ok(root: Path, relative: str, rule_name: str) -> str | None:
    """Check the thing a 'regenerable' claim actually depends on.

    A filename pattern is a guess about provenance, not evidence of recovery. A
    source capsule is only rebuildable while its manifest survives; a cumulative
    round snapshot is only droppable while the oracle receipts beside it exist.
    Where the dependency is absent the file is NOT regenerable and must fall
    through to at_risk.

    Returns None when the dependency holds, or a reason string when it does not.
    """
    path = root / relative
    if rule_name == "source_capsule":
        # The sibling manifest records git_blob_oid + sha256 for every entry and
        # is what tools/rebuild_source_capsule.py consumes.  A capsule is a
        # verbatim copy of a source tree, so its files nest arbitrarily deep --
        # anchor on the capsule DIRECTORY itself rather than walking a fixed
        # number of parents, which was wrong and demoted 5,136 recoverable files.
        parts = path.parts
        for index in range(len(parts) - 1, -1, -1):
            if parts[index].startswith("source_capsule"):
                capsule_parent = Path(*parts[:index])
                if any(capsule_parent.glob("source_capsule_manifest*.json")):
                    return None
                return (f"no source_capsule_manifest*.json in "
                        f"{capsule_parent.name or capsule_parent}")
        return "path does not contain a source_capsule component"
    if rule_name == "cumulative_round_snapshot":
        # campaign/round_N/<file> -> the ledger root is two levels up.
        campaign = path.parent.parent
        ledger = campaign.parent
        if (ledger / "oracle").is_dir() and any(
            (ledger / "oracle").glob("query_*/result.json")
        ):
            return None
        return "no oracle/query_*/result.json receipts beside it"
    if rule_name == "zero_oracle_gate_workspace":
        # The committed report beside it is the artifact worth keeping.
        for parent in list(path.parents)[:4]:
            if any(
                candidate.is_file() and candidate.name != "manifest.json"
                for candidate in parent.glob("*.json")
            ):
                return None
        return "no committed report found beside the dryrun workspace"
    if rule_name == "upstream_oracle_asset":
        # Redownloadable by definition; nothing local to depend on.
        return None
    if rule_name == "run_log":
        return None
    return None


def classify(relative: str) -> tuple[str, str, str] | None:
    for pattern, name, why, recipe in REGENERABLE_RULES:
        if fnmatch.fnmatch(relative, pattern):
            return name, why, recipe
    return None


def sha256_file(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(HASH_CHUNK), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def walk(root: Path, tracked: set[str], dirty: set[str]) -> dict:
    at_risk: list[dict] = []
    regenerable: dict[str, dict] = {}
    demoted: list[dict] = []
    tracked_count = tracked_bytes = 0
    dirty_count = 0

    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            path = Path(current) / name
            try:
                relative = str(path.relative_to(root))
            except ValueError:
                continue
            # In a git WORKTREE, .git is a FILE pointing at the parent repo, so
            # skipping directories by name does not catch it.
            if relative == ".git":
                continue
            # This tool's own output is not an asset at risk; before its first
            # commit it would otherwise report itself.
            if relative == str(MANIFEST):
                continue
            if relative in tracked and relative not in dirty:
                tracked_count += 1
                try:
                    tracked_bytes += path.stat().st_size
                except OSError:
                    pass
                continue
            if relative in dirty:
                # Tracked, but the bytes on disk are not the bytes git holds.
                dirty_count += 1
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                at_risk.append({"path": relative, "bytes": size,
                                "sha256": sha256_file(path),
                                "reason": "tracked but MODIFIED; current bytes "
                                          "are not committed anywhere"})
                continue
            if name.endswith(".pyc"):
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue

            rule = classify(relative)
            if rule is not None:
                name_, why, recipe = rule
                problem = recovery_dependency_ok(root, relative, name_)
                if problem is not None:
                    demoted.append({"path": relative, "rule": name_,
                                    "reason": problem})
                    at_risk.append({"path": relative, "bytes": size,
                                    "sha256": sha256_file(path),
                                    "reason": f"claimed {name_} but {problem}"})
                    continue
                bucket = regenerable.setdefault(
                    name_, {"why": why, "recipe": recipe, "files": 0, "bytes": 0}
                )
                bucket["files"] += 1
                bucket["bytes"] += size
                continue

            at_risk.append({"path": relative, "bytes": size,
                            "sha256": sha256_file(path)})

    at_risk.sort(key=lambda entry: -entry["bytes"])
    return {
        "tracked": {"files": tracked_count, "bytes": tracked_bytes},
        "regenerable": regenerable,
        "at_risk": at_risk,
        "demoted": demoted,
        "dirty_tracked": dirty_count,
    }


# ---- Verify ----


def verify(root: Path, backup: Path) -> int:
    """Check a backup against the SAVED MANIFEST, never against a fresh scan.

    Verifying against a freshly-walked inventory is worse than useless: a file
    that was recorded earlier and has since been DELETED is absent from the new
    scan, so it is never looked for, and the backup passes. The saved manifest
    is the only record of what was supposed to exist.
    """
    manifest_path = root / MANIFEST
    if not manifest_path.exists():
        print(f"FAIL: {MANIFEST} does not exist. Nothing to verify against.",
              file=sys.stderr)
        print("  Run with --write-manifest first, and commit the result.",
              file=sys.stderr)
        return 2
    try:
        saved = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        print(f"FAIL: cannot read {MANIFEST}: {error}", file=sys.stderr)
        return 2

    recorded = saved.get("at_risk") or []
    print(f"\nVerifying {len(recorded)} recorded at-risk file(s) against {backup}")
    print(f"  manifest written {saved.get('created_at_utc')}")

    missing = differing = matched = gone_locally = 0
    for entry in recorded:
        candidate = backup / entry["path"]
        if not (root / entry["path"]).exists():
            gone_locally += 1
        if not candidate.exists():
            missing += 1
            print(f"  ABSENT FROM BACKUP: {entry['path']}"
                  + ("   (and gone locally -- this file may be LOST)"
                     if not (root / entry["path"]).exists() else ""))
            continue
        if sha256_file(candidate) == entry["sha256"]:
            matched += 1
        else:
            differing += 1
            print(f"  DIFFERS: {entry['path']}")

    print(f"  matched {matched} / {len(recorded)};  absent from backup {missing};  "
          f"differing {differing}")
    if gone_locally:
        print(f"  {gone_locally} recorded file(s) no longer exist in the working "
              f"tree either.")
    return 0 if (missing == 0 and differing == 0) else 1


# ---- Entry point ----


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inventory files git does not protect. Reads only; never "
                    "deletes, moves or uploads."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--write-manifest", action="store_true",
                        help=f"write {MANIFEST} (commit it; a report protects nothing)")
    parser.add_argument("--verify", type=Path, metavar="DIR",
                        help="check a backup directory against the stored manifest")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args(argv)

    root = args.repo_root.resolve()
    tracked = git_tracked(root)
    if tracked is None:
        print(f"FAIL: {root} is not a git repository, or git is unavailable.",
              file=sys.stderr)
        print("  This tool answers 'what is NOT protected by git', which has no "
              "meaning\n  without a repository to compare against. Run it inside "
              "the working\n  tree, not in a source export or an unpacked "
              "archive.", file=sys.stderr)
        return 2

    print("=" * 78)
    print("PRESERVATION INVENTORY -- what git does NOT protect")
    print("=" * 78)
    print(f"  checkout: {root}")

    replication = git_branch_replication(root)
    print(f"  branch:   {replication.get('branch')} @ "
          f"{str(replication.get('head'))[:12]}")
    status = replication.get("status")
    if status == "replicated":
        print("  remote:   REPLICATED -- origin is at this exact commit")
    elif status == "AHEAD_OF_REMOTE":
        print(f"  remote:   AHEAD BY {replication.get('unpushed_commits')} COMMIT(S) "
              f"-- tracked work is NOT fully replicated")
    elif status == "ABSENT_ON_REMOTE":
        print("  remote:   ABSENT ON ORIGIN -- this branch exists on one disk")
    else:
        print(f"  remote:   {status} ({replication.get('reason')})")

    dirty = git_dirty(root)
    inventory = walk(root, tracked, dirty)

    tracked_stats = inventory["tracked"]
    print(f"\n  tracked:      {tracked_stats['files']:>6} files  "
          f"{tracked_stats['bytes'] / 1048576:>9.1f} MB   (committed AND unmodified)")
    if inventory.get("dirty_tracked"):
        print(f"  MODIFIED:     {inventory['dirty_tracked']:>6} files             "
              f"   tracked but uncommitted -- counted AT RISK below,")
        print("                                          because the bytes on "
              "disk are in no commit")

    regen_files = sum(b["files"] for b in inventory["regenerable"].values())
    regen_bytes = sum(b["bytes"] for b in inventory["regenerable"].values())
    print(f"  regenerable:  {regen_files:>6} files  {regen_bytes / 1048576:>9.1f} MB   "
          f"(recipe recorded, see below)")

    risk_bytes = sum(e["bytes"] for e in inventory["at_risk"])
    print(f"  AT RISK:      {len(inventory['at_risk']):>6} files  "
          f"{risk_bytes / 1048576:>9.1f} MB   (no recipe, not in git)")

    if inventory.get("demoted"):
        print(f"\n  DEMOTED TO AT-RISK ({len(inventory['demoted'])}) -- matched a "
              f"regenerable pattern but")
        print("  the thing its recovery actually depends on is missing:")
        seen = set()
        for item in inventory["demoted"]:
            key = (item["rule"], item["reason"])
            if key in seen:
                continue
            seen.add(key)
            count = sum(1 for d in inventory["demoted"]
                        if (d["rule"], d["reason"]) == key)
            print(f"    {item['rule']}: {item['reason']}  ({count} files)")

    if inventory["regenerable"]:
        print("\n  REGENERABLE CLASSES (recovery dependency verified present)")
        for name, bucket in sorted(inventory["regenerable"].items(),
                                   key=lambda kv: -kv[1]["bytes"]):
            print(f"    {name}  ({bucket['files']} files, "
                  f"{bucket['bytes'] / 1048576:.1f} MB)")
            print(f"      why:    {bucket['why']}")
            print(f"      recipe: {bucket['recipe']}")

    if inventory["at_risk"]:
        print(f"\n  AT-RISK FILES (top {args.top} by size)")
        for entry in inventory["at_risk"][:args.top]:
            digest = (entry["sha256"] or "UNREADABLE")[:16]
            print(f"    {entry['bytes'] / 1048576:>9.2f} MB  {digest}  {entry['path']}")
        if len(inventory["at_risk"]) > args.top:
            print(f"    ... and {len(inventory['at_risk']) - args.top} more "
                  f"(full list in the manifest)")
    else:
        print("\n  No at-risk files: everything present is committed, unmodified, "
              "or\n  regenerable by a recipe whose dependency was checked.")

    print("\n  THIS IS NOT PERMISSION TO DELETE ANYTHING.")
    print("  'regenerable' means a recipe exists and its dependency is present "
          "TODAY.")
    print("  It does not mean the recipe has been RUN, that it reproduces the "
          "bytes,")
    print("  or that re-running it is free. Verify a backup with --verify "
          "<dir> before")
    print("  removing anything, and prefer preserving over reclaiming space.")

    payload = {
        "schema_version": "preservation_inventory_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkout": str(root),
        "git": replication,
        "summary": {
            "tracked_files": tracked_stats["files"],
            "tracked_bytes": tracked_stats["bytes"],
            "regenerable_files": regen_files,
            "regenerable_bytes": regen_bytes,
            "at_risk_files": len(inventory["at_risk"]),
            "at_risk_bytes": risk_bytes,
        },
        "regenerable": inventory["regenerable"],
        "at_risk": inventory["at_risk"],
    }

    if args.verify:
        return verify(root, args.verify.resolve())

    if args.write_manifest:
        target = root / MANIFEST
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(f"\n  manifest written: {MANIFEST}")
        print("  COMMIT IT. A report in a terminal protects nothing.")
    else:
        print("\n  (no manifest written; pass --write-manifest)")

    print("\n" + "=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
