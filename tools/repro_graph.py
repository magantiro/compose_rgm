"""Trace each major result back to the code that produced it, and report broken links.

The chain a published number depends on::

    reported result -> result artifact -> run/contract id -> pinned source hashes
        -> commit/branch -> environment -> external weights and ledgers

This walks the part of that chain that is checkable from the repository, for the
result families named in the migration contract: the frozen T4 panel, the T4
controller campaigns, the PMO scored runs and the PMO diagnostics.

Three link types are checked, and the third is the one usually missed:

``payload_sha256``
    Where an artifact carries its own payload digest, it is recomputed. A result
    whose self-hash does not verify has been edited in place.

``pin``
    Every repo-relative path the artifact pins is hashed and compared. Stale is
    normal for a launch record; absent is the signal.

``branch``
    Branch names appear inside committed artifacts as reproducibility links, not
    as labels: they are how a number gets back to the code that produced it.
    A name that no longer resolves on the remote is a broken link, and it is
    invisible to every hash check. Resolution is queried with ``git ls-remote``
    rather than from local refs, because a pushed branch nobody has fetched is
    absent from ``git log --all`` and reads as lost. This tool was written after
    that exact mistake: 26 files were first reported as permanently lost and are
    in fact held on three unfetched remote branches.

Usage::

    python3 tools/repro_graph.py --out repro/reproducibility_graph_v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys
from typing import Any

HEX64 = re.compile(r"^[0-9a-f]{64}$")
# A dated branch name: two or more hyphenated tokens then a yyyymmdd stamp.
BRANCH_NAME = re.compile(r"\b[a-z0-9]+(?:[-_][a-z0-9]+){1,}-20\d{6}\b")

RESULT_FAMILIES = {
    "t4_frozen_panel": ["diagnostics/T4_FROZEN_RESULT_v1.json"],
    "t4_controller_campaigns": [
        "configs/t4_region_repair_rescue_plan_v1.json",
        "configs/t4_region_repair_rescue_braf_d06_v1.json",
    ],
    "pmo_scored_runs": [
        "configs/pmo_population_controller_v1.json",
        "diagnostics/pmo_realization_repair_v1.json",
    ],
    "pmo_diagnostics": [
        "diagnostics/pmo_realization_binder_cost_v1.json",
        "diagnostics/compose_progress_scoreboard_v1.json",
    ],
}


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def remote_branches(root: pathlib.Path) -> set[str]:
    completed = subprocess.run(
        ["git", "ls-remote", "--heads", "origin"],
        cwd=str(root), capture_output=True, text=True, check=False, timeout=180,
    )
    return {
        line.split("refs/heads/", 1)[1]
        for line in completed.stdout.splitlines()
        if "refs/heads/" in line
    }


def _iter_mappings(node: Any):
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            yield current
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)


def trace(root: pathlib.Path, relative: str, branches: set[str]) -> dict[str, Any]:
    path = root / relative
    if not path.is_file():
        return {"artifact": relative, "present": False}
    document = json.loads(path.read_text(encoding="utf-8"))
    raw = json.dumps(document)

    self_hash = None
    if isinstance(document, dict) and "payload" in document and "payload_sha256" in document:
        canonical = json.dumps(
            document["payload"], sort_keys=True, separators=(",", ":")
        ).encode()
        self_hash = hashlib.sha256(canonical).hexdigest() == document["payload_sha256"]

    pins_total = pins_live = pins_stale = pins_absent = 0
    absent_paths: list[str] = []
    for mapping in _iter_mappings(document):
        for key, value in mapping.items():
            if not (isinstance(key, str) and isinstance(value, str) and HEX64.match(value)):
                continue
            if key.startswith("/") or "/" not in key:
                continue
            pins_total += 1
            target = root / key
            if not target.is_file():
                pins_absent += 1
                absent_paths.append(key)
            elif hashlib.sha256(target.read_bytes()).hexdigest() == value:
                pins_live += 1
            else:
                pins_stale += 1

    named = sorted(set(BRANCH_NAME.findall(raw)))
    # A dated token is only a branch link if it actually names a branch.
    branch_links = [n for n in named if n in branches]
    dangling = [n for n in named if n not in branches and "-20" in n and n.count("-") >= 2]

    return {
        "artifact": relative,
        "present": True,
        "payload_self_hash_verifies": self_hash,
        "pins_total": pins_total,
        "pins_resolving": pins_live,
        "pins_stale": pins_stale,
        "pins_absent": pins_absent,
        "absent_pinned_paths": sorted(set(absent_paths))[:20],
        "branch_links_resolving_on_origin": branch_links,
        "dated_tokens_not_resolving_as_branches": dangling,
        "referenced_contracts": sorted(set(re.findall(r"configs/[A-Za-z0-9_./-]+\.json", raw)))[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="repro/reproducibility_graph_v1.json")
    args = parser.parse_args()
    root = repo_root()

    branches = remote_branches(root)
    families = {
        name: [trace(root, relative, branches) for relative in members]
        for name, members in RESULT_FAMILIES.items()
    }

    broken = []
    for name, traces in families.items():
        for entry in traces:
            if not entry["present"]:
                broken.append({"family": name, "artifact": entry["artifact"], "why": "absent"})
                continue
            if entry["payload_self_hash_verifies"] is False:
                broken.append(
                    {"family": name, "artifact": entry["artifact"], "why": "payload self-hash fails"}
                )
            if entry["dated_tokens_not_resolving_as_branches"]:
                broken.append(
                    {
                        "family": name,
                        "artifact": entry["artifact"],
                        "why": "names a dated token that does not resolve as a remote branch",
                        "tokens": entry["dated_tokens_not_resolving_as_branches"],
                    }
                )

    report = {
        "schema_version": 1,
        "remote_branch_count": len(branches),
        "method_note": (
            "Branch resolution uses git ls-remote, not local refs. A pushed branch nobody has "
            "fetched is absent from `git log --all` and reads as lost; that mistake produced a "
            "false report of 26 permanently lost files in this same audit."
        ),
        "families": families,
        "broken_links": broken,
        "broken_link_count": len(broken),
        "verdict": "NO_BROKEN_LINKS" if not broken else "BROKEN_LINKS_PRESENT",
    }
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for name, traces in families.items():
        print(f"\n{name}")
        for entry in traces:
            if not entry["present"]:
                print(f"   {entry['artifact']}: ABSENT")
                continue
            print(
                f"   {entry['artifact']}: self_hash={entry['payload_self_hash_verifies']} "
                f"pins {entry['pins_resolving']}/{entry['pins_total']} live, "
                f"{entry['pins_stale']} stale, {entry['pins_absent']} absent, "
                f"branches={entry['branch_links_resolving_on_origin']}"
            )
    print(f"\nverdict: {report['verdict']} ({report['broken_link_count']} broken)")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
