"""Stamp this lane's pushed HEAD into the handoff manifest.

A manifest cannot contain its own commit hash -- recording it changes the file
and therefore the hash. So the stamp records the **content head**: the commit
carrying all lane work, which is the PARENT of the stamp commit itself.

That makes it verifiable rather than approximate:

    git rev-parse origin/codex/compose-claim2-trajectory^   ==   pushed_head

The script refuses to stamp a commit that is not actually on the remote, since
a manifest claiming a pushed head that was never pushed is worse than one with
no head at all.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

BRANCH = "codex/compose-claim2-trajectory"


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", default=BRANCH)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/workstreams/claim2-trajectory/handoff.json"),
    )
    args = parser.parse_args()

    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", f"origin/{args.branch}")
    if head != remote:
        raise SystemExit(
            f"HEAD {head[:12]} is not the pushed tip {remote[:12]}; push first. A "
            "manifest claiming an unpushed head is worse than one with none."
        )
    if git("status", "--porcelain"):
        raise SystemExit("working tree is dirty; commit before stamping")

    manifest = json.loads(args.manifest.read_text())
    manifest["pushed_head"] = head
    manifest["branch"] = args.branch
    manifest["remote"] = git("remote", "get-url", "origin")
    manifest["pushed_head_verification"] = (
        f"pushed_head is the CONTENT head: the commit carrying all lane work. The "
        f"stamp commit that records it sits one above, so verify with "
        f"`git rev-parse origin/{args.branch}^` == pushed_head. A manifest cannot "
        "contain its own hash without changing it."
    )
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"stamped pushed_head {head[:12]} into {args.manifest}")
    print(f"  verify: git rev-parse origin/{args.branch}^")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
