"""Reconstruct each T4 cell's true best-so-far from round locks, not just its checkpoint.

A checkpoint is written after a round completes; a round lock is published before that
round's queries are docked and carries, for every query, the parent it was drawn from
together with that parent's measured score.  A parent score is therefore direct evidence
that the molecule was docked and admitted to the archive, and it survives independently of
the checkpoint.

That matters because a preempted container can resume from a stale view of the volume and
overwrite a newer checkpoint with an older one.  It happened once today: the jak2 delta=0.6
cells rolled back from 25 charged calls to 9, and jak2_1's genuine -10.7 incumbent
disappeared from the archive while remaining plainly visible in round_004's locks.

Conflicts are REPORTED, never silently resolved to the better or the worse value.  A
molecule recorded at two different scores is a fact about the run that a reconciliation
tool has no authority to average away.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ARMS = {
    "parp1_d06": ("compose-t4-held-target-distilled-parp1-d06-250", "parp1"),
    "jak2_d04": ("compose-t4-held-target-distilled-jak2-d06-250", "jak2"),
    "braf_d06": ("compose-t4-held-target-distilled-braf-d06-250", "braf"),
    "5ht1b_d06": ("compose-t4-held-target-distilled-5ht1b-d06-250", "5ht1b"),
    "fa7_d06": ("compose-t4-held-target-distilled-fa7-d06-250", "fa7"),
    "jak2_d06": ("compose-t4-held-target-jak2-true-d06-250", "jak2"),
    "parp1_d04": ("compose-t4-held-target-distilled-parp1-d04-250", "parp1"),
    "braf_d04": ("compose-t4-held-target-distilled-braf-d04-250", "braf"),
    "fa7_d04": ("compose-t4-held-target-distilled-fa7-d04-250", "fa7"),
    "5ht1b_d04": ("compose-t4-held-target-distilled-5ht1b-d04-250", "5ht1b"),
}


def _modal(args: list[str], timeout: int = 60) -> str:
    done = subprocess.run(
        ["modal", *args], capture_output=True, text=True, timeout=timeout, check=False,
        env={**__import__("os").environ, "MODAL_PROFILE": "nitya"},
    )
    return done.stdout if done.returncode == 0 else ""


def _payload(path: Path) -> dict | None:
    try:
        body = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return body.get("payload", body)


def reconcile_cell(volume: str, run: str, cell: str, work: Path) -> dict:
    """Union the checkpoint archive with every score the locks prove was measured."""
    listing = _modal(["volume", "ls", volume, f"{run}/{cell}"]).splitlines()
    locks = sorted(x.strip().rsplit("/", 1)[-1] for x in listing if "_lock.json" in x)
    evidence: dict[str, set[float]] = {}
    lock_charged = 0

    for name in locks:
        target = work / f"{cell}_{name}"
        _modal(["volume", "get", volume, f"{run}/{cell}/{name}", str(target)])
        payload = _payload(target)
        if payload is None:
            continue
        lock_charged = max(lock_charged, int(payload.get("charged_before") or 0))
        for query in payload.get("queries") or ():
            parent, score = query.get("parent"), query.get("parent_score")
            if parent and score is not None:
                evidence.setdefault(parent, set()).add(float(score))

    target = work / f"{cell}_checkpoint.json"
    _modal(["volume", "get", volume, f"{run}/{cell}/checkpoint.json", str(target)])
    checkpoint = _payload(target) or {}
    for endpoint, score in (checkpoint.get("archive") or {}).items():
        if isinstance(score, (int, float)):
            evidence.setdefault(endpoint, set()).add(float(score))

    conflicts = {k: sorted(v) for k, v in evidence.items() if len(v) > 1}
    every = [s for scores in evidence.values() for s in scores]
    checkpoint_scores = [
        s for s in (checkpoint.get("archive") or {}).values() if isinstance(s, (int, float))
    ]
    return {
        "cell": cell,
        "molecules_with_evidence": len(evidence),
        "reconciled_best": min(every) if every else None,
        "checkpoint_best": min(checkpoint_scores) if checkpoint_scores else None,
        "checkpoint_charged_calls": checkpoint.get("charged_calls"),
        "lock_charged_before_max": lock_charged,
        "round_locks": len(locks),
        "score_conflicts": conflicts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="", help="one key of ARMS; default every arm")
    parser.add_argument("--out", required=True)
    parser.add_argument("--work", default="")
    args = parser.parse_args()

    work = Path(args.work) if args.work else Path("/tmp/t4_reconcile")
    work.mkdir(parents=True, exist_ok=True)
    arms = {args.arm: ARMS[args.arm]} if args.arm else ARMS

    report: dict = {"schema_version": "t4_ledger_reconciliation_v1", "arms": {}}
    for label, (volume, prefix) in arms.items():
        runs = [x.strip() for x in _modal(["volume", "ls", volume]).splitlines() if x.strip()]
        rows = []
        for run in runs:
            for index in range(3):
                row = reconcile_cell(volume, run, f"{prefix}_{index}", work)
                if row["round_locks"] or row["checkpoint_charged_calls"]:
                    row["run"] = run[:12]
                    rows.append(row)
        report["arms"][label] = rows
        for row in rows:
            recovered = (
                row["reconciled_best"] is not None
                and row["checkpoint_best"] is not None
                and row["reconciled_best"] < row["checkpoint_best"] - 1e-9
            )
            flag = "  <-- LOCKS HOLD A BETTER SCORE" if recovered else ""
            print(
                f"{label:11s} {row['cell']:9s} {row['run']} "
                f"ck={row['checkpoint_best']} locks+ck={row['reconciled_best']} "
                f"calls ck={row['checkpoint_charged_calls']} lock={row['lock_charged_before_max']}"
                f"{flag}",
                flush=True,
            )
    Path(args.out).write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
