"""Reconstruct each T4 cell's true best-so-far and true charged-call count from round locks.

A checkpoint is published only after a round finishes, while that round's lock is published
BEFORE any of its queries are docked.  Each lock carries, for every query, the parent it was
drawn from together with that parent's measured score.  A ``parent_score`` is therefore direct
evidence that the molecule was docked and admitted to the archive, and it survives independently
of the checkpoint.

That matters because ``run_cell`` in the held-target apps reads ``result.json`` and
``checkpoint.json`` straight off the mounted volume with no ``volume.reload()``.  A Modal volume
mount shows a snapshot taken at mount time, so a preempted container that restarts sees a STALE
view, resumes from an older checkpoint, and overwrites the newer one.  The locks are not
overwritten, so they remain the only durable record of what the run actually reached.

Reconciliation rules (both from the task contract):

    true best-so-far   = min over (every lock ``parent_score``) union (committed archive values)
    true charged calls = max(committed ``charged_calls``, newest lock ``charged_before``)

Conflicts are REPORTED, never silently resolved to the better or the worse value.  A molecule
recorded at two different scores is a fact about the run that a reconciliation tool has no
authority to average away.

Read-only: this tool never writes, resumes, relaunches or stops anything on any volume.

Two fetch hazards this tool defends against, both observed:
  * ``modal volume get <vol> <dir> <dest>`` with a destination that does not already exist as a
    directory writes EVERY file in the directory onto the single path ``<dest>`` -- 21 files
    collapse into one and the command still reports success.  The destination is pre-created.
  * a fetch can return nothing at all within a short timeout.  Every expected file is verified
    present, non-empty and JSON-parseable before any verdict is computed; an unverified cell is
    reported as a fetch failure rather than as a result.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

# ---- Campaign map ----------------------------------------------------------------------
# delta is NOT taken from the volume name: compose-t4-held-target-distilled-jak2-d06-250 is
# named d06 but its frozen contract declares delta=0.4.  The contract is authoritative.

ARMS: dict[str, tuple[str, str, str]] = {
    "parp1_d06": (
        "compose-t4-held-target-distilled-parp1-d06-250",
        "parp1",
        "configs/t4_held_target_distilled_parp1_d06_250.json",
    ),
    "braf_d06": (
        "compose-t4-held-target-distilled-braf-d06-250",
        "braf",
        "configs/t4_held_target_distilled_braf_d06_250.json",
    ),
    "fa7_d06": (
        "compose-t4-held-target-distilled-fa7-d06-250",
        "fa7",
        "configs/t4_held_target_distilled_fa7_d06_250.json",
    ),
    "5ht1b_d06": (
        "compose-t4-held-target-distilled-5ht1b-d06-250",
        "5ht1b",
        "configs/t4_held_target_distilled_5ht1b_d06_250.json",
    ),
    "jak2_named_d06": (
        "compose-t4-held-target-distilled-jak2-d06-250",
        "jak2",
        "configs/t4_held_target_distilled_jak2_d06_250.json",
    ),
    "jak2_true_d06": (
        "compose-t4-held-target-jak2-true-d06-250",
        "jak2",
        "configs/t4_held_target_distilled_jak2_true_d06_250.json",
    ),
    "parp1_d04": (
        "compose-t4-held-target-distilled-parp1-d04-250",
        "parp1",
        "configs/t4_held_target_distilled_parp1_d04_250.json",
    ),
    "braf_d04": (
        "compose-t4-held-target-distilled-braf-d04-250",
        "braf",
        "configs/t4_held_target_distilled_braf_d04_250.json",
    ),
    "fa7_d04": (
        "compose-t4-held-target-distilled-fa7-d04-250",
        "fa7",
        "configs/t4_held_target_distilled_fa7_d04_250.json",
    ),
    "5ht1b_d04": (
        "compose-t4-held-target-distilled-5ht1b-d04-250",
        "5ht1b",
        "configs/t4_held_target_distilled_5ht1b_d04_250.json",
    ),
}

TOLERANCE = 1e-9


# ---- Modal access (read-only) ----------------------------------------------------------


def _run_modal(args: list[str], timeout: int) -> subprocess.CompletedProcess:
    env = {**os.environ, "MODAL_PROFILE": os.environ.get("MODAL_PROFILE", "nitya")}
    return subprocess.run(
        ["modal", *args], capture_output=True, text=True, timeout=timeout, check=False, env=env
    )


def volume_ls(volume: str, path: str = "", *, attempts: int = 4, timeout: int = 300) -> list[dict]:
    """List one volume path.  Retries; an exhausted listing raises rather than returning []."""
    args = ["volume", "ls", volume] + ([path] if path else []) + ["--json"]
    last = ""
    for _ in range(attempts):
        try:
            done = _run_modal(args, timeout)
        except subprocess.TimeoutExpired:
            last = "timeout"
            continue
        if done.returncode != 0:
            last = (done.stderr or "").strip()[-300:]
            if "not found" in last.lower() or "does not exist" in last.lower():
                return []
            continue
        try:
            return json.loads(done.stdout)
        except json.JSONDecodeError:
            last = "unparseable listing"
    raise RuntimeError(f"volume ls failed for {volume}:{path or '/'} ({last})")


def _read_envelope(path: Path) -> dict | None:
    """Return the payload of a published envelope, or None if the file is unusable."""
    try:
        if path.stat().st_size == 0:
            return None
        body = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(body, dict):
        return None
    payload = body.get("payload", body)
    return payload if isinstance(payload, dict) else None


def fetch_cell(volume: str, run: str, cell: str, cache: Path, *, timeout: int) -> tuple[dict, dict]:
    """Download one cell directory and verify every listed file before returning its payloads.

    Returns (payloads_by_filename, fetch_report).  A file that never verifies is left out of
    the payloads and named in the report, so the caller can refuse to publish a verdict for
    that cell instead of quietly reporting a partial one.
    """
    listing = volume_ls(volume, f"{run}/{cell}")
    expected = sorted(
        row["Filename"].rsplit("/", 1)[-1] for row in listing if row.get("Type") != "dir"
    )
    destination = cache / volume / run[:12] / cell
    # Pre-create: a directory `get` into a non-existent destination collapses every file onto
    # the single destination path and still reports success.
    destination.mkdir(parents=True, exist_ok=True)

    # Round locks are immutable per round index (verified by the monotone charged_before
    # ladder), so they cache.  checkpoint.json and result.json are REWRITTEN as the campaign
    # runs, so a cached copy would silently reconcile against a stale committed state -- the
    # same class of error as the defect being measured.  Always refetch them.
    for volatile in ("checkpoint.json", "result.json"):
        (destination / volatile).unlink(missing_ok=True)

    def verified() -> dict:
        found = {}
        for name in expected:
            payload = _read_envelope(destination / name)
            if payload is not None:
                found[name] = payload
        return found

    payloads = verified()
    attempts = 0
    while len(payloads) < len(expected) and attempts < 4:
        attempts += 1
        try:
            _run_modal(
                ["volume", "get", "--force", volume, f"{run}/{cell}", str(destination.parent)],
                timeout,
            )
        except subprocess.TimeoutExpired:
            pass
        payloads = verified()
        if len(payloads) < len(expected):
            # Fall back to per-file fetches for whatever is still missing.
            for name in expected:
                if name in payloads:
                    continue
                try:
                    _run_modal(
                        [
                            "volume",
                            "get",
                            "--force",
                            volume,
                            f"{run}/{cell}/{name}",
                            str(destination / name),
                        ],
                        timeout,
                    )
                except subprocess.TimeoutExpired:
                    pass
            payloads = verified()

    report = {
        "expected_files": len(expected),
        "verified_files": len(payloads),
        "fetch_attempts": attempts,
        "missing_files": [n for n in expected if n not in payloads],
    }
    return payloads, report


# ---- Reconciliation --------------------------------------------------------------------


def reconcile_cell(payloads: dict, fetch: dict, *, cell: str, run: str) -> dict:
    """Union the committed archive with every score the locks prove was measured."""
    checkpoint = payloads.get("checkpoint.json")
    result = payloads.get("result.json")
    locks = []
    for name, payload in payloads.items():
        if name.startswith("round_") and name.endswith("_lock.json"):
            try:
                index = int(name.split("_")[1])
            except ValueError:
                continue
            locks.append((index, payload))
    locks.sort()

    # evidence: molecule -> {score -> [sources]}.  Keeping the sources makes a conflict
    # attributable instead of merely visible.
    evidence: dict[str, dict[float, list[str]]] = {}

    def observe(smiles: str, score: float, source: str) -> None:
        evidence.setdefault(smiles, {}).setdefault(round(float(score), 9), []).append(source)

    lock_charged_before = None
    newest_lock_index = None
    lock_scores: list[float] = []
    # charged_before must rise with the round index.  A lock is published to a fixed path per
    # round, so a stale restart that re-runs a round REWRITES that round's lock with a lower
    # charged_before.  A non-monotone ladder therefore means the lock evidence has itself been
    # partially overwritten and the reconciled bound is weaker than it looks.
    ladder = [pl["charged_before"] for _, pl in locks if isinstance(pl.get("charged_before"), int)]
    ladder_monotone = all(b >= a for a, b in pairwise(ladder))
    for index, payload in locks:
        charged = payload.get("charged_before")
        if isinstance(charged, int) and (
            lock_charged_before is None or charged > lock_charged_before
        ):
            lock_charged_before = charged
        if newest_lock_index is None or index > newest_lock_index:
            newest_lock_index = index
        for query in payload.get("queries") or ():
            parent = query.get("parent")
            score = query.get("parent_score")
            if parent and isinstance(score, (int, float)):
                observe(parent, score, f"round_{index:03d}_lock")
                lock_scores.append(float(score))

    def archive_of(payload: dict | None) -> dict[str, float]:
        if not payload:
            return {}
        return {
            k: float(v)
            for k, v in (payload.get("archive") or {}).items()
            if isinstance(v, (int, float))
        }

    checkpoint_archive = archive_of(checkpoint)
    result_archive = archive_of(result)
    for smiles, score in checkpoint_archive.items():
        observe(smiles, score, "checkpoint")
    for smiles, score in result_archive.items():
        observe(smiles, score, "result")

    # The driver reports result.json when it exists and falls back to the checkpoint, so that
    # pair is what the published tables were built from.
    if result is not None and result_archive:
        committed_source = "result"
        committed_archive = result_archive
        committed_calls = result.get("charged_calls")
    elif checkpoint is not None:
        committed_source = "checkpoint"
        committed_archive = checkpoint_archive
        committed_calls = checkpoint.get("charged_calls")
    elif result is not None:
        committed_source = "result_without_archive"
        committed_archive = {}
        committed_calls = result.get("charged_calls")
    else:
        committed_source = "none"
        committed_archive = {}
        committed_calls = None

    committed_best = min(committed_archive.values()) if committed_archive else None
    every_score = [score for scores in evidence.values() for score in scores]
    reconciled_best = min(every_score) if every_score else None

    if committed_calls is None and lock_charged_before is None:
        reconciled_calls = None
    else:
        reconciled_calls = max(
            committed_calls if isinstance(committed_calls, int) else 0,
            lock_charged_before if isinstance(lock_charged_before, int) else 0,
        )

    conflicts = {
        smiles: {
            "scores": sorted(scores),
            "sources": {str(score): sorted(set(src)) for score, src in scores.items()},
        }
        for smiles, scores in evidence.items()
        if len(scores) > 1
    }

    # Archive entries the locks prove were measured but the committed archive has lost.
    lost = {}
    for smiles, scores in evidence.items():
        if smiles in committed_archive:
            continue
        lock_only = [s for s, src in scores.items() if all(x.startswith("round_") for x in src)]
        if lock_only:
            lost[smiles] = min(lock_only)
    lost_better = sorted(lost.values())[:5]

    # A conflict on the molecule that defines the reported best puts the headline number
    # itself in question, which is a different severity from a conflict deeper in the archive.
    # The checkpoint records the round results it holds.  Gaps are rounds the app forfeited on
    # a resume; a checkpoint whose last round sits below the newest surviving lock is a
    # checkpoint that lost completed rounds.
    committed_payload = result if committed_source == "result" else checkpoint
    round_ids = sorted(
        row["round"]
        for row in ((committed_payload or {}).get("rounds") or [])
        if isinstance(row, dict) and isinstance(row.get("round"), int)
    )
    checkpoint_last_round = round_ids[-1] if round_ids else None
    forfeited_rounds = (
        [i for i in range(1, checkpoint_last_round + 1) if i not in set(round_ids)]
        if checkpoint_last_round is not None
        else []
    )
    rounds_behind = (
        newest_lock_index - checkpoint_last_round
        if (newest_lock_index is not None and checkpoint_last_round is not None)
        else None
    )

    conflict_on_best = sorted(
        smiles
        for smiles, scores in evidence.items()
        if len(scores) > 1
        and reconciled_best is not None
        and min(scores) <= reconciled_best + TOLERANCE
    )

    best_gap = (
        None
        if (reconciled_best is None or committed_best is None)
        else round(reconciled_best - committed_best, 9)
    )
    calls_gap = (
        None
        if (reconciled_calls is None or not isinstance(committed_calls, int))
        else reconciled_calls - committed_calls
    )
    rolled_back = bool(
        (calls_gap is not None and calls_gap > 0)
        or (best_gap is not None and best_gap < -TOLERANCE)
        or (rounds_behind is not None and rounds_behind > 1)
    )

    return {
        "cell": cell,
        "run": run,
        "run_short": run[:12],
        "fetch": fetch,
        "fetch_complete": not fetch["missing_files"],
        "status": (result or checkpoint or {}).get("status"),
        "committed_source": committed_source,
        "checkpoint_present": checkpoint is not None,
        "result_present": result is not None,
        "round_locks": len(locks),
        "lock_ladder_monotone": ladder_monotone,
        "checkpoint_last_round": checkpoint_last_round,
        "checkpoint_forfeited_rounds": forfeited_rounds,
        "rounds_behind_newest_lock": rounds_behind,
        "newest_lock_index": newest_lock_index,
        "newest_lock_charged_before": lock_charged_before,
        "checkpoint_best": min(checkpoint_archive.values()) if checkpoint_archive else None,
        "checkpoint_charged_calls": (checkpoint or {}).get("charged_calls"),
        "result_best": min(result_archive.values()) if result_archive else None,
        "result_charged_calls": (result or {}).get("charged_calls"),
        "committed_best": committed_best,
        "committed_charged_calls": committed_calls,
        "lock_only_best": min(lock_scores) if lock_scores else None,
        "reconciled_best": reconciled_best,
        "reconciled_charged_calls": reconciled_calls,
        "best_gap": best_gap,
        "calls_gap": calls_gap,
        "locks_hold_better_score": bool(best_gap is not None and best_gap < -TOLERANCE),
        "rolled_back": rolled_back,
        "molecules_with_evidence": len(evidence),
        "committed_archive_size": len(committed_archive),
        "archive_entries_lost_from_committed": len(lost),
        "lost_entry_best_scores": lost_better,
        "score_conflict_count": len(conflicts),
        "conflict_on_reported_best": conflict_on_best,
        "score_conflicts": conflicts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", default=[], help="repeatable; default all arms")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache", default="")
    parser.add_argument("--timeout", type=int, default=600, help="seconds per modal call")
    parser.add_argument("--baseline", default="configs/t4_published_invirtuogen_baseline.json")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    cache = Path(args.cache) if args.cache else Path.home() / ".cache" / "t4_reconcile"
    cache.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((repo / args.baseline).read_text())
    arms = {k: ARMS[k] for k in (args.arm or ARMS)}

    report: dict = {
        "schema_version": "t4_ledger_reconciliation_v1",
        "generated_at_utc": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "snapshot_note": "These campaigns were RUNNING when this was taken; committed values "
        "advance. Locks and checkpoints were read live, checkpoint.json and result.json always "
        "refetched.",
        "sign_convention": "docking score, more negative is better; gap = COMPOSE - IVG; "
        "negative means COMPOSE wins",
        "reconciliation_rules": {
            "true_best": "min over all lock parent_score values union committed archive values",
            "true_charged_calls": "max(committed charged_calls, newest lock charged_before)",
            "conflicts": "reported, never resolved",
        },
        "method_limits": [
            (
    "Round locks are published to a fixed path per round index, so a rollback that is "
                "later REDONE overwrites the locks of the redone rounds with lower charged_before "
                "values. This tool detects a rollback only while it is still visible -- that is, "
                "while the committed checkpoint sits behind a surviving lock. A cell that rolled "
                "back and has since advanced past the lost segment leaves no trace, and its true "
                "lifetime oracle spend is strictly greater than any surviving artifact records."
            ),
            (
    "reconciled_charged_calls is a LOWER BOUND: charged_before on the newest lock counts "
                "calls charged before that round's queries were docked, and says nothing about "
                "whether that round's own queries were then charged."
            ),
            (
    "A parent_score proves the molecule was docked and admitted to the archive. It does "
                "not prove the archive ever contained a better molecule that no lock sampled as a "
                "parent, so reconciled_best is itself a bound, not a reconstruction."
            ),
        ],
        "baseline_source": args.baseline,
        "arms": {},
        "cells": [],
    }

    for label, (volume, target, contract_path) in arms.items():
        contract_body = json.loads((repo / contract_path).read_text())
        contract = contract_body.get("payload", contract_body)
        delta = contract.get("delta")
        budget = contract.get("charged_calls_per_cell")
        print(f"\n=== {label}  vol={volume}  target={target}  delta={delta}", flush=True)
        root = volume_ls(volume)
        runs = sorted(row["Filename"] for row in root if row.get("Type") == "dir")
        arm_rows = []
        for run in runs:
            sub = volume_ls(volume, run)
            cells = sorted(
                row["Filename"].rsplit("/", 1)[-1] for row in sub if row.get("Type") == "dir"
            )
            for cell in cells:
                payloads, fetch = fetch_cell(volume, run, cell, cache, timeout=args.timeout)
                row = reconcile_cell(payloads, fetch, cell=cell, run=run)
                row.update(
                    {
                        "arm": label,
                        "volume": volume,
                        "target": target,
                        "delta": delta,
                        "budget_per_cell": budget,
                        "seed_index": int(cell.rsplit("_", 1)[-1]),
                    }
                )
                arm_rows.append(row)
                report["cells"].append(row)
                flags = []
                if not row["fetch_complete"]:
                    flags.append(f"FETCH-INCOMPLETE {row['fetch']['missing_files']}")
                if row["locks_hold_better_score"]:
                    flags.append("LOCKS HOLD A BETTER SCORE")
                if row["calls_gap"]:
                    flags.append(f"CALLS +{row['calls_gap']}")
                if row["score_conflict_count"]:
                    flags.append(f"conflicts={row['score_conflict_count']}")
                print(
                    f"  {row['run_short']} {cell:9s} src={row['committed_source']:10s} "
                    f"locks={row['round_locks']:2d} "
                    f"best ck={row['committed_best']} rec={row['reconciled_best']} "
                    f"calls ck={row['committed_charged_calls']} rec={row['reconciled_charged_calls']}"
                    + ("  <-- " + "; ".join(flags) if flags else ""),
                    flush=True,
                )
        report["arms"][label] = {
            "volume": volume,
            "target": target,
            "delta": delta,
            "budget_per_cell": budget,
            "contract": contract_path,
            "runs": runs,
            "cell_rows": len(arm_rows),
        }

    _standings(report, baseline)
    out = repo / args.out if not os.path.isabs(args.out) else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


# ---- Standing --------------------------------------------------------------------------


def _pick_authoritative(rows: list[dict]) -> dict:
    """The furthest-progressed attempt for a cell: most reconciled calls, then most evidence."""
    return max(
        rows,
        key=lambda r: (
            r["reconciled_charged_calls"] or 0,
            r["molecules_with_evidence"],
            r["round_locks"],
        ),
    )


def _standings(report: dict, baseline: dict) -> None:
    usable = [r for r in report["cells"] if r["fetch_complete"]]
    grouped: dict[tuple, list[dict]] = {}
    for row in usable:
        grouped.setdefault((row["target"], row["seed_index"], row["delta"]), []).append(row)

    print("\n" + "=" * 118)
    print("PER-CELL LEDGER  (authoritative attempt per cell; more negative is better)")
    print("=" * 118)
    header = (
        f"{'arm':15s} {'cell':9s} {'run':12s} {'ck best':>8s} {'rec best':>8s} "
        f"{'ck calls':>8s} {'rec calls':>9s} {'confl':>5s} {'rolled back':>11s}"
    )
    print(header)
    print("-" * 118)

    standing: list[dict] = []
    for key in sorted(grouped, key=lambda k: (k[2], k[0], k[1])):
        rows = grouped[key]
        chosen = _pick_authoritative(rows)
        target, seed, delta = key
        panel = "delta_0_6" if abs((delta or 0) - 0.6) < 1e-9 else "delta_0_4"
        ivg = (baseline.get(panel, {}).get(target) or [None, None, None])[seed]
        entry = {
            "target": target,
            "seed_index": seed,
            "delta": delta,
            "panel": panel,
            "arm": chosen["arm"],
            "run_short": chosen["run_short"],
            "attempts_on_volume": len(rows),
            "other_attempt_runs": [r["run_short"] for r in rows if r is not chosen],
            "checkpoint_best": chosen["committed_best"],
            "reconciled_best": chosen["reconciled_best"],
            "checkpoint_calls": chosen["committed_charged_calls"],
            "reconciled_calls": chosen["reconciled_charged_calls"],
            "status": chosen["status"],
            "budget_per_cell": chosen["budget_per_cell"],
            "maturity": _maturity(chosen),
            "rounds_behind_newest_lock": chosen["rounds_behind_newest_lock"],
            "lock_ladder_monotone": chosen["lock_ladder_monotone"],
            "checkpoint_forfeited_rounds": chosen["checkpoint_forfeited_rounds"],
            "score_conflict_count": chosen["score_conflict_count"],
            "conflict_on_reported_best": bool(chosen["conflict_on_reported_best"]),
            "rolled_back": chosen["rolled_back"],
            "ivg_baseline": ivg,
            "gap_checkpoint": (
                None
                if (chosen["committed_best"] is None or ivg is None)
                else round(chosen["committed_best"] - ivg, 4)
            ),
            "gap_reconciled": (
                None
                if (chosen["reconciled_best"] is None or ivg is None)
                else round(chosen["reconciled_best"] - ivg, 4)
            ),
        }
        entry["verdict_checkpoint"] = _verdict(entry["gap_checkpoint"])
        entry["verdict_reconciled"] = _verdict(entry["gap_reconciled"])
        entry["verdict_flipped"] = (
            entry["verdict_checkpoint"] != entry["verdict_reconciled"]
            and entry["gap_checkpoint"] is not None
            and entry["gap_reconciled"] is not None
        )
        standing.append(entry)
        print(
            f"{chosen['arm']:15s} {chosen['cell']:9s} {chosen['run_short']:12s} "
            f"{_f(chosen['committed_best']):>8s} {_f(chosen['reconciled_best']):>8s} "
            f"{chosen['committed_charged_calls']!s:>8s} "
            f"{chosen['reconciled_charged_calls']!s:>9s} "
            f"{chosen['score_conflict_count']:>5d} {('YES' if chosen['rolled_back'] else '-'):>11s}"
        )

    report["standing"] = standing
    report["blast_radius"] = _blast_radius(report, standing)
    _panels(standing, report)
    _print_blast(report["blast_radius"])


def _maturity(row: dict) -> str:
    """How far a cell actually got.  A standing mixes these at its peril."""
    calls = row["reconciled_charged_calls"]
    if calls is None:
        return "never_started"
    if row["status"] == "candidate_exhaustion" or calls <= 1:
        # The cell docked its seed and then found no admissible edit, so its "best" is the
        # seed molecule's own score rather than anything the search produced.
        return "aborted_no_search"
    budget = row["budget_per_cell"]
    if isinstance(budget, int) and calls >= budget:
        return "complete"
    return "in_flight"


def _verdict(gap: float | None) -> str:
    if gap is None:
        return "no_data"
    if gap < -1e-9:
        return "COMPOSE_wins"
    if gap > 1e-9:
        return "IVG_wins"
    return "tie"


def _f(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _panels(standing: list[dict], report: dict) -> None:
    panels = {}
    for panel in ("delta_0_6", "delta_0_4"):
        rows = [r for r in standing if r["panel"] == panel]
        if not rows:
            continue
        print("\n" + "=" * 118)
        print(f"CORRECTED STANDING -- {panel}   (gap = COMPOSE - IVG; NEGATIVE means COMPOSE wins)")
        print("=" * 118)
        print(
            f"{'target':8s} {'seed':>4s} {'IVG':>7s} {'ck best':>8s} {'gap ck':>7s} "
            f"{'rec best':>8s} {'gap rec':>7s} {'calls':>6s} {'maturity':>17s} "
            f"{'verdict rec':>13s} {'flip':>5s}"
        )
        print("-" * 118)
        for row in sorted(rows, key=lambda r: (r["target"], r["seed_index"])):
            print(
                f"{row['target']:8s} {row['seed_index']:>4d} {_f(row['ivg_baseline']):>7s} "
                f"{_f(row['checkpoint_best']):>8s} {_f(row['gap_checkpoint']):>7s} "
                f"{_f(row['reconciled_best']):>8s} {_f(row['gap_reconciled']):>7s} "
                f"{row['reconciled_calls']!s:>6s} {row['maturity']:>17s} "
                f"{row['verdict_reconciled']:>13s} "
                f"{('YES' if row['verdict_flipped'] else '-'):>5s}"
            )
        scored = [r for r in rows if r["gap_reconciled"] is not None]
        summary = {
            "cells": len(rows),
            "cells_scored": len(scored),
            "compose_sum_checkpoint": round(
                sum(r["checkpoint_best"] for r in rows if r["checkpoint_best"] is not None), 3
            ),
            "compose_sum_reconciled": round(
                sum(r["reconciled_best"] for r in rows if r["reconciled_best"] is not None), 3
            ),
            "ivg_sum_over_scored": round(sum(r["ivg_baseline"] for r in scored), 3),
            "wins_checkpoint": sum(1 for r in rows if r["verdict_checkpoint"] == "COMPOSE_wins"),
            "wins_reconciled": sum(1 for r in rows if r["verdict_reconciled"] == "COMPOSE_wins"),
            "mean_gap_checkpoint": _mean([r["gap_checkpoint"] for r in rows]),
            "mean_gap_reconciled": _mean([r["gap_reconciled"] for r in rows]),
            "verdict_flips": [
                f"{r['target']}_{r['seed_index']}" for r in rows if r["verdict_flipped"]
            ],
            "cells_aborted_no_search": sum(1 for r in rows if r["maturity"] == "aborted_no_search"),
            "cells_never_started": sum(1 for r in rows if r["maturity"] == "never_started"),
            "cells_in_flight": sum(1 for r in rows if r["maturity"] == "in_flight"),
            "cells_complete": sum(1 for r in rows if r["maturity"] == "complete"),
            "searched_only": _searched_only(rows),
        }
        panels[panel] = summary
        print(
            f"\n  COMPOSE sum   checkpoint {summary['compose_sum_checkpoint']:>8.2f}  ->  "
            f"reconciled {summary['compose_sum_reconciled']:>8.2f}"
        )
        print(
            f"  cells won     checkpoint {summary['wins_checkpoint']:>8d}  ->  "
            f"reconciled {summary['wins_reconciled']:>8d}   of {len(scored)} scored"
        )
        print(
            f"  mean gap      checkpoint {_f(summary['mean_gap_checkpoint']):>8s}  ->  "
            f"reconciled {_f(summary['mean_gap_reconciled']):>8s}"
        )
        searched = summary["searched_only"]
        print(
            f"  cells that actually searched: {searched['cells']} of {len(rows)} "
            f"(aborted at the seed: {summary['cells_aborted_no_search']}, "
            f"never started: {summary['cells_never_started']}); "
            f"of those, reconciled wins {searched['wins_reconciled']} "
            f"(checkpoint {searched['wins_checkpoint']}), "
            f"mean gap {_f(searched['mean_gap_reconciled'])}"
        )
        if summary["verdict_flips"]:
            print(f"  VERDICT FLIPS: {', '.join(summary['verdict_flips'])}")
    report["panel_summary"] = panels


def _searched_only(rows: list[dict]) -> dict:
    """The same standing restricted to cells whose search ran at all."""
    kept = [r for r in rows if r["maturity"] in ("in_flight", "complete")]
    return {
        "cells": len(kept),
        "wins_checkpoint": sum(1 for r in kept if r["verdict_checkpoint"] == "COMPOSE_wins"),
        "wins_reconciled": sum(1 for r in kept if r["verdict_reconciled"] == "COMPOSE_wins"),
        "mean_gap_checkpoint": _mean([r["gap_checkpoint"] for r in kept]),
        "mean_gap_reconciled": _mean([r["gap_reconciled"] for r in kept]),
        "cell_ids": [f"{r['target']}_{r['seed_index']}" for r in kept],
    }


def _mean(values: list) -> float | None:
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 4) if clean else None


def _blast_radius(report: dict, standing: list[dict]) -> dict:
    usable = [r for r in report["cells"] if r["fetch_complete"]]
    better = [r for r in standing if r["reconciled_best"] is not None
              and r["checkpoint_best"] is not None
              and r["reconciled_best"] < r["checkpoint_best"] - TOLERANCE]
    more_calls = [r for r in standing if (r["reconciled_calls"] or 0) > (r["checkpoint_calls"] or 0)]
    return {
        "cell_directories_scanned": len(report["cells"]),
        "cell_directories_verified": len(usable),
        "cell_directories_fetch_failed": len(report["cells"]) - len(usable),
        "campaign_cells": len(standing),
        "cells_rolled_back": sum(1 for r in standing if r["rolled_back"]),
        "cells_with_better_locked_score": len(better),
        "cells_with_undercounted_calls": len(more_calls),
        "total_uncounted_charged_calls": sum(
            (r["reconciled_calls"] or 0) - (r["checkpoint_calls"] or 0) for r in more_calls
        ),
        "best_score_recovered_total": round(
            sum(r["checkpoint_best"] - r["reconciled_best"] for r in better), 3
        ),
        "largest_single_cell_recovery": round(
            max((r["checkpoint_best"] - r["reconciled_best"] for r in better), default=0.0), 3
        ),
        "cells_whose_lock_ladder_was_overwritten": [
            f"{r['target']}_{r['seed_index']}_d{r['delta']}"
            for r in standing
            if not r["lock_ladder_monotone"]
        ],
        "cells_with_forfeited_rounds": sum(
            1 for r in standing if r["checkpoint_forfeited_rounds"]
        ),
        "rounds_lost_to_rollback": sum(
            max(r["rounds_behind_newest_lock"] or 0, 0) - 1
            for r in standing
            if (r["rounds_behind_newest_lock"] or 0) > 1
        ),
        "cells_with_score_conflicts": sum(1 for r in standing if r["score_conflict_count"]),
        "total_score_conflicts": sum(r["score_conflict_count"] for r in standing),
        "cells_whose_reported_best_is_itself_conflicted": sum(
            1 for r in standing if r["conflict_on_reported_best"]
        ),
        "verdict_flips": [
            f"{r['target']}_{r['seed_index']}_d{r['delta']}" for r in standing if r["verdict_flipped"]
        ],
    }


def _print_blast(blast: dict) -> None:
    print("\n" + "=" * 118)
    print("BLAST RADIUS")
    print("=" * 118)
    for key, value in blast.items():
        print(f"  {key:38s} {value}")


if __name__ == "__main__":
    sys.exit(main())
