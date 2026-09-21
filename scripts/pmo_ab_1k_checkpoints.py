#!/usr/bin/env python3
"""Report the matched A/B at 250/500/750/1000 charged calls, from durable state only.

Every number here comes from the query ledger (the charged-call authority) and the
round locks (the campaign's committed state).  Nothing is read from a mutable
checkpoint and no oracle is called, so the report can be rebuilt at any time and is
valid mid-run as well as at completion.

The two attribution columns -- what fraction of proposals came from the structural
memory, and what fraction of frontier improvements came from memory-derived
descendants -- are computed from the provenance tag the memory channel writes, not
from a self-reported flag.  Arm A has no memory, so both are structurally zero there
and that is the control, not a missing measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compose_v4.experiments.pmo_dynamic_v21 import pmo_top_ten_auc

ARMS = {
    "A_baseline": "8d02a667384b248fed0a5fbb432a93714cc2a3a127ffdc05a4153f37ac3f89a0",
    "B_memory": "37bb856e15309543795e14cdccf44d1b672e957cb9af1e902af08d630f3e241b",
}
TASK = "celecoxib_rediscovery"
MEMORY_TAG = "pmo_online_memory"


def _read(volume, path):
    return json.loads(b"".join(volume.read_file(path)))


def _read_optional(volume, path):
    """Read a durable JSON file, or None if it does not exist yet.

    Absence is the ONLY tolerated failure: a query directory with no result is a
    call in flight.  A corrupt or otherwise unreadable file must raise, because
    silently skipping it would undercount charged calls -- and an undercounted
    ledger is the one error that makes a budget claim wrong.
    """
    try:
        return _read(volume, path)
    except FileNotFoundError:
        return None
    except Exception as error:  # narrowed by message below, then re-raised
        if "not found" in str(error).lower() or "no such file" in str(error).lower():
            return None
        raise


def _ledger_rows(volume, base):
    """Charged queries in charged order. `index` is the ledger's own ordering."""
    rows = []
    for entry in volume.listdir(f"{base}/oracle"):
        name = entry.path.split("/")[-1]
        if not name.startswith("query_"):
            continue
        row = _read_optional(volume, f"{base}/oracle/{name}/result.json")
        if row is None:
            continue
        if row.get("status") != "complete":
            raise RuntimeError(f"charged query is not complete: {name}")
        rows.append(row)
    rows.sort(key=lambda row: row["index"])
    return rows


def _latest_snapshot(volume, base):
    names = sorted(
        e.path.split("/")[-1]
        for e in volume.listdir(f"{base}/campaign")
        if e.path.split("/")[-1].startswith("round_")
    )
    for name in reversed(names):
        lock = _read_optional(volume, f"{base}/campaign/{name}/complete.json")
        if lock is not None:  # the newest round may still be pending
            return lock["snapshot"], name
    raise RuntimeError(f"no committed round lock under {base}")


def _memory_by_endpoint(snapshot):
    """Which archived endpoints came from a memory-derived proposal.

    Read from the provenance tag the memory channel writes at synthesis time, not
    from a self-reported flag on the controller.
    """
    by_endpoint = {}
    for entry in (snapshot.get("entries") or {}).values():
        metadata = (entry.get("provenance") or {}).get("metadata") or {}
        by_endpoint[entry.get("endpoint")] = MEMORY_TAG in metadata
    return by_endpoint


def _attribution(rows, by_endpoint):
    """Attribution over a CHARGED-CALL PREFIX.

    A frontier improvement is a charged call that RAISED the running best, so it is
    defined on the charged sequence: the archive has no order, and "improvement" is
    meaningless without one.  Computing this over the whole archive instead of the
    prefix would report a single number that belongs to no checkpoint.
    """
    memory_calls = sum(1 for row in rows if by_endpoint.get(row["endpoint"], False))
    improvements = memory_improvements = 0
    best = None
    for row in rows:
        score = float(row["score"])
        if best is not None and score > best:
            improvements += 1
            memory_improvements += int(by_endpoint.get(row["endpoint"], False))
        if best is None or score > best:
            best = score
    return {
        "memory_derived_scored_calls": memory_calls,
        "memory_proposal_fraction": memory_calls / len(rows) if rows else None,
        "frontier_improvements": improvements,
        "memory_derived_frontier_improvements": memory_improvements,
        "memory_improvement_fraction": (
            memory_improvements / improvements if improvements else None
        ),
    }


def checkpoint(rows, at, by_endpoint):
    prefix = rows[:at]
    values = [float(row["score"]) for row in prefix]
    if len(values) < at:
        return None
    top = sorted(values, reverse=True)[:10]
    return {
        **_attribution(prefix, by_endpoint),
        "charged_calls": at,
        "best_score": max(values),
        "top10_mean": sum(top) / len(top),
        # The denominator is the checkpoint, not 1000: `top_auc` trapezoids up from
        # (0, 0), so scoring a 250-call prefix against a 1000-call budget would
        # understate it by construction.
        "auc_top10_at_budget": pmo_top_ten_auc(values, budget=at, finish=True),
        "unique_scored_molecules": len({row["endpoint"] for row in prefix}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[250, 500, 750, 1000])
    parser.add_argument("--out", default="diagnostics/pmo_ab_1k_checkpoints_v1.json")
    args = parser.parse_args()
    import modal

    volume = modal.Volume.from_name("compose-v4-artifacts")
    report = {"schema_version": "pmo_ab_1k_checkpoints_v1", "task": TASK, "arms": {}}
    for arm, run_id in ARMS.items():
        base = f"pmo_population_controller_v1/{run_id}/{TASK}"
        rows = _ledger_rows(volume, base)
        snapshot, round_name = _latest_snapshot(volume, base)
        report["arms"][arm] = {
            "run_id": run_id,
            "charged_calls_observed": len(rows),
            "latest_committed_round": round_name,
            "checkpoints": [
                c
                for c in (
                    checkpoint(rows, at, _memory_by_endpoint(snapshot))
                    for at in args.checkpoints
                )
                if c
            ],
            "online_memory_reconstruction": snapshot["pmo_population"].get(
                "online_memory_reconstruction"
            ),
        }
        print(arm, len(rows), "charged", flush=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for arm, row in report["arms"].items():
        print(f"\n=== {arm} ({row['charged_calls_observed']} charged) ===")
        for c in row["checkpoints"]:
            print(f"  {c['charged_calls']:>5}  best {c['best_score']:.4f}  "
                  f"top10 {c['top10_mean']:.4f}  auc {c['auc_top10_at_budget']:.4f}  "
                  f"unique {c['unique_scored_molecules']:>4}  "
                  f"mem-calls {c['memory_derived_scored_calls']:>4}  "
                  f"mem-improv {c['memory_derived_frontier_improvements']}/"
                  f"{c['frontier_improvements']}")


if __name__ == "__main__":
    main()
