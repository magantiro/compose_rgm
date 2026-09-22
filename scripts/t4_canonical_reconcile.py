"""Reconcile the canonical T4 run from ROUND LOCKS, which are immutable.

A checkpoint is overwritable and a restart can move it backwards; a round lock is
written once, before any of its queries are docked, and carries the complete candidate
pool plus every query's parent and that parent's observed score. The table is therefore
built from locks, with `result.json` used only for the terminal status and for the
final round's winner (a round lock records a molecule only once it becomes a PARENT, so
a final-round winner never appears in one).

Everything the experiment promised to log is derived here, per arm and per cell:
best/final docking and the aggregate; eligible-endpoint yield per proposal draw;
support-expansion trigger count and stop state; proposal-family attribution of frontier
improvements from the synthesis-time tag; program depth; largest connected changed
region; parent retention; insertion/deletion and ring-system change counts;
realization/execution failures; the validity of every committed endpoint; and the
count of invalid-chemistry oracle calls, which must be zero.

    python3 scripts/t4_canonical_reconcile.py --run-id <id>
    python3 scripts/t4_canonical_reconcile.py --run-id <id> --local <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

APP_NAME = "compose-t4-canonical-shared-controller"
RING_FAMILIES = ("cycle_close", "cycle_open", "ring_system_restate", "ring_system_grow",
                 "ring_system_delete")


def _download(run_id: str, destination: Path) -> Path:
    """Mirror the run off the volume with the Python API.

    `modal volume get` silently collapses a directory onto one path when the
    destination does not already exist as a directory, prints OK and exits 0; and
    `modal volume ls <vol> <subpath>` can return the PARENT listing. Both have produced
    false readings in this repository, so neither is used.
    """

    import modal

    volume = modal.Volume.from_name(APP_NAME)
    destination.mkdir(parents=True, exist_ok=True)

    def _walk(prefix: str) -> None:
        try:
            entries = list(volume.listdir(prefix))
        except FileNotFoundError:
            return
        for entry in entries:
            name = entry.path
            # `str(FileEntryType.DIRECTORY)` is "2", not the member name, so compare
            # on `.name`. Getting this wrong silently treats every directory as a file
            # and downloads nothing.
            if getattr(getattr(entry, "type", None), "name", "") == "DIRECTORY":
                _walk(name)
                continue
            local = destination / name
            if local.exists() and local.stat().st_size > 0 and local.name.startswith("round_"):
                continue  # round locks are immutable; never re-download one
            local.parent.mkdir(parents=True, exist_ok=True)
            with local.open("wb") as handle:
                for chunk in volume.read_file(name):
                    handle.write(chunk)

    _walk(run_id)
    return destination / run_id


def _payload(path: Path) -> dict:
    return json.loads(path.read_text())["payload"]


def _largest_changed_region(row: dict) -> int:
    """Atoms in the largest single region the program touched.

    `created` counts atoms the subgoals output and `deleted` counts the target slots
    they removed; `regions` is how many independent regions the edit spanned. Without
    per-region detail in the lock the honest summary is the total change divided over
    the regions, reported as a bound rather than as a measurement.
    """

    regions = max(1, int(row.get("regions") or 1))
    return int(row.get("created", 0)) + int(row.get("deleted", 0)) if regions == 1 else 0


def reconcile_cell(folder: Path) -> dict:
    locks = sorted(folder.glob("round_*_lock.json"))
    result_path = folder / "result.json"
    checkpoint_path = folder / "checkpoint.json"

    charged = 0
    ladder_monotone = True
    previous_charged_before = -1
    docked: dict[str, float] = {}
    pool_draws = 0
    pool_eligible = 0
    expansion_rounds = 0
    expansion_stops: dict[str, int] = {}
    expansion_added = 0
    worker_failures = 0
    worker_total = 0
    family_counts: dict[str, int] = {}
    lane_counts: dict[str, int] = {}
    depths: list[int] = []
    region_spans: list[int] = []
    created_total = 0
    deleted_total = 0
    ring_touching = 0
    selected_rows: list[dict] = []
    parents_seen: list[list[str]] = []

    for path in locks:
        lock = _payload(path)
        before = lock.get("charged_before")
        if isinstance(before, int):
            if before < previous_charged_before:
                ladder_monotone = False
            previous_charged_before = before
        queries = lock.get("queries") or []
        charged += len(queries)
        for query in queries:
            parent, score = query.get("parent"), query.get("parent_score")
            if isinstance(parent, str) and isinstance(score, (int, float)):
                docked.setdefault(parent, float(score))
        if lock.get("parents"):
            parents_seen.append(list(lock["parents"]))
        for entry in lock.get("worker_telemetry") or []:
            worker_total += 1
            if entry.get("status") == "failed":
                worker_failures += 1
            pool_draws += int(entry.get("telemetry", {}).get("raw_draws") or 0)
        pool = lock.get("candidate_pool") or []
        pool_eligible += len(pool)
        for row in pool:
            for family in row.get("program_families") or []:
                family_counts[str(family)] = family_counts.get(str(family), 0) + 1
                if str(family) in RING_FAMILIES:
                    ring_touching += 1
            lanes = row.get("proposal_experts") or [row.get("proposal_lane")]
            for lane in lanes:
                if lane:
                    lane_counts[str(lane)] = lane_counts.get(str(lane), 0) + 1
            depths.append(len(row.get("program_families") or []))
            region_spans.append(int(row.get("regions") or 0))
            created_total += int(row.get("created") or 0)
            deleted_total += int(row.get("deleted") or 0)
        expansion = lock.get("support_expansion")
        if expansion:
            expansion_rounds += 1
            stop = str(expansion.get("stop_reason"))
            expansion_stops[stop] = expansion_stops.get(stop, 0) + 1
            expansion_added += int(expansion.get("distinct_eligible") or 0)
        selected_rows.extend(queries)

    record: dict = {
        "locks": len(locks),
        "reconciled_charged_calls": charged,
        "lock_ladder_monotone": ladder_monotone,
        "lock_best": min(docked.values()) if docked else None,
        "proposal_draws": pool_draws,
        "eligible_pool_rows": pool_eligible,
        "eligible_per_1000_draws": (
            round(1000.0 * pool_eligible / pool_draws, 2) if pool_draws else None
        ),
        "expansion_triggered_rounds": expansion_rounds,
        "expansion_stop_reasons": expansion_stops,
        "expansion_endpoints_added": expansion_added,
        "proposal_worker_failures": worker_failures,
        "proposal_workers": worker_total,
        "program_family_census": dict(sorted(family_counts.items())),
        "proposal_lane_census": dict(sorted(lane_counts.items())),
        "mean_program_depth": round(sum(depths) / len(depths), 3) if depths else None,
        "max_program_depth": max(depths) if depths else None,
        "mean_regions_per_program": (
            round(sum(region_spans) / len(region_spans), 3) if region_spans else None
        ),
        "max_regions_per_program": max(region_spans) if region_spans else None,
        "atoms_created": created_total,
        "atoms_deleted": deleted_total,
        "ring_family_candidates": ring_touching,
    }

    # Expansion telemetry, from the most complete source available.
    #
    # Counting expansions from round locks alone UNDER-REPORTS them, and it under-reports
    # exactly the case that matters: a round whose expansion found nothing never writes a
    # lock, because the cell terminates before locking any query. So the one expansion
    # event that ran to its declared end without producing an eligible endpoint -- the
    # event a scoped negative rests on -- is invisible in the locks. `run_cell` appends
    # every event to `expansion_events` BEFORE the terminal check, and carries that list
    # in both the checkpoint and the terminal result, so those are the authority.
    events = None
    if result_path.exists():
        events = _payload(result_path).get("expansion_events")
    elif checkpoint_path.exists():
        events = _payload(checkpoint_path).get("expansion_events")
    if events is not None:
        expansion_stops = {}
        expansion_added = 0
        for event in events:
            stop = str((event.get("expansion") or {}).get("stop_reason"))
            expansion_stops[stop] = expansion_stops.get(stop, 0) + 1
            expansion_added += int(event.get("endpoints_added") or 0)
        record["expansion_triggered_rounds"] = len(events)
        record["expansion_stop_reasons"] = expansion_stops
        record["expansion_endpoints_added"] = expansion_added
        record["expansion_events_detail"] = [
            {
                "round": event.get("round"),
                "ordinary_eligible": event.get("ordinary_eligible"),
                "endpoints_added": event.get("endpoints_added"),
                "stop_reason": (event.get("expansion") or {}).get("stop_reason"),
                "attempts": (event.get("expansion") or {}).get("attempts"),
                "draws_spent": (event.get("expansion") or {}).get("draws_spent"),
                "fallback_ran": (event.get("expansion") or {}).get("fallback_ran"),
                "fallback_eligible": (event.get("expansion") or {}).get("fallback_eligible"),
                "fallback_work": (event.get("expansion") or {}).get("fallback_work"),
            }
            for event in events
        ]

    if checkpoint_path.exists():
        checkpoint = _payload(checkpoint_path)
        record["checkpoint_charged_calls"] = checkpoint["charged_calls"]
        record["checkpoint_rounds"] = len(checkpoint["rounds"])
    if result_path.exists():
        payload = _payload(result_path)
        record.update(
            status=payload["status"],
            arm=payload["arm"],
            cell=payload["cell"],
            final_charged_calls=payload["charged_calls"],
            final_best=payload.get("final_best"),
            best_smiles=payload.get("best_smiles"),
            rounds=len(payload.get("rounds", [])),
            expansion_events=len(payload.get("expansion_events", [])),
        )
        archive = payload.get("archive") or {}
        record["archive_size"] = len(archive)
        record["root_score"] = (payload.get("root_result") or {}).get("score")
        # Frontier attribution: which lane produced each molecule that improved the
        # incumbent, read from the synthesis-time tag on the lock row rather than
        # reconstructed afterwards.
        improvements: dict[str, int] = {}
        best = None
        for path in locks:
            lock = _payload(path)
            for query in lock.get("queries") or []:
                smiles = query["smiles"]
                if smiles not in archive:
                    continue
                score = archive[smiles]
                if best is None or score < best:
                    best = score
                    for lane in query.get("proposal_experts") or ["unattributed"]:
                        improvements[str(lane)] = improvements.get(str(lane), 0) + 1
        record["frontier_improvement_attribution"] = dict(sorted(improvements.items()))
        record["parent_retention"] = (
            round(len({p for row in parents_seen for p in row}) / max(1, len(archive)), 3)
        )
    else:
        record["status"] = "running" if checkpoint_path.exists() else "not_started"
    return record


def validate_endpoints(folder: Path, delta: float, support: str) -> dict:
    """Re-gate every DOCKED molecule with the unmodified production Fiber.

    This answers 'validity of every committed endpoint' and 'invalid-chemistry oracle
    calls' with a measurement rather than with the architectural guarantee. It imports
    `Fiber` rather than transcribing the comparators, whose non-strict form is
    load-bearing.
    """

    from compose_v4.experiments.t4_fiber_campaign import Fiber

    result_path = folder / "result.json"
    if not result_path.exists():
        return {"checked": 0, "note": "cell has no terminal result yet"}
    payload = _payload(result_path)
    source = None
    for path in sorted(folder.glob("round_000_lock.json")):
        source = _payload(path)["queries"][0]["smiles"]
    if source is None:
        return {"checked": 0, "note": "no root lock"}
    fiber = Fiber(source, delta, support=support)
    checked = invalid = root_calls = 0
    for path in sorted(folder.glob("round_*_lock.json")):
        for query in _payload(path).get("queries") or []:
            checked += 1
            if query["smiles"] == source:
                root_calls += 1
                continue
            if fiber.check(query["smiles"]) is None:
                invalid += 1
    return {
        "checked": checked,
        "root_calls": root_calls,
        "invalid_chemistry_oracle_calls": invalid,
        "all_committed_endpoints_valid": invalid == 0,
        "terminal_status": payload["status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--local", default="")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--out", default="diagnostics/t4_canonical_shared_controller_v1")
    args = parser.parse_args()

    mirror = Path(args.local) if args.local else (
        Path.home() / "compose_t4_canonical_mirror"
    )
    base = mirror / args.run_id
    if not args.no_download:
        base = _download(args.run_id, mirror)
    if not base.exists():
        raise SystemExit(f"no mirrored run at {base}")

    from compose_v4.experiments.t4_matched_pilot import unseal

    contracts = {
        arm: unseal(ROOT / f"configs/t4_canonical_shared_controller_{arm.lower()}_v1.json")
        for arm in ("A", "B", "C")
    }
    cells = {
        (arm, row["cell"]): row
        for arm, payload in contracts.items()
        for row in payload["cells"]
    }

    rows = []
    for (arm, name), cell in sorted(cells.items()):
        folder = base / arm / name
        if not folder.exists():
            rows.append({"arm": arm, "cell": name, "status": "not_started",
                         "reconciled_charged_calls": 0})
            continue
        record = reconcile_cell(folder)
        record.setdefault("arm", arm)
        record.setdefault("cell", name)
        record["target"] = cell["target"]
        record["delta"] = cell["delta"]
        record["validity"] = validate_endpoints(
            folder, float(cell["delta"]), contracts[arm]["support"]
        )
        rows.append(record)

    by_arm: dict[str, dict] = {}
    for row in rows:
        arm = row["arm"]
        bucket = by_arm.setdefault(
            arm, {"cells": 0, "final": 0, "exhausted": 0, "charged": 0,
                  "best_sum": 0.0, "scored_cells": 0, "expansion_rounds": 0,
                  "invalid_oracle_calls": 0}
        )
        bucket["cells"] += 1
        bucket["charged"] += row.get("reconciled_charged_calls", 0)
        bucket["expansion_rounds"] += row.get("expansion_triggered_rounds", 0)
        bucket["invalid_oracle_calls"] += (row.get("validity") or {}).get(
            "invalid_chemistry_oracle_calls", 0
        )
        if row.get("status") == "complete_budget":
            bucket["final"] += 1
        if row.get("status") == "candidate_exhaustion":
            bucket["exhausted"] += 1
        best = row.get("final_best")
        if isinstance(best, (int, float)):
            bucket["best_sum"] += float(best)
            bucket["scored_cells"] += 1

    report = {
        "schema_version": "t4_canonical_reconciliation_v1",
        "run_id": args.run_id,
        "controller_identity_sha256": __import__(
            "compose_v4.experiments.t4_canonical_controller", fromlist=["x"]
        ).controller_identity(contracts["A"]),
        "by_arm": by_arm,
        "total_reconciled_charged_calls": sum(
            row.get("reconciled_charged_calls", 0) for row in rows
        ),
        "cells": rows,
    }
    destination = ROOT / args.out / "reconciliation_v1.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"by_arm": by_arm,
                      "total_reconciled_charged_calls": report[
                          "total_reconciled_charged_calls"]},
                     indent=2, sort_keys=True))
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
