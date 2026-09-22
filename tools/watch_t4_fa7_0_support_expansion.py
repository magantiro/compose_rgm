"""Read the fa7_0 support-expansion run's state from its Modal volume.

Uses the Python API, not `modal volume ls`: the CLI silently returns the PARENT
listing for a valid subpath, which has produced false readings in this repository
before.  Round locks are the authority for charged calls -- a checkpoint is
overwritable and a container that restarts after preemption can read a stale one.
"""

from __future__ import annotations

import argparse
import json

import modal

VOLUME = "compose-t4-fa7-0-support-expansion"
CELL = "fa7_0"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    volume = modal.Volume.from_name(VOLUME)
    prefix = f"{args.run_id}/{CELL}"
    try:
        entries = sorted(entry.path for entry in volume.listdir(prefix))
    except Exception as error:  # noqa: BLE001 - a missing folder is a real state
        print(f"no cell folder yet under {prefix}: {error!r}")
        print("top level:", sorted(e.path for e in volume.listdir("/"))[-5:])
        return 0

    locks = [path for path in entries if path.endswith("_lock.json")]
    print(f"run {args.run_id}\n  entries {len(entries)}  locks {len(locks)}")

    def _read(path: str) -> dict:
        chunks = b"".join(volume.read_file(path))
        return json.loads(chunks)["payload"]

    for path in locks:
        payload = _read(path)
        queries = payload.get("queries") or []
        expansion = payload.get("support_expansion")
        line = (
            f"  {path.rsplit('/', 1)[-1]}: round={payload.get('round')} "
            f"charged_before={payload.get('charged_before')} queries={len(queries)}"
        )
        if expansion:
            line += (
                f" | EXPANSION stop={expansion['stop_reason']} "
                f"eligible={expansion['distinct_eligible']} "
                f"fallback={expansion['fallback_eligible']} "
                f"attempts={expansion['attempts']} draws={expansion['draws_spent']}"
            )
        print(line)
        if args.verbose and queries:
            for query in queries[:3]:
                print(f"      {query.get('smiles')}")

    # A driver-level failure publishes summary.json with status=failed and never
    # writes a cell result, so watching only the cell folder would read a crash as
    # "still running" -- silence is not success.
    try:
        top = {entry.path for entry in volume.listdir(args.run_id)}
    except Exception:  # noqa: BLE001
        top = set()
    if f"{args.run_id}/summary.json" in top:
        summary = _read(f"{args.run_id}/summary.json")
        for record in summary.get("records") or []:
            print(f"  summary: {record}")

    for name in ("checkpoint.json", "result.json"):
        if f"{prefix}/{name}" in entries:
            payload = _read(f"{prefix}/{name}")
            best = payload.get("final_best")
            if best is None and payload.get("archive"):
                best = min(payload["archive"].values())
            print(
                f"  {name}: status={payload.get('status')} "
                f"charged={payload.get('charged_calls')} best={best} "
                f"archive={len(payload.get('archive') or {})}"
            )
            if name == "result.json" and payload.get("support_expansion"):
                print(f"    terminal expansion: {json.dumps(payload['support_expansion'])[:400]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
