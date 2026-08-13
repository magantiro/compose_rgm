"""Block until a detached Modal app leaves the running state.

Polling ``modal app list`` rather than tailing logs, because a log tail dies
with the client and this lane has already lost work to that twice today. The
app itself is detached; this is only a local watcher and killing it is safe.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

RUNNING = "ephemeral (detached)"


def state_of(app_id: str) -> str | None:
    result = subprocess.run(
        ["modal", "app", "list", "--json"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        return None
    try:
        rows = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    for row in rows:
        if row.get("App ID") == app_id:
            return str(row.get("State"))
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("app_id")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--timeout", type=float, default=7200.0)
    args = parser.parse_args()

    deadline = time.time() + args.timeout
    while time.time() < deadline:
        state = state_of(args.app_id)
        if state is None:
            print(f"{args.app_id}: not listed (transient or finished)", flush=True)
        elif state != RUNNING:
            print(f"{args.app_id}: {state}", flush=True)
            return 0
        time.sleep(args.interval)
    print(f"{args.app_id}: still running at timeout", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
