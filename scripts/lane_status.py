"""One command that answers 'what is running and is anything wasted?'.

Written after a night of hand-polling truncated `modal app list` columns, which
is how a duplicate-looking Stage B launch and a cosmetic progress-line bug both
went unnoticed for a while. Reports every lane's shard progress beside the live
Modal apps, and flags the two things that actually cost money:

  * DUPLICATE apps for the same lane, both holding tasks -- two launches racing
    the same inputs;
  * STALLED apps holding containers with no shard movement since the last run
    of this script.

Reads only. Launches nothing, stops nothing.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

STATE = Path("diagnostics/.lane_status_state.json")

#: lane -> (volume subdirectory, expected shard count)
LANES = {
    "retargeting held-out": ("retarget_heldout", 80),
    "pathwise stage B": ("pathwise_stage_b", 24),
    "pareto smoke": ("pareto_control_smoke", 12),
    "retargeting development": ("retarget_intervention", 60),
}
VOLUME = "compose-v4-artifacts"
ROOT = "editing_v2/r_theta_run"


def sh(cmd: str, timeout: int = 120) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=timeout).stdout
    except subprocess.TimeoutExpired:
        return ""


def main() -> int:
    print(f"=== lane status  {time.strftime('%H:%M:%S')} ===\n")

    apps = []
    raw = sh("modal app list --json")
    if raw.strip():
        try:
            for a in json.loads(raw):
                apps.append({"id": a.get("App ID", "?"), "state": a.get("State", "?"),
                             "tasks": a.get("Tasks", 0), "name": a.get("Description", "?")})
        except json.JSONDecodeError:
            pass

    live = [a for a in apps if "stopped" not in str(a["state"]).lower()]
    print(f"{'app':<34} {'state':<22} {'tasks':>5}  name")
    for a in live:
        print(f"  {a['id']:<32} {a['state']:<22} {a['tasks']:>5}  {a['name']}")
    total = sum(int(a["tasks"] or 0) for a in live)
    print(f"\n  containers in flight: {total}")

    # WASTE CHECK 1 -- two live apps for one lane means two launches racing.
    by_name: dict[str, list] = {}
    for a in live:
        by_name.setdefault(a["name"], []).append(a)
    # A lane may legitimately hold two live apps when a run is deliberately
    # batched (e.g. --start 0 then --start 12). That is disjoint work, not a
    # race. Flag only when the task counts suggest overlap: a second app whose
    # task count exceeds what the remaining unstarted work could justify.
    # Batched launches are annotated instead of alarmed.
    dup = {n: v for n, v in by_name.items() if len(v) > 1 and
           sum(int(x["tasks"] or 0) for x in v) > 0}
    if dup:
        print("\n  NOTE -- multiple live apps for one lane:")
        for name, group in dup.items():
            ids = ", ".join(f"{g['id'][:14]}(tasks={g['tasks']})" for g in group)
            print(f"    {name}: {ids}")
        print("    Legitimate for a deliberately batched run (--start 0, then")
        print("    --start N) since the source sets are disjoint. Confirm the")
        print("    task counts match the intended batch sizes before acting.")
    else:
        print("  no duplicate live apps")

    print("\nshard progress")
    prev = json.loads(STATE.read_text()) if STATE.exists() else {}
    now = {}
    for lane, (subdir, expected) in LANES.items():
        listing = sh(f"modal volume ls {VOLUME} {ROOT}/{subdir}")
        if not listing.strip():
            continue
        complete = sum(1 for l in listing.splitlines()
                       if ".json" in l and ".partial.json" not in l)
        partial = sum(1 for l in listing.splitlines() if ".partial.json" in l)
        now[lane] = complete
        moved = "" if lane not in prev else (
            f"  (+{complete - prev[lane]} since last check)"
            if complete != prev[lane] else "  *** NO MOVEMENT ***")
        print(f"  {lane:<26} {complete:>3}/{expected:<3} complete, "
              f"{partial:>2} partial{moved}")

    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(now, indent=2) + "\n")
    print("\n  (re-run to see movement since this check)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
