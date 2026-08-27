"""Launch the OFFICIAL GrIDDD/Jin QED 800 against the DEPLOYED hphi-h40head-ab app.

WHY A LAUNCHER AND NOT `modal run`. `modal run` creates an EPHEMERAL app: when
the local entrypoint returns, the app is torn down and every call it spawned
dies with it. A drive() that lives server-side is not enough on its own -- the
app it belongs to has to outlive the shell. Spawning against the DEPLOYED app
(`modal deploy` first) removes the tether entirely; the laptop can close.

Independently of that, run_source commits each source's record to the Volume
BEFORE its call returns, so a driver death loses only in-flight sources and the
resume check skips everything already written.

The task dicts come from official_tasks() in the app module itself, so the
launcher and the `--official` entrypoint branch cannot drift apart.

    python scripts/hphi_official800_launch.py parity 10
    python scripts/hphi_official800_launch.py smoke
    python scripts/hphi_official800_launch.py rest
    python scripts/hphi_official800_launch.py ext      # later: K=8..20
    python scripts/hphi_official800_launch.py cap
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

ROOT = Path("/Users/rmaganti/compose_v2_work")
sys.path.insert(0, str(ROOT))
from modal_apps.hphi_h40head_ab_app import official_tasks  # noqa: E402

APP = "hphi-h40head-ab"
OUT_DIR = "hphi_official800_k8"            # slots k = 0..7
# Extension slots. HPHI_EXT_K_END lets a campaign append a SHORTER extension
# (e.g. k=8..12, four more trajectories) instead of jumping straight to K=20.
# The directory is named for the actual range so two different extensions can
# coexist, and _record_name embeds k{start}-{end} so records never collide.
EXT_K_START = 8
EXT_K_END = int(os.environ.get("HPHI_EXT_K_END", "20"))
EXT_DIR = f"hphi_official800_k{EXT_K_START}{EXT_K_END}"
PARITY_DIR = "hphi_official_parity_check"
# The user's standing cap is 80; never exceed it. HPHI_MAX_CONTAINERS lets a
# single campaign run narrower (e.g. 20) without editing the default, and is
# clamped so a typo cannot raise it above the standing cap.
MAX_CONTAINERS = min(80, int(os.environ.get("HPHI_MAX_CONTAINERS", "80")))


def fn(name: str):
    return modal.Function.from_name(APP, name)


def set_cap() -> None:
    """80 containers, set on the DEPLOYED function's autoscaler.

    The decorator in the app file still reads max_containers=64 and is left
    BYTE-IDENTICAL on purpose: the official run has to be the same code as the
    banked 128, and an edit inside run_source's decorator is an edit inside
    run_source. The autoscaler override raises only the scheduling cap, which
    changes wall clock and nothing a record contains. NOTE: a redeploy resets
    it, so this is re-asserted before every launch.
    """
    fn("run_source").update_autoscaler(max_containers=MAX_CONTAINERS)
    print(f"run_source autoscaler max_containers -> {MAX_CONTAINERS}")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "cap"
    if mode == "cap":
        set_cap()
        return

    if mode == "parity":
        # PROOF, not assertion, that the official path is the banked path.
        # This task dict is what the untouched --valid branch builds for this
        # source, with ONLY out_dir redirected so the banked record cannot be
        # overwritten. seed_for keys on (arm, source, k) and never on the
        # source list, so the record must come back identical.
        i = int(sys.argv[2])
        vs = [x.strip() for x in
              (ROOT / "data/jin/hphi_valid_128.txt").read_text().split("\n")
              if x.strip()]
        assert len(vs) == 128
        task = {"index": i, "source": vs[i], "stratum": "prospective",
                "arm": "restart", "horizon": 40, "out_dir": PARITY_DIR,
                "head_dir": "hphi_v2", "budget_max": 24,
                "k_start": 0, "k_end": 8}
        set_cap()
        call = fn("run_source").spawn(task)
        print(f"PARITY valid[{i}] -> {PARITY_DIR}/{i:03d}_H40_hphi_v2_k0-8.json")
        print(f"spawned: {call.object_id}")
        return

    out_dir, k_start, k_end = OUT_DIR, 0, 8
    if mode == "smoke":
        lo, hi = 0, 32
    elif mode == "rest":
        lo, hi = 32, 800
    elif mode == "range":
        lo, hi = int(sys.argv[2]), int(sys.argv[3])
    elif mode == "resume":
        lo, hi = 0, 800
    elif mode == "ext":
        # K=8 -> K=20, APPEND ONLY. Slot indices are contiguous and seed_for
        # keys on (arm, source, k), so slots 8..19 are exactly what a single
        # k=0..20 run would have produced there. Slots 0..7 are NOT recomputed
        # and are NOT touched: a different out_dir, a different record name.
        out_dir, k_start, k_end = EXT_DIR, EXT_K_START, EXT_K_END
        lo, hi = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 800)
    else:
        raise SystemExit(f"unknown mode {mode!r}")

    tasks = official_tasks(ROOT, out_dir, k_start, k_end, lo, hi)
    if mode == "resume" or "--only-missing" in sys.argv:
        # Sources whose record is already on the Volume are dropped here as
        # well as skipped inside run_source, so a resume does not pay for 800
        # container starts to discover 790 no-ops.
        import subprocess
        r = subprocess.run(["modal", "volume", "ls", "compose-v4-artifacts",
                            f"editing_v2/r_theta_run/{out_dir}"],
                           capture_output=True, text=True)
        have = {line.strip().rsplit("/", 1)[-1] for line in r.stdout.splitlines()
                if line.strip().endswith(".json")}
        before = len(tasks)
        tasks = [t for t in tasks
                 if f"{t['index']:03d}_H40_hphi_v2_k{k_start}-{k_end}.json"
                 not in have]
        print(f"resume: {before - len(tasks)} already on the Volume, "
              f"{len(tasks)} to run")
        if not tasks:
            print("nothing to do")
            return
    set_cap()
    call = fn("drive").spawn(tasks)
    print(f"OFFICIAL 800 wave [{lo}..{hi}) = {len(tasks)} sources x "
          f"k={k_start}..{k_end}, H=40, N=32, head hphi_v2, budget_max=24, "
          f"arm=restart, stratum=official")
    print(f"out_dir: editing_v2/r_theta_run/{out_dir}")
    print(f"spawned: {call.object_id}")


if __name__ == "__main__":
    main()
