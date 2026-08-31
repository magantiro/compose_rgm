"""OFFICIAL T4 panel at the frozen controller (5e399bc), refine=2.

30 cells x 3 independent runs = 90 fresh runs from call zero. The 33 existing
runs came from the pre-REFINE_RING controller and are NOT part of this
aggregate.

Budget is LOCKED at 100 docking calls per cell: 10 rounds x 10 docks. No
1000-call extension.

Rolling pool rather than spawning all 90: each drive_episodes orchestrator is a
billed container, and with search_episode capped at 256 only ~4 runs progress at
once, so 86 idle orchestrators would burn core-hours doing nothing. The pool
keeps the episode cap saturated without paying for idlers.

The first completed runs are MEASUREMENT as well as result -- they count toward
the 90. Throughput is reported from them, then the panel continues on its own.
"""
import json, os, sys, time
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
# POOL=2, not 6. The workspace caps at 100 containers, so 6 runs x 64 episodes
# demand ~384 and every run crawls in lockstep -- aggregate throughput is
# identical (the cap binds either way) but the FIRST completion arrives ~3x
# later and long runs approach the 4h drive_episodes timeout. Two runs use
# ~98 of the 100 slots, saturating the cap while still finishing steadily.
POOL = int(os.environ.get("T4_POOL", "2"))
ROUNDS = int(os.environ.get("T4_ROUNDS", "10"))
DOCK_PER_ROUND = int(os.environ.get("T4_DOCK", "10"))
N_PART = int(os.environ.get("T4_PARTICLES", "64"))
# 200/201/202, not 0/1/2: r0-r3, r10-r12, r20-r22, r30-r32, r98, r99 already
# exist on the volume from the PRE-REFINE_RING controller, and refine=2 writes
# with no path suffix, so seeds 0/1/2 would overwrite that development evidence.
# run_seed only sets the RNG stream and the checkpoint path, so any free block
# gives three equally independent runs -- and this needs no code change after
# the frozen hash.
RUN_SEEDS = [200, 201, 202]
REPORT_AFTER = int(os.environ.get("T4_REPORT_AFTER", "4"))
STATE = ROOT / "diagnostics/t4_official_panel_progress.json"

CPU_RATE, GIB_RATE, MEM_GIB = 0.04716, 0.007992, 3.0

seeds = {f"{s['target']}_s{s['idx']}": s
         for s in json.loads((ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())}
cells = [f"{s['target']}_s{s['idx']}_d{d}"
         for s in json.loads((ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())
         for d in ("0.4", "0.6")]
jobs = [(c, rs) for c in cells for rs in RUN_SEEDS]
assert len(jobs) == 90, len(jobs)
# Cells already spawned by an earlier driver and still running server-side.
# They are valid official runs at this config, so they are neither cancelled nor
# re-queued -- re-spawning would duplicate work against the same checkpoint.
SKIP = {tuple(x.split(":")[0:1]) + (int(x.split(":")[1]),)
        for x in os.environ.get("T4_SKIP", "").split(",") if x.strip()}
if SKIP:
    before = len(jobs)
    jobs = [j for j in jobs if j not in SKIP]
    print(f"skipping {before - len(jobs)} already-spawned runs: {sorted(SKIP)}")

fn = modal.Function.from_name("macro-basin", "drive_episodes")


def payload_for(cell):
    tgt, si, dl = cell.rsplit("_", 2)
    sd = seeds[f"{tgt}_{si}"]
    return dict(cell=cell, seed=sd["smiles"], delta=float(dl[1:]),
                target=sd["target"], idx=sd["idx"])


def spawn(cell, rs):
    return fn.spawn(payload_for(cell), n_particles=N_PART, rounds=ROUNDS,
                    episode_len=5, dock_per_round=DOCK_PER_ROUND,
                    arm="pooled", run_seed=rs, semantics="legacy", refine=2)


def report(done, t_start):
    """Measured throughput -- never modelled."""
    durs = [d["wall_s"] for d in done]
    mean = sum(durs) / len(durs)
    # container-seconds per run: N_PART episodes x ROUNDS rounds, plus the
    # orchestrator itself for the run's whole wall time.
    cs = N_PART * ROUNDS * mean + mean
    cost = cs / 3600 * (CPU_RATE + MEM_GIB * GIB_RATE)
    elapsed = time.time() - t_start
    achieved = sum(durs) / elapsed if elapsed > 0 else 0
    remaining = 90 - len(done)
    eta_h = (remaining / max(achieved, 1e-9)) * mean / 3600 if achieved else float("nan")
    print("\n" + "=" * 68)
    print(f"MEASURED THROUGHPUT after {len(done)} official runs")
    print("=" * 68)
    print(f"  wall time / run          {mean:,.0f} s  ({mean/60:.1f} min)")
    print(f"                           range {min(durs):,.0f}-{max(durs):,.0f} s")
    print(f"  container-seconds / run  {cs:,.0f}  ({cs/3600:.1f} core-h)")
    print(f"  cost / run               ${cost:,.2f}")
    print(f"  achieved concurrent runs {achieved:.1f}")
    print(f"  cost for all 90          ${cost*90:,.0f}")
    print(f"  ETA remaining {remaining} runs   {eta_h:.1f} h")
    print("=" * 68 + "\n", flush=True)


def completed_on_volume():
    """Runs whose checkpoint already shows the final round.

    The driver loop is a LOCAL process; the runs themselves are server-side
    spawns and survive it. So a laptop sleep kills the queue, not the work --
    and without this a restart would redo all 90. drive_episodes has no
    mid-run resume, so a partially finished run is correctly re-run whole.
    """
    import subprocess, re
    dst = Path("/tmp/t4_panel_ck")
    dst.mkdir(parents=True, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "compose-v4-artifacts",
                    "macro_basin", str(dst), "--force"],
                   capture_output=True, text=True)
    d = dst / "macro_basin"
    pat = re.compile(r"episodes_(.+)_pooled_r(\d+)\.json$")
    out = set()
    for f in (d.glob("episodes_*_pooled_r*.json") if d.exists() else []):
        m = pat.search(f.name)
        if not m:
            continue
        cell, rs = m.group(1), int(m.group(2))
        if rs not in RUN_SEEDS:
            continue
        try:
            j = json.loads(f.read_text())
        except Exception:
            continue
        pr = j.get("provenance", {}) or {}
        # A checkpoint only counts as done if it is the FINAL round AND was
        # produced by this frozen configuration. Anything else is re-run.
        if (int(j.get("round", -1)) + 1 >= ROUNDS
                and int(pr.get("refine", -1)) == 2
                and int(pr.get("rounds", -1)) == ROUNDS
                and int(pr.get("dock_per_round", -1)) == DOCK_PER_ROUND):
            out.add((cell, rs))
    return out


def main():
    t_start = time.time()
    pending = list(jobs)
    if "--resume" in sys.argv:
        have = completed_on_volume()
        before = len(pending)
        pending = [j for j in pending if (j[0], j[1]) not in have]
        print(f"resume: {len(have)} runs already complete at this config, "
              f"{before} -> {len(pending)} to run", flush=True)
    inflight, done = {}, []
    reported = False
    while pending or inflight:
        while pending and len(inflight) < POOL:
            cell, rs = pending.pop(0)
            h = spawn(cell, rs)
            inflight[h.object_id] = dict(h=h, cell=cell, rs=rs, t0=time.time())
            print(f"  spawn {cell} r{rs}  ({len(done)} done, {len(inflight)} inflight, "
                  f"{len(pending)} queued)", flush=True)
        time.sleep(20)
        for oid in list(inflight):
            it = inflight[oid]
            try:
                r = it["h"].get(timeout=0)
            except BaseException as e:
                # Only a TIMEOUT means "still running". Anything else is a real
                # failure and must not be silently treated as pending, or a
                # dead run would keep a pool slot forever.
                if isinstance(e, TimeoutError) or "Timeout" in type(e).__name__:
                    continue
                print(f"  FAIL {it['cell']} r{it['rs']}: {type(e).__name__} {str(e)[:150]}",
                      flush=True)
                done.append(dict(cell=it["cell"], rs=it["rs"], best=None,
                                 wall_s=time.time() - it["t0"], error=True))
                del inflight[oid]
                continue
            w = time.time() - it["t0"]
            print(f"  DONE  {it['cell']} r{it['rs']}  best={r.get('best')}  {w:,.0f}s "
                  f"({len(done)+1}/90)", flush=True)
            done.append(dict(cell=it["cell"], rs=it["rs"], best=r.get("best"),
                             wall_s=w, error=False))
            del inflight[oid]
            STATE.parent.mkdir(parents=True, exist_ok=True)
            STATE.write_text(json.dumps(dict(
                frozen_hash="5e399bc", refine=2, rounds=ROUNDS,
                dock_per_round=DOCK_PER_ROUND, n_particles=N_PART,
                budget_calls=ROUNDS * DOCK_PER_ROUND,
                completed=len(done), results=done), indent=2))
            ok = [d for d in done if not d["error"]]
            if not reported and len(ok) >= REPORT_AFTER:
                report(ok, t_start); reported = True
    ok = [d for d in done if not d["error"]]
    print(f"\nPANEL COMPLETE  {len(ok)}/90 succeeded, {len(done)-len(ok)} failed")
    if ok:
        report(ok, t_start)
    print(f"state -> {STATE}")


if __name__ == "__main__":
    main()
