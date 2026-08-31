"""Complete the 30-cell T4 table on the FROZEN NARROW controller.

Narrow = the exact vocabulary that produced the 30 banked runs across 10 cells:
three carbon ring specs at 0.08 initial mass each, no refinement. Against
GenMol's published Table 4, at 100 oracle calls versus their 1000, narrow is
6 better / 1 tie / 2 worse on the 9 matched cells.

This fills the remaining 20 cells x 3 independent runs. It does NOT change the
controller: a table mixing vocabularies would be two methods under one name.

Resumable: a cell/seed already holding a COMPLETED run (final round, 10x10
budget) is skipped, so a laptop sleep costs the queue, not the work.
"""
import json, os, re, subprocess, time
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
POOL = int(os.environ.get("FILL_POOL", "2"))
RUN_SEEDS = [0, 1, 2]
STATE = ROOT / "diagnostics/t4_narrow_fill_progress.json"

seeds = json.loads((ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())
by_key = {f"{s['target']}_s{s['idx']}": s for s in seeds}
cells = [f"{s['target']}_s{s['idx']}_d{d}" for s in seeds for d in ("0.4", "0.6")]

# GenMol Table 4, per cell. Passed so a run can stop once it clearly beats the
# comparator: best-so-far is monotone, so stopping early only lowers the score
# we report, and the budget is then stated as AT MOST 100 calls with the actual
# calls_spent recorded per cell. Warm start makes it reversible -- a banked cell
# can be resumed to the full budget later without repeating finished rounds.
_g = json.loads((ROOT / "docs/genmol_t4_targets.json").read_text())
_rows = {r["row"]: r for r in _g["rows"]}
def genmol_for(cell):
    tgt, si, dl = cell.rsplit("_", 2)
    idx = by_key[f"{tgt}_{si}"]["idx"]
    r = _rows.get(idx)
    if not r: return None
    # dl comes from rsplit as "d0.4"/"d0.6", NOT "0.4". Comparing to "0.4" sent
    # every d0.4 cell to the d0.6 column -- parp1_s2_d0.4 resolved to -9.2
    # instead of -11.3, which would have banked cells against a target 2.1
    # kcal/mol too easy.
    return r["genmol_d04"] if dl.lstrip("d") == "0.4" else r["genmol_d06"]


def completed():
    dst = Path("/tmp/fill_ck"); dst.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["modal", "volume", "get", "compose-v4-artifacts", "macro_basin",
                        str(dst), "--force"], capture_output=True, text=True)
    d = dst / "macro_basin"
    # A FAILED SYNC MUST NOT LOOK LIKE "NOTHING IS DONE".
    #
    # Measured: one launch reported `banked complete: 0 runs, to run: 90` while a
    # second driver three seconds later correctly saw 31. The volume pull had
    # silently returned nothing. Because warm start only resumes when
    # 0 < done < rounds, a FINISHED run re-queued that way starts fresh and
    # OVERWRITES a completed checkpoint -- so a transient sync failure would have
    # destroyed hours of finished work rather than merely wasting time.
    #
    # Refuse to proceed on an implausibly empty pull; an empty volume is only
    # legitimate on the very first run, which the operator can force.
    n_files = len(list(d.glob("episodes_*.json"))) if d.exists() else 0
    if n_files == 0 and not os.environ.get("FILL_ALLOW_EMPTY"):
        raise SystemExit(
            "ABORT: volume pull returned 0 episode files (exit=%s). Treating this "
            "as 'nothing banked' would re-run and OVERWRITE completed runs. "
            "Re-run when the volume is reachable, or set FILL_ALLOW_EMPTY=1 if "
            "the volume really is empty.\n  stderr: %s"
            % (r.returncode, (r.stderr or "")[:300]))
    done = set()
    # Accept BOTH filename forms. Runs made while drive_episodes defaulted to
    # refine=0 were written as ..._pooled_rf0_r<seed>.json; refine=0 IS the
    # narrow controller, so those are valid runs of exactly this configuration.
    # Globbing only the bare form made six COMPLETED runs invisible and the
    # fill re-ran them. Provenance (refine, rounds, dock_per_round) is what
    # decides validity, never the filename.
    pats = ["episodes_*_pooled_r[0-9].json", "episodes_*_pooled_rf0_r[0-9].json"]
    files = [f for pat in pats for f in (d.glob(pat) if d.exists() else [])]
    for f in files:
        m = re.match(r"episodes_(.+?)_pooled_(?:rf0_)?r(\d+)\.json$", f.name)
        if not m: continue
        try: j = json.loads(f.read_text())
        except Exception: continue
        pr = j.get("provenance", {}) or {}
        cell_i, seed_i = m.group(1), int(m.group(2))
        ok_cfg = ((pr.get("rounds"), pr.get("dock_per_round")) == (10, 10)
                  and int(pr.get("refine", 0) or 0) == 0          # narrow only
                  and pr.get("arm") == "pooled"
                  and j.get("best_ds") is not None)
        if not ok_cfg:
            continue
        full = int(j.get("round", -1)) + 1 >= 10
        # AN EARLY-STOPPED RUN IS COMPLETE, NOT UNFINISHED. It halts below
        # round 10 by design, once it beats GenMol by more than the 0.70
        # kcal/mol noise floor. Requiring round>=10 made every banked winner
        # look unfinished and re-queued it, which would have undone the entire
        # saving -- parp1_s2_d0.6 r0/r1 were re-spawned despite finishing at
        # 16 and 24 calls. Re-derive the stop condition from the recorded score
        # rather than trusting a flag the checkpoint does not carry.
        # EARLY-STOP COMPLETION RULE REMOVED. It was correct while early stop
        # was on. With the matched mean-of-3 protocol, a run that halted below
        # round 10 is UNDER-BUDGETED and drags its cell mean down, so it must be
        # re-queued and topped up. Warm start resumes it from its checkpoint, so
        # the finished rounds are not repeated -- only the missing ones run.
        # A run that early-stopped ALREADY BEAT GenMol, so topping it up cannot
        # change that cell's verdict -- only its margin. With 16 cells still
        # unmeasured, unmeasured cells are worth more than margin on won ones,
        # so treat a winning stopped run as complete and spend the containers
        # elsewhere. Losing runs are never early-stopped (the condition requires
        # beating GenMol), so nothing that could flip a verdict is skipped here.
        _t = genmol_for(cell_i)
        won = (_t is not None and float(j["best_ds"]) <= float(_t) - 0.70)
        if full or won:
            done.add((cell_i, seed_i))
    return done

have = completed()
# CELL_FILTER lets a second driver work a specific target concurrently without
# stopping the first. Both share the same resume check, so neither re-runs
# completed work; if they briefly overlap on a pending cell, warm start makes
# the duplicate resume rather than restart, so the waste is bounded to rounds.
# CELL_FILTER accepts a comma-separated list of exact cells or prefixes, so a
# second driver can be pointed at a chosen subset while the first keeps running.
#
# Priority is by SEED QED, not by delta. Every seed contributes both a d0.4 and
# a d0.6 cell, so "focus on one delta" is not a real choice; what predicts
# whether a cell yields a usable number is whether QED>=0.6 is reachable from
# the seed at all. Measured: seeds at QED 0.16-0.35 (all of braf, most of fa7)
# are returning zero feasible molecules across full 100-call runs, while seeds
# at QED 0.44-0.89 produce hundreds. Cells that return nothing contribute no
# paired value and cannot tighten the equivalence interval.
_filt = os.environ.get("CELL_FILTER", "").strip()
if _filt:
    pats = [x.strip() for x in _filt.split(",") if x.strip()]
    cells = [c for c in cells if any(c == p or c.startswith(p) for p in pats)]
    print(f"  CELL_FILTER={_filt}: {len(cells)} cells -> {cells}")
jobs = [(c, r) for c in cells for r in RUN_SEEDS if (c, r) not in have]
print(f"  banked complete: {len(have)} runs")
print(f"  to run: {len(jobs)} runs across {len({c for c,_ in jobs})} cells", flush=True)

fn = modal.Function.from_name("macro-basin", "drive_episodes")

def spawn(cell, rs):
    tgt, si, dl = cell.rsplit("_", 2)
    sd = by_key[f"{tgt}_{si}"]
    return fn.spawn(dict(cell=cell, seed=sd["smiles"], delta=float(dl[1:]),
                         target=sd["target"], idx=sd["idx"],
                         # EARLY STOP DISABLED. It was conservative under
                         # best-of-n (stopping can only lower a run that was not
                         # the best anyway). But GenMol Table 4 reports "the mean
                         # docking scores of the most optimized leads of 3 runs",
                         # and InVirtuoGen averages over 3 seeds -- so the matched
                         # statistic is the MEAN, under which every stopped run
                         # drags its cell mean down. stop_margin=None disables it.
                         genmol=genmol_for(cell), stop_margin=None),
                    n_particles=64, rounds=10, episode_len=5, dock_per_round=10,
                    arm="pooled", run_seed=rs, semantics="legacy")

t0 = time.time(); pending = list(jobs); inflight = {}; results = []
errs = {}          # consecutive transport errors per in-flight call
while pending or inflight:
    while pending and len(inflight) < POOL:
        c, rs = pending.pop(0)
        h = spawn(c, rs); inflight[h.object_id] = (h, c, rs, time.time())
        print(f"  spawn {c} r{rs}  ({len(results)} done, {len(pending)} queued)", flush=True)
    time.sleep(20)
    for oid in list(inflight):
        h, c, rs, ts = inflight[oid]
        try:
            r = h.get(timeout=0)
        except BaseException as e:
            # A TIMEOUT means "still running" -- that is the normal poll result.
            if isinstance(e, TimeoutError) or "Timeout" in type(e).__name__:
                errs.pop(oid, None); continue
            # A TRANSIENT TRANSPORT ERROR IS NOT A DEAD RUN. Measured: a
            # ConnectionError("Deadline exceeded") arrived for fa7_s3_d0.6 r2
            # while its checkpoint showed round 3/10 and advancing. Treating
            # that as failure abandoned a healthy run and recorded best=None,
            # which over an overnight panel silently punches holes in the table.
            # Tolerate a run of them and keep polling; only give up if they
            # persist, which is what a genuinely dead call looks like.
            _n = errs.get(oid, 0) + 1
            errs[oid] = _n
            if _n < 12:
                if _n in (1, 6):
                    print(f"  transient {type(e).__name__} on {c} r{rs} "
                          f"({_n}/12), still polling", flush=True)
                continue
            print(f"  FAIL {c} r{rs}: {type(e).__name__} {str(e)[:120]} "
                  f"after {_n} consecutive errors", flush=True)
            results.append(dict(cell=c, seed=rs, best=None))
            del inflight[oid]; errs.pop(oid, None); continue
        print(f"  DONE {c} r{rs} best={r.get('best')} calls={r.get('calls_spent')} "
              f"rounds={r.get('rounds_done')} ({time.time()-ts:.0f}s) "
              f"[{len(results)+1}/{len(jobs)}]", flush=True)
        results.append(dict(cell=c, seed=rs, best=r.get("best"),
                            calls_spent=r.get("calls_spent"),
                            rounds_done=r.get("rounds_done"),
                            genmol=genmol_for(c)))
        del inflight[oid]
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(dict(controller="narrow-3spec",
                                         budget_calls=100, results=results), indent=2))
print(f"\nfill complete: {len(results)} runs in {(time.time()-t0)/3600:.1f} h")
