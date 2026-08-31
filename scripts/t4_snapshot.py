"""Durable local snapshot of every T4 run on the volume.

The Modal volume already holds a per-ROUND checkpoint for every run, so nothing
is held in memory and a crash loses at most one round. This adds a second,
git-tracked copy of the parts that matter -- scores, molecules, traces -- so the
results survive the volume, the workspace and the laptop independently.

Re-runnable at any time; it overwrites the snapshot with whatever is on the
volume right now.
"""
import json, os, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostics/t4_snapshot.json"
RAW = Path("/tmp/t4_snap")

names = subprocess.run(["modal", "volume", "ls", "compose-v4-artifacts", "macro_basin"],
                       capture_output=True, text=True).stdout
files = sorted(set(re.findall(r"episodes_[A-Za-z0-9_.]+\.json", names)))
print(f"  {len(files)} episode files on the volume", flush=True)
RAW.mkdir(parents=True, exist_ok=True)
for i, f in enumerate(files, 1):
    if not (RAW / f).exists():
        subprocess.run(["modal", "volume", "get", "compose-v4-artifacts",
                        f"macro_basin/{f}", str(RAW), "--force"],
                       capture_output=True, text=True)
    if i % 25 == 0: print(f"    pulled {i}/{len(files)}", flush=True)

runs = []
for f in files:
    p = RAW / f
    if not p.exists(): continue
    try: j = json.loads(p.read_text())
    except Exception: continue
    pr = j.get("provenance", {}) or {}
    m = re.match(r"episodes_(.+?)_pooled(.*?)_?r(\d+)\.json$", f)
    cell = m.group(1) if m else f
    seed = int(m.group(3)) if m else None
    arm  = (m.group(2) or "").strip("_") if m else ""
    runs.append(dict(
        file=f, cell=cell, seed=seed, arm=arm or "pooled",
        rounds_done=int(j.get("round", -1)) + 1,
        rounds_cfg=pr.get("rounds"), dock_per_round=pr.get("dock_per_round"),
        refine=pr.get("refine"), config_sha256=pr.get("config_sha256"),
        best_ds=j.get("best_ds"), n_feasible=j.get("n_feasible"),
        budget_curve=j.get("budget_curve"),
        # every docked molecule and its score
        docked={k: v for k, v in (j.get("docked") or {}).items()},
        # every feasible endpoint with the macro trace that built it
        archive=[dict(smiles=a.get("smiles"), ds=None, heavy=a.get("heavy"),
                      sim=a.get("sim"), qed=a.get("qed"), sa=a.get("sa"),
                      trace=a.get("trace"), ai=a.get("_ai"))
                 for a in (j.get("archive") or [])],
        action_probs=j.get("action_probs"),
    ))

complete = [r for r in runs if r["rounds_done"] >= 10 and (r["rounds_cfg"], r["dock_per_round"]) == (10, 10)]
n_mols = sum(len(r["docked"]) for r in runs)
n_arch = sum(len(r["archive"]) for r in runs)
snap = dict(
    taken_at=subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                            capture_output=True, text=True).stdout.strip(),
    n_runs=len(runs), n_complete_10x10=len(complete),
    n_docked_molecules=n_mols, n_archive_entries=n_arch, runs=runs)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(snap, indent=1))
print(f"\n  runs captured      {len(runs)}")
print(f"  complete (10x10)   {len(complete)}")
print(f"  docked molecules   {n_mols}")
print(f"  archive endpoints  {n_arch}")
print(f"  wrote {OUT}  ({OUT.stat().st_size/1024/1024:.1f} MB)")
