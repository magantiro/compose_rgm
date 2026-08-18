"""Recall(K) for the executed action inside the exact legal fiber."""
import json, subprocess
from pathlib import Path
import statistics as st
W=Path('/Users/rmaganti/compose_v2_work'); D=Path('/tmp/sl'); D.mkdir(exist_ok=True)
KS=(4,8,16,32,64,128)
sel=json.load(open(W/'docs/SHORTLIST_TASKS.json'))['tasks']
g=json.load(open(W/'docs/HPHI_SMC_64_GATE_BANKED.json'));l=json.load(open(W/'docs/HPHI_COVERAGE_LADDER_BANKED.json'));r3=json.load(open(W/'docs/HPHI_LADDER_RUNG3.json'))
rel=set(g['solved_sources']);marg=set(l['rung2']['new_sources'])|set(r3['conversions'])
strat={i:('reliable' if i in rel else 'marginal' if i in marg else 'hard') for i in range(64)}
def get(rel_,dst):
    if dst.exists(): return True
    subprocess.run(["modal","volume","get","compose-v4-artifacts",rel_,str(dst),"--force"],capture_output=True,text=True)
    return dst.exists()
runs={}
for i,k in sel:
    dst=D/f'{i:03d}.json'
    if get(f'editing_v2/r_theta_run/hphi_shortlist_v11/{i:03d}_H40_hphi_v2_k{k}-{k+1}.json',dst):
        runs[i]=json.load(open(dst))['arms']['restart']
print(f"runs landed {len(runs)}/{len(sel)}")
miss=sum(1 for r in runs.values() for x in r.get('ranks',[]) if x.get('MISSING_FROM_LAW'))
if miss:
    print(f"*** {miss} drawn marks ABSENT from the eager law -- ranks invalid, stopping ***")
    raise SystemExit(1)
pops={'all':[], 'hard':[], 'marginal':[], 'reliable':[], 'target_entry':[]}
per_traj={}
for i,r in runs.items():
    rk=[x for x in r.get('ranks',[]) if x.get('rank')]
    pops['all']+=rk; pops[strat[i]]+=rk
    pops['target_entry']+=[x for x in rk if x.get('entered')]
    if rk: per_traj[i]=rk
print(f"\n{'population':<14}{'n':>7}{'fiber med':>11}" + "".join(f"{'K'+str(k):>8}" for k in KS) + f"{'median':>8}{'p90':>7}{'p99':>7}")
out={}
for name in ('all','hard','marginal','reliable','target_entry'):
    v=pops[name]
    if not v: continue
    rs=[x['rank'] for x in v]; fb=[x['fiber'] for x in v]
    rec={'n':len(v),'fiber_median':st.median(fb),
         'recall':{str(k):sum(1 for x in rs if x<=k)/len(rs) for k in KS},
         'median':st.median(rs),
         'p90':sorted(rs)[int(.9*len(rs))-1],'p99':sorted(rs)[int(.99*len(rs))-1]}
    out[name]=rec
    print(f"{name:<14}{len(v):>7}{st.median(fb):>11.0f}" +
          "".join(f"{rec['recall'][str(k)]*100:>7.1f}%" for k in KS) +
          f"{st.median(rs):>8.0f}{rec['p90']:>7}{rec['p99']:>7}")
# per-trajectory: worst rank on each trajectory, the thing a shortlist must clear
print(f"\nper-TRAJECTORY worst rank (a shortlist must cover the whole route):")
for name in ('hard','marginal','reliable'):
    ws=[max(x['rank'] for x in rk) for i,rk in per_traj.items() if strat[i]==name]
    if not ws: continue
    print(f"  {name:<10} n={len(ws):>3}  median {st.median(ws):>6.0f}  max {max(ws):>6}"
          f"   trajectories fully inside K=32: {sum(1 for w in ws if w<=32)}/{len(ws)}"
          f"   K=128: {sum(1 for w in ws if w<=128)}/{len(ws)}")
    out[f'{name}_traj_worst']={'median':st.median(ws),'max':max(ws),
        'within_32':sum(1 for w in ws if w<=32),'within_128':sum(1 for w in ws if w<=128),'n':len(ws)}
json.dump(out,open(W/'docs/SHORTLIST_RETENTION.json','w'),indent=1)
print("\nwrote docs/SHORTLIST_RETENTION.json")
