"""Read the ONE prospective validation. Full curve, no selected point."""
import json, subprocess
from pathlib import Path
D=Path('/tmp/v128'); D.mkdir(exist_ok=True)
K=8
def get(rel,dst):
    if dst.exists(): return True
    subprocess.run(["modal","volume","get","compose-v4-artifacts",rel,str(dst),
                    "--force"],capture_output=True,text=True)
    return dst.exists()
recs={}
for i in range(128):
    dst=D/f'{i:03d}.json'
    if get(f'editing_v2/r_theta_run/hphi_valid128_k8/{i:03d}_H40_hphi_v2_k0-{K}.json',dst):
        recs[i]=json.load(open(dst))['arms']['restart']
print(f"landed {len(recs)}/128")
if len(recs)<128:
    print("INCOMPLETE -- not reporting a coverage number on a partial panel.")
    raise SystemExit(0)
cum=[sum(1 for r in recs.values() if any(c.get('success') for c in r['candidates'][:k+1])) for k in range(K)]
inc=[cum[0]]+[cum[k]-cum[k-1] for k in range(1,K)]
print(f"\n{'candidates':<12}"+"".join(f"{k+1:>6}" for k in range(K)))
print(f"{'cumulative':<12}"+"".join(f"{c:>6}" for c in cum))
print(f"{'increment':<12}"+"".join(f"{x:>6}" for x in inc))
print(f"\nPROSPECTIVE COVERAGE @8 = {cum[-1]}/128 = {cum[-1]/128*100:.1f}%")
print(f"development @8 was 41/64 = 64.1%   @20 was 47/64 = 73.4%")
contact=sum(1 for r in recs.values() for c in r['candidates'] if c.get('contact'))
runs=sum(len(r['candidates']) for r in recs.values())
ever=sum(1 for r in recs.values() if any(c.get('contact') for c in r['candidates']))
ext=sum(1 for r in recs.values() for c in r['candidates'] if c.get('extinct'))
div=[r['distinct_returned'] for r in recs.values()]
print(f"\ncontact runs {contact}/{runs} ({contact/runs*100:.1f}%)   "
      f"sources with any contact {ever}/128 ({ever/128*100:.1f}%)")
print(f"extinct runs {ext}/{runs} ({ext/runs*100:.1f}%)")
print(f"distinct returned molecules per source: mean {sum(div)/len(div):.2f} of {K}")
w=sum(r['work_transitions'] for r in recs.values())
print(f"work transitions {w:,}   seconds {sum(r['seconds'] for r in recs.values()):,.0f}")
json.dump({'k':K,'cumulative':cum,'increments':inc,'n':128,
           'coverage':cum[-1]/128,'contact_runs':contact,'runs':runs,
           'sources_any_contact':ever,'extinct_runs':ext,
           'mean_distinct_returned':sum(div)/len(div),'work_transitions':w},
          open('docs/VALID128_K8_RESULT.json','w'),indent=1)
print("\nwrote docs/VALID128_K8_RESULT.json")
