"""Merged validation curve: k=0..8 plus the k=8..12 extension."""
import json, subprocess
from pathlib import Path
D=Path('/tmp/v128'); D2=Path('/tmp/v128b'); D2.mkdir(exist_ok=True)
def get(rel,dst):
    if dst.exists(): return True
    subprocess.run(["modal","volume","get","compose-v4-artifacts",rel,str(dst),
                    "--force"],capture_output=True,text=True)
    return dst.exists()
cands={}; miss8=[]; miss12=[]
for i in range(128):
    c=[]
    d1=D/f'{i:03d}.json'
    if get(f'editing_v2/r_theta_run/hphi_valid128_k8/{i:03d}_H40_hphi_v2_k0-8.json',d1):
        c+=json.load(open(d1))['arms']['restart']['candidates']
    else: miss8.append(i)
    d2=D2/f'{i:03d}.json'
    if get(f'editing_v2/r_theta_run/hphi_valid128_k912/{i:03d}_H40_hphi_v2_k8-12.json',d2):
        c+=json.load(open(d2))['arms']['restart']['candidates']
    else: miss12.append(i)
    if c:
        c=sorted(c,key=lambda x:x['k'])
        ks=[x['k'] for x in c]
        assert ks==list(range(len(ks))), f'src {i} indices {ks}'
        cands[i]=c
K=max((len(v) for v in cands.values()), default=0)
print(f"sources {len(cands)}/128   missing k0-8 {miss8}   missing k8-12 {len(miss12)}")
print(f"\n{'candidates':<12}"+"".join(f"{k+1:>5}" for k in range(K)))
cum=[sum(1 for v in cands.values() if any(x.get('success') for x in v[:k+1])) for k in range(K)]
print(f"{'cumulative':<12}"+"".join(f"{c:>5}" for c in cum))
print(f"{'increment':<12}"+"".join(f"{(cum[0] if k==0 else cum[k]-cum[k-1]):>5}" for k in range(K)))
n=len(cands)
for k in (7,11):
    if k<K: print(f"\n@{k+1} = {cum[k]}/{n} = {cum[k]/n*100:.1f}%")
print(f"\nGrIDDD 45.1% @20 candidates")
ext=sum(1 for v in cands.values() for x in v if x.get('extinct'))
runs=sum(len(v) for v in cands.values())
print(f"extinct {ext}/{runs} ({ext/runs*100:.1f}%)")
json.dump({'n':n,'K':K,'cumulative':cum,'coverage':{str(k+1):cum[k]/n for k in range(K)}},
          open('docs/VALID128_CURVE.json','w'),indent=1)
