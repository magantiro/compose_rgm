import json, subprocess, sys
from pathlib import Path
PANEL={"reliable":[3,15,29,60],"marginal":[5,22,46,49],"hard":[0,25,42,63]}
D=Path('/tmp/ab_v10'); D.mkdir(exist_ok=True)
def get(rel,dst):
    if dst.exists(): return True
    r=subprocess.run(["modal","volume","get","compose-v4-artifacts",rel,str(dst),"--force"],
                     capture_output=True,text=True)
    return dst.exists()
rows=[]
for st,idxs in PANEL.items():
    for i in idxs:
        rec={}
        for arm,hd in (("old","hphi_v2"),("new","hphi_v2_h40")):
            rel=f"editing_v2/r_theta_run/hphi_h40head_v10/{i:03d}_H40_{hd}.json"
            dst=D/f"{i:03d}_{arm}.json"
            if not get(rel,dst): rec[arm]=None; continue
            a=json.load(open(dst))["arms"]["restart"]
            rec[arm]=a
        rows.append((st,i,rec))
print(f"{'stratum':<10}{'src':>4}  {'OLD success':>12}{'NEW success':>12}"
      f"{'OLD contact':>13}{'NEW contact':>13}{'OLD first':>10}{'NEW first':>10}")
agg={}
for st,i,r in rows:
    o,n=r.get("old"),r.get("new")
    if not o or not n: print(f"{st:<10}{i:>4}  MISSING"); continue
    oc=sum(1 for c in o["candidates"] if c.get("contact"))
    nc=sum(1 for c in n["candidates"] if c.get("contact"))
    a=agg.setdefault(st,{"n":0,"os":0,"ns":0,"oc":0,"nc":0})
    a["n"]+=1; a["os"]+=int(o["success"]); a["ns"]+=int(n["success"])
    a["oc"]+=oc; a["nc"]+=nc
    print(f"{st:<10}{i:>4}  {str(o['success']):>12}{str(n['success']):>12}"
          f"{oc:>10}/4{nc:>10}/4{str(o['first_success_at']):>10}{str(n['first_success_at']):>10}")
print(f"\n{'stratum':<10}{'sources':>8}{'OLD succ':>10}{'NEW succ':>10}"
      f"{'OLD contact':>14}{'NEW contact':>14}")
T={"n":0,"os":0,"ns":0,"oc":0,"nc":0}
for st in ("reliable","marginal","hard"):
    if st not in agg: continue
    a=agg[st]
    for k in T: T[k]+=a[k]
    print(f"{st:<10}{a['n']:>8}{a['os']:>10}{a['ns']:>10}"
          f"{a['oc']:>11}/{a['n']*4}{a['nc']:>11}/{a['n']*4}")
print(f"{'TOTAL':<10}{T['n']:>8}{T['os']:>10}{T['ns']:>10}"
      f"{T['oc']:>11}/{T['n']*4}{T['nc']:>11}/{T['n']*4}")
print(f"\nDECISION STRATA (marginal+hard): "
      f"old {agg.get('marginal',{}).get('os',0)+agg.get('hard',{}).get('os',0)}"
      f"/{agg.get('marginal',{}).get('n',0)+agg.get('hard',{}).get('n',0)} sources, "
      f"new {agg.get('marginal',{}).get('ns',0)+agg.get('hard',{}).get('ns',0)}"
      f"/{agg.get('marginal',{}).get('n',0)+agg.get('hard',{}).get('n',0)}")
