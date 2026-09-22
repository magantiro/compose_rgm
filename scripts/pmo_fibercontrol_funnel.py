import json, sys, collections, math, statistics as st
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from tdc.chem_utils.oracle.oracle import rediscovery_meta
TARGET='CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F'
official=rediscovery_meta(TARGET,fp='ECFP4')

d=json.load(open(sys.argv[1])); snap=d["campaign"]["snapshot"]; hist=snap["history"]
score_by_ep={o["endpoint"]:o["score"] for o in snap["observations"].values()}

def rings(s):
    m=Chem.MolFromSmiles(s); return None if m is None else m.GetRingInfo().NumRings()
def heavy(s):
    m=Chem.MolFromSmiles(s); return None if m is None else m.GetNumHeavyAtoms()

def fam(ac, dR):
    D=ac.get("deleted_original_atoms",0) or 0
    N=ac.get("surviving_new_atoms",0) or 0
    C=ac.get("changed_site_count",0) or 0
    if dR is not None and dR!=0:  return "4 ring/topology change"
    if C>=2:                      return "6 multi-region"
    if D<=1 and N<=1:             return "1 local edit"
    if D==0 and N>=2:             return "2 decoration/add"
    if N==0 and D>=2:             return "3 region delete"
    return "5 region replace"

rows=[]
for ri,r in enumerate(hist):
    b=r["batch"]
    el={c["endpoint"] for c in b["eligible_pool"]["candidates"]}
    sc={c["endpoint"] for c in b["proposal_pool"]["candidates"]}
    for a in b["attempts"]:
        ep=a.get("endpoint"); ac=a.get("actual_changes") or {}
        md=a.get("metadata") or {}
        comp=md.get("dynamic_generic_composition") or md.get("dynamic_v1_structured_program") or {}
        mods=[m.get("family") for m in (comp.get("modules") or [])]
        pscore=a.get("parent_measured_score")
        pctx=a.get("planner_context") or {}
        pep = pctx.get("parent_endpoint") if isinstance(pctx,dict) else None
        dR=None
        if ep and pep:
            re_,rp=rings(ep),rings(pep)
            if re_ is not None and rp is not None: dR=re_-rp
        rows.append(dict(rnd=ri,ep=ep,status=a.get("status"),ch=a.get("planner_channel"),
            prog=a.get("program_size"),parent=pscore,parent_ep=pep,modules=mods,ac=ac,dR=dR,
            mem=bool(md.get("pmo_online_memory")),
            elig=bool(ep and ep in el), scored=bool(ep and ep in sc),
            score=score_by_ep.get(ep)))

print("="*92); print("PHASE 1  FiberControl proposal -> scored funnel  (arm B, celecoxib, 250 calls, 0 new oracle calls)"); print("="*92)
tot=len(rows); prod=[r for r in rows if r["ep"]]
print(f"  attempts {tot} | produced endpoint {len(prod)} ({100*len(prod)/tot:.1f}%) | distinct {len({r['ep'] for r in prod})}"
      f" | eligible {sum(r['elig'] for r in rows)} | SCORED {sum(r['scored'] for r in rows)}")
print(f"  status: {dict(collections.Counter(r['status'] for r in rows))}")

print("\n  --- BY PLANNER CHANNEL ---")
print(f"  {'channel':34s} {'attempts':>8} {'endpoint':>8} {'eligible':>8} {'scored':>7} {'%of attempts':>12} {'%of scored':>10}")
ns=sum(r['scored'] for r in rows)
for ch,c in collections.Counter(r["ch"] for r in rows).most_common():
    g=[r for r in rows if r["ch"]==ch]
    print(f"  {ch:34s} {len(g):8d} {sum(1 for r in g if r['ep']):8d} {sum(r['elig'] for r in g):8d} "
          f"{sum(r['scored'] for r in g):7d} {100*len(g)/tot:11.1f}% {100*sum(r['scored'] for r in g)/ns:9.1f}%")

for r in prod: r["fam"]=fam(r["ac"], r["dR"])
print("\n  --- BY REALIZED STRUCTURAL FAMILY (from graph delta, not module names) ---")
print(f"  {'family':26s} {'attempts':>8} {'distinct':>8} {'eligible':>8} {'scored':>7} {'mean score':>11} {'mean d vs parent':>17} {'frontier':>9}")
best=0.0; frontier=collections.Counter()
ordered=[r for r in prod if r["scored"]]
for r in sorted(ordered,key=lambda x:x["rnd"]):
    if r["score"] is not None and r["score"]>best: best=r["score"]; frontier[r["fam"]]+=1
for f in sorted({r["fam"] for r in prod}):
    g=[r for r in prod if r["fam"]==f]; s=[r for r in g if r["scored"] and r["score"] is not None]
    ds=[r["score"]-r["parent"] for r in s if r["parent"] is not None]
    print(f"  {f:26s} {len(g):8d} {len({r['ep'] for r in g}):8d} {sum(r['elig'] for r in g):8d} {len(s):7d} "
          f"{(st.mean(r['score'] for r in s) if s else float('nan')):11.4f} {(st.mean(ds) if ds else float('nan')):17.4f} {frontier[f]:9d}")

print("\n  --- MODULE FAMILY (cross-check, names not delta) ---")
mc=collections.Counter(m for r in prod for m in r["modules"])
for k,v in mc.most_common(24): print(f"    {k:32s} {v}")
