"""Merge banked candidates 1-4 with freshly generated 5-6 and read the curve."""
import json, subprocess, sys
from pathlib import Path
W = Path('/Users/rmaganti/compose_v2_work')
BANK = Path('/tmp/rv5/hphi_recede_v5')
D = Path('/tmp/k56'); D.mkdir(exist_ok=True)

def get(rel, dst):
    if dst.exists(): return True
    subprocess.run(["modal","volume","get","compose-v4-artifacts",rel,str(dst),
                    "--force"], capture_output=True, text=True)
    return dst.exists()

g=json.load(open(W/'docs/HPHI_SMC_64_GATE_BANKED.json'))
l=json.load(open(W/'docs/HPHI_COVERAGE_LADDER_BANKED.json'))
r3=json.load(open(W/'docs/HPHI_LADDER_RUNG3.json'))
rel=set(g['solved_sources']); marg=set(l['rung2']['new_sources'])|set(r3['conversions'])
strat={i:('reliable' if i in rel else 'marginal' if i in marg else 'hard') for i in range(64)}

cands={}
missing=[]
for i in range(64):
    c=[dict(x, k=j) for j,x in enumerate(
        json.load(open(BANK/f'{i:03d}_H40.json'))['arms']['restart']['candidates'])]
    # Each extension slice is appended in candidate order. Slice parity showed
    # a slice reproduces what a full run would have produced at those k, so the
    # glued curve is the curve of a single long run.
    for tag, sub, suf in (("k56", "hphi_recede_v5_k56", "k4-6"),
                          ("k710", "hphi_recede_v5_k710", "k6-10"),
                          ("k1120", "hphi_recede_v5_k1120", "k10-20")):
        dst = D / f'{i:03d}_{tag}.json'
        if get(f'editing_v2/r_theta_run/{sub}/{i:03d}_H40_hphi_v2_{suf}.json', dst):
            c += json.load(open(dst))['arms']['restart']['candidates']
        else:
            missing.append((i, tag))
    c = sorted(c, key=lambda x: x['k'])
    ks = [x['k'] for x in c]
    assert ks == list(range(len(ks))), f'src {i}: candidate indices {ks}'
    cands[i]=c
if missing: print(f"MISSING k56 for {len(missing)} sources: {missing[:12]}")

# ---- slice parity: k0-1 run alone must reproduce banked candidate 1 --------
P=Path('/tmp/kpar'); P.mkdir(exist_ok=True)
ok=bad=absent=0; diffs=[]
for i in range(64):
    dst=P/f'{i:03d}.json'
    if not get(f'editing_v2/r_theta_run/hphi_slice_parity/{i:03d}_H40_hphi_v2_k0-1.json', dst):
        absent+=1; continue
    a=json.load(open(dst))['arms']['restart']['candidates'][0]
    b=cands[i][0]
    if a['returned']==b['returned'] and a['n_transitions']==b['n_transitions']: ok+=1
    else: bad+=1; diffs.append(i)
print(f"\nSLICE PARITY (k=0..1 alone vs banked candidate 1): {ok} match, "
      f"{bad} differ, {absent} absent" + (f"  differing: {diffs[:10]}" if diffs else ""))

K=max(len(v) for v in cands.values())
print(f"\nmax candidates available: {K}")
print(f"\n{'scope':<10}{'of':>4}" + "".join(f"{'@'+str(k+1):>6}" for k in range(K)))
out={'cumulative':{}, 'slice_parity':{'match':ok,'differ':bad,'absent':absent}}
for scope in ('ALL','reliable','marginal','hard'):
    ids=[i for i in range(64) if scope=='ALL' or strat[i]==scope]
    cum=[]
    for k in range(K):
        cum.append(sum(1 for i in ids
                       if any(c.get('success') for c in cands[i][:k+1])))
    out['cumulative'][scope]={'curve':cum,'of':len(ids)}
    print(f"{scope:<10}{len(ids):>4}" + "".join(f"{c:>6}" for c in cum))
print(f"\n{'scope':<10}" + "".join(f"{'+'+str(k+1):>6}" for k in range(K)))
for scope in ('ALL','marginal','hard'):
    c=out['cumulative'][scope]['curve']
    inc=[c[0]]+[c[k]-c[k-1] for k in range(1,K)]
    print(f"{scope:<10}" + "".join(f"{x:>6}" for x in inc))
    out['cumulative'][scope]['increments']=inc
# contact at the new candidates
for scope in ('marginal','hard'):
    ids=[i for i in range(64) if strat[i]==scope]
    for lo,hi,tag in ((0,4,'cand1-4'),(4,K,f'cand5-{K}')):
        if hi<=lo: continue
        tot=sum(1 for i in ids for c in cands[i][lo:hi] if c.get('contact'))
        n=sum(len(cands[i][lo:hi]) for i in ids)
        print(f"contact {scope:<9} {tag:<10} {tot}/{n}")
json.dump(out, open(W/'docs/EXTENDED_CURVE_64.json','w'), indent=1)
print('\nwrote docs/EXTENDED_CURVE_64.json')
