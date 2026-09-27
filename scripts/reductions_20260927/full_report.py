import json, re, statistics as st, math
from datetime import datetime
try: from scipy import stats as sp; HAVE=True
except Exception: HAVE=False
H=__import__("pathlib").Path.home()
# ---------- T4 ----------
DELTA={'43c59252':0.6,'b1126443':0.4,'10da2bfc':0.4,'c89be1e8':0.6,'0dd3cb00':0.4,
 '59d59e1b':0.6,'a6b780b2':0.6,'dcf0d2ba':0.4,'8593e6a8':0.6,'86d34bfb':0.4,
 '20064ca1':0.4,'0fa661d9':0.4,'8272b6ac':0.6,'08233d44':0.6,'a9f80eed':0.4,'ee150ed2':0.6}
panel=json.load(open('/tmp/t4_panel4.json')); rev=json.load(open('/tmp/t4_rev4.json'))
for r in rev:
    if r.get('delta') is not None: DELTA.setdefault(r['run'], r['delta'])
live={}
for r in panel+rev:
    dd=DELTA.get(r['run']) or r.get('delta')
    m=re.match(r"(.+)_(\d)_r(\d)$", r['cell'])
    if not m or dd is None: continue
    k=(m.group(1),int(m.group(2)),dd,int(m.group(3))); p=live.get(k)
    if p is None or (r.get('done') and not p.get('done')) or \
       (bool(r.get('done'))==bool(p.get('done')) and (r.get('charged') or 0)>(p.get('charged') or 0)):
        live[k]=r
PROT={'PARP1':'parp1','FA7':'fa7','5HT1B':'5ht1b','BRAF':'braf','JAK2':'jak2'}
BASE={'parp1':0,'fa7':3,'5ht1b':6,'braf':9,'jak2':12}
froz={}; cur=None
for ln in open('diagnostics/T4_FROZEN_RESULT_v1.md'):
    h=re.match(r"##\s*δ\s*=\s*(0\.\d)", ln)
    if h: cur=float(h.group(1)); continue
    if ln.startswith("## "): cur=None
    m=re.match(r"\|\s*(PARP1|FA7|5HT1B|BRAF|JAK2)\s*\|\s*(\d)\s*\|\s*\**(-?[\d.]+|—)\**\s*\|\s*\**(-?[\d.]+)\**", ln)
    if m and cur is not None:
        froz[(PROT[m.group(1)],int(m.group(2))-1,cur)]=(None if m.group(3)=='—' else float(m.group(3)), float(m.group(4)))
BL={}
for r in json.load(open('diagnostics/t4_combined_table.json'))['rows']:
    BL[(r['target'],r['seed'],r['delta'])]=r
print(f"# COMPOSE results report — {datetime.now().strftime('%Y-%m-%d %H:%M')} local\n")
print("**Status.** T4: 10 of 60 panel cells complete, 11 of 15 revival cells complete, panel ETA ~37 h")
print("(braf-bound). PMO A/B: 14 of 18 matched pairs complete, 2 running (~9 h), 2 failed.")
print("PMO C: 17 of 18 complete, 1 failed. **Nothing here is a finished panel; every unfinished value")
print("is best-so-far and, since lower/higher is better respectively, can only improve.**\n")
print("---\n## 1. T4 — constrained lead optimization\n")
print("QuickVina2 docking score of the best eligible molecule (kcal/mol, lower is better), subject to")
print("QED >= 0.6, SA <= 4.0, Tanimoto >= delta to the starting lead. 250 charged oracle calls per cell")
print("against InVirtuoGen's 1,000 and GenMol's 3,000. Replicate 1 is the frozen complete panel;")
print("replicates 2 and 3 are still running. `D` = that replicate finished, `NNr` = rounds so far.\n")
for dd in (0.4,0.6):
    print(f"### delta = {dd}\n")
    print("| Target | Lead | Start | GraphGA | RetMol | GenMol | InVirtuoGen | COMPOSE seed 1 | COMPOSE mean of 3 | r2 | r3 | state |")
    print("|:---|:-:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|")
    S=[];M=[];I=[];G=[];wi=[0,0];wg=[0,0];bothS=[];bothM=[]
    for prot in ('PARP1','FA7','5HT1B','BRAF','JAK2'):
        p=PROT[prot]
        for cid in (0,1,2):
            f1,ivg=froz.get((p,cid,dd),(None,None))
            v2=live.get((p,cid,dd,2)) or {}; v3=live.get((p,cid,dd,3)) or {}
            vals=[f1,v2.get('best'),v3.get('best')]
            got=[v for v in vals if isinstance(v,(int,float))]
            mn=st.mean(got) if got else None
            bb=BL.get((p,BASE[p]+cid,dd),{}); gen=bb.get('genmol')
            row={'graphga':bb.get('graphga'),'retmol':bb.get('retmol'),'genmol':gen,'ivg':ivg,'mn':mn}
            nums=[v for v in row.values() if isinstance(v,(int,float))]
            best=min(nums) if nums else None
            def f(v):
                if not isinstance(v,(int,float)): return "–"
                s=f"{v:.1f}"
                return f"**{s}**" if (best is not None and abs(v-best)<1e-9) else s
            g=lambda v: f"{v:.1f}" if isinstance(v,(int,float)) else "–"
            s2="D" if v2.get('done') else (f"{v2.get('rounds')}r" if v2 else "–")
            s3="D" if v3.get('done') else (f"{v3.get('rounds')}r" if v3 else "–")
            print(f"| {prot} | {cid+1} | {g(bb.get('seed_score'))} | {f(row['graphga'])} | {f(row['retmol'])} | "
                  f"{f(gen)} | {f(ivg)} | {g(f1)} | {f(mn)} | {g(v2.get('best'))} | {g(v3.get('best'))} | {s2},{s3} |")
            if isinstance(f1,(int,float)): S.append(f1)
            if isinstance(mn,float):
                M.append(mn)
                if ivg is not None: I.append(ivg); wi[0]+=mn<ivg; wi[1]+=mn>ivg
                if gen is not None: G.append(gen); wg[0]+=mn<gen; wg[1]+=mn>gen
            if isinstance(f1,(int,float)) and isinstance(mn,float): bothS.append(f1); bothM.append(mn)
    print(f"| **Mean** | | | | | **{st.mean(G):.2f}** | **{st.mean(I):.2f}** | "
          f"**{st.mean(S):.2f}** | **{st.mean(M):.2f}** | | | |")
    print(f"\n- COMPOSE mean of 3: **vs InVirtuoGen {wi[0]}W {wi[1]}L of {len(I)}**, "
          f"**vs GenMol {wg[0]}W {wg[1]}L of {len(G)}**")
    print(f"- seed 1 mean {st.mean(S):.2f} (n={len(S)}); mean-of-3 {st.mean(M):.2f} (n={len(M)}); "
          f"like-for-like on the {len(bothS)} shared cells: seed 1 {st.mean(bothS):.2f} vs "
          f"mean-of-3 {st.mean(bothM):.2f} ({st.mean(bothM)-st.mean(bothS):+.2f})\n")
print("### T4 statistics — paired across cells, COMPOSE mean of 3 minus baseline\n")
print("| baseline | delta | n | mean diff | s.d. | paired t p | Wilcoxon p | sign test |")
print("|---|:-:|:-:|--:|--:|--:|--:|--:|")
for base in ('InVirtuoGen','GenMol'):
    for dd in (0.4,0.6,None):
        d=[]
        for prot in PROT.values():
            for cid in (0,1,2):
                for ddd in ((dd,) if dd else (0.4,0.6)):
                    f1,ivg=froz.get((prot,cid,ddd),(None,None))
                    vals=[f1]+[(live.get((prot,cid,ddd,rp)) or {}).get('best') for rp in (2,3)]
                    got=[v for v in vals if isinstance(v,(int,float))]
                    b = ivg if base=='InVirtuoGen' else BL.get((prot,BASE[prot]+cid,ddd),{}).get('genmol')
                    if got and b is not None: d.append(st.mean(got)-b)
        if len(d)<2: continue
        tp=f"{sp.ttest_1samp(d,0).pvalue:.4f}" if HAVE else "–"
        wp=f"{sp.wilcoxon(d).pvalue:.4f}" if HAVE else "–"
        sg=f"{sum(1 for x in d if x<0)}/{len(d)}"
        print(f"| {base} | {dd if dd else 'both'} | {len(d)} | {st.mean(d):+.3f} | {st.pstdev(d):.3f} | {tp} | {wp} | {sg} |")
# ---------- PMO ----------
A=json.load(open(H/"compose_pmo_ablation/inputs/pmo_1k_final.json"))
B={(r["task"],r["seed"]):r for r in json.load(open("/tmp/armB_final.json"))}
C={(r["task"],r["seed"]):r for r in json.load(open("/tmp/armC_metrics.json"))}
TASKS=["albuterol_similarity","celecoxib_rediscovery","gsk3b","isomers_c9h10n2o2pf2cl","ranolazine_mpo","scaffold_hop"]
a=lambda t,s,k:(A.get(t,{}).get(str(s)) or {}).get(k)
print("\n---\n## 2. PMO — structured proposals vs length-matched uniform legal edit chains\n")
print("**A (COMPOSE)** structured molecular proposals, unchanged configuration, pre-existing campaigns")
print("reused (0 new oracle calls). **B** the arm-A synthesis supplies the realized primitive-edit count,")
print("the program is discarded and replaced by that many uniform draws from the legal Active8 fiber,")
print("recomputed after every intermediate. Executor, endpoint eligibility, candidate pool, initialization")
print("and the online selection/allocation rules are retained in both arms. 6 objectives x 3 seeds,")
print("1,008-call budget, metrics on the first 1,000 resolved receipts, no-prescreen.\n")
at=[];bt=[];aa=[];ba=[]
for t in TASKS:
    for s in sorted({s for (tt,s) in B if tt==t}):
        b=B.get((t,s),{})
        if b.get("state")!="COMPLETE" or not isinstance(a(t,s,"top10"),float): continue
        at.append(a(t,s,"top10")); bt.append(b["top10"]); aa.append(a(t,s,"auc")); ba.append(b["auc"])
print("| Proposal mechanism | Final Top-10 | AUC-Top10 |")
print("|---|--:|--:|")
print(f"| **COMPOSE (structured proposals)** | **{st.mean(at):.3f}** | **{st.mean(aa):.3f}** |")
print(f"| Uniform legal edit chains | {st.mean(bt):.3f} | {st.mean(ba):.3f} |")
print(f"| *difference* | *{st.mean(bt)-st.mean(at):+.3f}* | *{st.mean(ba)-st.mean(aa):+.3f}* |")
print(f"\nMean over {len(at)} matched objective-seed cells.\n")
print("| metric | n | mean B−A | median | s.d. | A better | paired t p | Wilcoxon p |")
print("|---|:-:|--:|--:|--:|:-:|--:|--:|")
for lab,X,Y in (("Final Top-10",bt,at),("AUC-Top10",ba,aa)):
    d=[p-q for p,q in zip(X,Y)]
    tp=f"{sp.ttest_1samp(d,0).pvalue:.4f}" if HAVE else "–"
    wp=f"{sp.wilcoxon(d).pvalue:.4f}" if HAVE else "–"
    print(f"| {lab} | {len(d)} | {st.mean(d):+.4f} | {st.median(d):+.4f} | {st.pstdev(d):.4f} | "
          f"{sum(1 for v in d if v<0)}/{len(d)} | {tp} | {wp} |")
print("\n### Per-objective\n")
print("| objective | seeds | A Top-10 | B Top-10 | d Top-10 | A AUC | B AUC | d AUC |")
print("|---|:-:|--:|--:|--:|--:|--:|--:|")
for t in TASKS:
    L={'at':[],'bt':[],'aa':[],'ba':[]}
    for s in sorted({s for (tt,s) in B if tt==t}):
        b=B.get((t,s),{})
        if b.get("state")!="COMPLETE" or not isinstance(a(t,s,"top10"),float): continue
        L['at'].append(a(t,s,"top10")); L['bt'].append(b["top10"]); L['aa'].append(a(t,s,"auc")); L['ba'].append(b["auc"])
    if not L['at']: continue
    m={k:st.mean(v) for k,v in L.items()}
    print(f"| {t} | {len(L['at'])} | {m['at']:.4f} | {m['bt']:.4f} | {m['bt']-m['at']:+.4f} | "
          f"{m['aa']:.4f} | {m['ba']:.4f} | {m['ba']-m['aa']:+.4f} |")
print(f"| **all** | **{len(at)}** | **{st.mean(at):.4f}** | **{st.mean(bt):.4f}** | "
      f"**{st.mean(bt)-st.mean(at):+.4f}** | **{st.mean(aa):.4f}** | **{st.mean(ba):.4f}** | "
      f"**{st.mean(ba)-st.mean(aa):+.4f}** |")
print("\n### Full per-cell, all 18 cells\n")
print("| objective | seed | B state | A best | A Top-10 | A AUC | B best | B Top-10 | B AUC | d AUC |")
print("|---|--:|:--|--:|--:|--:|--:|--:|--:|--:|")
F=lambda v: f"{v:.4f}" if isinstance(v,float) else "–"
for t in TASKS:
    for s in sorted({s for (tt,s) in B if tt==t}):
        b=B.get((t,s),{})
        dd=(f"{b['auc']-a(t,s,'auc'):+.4f}" if isinstance(b.get("auc"),float) and isinstance(a(t,s,"auc"),float) else "–")
        print(f"| {t} | {s} | {b.get('state','–')} | {F(a(t,s,'best'))} | {F(a(t,s,'top10'))} | {F(a(t,s,'auc'))} | "
              f"{F(b.get('best'))} | {F(b.get('top10'))} | {F(b.get('auc'))} | {dd} |")
cd=[]
for t in TASKS:
    for s in sorted({s for (tt,s) in C if tt==t}):
        c=C.get((t,s),{})
        if c.get("state")=="COMPLETE" and isinstance(a(t,s,"auc"),float) and isinstance(c.get("auc"),float):
            cd.append(c["auc"]-a(t,s,"auc"))
print("\n---\n## 3. PMO secondary experiment (arm C) — appendix only\n")
print("Arm C keeps the structured recipe intact (operation sequence, length, block boundaries, payloads and")
print("every binding to an original source atom preserved byte-for-byte) and resamples only the operands")
print("referring to atoms created by earlier operations, among the currently live created atoms that leave")
print("the operation admissible.\n")
tp=f"{sp.ttest_1samp(cd,0).pvalue:.3f}" if HAVE else "–"; wp=f"{sp.wilcoxon(cd).pvalue:.3f}" if HAVE else "–"
print(f"AUC-Top10 paired against A over {len(cd)} completed cells: mean **{st.mean(cd):+.4f}**, median "
      f"{st.median(cd):+.4f}, A better on {sum(1 for v in cd if v<0)}/{len(cd)}, t p={tp}, Wilcoxon p={wp}.\n")
print("The intervention produced a smaller mean AUC reduction than the broad proposal replacement, with")
print("substantial variation across objectives and runs. **This comparison does not resolve how much of the")
print("full proposer's advantage is specifically attributable to these bindings**, and its effect must not be")
print("treated as an additive fraction of the A/B effect.\n")
print("---\n## 4. Caveats that must travel with these numbers\n")
print("1. **Nothing is a finished panel.** T4 replicates 2-3 and 2 arm-B campaigns are running; every")
print("   unfinished value is best-so-far and can only improve.")
print("2. **`gsk3b` is a reward-hackable ML-predictor oracle in every PMO arm.** Its leaders are off-manifold")
print("   (QED 0.03-0.23, SA 6.2-6.6, bare phosphorus, stacked hydrazines). Not a chemistry result for any arm.")
print("3. **The PMO effect is task-dependent**: clear on albuterol, gsk3b, ranolazine, celecoxib; neutral to")
print("   slightly negative on isomers and scaffold_hop.")
print("4. **B inherits arm A's synthesis-success filter, its realized edit length, and its arbitration random")
print("   stream.** It replaces edit choices, construction priors and program reuse together; it does not")
print("   isolate dependency scheduling.")
print("5. **AUC-Top10 trapezoids from (0,0)**, so a 1,000-call AUC is structurally depressed ~5% and must")
print("   never be compared to a published 10,000-call figure.")
print("6. **T4 docking is not reproducible at the single-molecule level**: re-docking one molecule spans")
print("   1.3 kcal/mol because `obabel --gen3D` is unseeded. Replicate s.d. on a cell mean is 0.46 (median),")
print("   which is the right per-cell uncertainty; do not bold a single-run cell on a sub-1.3 margin.")
print("7. **Two named phases sit inside the T4 COMPOSE columns**: the revival arm (cells re-run after the")
print("   round-0 preemption window) and, in replicate 1, several support-expansion cells.")
print("8. **T4 GraphGA/RetMol coverage is thin** (2-10 cells, older combined table); verify provenance or drop.")
print("9. **No PMO arm's proposal-synthesis path is in the contract `implementation_sha256`.** Arm identity")
print("   rests on the launch receipt `git_commit` plus the deployed image.\n")
print("---\n## 5. Durable artifacts\n")
print("- PMO A/B/C reduction: `compose_pmo_chain/diagnostics/pmo_abc_ablation_v1/reduction_v1.json` (commit `3e01fed6`)")
print("- Arm C verification: `compose_pmo_chain/diagnostics/pmo_armc_verification_v1/VERIFICATION.md` (commit `9af2beb8`)")
print("- Arm C frozen implementation: commit `e11cd89d`, branch `pmo-chain-ablation-20260925` (pushed)")
print("- T4 frozen replicate 1: `compose_t4_nitya/diagnostics/T4_FROZEN_RESULT_v1.{json,md}`")
