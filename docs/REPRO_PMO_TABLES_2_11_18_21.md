# Reproduction handoff — PMO Tables 2, 11 and 18–21, ICLR 2027 submission

Read-only. No oracle calls, no launches, no edits to frozen artifacts, no
contract re-pinning. Scope is
`/Users/rmaganti/Desktop/19340_COMPOSE_Molecular_Genera.pdf`
(sha256 `a609e9dc9c43ad22d8b4fa2a…`; **differs** from
`COMPOSE_ICLR_2027_zip5_appendix_results/main.pdf`). Tables 2, 11, 18, 19, 20
and 21 were read from the PDF and their values verified against artifacts
below.

---

## 1. Two different experiments, two different artifacts

| Paper | What | Backing artifact |
|---|---|---|
| **Tables 2, 11** | 22-objective PMO-1K headline, 3 seeds | `diagnostics/pmo_ablation_frozen_v1/inputs/pmo_1k_final.json` |
| **Tables 18–21** | structured-proposal A/B ablation, 6 objectives | `diagnostics/pmo_ablation_frozen_v1/frozen_config_v1.json` (protocol) + `pmo_1k_final.json` (arm A) + `diagnostics/compose_status_report/inputs/armB_final.json` (arm B) |

`pmo_1k_prov.json` sits beside `pmo_1k_final.json` and, despite the name,
**contains scores, not provenance** — same `{task: {seed: {best, top10, auc}}}`
shape. Do not expect namespaces from it.

---

## 2. Tables 2 and 11 — the 22-objective headline

### Reduction rule, verified

`pmo_1k_final.json` holds 23 objectives. Five carry a **fourth** seed
(albuterol, celecoxib, mestranol, thiothixene, troglitazone); the paper reports
three. **Dropping the seed whose id ends `983`** reproduces the published mean
*and* standard deviation on all five. The reported mean spans 22 objectives and
excludes `valsartan_smarts`.

Verified this session: **21 of 22 objectives match Table 2 exactly**, 21 of 22
match Table 11 exactly, and **both headline means reproduce — 0.563 and 0.482.**

```bash
python3 - <<'PY'
import json, statistics
d=json.load(open('diagnostics/pmo_ablation_frozen_v1/inputs/pmo_1k_final.json'))
sel=lambda t: {k:v for k,v in d[t].items() if not k.endswith('983')} if len(d[t])==4 else d[t]
t10=[];auc=[]
for t in sorted(d):
    s=sel(t); a=statistics.fmean([x['top10'] for x in s.values()])
    b=statistics.fmean([x['auc'] for x in s.values()])
    if t!='valsartan_smarts': t10.append(a); auc.append(b)
    print(f"{t:<26}{a:.3f}+-{statistics.stdev([x['top10'] for x in s.values()]):.3f}"
          f"   {b:.3f}+-{statistics.stdev([x['auc'] for x in s.values()]):.3f}")
print(f"MEAN(22) top10={statistics.fmean(t10):.4f}  auc={statistics.fmean(auc):.4f}")
PY
```
Expected: `MEAN(22) top10=0.5632  auc=0.4818` → paper's **0.563** and **0.482**.

### The JNK3 row — investigated, not substituted

Committed JNK3 seeds and values:

| seed | top-10 | AUC |
|---|---|---|
| 20269925 | 0.2870 | 0.2441 |
| 20269926 | 0.2410 | 0.1967 |
| 20270003 | 0.2100 | 0.1886 |

As a set of three: **0.2460 ± 0.0387** top-10, **0.2098 ± 0.0300** AUC.
The paper prints **0.245 ± 0.041** and **0.208 ± 0.032**. So the committed trio
is *not* the trio the paper used.

Solving for a single unknown replacement `x` under the printed mean, then
checking the printed SD as an **independent** constraint:

| dropped | top-10 x → SD | AUC x → SD |
|---|---|---|
| 20269925 | 0.2840 → 0.0372 (paper 0.041) | 0.2387 → 0.0269 (paper 0.032) |
| 20269926 | 0.2380 → 0.0390 | 0.1913 → 0.0313 |
| **20270003** | **0.2070 → 0.0401** ✓ | **0.1831 → 0.0320** ✓ |

**Only dropping `20270003` is consistent on both metrics simultaneously.** The
implied replacement has top-10 ≈ **0.2070** and AUC ≈ **0.1831**.

**Candidate, not confirmed:** a fourth JNK3 run exists —
`scored_jnk3_seed20269928_mscore_20260924T115852Z`, volume
`pmo-fiber-jnk3-seed20269928-mscore`, call `fc-01M39MGSWJ1ZXB3HEWAXV4X0VA`,
budget 3000, replicate 4, campaign `pmo_learned_trio_v1`. **Its score is not
committed anywhere in the repository**, so I cannot confirm it is the source of
0.2070/0.1831. **I have not substituted it.** Resolving this requires reading
that volume's `canary_v1.json` or its first 1,000 `oracle/query_*/result.json`
receipts.

I found **no artifact describing an "815-call provisional" JNK3 run.** Marked
MISSING — the string does not appear in the frozen PMO record, and JNK3 is not
one of the six ablation objectives whose receipt counts are documented.

### Per-seed provenance for Tables 2/11 — largely MISSING

Joining `pmo_1k_final.json`'s 74 seed-entries against every committed launch
receipt: **33 matched, 41 unmatched.**

| seed suffix | matched | missing | note |
|---|---|---|---|
| `…923` | 20 | 0 | replicate 0; the 10,000-budget campaigns |
| `…925` | 5 | 18 | replicate 1 |
| `…926` | 3 | 20 | replicate 2 |
| `…983` | 5 | 0 | the dropped replicate |
| `…003` | 0 | 3 | incl. gsk3b 20269003 |

Matched example (albuterol 20261923): campaign `pmo_fibercontrol_targets_v1`,
volume `pmo-fiber-albuterol-similarity-seed20261923`, budget **10000**,
commit `3e6baf94`, call `fc-01M37WZCX2VEKDK3S7HN7R66QE`. Namespace format is
`scored_<task>_seed<SEED>_<flag>_<UTC>`.

**Budgets are 10,000 or 3,000, not 1,000.** The PMO-1K metrics are taken from
the **first 1,000 resolved receipts in charge order** of longer runs — see §3.

---

## 3. Tables 18–21 — the structured-proposal ablation

`frozen_config_v1.json` is the authoritative protocol record.

### Arms

- **A** — structured programs, unchanged COMPOSE configuration; **pre-existing
  completed campaigns reused, 0 new oracle calls**.
- **B** — length-matched uniform legal edit chains. Arm-A synthesis supplies the
  realized primitive count `K`; the program is then discarded and replaced by
  `K` uniform draws from the **recomputed** legal Active8 fiber.
  **18,144 new oracle calls.**
- **C** — created-atom rebinding (a *third* arm, not in the paper's Tables
  18–21): the structured recipe is kept byte-for-byte and only operands
  referring to earlier-created atoms are resampled.

Run budget **1,008** charged calls; reporting prefix **1,000**.

### Charged-call accounting — read this before counting anything

> `authoritative_charged_count`: resolved oracle receipts —
> `oracle/query_NNNNNN/` containing `result.json`, exactly `ProgramQueryLedger.rows`.
> `do_not_use`: `progress.json:charged_oracle_calls` — a **per-round snapshot
> that lags** the final calls.

Measured evidence in the record: gsk3b 20269003 shows `progress_charged: 988`
against `query_dirs: 1000`; albuterol 20261923 shows 5,991 receipts against
5,975 charged in its own snapshot. **The 988 is a lagging snapshot, not a
shortfall.**

Arm-A receipts per seed (`18 of 18 PASS`, requirement ≥1,000):

```
albuterol   20261923:5991  20261925:1008  20261983:3367
celecoxib   20260923:4199  20260925:1008  20260926:1008
gsk3b       20268925:1008  20268926:1008  20269003:1000
isomers_C9  20280923:4433  20280925:1008  20280926:1008
ranolazine  20274923:2615  20274925:1008  20274926:1008
scaffold    20279923:2880  20279925:1008  20279926:1008
```

**A correction already recorded in the artifact, which must not be undone.** An
earlier verification pass wrongly reported albuterol 20261923 as a 112-call
smoke and substituted 20261983, because the verifier skipped any namespace whose
`provenance.json` was absent (`if p.get('returncode') is None: continue`) and
the real run carries none. *"The substitution is WITHDRAWN and the reported
albuterol row 0.782 ± 0.020 stands."* Also: ranolazine 20274925/20274926 appear
on workspace `nitya` as 96-budget smokes with 0 receipts; those are **superseded**
by the `kosha-labs` 1008-receipt runs of the same seeds.

### The 14 completed matched pairs — verified

Arm B's 18 planned runs resolve as **14 COMPLETE, 2 FAILED, 2 PARTIAL**:

| task | seed | state | cause |
|---|---|---|---|
| albuterol_similarity | 20261923 | **FAILED** | `IndexError` in `pmo_contextual_macro.edge_features` (shared contextual value model, triggered by arm-B molecules). **Arm-dependent, not random attrition.** 64 charged. |
| celecoxib_rediscovery | 20260923 | **FAILED** | pending round needs receipt-based recovery after preemption. 431 charged. |
| ranolazine_mpo | 20274923 | **PARTIAL@512** | did not reach the reporting budget |
| ranolazine_mpo | 20274925 | **PARTIAL@528** | did not reach the reporting budget |

18 − 4 = **14**, and the four excluded seeds are exactly those absent from
Table 21. All four have `auc: null`.

**Table 21 verified: 32 of 32 printed values matched** (8 rows × {A,B} ×
{Top-10, AUC}) against `pmo_1k_final.json` (arm A) and `armB_final.json`
(arm B), with zero mismatches.

### 14 vs 17 vs "16"

- The paper's B−A comparison is **n = 14**, matching Table 20.
- `diagnostics/pmo_abc_ablation_v1/reduction_v1.json` gives
  `auc_B_minus_A.n = 14` (agrees) and **`auc_C_minus_A.n = 17`** — arm C
  completed 17 of 18 (ranolazine 20274925 FAILED at 400 charged).
- **A "16-pair reduction" is not present in any committed artifact.** The
  arithmetic that would produce 16 is visible: 18 planned − 2 hard FAILED = 16,
  reachable only if the two PARTIAL ranolazine B runs later completed. The
  latest artifact (`armB_final.json`, 2026-09-26) still shows them
  **PARTIAL@512 / PARTIAL@528**. Marked **MISSING**: either it postdates every
  artifact here, or it refers to the 17-cell arm-C reduction.
- `reduction_v1.json` carries `STATUS: PROVISIONAL … Do NOT publish as final.`
  **It is not the paper's table.**

### Reduction command

```bash
python3 - <<'PY'
import json, statistics
A=json.load(open('diagnostics/pmo_ablation_frozen_v1/inputs/pmo_1k_final.json'))
B={(r['task'],r['seed']):r for r in
   json.load(open('diagnostics/compose_status_report/inputs/armB_final.json'))}
pairs=[(t,int(s)) for (t,s) in ((t,s) for t in A for s in A[t])
       if (t,int(s)) in B and B[(t,int(s))]['state']=='COMPLETE'
       and B[(t,int(s))].get('auc') is not None]
for m in ('top10','auc'):
    a=[A[t][str(s)][m] for t,s in pairs]; b=[B[(t,s)][m] for t,s in pairs]
    d=[y-x for x,y in zip(a,b)]
    print(f"{m}: n={len(d)}  A={statistics.fmean(a):.4f}  B={statistics.fmean(b):.4f} "
          f" B-A mean={statistics.fmean(d):+.4f} median={statistics.median(d):+.4f} "
          f" A higher={sum(1 for x in d if x<0)}/{len(d)}")
PY
```
Expected, matching Tables 18 and 20: A 0.621 / B 0.558 on Top-10, A 0.503 /
B 0.465 on AUC; B−A mean −0.0630 (Top-10) and −0.0379 (AUC); A higher in 10 of
14 on each.

---

## 4. Is a learned reference numerically used in PMO? **No.**

Stated explicitly in `frozen_config_v1.json → known_asymmetries_to_disclose`:

> there is **NO learned rate model on the scored PMO path** (verified: zero
> `torch.load` / rate-model instantiation in the import closure), so the "frozen
> local reference" for the chain arm is the **uniform law over the legal Active8
> fiber**, not a learned `R_θ`.

Two asymmetries disclosed alongside it:

- the **jump lane reads a teacher-derived plan library**
  (`diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json`,
  `fit_scope shared_all_routes`) for which the chain arm has no analogue;
  measured footprint **~0.39% of attempts**;
- the credit feature vector carries one-hot channel indicators; anchored and
  transplant already encode as all-zero, so a chain proposal inherits an
  existing valid encoding rather than a novel missing-feature penalty.

This applies to **the scored PMO procedure in Tables 2, 11 and 18–21.** It does
**not** contradict the paper's §3.4 statement that "the GuacaMol-trained `R_θ`
and rewrite system remained fixed" — `R_θ` is fixed and the rewrite system is
shared, but no learned rate model is *evaluated numerically* on this path.
Anyone restating the PMO result should say which of the two they mean.

Arm-A closure identity: *"every arm-A run is byte-identical on the scored PMO
import closure"*, commits pooled `3e6baf94`, `c7fdddca`, `0b9666f4`, method =
import closure taken by execution intersected with `git diff --name-only`
between each commit pair, **0 closure files changed**.

---

## 5. Environment

```bash
python 3.11 · rdkit 2023.9.6 · PyTDC 1.1.15 (--no-deps) · numpy 1.26.4 · setuptools 69.5.1
```

PyTDC 1.1.15 pins `rdkit>=2023.9.5,<2024.3.1`, so installing against 2024.3.5 is
unsatisfiable; production adds an `rdkit.six` shim. `setuptools<81` because
`tdc/metadata.py` imports `pkg_resources`. The image holds **one** rdkit, so the
PMO runtime — oracle *and* executor — is 2023.9.6, a different kernel from the
rest of the paper.

Oracle assets: `drd2`, `gsk3b`, `jnk3` load a pickle from a **relative** path.
`gsk3b` and `drd2` load **lazily on first call**; `jnk3` loads eagerly at
construction. `tdc.Oracle.__call__` wraps the evaluator in a bare `except`
returning `0.0`, so a missing asset yields a silent all-zero ledger. Any rerun
must assert a **known active scores > 0** before the first charged call.

None of the reduction commands in this document need PyTDC or an oracle.

---

## 6. Evidence and limitations ledger

| Item asked for | Status | Detail |
|---|---|---|
| Table 2 per-objective → artifact | **verified** | 21/22 exact; JNK3 investigated, §2 |
| Table 11 per-objective → artifact | **verified** | 21/22 exact |
| Both headline means | **verified** | 0.5632 → 0.563; 0.4818 → 0.482 |
| Seed-selection rule | **verified** | drop `…983`; reproduces mean and SD on all five 4-seed objectives |
| Table 21's 14 pairs | **verified** | 32/32 printed values matched |
| Tables 18/20 aggregates | **verified** | reduction command §3 reproduces them |
| 4 excluded B cells + causes | **verified** | §3; 2 FAILED, 2 PARTIAL@512/@528 |
| 14 vs 17 vs 16 | **verified / MISSING** | 14 (paper, B−A) and 17 (C−A) confirmed; **16 not in any artifact** |
| Arm C distinguished | **verified** | separate arm, `reduction_v1.json`, `STATUS: PROVISIONAL` |
| Learned reference used numerically? | **verified: NO** | §4, from the frozen record |
| Arm-A receipts per seed | **verified** | §3, 18/18 PASS |
| Charged-call authority | **verified** | resolved receipts, not `progress.json` |
| Arm C commit / workspaces | **verified** | C `e11cd89d64c03a…`, C on `kosha-labs`, B on `nitya` |
| **Per-seed namespace/volume for Tables 2/11** | **41 of 74 MISSING** | §2; only 33 seed-entries have a committed launch receipt |
| **First 1,000 receipts, locally** | **MISSING** | receipts live on Modal volumes; only per-seed reductions are committed |
| **JNK3 replacement value** | **MISSING** | implied 0.2070 / 0.1831; candidate seed 20269928 has no committed score |
| **"815-call provisional" JNK3** | **MISSING** | no artifact mentions it |
| **Deployed image digest** | **MISSING** | not recorded in any committed artifact |
| **Input hashes for PMO runs** | **PARTIAL** | contract pins exist; *"no arm's proposal-synthesis path is in the contract's `implementation_sha256`; arm identity rests on the launch receipt `git_commit` plus the deployed image"* — a stated limitation of the ablation itself |

### Caveats that belong with any restatement

- **AUC is budget-dependent** and trapezoids from (0,0). A 1,000-call AUC is
  structurally depressed ~5% and **must never** be compared to a published
  10,000-call figure. Below one logging period the depression is exactly 50%.
- **gsk3b is a reward-hackable ML-predictor oracle in every arm.** Its leaders
  are off-manifold (QED 0.03–0.23, SA 6.2–6.6, bare phosphorus, stacked
  hydrazines). The paper retains the objective in the declared reduction but does
  not interpret its scores as chemical quality. The same applies to jnk3 and drd2.
- **Arm B's completed subset excludes `ranolazine_mpo` entirely** in
  `reduction_v1.json`, so any B-vs-A figure there rests on a task mix missing
  one sixth of the panel. The paper's 14 pairs *do* include ranolazine (seed
  20274926, n=1), and Table 19 notes no SD is defined for that single pair.
- Tables 18–21 are **exploratory cell-level comparisons on a fixed
  objective-seed panel**, not estimates of generalization over independently
  sampled objectives. The paper says this.

Nothing was run, launched, re-pinned or edited. Score artifacts are currently
gitignored but present on disk and backed up to
`~/compose_paper_artifacts_backup`; every command above reads them in place.
