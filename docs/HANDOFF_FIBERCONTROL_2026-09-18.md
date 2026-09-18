# FiberControl handoff — 2026-09-18

Lossless record of the autonomous-controller work on branch
`t4-objective-dynamic-reset-20260916`, from commit `938f26a0` (the last state described
by the previous handoff) through `73c16979`. Written for a collaborator picking this up
cold. Everything claimed here is either a measurement with its artifact named, or is
explicitly flagged as unmeasured.

---

## 1. One-paragraph status

The autonomous controller runs end to end on JAK2 `jak2_1` at delta=0.6 and reaches
**-10.30** in 105 charged docking calls from a -8.10 seed, against InVirtuoGen's published
**-10.4 +/- 0.1** for that exact cell. Reward-adaptive selection beats reward-blind
selection on the same candidate pools in 10 of 13 matched rounds (sign test p=0.092) and
its model-chosen calls beat its own random quota by 0.405. The controller is therefore
working; the binding constraint is the PROPOSAL LAW, which places about 0.067% of its mass
on the structural architecture every known strong delta=0.6 JAK2 molecule shares, and 0 of
those proposals survive the similarity gate. The next planned experiment -- the same
controller on `braf_9` delta=0.6 -- is **blocked**: 240 free programs from that seed yield
**zero** feasible endpoints, because the braf seeds start at QED 0.235-0.346 against the
task's 0.6 floor and nothing in the current proposal vocabulary repairs that.

---

## 2. Results, with exact numbers

### 2.1 The headline run (`RUN4_jak2_1_final.*`)

Target `jak2`, seed `jak2_1` = `COC(=O)CC2Nc1ccccc1c3ccnc4[nH]cc2c34`, delta=0.6,
112-call budget per arm (105 spent), batch 8 (6 model picks + 2 random), 480 raw programs
per parent, 4 parents per arm, program depth 3, seed 20260918, support `compose_valid`.

| | R_official | R_vetted |
| --- | ---: | ---: |
| adaptive | **-10.30** | -10.00 |
| reward-blind | -10.00 | -9.90 |
| IVG published, jak2_1 delta=0.6 | **-10.4 +/- 0.1** | - |

`R_official` = best under the task criterion. `R_vetted` = best that also passes the
frozen secondary chemistry screen. The adaptive arm's top FIVE molecules all carry
`N-O-acyl` and are refused by that screen, which was frozen BEFORE this run.

Best-so-far by call count (adaptive / blind):
9 `-9.30/-8.80`, 17 `-9.30/-9.60`, 25 `-9.50/-9.60`, 33 `-9.80/-9.60`, 41 `-9.80/-9.60`,
49 `-10.00/-9.70`, 57 `-10.00/-9.70`, 65 `-10.20/-9.70`, 73 `-10.20/-9.90`,
81 `-10.30/-9.90`, 89 `-10.30/-9.90`, 97 `-10.30/-9.90`, 105 `-10.30/-10.00`.

Top adaptive molecules:
```
-10.30  C=C1N(COC(=O)CC2Nc3ccccc3-c3ccnc4[nH]cc2c34)CC(=O)ON1C     N-O-acyl, screened
-10.20  CN1CN(COC(=O)CC2Nc3ccccc3-c3ccnc4[nH]cc2c34)CC(=O)O1       N-O-acyl, screened
-10.00  CC1CC(COC(=O)CC2Nc3ccccc3-c3ccnc4[nH]cc2c34)CC(=O)O1       CLEAN  <- R_vetted
```

### 2.2 Does reward adaptation beat reward-blind selection?

Three independent readings, all from the same run, all favourable:

- **Matched-round sign test.** Both arms select from ONE shared pool each round, so each
  round is a matched pair. Adaptive better in **10 of 13**, blind in 3, mean paired
  difference **-0.423**, two-sided **p = 0.092**. Suggestive, not significant at n=13.
- **Model picks vs its own random quota** (within-arm, same round, same pool, same
  parents): model-selected **-9.074** over 78 calls, random quota **-8.669** over 26
  calls, difference **-0.405**.
- **Top-tail ranking.** Median percentile of the round's ACTUAL best molecule in the
  model's pre-docking ranking of the whole pool: **2.3%**. In the top decile on 9 of 12
  rounds; ranked #1 on 8 of 13. Round 10 ranked the eventual -10.30 winner **1 of 168**.
  Regret 0.00 on 10 of 12 rounds.
- **Calibration.** `corr(predicted endpoint, realized)` moved **-0.463 -> +0.262** over
  104 calls. The early anti-correlation was small-sample, not a defect.

### 2.3 Why it stops at -10.30 (`BASIN_AUTOPSY.md`)

Every molecule at or below -10.5 that is feasible at delta=0.6 shares ONE architecture:
the seed's methyl ester replaced by an amide to a saturated 1,2-diamine ring, that ring
carrying a urea or carboxamide. Seven of seven.

```
-11.30  sim 0.629  CC1N(C(N)=O)CCN1C(=O)CC1Nc2ccccc2-c2ccnc3[nH]cc1c23   (route-assisted, confirmed -11.32 +/- 0.04 over 4 seeds)
-11.00  sim 0.639  CC1N(C(=O)O )CCN1C(=O)CC1Nc2ccccc2-...                (archive best)
-10.90  sim 0.600  CC1=CN(C(=O)CC2Nc3ccccc3-...)CN1C(N)=O
-10.80  sim 0.619  CC1(C)N(C(N)=O)CCN1C(=O)CC1Nc2ccccc2-...
-10.70  sim 0.600  CC1CN(C(=O)CC2Nc3ccccc3-...)CN1C(N)=O
-10.60  sim 0.619  CC1N(C=O)CCN1C(=O)CC1Nc2ccccc2-...
-10.60  sim 0.623  NC(=O)N1CCN(C(=O)CC2Nc3ccccc3-...)C1
```

Feature rates, strong basin vs what the controller docked:

| feature | strong d0.6 | strong d0.4+ | autonomous docked |
| --- | ---: | ---: | ---: |
| amide linkage `N-C(=O)-CH2` | 7/7 100% | 21/23 91% | 13/193 7% |
| **ester retained** | 0/7 0% | 0/23 0% | **126/193 65%** |
| ring `N-C-N` | 7/7 100% | 18/23 78% | 7/193 4% |
| ring `N-C-C-N` | 6/7 86% | 16/23 70% | 1/193 0.5% |
| urea/carbamoyl on ring N | 5/7 71% | 14/23 61% | 5/193 3% |
| acyloxymethyl (our motif) | 0/7 0% | 0/23 0% | 22/193 11% |

The controller DECORATES the ester; the basin REPLACES it.

### 2.4 Factoring the proposal probability (`AUDIT_program_factors.json`)

3,000 programs from the root, 41,591 realized variants, 1,610 feasible endpoints. Root
slots read off the padded graph: slot 0 methyl C, **slot 1 ester O**, slot 2 carbonyl C,
slot 3 carbonyl O.

| factor | rate |
| --- | ---: |
| q(R): WHERE touches the ester group (slots 0-3) | **65.4%** (1949/2978) |
| q(R): WHERE touches the ESTER OXYGEN | **55.4%** (1649/2978) |
| q(A\|R): ester oxygen retyped to N | **0.77%** of variants (321/41591), 48 feasible |
| q(amide \| R hit, feasible) | 6.6% (55/828) |
| q(diamine ring \| R hit, feasible) | 0.12% (1/828) |
| q(amide AND ring) | **0/828**, `< 0.36%` by rule of three |
| q(urea on ring N) | 0/1610, `< 0.19%` |

**WHERE is not the defect** (matches the historical 96.6% JAK2 WHERE coverage).
**The retype is not the defect** -- 308 of the 321 retypes come from `retained_element`,
the capability added at `df807f44`. **Coordination is.**

### 2.5 Why generated diamine endpoints die (`AUDIT_diamine_survival.json`)

50 diamine-ring endpoints from 2,973 programs, classified by the FIRST gate refusing them:

```
similarity                    49/50   98.0%
structural:strained_NN_ring    1/50    2.0%

similarity - 0.60   median -0.420   min -0.574   max -0.063   violating 50/50
4.0 - sa            median -1.092   min -2.661   max +0.376   violating 46/50
qed - 0.60          median +0.012   min -0.483   max +0.262   violating 24/50
```

Best of the fifty: similarity 0.537, still outside the corridor. No reward model can
select candidates refused before reward is consulted.

**A coarse-SMARTS artifact, corrected.** `[NX3;R][CX4;R][NX3;R]` needs sp3 ring carbons,
and every nitrogen in the root is aromatic, so the pattern is satisfiable by
DE-AROMATISING the core. `ring_system_restate` is 6.4x enriched in these endpoints (0.82
per diamine endpoint vs 0.13 per program). Re-measured with a definition that cannot be
met by wrecking the core -- a saturated ring with >=2 N sharing no atom with any aromatic
ring -- and validated by requiring the three known strong molecules to satisfy it (they
do):

| | count |
| --- | ---: |
| coarse "diamine" endpoints | 50 |
| aromatic core preserved (3+ aromatic rings, as the root) | **10/50** |
| genuine pendant saturated diamine ring | **7/50** |
| **both, i.e. the basin architecture** | **2/50** |

```
core preserved = True    n=10   median sim 0.385   max 0.537
core preserved = False   n=40   median sim 0.138   max 0.394
```

**The two numbers any q0 change must move:**
```
P(basin architecture)              = 2/2973 = 0.067%
P(sim >= 0.6 | basin architecture) = 0/2
```

Growth is NOT what costs similarity: the known basin grows 22 -> ~28 heavy atoms and still
sits at 0.629, because it keeps all three aromatic rings and the whole
`C(=O)CC1Nc2ccccc2-c2ccnc3[nH]cc1c23` fragment, replacing only the methoxy. MCS/seed for
the basin is **0.91**, the highest in the whole cross-cell table.

### 2.6 Cross-cell audit (`AUDIT_cross_cell_transformations.json`)

25 published IVG winners across the 5 cells that have released structures, vs the JAK2
basin. Zero oracle calls.

| lane | cells | similarity | Δ heavy atoms | aromatic rings kept | MCS/seed | direction |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| delta=0.4 | 5ht1b_s7, parp1_s0 | 0.449 | **+14** (+11..+15) | 10/10 | 0.40 | **GROW** |
| delta=0.6 | braf_s9, s10, s11 | 0.610 | **-10** (-14..-7) | **3/15** | 0.60 | **SHRINK** |
| **JAK2 d0.6** (ours) | jak2_1 | 0.619 | **+7** | **7/7** | **0.91** | **GROW** |

**Three different transformation regimes, and JAK2's is not the same as the other delta=0.6
cells.** BRAF delta=0.6 winners PRUNE 7-14 heavy atoms from 37-40-atom seeds, which
COMPOSE already covers (`substituent_delete`, `segment_shrink`, `cycle_open`,
`ring_system_delete`). So an anchored-replacement primitive built for JAK2 is NOT
established as a general missing capability.

### 2.7 The BRAF experiment is blocked (measured, not inferred)

```
braf_9  CCN(CC)CCNC(=O)c3cnn4c(c2cccc(NC(=O)Nc1ccc(Cl)c(C(F)(F)F)c1)c2)ccnc34
        40 heavy atoms, QED 0.235, SA 2.69     -> FAILS the task gate itself
braf_10 39 HA, QED 0.346        braf_11 37 HA, QED 0.255

free pool from braf_9: 240 raw programs -> 0 FEASIBLE endpoints (55 s)
```

The braf seeds start BELOW the QED floor, so the control problem there is to repair a
violated constraint while holding the similarity corridor -- which is exactly why the
published winners shrink. The current proposal vocabulary produced no feasible endpoint at
all in 240 programs, so the planned BRAF controller run cannot start. **This check cost 0
docking calls and saved the 56 that were about to be spent.**

### 2.8 Pool width (`AUDIT_pool_width.log`)

```
serial     90 draws -> 29 feasible in 62.2 s
8-shard   720 draws -> 176 feasible in 70.5 s      6.1x width for 1.1x wall clock
```

Accessibility at delta=0.6 is ~1.1% of proposed endpoints. Pools went from 9-77 (first
run) to 104-286 (final run). Pool width per parent FALLS as the frontier leaves the root:
139 from 1 parent, then ~38/parent, then ~18/parent.

---

## 3. What was built

### 3.1 Runtime (`src/`)

| file | role |
| --- | --- |
| `control/fiber_control.py` | `SearchState`, `program_features` (15), `ProgramValue` ridge, `endpoint_utility`, `acquisition` (6 model + 2 random), `should_stop` |
| `control/chemistry_screen.py` | frozen secondary med-chem screen, admission-tested against a reference population |
| `control/feasibility_value.py` | `psi_h` estimator. Measures the geometry; the policy deliberately does NOT use it |
| `experiments/t4_fiber_campaign.py` | `Fiber` (three nested supports), `_variants`, `expand`, `prepare` |
| `experiments/t4_fiber_expansion.py` | `expand_wide`, `expand_frontier` (subprocess shards), `branch_value` (= Q3, implemented and DELIBERATELY UNUSED) |
| `gates/med_chem_gate.py` | pre-existing structural gate; sulfur rule corrected this session |

### 3.2 Drivers (`scripts/`)

| file | role |
| --- | --- |
| `t4_fiber_paired_campaign.py` | the experiment. Both arms in lockstep on ONE shared pool, `--resume`, `--support` |
| `t4_fiber_paired_report.py` | matched-round sign test, tail-ranking table, model-vs-quota split, calibration |
| `t4_teacher_program_factors.py` | the q0 factor decomposition |
| `t4_diamine_survival_audit.py` | first-gate-failed classification, graph-level ring membership |
| `t4_basin_autopsy_probe.py` | basin feature rates on a free pool |

### 3.3 Tests

`tests/test_fiber_control.py` (24), `tests/test_fiber_expansion.py` (11),
`tests/test_chemistry_screen.py` (12), `tests/test_med_chem_gate.py` (17). **64 total, all
passing.** Every selection fix is mutation-checked: reverting it fails the guard written
for it.

### 3.4 Artifacts (`diagnostics/t4_fiber_control/`)

`README.md` (design + what it may claim), `BASIN_AUTOPSY.md` (the full diagnosis),
`RUN1_noise_dominated.*`, `RUN2_quota_legacy_fiber.*`, `RUN3_partial_2rounds.*`,
`RUN4_jak2_1_final.*` (the headline), `AUDIT_*.json` (six free audits).

---

## 4. The three nested supports — a naming trap to avoid

```
Q_legacy_screened  subset of  Q_compose_valid  subset of  Q_benchmark_only
```

- `BENCHMARK_ONLY` — the task contract verbatim: `QED>=0.6, SA<=4, sim>=delta`, valid and
  connected. **Admits radicals**, which is why it is not used for search.
- `COMPOSE_VALID` — the above plus `med_chem_gate.is_valid`. **This is what runs.** It is a
  STRICT SUBSET of the benchmark support, not a restatement of it.
- `LEGACY_SCREENED` — plus `queryable_fiber.instability`, which refuses **7.0% of the
  benchmark's own Jin QED leads** and 13.8% of GuacaMol, chiefly via a plain-enone rule at
  9.0% of GuacaMol. Retained only to reproduce RUN1/RUN2.

Accepting all 25 published IVG winners shows `med_chem_gate` does not remove what IVG
found. It does **not** show the gate equals the task criterion. Do not conflate those.

---

## 5. Bugs found and fixed (do not reintroduce)

1. **Acquisition ranked improvement-over-parent, not endpoint score** (`8823500c`).
   Improving on a weak parent is easier, so it bought the worst parents; round 4 of the
   first run docked -6.8/-7.0/-7.3 with a -9.10 incumbent. Now ranks `parent_score - gain`.
2. **Exploration noise 2.5-5x the model's own signal** (`5b3ca081`). Gaussian noise at 0.5
   (the oracle's single-call reproducibility) was added to predictions spanning 0.10-0.21,
   making the reward arm approximately its own control (3-2 over 5 rounds, p=1.00).
   Replaced by a quota: 6 model picks + 2 random.
3. **Stripping the med-chem bundle deleted the RADICAL check with it** (`fd1b36d8`). Two
   rounds later the leading endpoint was an aminyl radical at -9.70. Structural validity is
   now separate and always applied.
4. **`med_chem_gate` missed TERMINAL hypervalent sulfur** (`fd1b36d8`). The rule required
   `GetDegree() >= 2`, so `CC([SH3])...` (one C neighbour, three H) never fired. Now counts
   bonds to non-oxygen, so thiol/thioether/thiophene/sulfoxide/sulfone/sulfonamide/
   disulfide all still pass.
5. **Subprocess stderr deadlock** (`9e88d8ba`). Shard diagnostics went to a pipe nobody
   drained until the parent reached that shard's `communicate()`; a chatty shard blocked on
   a full 64 KiB buffer. Two shards sat at 6:25 elapsed, one spinning, one at 0.0% CPU. Now
   per-shard files.
6. **Relaunching to the same `--out` destroyed a finding's evidence** (`dec1de6e`). An
   existing result is now renamed with a timestamp before a new run starts.
7. **Boundary-exclusive thresholds** (`8cc7e507`). The gate required QED strictly > 0.6 and
   SA strictly < 4.0 where the contract says `>=0.6` and `<=4`. Matters because strong
   solutions hug constraints -- median similarity margin +0.019, several at exactly 0.600.
8. **An 18-heavy-atom floor** that appears nowhere in the task (`8cc7e507`).

---

## 6. Claims retracted or corrected this session

Recorded because a reader will otherwise find them in earlier commit messages.

| claim | status |
| --- | --- |
| "we matched IVG with ~95x fewer oracle calls" | **WRONG.** The 10,000-evaluation budget in the IVG paper is the PMO benchmark. The paper does NOT state the lead-optimization budget. No call-efficiency claim is supportable. |
| "the honest number is -10.10, not -10.30" | **Too strong.** Two different quantities: `R_official` -10.30 governs the IVG comparison, `R_vetted` -10.00 is a secondary analysis. Do not handicap COMPOSE with constraints IVG did not have. |
| "zero support for the basin conjunction" | **Overstated a zero count.** 0 of 2,400 bounds the rate at ~3/2400 = 0.00125. It is a proposal-MASS problem. |
| "there is no reward valley to cross" | **Does not follow.** The endpoint's 0.629 similarity means it satisfies the constraint, not that no route to it passes through low-reward intermediates. Needs route-prefix scoring, not run. |
| "the catalog cannot build a diamine ring at all" | **False.** The catalog caps at ONE nitrogen per constructed ring (20 options, max N=1, verified), but diamine motifs still arise at 1.7% of programs by other routes. |
| "ring construction is rare" | **False.** 99.07% of programs gain a ring. |
| "cycle_close is downweighted to 0.5" | **Wrong table.** That is `NEAR_CAPACITY_MODULE_WEIGHTS`, which applies only above 36 heavy atoms. The root is 22. |
| "the forced conditionals show the pieces are antagonistic" | **Underpowered.** 0 of 29 retyped-and-feasible carried a diamine ring against an independent expectation of 0.43. |
| "a blanket `C=NH` rule is justified" | **No.** Primary-imine base rate is 0.12% of Jin leads and 1.40% of GuacaMol -- rare but real chemistry. The screen uses a narrow form excluding amidines, imidates, oximes, hydrazones. |
| "my diamine SMARTS measures the basin" | **Too coarse.** Satisfiable by de-aromatising the core; corrected count 50 -> 2. |

---

## 7. Open questions, none of them measured

1. **Reward valley along the teacher route.** Costs oracle calls. Note the executor already
   gives protected semantics -- `expand` gates only the ENDPOINT, never internal states --
   so if the basin's program is one protected step the question may be moot.
2. **IVG's jak2 basin.** Unmeasurable: the paper reports jak2 scores but publishes no jak2
   SMILES. `diagnostics/ivg_winners.json` legitimately has no jak2 cell.
3. **IVG's lead-optimization oracle budget.** Undocumented in the paper.
4. **Whether anchored replacement generalizes.** Not established. BRAF delta=0.6 needs
   deletion, which COMPOSE already has. fa7 has no published winners at all.
5. **Multi-region proposal share** is 0-7% of any pool; `joint_multi_site` is still not
   wired into the controlled action.

---

## 8. Exact next steps, in the order the evidence supports

1. **Unblock BRAF, or drop it.** `braf_9` yields 0 feasible endpoints in 240 programs
   because the seed violates QED. Either add a QED-repair capability (the published winners
   prune 7-14 atoms) or pick a different transfer cell. **Do not spend docking calls on
   braf until a free pool yields feasible endpoints.**
2. **The generic primitive for JAK2**, if pursued: `replace_at_role(G, r, ring_spec)` --
   pick a retained anchor, delete only the substituent being replaced, optionally retype
   the anchor, construct a ring through it, execute the closure atomically, and **preserve
   the aromatic ring system** (80% of the loss comes from de-aromatisation). Nothing about
   piperazine or JAK2 in the primitive.
3. **Acceptance test, free:** rerun `t4_diamine_survival_audit.py` and require
   `P(basin architecture)` and `P(sim>=0.6 | basin architecture)` to become materially
   nonzero from 0.067% and 0/2. No docking until they move.
4. **Then** rerun the JAK2 campaign with the unchanged controller. That is a clean ablation:
   same reward controller, old q0 vs new q0.
5. **Hold at least one target untouched** as a genuine transfer test. fa7 is the natural
   choice -- no published winners were inspected for it.

---

## 9. How to run things

```bash
cd <this worktree>
export PYTHONPATH=src:scripts KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1

# tests
.venv/bin/python -m pytest tests/test_fiber_control.py tests/test_fiber_expansion.py \
    tests/test_chemistry_screen.py tests/test_med_chem_gate.py -q     # 64 pass

# the experiment (both arms, one shared pool per round)
.venv/bin/python -u scripts/t4_fiber_paired_campaign.py \
  --target jak2 --root 'COC(=O)CC2Nc1ccccc1c3ccnc4[nH]cc2c34' --delta 0.6 \
  --support compose_valid --budget 112 --batch 8 --exploration 2 \
  --draws 480 --shards 10 --parents 4 --horizon 3 \
  --out diagnostics/t4_fiber_control/RUN5.json

# resume after an interruption (reboots killed two runs)
#   ... --resume diagnostics/t4_fiber_control/RUN5.json --out .../RUN5b.json

# the report
.venv/bin/python scripts/t4_fiber_paired_report.py \
  --result diagnostics/t4_fiber_control/RUN4_jak2_1_final.json --out /tmp/report.md
```

Docking is Modal (`jak2-contrastive-dock`, function `dock_batch`, target-generic via
`BOXES[target]`). Expansion is local: 10 subprocess shards, ~350-450 s per round for 8
parents at 480 draws. A round is a synchronisation barrier, so Modal parallelism would cap
near 4-6x; see the note in the session log about the PMO-vs-lead-optimization budget
confusion before quoting any speed claim.

## 10. Operational hazards paid for already

- **`/private/tmp` is reaped mid-session and the laptop rebooted twice.** Both destroyed
  running campaigns. All artifacts now live under `diagnostics/` and the driver has
  `--resume`. `compose_v4.data.durable_path` refuses a reapable source-of-truth path.
- **Single-call docking reproducibility is about 0.5** on this target: the same root
  returned -8.00, -8.50 and -8.10. Any champion needs replicate seeds, as the -11.32 got
  four. **The -10.30 headline is a single call and is NOT yet confirmed.**
