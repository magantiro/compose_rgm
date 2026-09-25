# Handoff — 2026-09-24 session

Everything that ran, what it measured, what was retracted, and what is still open.
Written for whoever picks this up next, human or agent.

## 1. Headline state

| workstream | status |
|---|---|
| **T4 vs GenMol at matched 250 calls** | **DONE and strong.** COMPOSE wins 24/24 paired cells vs GenMol mean-of-8, 20/24 vs best-of-8, plus 5 cells GenMol never solves. |
| **T4 replicates 2 and 3** | Gate **PASS**, 10 arms authorized at 15,000 calls, launch in progress. |
| **IVG T4 comparison** | **IMPOSSIBLE.** Lead-optimization runner is absent from the released code. |
| **PMO no-prescreen 1K** | 20 tasks x 2-3 seeds extracted; 3 protein tasks at n=1 locally. |
| **PMO 10k campaigns** | **48 of 50 DEAD** on a contract/source hash mismatch. Needs redeploy + resume. |
| **PMO mechanisms** | Construction prior: positive on celecoxib 3/3, inconclusive on gsk3b. Online memory: **withdrawn**. |

## 2. T4

### 2.1 The comparison (complete)

GenMol baseline: 240 runs, 8 replicate passes, on Modal native linux/amd64.
**Only deviation from the authors' defaults: `--num_gen 100 -> 25`** (budget 1000 -> 250);
`docking.py` byte-identical to `docking.py.orig`. Checkpoint `nvidia/NV-GenMol-89M-v1`,
sha256 `2c5c4c34...`. Positive control reproduces the authors' own `actives.csv`
(-7.3, -7.8 exact) in both native and emulated environments.

Verified comparable: `qvina02` and ALL FIVE receptors byte-identical between our pinned
hashes and NVIDIA's release; all five box centres and sizes identical.

    delta   statistic          cells  COMPOSE  GenMol  COMPOSE wins
    0.4     vs mean-of-8         14   -159.8   -142.8       14/14
    0.4     vs best-of-8         14   -159.8   -153.2       10/14
    0.6     vs mean-of-8         10   -114.8    -97.5       10/10
    0.6     vs best-of-8         10   -114.8   -101.8       10/10
    plus 5 cells where only COMPOSE produces a result.

GenMol's own aggregate reproduces its published 1000-call result at quarter budget
(delta=0.6 exactly equal on the 3-pass/10-cell set), so the baseline is not starved.

### 2.2 THE STATISTIC IS LOAD-BEARING

**best-of-N is biased and its bias GROWS with N.** Same 240 runs, same cells, only the
summary statistic changing: delta=0.4 best-of-3 -147.3, best-of-8 -153.2, mean-of-8 -142.8.
**A 10.4 kcal/mol swing on a fixed cell set**, larger than any per-cell effect and larger
than the margins the verdict turns on. Report mean WITH n_success on both sides; best-of-N
only as a labelled secondary column; never best-to-best, never across different N.
`mean_of_successful` conditions on success and GenMol's delta=0.6 success rate is 54.2%.

**AND: GenMol has FOUR d0.6 cells at n_success = 0** -- fa7_2, 5ht1b_2, braf_0, jak2_2.
Three are COMPOSE's own historically exhausted cells. `mean_of_successful` is UNDEFINED
there, so those rows are "COMPOSE produced a molecule, GenMol produced none", NOT a score
comparison. Say that explicitly or it reads as a win margin.
The reporting statistic is sealed BEFORE any replicate-2 number exists at
`diagnostics/t4_replicate_launch_authorization/reporting_contract_v1.json`.

### 2.3 Why GenMol fails where it fails (mechanism, not variance)

GenMol seeds its fragment population ONLY from the start molecule and grows it solely from
molecules passing the FULL four-way gate (DS > start AND QED >= 0.6 AND SA <= 4 AND
sim >= delta). Where the start molecule's own QED is below 0.6, a run must land all four
constraints at once before the population can grow. **20 of 30 cell-instances are exposed;
9 are measurably bimodal.** All 6 zero-success instances are exposed and 9 of 10 partials --
but `5ht1b id1 d0.6` is 6/8 while exposed on neither criterion, so it is a strong predictor,
not a law. `5ht1b id2` is the second mechanism: start QED fine (0.716), start **SA 4.69**
above the ceiling, so it must repair SA by 0.69 while holding similarity -- independently
reproducing our own 5ht1b_2 diagnosis from a different codebase.

### 2.4 Replicates 2 and 3 — LAUNCHED 2026-09-24

All ten arms are live and charging. Staggered 1/4/5; wave 1 was fa7_d06 ALONE with its root
docking confirmed end-to-end before the other nine, because a prior session lost four root
dockings to four sequential launch defects on virgin code. All ten root dockings confirmed
FROM THE VOLUME ARTIFACT, never from `FunctionCall.get`. Branch HEAD `4a600557`, pushed.

    arm        function_call_id                  | arm        function_call_id
    fa7_d06    fc-01M3B8KYRAVVC1D6R1Q9X5RX35     | fa7_d04    fc-01M3B8W1WPZB14Q0TM3R7Q0CB4
    braf_d06   fc-01M3B8PCS3MR579TGPBTX7P242     | braf_d04   fc-01M3B8X4JTMZ6G0M3V39HTMPHP
    5ht1b_d06  fc-01M3B8QDN5VZXSD7M9B0DMTGXG     | 5ht1b_d04  fc-01M3B8Y4PXFH0RBRRFPJR3Y55A
    jak2_d06   fc-01M3B8RJBGYD29RACPZ6SQX66D     | jak2_d04   fc-01M3B8ZDRA63BN9D1BFC8F2G6V
    parp1_d06  fc-01M3B8SQ5Z5VQTS5N79KM97BG1     | parp1_d04  fc-01M3B90FBHCMMAGJKDVFRJR307

Status at launch: 3 of 6 cells started per arm (`run_cell` caps at `max_containers=3`),
**62 / 15,000 charged (0.4%)**, 0 cells finished, 103 live tasks across 15 apps. 20h per-cell
timeout; hours from done. **Report NO score from checkpoints** -- reconcile from round locks
with `scripts/t4_all_runs_reconcile.py`, which already handles the r23 layout.

CORRECTION to an earlier brief: the blocker was NOT "fa7_0's shard missing". The committed
gate read `trigger_rows_present: 10`, and all 15 trigger shards plus all 20 controls were
ALREADY on the volume -- the previous agent died between the last shard landing and the
reducer running. Only the reduction was missing. Re-running fa7_0 would have burned ~86
minutes recomputing an existing result.

GOTCHA, sealed: pre-authorization the replicate test suite read 22 passed / 21 SKIPPED,
every skip "not authorized yet"; post-flip it is 43 passed / 0 skipped. The reconstruction
guards only activate AFTER authorization, so the pre-flip green was not coverage.

### 2.4b Original launch plan (superseded by 2.4)

Worktree `.worktrees/t4-matched-panel-20260924`, branch pushed. Gate **PASS** (15/15
triggers, 20/20 controls). Ten arms authorized, 6 cells each, 1,500 ceiling each = **15,000**.
Coverage verified: 5 targets x 3 molecules x 2 deltas = 30 cells, x 2 replicates = 60 runs.
30/30 groups carry two distinct replicate seeds; 0 collide with replicate 1.

**Replicate 1 = the historical frozen panel**, admitted under BEHAVIOURAL equivalence:
7/7 rows that historically fired a fallback route to the branch the unified router selects
from the same molecular state (5 region_repair -> q_region, 2 protonation -> q_state),
30/30 seed correspondence. The criterion is behavioural, NOT implementation identity -- an
earlier verdict counted 15 app modules / 10 fingerprints / 15 contract hashes and was wrong.
Do not resurrect that argument.

Caveats to carry: **fa7_0 d0.6 is n=2** (replicate 1 blank, charged 0); replicate 1 charged
**4,114** calls total so replicates 2/3 will likely charge MORE (the ladder restores support
on cells that previously exhausted in a handful of calls -- braf_0 d0.6 spent 33);
`docking_seed` non-uniform by faithful inheritance (braf/parp1 20260918, others 20260919);
**d0.4 gating is INFERRED from delta-monotonicity, not measured**.
`T4_FROZEN_RESULT_v1` REMAINS AUTHORITATIVE for the IVG comparison.

### 2.5 IVG

**The lead-optimization runner is not in the released code.** `in_virtuo_reinforce/ppo_docking.py`
is a LaTeX table generator (7 unrelated flags). Zero files in either branch reference
`qvina`, `DockingVina`, `pdbqt` or `no_sim_constraint`; `genetic_ppo.py` is the PMO runner
with no docking path; `main` is one squashed commit; 3 of 5 receptors missing.
LICENSE CC BY-NC-SA 4.0 (non-commercial); GenMol is Apache-2.0.

## 3. PMO

### 3.1 no-prescreen 1K, best score @ 1000 charged calls

Extracted from the 10k campaigns' charged prefix (every campaign passed 1000 before dying).

    task                      n   mean     sd     range           cv
    isomers_c7h8n2o2          2  0.9877  0.0175  0.975-1.000    0.018
    qed                       2  0.9471  0.0020  0.946-0.948    0.002
    isomers_c9h10n2o2pf2cl    2  0.8842  0.0781  0.829-0.939    0.088
    osimertinib_mpo           2  0.8280  0.0348  0.803-0.853    0.042
    albuterol_similarity      3  0.8269  0.0622  0.787-0.899    0.075
    fexofenadine_mpo          2  0.7597  0.0503  0.724-0.795    0.066
    ranolazine_mpo            2  0.7585  0.0565  0.719-0.798    0.074
    deco_hop                  2  0.7288  0.2008  0.587-0.871    0.276
    amlodipine_mpo            2  0.6156  0.1344  0.521-0.711    0.218
    mestranol_similarity      3  0.6132  0.2418  0.425-0.886    0.394
    perindopril_mpo           2  0.5779  0.0641  0.533-0.623    0.111
    scaffold_hop              2  0.5043  0.0655  0.458-0.551    0.130
    zaleplon_mpo              2  0.4954  0.1016  0.423-0.567    0.205
    sitagliptin_mpo           2  0.4931  0.1698  0.373-0.613    0.344
    valsartan_smarts          2  0.4886  0.6909  0.000-0.977    1.414
    thiothixene_rediscovery   3  0.3838  0.0548  0.328-0.438    0.143
    celecoxib_rediscovery     3  0.3723  0.1091  0.255-0.471    0.293
    troglitazone_rediscovery  3  0.3597  0.0869  0.286-0.455    0.242
    median1                   2  0.3142  0.0143  0.304-0.324    0.045
    median2                   2  0.2460  0.0888  0.183-0.309    0.361
    ---- local, n=1, NO variance ----
    drd2                      1  1.0000    --    top10 0.9992  auc 0.9180
    gsk3b                     1  0.4100    --    top10 0.3930  auc 0.3058
    jnk3                      1  0.2700    --    top10 0.2490  auc 0.1918

Suite sum over the 20 replicated tasks: **12.185**, propagated SE **+-0.575** (dominated by
valsartan alone at SEM 0.489; drop it and the SE falls to ~0.30).

**15 of 20 tasks are at n=2**, where an "sd" is a range wearing a lab coat. Report mean with
min-max and explicit n. **This column is BEST SCORE, not auc_top10** -- do NOT place 12.185
beside IVG's 16.676, which is an AUC column.

### 3.2 THE 10k CAMPAIGNS ARE DEAD

**48 of 50 campaigns exited with `returncode: 1`**, all on the identical error:

    ValueError: input identity mismatch:
      src/compose_v4/control/dynamic_program_synthesis_v21.py:
      expected 3e591297..., got fc769f2c...

Same two hashes as the donor-arm failure earlier the same day: a redeploy whose baked
SOURCE moved while its baked CONTRACT still pinned the pre-edit hash. Sequence that fits:
each campaign ran normally (1,678-4,512 charged), was preempted, and the RETRY restarted
into the rebuilt image, died at startup, and overwrote `provenance.json` with returncode 1.
Two survive: albuterol seed ...923 (5,326) and thiothixene seed ...923 (4,305).

**FIX = redeploy so baked contract and baked source agree, then resume.** Campaigns are
stateful and checkpointed; verify resumption picks up rather than restarting.

**Volumes:** `compose-pmo-fibercontrol` and `compose-pmo-fibercontrol-mpo`, profile **nitya**.
`compose-pmo-fibercontrol-replication` exists and is EMPTY -- ignore it.
drd2 / gsk3b / jnk3 have NO Modal campaign; they exist only as local 1K runs.

### 3.3 Reward hacking (scoped, tested prospectively)

    task        oracle          n    best   r(score,QED)  QED@best  on-manifold best
    gsk3b       ML predictor   936   0.410     -0.705      0.038        0.130
    jnk3        ML predictor   384   0.230     -0.750      0.035        0.100
    celecoxib   similarity     919   0.299     +0.229      0.426        0.293

The predictor oracles are climbed by leaving the drug-like manifold. gsk3b's 0.410 is
`CCCN(N)Cc1ccnn1C[IH3]CC(...)` -- two hypervalent iodines, QED 0.038. jnk3's 0.230 carries
boron, hypervalent sulfur and an `OOO` trioxide chain. Neither is a molecule.
On similarity tasks the on-manifold best essentially EQUALS the overall best (0.293 vs
0.299) -- restricting costs nothing where the oracle is honest and everything where it is not.
**53.3% of gsk3b's 1000-call budget went to molecules with QED < 0.2**, only 8.4% to QED >= 0.5,
so its "on-manifold ceiling" of 0.130 is a best-of-79, not a measured ceiling.

**CAVEAT added later the same day:** across 6 celecoxib arms `r(score,QED)` ranges
**-0.620 to +0.492**. So "similarity oracles are honest" is a PER-RUN property, not a task
property. The extremes hold (gsk3b/jnk3 consistently -0.70/-0.73); the clean dichotomy does not.

### 3.4 Mechanisms

**Construction prior** (`ringcore_a7546e2_best.pt`, DEVELOPMENT_ONLY, task-independent):
- gsk3b, 3 seeds: benchmark objective 2/3 (NOT established -- seed 1 gave B/A AUC **2.190**
  and seed 2 REVERSED it to 0.886); manifold adherence 3/3.
- celecoxib, 3 seeds: **AUC 3/3 (ratio 1.145), best 3/3 (ratio 1.255)** -- the first
  replicated positive PMO signal. Predicted in advance and sealed before running.
- But on-manifold sampling was higher only 2/3 on celecoxib, so the manifold mechanism does
  NOT generalise across oracles.

**Online memory: WITHDRAWN.** The celecoxib +/- memory 1K pair reporting B/A AUC 1.165 is
retracted. Three proposal-path files (`dynamic_program_synthesis.py`, `current_state_edits.py`,
`pmo_online_memory.py`) were edited at 15:26-15:33 while the arms ran, **none is
contract-pinned**, and the artifact records no implementation hash -- so there is no evidence
both arms ran identical bytes.

## 4. Retractions and corrections made this session

1. **2.19x construction-prior effect** -- trajectory variance; reversed on seed 2.
2. **"Identical on-manifold best across seeds is structural"** -- different molecules;
   gsk3b is a random forest whose output is quantized to 0.01, so ties are unremarkable.
3. **Memory-gate 1.165x** -- withdrawn, working-tree race (above).
4. **"COMPOSE wins 20/24"** -- compared COMPOSE best-of-1 to GenMol best-of-3, mismatched
   statistics. The mismatch favoured the RIVAL, so the conclusion survived; that was luck.
5. **"Similarity oracles are honest"** -- per-run, not per-task (3.3).
6. **"The PMO seeds never launched"** -- I checked an empty volume and declared absence
   across three messages. They were landing in `compose-pmo-fibercontrol-mpo` all along.
7. **Ladder rungs** -- stated as 480/960/1920/3840; it is `draw_ladder=(960,1920,3840)`
   plus a lane-adding rung 0. The 480 is the shallow lane's draw count.

## 5. Open decisions

1. **PMO 10k redeploy + resume** -- the only thing between us and the 10k numbers.
2. **Route-lane provenance** -- the T4 route expert is fit on 77 locked routes with
   leave-one-target-out (`held_target_absent_from_training: true`, `runtime_target_conditioning:
   false`, `new_oracle_calls: 0`). The corpus is competitor-derived and NC-licensed. The lane
   returns 0 eligible in 8 of 9 cells, so disabling costs almost nothing -- but disabling
   mid-audit would destroy the equivalence being established. Owner call.
3. **Retry patch** for T4 (`run_cell` 5, `proposal_worker` 3, `dock_worker` deliberately 0)
   is written, tested and REVERTED because it breaks all five frozen contract pins. Separate
   re-seal if wanted.
4. **Whether to test the construction prior on a second honest task** -- the celecoxib
   result is 3/3 on one task and wants a second before promotion.

## 6. Environment facts that bite

- **THREE chemistry kernels.** PMO = rdkit **2023.9.6**; T4/editing = **2024.3.5**; laptop
  `.venv` = 2026.03.6. A local PMO number computed under 2024.3.5 needs a parity statement.
- **T4 docking is not reproducible across runs**: qvina02 is seeded but `obabel --gen3D` is
  not. One molecule scored -7.5 / -8.30 / -8.8. Per-cell margins under ~1 kcal/mol are not
  resolvable at n=1.
- `modal volume ls <vol> <subpath>` silently returns the PARENT listing. `modal volume get`
  collapses a directory onto one path. `modal app list` TRUNCATES the description column.
  Use `modal.Volume.listdir` from Python; read the task-count column.
- **Profile matters.** PMO campaigns are on **nitya**; T4 on **rahul-94866**. Cross-profile
  they read as absent.
- `FunctionCall.get(timeout=0)` raising TimeoutError means NOT FINISHED -- it covers queued
  and dying calls. Check `returncode` in each namespace's provenance instead.
