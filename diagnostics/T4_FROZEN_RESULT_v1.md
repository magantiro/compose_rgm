# T4 — FROZEN RESULT (authoritative)

**FROZEN — this is the authoritative T4 result. Do not re-derive from a builder.**

- Comparator: **InVirtuoGen (IVG), no-prescreen**
- Budget: **COMPOSE 250 charged docking calls per cell vs InVirtuoGen 1000**
- Objective: QuickVina2 docking of the best eligible lead, lower is better, subject to QED>=0.6, SA<=4.0, Tanimoto>=delta to the seed
- Frozen: 2026-09-22T03:59:40.736075+00:00
- Artifact: `diagnostics/T4_FROZEN_RESULT_v1.json` (payload sha256 acaf7dad784cf0ae…)

> Do NOT re-derive this from `scripts/t4_ivg_convention_table.py`. That builder pins one
> run per volume, and work migrates to a new `run_id` after a resume, so it silently
> reports an empty run as a cell's result. This file supersedes it.


## δ = 0.4 — coverage **15/15**

| Target | Seed | COMPOSE | IVG | gap | calls | source |
|---|---|---|---|---|---|---|
| PARP1 | 1 | -11.9 | -14.1 | +2.2 | 73 | panel |
| PARP1 | 2 | -12.3 | -13.4 | +1.1 | 81 | panel |
| PARP1 | 3 | **-14.1** | -9.0 | -5.1 | 129 | panel |
| FA7 | 1 | **-9.3** | -8.4 | -0.9 | 89 | panel |
| FA7 | 2 | **-9.4** | -8.9 | -0.5 | 217 | panel |
| FA7 | 3 | **-9.6** | -8.0 | -1.6 | 105 | support_expansion |
| 5HT1B | 1 | -12.4 | -13.3 | +0.9 | 130 | panel |
| 5HT1B | 2 | -11.9 | -12.0 | +0.1 | 161 | panel |
| 5HT1B | 3 | **-12.5** | -10.9 | -1.6 | 249 | support_expansion |
| BRAF | 1 | **-12.2** | -10.1 | -2.1 | 77 | panel |
| BRAF | 2 | -10.5 | -10.8 | +0.3 | 113 | panel |
| BRAF | 3 | **-10.9** | -10.6 | -0.3 | 73 | panel |
| JAK2 | 1 | **-10.4** | -10.2 | -0.2 | 225 | panel |
| JAK2 | 2 | **-11.3** | -10.5 | -0.8 | 145 | panel |
| JAK2 | 3 | **-10.7** | -10.2 | -0.5 | 129 | panel |
| **SUM** | | **-169.4** | **-160.4** | **-9.0** | | 10/15 wins, mean **-0.600** |

## δ = 0.6 — coverage **14/15**

| Target | Seed | COMPOSE | IVG | gap | calls | source |
|---|---|---|---|---|---|---|
| PARP1 | 1 | **-13.6** | -12.3 | -1.3 | 246 | panel |
| PARP1 | 2 | **-12.7** | -11.7 | -1.0 | 244 | panel |
| PARP1 | 3 | **-11.3** | -10.7 | -0.6 | 249 | panel |
| FA7 | 1 | — | -7.7 | — | 0 | panel |
| FA7 | 2 | **-7.8** | -7.5 | -0.3 | 249 | panel |
| FA7 | 3 | **-8.5** | -7.4 | -1.1 | 40 | support_expansion |
| 5HT1B | 1 | **-13.3** | -12.4 | -0.9 | 217 | panel |
| 5HT1B | 2 | -11.3 | -12.0 | +0.7 | 222 | panel |
| 5HT1B | 3 | **-10.8** | -10.6 | -0.2 | 126 | support_expansion |
| BRAF | 1 | -9.0 | -9.7 | +0.7 | 33 | support_expansion |
| BRAF | 2 | **-11.0** | -10.4 | -0.6 | 89 | support_expansion |
| BRAF | 3 | **-11.5** | -10.3 | -1.2 | 104 | panel |
| JAK2 | 1 | **-10.6** | -9.7 | -0.9 | 97 | panel |
| JAK2 | 2 | **-10.8** | -10.4 | -0.4 | 89 | panel |
| JAK2 | 3 | **-10.8** | -10.3 | -0.5 | 113 | panel |
| **SUM** | | **-153.0** | **-145.4** | **-7.6** | | 12/14 wins, mean **-0.543** |

## Correction applied to the previous table

**5HT1B seed 3, delta 0.4: -12.3 → -12.5.** Round locks record a molecule only once it becomes a PARENT, so the winner scored in the final round never appears in a lock. The completed run archive (249 scored molecules, status complete_budget) holds -12.5 as its best, and that molecule was independently re-scored under the pinned T4 kernel and passes every gate.

`Nc1nncn1C1=CC2C(CCCN3CCN(C(=O)Oc4ccc5c(c4)CCC5)CC3)=CN=C2C=C1`

Verified under the pinned T4 kernel: similarity **0.4000**, QED 0.6736, SA 3.9776, 36 heavy atoms. similarity sits EXACTLY on the 0.4 bound; it is eligible only because the shipped comparator is non-strict.
Round-lock lower bound was -12.3.

## The one blank cell

**FA7 seed 1, delta 0.6 — SCOPED NEGATIVE, deliberate.** IVG has -7.7.

Reachable but not reached. All five stored witnesses pass the production Fiber.check and a production configuration generated one eligible endpoint, but only ONE across 3 seeds x 480 draws at a single seed -- not reproducible seed-to-seed, so docking calls were declined. The barrier is architectural: expand does not gate the molecule a proposal law conditions (recovered_fraction = 0.0000 over 150 draws).

Record: `diagnostics/t4_fa7_0_scoped_negative.md (commit 01baa8a6)`

**Open option:** Dock the one credible eligible endpoint (COC(=O)N(CCC(C)C)Cc1ccc2ccc(C(=N)N)cc2c1, sim 0.6610 / QED 0.6264 / SA 2.331, retains the amidine pharmacophore) and mark it a single-witness rescue.

## Adaptive support expansion

Cells marked `support_expansion` ran under their own versioned contract with a proposal law that
DIFFERS from the v1 arm. Their claim boundaries forbid splicing them into an
"unchanged-v1" panel without naming them. They are named here.


## Successor in progress (v1 remains authoritative)

A **v2** panel is being prepared from ONE FROZEN ALGORITHM — primary controller plus a generic
zero-support fallback available to all 15 cells, sealed prospectively before any docking
(branch `t4-generic-fallback-20260922`). Its purpose is to remove the `support_expansion` column: a row
produced by the declared algorithm is an ordinary result, and the fallback is described once in
Methods.

**Until v2 is sealed and run, the numbers above are the ones to quote.** Do not mix v1 and v2 rows.

## Claim framing (owner decision, 2026-09-22)

**Adaptive support expansion is a first-class controller capability, not a patch.** Say:

> A single COMPOSE controller spans all T4 targets by combining constrained local optimization
> with adaptive proposal escalation for states where the primary proposal distribution becomes
> support-limited.

**May claim:** shared controller framework; adaptive proposal escalation; same executor and
constraints; state-dependent program support; search configurations frozen per target.

**Must NOT claim:** identical configuration for every target; the same proposal distribution
everywhere; that one identical frozen fallback implementation ran on every support-limited cell.

**Banned words:** rescue, fallback, recovery.

The appendix discloses in one sentence that the support-expansion configuration differed across
the few support-limited instances. That is the standard GenMol and InVirtuoGen meet — both
compose multiple mechanisms without claiming homogeneity.

## Docking reproducibility caveat (discovered 2026-09-22)

**MEASURED: the fa7_0 seed molecule scored −7.5, −8.30 and −8.8 across three runs — a 1.3
kcal/mol spread on ONE molecule, one target, one fixed box.** `qvina02` is seeded, but the
conformer is built by `obabel --gen3D`, which takes no seed (verified in source; inferred as
the cause, no repeated-conformer experiment run).

**Read the aggregate and the win count with confidence; do not lean on any single row's
margin.** Per-cell gaps of 0.2–0.7 appear throughout this table and cannot be called
reproducible against a 1.3 spread. The aggregates (δ0.4 −9.0 over 15 cells, δ0.6 −7.6 over 14)
are far less exposed.

This does **not** affect the within-run search: candidates and the incumbent share one pipeline
and one conformer generator, so selection inside a run is fair. Not fixed, deliberately —
seeding conformer generation changes the docking adapter that every T4 contract pins.

## Claim framing (LOCKED 2026-09-22)

**Name: a population-based controller with adaptive proposal escalation.**
Rule: `Q_t = empty` => escalate proposal effort over the same support lanes along a
bounded ladder; publish `candidate_exhaustion` only if the ladder runs to its declared end.

**Do not call it replenishment.** "Archive-backed proposal replenishment" was considered
and rejected: the implementation escalates the DRAW BUDGET over the SAME lanes on the SAME
parent, and never draws different parents from the archive. That name would claim an
algorithm we did not run.

Banned: rescue, fallback, recovery, replenishment. Avoid "hierarchical" -- there are not
multiple explicit policy levels here and a reviewer will ask.

Three objections a reviewer can raise, and where we actually stand:

| objection | status |
|---|---|
| manually invoked rather than automatically triggered | **partial** -- automatic within a run on an empty pool, but arms were deployed per cell by human launch, not uniformly across the panel |
| behaviour chosen per target after seeing results | **not clean** -- the escalation operator differed across the few support-limited instances; these are development experiments, not one frozen algorithm |
| extra oracle budget vs baselines | **clean, and the strongest point** -- escalation spends CPU and never oracle calls; records are locked and docked through the unchanged round path, budget ceiling untouched. COMPOSE ran 250 calls/cell against IVG's 1000 |

To make the claim fully clean: one frozen, target-agnostic escalation configuration run
over the whole panel, triggered only by the mechanical empty-pool event. Until then the
honest claim is a shared control FRAMEWORK with a per-instance escalation configuration,
disclosed once in the appendix.
