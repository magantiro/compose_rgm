# T4 replicate panel — situation brief for an ideating agent

Written 2026-09-25 ~04:00 EDT. Paper submission deadline ~2026-09-26 08:49 EDT (~29 h).
Purpose: give a fresh agent everything needed to propose a better plan. This is a
*situation* document, not an instruction. Nothing here is settled except the measurements.

## 1. The scientific object

T4 is a constrained lead-optimisation benchmark. 5 protein targets (fa7, braf, 5ht1b,
jak2, parp1) x 3 starting molecules x 2 similarity thresholds (delta 0.4, 0.6) = **30 cells**.
Each cell: start from a given lead, propose edits, dock candidates, keep the best endpoint
that satisfies `similarity >= delta AND QED >= 0.6 AND SA <= 4`. Budget **250 charged
docking calls per cell**.

**The claim under test.** Historically each cell was run by whichever hand-picked arm worked
for it — 15 distinct app modules, 10 controller fingerprints, 15 contract hashes. Three of
those were bespoke rescues. The unification claim is that all of those are branches of ONE
state-adaptive policy:

    support state -> applicability mask -> {q_local, q_region, q_state}
    + escalation ladder on terminal candidate_exhaustion

with branch selection driven by MOLECULAR STATE, never by target identity. A
behavioural-equivalence audit certified this: **7/7 historically-rescued rows route to the
branch the unified router picks from their own state**, 30/30 seed correspondence. So the
frozen historical panel counts as **replicate 1**, and this run is **replicates 2 and 3**:
30 cells x 2 replicates = **60 runs x 250 calls = 15,000 charged docking calls**.

## 2. Why this matters for the paper

- **The paper already has a complete T4 result without this run.** `T4_FROZEN_RESULT_v1`
  (replicate 1) beats GenMol on **24/24 paired cells** against GenMol's mean-of-8, with
  byte-identical qvina02, byte-identical receptors on all five targets, and identical boxes.
  IVG cannot be compared at all — its lead-optimisation runner is absent from its release.
- **These replicates are UPSIDE**: they add variance bars and let the paper say "one
  state-adaptive controller across all thirty cells, three seeds" instead of one seed.
- Therefore: losing this run is survivable. Corrupting the frozen panel is not.

## 3. Current state (measured 04:00, 4.4 h after launch)

    cells started                     45 of 60
    cells with NO checkpoint          3   (unresumable, see s6)
    charged (lock lower bound)        ~1,264 of 15,000  (8.4%)
    measured rate                     ~286 calls/h
    elapsed                           4.4 h
    drivers die at                    21 h  -> ~16.6 h from now
    projected need at current rate    ~52 h

**The run cannot finish.** At 286 calls/h the panel needs ~52 h; the `drive` function times
out at 21 h. Expect ~38-45% charged when the drivers die around 20:00 tonight.

## 4. The binding constraints — all measured, do not re-derive

| constraint | value | can it change? |
|---|---|---|
| Modal workspace container cap | **100 concurrent tasks** (measured at exactly 100/101) | owner says real |
| `run_cell` timeout | 20 h | NO — see below |
| `drive` timeout | 21 h | NO |
| `run_cell` max_containers | 3 per app (=3 cells/arm) | technically yes, see s5 |
| `proposal_worker` max_containers | 36 per app | yes but not binding |
| `dock_worker` max_containers | 24 per app | yes but not binding |
| container size | 1 CPU, 4 GB | yes |

**Timeouts cannot be raised.** `modal_apps/t4_unified_controller_app.py` is one of 29 entries
in every contract's `runtime_inputs_sha256`, re-hashed INSIDE the container by `_validate_task`.
Editing it ⇒ all 10 contracts fail ⇒ re-seal ⇒ `contract_payload_sha256` moves ⇒ each carries
`status: AUTHORIZED_FOR_SCORED_LAUNCH` naming the CURRENT hash, so re-pointing without the
owner naming the new value would manufacture consent. Also `resume_predecessor` is `None` on
all 10. And a Modal timeout cannot be extended for a call already in flight anyway.

## 5. What has been RULED OUT, with evidence (do not re-propose these)

- **Raising per-app `max_containers` does nothing.** The ten arms are using **92 of the 630
  containers their own limits already permit** — ~15%. The per-app caps are NOT binding; the
  100-container WORKSPACE cap is. Raising per-app limits cannot conjure containers.
- **`scaledown_window: 20` is not the cost.** The fastest workers complete in 9-17 s end to
  end. It entered as a copied house convention (git log -S shows it arrived wholesale with
  the app's first commit; four other T4 apps use the same value).
- **Raising `run_cell max_containers` 3 -> 6 is probably NEGATIVE.** ~30 of the ~100 slots are
  already `run_cell` containers that ORCHESTRATE rather than compute. More orchestrators means
  fewer workers.
- **Redeploying for performance is moot for the running arms.** They are ephemeral/detached
  (confirmed: `Function.from_name` returns NotFoundError for all 40 (app, function) pairs), so
  a redeploy does not touch them; capturing any change requires stop + relaunch.

## 6. Where the time actually goes — measured from `worker_telemetry`

Each round issues **12 proposal requests** = 4 parents x 3 lanes (`shallow`,
`anchored_replacement`, `route_complete_region`), then docks 8 selected candidates.

    per-round worker elapsed (jak2_0_r2 round 5), seconds:
      [11.8, 13.2, 16.4, 21.7, 260.9, 377.9, 399.3, 473.8, 576.9, 850.7, 935.6, 954.7]
      serial sum 4,893 s      longest single worker 955 s

- **The per-round critical path is ONE worker at 10-19 minutes of genuine single-threaded
  computation.** No repacking reduces it. At ~31 rounds/cell that is a ~8-10 h per-cell floor.
- **The spread is ~80x** (11.8 s to 954.7 s). Eleven workers finish and idle while the slowest
  runs. This is the main source of the measured **~42% utilisation**.
- Total work ~2,273 container-hours ⇒ **22.7 h floor at 100 containers**, which still exceeds
  the 21 h driver timeout.
- `anchored_replacement` costs ~216 s/parent and contributed **0 of 8 selected** on a sampled
  parp1 cell. It is NOT rescue machinery — it has been in the portfolio since replicate 1.
- The ladder is correctly DORMANT: `support_expansion: None` on every round lock inspected.
  It only fires on terminal exhaustion.

## 7. Resume — VERIFIED FAITHFUL (this is the key enabler)

Verified against two real cells' durable state by running the PRODUCTION `_resume_state`:

- `rng_state` is stored as exact Python ints (PCG64, 128-bit `inc`, 127-bit `state`), survives
  JSON round-trip identically, and is RESTORED not re-seeded. Every round also records
  `rng_state_after`, matching the checkpoint — a per-round audit trail.
- ⇒ a resumed cell continues the same parent stream ⇒ same `controller_seed +
  1_000_003*round + 10_007*parent + 101*expert` seeds ⇒ **same experiment**.
- Double-charging is structurally impossible: an interrupted round's queries are debited in
  full and never re-docked; query ids embed the round index.
- `mode=resume --run-id <id>` works with the code EXACTLY AS IS — no re-seal, no
  re-authorization, same run_ids. Completed cells return instantly without charging.
- Cost of pausing: <=8 forfeited (debited, never re-docked) calls per interrupted cell.
- **HARD RULE: never resume an arm while its original driver is alive** — two `run_cell`
  containers writing one cell folder is a known checkpoint-overwrite race.
- NOT closed: no live forfeit instance was observed, and the unified resume path has NO test
  coverage for RNG or forfeit. Evidence is code structure + the live-data probe.

## 8. A present loss

`fa7_1_r3`, `fa7_2_r2`, `fa7_2_r3` each hold a `round_000_lock.json` with NO checkpoint and
have not progressed in >2.5 h. `_resume_state` refuses exactly this shape ("an unfinished
query lock exists with no recoverable checkpoint"). **These 3 of 60 cells are unresumable**
and their single root docking call each is of uncertain charge status. This is the round-0
preemption window, narrowed but not closed by the round-0 checkpoint commit.

## 9. The decision space

At the 21 h wall, ~38-45% of 15,000 will be charged. The question is what shape that takes.

**Option A — let it run (current).** All 60 cells reach ~40%. Yields **ZERO complete
replicates**; a cell at 100/250 is not a replicate of anything. Paper reports replicate 1 only.

**Option B — concentrate.** Stop some arms, let the rest have the full 100 containers, so
some cells COMPLETE. E.g. delta=0.6 replicate 2 only = 15 cells x 250 = 3,750 calls ≈ 13.7 h
at the current rate — fits inside the wall. Yields **n=2 across a full delta**. Paused cells
resume later for camera-ready. Costs <=8 forfeited calls per paused cell.

**Option C — something better.** This is what the brief is for.

Constraints any proposal must respect:
- Do NOT change the controller, ladder, region law, route lane, proposal draws, or any
  contract field the equivalence audit certified. Container counts and idle windows are
  execution; anything else is policy and invalidates the audit.
- 15,000 charged calls is a hard ceiling; ~1,264 are spent.
- Never convert an uncertain call into a free call.
- Round locks are authoritative; checkpoints are not (they are hash+contract verified here,
  but the standing rule holds).
- The unified lock schema does NOT carry each query's own docked score, only its parent's, so
  `t4_all_runs_reconcile.cell_best` returns None for every cell here. Reconstruct best from
  `parent_score` and treat both scores and charged counts as LOWER BOUNDS.

## 10. Things an ideating agent might usefully question

- Is per-cell completion really the right unit? Could a partial panel be reported honestly in
  some form that is still scientifically meaningful (e.g. "at N charged calls, COMPOSE reaches
  X" as a budget-matched curve rather than an endpoint)?
- The 80x worker spread suggests the 4-parents-x-3-lanes fan-out is badly balanced. Is there
  a scheduling change that is provably draw-identical?
- Is there a way to complete cells that matter most (e.g. the 5 historically-exhausted cells,
  which are the ones the unification claim actually rests on) rather than a delta-slice?
- Could replicate 3 be abandoned entirely, making this a 2-replicate study, with the freed
  capacity completing replicate 2?
