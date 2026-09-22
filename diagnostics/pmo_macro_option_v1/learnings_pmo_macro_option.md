# Learnings — PMO macro-options (branch `pmo-macro-options-20260921`)

Written for merge into `.claude/context/learnings.md`. Kept in the `pmo_macro_option_v1`
namespace rather than appended directly, because other agents were editing the shared
file concurrently and it is append-only.

All numbers MEASURED under the PMO production kernel (python 3.11.13 / **rdkit 2023.09.6**
/ numpy 1.26.4 / torch 2.4.0), `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1`.
**ZERO charged benchmark oracle calls** across every result below: the scorer is the
declared synthetic `valley_similarity_v1`, which carries no benchmark answer.

## 2026-09-22 (macro-options: the reservation sat one hop upstream of the discard)

- **A reservation only binds where the discard actually is, and the geometry that makes
  it binding is a PRODUCTION FACT that a harness can silently break.** The macro-option
  reservation sits in the controller's `_allocate`. It is the binding gate only because
  production PMO sets `candidates_per_batch == QUERIES_PER_ROUND`
  (`pmo_population_v1.py:57, :226`), which sends `prepare_query_batch` down its
  `len(candidates) <= count` branch so `lock_query_subset` discards nothing.
  **MEASURED at 8 allocated against 4 charged: the reserved stage-0 leg was generated,
  reserved and CHOSEN by `_allocate`, and then silently dropped by `select_parent_edits`
  in `parent_edit_search.py`, which picks `queries_per_round` of the allocated pool and
  never sees the reservation.** Across a whole campaign no bridge was ever charged, no
  protection window ever opened, and **nothing raised**. Re-run at the matched geometry on
  the same seed: the leg is locked into round 4's batch, the bridge is charged, the window
  opens. This is the 2026-09-21 rule ("a proposal law bites only where something is
  DISCARDED") one level down — the discard existed, it was just one hop further
  downstream than the mechanism.
- **The tell was a test failing for a reason that had nothing to do with the mechanism.**
  Eight wiring tests failed with "no published round held an open protection window",
  which reads as a mechanism defect. Diagnosing it by instrumenting the real path rather
  than by adjusting the fixture is what found the geometry. `run_arm` now REFUSES a
  mismatched geometry, and the premise is asserted against the production module so it
  fails if production ever stops charging what it allocates.

- **A CONSUMPTION CHECK THAT RUNS ITS CLOSURE ONCE PER HOOK MUST TOLERATE THE CLOSURE
  COMPLETING.** With protection ON the first probe raises from inside `selection()`, so
  `propose_batch` never finishes and leaves no state. With protection OFF -- the negative
  control, the one case that proves the check can fail -- the first probe is never
  reached, `propose_batch` RUNS TO COMPLETION and leaves a pending batch, and the second
  probe dies with "resolve the pending PMO population batch first" instead of reporting
  NOT consumed. **The negative control was failing for the wrong reason, which is
  indistinguishable from a broken harness.** Fix: record a non-probe refusal rather than
  propagating it (it can only push the verdict TOWARDS not-consumed, since a flag is set
  only by the probe being entered), and hand the check a factory that restores a fresh
  controller per probe.
- **Do not build an "unprotected" control by resuming a protected run unprotected.**
  `restore` refuses that outright and is right to: it is the silent arm change the whole
  matched comparison exists to prevent. Flip the flag in the published snapshot and
  re-derive its identity instead, then ASSERT the control holds the same open window as
  the positive arm -- otherwise a NOT-consumed verdict can come from an empty registry.

- **THE LANDSCAPE VOIDED, AND THE GENERATOR WAS THE CAUSE, NOT THE SCORER.** The
  predeclared instrument check fired on the two-seed calibration: `bridges_in_valley`
  0 of 21 in BOTH arms. Cause: `_synthesize_chain` chains `synthesize_structured_program`,
  which this repo has already measured to be additive (atom_insert:atom_delete 2.42:1,
  mean +1.16 heavy atoms), so **6 of 6 declared chains grew monotonically -- 20 -> 25 ->
  26 -> 31 -> 36 heavy atoms** and no intermediate was ever smaller than its own source.
  A valley predicate defined faithfully as "smaller than the run's own starting molecule"
  therefore CANNOT fire. The option generator could not produce the transformation class
  the mechanism exists to protect.
  **The recalibration went to the GENERATOR: a macro option must PRUNE before it
  installs.** After the change, same seed: 8 of 8 options declared, 8 of 8 prune first
  (chains like 20 -> 16 -> 20 -> 25 -> 28), `bridges_in_valley` 0/21 -> 6/22. The valley
  predicate is UNCHANGED -- loosening it to fire on growth would have manufactured a dip
  the motivating measurement does not describe.
- **Report a dip's DEPTH, not just its existence.** 3 of 6 scorable options did have a
  bridge scoring below its own origin even before the fix, but by a few thousandths
  (0.132 -> 0.111) against a measured median relative depth of 0.659 on real transports.
  "A dip exists" without its depth would have made the VOID landscape sound adequate.
- **Calibrate before sweeping when the predeclaration has VOID conditions.** The
  two-seed calibration cost 21 minutes and caught a condition that would have VOIDed a
  three-hour sweep. The predeclaration explicitly sanctions recalibration on a failed
  instrument check, which is what makes this a use of the design rather than a rescue.

- **A SEALED POWER TABLE CAN ASSUME A BASE RATE THE LANDSCAPE DOES NOT DELIVER, AND THEN
  THE MAGNITUDE VERDICT IS DECIDED IN ADVANCE.** The seal's power was computed at an
  assumed base reach rate p0 = 0.40 with a minimum effect of interest of 0.25. When the
  protected arm's OWN reach rate falls below 0.25, no outcome can clear the threshold and
  REJECTED is guaranteed before any data — which reads as evidence about the mechanism
  when it is evidence about an assumption. The sealed verdict still governs, but the
  artifact now also reports the DISCORDANT-PAIR DIRECTION, which exact McNemar conditions
  on and which is therefore base-rate free, explicitly labelled as NOT predeclared.
  Lesson: a predeclared effect size needs a predeclared ATTAINABILITY check beside it.

- **A group floor divided among many simultaneous holders is a per-holder floor of
  nothing.** `protected_parent_weights` guarantees the protected bridges 0.25 of parent
  mass AS A GROUP, which is correct -- a per-bridge floor would be unbounded in aggregate.
  But with 8 options in flight each bridge holds ~0.031, and at ~8 parent draws per round
  its chance of being drawn falls from ~90% (one holder) to ~22%. MEASURED: the dominant
  failure in BOTH arms is `bridge_not_drawn_in_window` -- the window opened, the bridge
  was never drawn, so its continuation was never even offered. The motivating measurement
  (residual 5 of 35 transports, median and MAXIMUM 1 protected round) describes an
  UNCROWDED regime, and running 8 options per seed for sample-size economy is a confound
  the harness introduced, not a property of the mechanism.
- **Decompose the window failure, because the outcome metric cannot.** "Destination
  reached" cannot separate a floor that was applied and still lost from a floor that was
  never large enough to matter. Three consecutive requirements per crossing -- the window
  OPENS, the bridge is DRAWN, the leg is CHARGED -- each fail alone and mean different
  things. It is computed inside `run_arm`, because the driver deletes each seed's campaign
  directory once its artifact lands and a post-hoc reducer has nothing to read.

- **The archive has NO eviction path, verified independently rather than assumed.**
  `ProgramOptimizer.entries` is initialized once (`adaptive_program_optimizer.py:230`) and
  its ONLY write is insert-if-absent (:302-303); there is no `del`, `.pop`, `.clear` or
  reassignment. So a bridge is never EVICTED, and a mechanism that claimed to prevent
  eviction would be claiming to prevent something that cannot happen. The two real
  prunings are both selection: expansion starvation (`selection()` weights by
  1/score_rank, and a bridge scores worse than its own parent by construction) and
  allocation discard.

- **The jump lane spends its whole wall every round (20.1s against a 20.0 budget) while
  the structured lane finishes inside it**, so the pool a declared leg competes against is
  LOAD-DEPENDENT -- a live confound on a machine sitting at load 11 of 12 from unrelated
  agents. The lane was deliberately NOT removed: deleting the main competitor would make
  the mechanism look better for free while halving the run time. The confound is recorded
  per arm instead (attempts, pool candidates, rounds) so a reader can check the two arms
  of a pairing faced comparable work.

- **A mutation pointed at a test that cannot reach it scores as a survivor with no guard
  to repair.** `legs_offered_without_a_drawn_parent` was aimed at the options-DISABLED
  test, where `_macro_option_candidates` is never called at all. The drawn-parent gate --
  the thing that makes the parent-mass floor load-bearing rather than decoration -- had no
  test. Pre-verifying that every mutation string matches exactly once AND that every named
  test exists costs one second and would not have caught this; only asking what each named
  test actually exercises does.
