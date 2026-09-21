# Mission (standing goal)

Make COMPOSE as competitive as possible on the **complete T4 and no-prescreen PMO
benchmarks**, and produce a defensible ICLR paper. Be aggressive about it.

**Standing authorization.** Inspect, diagnose, develop general improvements, write tests,
commit and push scoped changes, and **launch the bounded experiments below once their
checks pass**. Do not re-ask for a "go" on an already-authorized step. Do not polish
diagnostics while the experiment they enable sits unlaunched. Not authorized: unbounded
spend, off-ledger oracle calls, destructive cleanup. Every scored run needs an explicit
budget, identity and ledger.

**Loop:** diagnose the real failure -> smallest general mechanism -> test -> measure
objective performance -> keep, revise, or discard. "Infrastructure works" and "support
improved" are not success while optimization is weak. One bad early score is not a reason
to redesign a working algorithm.

## Measurable progress overrides activity

The primary outputs are exactly two:

1. a complete, correctly reconciled T4 table at delta=0.6 and delta=0.4;
2. increasingly competitive **scored** PMO results.

Agents spawned, diagnostics written, tests passed, commits made and hypotheses developed
are **not** progress toward those outputs. Do not report them as if they were.

**T4 -- patience with healthy computation, impatience with blocked cells.** Keep every healthy
cell running and use authorized Modal capacity aggressively to cut wall clock; never interrupt a
good run merely because it is incomplete. Track per cell: charged calls / budget, best score, IVG
gap, and state = healthy / exhausted / rescue / complete. Finish the pruning-capable rescue gate
and move straight into scored rescue runs when it passes. Add no new T4 mechanism work unless an
existing rescue fails or a live cell reveals a new blocker. **Seed-only rows are not generated
COMPOSE successes.** The target is all 15 cells resolved at both thresholds, not further polishing
of cells already winning.

**PMO -- impatience with further unscored analysis.** Shift from diagnosis to scored iteration.
Corrected GSK3B v1 must reach a real scored 250-call run, and it must NOT block v2. The
attachment/realization repair must reach a matched scored 250-call v2 experiment on
Perindopril/Celecoxib, plus GSK3B once its oracle is ready. Do not run another broad diagnostic
cycle before scoring the strongest currently justified controller revision. After each scored
version decide: promote to 1k, revise on trajectory evidence, or reject. Hierarchical credit, new
priors and additional machinery enter only when measured trajectories justify them.

**Scoreboard.** Maintain a compact progress scoreboard updated from authoritative artifacts --
round locks for T4, ledgers for PMO. T4: completed cells, active calls, rescue status,
searched-cell W/T/L. PMO: controller version, task, calls, best score, AUC/top-k, and delta versus
the previous matched version.

Orient parallelism toward shortening the path to these outputs. Once an experiment is validated
enough to answer its question, **run it**. Avoid low-value housekeeping, repeated audits and
infrastructure that does not unblock a scored experiment. Be principled, but bias toward execution
whenever a bounded experiment can resolve the uncertainty.

## Architecture to preserve

    state + constraints -> structural/dependency-region program policy
      -> constrained realization -> exact execution -> reward feedback + recursive control

A structural action is `Z = (R, H, alpha, D)` -- retained/released region, topology,
attachment, dependencies -- parameterized by scale, mode, duration.

**Do not replace this with primitive autoregression plus beam search.** No production fix
built on widening a beam or keeping hash-ranked prefixes; a beam comparison is a
*diagnostic of the current implementation only*. Never substitute an arbitrary executable
prefix for a requested complete transformation. Do not swap in MCTS/PUCT/GFlowNet/SMC
because the name sounds newer. Resolve ambiguity with constraints, dependencies and
compatibility propagation.

Exact execution does not imply cheap planning or transfer; a dependency graph is not a
proven quotient; a heuristic allocation is not a Doob transform.

## Controller scope: shared process, task-appropriate control

The claim is **a common validity-closed generative process over molecular graphs**, with task
constraints and objectives expressed through a shared structural control interface -- NOT one
identical controller configuration for every benchmark. The strongest baselines do not impose that
on themselves: GenMol changes its inference procedure per task (fragment remasking for PMO/lead-op,
molecular-context guidance for guided generation, separate pipelines per family, and V2 beats V1 on
de novo/fragments while V1 remains better on PMO); InVirtuoGen uses direct flow sampling for de
novo, a dedicated conditioning procedure for fragments, GA+PPO for PMO, a docking-PPO pipeline for
lead optimization, and even a different checkpoint for de novo/fragments than for PMO.

Three levels, and they are different:

- **ACROSS benchmark families -- different controllers are legitimate.** Fragment-constrained
  generation, black-box optimization and constrained lead optimization are different problems; de
  novo has no objective feedback at all. Forcing one algorithm across all four is artificial and
  costs performance.
- **WITHIN a family -- the algorithm and its hyperparameters are FROZEN across instances.** One PMO
  controller over all 23 oracles, only the oracle changing. One lead-optimization controller across
  every protein, seed and delta -- target, start molecule and delta are benchmark INPUTS, never
  reasons to retune. One fragment controller across every drug and task type.
- **NEVER per-instance.** No GSK3B controller, no BARICITINIB attachment rule, no BRAF-specific
  proposal probabilities, no `if target == 5ht1b`. A mechanism must activate from structural
  features or a constraint specification, never from an identity. That is the line between a
  general capability and benchmark engineering.

A fragment controller consuming the benchmark's own declared constraints -- motif, attachment sites,
locked atoms, linker endpoints -- is NOT a hack; it is what a fragment-conditioned generator is
supposed to do, and it may legitimately behave differently for linker design than for motif
extension because the constraint itself differs.

**Consequence for T4.** The region-repair and protonation mechanisms are general structural proposal
channels, not per-target rescues. The final panel should be ONE T4 controller carrying shallow,
structured, region and protonation-aware channels, routed by molecular-state features rather than
protein identity, then frozen and run over the whole panel. The current per-cell rescue arms are
development experiments that establish which mechanisms work; they are not required to be the
canonical final evaluation.

## Reasoning discipline

Per intervention record: observation -> a competing explanation -> cheapest discriminating
test -> what changes vs stays fixed -> what outcome promotes or kills it. Keep separate:

- not sampled != impossible; search timed out != no binding exists
- teacher route replays != the deployed generator can find it
- eligible molecule != docking quality; shared tokens != demonstrated transfer
- late improvement != attribution; similar mean != "differs only in budget"
- cold-start counterfactual != causal decomposition of an adaptive campaign

Inspect real molecules and lineages when they discriminate. Reconcile by exact identity
(task, source, delta, controller version, seed, run id). **Never merge run directories by
picking whichever has the best score or most calls.**

## T4

Continue healthy runs at both thresholds; do not restart to normalize inherited settings,
but record them -- never claim identical hyperparameters. Original source stays the
similarity reference.

Rescue triggers on **measured candidate exhaustion**, never on target identity or on losing
to IVG. One general mechanism: normal proposal -> empty eligible pool -> bounded
feasibility-directed recovery -> hand candidates back to the existing loop. Defaults, not
sacred: 256/512/1024/2048 extra attempts, <=3840 per event, stop at ~4 distinct clean
eligible; use 1-3 rather than discard valid support; explicit wall-clock and per-cell caps;
reuse cached evaluations. More attempts alone do not fix a distribution that misses the
transformation.

Keep the chemistry screen narrow. Separate benchmark eligibility / quality screening /
synthetic-suitability claims. Do not add filters until support vanishes.

Subtract the **reconciled** charged-call count, never an assumed history. **Version every
rescue.** It must never splice silently into an "unchanged v1" table -- name the
configuration and the rescue phase, or run the matched confirmation.

## PMO

**RESET 2026-09-21.** The objective is no longer to recover teacher-shaped structural jumps.
It is to **improve the top-ten score curve with a general feedback-driven controller**. The old
mechanism diagnostics -- a ~31-primitive median, retained fraction well below 1, teacher-geometry
recovery -- are **NOT admission requirements** for useful PMO search and must never gate whether a
promising optimizer continues. Measured: support-conditioned pairing lifted exact realization
10.8x (0.275% -> 2.96%) and every success was still purely additive, which proves the transplant
MECHANISM transfers poorly, not that PMO needs deletion-heavy jumps.

**One controller plus a task adapter, shared with T4.** The loop is the same in both:
`choose parent -> choose legal region -> choose structural program -> execute exactly -> feed back`.
Only the adapter differs -- T4 supplies a lead plus similarity/QED/SA constraints and docking and
optimizes best-eligible; PMO supplies a task-independent initial population and a scalar oracle and
optimizes the top-10 frontier. Never grow a separate pile of PMO-specific hacks, and never import
T4's SA/QED eligibility into PMO, where the task does not supply it.

**Proposals are PARENT-FIRST:** `G -> R subset G -> (H, alpha, D) | G, R -> G'`. Choose the region
on the actual parent, then construct a compatible refinement, replacement or recombination. Never
draw a historical plan and ask the current molecule to resemble that plan's original source. Hard
constraints stay hard (valence, connectivity, real attachment obligations, declared program
semantics); source-side *descriptors* may be preferences -- separate those deliberately rather than
loosening every matcher until something binds. Retire expensive source-specific transplanted plans
from default allocation; do NOT replace them with production beam search.

**Information boundary.** Task-independent chemical prior + current-run COUNTED feedback. Online
adaptation from scored observations is legitimate no-prescreen optimization. Oracle internals,
target SMILES, hidden component scores and uncounted same-task history are NOT. If QED is the
objective, computing it on uncounted candidates for selection IS objective evaluation.

**Learn structural CONTENT, not just allocation.** Two memories: donor regions from high-scoring
molecules (associations, not causal labels), and edit outcomes `(G, Z, G', f(G), f(G'))`. Update
what gets PROPOSED, not merely which channel receives budget. Reuse the existing archive,
ProgramValue, credit and operator infrastructure -- verify current behaviour first so components
are not duplicated, and remember a validated repair behind an opt-in keyword is INERT until a
caller passes it.

**Allocation targets frontier utility.** `U10 = mean of the top ten scored`; prioritize by
predicted endpoint value and `delta U10`, using parent-relative change as learning evidence rather
than the sole objective. Keep explicit exploration and some structurally distinct non-elite
parents. Track proposal COST separately from oracle utility: an easy generator must not monopolize
a run by completing reliably, and an expensive one must not hold a fixed share by sounding
powerful.

No-prescreen only. **1,000 calls is the development horizon, 250 an intermediate diagnostic.**
Hold initialization FIXED across matched arms. Development set: gsk3b, celecoxib_rediscovery,
perindopril_mpo, then C7 isomers, Median1, Valsartan SMARTS -- freeze the recipe before the
remaining tasks, report all 23 with development status clear. **Never scale a 250-call AUC and
compare it to a 10k result.** Reproduce the official AUC implementation for official comparisons.

Oracle assets: resolve for the oracle's full LIFETIME (not a constructor-only chdir), and before
any scored rerun run a reference panel that **actually calls** the production oracle in a fresh
process against pinned values including a nontrivial intermediate -- not merely "an active is
positive". Keep reference molecules out of initialization, archives and selection.

## QED and fragments

QED gets a **dedicated task** -- never a T4 label that silently imports docking thresholds
or the SA <= 4 restriction. Verify operative thresholds, maximization direction, source
panel, returned-output rule and atom-capacity semantics. Match sources and **actual work**,
not returned K alone: primitive transitions, molecules scored, endpoints generated and CPU
time are not interchangeable. Verify the GrIDDD source and panel before any comparison.

Fragments are lower priority and must not delay PMO scoring or T4 rescues. The gap is the
repeated-completion sampler, not the executor. Test constraint satisfaction, uniqueness,
diversity and the official quality metrics -- not validity alone.

## Runtime correctness

One executable configuration authority for budgets, thresholds, task identity, lanes,
assets, seeds, horizons. Verify **runtime-resolved** values before launch -- a signed JSON
does not prove its fields are consumed. Three checks: targeted tests that fail when the fix
is removed; synthetic-scorer end-to-end including restore, budget termination and worker
serialization; a small real-oracle reference check through the deployed adapter. Revalidate
by impact.

State recovery is not optional. A run that rolls back its ledger is not self-healing.
Persist reservations, archive, credit state, RNG, pool identity, remaining budget. Never
resolve conflicts by blind min/max. Uncertain calls must not become free calls. **Never delete an active artifact**; preserve
before removing, and verify the copy is readable from its durable home first.

## Agents

**Every delegated agent must use Fable.** One coordinator owns integration and launches. Each writer gets its own
worktree and output namespace; assign file ownership. No shared-scratch globbing, no broad
`git add`, no concurrent edits to one pinned module. Reuse existing agents before spawning
duplicates. Schedule computation against real resources.

Handback: finding + confidence; artifact/commit; what it falsified; proposed change; tests
and negatives; cost; next executable action. Then decide: integrate, score, revise, park.

## Execution order

1. Verify and continue healthy T4 campaigns
2. Complete qualified T4 rescues, unblocked by unrelated research
3. Fix and positively verify GSK3B, run the fresh 250
4. Finish the PMO autopsy and source-compatible realization repair
5. Score the next justified PMO version; test hierarchical credit where evidence supports
6. Advance promising versions through 1k and the broader suite
7. Repair and run the QED comparison
8. Spare capacity: fragment sampler, evaluation cleanup

The aim is not to make development results look good. It is **as many genuine,
reproducible T4 and PMO wins as possible**, with implementation and scientific explanation
in agreement.
