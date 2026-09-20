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

Fix oracle asset resolution for the oracle's full lifetime (absolute paths or a stable
worker dir, not a constructor-only chdir). Before any scored rerun, a reference panel that
**actually calls** the production oracle in a fresh process against pinned values including
a nontrivial intermediate -- not merely "an active is positive". Cover every asset-backed
evaluator; keep reference molecules out of initialization, archives and selection.

Priority: **realization/attachment first** (alpha is the leading hypothesis, not a proven
universal cause), then trajectory autopsy, then hierarchical credit where evidence supports
it. Program length is separate from binding; a 32-primitive cap is not a theorem about
reachability.

No-prescreen only. Ladder 250 -> 1k -> broader panel -> 10k matched with multiple seeds.
**Never scale a 250-call AUC and compare it to a 10k result.** Reproduce the official AUC
implementation for official comparisons.

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
