# Project status — 2026-07-19

> **Live override:** both bounded continuations have completed and
> are retired.  The fresh-optimizer arm failed to improve the selected
> step-2,500 quotient checkpoint.  The legacy-transfer arm copied 109
> name-and-shape-compatible tensors and trained for 500 cached updates; its best
> validation point was step 400 at loss 12.7654 and 53.81% family accuracy,
> versus the incumbent's loss 11.6977 and 58.08%.  It was therefore rejected
> before rollout or FCD evaluation.  The original step-2,500 quotient weights
> remain selected.  Their 100-sample replay localizes the main chemistry error:
> every new three/four-member ring came from whole-ring grow, even though those
> topologies comprise only 3.01% of the catalog base measure.  The next bounded
> support-versus-neural-logit audit is now complete.  It found that late states
> can retain only 2--5 executable ring templates and that every surviving
> template is a 3/4-member topology, forcing a small ring after the family head
> selects ring-grow.  A matched inference-only support ablation is running; no
> new operator or full cache build is authorized before its result.  Both
> bounded cached retraining arms are now stopped.  The hierarchical control
> plateaued at 52.67% family accuracy at step 1,000.  The superposed marked-rate
> arm briefly reached 56.94% at step 750, then declined to 55.07% and 54.61%
> at steps 1,000 and 1,250.  Neither approaches the retained pancake
> checkpoint's 84.44%, so no more GPU updates are authorized on those
> formulations.  Work has returned to the exact step-6,250 pancake model: a
> 100-sample inference rescue is running with canonical self-Grafts removed,
> and a validity-certified early-ring schedule is locally verified before any
> retraining decision.
> The current decision schedule and 48-hour work boundary are in
> [`48_HOUR_RESULTS_TRACKER.md`](48_HOUR_RESULTS_TRACKER.md).

This is the operational status for the current working tree. It separates
verified implementation, completed cloud gates, the next launch, and legacy
evidence so a collaborator can tell what is actually running.

## Completed and retained

- The lossless Git handoff, three canonical HTML plans, ICLR draft, and legacy
  step-6,250 trajectory archive are versioned in this repository.
- Canonical successor aggregation removes molecular self-Grafts and sums alias
  rates reaching the same chemical successor.
- The factorized ring decoder uses shared executor-aware semantic support for
  teacher likelihoods and ancestral sampling. The retained teacher audit has
  zero missing teacher marks.
- Flexible-size carbon-tree transport, atom grow/shrink/retype, Graft,
  bond-order changes, and atomic whole-ring-system grow/delete are integrated
  into target-free ancestral CTMC sampling.
- Validation-based early stopping is configured every 250 updates after a
  500-update warmup, with six materially non-improving evaluations of patience
  and exact recovery of the selected checkpoint.
- The canonical early ancestral preview gate now requires at least 65%
  validation-loss improvement, family accuracy of at least 0.75, and a selected
  step of at least 1,000. These thresholds are persisted in the launch marker.
  The active immutable `2be9258` run predates this change; any preview launched
  under its 50%/0.60/250 rule is preliminary and does not replace the stricter
  100-sample chemistry review.
- Production training now saves the selected checkpoint and releases the A100
  before the final 2,000-sample/FCD job begins on CPU. The CPU evaluator is
  retry-safe: request identity, checkpoint hash, rollout signature, cached
  attempts, and metrics are persisted atomically.
- Recovery preserves the original step-zero validation baseline, rejects
  changed source/recipe/data provenance, and treats a checkpoint saved at an
  already-triggered early stop as terminal rather than training extra updates.
- Compact per-event trajectory evidence now records canonical molecular states,
  exact atom counts, authoritative validity/connectivity flags, self-events,
  immediate reversals, and delete-to-one/regrow behavior. Fresh evaluation
  caches require this evidence and reject corrupt or terminal-only payloads.
- Fixed validation/test features are now content-addressed by their complete
  scientific configuration, built once on a CPU-only stage, and reused across
  runs. Production GPU stages fail fast if this cache is absent.
- Model evaluation streams the frozen validation/test sets in 64-example
  microbatches and reuses the already-computed step-zero metrics. Training
  batch size remains 64; the streamed reduction is mathematically the same
  full-set evaluation without full-set GPU residency.
- Exact training ring support is now a content-addressed sparse cache compiled
  entirely before GPU allocation. A trainer must load the requested rows and
  cannot silently fall back to online chemistry. Cache identity includes the
  path/sampling signature and ring-support semantics, so neural and optimizer
  changes reuse it while rewrite-semantic changes invalidate it.
- The full local suite passes **339 tests** with `PYTHONPATH=.:src`, with one
  pre-existing non-failing tensor-conversion warning.  This includes the new
  ring-calibration helper, checkpoint-evaluator integration, sampling-support
  fallback, and distributed-rollout cache identity checks.
- This includes exact adjacent-commutation scheduling and legacy pancake
  checkpoint compatibility.

## Completed cloud gates

Three clean 200-update gates established the production resource choice:

| Configuration | Final interval | Wall time | Selected validation | Family accuracy |
|---|---:|---:|---:|---:|
| A100, 24 CPU, 16 data workers | 45.20 examples/s | 435.82 s | 19.0701 | 0.7436 |
| H100, 24 CPU, 16 data workers | 51.80 examples/s | 393.86 s | 19.0842 | 0.7436 |
| A100, 32 CPU, 24 data workers | **54.52 examples/s** | **391.95 s** | **19.0701** | **0.7436** |

The scaled A100 configuration is the production choice: it is faster than the
measured H100 configuration and materially cheaper at current Modal rates. At
the observed training rate, 30,000 updates project to roughly 9.8 raw GPU hours
and about $40 of A100/CPU/RAM container cost before any early stop.

All 16 scaled-A100 rollouts were valid, connected, non-null, unique, and novel,
with zero event-budget exhaustion. The forward event mix was Graft 37.5%, atom
insert 16.8%, delete 12.9%, restate 17.9%, bond reorder 5.6%, ring grow 9.1%,
and ring delete 0.2%. This is not delete-to-one collapse or single-family
collapse.

The chemistry is deliberately not declared solved from 16 samples after 200
updates: 7/16 contained a three- or four-membered ring and every sample was
polycyclic. Macrocycle and cage-candidate counts were zero. That is an
undertraining warning to test at 100 samples, not a production result.

The commit-`2be9258` deployment-boundary smoke also passed end to end on
2026-07-19: CPU compilation completed, the 16-example teacher audit had zero
failures, A100 training skipped in-container rollouts, and a separate CPU job
loaded the selected checkpoint and produced 16/16 valid samples with complete
trajectory diagnostics and a finite infrastructure-only FCD. Its two-update
FCD is not a model-quality result.

## Latest production attempt

- The commit-`2be9258` production attempt under
  `compose-v4-stage3-flexible-graft-prod-2be9258-v1` was stopped before any
  optimizer update. It spent 24m49s building 2,048 validation examples and
  27m33s building 4,096 test examples inside the A100 container. The first
  full-set validation occupied nearly the entire 40-GB A100; the duplicate
  step-zero validation then failed while requesting another 15.62 GiB. Modal's
  deterministic retry was stopped before it repeated the same failure.
- This was an evaluation-pipeline scaling failure, not evidence of slow or
  unstable training. The retained compiled path shards and zero-failure
  teacher audit remain valid and can be reused losslessly by the corrected run.
- No corrected-checkpoint 100-sample preview or 2,000-sample FCD estimate exists
  yet.
- The archived step-6,250 samples and FCD dashboard remain retired,
  pre-quotient diagnostics only.
- The corrected production evaluation cache completed on a 64-CPU stage under
  `compose-v4-stage3-flexible-graft-prod-7ea89d0-v1`. Validation feature
  construction took 711.61 s and test construction took 1,215.76 s; the full
  remote compile stage took 2,225.99 s (37m06s), including startup, loading the
  retained path shards, and finalizing ring support. The resulting 577-MB
  content-addressed artifact covers the fixed 2,048 validation and 4,096 test
  examples. Runs with the same scientific signature reuse it without chemistry
  recompilation, saving about 37 minutes relative to this CPU path and 52
  minutes relative to the retired in-A100 path.
- The next bottleneck was isolated to exact ring-support construction for the
  deterministic training stream. Computing it inside the A100 DataLoader took
  roughly 7.5 minutes per 64-row batch and left the GPU idle. A 60-worker
  single-container compiler restored parallel CPU use, but each worker retained
  enough of the 4,096-template chemistry engine that memory exceeded 372 GB
  before the first 16,000-row shard completed. That CPU-only gate was stopped;
  no GPU was allocated.
- Support compilation now accepts shard-aligned absolute step ranges. The
  launch surface partitions them across isolated 14-core/12-worker Modal
  containers with unique immutable run directories and non-overlapping atomic
  shard files. This changes only execution placement: exact masks, deterministic
  row indices, and the training objective are bit-for-bit unchanged.

## Immediate sequence

1. **Done:** commit and push the trajectory-audited deployment code; pass the
   real CPU-to-A100-to-CPU boundary smoke.
2. **Done:** reuse the immutable `2be9258` path cache, build the signed fixed
   evaluation cache on a 64-CPU stage, and pass the streamed-evaluation
   integration gate.
3. **Active:** the first isolated 16,000-row gate completed at 5.15 rows/s and
   atomically validated its shard. Compile the seven remaining early-training
   shards in parallel across seven 14-core containers (98 allocated CPUs), then
   verify every sparse shard before allocating an A100.
4. Relaunch A100/32-CPU/24-worker training from the commit containing the exact
   successor optimizations and required support-cache consumer. At the first
   credible checkpoint, inspect 100
   ancestral samples and their compact full-trajectory diagnostics.
   Hard-stop on any validity/connectivity failure, event-budget exhaustion,
   canonical self-event, delete-to-one recurrence, severe operator collapse,
   or the preregistered distribution/ring thresholds. Treat n=100 FCD and
   missing rare spiro/bridged modes as warnings, then recheck at step 1,000.
5. Let validation early stopping select the checkpoint. After the GPU is
   released, run the automatic CPU 2,000-sample/FCD evaluation with full
   trajectory, operator, size, element, bond-order, and ring diagnostics.
6. Once unconditional sufficiency is established, implement valid-rewrite
   recovery and run the matched QED/conditional-guidance gate required before
   COMPOSE-Lipid candidate generation.

## Lossless successor-computation implementation

The four planned exact optimizations are now implemented locally:

1. An LRU keyed by the complete slot-aware molecular state reuses topology,
   application conditions, resonance-invariant bonds, Graft quotient groups,
   and ring-delete actions. The cache is bounded per persistent worker.
2. Exact colored-tree vertex and ordered-pair automorphism-orbit identifiers
   eliminate redundant Graft successor work on sufficiently symmetric trees
   while retaining every labeled Graft logit and its total successor rate.
3. Local Graft reroutes reuse unaffected directed-branch identifiers and
   recompute only branches affected by the cut and attachment instead of
   recanonicalizing the complete colored tree for every candidate.
4. Exact ring-template support is serialized and transported as CSR indices;
   the dense 4,096-template mask is materialized only at the neural decoder
   boundary on the target device. Existing version-1 evaluation caches are
   migrated losslessly in memory and remain reusable.

The optimized and exhaustive Graft masks, removed-neighbor arrays, and
canonical-successor groups agree exactly on chains and random colored trees up
to 40 atoms. Dense and sparse ring support produce identical enabled families,
selected-mark log probabilities, and Generator Matching loss. A repeated-state
40-atom collation microbenchmark improved from 3.64 s to 0.11 s (31.8x) in the
cache-hit-heavy case; this is an upper-bound microbenchmark, not a claimed
end-to-end training speedup. Representative 64-by-4,096 ring support occupied
3.7 KB in CSR form versus 262 KB dense (71x smaller). Small or asymmetric trees
fall back to the exhaustive Graft path when orbit reuse is not beneficial.

These changes do not alter rewrite rules, the canonical-successor quotient,
teacher rates, reachable states, batch size, or evaluation examples. Persistent
fixed-evaluation features remain content-addressed and reusable across runs.

## Claim boundary still open

The successor quotient resolves measured presentation-level Graft churn, but
full invariance of learned rates to arbitrary atom-slot permutations remains a
formal and empirical audit obligation. Validity-closed guidance has a strong
architectural advantage because every reward or oracle query sees an executable
molecule, but better conditional sample efficiency must be demonstrated against
matched filtering, repair, and guidance baselines rather than assumed.
