# Project status — 2026-07-19

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
- The first early ancestral preview launches from an immutable checkpoint only
  after at least 50% validation-loss improvement, family accuracy of at least
  0.60, and a selected step of at least 250.
- Production training now saves the selected checkpoint and releases the A100
  before the final 2,000-sample/FCD job begins on CPU. The CPU evaluator is
  retry-safe: request identity, checkpoint hash, rollout signature, cached
  attempts, and metrics are persisted atomically.
- Recovery preserves the original step-zero validation baseline, rejects
  changed source/recipe/data provenance, and treats a checkpoint saved at an
  already-triggered early stop as terminal rather than training extra updates.
- The full local suite passes: **266 tests**, with two non-failing warnings.

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

## Not currently running

- No 30,000-update quotient-correct production job is active yet.
- No corrected-checkpoint 100-sample preview or 2,000-sample FCD estimate exists
  yet.
- The archived step-6,250 samples and FCD dashboard remain retired,
  pre-quotient diagnostics only.

## Immediate sequence

1. Commit and push the verified GPU-release, CPU-evaluation, immutable-run, and
   exact-resume changes.
2. Verify that a commit-bearing production run label is absent on the Modal
   volume, then launch the A100/32-CPU/24-worker pipeline.
3. At the first credible step-500 checkpoint, inspect 100 ancestral samples.
   Hard-stop on any validity/connectivity failure, event-budget exhaustion,
   canonical self-event, delete-to-one recurrence, severe operator collapse,
   or the preregistered distribution/ring thresholds. Treat n=100 FCD and
   missing rare spiro/bridged modes as warnings, then recheck at step 1,000.
4. Let validation early stopping select the checkpoint. After the GPU is
   released, run the automatic CPU 2,000-sample/FCD evaluation with full
   trajectory, operator, size, element, bond-order, and ring diagnostics.
5. Once unconditional sufficiency is established, implement valid-rewrite
   recovery and run the matched QED/conditional-guidance gate required before
   COMPOSE-Lipid candidate generation.

## Claim boundary still open

The successor quotient resolves measured presentation-level Graft churn, but
full invariance of learned rates to arbitrary atom-slot permutations remains a
formal and empirical audit obligation. Validity-closed guidance has a strong
architectural advantage because every reward or oracle query sees an executable
molecule, but better conditional sample efficiency must be demonstrated against
matched filtering, repair, and guidance baselines rather than assumed.
