# Online complete-plan proposal learning

## Identity and scope

Problem: useful executable edits are rarely proposed. Output: a conditional
distribution over complete molecular edit plans, followed by primitive executor
compilation, not a new molecular operator or endpoint teleport.

Hypothesis: learning from newly scored complete edits changes subsequent proposal
probabilities and increases useful offspring relative to the identical frozen
policy. This is a warm exposed Perindopril development comparison, not official
PMO AUC, exact Doob control, or a test of learned future value.

The user approved trying the proposed controller and continued implementation.
Bound: one paired comparison, four rounds, sixteen parents per arm, at most 128
new oracle calls total, at most 29 CPU workers plus one driver, $10, no GPU.
Expected proposal time is 5–15 minutes based on prior broad-reference runs.
Driver stop: 900 seconds; individual workers: 180 seconds. CPU fitting is bounded
to the fixed recipe below and performed separately before deployment. No other
task or replicate is authorized by this recipe.

## Frozen recipe

- Start both arms from the same best sixteen available exact saved development
  states, preserving the current 0.6835298931 champion. Retain the original 116
  donor molecules. All inputs are hashed; legacy bank provenance remains incomplete.
- Keep a 50% unchanged broad reference channel, including generic/local/global
  chemistry. The remaining 50% is the existing optional donor-program channel.
  No new whole-ring catalog, executor change, or reference-model training.
- For a donor proposal, sample 64 donor/cut plans from the existing prior. Remove
  unsupported-size, invalid and self plans with explicit counts; canonical-dedup
  the remainder. Choose one with 20% uniform exploration plus 80% learned softmax.
  Plans are cheap hypothetical endpoints. Only the chosen plan is compiled; every
  intermediate and endpoint must pass the existing executor and exact replay.
  A failed compilation is an explicit no-offspring event, not an invented score.
- Initialize a molecular encoder on paid endpoint labels, with a source/scaffold
  grouped diagnostic split frozen before fitting. Fixed architecture/optimizer and
  epochs, no validation-driven hyperparameter selection. The endpoint head is an
  initialization signal, not a future-value model. Preserve all label provenance.
- Add a context-dependent proposal residual on source/product embeddings and
  their difference. Both arms start identical. Only the learning arm updates this
  residual after a complete round, using observed score-minus-parent advantages
  and clipped policy ratios on the recorded proposal pools. Freeze the encoder
  and endpoint head during this short comparison. Snapshot each policy and its
  training receipts before the next round. No within-round oracle feedback.
- Preserve the common archive/parent allocation. Private arm histories drive
  updates; shared physical query cache does not disclose unrequested scores.
- Programs use the valid executor, but their probability under original R_theta
  is not certified. Do not claim the original-reference KL or Doob identity.

## Decision and accounting

Primary: final archive top-ten mean and mean round-end top-ten mean improve over
the frozen arm, with no lower best score. Also report improving-offspring fraction,
distinct molecules, intended release versus realized scale, program completion,
primitive depth, ring/cycle changes, and proposal/training/oracle time. Report
learning-induced probability changes separately from benchmark improvement.

A positive result earns a proposed fresh-seed replication, not a SOTA claim.
A null/reversal stops this unchanged recipe. Few completed novel plans, negligible
policy change, or near-identical initial and final sampling makes the result
inconclusive for learning; report that failure mode rather than expanding the run.
All prior prescreen/development labels and fresh physical calls remain explicit.

Checks: plan-versus-replayed endpoint equality, normalized positive proposal law,
reward-direction update on a small fixture, fixed-arm immutability, no cross-arm
training, deterministic snapshots, and budget/lock/restart behavior of the reused
runner. Focused checks during development; no claim of a completed benchmark or
release milestone without its broader verification.
