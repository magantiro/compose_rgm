# Locked complete-plan pool prevalence assay

## Question and scope

The learned complete-plan comparison changed proposal probabilities but did not
improve the retained Perindopril MPO champion. Its post-run audit cannot determine
whether the recorded pools lacked useful products or whether the controller ranked
useful unscored products away: 814 of 890 distinct pool products had no pre-run
label. This assay separates those explanations on the already generated pools.

The preparation stage is offline and adds zero oracle calls. It uses the complete
run artifacts from plan-policy run
`170ca5944c646076d6adf1be75edd5ba781119eddc23d12fa6d2d11b45510e4b`.
It does not generate new plans, train a model, change the executor or reference
process, or use a public winner. A later scoring stage is not authorized by this
document.

## Frozen lock rule

Retain plan pools whose exact parent score is at least 0.65. Order distinct parent
structures by descending score and canonical identity. Traverse their pools in
round-robin order, with each parent's pools ordered by round, slot, and worker
identity, and retain the first 32 pools.

Remove every product scored before or during the completed comparison. Within each
retained pool, form three distinct queues of at most three products each:

1. `actor_top`: descending recorded behavior-policy probability;
2. `uniform_hash`: ascending SHA-256 order under the declared assay seed;
3. `largest_release`: descending intended released fraction.

Break every tie by canonical product identity. A product may occur in only one
queue across the complete lock. Queue membership is fixed before compilation.
For each queue, compile candidates in order through the existing pendant-transplant
compiler until the first successful executable program. Compilation status may
choose the first executable member but task value may not. Preserve every attempted
compile receipt and exact replay state. A queue with no compiled member abstains.

The proposed later label cap is 96 unique compiled products, one per role and pool.
All products must be locked before any requested score is exposed. Previously paid
scores are request-only cache hits and are excluded from this new-label census.

## Decision after labels

- If any locked product exceeds the current 0.6835298930947339 champion, bank the
  improvement and compare which role exposed it.
- If random or largest-release choices improve parents while actor choices do not,
  repair pool ranking and uncertainty-aware exploration before changing proposals.
- If actor choices improve parents but not the champion, the channel is locally
  fertile but insufficient on its own; combine it with broad option pools rather
  than scaling the same transplant actor.
- If no role improves any parent, reject these donor-transplant pools as the next
  search substrate and move the same locked-pool design to the full
  WHERE/WHAT/HOW macro and generic proposal mixture.

Parent improvement, champion improvement, role-wise hit rate, score-delta
distribution, structural diversity, intended release, realized primitive depth,
compilation coverage, proposal time, and all oracle accounting must be reported.
This is exposed warm development, not official PMO area under the curve or evidence
of external superiority.

