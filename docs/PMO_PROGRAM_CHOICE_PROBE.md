# Fresh donor-program preference comparison

Authorized by the active T4+PMO controller goal. This is one bounded component
comparison, not replacement of COMPOSE's broad controller or a PMO benchmark.

Use the unchanged preference model from `diagnostics/pmo_program_ranking/model.json`
(file SHA-256 `2069958078422b42a35cac0428904eb8eda8f65b4e3647c8efbe1669ebaf9535`).
Its training uses 256 historical complete edits from 156 parents. The exposed
later-run donor check had 51 endpoint-disjoint edges: 11/17 predicted positive
changes were positive, recovering 11/15 positives. Donor pair ranking was 10/16
correct across eight multi-candidate parents. This is limited evidence, not
calibrated future value. In particular, it ranked the 47-step replication gain
above two alternative edits but below its parent. Do not filter on predicted
positive gain or use this model to prohibit lower-valued intermediates.

Freeze top 16 canonical exact parents from the completed archive comparison,
best 0.6030226892. Keep the original 100 prescreened donors. For each parent,
attempt eight independently drawn donor programs, uniformly choosing a donor and
an oriented single-bridge cut on each side. No replacement attempt after failure.
Use the existing compiler (64 primitive steps, 128 expansions) and executor replay.
Every committed intermediate remains a valid supported molecule. Plans are not
new primitive transitions. Keep exact states, cut operands, actions and failures.

Within each parent, merge completed candidates by canonical identity and aggregate
draw multiplicity. Exclude already-observed canonical endpoints from both query
pools. Baseline selects from this empirical law; learned choice uses the existing
KL=1 preference tilt with 10% reference exploration. These probabilities concern
the sampled complete-program pool, not the original R_theta kernel. Batch feature
computation once across all pools. Couple the two choices using one uniform draw
per parent, seed 20260930. Lock every pool, prediction and selection before any
new oracle outcome. No fitting or within-batch adaptation.

Query only the selected endpoints, at most 32 physical calls, canonical duplicates
charged once. Historical prescreen and development labels remain accounted for.
The endpoint query budget does not hide the 128 internal proposal attempts.
No known public winner is a parent, donor, template, or reward. No docking.

Report paired selected scores and parent-to-child changes, new best, improvement
precision, candidate diversity, realized cycle/size changes, compiler coverage,
failed and duplicate proposals, model time, proposal time, and physical queries.
A fresh ranking advantage plus constructive improvements earns integration into
the broad optimizer for a short multi-round comparison. Avoiding damage alone
without constructive gains is insufficient. If the learned choice is no better
than random, stop that unchanged model. Few distinct choices are inconclusive,
not a pass. This one-component check cannot establish generalization or SOTA.

Execution is local CPU with the already qualified RDKit 2024.03.5/PyTDC 0.3.6
oracle environment and unchanged compiler. No neural R_theta enumeration, GPU,
Modal allocation or deployment. Maximum 300 seconds per invocation; restart from
completed proposal receipts, never implicitly repeat an unresolved oracle call.
Expected 30-90 seconds based on the earlier 128-program, 30.884-second probe.
Record input/code/recipe hashes, exact software/hardware, source worktree state,
atomic candidate and oracle receipts. No remote dollar spend.
