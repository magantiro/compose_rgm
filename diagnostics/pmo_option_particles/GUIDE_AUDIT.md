# The saved achieved-route head did not rank realized reference futures well

This is a retrospective descriptive check of the completed run, using only its
saved pre-outcome predictions and independently sampled reference trajectories.
No new oracle evaluations, model evaluations or rollouts were performed.
`guide_audit.json` binds the verified run artifact, analyzer and software.

At the first option boundary, the head's highest-valued candidate had current
score 0.002166 and predicted achieved value 0.780725. Its sampled remaining path
only reached 0.079737. The immediately best candidate was 0.522233 and its
sampled suffix maximum remained 0.522233. Among eight reference trajectories,
first-boundary Spearman correlation with eventual terminal score was -0.238
for the head versus +0.524 for the current score. The head's terminal-score
ranking was worse than the current score's at all five nonterminal boundaries.
These are correlated observations from eight trajectories and five distinct
warm starting molecules, not independent tests or calibrated expected values.
A poor single suffix does not bound all possible continuations from its source.

There is also a concrete conditioning mismatch. At boundary one the head was
given a 55-primitive budget, while the five remaining sampled options executed
7–17 primitives. At boundary five it was given 11, while the single remaining
option executed 3–6. The head was trained on best witnessed teacher-path scores
within a primitive horizon, not the expectation under this stochastic option
policy. Its output is therefore not qualified foresight for the tested process.
This mismatch is observed in code/receipts; its separate causal contribution to
the null performance has not been isolated.

Keep the completed SMC result as negative evidence. Do not repair it merely by
turning up guidance or calling the same head a committor. First inspect the
normal-proposal replacement audit. A subsequent continuation estimator should
condition on the actual remaining option decisions and execution policy, with
targets from that policy's continuations. Teacher-path maxima may supervise a
separate proposal skill, but must not masquerade as reference expectations.

Reproduce with `tools/pmo_saved_guide_audit.py RESULT --verified-report
diagnostics/pmo_option_particles/report.json --output
diagnostics/pmo_option_particles/guide_audit.json`, using the pinned chemistry
environment. The supplied `RESULT` path must match the verified report's bound
input identity. An initial mistyped path was rejected before any output existed.
