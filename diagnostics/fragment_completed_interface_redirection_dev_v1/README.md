# Completed-interface redirection development pilot

This is a failed development gate, not an official fragment benchmark result or
a promoted sampler. The pre-outcome decision rule is in
`docs/FRAGMENT_COMPLETED_INTERFACE_REDIRECTION_DEV_2026-09-24.md`. The frozen
matched run used four development scaffold-decoration prompts, 20 attempts per
prompt per arm, and two negative-control prompts. Inputs, seeds, software,
evaluator hashes, exact accepted actions, all attempt outcomes, and elapsed time
are in `summary.json` and the 12 `units/*.json` files.

The proposed rule was accepted only 4 times, below the required 5. Staging
refusals fell from 491 to 450 (8.35%), below the required 10%. Full task success
was 77/80 in both arms. Every one of the 80 committed endpoints in each arm was
chemically valid and fragment-preserving; negative controls were attempt-identical.
Mean within-prompt diversity was 0.6160 vs 0.6180. The mechanism fired only
on MARIBAVIR, so these data do not support a general attachment-control fix.
The candidate must not be promoted or selected on held prompts.

`replay_audit.json` independently replayed all 240 attempts, including both
negative controls, and verified all 3,032 accepted executor steps against the
exact graph lock. It also corrected an interpretation error without changing
the original result: the contract allowed a **five-percentage-point** quality
loss, but the original runner checked a loss of 0.05 on a 0–100 scale. Both
versions pass on these data; the overall development gate still fails.

The audit additionally separates two evaluand populations using the pinned
InVirtuoGen metric function. `official_metrics_on_all_committed` scores every
chemically valid endpoint, even if it misses a declared attachment site;
`official_metrics_on_constraint_emitted` scores the same attempts after the
current harness replaces task failures with an invalid placeholder. On the
candidate BARICITINIB decoration prompt, these give validity 100% vs 95% and
quality 65% vs 60%; on MARIBAVIR they give validity 100% vs 90% and quality
25% vs 20%. The latter is an explicitly stricter task-success composite, not
the published baseline's plain chemical-validity metric. Report both, with
their denominators, rather than calling one the other.

Next safe action: use the saved failure and action traces to localize why the
other three decoration prompts never offered this move, then predeclare a new
generic, attachment-aware binding hypothesis. Do not tune this failed pilot
on held prompts. This 20-attempt-per-prompt pilot cannot establish a paper
comparison to GenMol or InVirtuoGen.
