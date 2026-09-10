# Winner-blind completed-option beam, 2026-09-09

User authorization: proceed with the proposed small winner-blind multi-option
search after the four-point conditional diagnosis. This is a zero-new-docking
development experiment, not the older unlaunched 40-call frontier comparison.
No reference training, executor change, prior retuning or further experiment is
authorized by this recipe.

COMPOSE remains a frozen executable molecular process. This experiment asks
whether task-informed retention of complete options can chain useful edits more
effectively than the same generator with post-hoc task scoring. Output is exact
primitive trajectories, complete-option candidate locks and a paired diagnostic.
Support stays 1..40 active atoms, 48 persistent slots, charge-preserving achiral
graphs, the existing executor and existing options including generic and the
opt-in carbonyl channels. No IVG structure or path enters generation or fitting.

Four independent CPU workers: two arms, each from the original benchmark seed
and the best eligible measured parent in the same complete 51-call archive.
They consume one byte-identical saved DockingValue snapshot, fit only to that
archive. These roots are chosen by task provenance/observed scores, not winners.
No inference about held-out generalization or actual docking improvement follows.

Each worker makes four completed-option decisions in depth, keeps at most three
distinct molecular frontier states, and draws three (region, option) bundles per
parent. At most 3 + 9 + 9 + 9 = 30 program attempts per worker. WHERE and WHAT
use the existing reference rows and exploration floors without retuning. HOW
samples the full lazy option reference until the registered program completes
or has a real support dead end. No scoring or pruning interrupts a program.
The 44-primitive bookkeeping horizon accommodates four longest current 11-step
programs; the experimental stop is four option decisions, not an arbitrary cut
inside a program. Every primitive is recorded and replayed through the executor.

At each option boundary, canonical-deduplicate complete outputs for retention.
Keep one quality/exploration draw, one maximum-minimum Morgan-distance alternative
and one uniform exploratory draw without replacement. The guided arm uses the
existing kappa=1 tilt of bounded desirability over this finite empirical pool for
the first slot, mixed with a 0.1 uniform floor. The post-hoc arm uses a uniform
first slot and never invokes task scoring during generation. Both use identical
diversity/exploration selection and deterministic tie rules. Ineligible complete
molecules remain available for search; only final oracle eligibility is gated.

This is finite-pool beam optimization, not an exact Doob transform or an unbiased
reference-law estimator. Its retention KL is relative to the empirical candidate
pool, not claimed relative to the full molecular kernel. No uncertainty estimate
is used. Each arm receives the same maximum number of bundles and option depth,
not guaranteed equal primitive work because selected programs differ.

Save every attempt, RNG, exact state, selection row and beam decision; retain dead
ends and duplicates in the ledger. Save the final generation lock before scoring
the post-hoc arm. Completed attempts and levels are reused on resume. Reference
law receipts are reused only after exact input/dependency checks, with fresh rows
cached by exact state. The previous conditional winner locks are not search roots.

Report completed attempts, retained option chains, local/global intended versus
realized change, ring-system and cycle-rank deltas, eligible canonical diversity,
surrogate outcomes, replay status, actual generator/executor work and proposal
time separately from initialization. A failed, empty or uncompetitive search is
a valid result. No docking follows automatically.

Compute: four one-core 8-GiB CPU workers, no GPU, retries=0, 30-second heartbeats,
1800-second administrative timeout each. Expected 5–20 minutes based on the
previous diagnosis's 40 fresh laws and per-worker 36–144 seconds of non-startup
work for four attempts plus known-path inspection. This new larger search is not
yet timed. Timeout envelope: two configured CPU-hours and sixteen GiB-hours,
excluding build/startup. Dollar rates are unverified. Timeout means incomplete,
not a molecular support failure. Do not automatically increase this envelope.

Narrow launch checks cover target-free sampling, completed-option-only retention,
post-hoc isolation, exact replay, resume determinism and the four-worker launch.
Use clean committed source, strict preflight, deploy genmol_t4_opt_app.py, then
`python tools/t4_launch.py --macro-beam`. Full-suite verification remains a
milestone requirement, not a gate for this bounded development run.
