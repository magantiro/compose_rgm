# Task-trained executable proposal, bounded development

User authorization, 2026-09-11: try proposal learning and resume the PMO goal.
This supersedes the no-option-training restriction of the initial macro audit
only for this controller-side experiment. The molecular executor, reference
checkpoint, region law, support, task oracle and benchmark rules remain frozen.

## Question and scope

Can learning from scored complete-option trajectories improve newly generated
offspring, rather than only choosing among endpoints of an unchanged generator?
The output is a stochastic proposal over applicable options and their executable
primitive continuations. This is one experimental controller on COMPOSE's
complete-state editing process, not a new definition of the platform.

Support remains connected 1..40-heavy-atom, charge-preserving molecular graphs
under the frozen vocabulary, executor and canonicalization, without represented
stereochemistry. Existing generic, local, construction, retyping, carbonyl and
region-replacement channels remain available. No ring endpoint catalog, winner
structure input, new primitive, intermediate benchmark filter or reference
training is introduced. WHERE stays the existing local-to-global region law.

## Learning recipe, before fresh outcomes

Reuse all completed, scored options from the prior option-particle, replacement
and replacement-continuation runs. Deduplicate shared worker draws, not merely
canonical endpoints; retain their ancestry and oracle provenance. The initial
roots and earlier development already have winner exposure. These are training
data, not held-out tests. Count their prior oracle charges separately. Do not
train on unscored endpoints, failed options with invented scores, or winner-route
teachers. Report failures and exclusions in the preparation census.

At each recorded WHAT decision, sample four noise alternatives from its saved
historical applicable row; for HOW, sample from the current unguided conditional
proposal using saved exact states and qualified reference rows.
No new neural enumeration or oracle call is needed for preparation. For
HOW, use the same executor and option state, including construction progress.
The observed complete-option return is assigned to its decisions, including
temporarily unattractive intermediates. This is return-weighted policy fitting,
not regression to a fabricated intermediate score or a learned committor.

Fit a fixed hashed-feature linear energy with noise-contrastive, return-weighted
conditional cross-entropy. Each option's weight is exp(10*(endpoint-parent));
first give each exact parent equal exposure by dividing by its observed option
count, then apply the return weights and normalize globally. Do not normalize
away returns separately for each parent, which would erase the task signal for
parents with only one observed option. Split each option's weight between WHAT and HOW,
and divide HOW weight by its number of primitive steps so long programs do not
receive extra training mass. Four proposal-noise alternatives per decision,
4096 fixed signed-hash features, L2 regularization 0.001, at most 200 L-BFGS
iterations, float64 CPU arithmetic, deterministic seed 20260924. Features use
molecular graph environments, their changes, region context and option phase;
no slot number, winner identity, task target or literal-SMILES edit distance.
Earlier traces predate region replacement. The preparation records those
controller-source differences and validates identical executor/chemistry inputs;
it does not assert identical historical option laws. The finite-noise objective is an approximation;
no calibrated uncertainty or
exact future-value claim is made. No hyperparameter sweep on new outcomes.

Prepare immutable training records first and bind all source receipt hashes,
then fit the controller once. Freeze its bytes before the new oracle comparison.
No reference-model parameters change. Training diagnostics are not evidence of
improved optimization. Failure to fit or absent scored executable data aborts.

## In-loop proposal and mathematical boundary

Let B be the existing augmented-state option proposal. For WHAT and each HOW
step, use q = 0.1 B + 0.9 B exp(f)/Z, where f is the fitted energy clipped to
[-sqrt(2), sqrt(2)]. WHAT is normalized on its full applicable row. HOW uses
exact rejection sampling from the existing lazy B row, with acceptance
exp(f-sqrt(2)); no top-k, rejection-count cap or new support mask is introduced.
Every accepted state still passes the same executor. Generic retains positive
probability. Record proposal attempts and acceptance work.

The score range is 2*sqrt(2). The exponential-family identity
KL(B exp(f)/Z || B) = integral_0^1 t Var_{B exp(tf)/Z_t}(f) dt
and Var(f) <= range(f)^2/4 give KL <= 1. Convexity gives the same bound for
the reference mixture. This conservative bound needs no full primitive-row
expansion and does not change kappa. Energy depends on the resulting graph and
declared option context, not mark aliases. B is the existing option-conditioned
augmented process, not the unconditioned raw R_theta kernel.

The learned proposal changes the path reference used by the optimizer. Both
comparison arms use identical actual-score particle selection. There is no
claim that the arms target the same terminal distribution or that this is an
exact Doob transform of the original reference. Correcting the learned proposal
back to B would remove the intended proposal intervention.

## Fresh comparison and compute

Use the same nine starting molecules as replacement continuation, two arms
(unchanged proposal and trained proposal), four complete-option boundaries,
one option per particle per boundary, actual-score SMC in both arms, beta=10,
ESS<N/2 resampling only at nonterminal boundaries, primitive limit 40 per option.
Retain every scored candidate in its arm archive, including worse children.
Lock proposals before the round's oracle calls. Do not use new labels for policy
refitting in this first causal comparison. There are at most 36 query requests
per arm and 72 new physical calls total; independently selected duplicate
endpoints reuse deterministic labels and remain in each arm's logical ledger.
Empty supports abstain; no retries or budget extension. The historical label
bank prevents charging repeated physical evaluations but is not free benchmark
initialization. This warm development result is not matched PMO top-ten AUC.

At most 18 one-core CPU proposal workers and one CPU driver, no GPU. Preparation
reuses saved laws; training uses a bounded CPU fit. Each proposal worker has a
300-second hard timeout, driver 1200 seconds, no automatic retry. Prior proposal
boundaries cost 38..125 seconds; allow approximately 6..15 minutes for this
comparison, with acceptance overhead measured, not assumed free. Reserve at
most 6.34 CPU-hours and $10 for this development loop. Heartbeats every 30 seconds,
durable per-primitive progress and immutable candidate/oracle locks. Deployment
uses preflight, clean committed source, deploy, then durable spawn.

Report best and top-ten archive scores, new-only offspring scores and parent
improvements, logical/physical query counts, selected options, intended versus
realized scale, cycle-rank/ring-system changes, canonical diversity, failures,
fit time, proposal time and rejection overhead. Flat best scores or weaker
offspring are negative results. Do not continue unchanged if learned proposals
have no useful contrast or their runtime dominates. Focused dependency tests
cover training separation, probability/support bounds and actual in-loop use;
full-suite milestone/release qualification remains separate.

Decision: if learned proposals improve the best score over both the starting
archive and the baseline, follow with a matched-seed replication, not a claim of
benchmark success. Better offspring/top-ten scores without a best improvement
support improved local search only. If neither improves, abandon this fitted
proposal recipe rather than extend the unchanged run. If timeout or severe
rejection overhead prevents the comparison, treat efficacy as inconclusive and
repair the measured computational cause before spending more oracle calls.
