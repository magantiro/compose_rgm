# Bounded complete-option particle development

Status: proposed and locked before this experiment's outcomes, 2026-09-11.
Authorized by the continuing PMO controller goal and the explicit source-hashed
development snapshot approval. This is a new controller development arm, not a
change to the qualified region controller or a release/held-out benchmark.

## Question and smallest useful comparison

Does retaining and resampling complete-option trajectories improve autonomous
search from the five existing development starts? Compare eight particles per
arm, six completed-option opportunities, with identical reference proposal code:

- reference: independent trajectories, no intermediate selection;
- immediate: particle resampling with measured current-score potentials;
- future: particle resampling with the existing achieved-return model.

The initial empirical population cycles over the same five exact stored starts
in recorded order to fill eight slots. It is a warm development initialization,
not from-scratch PMO. One of those starts is the previously found incumbent.
No target molecule, target similarity, reconstructed winning edge, or route is
supplied to proposal selection. The future model was previously fitted using
answer-known development routes, so this is explicitly winner-informed development.
Do not call that head a calibrated committor or the test blind generalization.
No training occurs in this experiment. All old labels and checkpoint bytes are
reused with hashes; historical labels are not magically free benchmark calls.

## Reference and control law

B is the existing WHERE/WHAT/HOW option-augmented reference. WHERE is unchanged,
WHAT remains balanced/applicability-aware with generic active, and HOW uses the
existing lazy reference sampler. Each draw completes one existing option (up to
11 primitive steps), with every step executed and replayed on the frozen support.
Six option opportunities therefore allow up to 66 primitive steps. This is not
beam search, top-k enumeration, a ring-template catalog, or an endpoint teleport.
The chosen macro law is not identical to raw R_theta despite frozen weights.

For live states set terminal g=exp(10*score). At intermediate boundaries use
psi=1, exp(10*score), or exp(10*predicted_achieved_return), respectively. A score
difference of 0.1 thus means one log-weight unit; this single scale is declared
before new results and is not fitted to the winner. Initial psi=1 for all roots.
The future model receives (remaining option opportunities)*11, in 0..55 at the
first and subsequent updates, as a heuristic remaining primitive allowance.
It estimates neither an exact option-horizon value nor E_B[g].

With q=B, update log w += log psi_next - log psi_previous. Preserve weights
between boundaries; systematically resample only if ESS<N/2 and nonterminal.
After resampling reset weights uniformly and inherit the selected state's psi.
The exact terminal potential replaces every intermediate twist. A finite particle
system approximates the B path law tilted by g, and does not guarantee discovery.
The reference arm uses only terminal importance weights for its terminal draw.
All arms also report the full actually queried archive, a distinct search output.

Failed programs enter a cemetery state (zero continuation mass); an incomplete
intermediate is never silently returned as a completed option. Dead particles
may be replaced through ordinary SMC resampling, never by an unrecorded retry.
Extinction is reported. Duplicated particles retain multiplicity and independent
subsequent randomness. Exact shared tasks across arms can reuse computation, not
gain additional probability. Same slot/step seeds couple the arms where possible.

This new SMC path-selection law has no asserted kappa=1 effective-path bound.
It does not modify kappa or claim parity with the frozen KL-controlled process.
All scientific support, R_theta, canonicalization and task thresholds stay fixed.
PMO has no T4 QED/SA/seed-similarity filter on intermediate or returned molecules.

## Accounting, execution, and decision

At most 144 new oracle calls and 144 single-option worker tasks (8*6*3).
Shared tasks/identities and historical deterministic scores are reused. Lock the
entire completed-option batch before scoring; two durability barriers per batch,
not per molecule. Scores cannot alter proposals from the same boundary. Record
every queried/known candidate in each arm's archive, including unselected ones.
No root screening is performed, but prior selection/training exposure is explicit.

At most 24 concurrent worker containers plus one driver (25 total, below the
authorized 30); one CPU and 8 GiB per container; no retries. Each worker has a
180-second limit, the driver 900 seconds. Timeout is an operational failure, not
a chemistry rejection. Complete task receipts are restart units; checkpoints at
each boundary bind particles, weights, potentials, archives, and deterministic
seed derivation. Oracle attempts are durably reserved; ambiguous attempts abort.
Representative prior one-option times were 22-29 seconds. Six parallel waves
are expected to take about 4-10 minutes after deployment; this is an estimate,
not a promise. The configured reservation ceiling is 7.45 CPU-hours (144*180s
plus 900s); expected usage is much lower. Estimated cost is $1-8, not a billing
guarantee. Heartbeats every 30 seconds and per-worker completion logs are required.

Report per-boundary best and top-ten queried scores, diversity, unique exact and
canonical states, ESS/ancestry, selected option counts, intended release versus
realized change, ring-system/cycle-rank deltas, proposal/law/I/O/oracle times and
actual calls. Also report the terminal weighted draw separately. Do not compare
this short warm-start diagnostic's top-ten score with IVG's 10k-query AUC.

Retain nulls and failures. Improvement in scores is required to claim an improved
optimizer; changed particle ancestry or ring formation alone is not sufficient.
If the future arm fails but immediate SMC improves, retain that result and repair
the value model separately. If neither improves, use recorded proposal families
and trajectory survival to identify the next specific bottleneck. No automatic
full benchmark or additional training follows from a successful smoke result.
