# Persistent archive controller: smallest paired development loop

## Scope and claim

Authorized by the user on 2026-09-10: implement the proposed persistent optimizer
and compare two arms on perindopril MPO, one paired random seed, 100 unique oracle
queries per arm, including the four initial sources. This is a bounded development
pilot, not reference-model training, an official PMO result, or an ICLR milestone.
It does not reopen the editing training ladder or modify the completed beam probe.

Problem: allocate costly task observations over a known executable molecular edit
process. Output: a scored canonical archive plus exact persistent states and
primitive witnesses. Claim under test: online region/option allocation and delayed
credit can improve the short-prefix top-ten curve beyond persistent selection of
good parents alone. Both arms use the same frozen reference model, complete-option
executor, objective-blind roots, balanced applicability-aware reference policies,
and oracle. There is no prescreen, winner input, docking guide, or learned surrogate.

Support remains complete connected 1..40-heavy-atom graphs under the existing
element/valence/charge rules, without a stereochemical claim. Existing generic,
ordinary macro and parameterized pendant/fused-ring options remain available.
The option menu is unchanged from the previous probe and is not the generator's
full molecular support. No new ring templates or executor operators are introduced.
Region enumeration retains its existing 1..24-atom bounds and exploration floor.

## Mathematical problem versus the tested approximation

Let S contain the exact archived molecular states, canonical oracle ledger,
ancestry, option progress, policy statistics, and remaining query/compute budgets.
A decision a=(parent, region, option) induces a distribution over valid primitive
paths and their completed endpoints under the frozen option kernel K. Durations
and computation differ across options, so this is an archive-valued semi-Markov
control problem, not the finite-horizon Doob problem for one molecular trajectory.

For this pilot, U(A) is the sum of the ten largest observed scores divided by ten,
with absent slots filled by zero. The declared objective is

    maximize E_pi[ (1/B) sum_{t=1}^B U(A_t) ],  B=100.

Every previously unseen canonical endpoint consumes one query; repeats use this
case's charged ledger. All completed endpoints are scored immediately, including
unimproved endpoints. Endpoint feasibility is the unmodified PMO objective; no T4
similarity, QED, SA, or medicinal-chemistry filter is added. Exact oracle inputs are
locked before evaluation. Proposal failures consume computation, not fabricated
oracle labels. An operational timeout is reported as incomplete.

At the full state level, a budgeted Bellman equation would value each decision by
immediate archive reward plus the value of its resulting archive and remaining
resources. We do not evaluate that equation exactly or claim that the pilot's
statistics consistently estimate it. The tested approximation is deliberately
small and auditable:

1. Both arms choose canonical parents by inverse score rank, mixed with a 0.2
   uniform floor. Equal scores receive equal ranks. Exact-state variants are drawn
   uniformly *within* a canonical identity; extra aliases do not buy parent mass.
   All discovered endpoints remain available, including worse intermediates.
2. The balanced arm samples WHERE and WHAT from the existing reference rows.
3. The adaptive arm tilts those same rows using observed archive-improvement credit.
   Let r_j=10[U(A_j)-U(A_{j-1})]. On a newly scored discovery j, credit r_j to its
   creating edge and 0.9^d r_j to each ancestor edge d options back. Failures and
   canonical repeats add no discovery reward. An edge's accumulated credit G_e is
   an attributed return from the realized discovery forest, not an unbiased
   counterfactual action value or a calibrated uncertainty estimate.
4. Pool G_e by region scale (release <=0.2, <=0.5, >0.5) for WHERE, and option name
   for WHAT. Each attempted decision is one observation; descendant credit updates
   that observation instead of inventing extra trials. Estimate each cell's mean
   with two pooled-mean pseudocounts. Unknown cells receive the pooled mean.
5. For either row with reference probabilities mu, use h=exp(estimated credit).
   The existing tilt gives p_eta proportional to mu*h^eta, with KL(p_eta||mu)<=1
   and eta<=64. Sample q=0.9*p_eta+0.1*mu. Baseline has eta=0. This is a valid
   support-preserving, entropy-regularized allocation rule, not an exact Doob law.
   Convexity gives KL(q||mu)<=0.9 KL(p_eta||mu), and q>=0.1 mu. The cap and sparse
   empirical feedback make it an approximation, not an optimality guarantee.

The existing Q(M) reference geometry is retained; the PMO arm explicitly tests a
new task-dependent tilt on it. No QED value head is repurposed. HOW is identical in
both arms and samples the frozen complete-option kernel without additional task
guidance. Option decisions are augmented-state decisions; we do not call their
mark-level implementation representation-invariant canonical molecular control.

The campaign has no three-option or 110-primitive ancestry cutoff. Each new option
receives its registered local execution horizon (bookkeeping capacity 11), while
the exact graph, atom lineage, ancestry and cumulative primitive cost are retained.
Renewing this local horizon does not replenish the 100-query campaign budget.
Unfinished programs preserve their exact progress and RNG; they are never restarted
from canonical SMILES. Generic remains available.

## Efficient execution and evidence

Reuse compatible saved deterministic reference laws from the completed probe,
verified by their model/executor dependency closure. Never import its reward labels
or candidates as free initialization. Keep all raw candidate locks and oracle
attempt receipts. Buffer ordinary volume commits, retaining a <=30-second progress
checkpoint; force durability before each oracle call and after its result. Resume
reconstructs policy updates from locked attempt records and charged labels, not a
different random stream. An unresolved oracle attempt blocks implicit retry.

Profile actual proposal work and persistence before estimating the remaining run.
Log query count, current score, best, top-ten mean, option, ancestry depth, active
primitive, proposal time, exact-law counts, executor counts and persistence cost.
The first four proposal attempts are profiled within the real run, not a discarded
canary. Two independent CPU containers are sufficient for the paired arms. Record
separate initialization and proposal times. A 300-attempt / 75-minute operational
ceiling prevents unbounded duplicate or dead-end work; failure to reach 100 queries
is incomplete, not an equal-budget optimization result.

Report the paired 100-query top-ten curves (zero-padded metric above), final best,
parent/region/option allocation, direct and descendant credit, number/depth of
reused molecules, realized structural change and topology, diversity, and compute.
One pair can diagnose feedback and throughput, not establish superiority. These
tasks and sources have development exposure. Any IVG 10,000-query comparison needs
its own matched initialization, evaluator, accounting and repeated-run protocol.

Focused checks cover probability normalization/KL/floors, delayed-credit updates,
canonical accounting, exact-state reuse and interrupted resume. No unrelated full
suite is an iteration gate for this bounded user-authorized pilot. Full milestone
verification remains required before a release or scientific completion claim.

Launch: strict preflight, deploy `modal_apps/pmo_archive_pilot_app.py`, then durable
spawn via `tools/pmo_archive.py`. No `modal run --detach`. Commits are not pushed
without authorization. No scientific result exists when this plan is written.
