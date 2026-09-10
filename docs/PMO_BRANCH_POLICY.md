# Bounded deployed complete-edit policy update

Authorization, 2026-09-10: user requests implementing and deploying the learning
intervention now. Cap: 32 new PMO oracle calls on perindopril MPO, no docking,
no reference-model training, and no benchmark expansion. Prior failed predictors
remain failed; this separately authorized preference-policy intervention is not
their deployment or a retroactive change to their admission rule.

Problem/output: learn a state-conditioned stochastic choice among complete
executable molecular edits, then test whether it improves fresh search outcomes
against balanced choice with the same generator. COMPOSE's platform identity and
1..40-heavy-atom, charge-preserving, non-stereochemical support remain unchanged.
This is a task-trained policy over sampled parent/product graph pairs. It is not
a calibrated score regressor, an exact Doob law, or learned primitive foresight.

## Fixed recipe before new outcomes

- Warm start uses the two already-paid 100-query development histories, including
  their exact archived states and original root receipts. Report all 200 historical
  charges, unique labels, and the new-call ledger separately. No winner input or
  prescreen. This cannot be reported as a from-scratch 32-call PMO benchmark.
- Four starting parents are drawn without replacement from the scored canonical
  bank using the existing inverse-rank/20%-uniform parent allocation and a fixed
  seed. Duplicate exact representations do not increase canonical parent mass.
- Calibration: four complete option draws from each of the four parents. Lock all
  candidates and initial predictions before scoring. Score every unique new
  endpoint, at most 16 calls; retain failures and repeats explicitly. No retries
  to find a favorable outcome. Fit once on the historical replay plus these labels.
- Policy: same-parent, score-gap-weighted pairwise logistic loss, equal weight
  per informative parent, fixed regularization 0.01. Include each observed parent
  as a comparison anchor, not an executable self-edit. Fixed kernel averages
  parent similarity, product similarity and normalized molecular feature change;
  fixed molecular features are those in the preceding chronological check.
  No task-reference structure, uncertainty claim or hyperparameter sweep.
- Fresh evaluation: two round-synchronous rounds, four parent bundles per arm,
  four full option draws per bundle. The learned policy chooses a completed
  candidate before its new score; the control chooses from the empirical proposal
  law. Use the existing KL=1 tilt and 10% exploration mixture in the learned arm.
  The KL statement concerns the sampled canonical candidate pool, not the full
  reference path law. Empirical multiplicities are aggregated by canonical result.
- Both arms start from the identical warm archive including calibration results.
  Parent allocation stays the existing inverse-rank/20%-uniform rule. Keep all
  scored children, including worse ones, available for future parent selection.
  Freeze the fitted policy through both evaluation rounds; evaluation labels must
  not enter its training. Each arm makes at most eight evaluation query requests.
  Physical duplicate labels may be reused, but each arm's request ledger records
  them. No cross-arm observation enters the other's archive unless it independently
  selects that molecule. Maximum new physical calls is 16+8+8=32.
  Query allocation excludes endpoints already in that arm's scored archive;
  their exact states remain available as parents. Record these exclusions and
  every failed draw. An empty unqueried pool abstains, without resampling or
  charging an invented observation.
- Every candidate draw retains existing WHERE geometry, applicability-aware WHAT
  and exact complete-option HOW. Generic remains a proposal channel. No new rings,
  support restrictions, QED/SA/similarity filters or primitive-horizon reductions.
  Learned endpoint selection induces task-dependent region/option frequencies;
  the underlying region proposal law is unchanged. Unqueried candidates are
  computational proposals, not fabricated scored observations.

## Measurement and decision

Report both arms' initial/final best and top-ten sum, round curves, selected
options, intended regions, realized structural/topology changes, unique queries,
draw failures, duplicate fractions, proposal and fit time, and all probability
rows. Compare improvement of the actually selected fresh offspring and final
archive utility. One short developmental pair cannot establish IVG superiority,
generalization or an exact control guarantee. No automatic budget extension.

Source and model hashes, durable candidate locks, exact primitive replay and
oracle-start/result receipts are mandatory. Focused tests cover preference
direction, normalization/KL/floor, deterministic fitting and train/evaluation
separation. Run only affected tests before this development deployment; full-suite
verification remains a milestone requirement.

## Execution

One CPU driver, at most eight independent CPU proposal workers (nine concurrent
containers total), each worker one core/8 GiB, no accelerators. At most 80 option
draws before reuse of identical first-round bundles; original pilot option work
was about 18-23 seconds per attempt on average, before the verified enumeration
repair. Expect roughly 8-20 minutes including model/container initialization;
this is an estimate, not a deadline. Worker timeout 1800 seconds, driver timeout
3600 seconds, no automatic retries; conservative reserved bound five CPU-hours,
estimated cost $1-$5. Restart unit is an exact-parent four-draw bundle. Reuse
complete compatible bundles and saved primitive progress; unresolved oracle
attempts stop implicit retry. Heartbeats every 30 seconds.

Prepare inputs locally, batch one clean-source commit at the launch boundary,
run `tools/preflight.py --strict`, deploy `modal_apps/pmo_branch_policy_app.py`,
then durably spawn through `tools/pmo_branch_policy.py`. Never use
`modal run --detach`. No source change during the run and no push without approval.
