# In-loop continuation choice, bounded development

Authorized by the current user request: implement future-aware selection and use
public winner structures/routes as development evidence. This first paired loop
uses actual scored continuations, not a newly trained surrogate or reference law.
It does not change previous frozen experiments or the editing training ladder.

Problem: completed-option endpoint ranking can reject a useful first edit without
considering its descendants. Output: selected complete executable programs and
their newly scored follow-up molecules. Claim under test: downstream outcome
backups change which first edit is selected and improve fresh continuation scores.

Both arms share five exact saved parents (the last current best and all four
original roots), the same frozen COMPOSE generator, and a sampled two-option
tree. Each parent gets four first-option draws; each completed draw gets four
second-option draws. Every novel canonical product is scored and charged. Both
arms see identical first-option products and compute the same lookahead tree;
the immediate arm uses only first-option scores for its decision, while the
future arm backs up the best witnessed score of the first or second option.

The existing FrontierSearch max-backup policy is reused on this *sampled proposal
graph*. Its conditional choice obeys the existing kappa=1 tilt and 0.1 exploration
floor. This is a Monte Carlo planning approximation over sampled complete programs,
not an exact full molecular R_theta row or an exact Doob expectation. Failed draws
remain explicit abstentions. Their zero reward is a declared failed-proposal
return, not a fabricated molecular oracle label.

Each selected first-option endpoint receives four fresh reference continuations.
Those fresh outcomes are the primary paired diagnostic; lookahead scores used to
select an action are not independent validation. Identical selected parents share
computation, while each arm retains its logical candidate/query count. All scores
are development observations; none constitutes held-out task generalization.

Primary report: changed first choices, selected immediate versus backed-up values,
fresh best and mean scores, canonical diversity, option families, actual graph
changes, physical/logical oracle calls and proposal time. Report all five parents,
failures, ties and losses. Public target score 0.80883 is a development reference,
not a promised outcome or a matched published AUC comparison. Winner routes are
not injected as successful sampled branches in this paired generator comparison.
They remain available for separate target-informed controller development.

Support: unchanged 1..40 real atoms, 48 persistent slots, charge-preserving,
achiral production chemistry. Both arms retain the same region geometry,
applicability-aware options, generic channel and ordinary primitive edits.
Every complete program is executor-replayed; no endpoint teleportation.
Endpoint-only PMO scoring, no T4 similarity/QED/SA restrictions.

Maximum 140 new PMO calls: 100 shared planning products plus 40 fresh products.
No docking, prescreen, reference training or model selection. At most 35 four-draw
workers, up to 20 concurrently; each worker has a 20-minute administrative timeout
with no retry. Expected wall time 3..15 minutes using the qualified cached runtime.
All worker results and oracle locks are restart-safe, and progress is published.
Before launch, run focused policy/adapter tests, preflight and deploy clean source.
A full repository suite remains release qualification, not this iteration gate.
