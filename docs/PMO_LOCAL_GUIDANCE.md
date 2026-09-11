# Broad population search with a tested local selection channel

## Hypothesis and matched comparison

The endpoint selector passed two locked tests: within one neighborhood and then
on two different parents, finding more improving offspring than uniform queries.
Neither test raised the overall champion. Test whether selectively allocating
queries to these local moves helps the complete broad controller improve through
multiple rounds. This is not a future-value model, twisted SMC or exact Doob law.

Baseline is the strongest retained fixed-memory broad controller: 50% donor
transplants and 50% reference options, all existing chemistry and region
replacement enabled, 64-primitive maximum per option, sixteen archive-selected
parents and 0.2 parent exploration. Use the same 116 fixed donors as before.

The intervention replaces half of reference-channel draws with a local selector:
50% donor, 25% original reference and 25% local endpoint selection. All channels
remain proposal mechanisms through the same executor. Generic and multi-edit
reference options remain positive; no finite ring catalog or executor support
change is introduced. This explicitly changes the proposal mixture, not the
frozen R_theta parameters or primitive law. The existing region/option machinery
inside the unchanged reference channel is untouched.

For a local draw, enumerate all positive-mass primitive successors, canonical
deduplicate, and exclude previously paid initial labels plus the arm's own
requested archive. Predict every remaining successor with the exact 1044-label
model tested in the component experiments. Draw uniformly from its top sixteen
(or all if fewer remain), breaking prediction ties by canonical identity. This
is stochastic query allocation, not a claim of sampling the reference law. Empty
novel support is a recorded abstention, not an invented replacement. Do not refit
or update coefficients during the experiment. This bounded test explores broader
model applicability; it does not assume calibration beyond the tested chemistry.

The original donor/reference coin uses SeedSequence([20261008, round, slot, 776]).
Only a reference outcome draws a second coin, and values below 0.5 activate the
local channel in the intervention arm. Other RNG streams and baseline semantics
are unchanged. Local selection uses SeedSequence([20261008, round, slot, 777]).
Common parent/channel tasks are shared only when their full laws are identical.

## Initialization, budget and decision

Both arms start at the same sixteen highest-scoring distinct exact states in
completed own-controller and component records. Include the observed 0.6747477698
champion. Sort by score then canonical identity, retaining exact persistent-slot
witnesses and ancestor chains. No public winning endpoint is an input. Reuse
all compatible scored history as requested-only oracle cache, not as uncharged
training or generated results. The model was already trained on 1044 labels;
total prior costs are 249455 prescreen plus 2056 development physical calls.

Four synchronous rounds, sixteen attempts per arm per round, at most 128 new
canonical oracle calls across both arms. Score the globally locked union, then
update each arm's own archive; no within-round oracle feedback. Report archive
best, top-ten mean, unique improvements, structural-change and primitive-family
records, proposal time, neural work, physical queries and failures.

Positive requires the intervention's best to exceed both its initial champion
and the simultaneous baseline's best. Then earn one unchanged fresh-seed
replication from identical starts and model, with new labels used only as cache.
Null/reversal stops this unchanged mixture; distinguish absent useful local
products, endpoint prediction errors and failed global proposals using saved
candidate records, without another uninformative scaled run. Neither a component
pass nor one warm comparison establishes external PMO AUC superiority.

## Resource and execution contract

At most 29 CPU workers plus one driver, no GPU, one numeric thread, float32
frozen neural scoring and float64 endpoint scoring. Four rounds; at most 128
workers, 180 seconds per worker, 900 seconds driver, zero automatic retries.
Representative local cache census plus prediction took 24-26 seconds per parent;
prior broad replication took 399 seconds for six rounds. Expected 4-12 minutes,
$10 reserved cap. Worker/round checkpoints, 30-second heartbeats, exact input,
source and model identities, candidate locks and oracle ledger are mandatory.
Clean committed source, preflight, deploy then durable spawn. Reuse prior neural
law caches after the existing dependency-containment check. Preserve all tested
chemistry, negative outcomes and generated products in durable artifacts.
