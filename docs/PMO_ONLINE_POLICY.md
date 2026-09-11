# Repeated policy-feedback development intervention

User authorization, 2026-09-10/11: resolve the inference discrepancy and improve
the controller empirically and principledly while the user is away for an hour.
This is a new bounded development run, not a continuation or amendment of the
completed frozen-policy evaluation. No reference training or docking is permitted.

Problem/output: improve stochastic choice among complete executable molecular
edits using newly observed task outcomes. The reference process, declared
1..40-heavy-atom charge-preserving support, generic channel, region law and option
executor stay fixed. Executability is pathwise; benchmark properties are not
intermediate-state masks. No winner structures, prescreening or invented labels.

The current policy has a precise limited interpretation: contextual, pairwise
preference learning over sampled complete-option endpoints, followed by a KL=1
tilt with the existing 10% exploration mixture on the canonical candidate pool.
It is not a learned Doob value or a multi-option planning algorithm. Repeated
feedback is necessary to test policy improvement, but not sufficient to prove
long-horizon discovery. We do not promise that this intervention will close the
IVG gap. Its result determines whether to retain or reject this learning layer.

## Frozen comparison

- Start all arms from the same archive containing all 228 previously charged
  development calls. Merge exact duplicate observations with consistent labels;
  record the resulting unique-label count. No prior evaluation is now described
  as a held-out final benchmark. New outcomes are chronological development data.
- Arms: balanced proposal allocation, frozen learned endpoint allocation, and
  learned allocation refit after each complete round from that arm's own labels.
  The frozen/online comparison isolates updating; the balanced comparison checks
  whether either learned policy is actually useful. No cross-arm training labels.
- Eight rounds, four parent bundles per arm per round, four complete option draws
  per parent. Share exactly identical proposal tasks before arms diverge. Preserve
  inverse-rank/20%-uniform parent selection, canonical deduplication, full options
  and the existing pairwise policy recipe. No optimization-weight sweep.
- Lock every round's selections before any new score in that round. Retain worse
  scored children in the archive. Fit the online policy only after the round has
  completed, for use in the next round. No policy updates inside a round.
- Maximum 96 new PMO oracle calls on `perindopril_mpo`, 32 logical query requests
  per arm. Independently selected duplicate endpoints reuse physical labels but
  remain charged as logical requests to each arm. Empty pools abstain, no retries
  or budget extension. Count all 228 historical charges separately.
- Report best-so-far, top-ten mean/area under the request curve, selected scores,
  parent deltas, option frequencies, realized topology/scale, duplicate/failure
  fractions, policy hashes, fit times, query ledgers and proposal time. A flat
  curve or no advantage is a negative result, not a reason to tune on this run.

## Inference admission

The prior production parent's saved full law exactly matches the accelerated
probe digest. The differing export-worker digest therefore is not evidence of
a cache-induced difference; its numerical cause remains unknown. Reuse the
authenticated model package and passing three-state baseline/cache receipt.
Each fresh runtime compares its complete parent law with the existing production
record before any search: action records must be identical, probabilities finite,
maximum absolute error <=1e-7 and marked-measure L1/2 <=1e-6. These prospective
float32 portability limits are separate from the unchanged, already-passed exact
within-worker cache gate. Record raw values and exact/numerical comparison results;
fail before oracle work if either limit fails. Do not silently reorder or round
records, or claim cross-worker bitwise identity if only numerical agreement holds.

## Compute and reuse

One CPU driver and at most twelve one-CPU proposal workers (13 containers total),
8 GiB per worker, no GPUs. Worker limit 1200 seconds; driver 2700 seconds. The
driver's 45-minute timeout is a hard stop, no automatic retry. Up to 384 complete
draws before identical-task reuse; expected wall time 10-25 minutes based on the
measured 2.08-2.19x enumeration speedup and earlier proposal work, not guaranteed.
Conservative $5 spending bound for this development cycle. Reuse each completed
exact-parent task and primitive progress. Heartbeats every 30 seconds. Source,
weights, archive and score locks are bound before deployment. Unresolved oracle
starts forbid implicit retries. Focused dependency tests only during iteration;
full-suite release qualification remains separate.

## Comparator boundary

IVG's released no-prescreen PMO recipe uses repeated population search and PPO
updates with replay under 10,000 oracle calls. Its reported perindopril number is
top-ten AUC, not a single best score. Our short warm-start probes do not measure
the same statistic or budget. Source: the authors' released repository,
https://github.com/invirtuolabs/InVirtuoGen_results (read 2026-09-11).
This comparison motivates testing repeated learning; it does not establish that
our model, chemistry support or controller is equivalent to theirs.
