# First resumable-frontier docking comparison

Status: prospective development recipe, recorded before new docking outcomes.
The user explicitly approved the paired 40-call milestone on 2026-09-09 and
authorized up to 20 concurrent Modal containers. The initial two-worker launch
was blocked before spawning; this amendment partitions unchanged work by lineage.
Session: `compose_iclr`. This advances stage 1, not reference-model training.

Current recovery status, recorded before any new docking outcome: source run
`1782604b465e423e3fe9ed6e6f9b5bf41f8ddd4bc74062dc1d86372990e5dafe`
completed all 16 lineage partitions and then failed during deterministic
reduction, before an oracle barrier existed. Every scientific identity matched.
Two snapshot hashes differed only because host BLAS implementations changed the
51 ridge coefficients by at most `3.885780586188048e-16`; their maximum summed
kernel prediction difference is bounded by the coefficient L1 difference. The
recovery contract binds the source launch, failure, and every partition by
SHA-256, requires the source result, oracle barrier, and oracle-attempt artifacts
to be absent, and accepts only authenticated snapshots that agree exactly outside
the coefficients and differ by at most `1e-14` per coefficient. New snapshots
serialize coefficients to 14 decimal places. The recovery performs no molecular
enumeration and preserves the original paths, RNG streams, primitive horizons,
and candidate pool. It will rerun reduction, selected-path replay, and locking in
the pinned RDKit container before any of the authorized calls can occur.

Recovery run
`6742f226f6768273cb1f83f0fb7e58f0a75acfaf1f91ecf3c9ecb365fde181b8`
then authenticated and reduced the reused partitions, but stopped during the
first selected-path audit with zero oracle calls. The verifier had assumed that
one physical primitive mark must yield exactly one augmented option state. A
parameterized construction mark can legitimately match several option-progress
branches even though all remain ordinary executor transitions. In the failing
event the recorded exact augmented product matched one of two branches uniquely.
The repair therefore requires exactly one successor equal to the recorded full
option state and rejects zero matches or duplicate matches; it does not weaken
the executor, option contract, or molecular-product check.

## Question and support

Does short in-loop task guidance improve eligible molecular proposals over the
same executable local/global and option hierarchy with task scoring only after
generation? The output is complete exact-slot molecular trajectories and a locked
batch of new eligible endpoints. This is a controller experiment on COMPOSE's
frozen successor process, not the platform's identity or a universal-reachability
claim. The existing 1..40-atom, charge-preserving, non-stereochemical support,
executor, canonicalization and option registry remain unchanged.

Both arms start from the complete 51-call development archive (best measured
score -9.7). No IVG structures, edit paths, fragments or similarity targets enter
generation or task-value fitting. The same prior-round archive fits the same
round-frozen docking predictor. Its uncertainty is not claimed calibrated.

## Smallest oracle comparison

One round per arm, at most 20 new calls each, at most 40 in total. This is the
first round of the research memo's proposed three-round comparison; additional
rounds are not automatically authorized by this contract. Both arms retain eight
exact lineages and the existing 16-primitive horizon. WHERE remains the frozen
region reference plus its 0.2 scale floor. WHAT retains the applicability-aware
balanced option reference plus its 0.1 floor and permanent generic channel.
Compound options execute only ordinary valid primitives. Kappa stays 1.

- `post_hoc`: reference-only generation, no task-value evaluation or planning
  during generation. The predictor selects completed endpoints for docking.
- `in_loop`: the stage-1 resumable best-witness planner, two planning transitions
  before each committed hierarchical choice, with the same endpoint selector.

The comparison tests the combined guidance/planning mechanism, not either
component independently. It is matched in the maximum oracle allowance and
committed primitive horizon, not internal compute. Planning work and elapsed time
must be reported separately. Do not interpret one unreplicated development round
as evidence of benchmark superiority or generalization.

Prepare each arm to its declared primitive horizon or actual support dead end.
Checkpoint after every completed decision; scheduling quanta do not discard
unfinished programs. Offer completed committed option endpoints, canonically
deduplicated and screened by the unchanged T4 and medicinal-chemistry endpoint
rules. Ineligible intermediates remain legal search states. Allocate at most one
oracle slot per represented bundle in this first comparison, as stage 1 already
does. Underfilled batches are reported, not rescued with extra trajectories.

## Launch and interruption

Sixteen independent CPU preparation containers, one per (arm, parent lineage),
run alongside one driver, below the user's 20-container limit. The full eight-parent
census, lineage-derived RNGs and molecular paths are retained. A deterministic
reduction combines all eight lineages before global selection; missing lineages
cannot be silently dropped. Both audited
locks must exist before either arm docks. The driver records each oracle attempt
before calling QuickVina and publishes each result immediately. An interrupted
attempt with unknown outcome blocks automatic redocking and requires accounting
review. No generator rerun is needed to recover completed preparation.

No executor-count stop or wall-time stopping rule substitutes for the scientific
horizon. A one-hour administrative worker timeout (90 minutes for the driver)
marks incomplete work and preserves checkpoints; it is not a matched-compute
result. The working elapsed-time estimate is 10–30 minutes, not measured yet.
The original serial-within-arm estimate was 30–120 minutes, with substantial
uncertainty: the prior ring round used
10,570 executor applications in 680 s and the stalled task audit used 19,648 in
1,439 s, while full-row planning can exceed both. At sixteen concurrent 1-core/8-GiB
workers plus a driver, the timeout envelope is at most 17.5 container-hours.
Record actual resource rates/cost from the account rather than inventing dollars.
The first real preparation is the reusable timing benchmark, not a throwaway test.

The first partitioned launch stopped before molecular enumeration or docking: one
of 52 old archive rows stored a noncanonical spelling of the same molecule in its
metadata. The source file and exact persistent-slot states remain frozen. On
retry, canonicalize only the metadata string after independently decoding the
exact graph and proving RDKit canonical identity; record the source hash, row,
old string, canonical string, and exact-state hash. Any molecular mismatch still
fails. This is a deterministic metadata migration, not a relaxed identity gate.

Use a clean committed worktree. Run `python3 tools/preflight.py`, deploy
`modal_apps/genmol_t4_opt_app.py`, then spawn with
`python3 tools/t4_launch.py --frontier-compare`. No ephemeral detached apps.

## Acceptance and report

Before any oracle call, verify the exact warm prefix, frozen input hashes and
configuration; replay only committed primitive paths; check full recorded rows,
support, probability normalization, KL/floors, option continuity, canonical
identity, endpoint eligibility and candidate provenance. Verify exact path,
probability, RNG and global-pool parity against serial preparation on a small
production-executor fixture, and fail reduction for duplicate/missing lineages.
Use focused regression
tests for post-hoc isolation, replay, lock-before-docking and interrupted-oracle
accounting. These are not a new broad capability gate.

Report every attempted call, best feasible score versus new and total calls,
selected/represented options and bundles, intended versus realized scale,
cycle-rank and ring-system deltas, canonical diversity, failed/empty outcomes,
generator/executor/value work, proposal seconds and hardware. Persist exact paths,
checkpoints, candidate locks, archives and numerical receipts on the Modal volume;
version a concise interpretation with input/code identities. Preparation can
finish without filling 20 slots; a guidance null or chemistry shortfall is a
valid negative result. No macro weights will be selected from these scores.
