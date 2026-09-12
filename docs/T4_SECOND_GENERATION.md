# Measured second-generation program development

## Problem and authorized implementation

Use the four eight-endpoint measured archives from the completed curriculum to
produce useful descendants, not restart from the generic unscored library.
Primary output: a new complete program, exact execution trace and unscored
eligible endpoint. The hypothesis is that allocation using measured target
history improves the next generation's docking outcomes. This local preparation
cannot establish that outcome without subsequent charged measurements.

Retain the same T4 cells, original-seed endpoint constraints, frozen reference,
valid-state executor and maximum 40 heavy atoms. Use the same 70/20/10
mutation/recombination/broad interface, 20% endpoint exploration floor,
32-primitive/eight-block cap, 128 attempts and 60-second limit per arm/cell.
Stop at eight new eligible endpoints. No docking, new labels or model fitting.
The local mode continues to record unavailable broad draws explicitly; it is
not a complete remote mixed-controller benchmark.

## Changes and comparison

Attach whole-program source, peak and final atom counts and net change. For
constructor mutation, distinguish the original construction source from the
measured parent endpoint. Reuse the existing deterministic dependency/resource
scheduler and exact executor. A 39 -> 35 -> 40 program remains allowed; an
overshooting intermediate does not become legal because its endpoint fits.
These are structural accounting features, not a reward for shrinking.

The primary arm samples measured endpoints by inverse score rank, with the
existing mutation operators and bounded duplicate-exhaustion adjustment. Preserve
at least the declared uniform endpoint exploration mass after exhaustion is
applied. The score-blind control starts with the same measured programs but uses
equal quality weights before the identical exhaustion/exploration rule.

Do not use the legacy `adaptive=False` switch as this initial feedback ablation:
its initial static scores equal the measured scores, so it initially ranks the
archive identically. That switch remains available for a later online versus
frozen-preference experiment.

Each arm is an explicitly new experiment derived from the same hash-bound archive,
not a silent configuration change during resume. Seed 20260914 is shared across
arms and the four cells; each cell has its own oracle identity and RNG state.
No target-specific structural rules or target-specific tuning.

## Acceptance and next decision

Focused checks must protect peak versus net size, unchanged primitive support,
identity-bound archive reuse, exploration after duplicate exhaustion, score-blind
allocation, and exact snapshot/resume. Record every proposal and candidate,
remaining failures, parent-score allocation, actual size changes, executor calls
and proposal time. Store complete input/implementation hashes and both locked
arms before any second-generation docking.

Useful preparation means the scored archive produces new eligible candidates
within the common work cap and measured ranking actually changes allocation.
It does not mean those candidates dock better. If the neighborhood is exhausted
or a structural failure dominates, repair that shared mechanism before asking for
a larger scored campaign. Do not scale an unchanged empty proposal stream.

Proposed paid comparison, not yet authorized: score up to eight locked candidates
per arm/cell (64 maximum before cross-arm reuse), with a separately declared fresh
confirmation reserve. Compare improvement over each first-generation incumbent,
not merely over the original seed. Positive outcomes favor measured adaptation;
no improvement or a score-blind advantage is a negative controller result.

## First preparation and shared repair

The first attempt retained 39 cross-arm distinct candidates with zero new oracle
calls. Score-ranked pools filled 8/3/8/6 slots on JAK2/FA7/BRAF/5HT1B; score-blind
pools filled 8/6/8/8. Preparing all eight arm/cell batches took 67.06 seconds.
Measured ranking changed parent allocation, but did not improve proposal yield.
No docking outcome is inferred from that finding.

The attempt ledger shows a repeated avoidable failure: uniform mutation-kind
selection chooses contraction even when the complete program has no contractible
created segment. In ranked 5HT1B this caused 29 of 128 rejected attempts.
Repair the same mutation decoder for every target by normalizing over its
currently available conditional choices, including context-bound attachment
alternatives. This changes the optimizer proposal, not R_theta or executor
semantics. Every proposed transformation still needs complete exact replay.
Retain both preparations and repeat the same bounded zero-oracle comparison;
do not enlarge its work limit or change the starting archives.

## Locked second-generation pools

The repaired preparation is `diagnostics/t4_second_generation/attempt_2`.
Score-ranked arms filled 8/7/8/8 candidate slots; score-blind arms filled all
eight on each cell. The 63 arm/candidate placements contain 49 distinct
target/molecule queries, so cross-arm reuse saves 14 physical first evaluations.
FA7's unfilled ranked slot remains a yield failure, not a reason to extend its
128-attempt budget. No second-generation docking scores exist yet.

Preparation took 81.85 seconds, 9,534 executor calls, approximately 473 MB peak
process RSS and zero oracle calls. The conditioning repair requires more
attachment-census work, but removes empty conditional-mutation failures.
The first attempt is preserved; fewer executor calls alone are not a claim of
lower total runtime (the first attempt took 67.06 seconds).

Measured ranking raises the initial probability assigned to each cell's best
observed endpoint from 12.5% in the score-blind arm to approximately 30–32%.
This is a change in proposal allocation, not evidence that its descendants
score better. Both arms use identical measured archives, mutation support,
size accounting, endpoint gates and exploration/exhaustion rules. Completed
mutations include growth, contraction and size-neutral changes. No shrinkage
reward was added.

### Scoring allocation, approved 2026-09-12

**Update, 2026-09-12:** the user approved proceeding after clarification of
actual prior timings and cost. The allocation below is now authorized, at most
73 new calls and $20 reserved, not $20 expected spend. The prior 36-call round
completed in 61.16 driver seconds with 3.88 seconds per molecule on average.
For this round, expect a few minutes after launch at similar throughput; image
deployment, startup and variable preparation times are additional. CPU/RAM
execution is expected to be cents-scale at current published Modal rates, not
a provider billing receipt. Eight workers plus one driver, no GPU or retries.
The new allocation is separate from the old two-call remainder.

Use the already locked attempt_2 pools without regeneration or surrogate
screening. Score all 49 distinct target/molecule pairs once, sharing the first
physical receipt between arms only where both requested that endpoint before
scoring. Preserve each arm's logical call ledger, duplicate reuse and the FA7
shortfall. Each arm receives only labels for its own locked candidates.

Then nominate each arm/cell champion from those first scores and evaluate it
twice freshly. Repeat each of the four first-generation incumbents twice as
well. The maximum is 49 first evaluations plus 16 champion confirmations plus
eight incumbent confirmations, or **73 new docking calls**. Reuse repeat
receipts for identical arm champions rather than performing duplicate physical
evaluations. Compare fresh repeats separately from selection scores.

The approved new allocation has a $20 reserved ceiling, eight single-CPU
workers plus one driver, zero automatic oracle retries and unchanged
receptor/preparation/search settings. It is separate from the old allocation's
two remaining calls. No further paid round is automatic.

The engineering promotion question is whether feedback yields improvements
over the corresponding first-generation incumbents in more than one cell,
with fresh-repeat support, and whether it improves over score-blind selection
under the same proposal limits. A single improved endpoint does not establish
an aggregate benchmark win. Negative results remain part of the controller
decision ledger.

The exact scoring manifest is `configs/t4_second_generation_lock.json`.
The two-stage runner uses the unchanged production docking function, commits
every started query and result, and rejects ambiguous retries. Four focused
tests cover arm-specific champion selection, shared confirmations, failure and
budget accounting, exact trace replay, completed reuse and interrupted-start
rejection. One test fixture needed JSON normalization to match the persisted
production boundary; no production gate was weakened. Ruff and whitespace
checks pass. A full-suite milestone sign-off is not claimed.

From a clean committed source, deploy `modal_apps/t4_second_generation_app.py`
and run `PYTHONPATH=src:. python tools/t4_second_generation.py launch`, writing
the launch receipt outside the clean source checkout. Then retrieve the sealed
remote result and run `tools/t4_second_generation.py review --folder <receipts>`
in the pinned chemistry environment to update each arm's own scored archive.

### Verification and reproduction

The paid round is complete. See
`diagnostics/t4_second_generation/scoring_1/README.md` and its sealed
`review.json`. It used 69 calls, including all 49 first queries and 20 fresh
confirmations after sharing two identical arm champions. Parent ranking did
not earn promotion: it tied the control on JAK2/5HT1B, slightly improved FA7,
and lost BRAF. The new 5HT1B endpoint is unstable across docking seeds; retain
the existing incumbent. No additional paid generation is authorized.

Nineteen focused optimizer/mutation/scheduler tests passed, plus the added
conditional-choice regression and two attachment-transfer tests passed separately.
Tests cover shrink-then-regrow,
illegal intermediate peaks, explicit score-blind allocation, the exploration
floor after exhaustion, broad-channel preservation and exact snapshots.
Ruff checks and whitespace checks passed. No repository-wide suite was used as
this bounded preparation's iteration gate; the full milestone is not complete.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 python tools/prepare_t4_second_generation.py \
  --output diagnostics/t4_second_generation/new_preparation
```

Use the pinned RDKit 2024.03.5 environment. The script refuses to overwrite
existing candidate locks. Input receipts, archive identities, implementation
hashes, configuration, seeds, hardware, cost counters and pending snapshots
are retained. No remote service is called.
