# One feedback round after the partial docking diagnostic

User authorization, 2026-09-08: proceed with the proposed single bounded
twenty-call feedback round from all 33 accumulated evaluations. No automatic
next round, training, macro-weight tuning, or additional oracle budget.

## Question and frozen identity

COMPOSE produces complete supported molecular graphs through an executable
marked rewrite process. This inspected PARP1 seed0 d=0.4 development experiment
asks whether objective feedback can accumulate useful edits across generations.
Its baseline is the full previous 33-call history, with best score -9.3. A
matched same-generator reference continuation remains necessary for any causal
guidance claim, and is not authorized here. No IVG molecule, fragment, score,
or distance enters proposal or candidate selection.

Retain connected broad-organic, charge-preserving, non-stereochemical support,
40 active atoms, and the same executor. Keep Q(M), the balanced untrained Q(o),
generic, R_theta, committor, kappa=1, horizons, exploration, eight lineages,
three regions per lineage, and one particle unchanged. Use existing
feasible-then-docking parent selection and bootstrap-ridge candidate ranking.
The surrogate is an inherited heuristic, not calibrated uncertainty evidence.
Only the frozen QED >= 0.6, SA <= 4, original-seed similarity >= 0.4 endpoints
can be docked. No additional topology blacklist or intermediate-state screen.

## Evidence reuse and RNG boundary

Bind the physical hashes of the saved 20-call exact archive, partial candidate
lock, and 13-call docking receipt. Import every evaluated endpoint, including
failures if present, with its saved exact state, score, and ancestry. Do not
rebuild an executable state from SMILES. The archive contains 34 records with
the unevaluated original seed. Its second event is explicitly the partial
seven-parent diagnostic, not a completed optimizer round.

This is a new feedback episode, not a bit-identical resume of the failed round.
Initialize its RNG from the first 64 bits of the SHA-256 of canonical JSON
containing the three source identities and seed_rng=1000. No seed search or
score-dependent seed choice. Retain this derivation in the exact archive.
The new proposal is archive event/round 3. Previously evaluated molecules never
receive another call. Completed source parents and the failed eighth parent
are not regenerated: their scored outputs already enter the new archive.

## Compute, accounting, and acceptance

One CPU, 8 GiB, no GPU, one-hour whole-job limit, zero retries. The original
20,000 total public-executor ceiling remains. Allocation within that ceiling
is pending the explicit user choice; the draft contract is deliberately
unsealed and cannot launch until that choice is recorded.

The prior attempt spent 589.567 seconds on preparation before exhausting
20,000 calls, with seven parent units complete. Its ledger contains 18,186
distinct exact source/action pairs, so simple duplicate elimination accounts
for only 1,814/20,000 calls in a provisional read-only count (not yet a sealed
performance benchmark). This is not evidence that caching alone can make
the eighth parent finish. Expect minutes for the new round, not guaranteed
completion. Retain every stopped or invalid outcome and completed restart unit.

Dock at most 20 new canonical molecules, in one locked batch. A short eligible
batch is allowed, never backfill with infeasible chemistry. Use the same
unseeded QuickVina settings. Serial docking reuses the existing runner; the
previous 20 serial calls took 47.439 seconds, so generation is the load-bearing
bottleneck. Max cumulative calls are 53, leaving at least seven unused calls.
Update the archive only after the complete locked batch returns.

Report best feasible docking score versus calls and wall time, parent and
option coverage, intended versus realized scale, exact ancestry, topology,
candidate diversity, feasibility, executor counts and proposal time. No
improvement is a first-class result, not authority to tune weights or continue.

Focused tests cover exact-state/source reuse, all 33 scores, deterministic RNG,
unchanged original-seed constraints, candidate locking, one-round accounting,
and no implicit retries. Apply the user-approved bounded T4 verification policy
in AGENTS.md; do not repeat the unrelated non-green broad suite. Launch only
from a clean committed source after strict preflight, deployment, and
`python3 tools/t4_launch.py --feedback-round`.
