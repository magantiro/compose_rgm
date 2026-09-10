# Four-point option decision diagnosis, 2026-09-09

Measured: all 16 reference proposal attempts completed, producing 15 distinct
canonical molecules; 12/16 attempts met the existing oracle-eligibility gate.
All eight focal attempts completed and remained oracle-eligible. Each of the
four known continuations occurred in at least one focal sample. No new docking
calls, controller training, executor changes, or production WHERE changes ran.

| Conditional decision | Exact known product in focal samples | Known product's guide rank among unique eligible samples | Focal option prior |
| --- | ---: | ---: | ---: |
| Pendant six-membered aromatic ring | 1/2 | 2/4 | 0.983% |
| Fused six-membered nonaromatic ring | 1/2 | 2/2 | 1.020% |
| Add carbonyl | 2/2 | 3/3 | 3.870% |
| Insert core ring carbonyl | 1/2 | 1/2 | 4.770% |

The final focal sample in the core-carbonyl case is the complete first IVG
winner molecule, but it was generated from the saved penultimate state, not
autonomously from the benchmark seed. The four focal option types and their
starting states were answer-known development strata. Attachment sites and
the known path were absent from proposal generation; the path was checked only
after locking each worker's four attempts. The remaining two attempts per case
were balanced-prior option draws, not another matched focal arm.

The post-lock support audit found positive mass for all four exact-state
prefixes, respectively 0.05, 0.07142857, 0.14311781 and 0.05, conditional on
the option and fully mutable region. These are not total canonical endpoint
probabilities. Two trials per stratum do not estimate reliable success rates.

## What this locates, and what it does not

Construction is now demonstrated under the learned-law option kernel at these
states, not merely through a forced executor replay. Pendant construction adds
one cycle and one ring system; fused construction adds one cycle within an
existing system. The carbonyl operations preserve cycle rank. No sampled focal
program hit a support dead end.

The guide does see improvement over the starting state for every known
continuation: predicted docking changes are -0.271, -0.085, -0.061 and -0.123.
It does not consistently prefer that continuation over alternatives. In the
carbonyl case, fluorination and methyl addition both rank above the known
carbonyl. In the fused case, the preferred positional isomer differs by only
0.015 predicted docking units. These predictions are not observed docking
comparisons, and a known winner does not prove every intermediate or alternative
has inferior docking.

The roughly 1% prior mass of each focal ring variant describes its opportunity
rate at these particular states; it is not a reason to tune weights to this
winner. The present assay supplies focal opportunities explicitly. It therefore
does not establish autonomous WHAT selection, local/global WHERE allocation,
multi-option chaining, original-seed linker remodeling, or benchmark performance.

Proposed next experiment: a small winner-blind multi-option search that preserves
diverse completed transformations and tests whether task feedback can allocate
and chain them. Keep generic and local/global exploration. Do not add another
ring template, change the executor, tune a prior to these winners, or claim a
new docking result from this diagnosis. No next experiment was launched.

## Runtime and provenance

The four one-core CPU workers ran concurrently. Complete diagnostic times were
275.5, 329.9, 169.4 and 241.4 seconds, including initialization and post-lock
support inspection. Initialization took 131.8, 196.1, 133.8 and 148.3 seconds.
Post-initialization work took 143.8, 133.8, 35.7 and 93.1 seconds. Pure proposal
time was not isolated, so those times must not be labeled proposal-only latency.
The workers made 40 fresh marked-law enumerations and 3,005 total public executor
calls, including support inspection. Compatible old-law caches were inventoried
but yielded no exact-state hits; the reported cache hits are within-worker reuse.

Run: `f9dce46f60769f19299140c6c786edb22a161fa256b129b31345c5e231aa773b`.
Generation revision: `0e02cfd5f6bfd406b71d4a7b3699f311f8f8d08f`.
Modal volume: `compose-v4-artifacts`, under
`/t4_option_decision_audit/case_{0,1,2,3}/<run>/`.
Raw marked-law caches and executor receipts remain on that volume. Compact
results, exact-state locks, value snapshots, runtime gates, cache inventories,
final heartbeats and the serialized-source launch identity are preserved here.

All frozen physical inputs passed their remote checks. All four guide fits used
identical recipes, training rows and features. One worker's float64 coefficients
differed by at most 3.89e-16, producing a different snapshot hash. The initial
byte-identity assertion in the reducer failed; this is retained as a numerical
deviation, not described as byte-identical execution. Since the fixed kernel is
in [0,1], the coefficient L1 difference bounds prediction changes by 5.37e-15.
All compared distinct-molecule score gaps exceed twice this bound, so every
reported ordering is stable. Future fan-outs should distribute one saved value
snapshot rather than independently solve the same fit.

Reproduce the numeric reduction without Modal, RDKit, model fitting or docking:

```sh
python3 diagnostics/t4_option_decision_audit/summarize.py
```

`summary.json` binds physical input hashes and the reducer hash. The remote
results bind the frozen recipe, input hashes, exact generation revision, seeds,
CPU float32 reference model, elapsed time and peak RSS. The mounted image pins
Python 3.11, torch 2.4.0, NumPy 1.26.4, SciPy 1.13.1 and RDKit 2024.03.5;
RDKit identity is explicitly checked at worker entry.

Verification: 11 focused tests passed, with zero failures/errors/skips; strict
clean-source preflight and focused lint passed. The reducer validates sealed
artifacts, configuration/source consistency, complete attempts, zero-oracle
accounting and numerical ordering stability. No repository-wide suite ran for
this bounded development diagnosis; this is not a full milestone qualification.
