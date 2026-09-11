# Frozen-model inference speed result

The approved zero-oracle check passed its within-worker paired acceptance criteria
on all three exact archived states. The same selected model was used throughout.
The tested cache reduces repeated serialization and scans existing bonds sparsely;
it does not truncate actions, remove chemistry, or change the controller.

| Exact archived state | Legal marks | Baseline median (s) | Cached median (s) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Slow-worker parent | 1,196 | 10.072537 | 4.849325 | 2.077101x |
| Pendant six-ring draw 00 | 1,534 | 15.092354 | 6.898344 | 2.187823x |
| Pendant six-ring draw 03 | 1,598 | 15.039934 | 6.919901 | 2.173432x |

Each median uses three alternating repetitions. All ordered action/probability
records match exactly between baseline and cache, with no numerical tolerance.
Every executed exact product, canonical product identity, and validity outcome
also matches in the first repetition of each arm: 4,328 marks across the three
source states, replayed once per arm. Eighteen timed enumerations consumed
176.746221 seconds in total. Replay and orchestration are additional work and
are not included in that timing total. No new PMO labels or docking calls were
made. These timings do not establish a full-search wall-time improvement.

## Inference package

The original qualified runtime took 102.932283 seconds to initialize. Exporting
that selected CPU model and reloading it in the export worker took 0.081301
seconds for the load, with exact equality of all 1,196 marked probabilities on
the parent. The package loaded in 1.180563 seconds in the fresh probe worker.
These are model-initialization timings, not container boot or dependency-import
times. No weights were fitted, selected anew, or replaced.

The reusable package lives on `compose-v4-artifacts` at
`inference_packages/9dd2eaa142417ad9bbf7ca1bf18586e3372c3312dcbee59d40a3117bb5affe66`.
It binds physical model bytes, the selected tensor state, original checkpoint and
run-path hashes, and 116 source dependencies. Historical Gate 0 and process
identities remain unchanged. Package loading authenticates these inputs before
unpickling. The two source overrides are still explicitly probe-only; the old
search contracts have not been silently rebound to the optimized implementation.

## Scope and unresolved reproducibility detail

Exact equality was established within each worker: original versus reloaded
model in the export worker, and baseline versus cache in the probe worker.
The export worker's parent marked-law digest differs from the probe worker's
parent digest (`7ba34c96...` versus `1a6bd68a...`). The receipts retain digests,
not complete cross-worker probability arrays, so their cause and numerical size
are not determined by this check. Do not interpret the paired results as
cross-worker bitwise reproducibility, or assume the discrepancy is harmless
floating-point noise without comparing the records. This remains an integration
follow-up before making a stronger inference-runtime equivalence claim.

This result does not improve the measured PMO objective, show long-horizon ring
construction guidance, or establish generalization to other molecular states.
The last policy experiment still tied the balanced baseline on best and top-ten
archive score. A search-worker integration and a separately scoped repeated
feedback experiment remain subsequent work; neither was launched here.

## Provenance

- Contract: `94a31eae0c25b457039221c53cfcd7b498b2e555f046937015648e48d9429cb8`.
- Export producer: `ba83fcd3c41b`; run `41306bcba8ac5ce3de12e47c97cb9c96e822b16bbc81c6da65e37a400b985d1e`.
- Probe producer: `f35958e04e7d41c25f9666ac2860d579c1b37f31`; run `7418ed20cba9e885b94609216bddee550cd3a104fc667efac38d92d42d74754f`.
- Remote receipts: `pmo_inference_speed/<run>` on `compose-v4-artifacts`.
- Selected tensor SHA-256: `c977ee3fe0cfdcafa204a2add27f3feef59f5dfcf4f3384659d8902a9006167c`.
- `result_sealed.json`: `50f0a13f23165db6f1f3b475b3557ab08878e86aa0b7174256cb79a5dc08c829`.
- `export_sealed.json`: `831e640e41c7dc27d0af0cd29e4bd2b2dfbe61418c033231d60d6100fa33f79f`.
- `package_manifest.json`: `d89cfaf14e44fbdbc9e974c56ded9e6c7f09e27cbe624243a6a1faf7e8d0ea59`.

The frozen prepared file contains the exact slot-addressed source states,
original source-artifact hashes and baseline serializer text. No reconstruction
from SMILES or score-based state selection was used. The model is in evaluation
mode; law enumeration is deterministic conditional on the runtime, and there is
no random sampling in this probe. Paired arm order alternates deterministically.
Remote software: Python 3.11.12, Torch 2.4.0+cu121 on CPU, NumPy 1.26.4,
RDKit 2024.03.5. One CPU thread, float32, 8 GiB reserved, no GPU. Peak resident
memory and underlying host CPU model were not measured. No billed-cost receipt
was collected; the authorized limit was $2, not a measured expenditure.

## Checks and status

Ten focused tests passed in 2.72 seconds: inference reload, corrupt-byte rejection
before unpickling, unauthorized dependency changes, cache mutation keys, padding,
charge, invalid outcomes, bounded eviction and interruption cleanup. Strict
clean-source preflight and diff checks passed before each launch. The downloaded
seals, input identities, tensor hash, row identities/counts and reported speedup
arithmetic were checked against the pinned contract.

Ruff passed on the new export/probe code and focused tests. The two copied cache
source files retain pre-existing lint/format findings outside the optimization:
five remaining molecular-graph lint findings and legacy formatting in both files.
Those were confirmed against the baseline and were not repaired during a probe
requiring the exact predeclared source hashes. No repository-wide suite was run.
This is bounded development evidence, not a release or full milestone completion.
All code and evidence commits remain unpushed.

## Subsequent runtime admission, 2026-09-11

The original production worker's recovered full parent law
(`reference_law_sealed.json`, SHA-256
`1966ec6cddb20d2010b2ea7df3208c594cb1ff5c260c9dac87bc99cc34fc66f9`)
has digest `1a6bd68a...`, exactly matching the accelerated probe. A new worker
reproduced the other digest, `7ba34c96...`, and retained its complete arrays in
`portability_check_sealed.json`. Direct comparison measured identical ordered
actions, maximum probability difference `3.794342490204272e-08`, and marked-law
half-L1 difference `6.046095143637787e-08`. Thus the numerical magnitude is now
measured; the underlying cross-worker cause remains unproven. Neither changing
chemistry nor a probability truncation was used to obtain agreement.

The separate online-policy development contract declared portability limits of
`1e-7` maximum absolute difference and `1e-6` half-L1 before this comparison.
It retains the prior exact within-worker cache requirement. Its later admission
worker matched all 1,196 original probabilities exactly and loaded the package
in 0.176226 seconds. These are bounded runtime checks, not a claim of bitwise
equivalence across all future molecules or hardware. Original Gate 0 records
are unchanged. The separately authorized search and its outcomes are recorded
under `diagnostics/pmo_online_policy`; the earlier speed-only result above
remains zero-oracle evidence.
