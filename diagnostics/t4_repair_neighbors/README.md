# One-edit repair availability, 2026-09-10

## Finding

All three selected benchmark-ineligible intermediates have new eligible
one-edit continuations under the frozen production process. Across the three
roots there are 25 distinct eligible molecules absent from the prior 51-call
docking archive. Every eligible representative edge passed exact executor replay.
No docking calls or model training were performed.

| Root, prior option | Starting similarity | Unique one-edit products | Eligible products | New to prior archive | Returns to incumbent |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0, shrink | 0.397059 | 505 | 11 | 10 | 1 |
| 1, aromatize | 0.394366 | 632 | 6 | 5 | 1 |
| 2, open | 0.380282 | 569 | 11 | 10 | 1 |

Coverage is 3/3 selected roots with a new eligible continuation. Eligible
root-product pairs are 28/1,706, including three returns to the same incumbent.
Canonical deduplication is within each root and across new eligible products;
the denominator counts root-product pairs, not globally unique molecules.
Five further pairs pass the benchmark properties but fail the separate inherited
medchem screen. All new eligible repairs preserve graph cycle rank and ring-system
count relative to their respective starting intermediate. They are not evidence
of additional ring construction.

The root at similarity 0.397059 was discarded without a continuation in the
previous graded-beam run. The present census verifies ten alternative eligible
continuations from that exact persistent-slot state, besides undoing to the
incumbent. Thus benchmark failure at an intermediate does not imply the absence
of a successful continuation, even just one primitive edit later.

## Negative finding and interpretation

None of the 25 new eligible molecules is predicted to improve on the incumbent
by the same frozen docking surrogate. The best new prediction is -8.937673,
versus -9.148093 for the incumbent under that model. These are predictions,
not new docking observations; the previously observed incumbent score remains
-9.7. The model's extrapolation to these products is unvalidated.

This establishes repair availability, not that the optimizer samples the repairs,
that every detour is useful, or that docking performance improved. The panel is
the three lowest-violation distinct products of one inspected development arm,
not a held-out or representative sample. Most one-edit products remain ineligible.
No recovery probability or canonical-successor probability law was estimated.

The next proposed controller experiment is continuation-aware selection at
complete-option boundaries. It must preserve bundle accounting and exploration,
keep terminal eligibility unchanged, and value eventual task performance rather
than merely reward returning to the feasible region. A matched comparison must
test whether that selection actually yields useful eligible offspring. No such
follow-up or docking batch was launched by this diagnostic.

## Why the 0.4 threshold remains

The T4 benchmark uses original-seed Morgan-fingerprint Tanimoto similarity with
thresholds 0.4 and 0.6, alongside QED >= 0.6 and SA <= 4. The constraints are
stated in [InVirtuoGen section 3.4](https://arxiv.org/html/2509.26405v1#S3.S4),
which follows the GenMol benchmark. Here 0.4 is a returned-molecule eligibility
floor, not a docking objective, a biological-quality guarantee, or a percentage
of preserved atoms. All three probe roots are below that floor and are still
valid inputs to the executable process. The threshold was never relaxed.

## Compute and verification

One CPU worker completed the check in 218.51 seconds including initialization.
Enumeration, post-lock assessment and replay took 77.57 seconds. Two fresh
reference rows took 13.18 seconds; the third was reused after input and dependency
verification. The public executor meter counted 2,150 calls, including calls
inside enumeration and representative replay. CPU float32 model inference used
the pinned runtime; no GPU was requested. No failed root or truncated support
was converted into an absence-of-repair result.

Nine focused tests passed on the exact clean generation revision in 18.05 seconds.
Strict preflight, touched-code lint and formatting passed. The legacy app retains
17 unchanged lint findings, none in the new wrapper. No full repository suite was
run and no repository-wide milestone is claimed complete.

## Evidence

- Generation revision: `18cab20f7e277d2dda436c1244bd564f0d61b181`.
- [Contract](../../configs/t4_repair_neighbors.json) and
  [prospective decision](../../docs/T4_REPAIR_NEIGHBORS.md).
- [Summary](summary.json), [raw result](attempt_1/result.json),
  [panel lock](attempt_1/panel_lock.json), [launch receipt](attempt_1/launch.json),
  and [verification](verification.json).
- Volume: `compose-v4-artifacts`, prefix
  `t4_repair_neighbors/7bb480732e9ce78061ce4d1313fcfd1ff443319433aa7872a9862a0160bff67b`.
  Complete generation/scored locks, exact law files and executor receipts remain
  there. The summary binds physical hashes of all generation/scored locks used
  in its reduction. The compact local result retains every eligible witness.
- `fetch.py` reads existing artifacts only. It checks sealed payloads, exact
  input identity, generation/scoring correspondence, mark and canonical counts,
  endpoint eligibility, known/incumbent accounting and all three completed roots.
  Run with the pinned RDKit 2024.03.5 environment and Modal read access:
  `PYTHONPATH=src:. python diagnostics/t4_repair_neighbors/fetch.py`.

Two successive reductions at the same revision were byte-identical. The analysis
script hash and revision are stored in the summary. Repeat reductions
at another revision intentionally change provenance fields; they do not rerun
molecular enumeration or docking.
