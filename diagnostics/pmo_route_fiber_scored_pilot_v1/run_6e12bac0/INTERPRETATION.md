# PMO route-prior x FiberControl scored-pilot forensic audit

Run `6e12bac0e69c30e7cafcd94cc124936aa7f5f2754eba7bda1a687c658afa3088` completed all eight 48-call units, but it is not a valid two-task scientific comparison. The 192 GSK3B rows are PyTDC fallback zeros caused by an evaluator-adapter path defect, not GSK3B model predictions. The Perindopril rows remain interpretable as a bounded warm development factorial and do not meet the frozen promotion criterion.

The result envelope is internally intact. Its payload hash, run identifier, launch payload identity, 8 by 48 accounting, all 48-point AUC curves, the 16 candidate-pool file and payload hashes, selected-candidate membership, exact-replay and validity flags, and round-1 selector pairing all verify. The aggregate does not contain the physical query-lock envelopes or per-call ledgers, so those remote files were not independently rehashed here.

## Eight-arm result

| Task | Arm | AUC48 | Best | Top-10 mean at call 48 |
|---|---|---:|---:|---:|
| GSK3B | additive route, blind | 0.000000 | 0.000000 | 0.000000 |
| GSK3B | additive route, FiberControl | 0.000000 | 0.000000 | 0.000000 |
| GSK3B | old v0, blind | 0.000000 | 0.000000 | 0.000000 |
| GSK3B | old v0, FiberControl | 0.000000 | 0.000000 | 0.000000 |
| Perindopril MPO | additive route, blind | 0.168694 | 0.465475 | 0.254400 |
| Perindopril MPO | additive route, FiberControl | 0.177851 | 0.465475 | 0.286028 |
| Perindopril MPO | old v0, blind | 0.162192 | 0.465475 | 0.232805 |
| Perindopril MPO | old v0, FiberControl | 0.174894 | 0.465475 | 0.306484 |

The machine-readable audit records best and top-10 curves at calls 16, 24, 32, 40 and 48 for every arm.

## GSK3B failure

The candidates were not the problem. Every selected candidate belonged to its immutable declared pool and every pool candidate had exact replay and validity set true. The ledger also rejects an invalid molecule before evaluator invocation.

The pinned worker constructed `tdc.Oracle` while temporarily inside the asset directory, then restored its prior working directory before evaluation. PyTDC's GSK3B function does not load the model during construction. It loads `oracle/gsk3b_current.pkl` lazily from a relative path on the first call. PyTDC catches that evaluator exception and returns `default_property = 0.0`. The outer ledger therefore recorded finite completed zeros and could not see the hidden failure. This explains the exact 0.0 for all 64 initialization charges and 128 candidate charges. The asset itself passed its SHA-256 preflight; the defect is path binding at evaluation time.

The GSK3B task is scientifically invalid and must not be interpreted as a negative chemistry or controller result. Its 192 calls remain historically charged. Re-evaluation is not authorized by this audit.

## Perindopril interpretation

Under blind selection, adding the route lane improved AUC48 by 0.006501 over old v0. FiberControl improved AUC48 by 0.012702 on old v0 and by 0.009158 on the additive route pool. The combined arm was 0.015659 above old-v0 blind but only 0.002957 above the better single-factor arm, well below the frozen 0.02 promotion margin. Its factorial interaction was negative, -0.003544.

The benefit was archive breadth, not a new champion. All four arms retained the same 0.465475 initialization best; the strongest selected child was 0.424094. The old-v0 FiberControl arm actually ended with a higher call-48 top-ten mean than the combined arm, 0.306484 versus 0.286028, despite the combined arm's slightly higher integrated AUC48. FiberControl selected 14 route-lane candidates in the combined arm, and six route children improved their own parent.

## Support mismatch

The scored route sampler was capped at eight primitives and returned to the same 16 immutable initialization parents in every round. Candidate pools were locked before scoring and descendants could never become parents. This was therefore a root-conditioned selection assay, not dynamic optimization.

The sealed dependency-region corpus contains 184 unique exact teacher traces. Only one is at most eight primitives. Of the 106 traces within the declared 32-primitive runtime support, 105, or 99.06 percent, exceed the scored sampler's eight-primitive horizon. All 16 Perindopril teacher routes exceed 32 primitives, so none is a complete runtime-supported Perindopril route. Horizon saturation was visible in the locked scored pools: 47 of 64 GSK3B route-lane candidates and 40 of 63 Perindopril route-lane candidates stopped at exactly eight primitives.

## Decision

Retain the invalid GSK3B result and the valid Perindopril-only development evidence, but do not promote this revision and do not spend more calls now. The next PMO action should be a zero-oracle complete dependency-region support replay from the frozen sources under the existing 32-primitive and eight-component limits. It must measure exact execution precision, complete endpoint and transformation recall, source coverage and explicit budget abstention against the eight-primitive transition sampler. Perindopril should remain an explicit over-32 abstention unless a separate support expansion is authorized.

No Modal, oracle or docking call was made by this audit.
