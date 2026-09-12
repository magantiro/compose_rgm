# Coordinated programs reach a strong neighbor, not an IVG improvement

All 26 locked non-winner candidates returned a docking score. The best scored
-13.3; two fresh evaluations also scored -13.3. The reused published-winner
controls were -13.6, -13.6, -13.6. The +0.3 repeat-only difference is worse, not
a successful improvement. This is winner-informed PARP1 seed0, delta=0.4
development. The same known route supplied the program bank.

| Context | Proposal arm | Complete / 32 | Eligible / 32 | Unique newly scored | Best new score |
| --- | --- | ---: | ---: | ---: | ---: |
| Original seed | Uninterrupted serial | 12 | 9 | 3 | -12.6 |
| Original seed | Independent multi-site | 17 | 16 | 9 | -9.7 |
| Original seed | Joint multi-site | 28 | 28 | 2 | -13.3 |
| Post-linker | Uninterrupted serial | 13 | 8 | 3 | -12.6 |
| Post-linker | Independent multi-site | 18 | 16 | 9 | -13.3 |
| Post-linker | Joint multi-site | 30 | 24 | 5 | -13.3 |

Completion and eligibility count attempted programs, including repeated known
winner reconstruction. Unique newly scored columns exclude the known winner;
shared candidates occur in multiple arms but consumed only one first docking.
These are two contexts of one source, not independent target replications. Work
ceilings matched within context; actual work and candidate counts did not.
Every eligible locked molecule was docked, without surrogate selection.

The five non-winner joint post-linker variants scored -12.6, -12.6, -13.0,
-13.1, and -13.3. Only the overall best has fresh repeats. The joint bank
concentrates on useful neighboring completions but remains narrow; its advantage
does not establish transferable task learning. Independent combinations add
diversity but frequently miss the productive coordination. Retain both channels;
do not spend another round simply resampling this bank unchanged.

Best attachment variant:

```text
O=C1CCCc2cc(CCc3ccc4c(c3)CNC(=O)c3cccc(=O)n3-4)ccc21
```

Original-seed similarity 0.4126984, QED 0.7354256, SA 2.7318931. It was proposed
by the joint arm from both starts and by the independent arm post-linker. No
claim of molecule novelty is made. The inherited raw comparison helper uses the
phrase "winner-initialized refinement"; that is a labeling error for this pool.
The unmodified receipt is retained and `review.json` explicitly corrects the
interpretation. Its starts are the seed or the recorded post-linker state.

## Cost and provenance

- 28 new calls: 26 first evaluations, two confirmations; three reused controls.
- 54.6743 seconds of driver runtime; 94.9040 summed docking-worker seconds.
  Image deployment and local analysis are outside those timings. No GPU.
- Maximum eight workers plus a driver. Reserved $20, not a claim of billed cost.
- Source commit `1762f2db4ed3`; call `fc-01M2BM16P93H7P304NPDBN4FN1`.
- Modal volume `compose-v4-artifacts`, prefix
  `t4_program_pool/79c5b633c04b3f5cc1dc30ab729e8522b521a9ebf12118811fddaeaaebfb2563`.
- Completed at 2026-09-12T20:14:12.823977+00:00. No active continuation.
- Immutable lock `configs/t4_program_pool_lock.json`, file SHA-256
  `077daa1965bfa0ee53b14463a3ffe43af36ddbc2e4ba66b100ccb56c0ab5242e`.
- `remote_result.json`: unchanged sealed complete receipt. `review.json`:
  deterministic reduction binding all inputs and analysis implementation.
  Exact ligand/pose artifacts persist in the volume's `pool` and `confirmation`
  subtrees under the same run identity.

## Scored replay, prepared only

`scored_program_replay.json` retains 54 complete attachment/program
representations for 27 measured endpoints, including the reused winner. All
examples belong to one inspected source group; neither training nor validation
on these examples alone can demonstrate transfer. The 91 rejected or unscored
attempts are explicit exclusions without fabricated docking targets. Each
source receives equal mass, then each endpoint, then its distinct program
representations. Repeatedly sampled identical programs do not multiply mass.
These are structural replay weights, not policy advantages or future values.
Preparation made zero oracle and zero executor calls; no model fitting occurred.

Reproduce the review and preparation using the pinned chemistry environment:

```sh
PYTHONPATH=src:. python tools/review_t4_program_pool.py diagnostics/t4_program_pool/attempt_1/remote_result.json diagnostics/t4_program_pool/attempt_1/review.json
PYTHONPATH=src:. python tools/prepare_scored_program_replay.py diagnostics/t4_program_pool/attempt_1/review.json /tmp/scored_program_replay.json
```

The replay command refuses an existing output. Focused aggregation/balance tests
passed. Broader controller integration and source-disjoint task validation remain
unfinished; no full-suite or milestone-completion claim is made.
