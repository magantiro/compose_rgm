# Objective-aware exact ring support — 2026-07-18

## H100 preflight diagnosis

The first Stage-3 exact-support preflight used the immutable label
`compose-v4-stage3-exact-support-preflight-08fe2b1-v1`. Path compilation and
validation/test construction completed, and the model was numerically healthy:

- initial validation loss: 56.1582;
- step-50 validation loss: 25.1875;
- family accuracy: 0.0769 at initialization and 0.5897 at step 50;
- step-1 GPU forward/backward/optimizer time: approximately 1.11 seconds;
- step-1 data wait: 119.31 seconds;
- sustained steps 1–50: approximately 8.2 seconds per update.

The run was stopped deliberately after its step-50 recovery checkpoint. It
failed the throughput gate, not the learning or chemistry gate. Its artifacts
remain on `compose-v4-artifacts` under the run label above and must not be used
as a quality checkpoint.

## Exact optimization

The hierarchical marked-rate model uses a separate family logit. Consequently,
the complete within-family ring partition is required only when the supervised
teacher family is `ring_system_grow`. For every other teacher row, Generator
Matching needs only the exact Boolean application condition saying whether the
ring family is enabled.

The optimized loader therefore computes:

1. the complete executor-aware template support for ring-grow teacher rows;
2. an exact one-witness enablement certificate for all other rows, stopping at
   the first executor-supported template;
3. an exhaustive all-false certificate when no ring template is executable.

Catalog witnesses remain positive accelerators only. Failed or absent catalog
witnesses fall back to the complete semantic decoder, so the optimization does
not change the executable rewrite manifold. Sampling is unchanged: it still
refines to complete exact support when the ring family is selected.

## Frozen-cache benchmark

The benchmark used the same first 128 deterministic Stage-3 training examples.
All complete-mask baseline worker configurations produced the same mask hash.

| Workers | Prefetch | Baseline throughput | Baseline wall |
|---:|---:|---:|---:|
| 4 | 1 | 1.40 examples/s | 91.32 s |
| 4 | 2 | 1.54 examples/s | 83.35 s |
| 8 | 1 | 1.69 examples/s | 75.54 s |
| 8 | 2 | 1.91 examples/s | 67.19 s |
| 12 | 1 | 1.74 examples/s | 73.67 s |
| 12 | 2 | 1.83 examples/s | 69.83 s |

With objective-aware support at 8 workers and prefetch 2, two repeated runs
completed in 19.67 and 20.06 seconds (mean 6.44 examples/s). The output hashes
matched across both repetitions. Routing was exact: 17 ring teachers received
17 complete masks; 111 other rows received certificates; 87 certificates were
one-hot enabled and 24 were exhaustive disabled; none contained multiple
witnesses. This is a 3.38x total speedup and approximately 8.34x after worker
startup.

## Verification

- Certificate enablement matched complete executor support on the frozen
  diagnostic states.
- A non-ring teacher test verifies identical enabled families, family
  probabilities, selected-mark likelihood, loss, and parameter gradients under
  complete support versus the exact certificate.
- Ring-grow teacher rows are asserted to retain complete support.
- The refreshed frozen 128-example teacher audit reports zero failures across
  atom insert/delete/restate, bond reorder, Graft, ring grow, and terminal rows.
- Local regression: 242 tests passed with two pre-existing warnings.
- Modal entrypoints: 8 tests passed.

## Next gate

Run a fresh 200-update H100 preflight with 8 data workers, prefetch factor 2,
and cumulative loader telemetry. Production training remains blocked until the
preflight completes, ancestral rollouts succeed, and sustained data throughput
is commensurate with GPU update throughput. The production recipe validates
every 250 updates, uses early-stopping patience 6 after a 500-step warmup, and
exports `checkpoint.best_so_far.pt` every 500 updates. The first improved
post-warmup snapshot receives a concurrent 100-sample rollout evaluation before
the full training run is allowed to consume its complete budget.
