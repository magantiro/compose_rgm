# Scaled-data build + training-execution stages (authoritative)

**Status: AUTHORITATIVE** for everything between the data-starvation finding and the first scientific
editing-prior training run. Owner mandate, 2026-07-28.

**Governing principle — do not reorder:**

> correctness → offline preprocessing → vectorization → profiling → targeted GPU optimization
>
> *not* premature CUDA optimization before the data representation is stable.

**Sequence.** Stage 1 (chemistry shards) → Stage 2 (packed cache + vectorized scoring) → Stage 3
(GPU-native training loop) → short throughput/profile run → **STOP for the scientific schedule decision**.

No further local gates or tiny training runs may appear between the census and production training unless
the build violates an exact invariant.

---

## Stage 1 — chemistry compilation ✅ BUILT (`modal_apps/precompile_edit_data_app.py`)

`partition_sources` → `compile_shard` (N parallel) → `reduce_and_validate`.

- Every map worker writes a **unique shard + manifest**; no concurrent writes to a shared file.
- **Restartable**: a shard is reused only if its manifest parses, checksum matches, and its recorded
  codec / operator-registry / capability / partitioner hashes equal the current ones. Stale-contract shards
  recompile — never silently reused.
- **Both layers**: corruption (7 families) and cycle_ops (the 2 compositional families). A corruption-only
  build would scale seven families while starving the defining RingCore families.

**Measured throughput** (local fixtures): corruption **0.98 s/source, 9.55 transitions/source**;
cycle_ops **0.049 s/source, 7.73 transitions/source** (~20× cheaper).

**Build targets** (ranges, not quotas — stop on diversity curves):

| layer | sources | expected transitions |
|---|---|---|
| corruption | 20,000–30,000 | ~190,000–285,000 |
| cycle_ops | 30,000–40,000 **eligible** | ~230,000–310,000 |
| MMP | — | retain the full **363,456** |

Stop criteria: new canonical transitions per additional source · unique scaffolds · rare-family and subtype
coverage · cycle lengths and cycle-rank changes · fused/spiro/bridged contexts · element and charge contexts
· source/scaffold concentration.

> **Cycle "eligible source" is a distinct denominator.** Report cycle coverage relative to molecules where
> close/open are actually legal, NOT relative to all 466,483 corpus molecules.

> **Artifact size, source breadth, and sampler weight are three separate decisions.** Compile cycle data
> broadly because it is cheap; do NOT give it more training probability merely because more records exist.

---

## Stage 2 — packed tensor cache + vectorized scoring (NEXT)

**Two-level artifact design.** (A) canonical audit artifacts = source of truth (exact slot states, semantic
V2 actions, successor keys, replay + provenance). (B) packed training artifacts = a **versioned
deterministic derivative**. Never the reverse.

**Hot-path boundary — all of this happens offline, never per-batch:** molecule parsing/sanitization ·
exact-state reconstruction · executor replay · inverse checks · candidate legality enumeration (fixed state)
· canonical-successor grouping · teacher-in-candidate resolution · static graph tensorization.
At training startup: validate hashes + run a small fixed sentinel replay panel. Do **not** revalidate the
full corpus per epoch.

**Ragged representation** — flattened, with: `candidate_to_example`, `candidate_to_family`,
`candidate_to_successor`, `candidate_offsets`, `successor_offsets`, `target_candidate`,
`target_successor`. Canonical-successor probability = **segmented log-sum-exp/sum over candidate aliases**.

**Sampling.** Deterministic bucketing by graph size and legal-candidate count. Scientific sampling happens
FIRST; compute bucketing must not change record probabilities. Sample via **precomputed index manifests**,
never file scans or prefix reads. Preserve locked layer weights, scaffold-aware sampling, family/curriculum
semantics. Do not bias the sampler toward cheaper records.

**Storage.** Moderate packed shards, sequential-read/memmap friendly, deterministic random access, ragged
support, cold reload, schema validation, worker-local caching. Avoid one-file-per-record, tiny shards,
repeated random decompression, or loading the corpus into every worker.

**Cache invalidation — fail loudly on ANY mismatch:** program-contract fingerprint · operator-registry hash
· capability hash · action-codec version · canonicalization version · exact-state schema · tensorization
version · candidate-enumerator hash · partitioner version.

**Gating test:** the vectorized path must match the trusted reference path on a fixed family-complete panel
**before** becoming production-default. This is the most likely place for a silent discrepancy — segmented
reductions vs the existing `logsumexp`-over-groups.

---

## Stage 3 — GPU-native training execution (AFTER Stage 2 matches reference)

Operational optimization only. **Does not change the objective, sampler, model, or numerical meaning.**

1. **GPU hot-path contract.** Once a batch is on device, keep encoder, hazard, family/action/site scoring,
   masking/normalization, successor aggregation, target gathers, losses, backward, clipping and the
   optimizer step on GPU. No `.cpu()`, `.numpy()`, `.item()` or Python branching in the hot path — `.item()`
   forces synchronization. Aggregate logging tensors; read at scheduled intervals only.
2. **Ragged vectorized scoring.** Batched ops + segmented reductions; the trusted loop stays as the
   reference test path, not production.
3. **Compute-aware bucketing** by heavy-atom count, edge count, candidate count, successor-group count.
   Persist and report realized layer/family/scaffold frequencies to prove no sampler drift.
4. **A100 numerics.** BF16 autocast for encoder/dense heads; FP32 for segmented log-sum-exp, normalization
   and final likelihood accumulation. BF16 preferred over FP16 for dynamic range. Validate against FP32.
5. **Optimizer.** Fused AdamW where numerically equivalent; `zero_grad(set_to_none=True)`; clipping on GPU.
   Gradient accumulation ONLY if justified by memory or effective batch size — it can lower throughput.
6. **Overlap.** Persistent workers, pinned memory, prefetch, non-blocking transfer, worker-local shard
   handles, double buffering. Choose settings by benchmark in the **Modal** environment, not locally.
7. **Compilation.** `torch.compile` only after the vectorized eager path is correct. Drop it if dynamic
   shapes cause frequent recompilation or the speedup is negligible. **CUDA graphs are optional** and only
   after everything above is profiled.
8. **Profiling** (`torch.profiler`, real packed shards, Modal A100): data wait · H2D · encoder · candidate
   scoring · successor aggregation · loss · backward · optimizer · logging · checkpoint I/O. Report
   examples/s, **candidate actions/s** (the most informative unit — equal molecule counts can have very
   different fiber sizes), successor groups/s, GPU utilization, memory, data-wait fraction, kernel launches,
   compile/recompile counts, cold vs steady state.
9. **Equivalence gate** on a fixed family-complete panel: total/hazard/mark loss · canonical-successor NLL ·
   per-example target probability · gradients by parameter block · one optimizer update · save/reload.
   Plus 100% teacher-in-candidate, identical successor grouping, identical record identities, no sampler
   drift. FP32 must match tightly; BF16 to a **declared tolerance**; fail on non-finite.
10. **Pass condition is operational:** *the A100 is not materially starved by avoidable data preparation,
    Python action loops, or repeated CPU synchronization.* Do **not** require a contrived utilization
    number — irregular candidate sets may not sit at 99%. Do not infer model quality from this benchmark.

---

## Unified census — required before the GPU run

Train/val/test counts for **all three layers** · unique sources and scaffolds · canonical-transition counts
· **cross-layer transition overlap** · family and subtype counts · topology and cycle-rank distributions ·
close/open balance · source/scaffold Gini · rare-family minimum counts · expected layer AND family exposures
under the proposed schedule · all contract, codec, partitioner, executor and artifact hashes.

**Partition leakage — the precise condition:**

> ( ⋃ train sources/scaffolds across ALL layers ) ∩ ( ⋃ validation/test sources/scaffolds across ALL layers ) = ∅

The same source in both MMP and corruption *within* train is **not** leakage. The dangerous case is a source
or scaffold crossing a partition **via a different layer**.

**Cross-layer duplicates:** measure identical canonical transitions across MMP↔corruption,
corruption↔cycle, MMP↔cycle. Do **not** silently deduplicate. If two records share exact source state,
action/successor, time/curriculum context AND teacher rate, keeping both reweights that transition —
collapse or cap exact semantic duplicates while preserving provenance. Genuinely different contexts stay.

**The sampler must be explicit.** State the actual training probabilities for MMP / corruption / cycle.
Do NOT infer them from artifact sizes.

**Exposure:** `E_i = S · B · w_i / N_i` (S steps, B batch, w_i layer weight, N_i unique training records).
Target ≈ **0.75–1.25** passes over MMP, **1–3** over corruption, **1–3** over cycle supervision, while
checking rare families receive enough *absolute* updates. The earlier 10k–13k step estimate remains
plausible but **cannot be locked** until the three record counts and three sampler weights are known.

---

## Execution constraints

- The full audit-artifact and packed-cache builds do **NOT** run locally. Local execution is limited to
  correctness fixtures, the small cold-reload gate, and reference-equivalence tests.
- Both builds are restartable Modal **CPU** map/reduce stages.
- The A100 training path consumes packed shards and must not repeat RDKit parsing, executor replay,
  corruption generation, canonicalization, or fixed legal-fiber construction inside the batch loop.
- The throughput benchmark runs in the **Modal A100 environment** — local I/O and CPU are not
  representative — and must not become a model-quality preflight.

## Required return at the stop

Unified census · layer weights · exposure-derived schedule · packed-cache schema and hashes · cache size by
layer · cold-load time · steady-state throughput · selected worker/prefetch configuration · precision mode ·
compile setting · timing breakdown · GPU utilization and data-wait fraction · candidate-scoring throughput ·
remaining measured bottleneck · estimated full-run duration and cost. **Then stop once for owner approval.**
