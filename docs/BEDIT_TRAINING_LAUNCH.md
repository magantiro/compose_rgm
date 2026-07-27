# B-edit (broad-organic) training launch — reviewed spec

**Status: WIRED + verified, NOT yet launched.** The corpus is mined, the recipe is
locked. What remains before the A100 run is operational, not code: resolve the cache dependencies, choose
steps/count, and launch from a clean committed worktree with the prelaunch gate green.

> **⚠️ 2026-07-27 CORRECTIONS (this header supersedes the stale body below; authoritative status lives in
> `docs/PRODUCTION_PREFLIGHT_REPORT.md`):**
> 1. **Scope is max_atoms=40, not 48.** Corpus 466,483 eligible (not 490,466); scope_hash `3721d69851110fdd`;
>    pool `/artifacts/edit_mining_full_broad_40/edit_pool_full.jsonl` (363,456 records); manifest
>    `scaled_edit_data_manifest_40.json`.
> 2. **Warm-start flag: use `--initialize-compatible-from-source-checkpoint`.** The `--initialize-from-source-
>    checkpoint` (strict) command below **CRASHES** — B has CNOF 4-class heads, B-edit has organic 15-class;
>    strict `load_state_dict` raises a `restate_head/grow_root_head` size mismatch. The compatible path does
>    semantic partial transfer (shared C/N/O/F rows copied by label; new S/P/Cl/Br/I/B rows fresh).
> 3. **Base checkpoint** `…3k-1ac6f19-v1/checkpoint.best_so_far.pt` SHA `c9d927…`, trained on a deterministic
>    **50,000-molecule CNOF-neutral subset** (of 225,149 CNOF-eligible ≤40) — NOT all 500k; `b_train_sha256
>    83cebcef…`. B-edit mining/training must not be reduced to that 50k CNOF subset.
> 4. **Caches: v2 (`--scaled-manifest`) needs NO de-novo path cache and NO support cache** (`denovo_keep=0`
>    discards de-novo records; support is on-the-fly). The de-novo `--compile-only` step below is unnecessary
>    once the zero-mixture branch lands. See the preflight report §6.

## Inputs (all verified)

| Thing | Value |
|---|---|
| Base checkpoint (Lineage B, warm-start) | `compose-v4-stage3-flexible-graft-3k-1ac6f19-v1/checkpoint.best_so_far.pt` (on `compose-v4-artifacts`) |
| MMP pool (Layer 2) | `/artifacts/edit_mining_full_broad/edit_pool_full.jsonl` — **347,793 records** (mined this session) |
| Scaled manifest (locked recipe) | `diagnostics/composition/scaled_edit_data_manifest.json` → upload to `/artifacts/edit_mining_full_broad/scaled_edit_data_manifest.json` |
| Corpus | `guacamol_subset_500000_seed0.smiles` (broad-organic scope, 490,466 eligible) |
| Locked mixture / curriculum | corruption 0.55 / mmp 0.45 / scaffold 0 ; path-length bins [5, 9, 13] ; cold-element floor 0.02 |

## Wiring status: DONE (`36b741a`, 498 tests green)

Both pieces are wired, opt-in, and fail-safe (de-novo B byte-identical when the flags are absent):
- **Modal passthrough** — `analogue_trace_pool` / `analogue_trace_count` / `corrupted_prior_count` /
  `scaled_manifest` thread `main → train_stage.spawn → train_stage → _run_remote → recipe["arguments"]`.
- **Gate `--scaled-manifest`** — builds the hierarchical sampler from the layer slices + the manifest and
  passes it to `train_factorized_mark_model` (verified on real records: mmp draw fraction 0.448 ≈ 0.45).

## Two recipes: v1 (fast, cached) vs v2 (hierarchical)

The training-support-cache COMPILER draws records uniformly (a separate path from the dataset), so the
hierarchical sampler cannot reuse a pre-compiled cache — the gate GUARDS `--scaled-manifest` against a
pre-compiled cache.

- **v1 (recommended first run):** UNIFORM sampling + the support cache (fast, correct). Hit the 55/45
  mixture by COUNT — `--corrupted-prior-count`/`--analogue-trace-count`. Omit `--scaled-manifest`. This is
  the launch command above.
- **v2 (curriculum + cold-element floors):** add `--scaled-manifest ...`, and run WITHOUT
  `--require-training-support-cache` (support computed on the fly — slower, correct). Use once v1 trains.

## Timing / duration (honest)

Early stopping + checkpointing are implemented and were used to train B: `--early-stopping-patience 6`,
`evaluation_every 250`, min-delta 0.1%, restoring best state; periodic `checkpoint.best_so_far.pt` +
`checkpoint.recovery.pt`; the A100 `train_stage` runs `--detach`, 24h timeout, 1 retry that resumes from
recovery. So the run is **self-limiting** (stops at convergence, ~1–3k steps for a warm-start fine-tune) and
resumable. There is **no recorded A100 steps/sec for the organic 15-class config**, so any wall-clock is a
guess. To measure it: run a short `--train-only --training-steps 300` and read the logged steps/sec + the
one-time support-compile cost. That probe is a real A100 op with prerequisites — the base checkpoint is on
the volume (`compose-v4-stage3-flexible-graft-3k-1ac6f19-v1/checkpoint.best_so_far.pt`) but B's **path /
eval / support caches must be present (or recompiled)** first; verify before the probe.

## Prerequisite (VERIFIED gap): caches must be compiled for the broad-organic config

B's run dir (`compose-v4-stage3-flexible-graft-3k-1ac6f19-v1`) holds only `checkpoint.best_so_far.pt` +
`checkpoint.recovery.pt` + manifests — **no `compiled_paths.pt` / eval / support caches**. And B's path
cache was compiled for the **CNOF-neutral** split; B-edit uses the **broad-organic** split (a different
molecule set → different de-novo carbon-tree paths), so **B's cache is NOT reusable**. So before `--train-
only`, compile fresh for the broad-organic config:

1. `--compile-only --organic-vocabulary` → `compiled_paths.pt` for the broad-organic TRAIN split (+ eval
   cache). Confirm the compile stage loads the broad-organic split (it goes through the same
   `--organic-vocabulary` gate path).
2. `--support-compile-only --corrupted-prior-mix --organic-vocabulary --analogue-trace-pool <pool>
   [--corrupted-prior-count N]` → the mixed-tuple support cache (v1). (Skip for v2 `--scaled-manifest`,
   which is on-the-fly.)

Open question to resolve at setup: under `--scaled-manifest` the de-novo slice is dropped (denovo_keep=0)
yet `--train-only` still requires the path cache to build the (then-sliced-away) de-novo records — either
compile it anyway, or add a gate path that skips de-novo when denovo_keep=0. Straightforward, but decide it
deliberately.

Each stage is real (CPU) compute with config nuances — set it up + verify each stage green, do NOT batch-fire.

## Launch sequence (from a clean committed worktree at the launch tag)

```bash
# 0. upload the locked manifest next to the pool
modal volume put compose-v4-artifacts \
  diagnostics/composition/scaled_edit_data_manifest.json \
  edit_mining_full_broad/scaled_edit_data_manifest.json

# 1. prelaunch gate (full suite + ruff + clean tree + corpus + hashes) -- must be green
PYTHONPATH=src python scripts/prelaunch_gate.py --corpus guacamol_subset_500000_seed0.smiles \
  --expected-commit <LAUNCH_TAG>

# 2. recompile the training-support cache over the MIXED tuple (its signature changed) -- CPU
modal run modal_apps/train_tracelet_gm.py --support-compile-only \
  --corrupted-prior-mix --organic-vocabulary \
  --analogue-trace-pool /artifacts/edit_mining_full_broad/edit_pool_full.jsonl \
  --scaled-manifest /artifacts/edit_mining_full_broad/scaled_edit_data_manifest.json \
  --run-label compose-v4-bedit-broad-<COMMIT>-v1 \
  --source-run-label compose-v4-stage3-flexible-graft-3k-1ac6f19-v1

# 3. warm-start fine-tune B -> B-edit (A100). --train-only reuses B's path/eval caches + the recompiled
#    support cache; strict warm-start loads B into the flag-constructed (organic, enable_*) model.
modal run --detach modal_apps/train_tracelet_gm.py --train-only \
  --corrupted-prior-mix --organic-vocabulary \
  --analogue-trace-pool /artifacts/edit_mining_full_broad/edit_pool_full.jsonl \
  --analogue-trace-count 0 \
  --corrupted-prior-count <N_CORRUPTION> \
  --scaled-manifest /artifacts/edit_mining_full_broad/scaled_edit_data_manifest.json \
  --run-label compose-v4-bedit-broad-<COMMIT>-v1 \
  --source-run-label compose-v4-stage3-flexible-graft-3k-1ac6f19-v1 \
  --initialize-compatible-from-source-checkpoint \
  --initialization-source-run-label compose-v4-stage3-flexible-graft-3k-1ac6f19-v1 \
  --checkpoint-name checkpoint.best_so_far.pt \
  --training-steps <STEPS> --schedule-steps <STEPS>
```

- `--schedule-steps == --training-steps` so cosine LR fully decays in the short fine-tune.
- Sizing: `--corrupted-prior-count` sets how many corruption records are generated in-memory; with the
  hierarchical sampler the mixture is set by the manifest WEIGHTS (55/45), so the counts only need to be
  large enough to draw from (the pool is 347,793; a corruption count of order 1e5 gives comfortable
  coverage). Confirm the exact steps/count against compute budget before launch.

## Post-training gate (before ANY experiment)

```bash
PYTHONPATH=src python scripts/broad_preflight_gate.py \
  --checkpoint /artifacts/compose-v4-bedit-broad-<COMMIT>-v1/checkpoint.best_so_far.pt
```
Must return `GO_FOR_FULL_A100`... (here, GO for experiments). `NO_GO_BROAD_CHARGE_CONTEXT` on any
charged-context failure — no silent neutral fallback. It also enforces the `corpus_scope_hash` match
(`e59fb09801459470`).

## Standing constraints
- Never launch before `prelaunch_gate.py` is green; launch only from a clean committed worktree.
- Do not modify the audited sampler / GM loss / executor / operator semantics / checkpoint contract.
- The A100 run is the real spend — confirm steps + count + budget before `modal run --detach`.
