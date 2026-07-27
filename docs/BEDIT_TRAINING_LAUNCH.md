# B-edit (broad-organic) training launch — reviewed spec

**Status: reviewed, NOT yet launched.** The training corpus is mined + the recipe is locked. This spec is
the exact command + the small remaining wiring to hand off. Two bounded wiring changes remain (below); once
they land + the prelaunch gate is green, launch from a clean committed worktree.

## Inputs (all verified)

| Thing | Value |
|---|---|
| Base checkpoint (Lineage B, warm-start) | `compose-v4-stage3-flexible-graft-3k-1ac6f19-v1/checkpoint.best_so_far.pt` (on `compose-v4-artifacts`) |
| MMP pool (Layer 2) | `/artifacts/edit_mining_full_broad/edit_pool_full.jsonl` — **347,793 records** (mined this session) |
| Scaled manifest (locked recipe) | `diagnostics/composition/scaled_edit_data_manifest.json` → upload to `/artifacts/edit_mining_full_broad/scaled_edit_data_manifest.json` |
| Corpus | `guacamol_subset_500000_seed0.smiles` (broad-organic scope, 490,466 eligible) |
| Locked mixture / curriculum | corruption 0.55 / mmp 0.45 / scaffold 0 ; path-length bins [5, 9, 13] ; cold-element floor 0.02 |

## Remaining wiring (bounded — do first, then re-gate)

Both are opt-in and fail-safe; de-novo B is byte-identical when the new flags are absent.

1. **Modal app passthrough** (`modal_apps/train_tracelet_gm.py`): add 4 args and thread them through the
   entrypoint `main` (:2276) → `train_stage` (:1439) → `_run_remote` (:698), emitting into
   `recipe["arguments"]` (the recipe→argv converter already handles str/int: `--flag value`):
   - `analogue_trace_pool: str | None` → `recipe["arguments"]["analogue_trace_pool"]`
   - `analogue_trace_count: int = 0` → `["analogue_trace_count"]` (0 = whole pool)
   - `corrupted_prior_count: int | None` → `["corrupted_prior_count"]`
   - `scaled_manifest: str | None` → `["scaled_manifest"]`
2. **Gate hierarchical sampler** (`scripts/train_tracelet_cnof_gate.py`): add `--scaled-manifest`; capture
   the layer boundaries (`denovo_keep`, `len(edit_records)`, `len(analogue_records)` at :2106/:2126); when
   set, build `record_index_sampler = build_layered_sampler({corruption: edit slice, mmp: analogue slice},
   layer_weights=<manifest>, path_length_bins=<manifest [5,9,13]>, cold_element_floor=<manifest>,
   cold_elements_of=<non-CNOF of record.path.trace.target>)` and pass it to `train_factorized_mark_model`
   at :3057 (the param + full plumbing already exist + are tested). Set `denovo_keep=0` under
   `--scaled-manifest` (the mixture is manifest-controlled; de-novo retained via the warm-start).

Smoke both with `--smoke` before the real launch (the mixed-record path must build the sampler + backprop
FINITE, as the local training smoke already verifies).

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
  --initialize-from-source-checkpoint \
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
