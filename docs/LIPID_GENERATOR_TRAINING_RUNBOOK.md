# Lipid generator training runbook (Lineage B, unconditional, CNOF)

Everything needed to fire the **authorized exploratory lipid smoke** now, and the
full P2-G3 run once P1-G7 opens. Prereqs are all in place; the only remaining
steps need Modal access (corpus upload + `modal run`).

## Status of prerequisites
- ✅ Corpus (CNOF-generatable): `artifacts/datasets/compose_lipid_pretraining_v1/generator_corpus_cnof_v1.smiles` (429,335 SMILES; regen: `python3 scripts/export_generator_corpus.py --max-atoms 96`).
- ✅ Recipe (Lineage B): `recipes/lipid_unconditional_cnof_v1_smoke.json` (all 66 args validated).
- ✅ Local compile smoke: 100% (500/500 certified programs at `max_atoms=96`) — `scripts/smoke_lipid_path_compile.py`.
- ✅ Eval panel: `scripts/analyze_lipid_generator_panel.py` (P2-G3/L1 marginals).
- ✅ Region-aware model (optional enhancement): `compose_v4/lipids/region_aware_rate_model.py`.
- 🔒 Full training gated behind **P1-G7** (Codex Paper-1 unconditional + matched-QED proof). An **exploratory restricted-ring smoke is authorized now** (handoff §9).

## Step 1 — stage the corpus on a Modal volume
Generate a held-out reference (for the eval panel), then upload both to the `guacamol` volume (or a new `lipid-corpus` volume):
```bash
# held-out reference = last 5k SMILES not used for training (disjoint slice)
tail -5000 artifacts/datasets/compose_lipid_pretraining_v1/generator_corpus_cnof_v1.smiles \
  > /tmp/lipid_heldout_ref_5000.smiles
modal volume put guacamol \
  artifacts/datasets/compose_lipid_pretraining_v1/generator_corpus_cnof_v1.smiles \
  /lipid_corpus_cnof_v1.smiles
modal volume put guacamol /tmp/lipid_heldout_ref_5000.smiles /lipid_heldout_ref_5000.smiles
```

## Step 2 — point the launcher at the lipid corpus (minimal, backward-compatible)
In `modal_apps/train_tracelet_gm.py`, make three values lipid-aware. Cleanest as an
env-var default so the drug-like path is unchanged (default = current):
- `ROLLOUT_MAX_ATOMS` (line 44): `int(os.environ.get("COMPOSE_ROLLOUT_MAX_ATOMS", "40"))` → set `COMPOSE_ROLLOUT_MAX_ATOMS=96` for lipids.
- `train_file` (lines 917, 1503): `Path(os.environ.get("COMPOSE_TRAIN_SMILES", "/guacamol/guacamol_subset_500000_seed0.smiles"))` → set `COMPOSE_TRAIN_SMILES=/guacamol/lipid_corpus_cnof_v1.smiles`.
- `reference_file` (lines 918, 1504): `Path(os.environ.get("COMPOSE_REF_SMILES", "/guacamol/guacamol_heldout_val_5000_seed0.smiles"))` → set `COMPOSE_REF_SMILES=/guacamol/lipid_heldout_ref_5000.smiles`.
(Env vars must be visible inside the Modal functions — set them in the function `image`/`secret` or via `modal run` env; the app already mounts `/guacamol`.) **Shared file → flag to Codex in the sync; the change is a pure default-preserving wrap.**

## Step 3 — the 4 Modal stages (always `--detach`; `--recipe-name` needs `.json`)
Copy `recipes/lipid_unconditional_cnof_v1_smoke.json` into the launcher's recipe dir.
```bash
LB=lipid-smoke-<commit>-v1
# (a) compile teacher paths from the lipid corpus
modal run --detach modal_apps/train_tracelet_gm.py --compile-only \
  --recipe-name lipid_unconditional_cnof_v1_smoke.json --run-label $LB-paths
# (b) compile support
modal run --detach modal_apps/train_tracelet_gm.py --support-compile-only \
  --recipe-name lipid_unconditional_cnof_v1_smoke.json --run-label $LB-support \
  --source-run-label $LB-paths --support-steps 150 --support-containers 2 --support-workers 12
# (c) train (from scratch on the lipid corpus; Lineage B config)
modal run --detach modal_apps/train_tracelet_gm.py --train-only \
  --recipe-name lipid_unconditional_cnof_v1_smoke.json --run-label $LB-train \
  --source-run-label $LB-paths --training-steps 150 --schedule-steps 150
# (d) evaluate: sample + rollout
modal run modal_apps/evaluate_rollout_shards.py --run-label $LB-eval \
  --source-run-label $LB-train --checkpoint-name checkpoint.pt --samples 2000 --shards 50
```

## Step 4 — score the rollouts with the lipid panel (P2-G3 / L1)
Download the rollout SMILES, then:
```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src \
  python3 scripts/analyze_lipid_generator_panel.py \
    --generated <rollout>.smiles \
    --reference /tmp/lipid_heldout_ref_5000.smiles \
    --train artifacts/datasets/compose_lipid_pretraining_v1/generator_corpus_cnof_v1.smiles \
    --label lipid_smoke
```
Watch: 100% committed validity, uniqueness, exact/scaffold/**linker** novelty, and the
lipid architecture JS to the held-out reference (head/linker/tail, tail count/length/
branching, degradable-linker rate, protonation).

## Region-aware variant (measure the lift)
To train the region-aware model instead of the base, instantiate
`RegionAwareFactorizedTraceletRateModel(..., region_prior_table=RegionAwareFactorizedTraceletRateModel.prior_table_from_json("artifacts/datasets/compose_lipid_pretraining_v1/region_conditioned_prior_v1.json"))`
where the trainer builds the model (a small trainer switch), then compare the panel
marginals (region atom share, degradable-linker rate) vs the base run.

## BEAE fine-tune (Fig-6, after the base is trained)
- Substrate: `configs/lipid_reactions/beae_finetune_substrate_v1.json` (propiolate transform) + `scripts/enumerate_beae_substrate.py` (combinatorial head × tail × tail).
- Linker-frozen optimization: freeze `beae_linker_atoms(lead)` (`configs/lipid_reactions/beae_linker_v1.json`) as a hard legal-fiber condition; optimize head + tail1 + tail2. (Conditioning/Arm-B machinery is the other lane.)

## Gotchas (learned this session)
- `modal run --detach` always; `--recipe-name` needs the `.json`.
- New `--trainable-parameter-scope` values must be added to the argparse `choices` in `scripts/train_tracelet_cnof_gate.py`.
- `max_atoms=96` is ~5.8× the memory of 40 (dense n_slots² tensors) — cut `batch_size` if OOM (recipe starts at 16).
- Corpus is CNOF-only (the generative heads can't build S/P); S/P families (~7.6%) need the CNOSP kernel extension with Codex.
