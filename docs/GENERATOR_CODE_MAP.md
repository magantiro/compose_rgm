# Generator code map — the lipid port (synthesized from a full reading pass)

My working reference for porting the COMPOSE generator to lipids. Complements
`GENERATOR_INTERNALS_FROM_ZERO.md` (concepts) with exact file:line seams + the
lipid-specific findings. Verified against the tree on 2026-07-20.

## 0. Mental model (one line)
A CTMC over molecular graphs: seed = carbon tree; a factorized GNN
(`FactorizedTraceletRateModel`) predicts a **rate** for every legal rewrite;
`rate = total_hazard × P(family) × P(action | family)` over 10 families; the
executor/fiber decides legality (the net only shapes probability inside the
legal set); trained by Poisson-Bregman Generator-Matching on certified teacher
edit-programs compiled from the corpus. **Judge on rollouts, not loss.**

## 1. The pipeline + exact wiring (corpus → sample)
1. **Corpus** = a plain **SMILES file**, passed as the trainer's **positional arg**
   (`argv[0]`), NOT a recipe key. The launcher hardcodes it:
   `modal_apps/train_tracelet_gm.py:917` (`/guacamol/...500000...smiles`) and the
   rollout copy at `:1503`. Recipe (`recipes/*.json`) carries only hyperparameters
   (`experiments/recipe.py:31` emits `argv[0]=smiles_file` then `--flag value`).
2. **Loader** `data/cnof.py:35 load_cnof_corpus_split`: reads `line.split()[0]` per
   line, canonicalizes, dedups, **filters `_in_scope` (`:193`): neutral + connected
   + `n_real_atoms ≤ max_atoms` + all atoms ∈ {C,N,O,F}**, then splits train/val/test.
3. **Compile paths** (`--compile-only`): `compile_carbon_tree_to_target`
   (`rewrite/tree_transport.py:48`) builds a certified seed→target `RewriteTrace`
   per (coupling, target), replay-verified. `flexible_size_graft` resizes the seed
   to the target then couples via quotient-Graft + `atom_restate` (heteroatoms) +
   one `ring_system_grow` per ring.
4. **Compile support** (`--support-compile-only`) → per-step legal-action tensors.
5. **Train** (`--train-only`): `forward_mark_batch` → `factorized_mark_bregman_loss`
   (`model:4006`), AdamW, grad-clip 10.
6. **Eval** (`evaluate_rollout_shards.py`): sample N, metric panel
   (`analyze_unconditional_sufficiency.py`, swap the lipid reference).

## 2. THE CNOF GENERATION LIMIT (defines v1 scope) — most important finding
The encoder can *represent* 12 elements (incl. S/P, with lipid valences,
`molecular_graph.py:77-90`), but every **generative** head is hardcoded to
`CNOF_ATOM_TYPES=(C,N,O,F)` (4): `grow_root_head`, `grow_option`, `restate_head`,
`ring_system_atom_head`, + all 5 empirical-prior tables. The seed is pure carbon;
atoms enter only by insertion (CNOF-only). **⇒ the model can read but cannot
generate S/P.** So S/P-containing teachers won't compile (`atom_restate` can't
install S/P). Widening to CNOSP = coordinated shared-core change: `CNOF_ATOM_TYPES`
(`factorized_fiber.py:51`), every dependent head + valence/H masks, empirical
prior regen, and the two hard gates `_in_scope` (`data/cnof.py`) +
`_assert_supported_state` (`factorized_fiber.py:329`). **Coordinate with Codex.**

- **v1 trains on the CNOF subset (~82% of the corpus)** — amine heads, ester/amide/
  ether/aza-Michael/Ugi/Passerini/reductive-amination/epoxide. **BEAE is pure C/N/O
  (amine + acrylate + propiolate enamine-ester) → fully CNOF-generatable, unaffected.**
- **Deferred to v2 (CNOSP widening):** disulfide/thioether (S) + iPhos phosphate (P)
  ≈ 18% (the S/P degradable-linker families).

## 3. Where each lipid change goes (precise punch list)
| Change | Seam | Owner |
|---|---|---|
| Corpus | export a **plain SMILES file** (CNOF subset, realism-weighted); point launcher `train_file` `:917`+`:1503` at it + mount Modal volume | lipid lane |
| Size 40→96 | recipe `max_atoms=96` (no upper bound, `train_tracelet_cnof_gate.py:686`) **AND** launcher `ROLLOUT_MAX_ATOMS=40→96` (`train_tracelet_gm.py:44,:512`) — else eval truncates | lipid lane |
| Restrict rings to 5/6 | **automatic**: catalog is corpus-derived (`typed_ring_catalog.py:380`); ring-clean corpus ⇒ ring-clean catalog. Optional hard filter: `span∈{5,6}`+`single_ring` in `_finalize_typed_ring_catalog:548`. Recipe: `ring_template_factorization` off | lipid lane |
| Base config | `empirical_mark_prior_mode=corpus_residual_v1`, `ring_electronic_mode=factorized_local`, `source_prior=carbon_tree`+`tree_size_prior=empirical`, `teacher_ordering=sequential` | lipid lane (recipe) |
| Sampler knobs | `atom_delete_log_rate −0.5`, `small_ring −1.5` (thinning, `calibrated_rewrite_sampling.py`) | lipid lane |
| CNOSP elements (v2) | `CNOF_ATOM_TYPES` + heads + masks + priors + 2 hard gates | **shared → Codex** |

## 4. Oracle seam (Arm A/B) — for the pan-lung oracle
- **Interface:** `StateScorer = Callable[[MolecularGraph], float]`
  (`griddd_conditional.py:65`, `guided_rewrite_sampling.py:29`) — single state,
  scalar, **not batched**. Replace `qed_state_oracle` (`griddd_conditional.py:1040`)
  or pass `scorer=` to `GridDDConditionalEvaluator` (`:1897`).
- **Adapter:** `def pan_lung_oracle(state)->float`: `molecular_graph_to_smiles(state)`
  → `OracleNominator` → scalar delivery score. Scale to [0,1] or relax the
  `GridDDProtocol` QED-range asserts (`:723,:727,:1941`).
- **Arms:** direct = Arm A (conditioned rates, oracle doesn't steer selection);
  controller = **Arm B** (base rates + oracle tilt, `ExactBudgetLeadGuidedSampler`);
  combined = both. Exact matched oracle-call budget with padding (`:915-1037`).
- **Linker preservation = HARD:** a protected-atom mask in the **fiber** (removes any
  edit touching linker atoms before scoring; survives by construction), riding a
  protected-set tag on the state from `FixedMolecularStatePrior`. Fiber = shared core
  → Codex. (`RewriteSystem.constraints` in `kernel.py` is a correctness fallback.)
  Current preservation is only SOFT (similarity penalty + post-hoc Tanimoto 0.40).

## 5. Conditioning & the weak-steering fix
- Classifier-free: `condition_dropout_probability=0.15`; value standardized
  (`PropertyConditionNormalizer`) + fed via `sample_rewrite_mark_conditioned`
  (`property_values,property_mask`), zero-init encoder added to context → all heads.
  Add the pan-lung property to `PROPERTY_FUNCTIONS` (`molecular_property_conditioning.py:21`).
- **Why the sidecar was weak:** trained by **teacher imitation; the oracle never
  enters the loss** → no gradient toward high score; can't extrapolate past the
  teacher marginal (0/5, degraded QED). The doc's "loss ≠ rollouts" in the flesh.
- **Fix order:** (A) run **Arm B controller at full budget** (only positive arm; was
  under-budgeted 4 vs 48) — cheap, mostly built. (B) **reward-FT under a KL anchor**
  (fine-tune CTMC toward oracle, anchored to frozen base) — a *build*; seam = the
  sidecar's differentiable rate table + frozen base as anchor.

## 6. Fine-tune surface (for the Michael/BEAE linker, Fig 6)
- **Trainable scopes** (`configure_factorized_trainable_parameters`,
  `factorized_mark_conditional.py:1302`): `chemistry_marks_only`, `ring_topology_only`,
  `chemistry_and_topology` — none unfreeze family/Graft/hazard (deliberate, avoids
  drift). Freeze base, train local chemistry marks for a linker fine-tune.
- **Frozen residual sidecar** (`FrozenExactRDKitResidualAdapter`,
  `canonical_successor_distillation.py:1107`) = cleanest property-conditioned template;
  add the property key (currently `{qed,molecular_weight,crippen_logp}`).

## 7. Open risks / to verify at smoke
- **max_atoms=96 = ~5.8× memory** (dense `n_slots²` tensors; `graft_features` biggest)
  → cut batch size / hidden_dim; lipid trees always hit O(n²) Graft canonicalization
  on the CPU compile path (slow but not broken).
- **`ROLLOUT_MAX_ATOMS`** must be bumped too (else eval truncates at 40).
- **`MAX_H_COUNT=4`** drops teachers needing transient H>4 (`compiler.py:139`).
- **Stereo:** corpus is 88% stereocenters — verify whether the kernel preserves or
  strips stereo (bond_representation=aromatic; likely stereo-agnostic → declare that).
- **Caches** are content-addressed on max_atoms/ring/source/transport → new corpus
  forces a full path recompile (expensive CPU stage).

## 8. v1 build order (all lipid-lane except CNOSP)
1. Export CNOF-subset realism-weighted SMILES file from the corpus.
2. Lipid recipe (max_atoms=96, restricted rings, base config) + a lipid-scope note
   (or keep CNOF loader as-is for v1 — it correctly matches the CNOF kernel).
3. Wire launcher: corpus path + `ROLLOUT_MAX_ATOMS=96` + Modal volume.
4. Restricted-ring smoke at max_atoms=96 (compile→support→few train steps→eval);
   tune batch size for the 5.8× memory. (Authorized now; full training gated by P1-G7.)
5. Arm A (oracle ranking) with the pan-lung adapter; then Arm B controller at full budget.
6. v2: CNOSP head-widening with Codex; then S/P families + the Michael/BEAE fine-tune.
