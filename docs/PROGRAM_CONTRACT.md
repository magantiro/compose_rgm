# Program Contract — the ONE canonical scientific program (RingCore-V1)

Authoritative production system, derived from committed code (HEAD 4204c9d), not memory.
**Production contract fingerprint: `3917a0bf4a3de9830722663f`** (`scripts/coherence/program_contract.py` →
`diagnostics/coherence/program_contract.json`). Entry point: `train_tracelet_cnof_gate.py` via
`modal_apps/train_tracelet_gm.py` with `--corrupted-prior-mix --cycle-op-mix --disable-ring-grow-macro
--organic-vocabulary --scaled-manifest ... --train-only`.

## 1. State space
Connected molecular graphs; **ORGANIC_VOCABULARY** (C/N/O/F/S/P/Cl/Br/I/B, 15 (element,valence) classes);
**max_atoms=40** (asserted `gate:1657`; `organic_corpus.py:143`); charge-preserving (`retain_representable_
charges`); stereo/isotope dropped, radical/bad-valence → unsupported class; validity via `is_valid_state` +
`is_connected_or_null` (`chem/state.py`); padded-slot (slot-stable) states. Scope hash `3721d69851110fdd`.

## 2. Reference model
`FactorizedTraceletRateModel` (`model/factorized_tracelet_rate_model.py:1639`), warm-started from base **B**
(`c9d927…`, CNOF de-novo) via `--initialize-compatible-from-source-checkpoint` (semantic partial transfer:
shared rows copied by (element,valence), new rows fresh). Input `(x, t)` (current state + frozen time); output
total hazard Λ_θ + factorized mark distribution p_θ(a|x,t). Source-agnostic universal edit prior (source =
initial condition only). hidden_dim 256, mp_steps 6.

## 3. Editing process
Source = corrupted real molecule (`source_corruption.make_edit_pair`); fixed-step embedded molecular jump
chain (editing discards hazard, walks a fixed edit budget — de-novo uses rates/holding-times); canonical
molecular-successor aggregation (`canonical_state_key`); **no carbon-tree initialization** (zero-mixture:
`denovo_keep=0`, `tree_source_prior` never built under `--scaled-manifest`).

## 4. Ring process
RingCore-V1 compositional cycle support: **cycle_close** (`bond_insert`) + **cycle_open** (`bond_delete`),
support-complete (14/14 topology classes). Legacy `ring_system_grow` macro **DISABLED** (masked dead, params
retained for warm-start). `enable_ring_macros=False`. **Ring GENERATION is compositional (catalog-independent);
ring DELETION (`ring_system_delete`/`ring_delete`) IS catalog-bounded** (`enumerate_clean_ring_system_deletes
(state, catalog)`, 5-seed catalog `639ff6078c32d43c` → limited to the seed topologies) — see R1. Restate
(`ring_aromaticity_restate`) is catalog-independent.

## 5. Operator process
9 production-enabled families (see `OPERATOR_ONTOLOGY.md`): atom_insert/delete/restate, bond_reorder, graft
(bond_reroute), **cycle_close** (slot cycle_insert), **cycle_open** (slot cycle_attach), ring_delete
(ring_system_delete), ring_aromaticity_restate (ring_system_restate). `ring_system_grow` DISABLED. Internal
slot names cycle_insert/cycle_attach are engineering aliases — public = cycle_close/cycle_open. One executor
(`RewriteSystem.apply`, `kernel.py:51`), one canonical successor (`canonical_state_key`, `kernel.py:95`).

## 6. Data recipe
Scaled edit manifest `ring_core_v1_scaled_manifest.json`: layer weights **corruption 0.55 / mmp 0.45**,
denovo 0.0; curriculum bin edges [5,9,13]. Corruption regenerated in-memory (serial, ~0.5s/sample — see the
full-run precompile prerequisite); MMP pool `edit_pool_full.jsonl` loaded from volume; cycle-op supervision
folded into corruption. **Teacher-representability filter** v1 (`f8137995181eeadc`): every teacher ∈ the exact
dynamic candidate set for its state (drops ~2% unrepresentable grow-fused inverse restates at the data source).
Validation = held-out edit records from `split.validation` (disjoint from `split.train`). Scaffold-disjoint
split; teacher-in-candidate invariant enforced training + eval.

## 7. Warm-start
Body / atom_embedding / total_hazard COPIED_EXACT; shared CNOF (element,valence) rows copied semantically;
new S/P/Cl/Br/I/B rows fresh; cycle heads fresh; `ring_system_template_key` fresh (catalog mismatch);
calibration policy `f534c0233e3bf4a0` (diagnostic, not applied to optimizer); all trainable params in the
optimizer.

## 8. Control interface
Canonical-successor kernel P_θ(y|x,t)=Σ_{a:T(x,a)≅y} p_θ(a) as the substrate; source/objective-conditioned
controllers (`griddd_value_guided_smc_controller.py`, V0 immediate-reward deployed; V1 twist / V2 lookahead
experimental) operate on the embedded chain with a remaining edit budget. Exact Doob control = toy-only.
Editing/anytime-Pareto controllers = EXPERIMENTAL (separate subsystem, not the RingCore training contract).

**Hashes:** operator-registry `9197401e8dc3a7ae` · capability `330473e319bfec19` · cycle-op semantic
`27a823aeb6cf7548` · scope `3721d69851110fdd` · eval-capability `5bedd317328c9ed7` · teacher-filter
`f8137995181eeadc` · base-B `c9d927…` · ring-catalog `639ff6078c32d43c`.
