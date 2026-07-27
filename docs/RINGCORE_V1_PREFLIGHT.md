# RingCore-V1 preflight — bank the compositional ring core

**Mandate:** freeze the supervised compositional ring core as a separately-versioned production candidate
`RING_CORE_V1` (`enable_cycle_ops=true`, `enable_ring_macros=false`, legacy `ring_system_grow` macro
**disabled**), run the bounded preflight, and return a macro decision. **Do NOT** implement P4 into this
production path. **Do NOT** launch the full A100 run — bounded preflight only (≤750 steps).

Ring reachability is defined by the compositional operators; a finite ring catalog is a LATER
(`RING_HYBRID_V2` / P4) acceleration layer, never the definition of support.

Status legend: ✅ done · 🔶 in progress · ⏳ pending.

---

## 1. Freeze — RING_CORE_V1 ✅
- **Git tag `ring-core-v1`** → commit `491121d` (manifest); frozen core code at `0016873`.
- **Capability hash `330473e319bfec19`** · operator-registry hash `9197401e8dc3a7ae` · cycle-op semantic
  hash `27a823aeb6cf7548` · ring-support-semantics-version `1`.
- Config: `enable_cycle_ops=True`, `enable_ring_grow_macro=False`, `enable_ring_macros=False`,
  `corrupted_prior_mix=True`, `organic_vocabulary=True` (15 classes), scope `3721d69851110fdd`, production
  ring catalog `639ff6078c32d43c`, base `c9d927…`.
- **Legacy grow disabled** (`enable_ring_grow_macro=False`): grow family masked dead — 0/400 draws vs 74/400
  enabled; grow head params retained (warm-start-safe, byte-identical when enabled). Ring ADDITION is purely
  compositional (`cycle_close`).
- **De-ambiguated public↔internal operator map** (internal `cycle_insert`/`cycle_attach` are engineering
  aliases, never exposed): public `cycle_close`(bond_insert) · `cycle_open`(bond_delete) ·
  `ring_delete`(whole removable-ring, decoration-preserving) · `ring_aromaticity_restate`. Deferred to P2:
  `ring_ear_insert`, `ring_spiro_attach`. Disabled: `ring_grow_macro`.
- **Latent Modal-crash blocker fixed:** the real training path (`FactorizedMarkDataset.__getitem__`) rejected
  the cycle-op executor teacher names (`bond_insert`/`bond_delete`) — the prepare-batch tests bypassed it. Now
  aliased to `cycle_insert`/`cycle_attach`. RingCore training smoke: loss finite, grad_norm 0.75. Full suite
  **530 passed**.
- Artifacts: `diagnostics/production_preflight/ring_core_v1_manifest.json`; `scripts/freeze_ring_core_v1.py`.

## 2. Ring-topology capability gate ✅ `GO_RING_TOPOLOGY_CAPABILITY`
Executable compositional construction (`cycle_close`) + deletion (`cycle_open`) per topology class, decomposed
via a spanning tree (non-tree edges = ring-closing bonds), with all 14 required properties.

**Topology matrix — 14/14 `SUPPORTED_BY_CORE`, 0 `REQUIRES_STRUCTURED_P2_MODE`, 0 `UNSUPPORTED`:**

| Class | ring bonds | Class | ring bonds |
|---|---|---|---|
| simple_saturated | 1 | fused_saturated (decalin) | 2 |
| aromatic_carbocycle (benzene) | 1 | spiro | 2 |
| aromatic_heterocycle (pyridine) | 1 | bridged_bicyclic (norbornane) | 2 |
| saturated_heterocycle (piperidine) | 1 | polycyclic_cage (adamantane) | 3 |
| macrocycle (12-ring) | 1 | sulfur_ring_aromatic (thiophene) | 1 |
| fused_aromatic (naphthalene) | 2 | sulfur_ring_saturated | 1 |
| phosphorus_ring | 1 | charge_preserving_ring (N⁺ + cyclohexane) | 1 |

Every class passed all of: valid start, legal compositional sequence, all-intermediate validity+
connectedness, ≤40 atoms every state, endpoint isomorphism, correct cycle-rank changes, charge preservation,
canonical-successor identity, exact inverse (open↔close), positive supervision.

**⇒ The mandate's P2 conditional is NOT triggered — the compositional core is support-complete across all
topology classes; no minimal structured P2 mode is required for support.** Fused/spiro/bridged/cage
construction is reachable compositionally (efficiency is measured separately in §3).

**Core mechanism (topology-agnostic, once over the union):** finite gradients ✅ (grad_norm 1.18) ·
forced sampling executes ✅ · natural sampling after a 120-step overfit ✅ (**100%** of draws on precursor
states are cycle ops) · save/reload equivalent ✅. Artifact:
`diagnostics/production_preflight/ring_topology_capability.json`.

## 3. Compositional path-cost study ✅
The pilot analogue pool has **no** ring-changing pairs (53 pairs, all `variable_ring_atoms=0`, atom_insert/
delete only), so path cost is measured directly on **800 real held-out GuacaMol ring systems** via the
executor-verified spanning-tree decomposition (ring-system ↔ ring-opened precursor is the canonical
ring-changing transformation; its compositional edit length is exactly the cycle_close/cycle_open count).
**100% compilation success**, max temp atoms 40 (bound respected). Budget = 16 (sampler horizon).

Two costs (`diagnostics/production_preflight/ring_path_cost.json`):
- **Ring restructuring** (cycle_close/open when the ring atoms are present — the common lead-opt ring edit:
  open/close/reshape/fuse): median **3**, p75 4, p90 5, **max 7**. **86.6% ≤4 edits, 100% ≤8, 0% exceed the
  budget.** Cheap for *every* topology (per-topology medians 2–4.5, max 7 incl. fused/spiro/bridged/aromatic/
  hetero). ⇒ the whole-ring macro adds little to ring restructuring.
- **Ring introduction** (build a whole new ring system from fresh atoms = atom_inserts + closures — the
  UPPER bound a macro collapses to ~1 step): median **20**, **73.5% exceed the budget** (fused/hetero 24 /
  93.6%, spiro 26 / 100%, monocarbocycle 14 / 15%). ⇒ de-novo whole-ring construction is where a macro
  would help — but that is a de-novo move, not the dominant lead-editing operation.

**Read for the macro decision:** restructuring existing rings (the editing regime's ring work) is
compositionally cheap and always within budget; only building a *whole new* fused/hetero ring de-novo is
expensive. Whether that bites RingCore in practice depends on how often trained ring edits are whole-ring
introductions vs restructurings — measured in §6.
## 4. P5 calibration for the compositional core ⏳
## 5. Regenerated RingCore manifest + cache hashes ⏳
## 6. Bounded RingCore preflight (≤750 steps) ⏳
## 7. Unseen-lead + topology-stratified rollouts ⏳
## 8. Measured throughput ⏳
## 9. Projected full-run cost ⏳
## 10. Decision ⏳
`CORE_SUFFICIENT_FOR_FULL_RUN` / `MACROS_REQUIRED_BEFORE_FULL_RUN` / `MACROS_OPTIONAL_ABLATION` /
`NO_GO_RINGCORE_CORRECTNESS`.

## Paper position (§9)
The method is written around the compositional ring core; the theoretical support claim does NOT depend on a
finite macro dictionary. Macros, if later adopted (`RING_HYBRID_V2`), are optional learned shortcuts for
frequent ring transformations. The compositional-only model is the **support-complete** variant, not a
reduced-capability one — macros affect efficiency and finite-budget reachability. Required ablation:
compositional-only / catalog-only / compositional+macros.
