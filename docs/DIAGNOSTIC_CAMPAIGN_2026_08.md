# Search-geometry diagnostic campaign — August 2026

Closed out before the pivot to **variable-scope executable region resampling**.
Everything here is CPU-only, diagnostic, and deploys nothing. No `R_theta`
retraining, no `carbon_rich` patch, no macro built.

## What was established

| # | Finding | Artifact |
|---|---|---|
| 1 | **T4 is refinement-limited, not entry-limited.** Both COMPOSE and IVG leave the seed's coarse stratum on growth cells; IVG grows *more*. The gap is one ring's composition. All-carbon share of added rings: ours 83%, IVG 51%. | `diagnostics/frozen/composition_mass_audit.FROZEN.json` |
| 2 | **Composition is a controller failure, not `R_theta`.** At true mid-chain growth states `R_theta` *upweights* N 2.19x over support. The frozen `carbon_rich` spec captures 0.284 of the step's mass where `mixed` captures 0.955. | same |
| 3 | **Exposure alone is not enough.** Opening the stoichiometry space took hetero-ring endpoints 0/48 -> 29/48 at matched budget, but T4 feasibility was flat at delta=0.4 and *worse* at delta=0.6 (similarity floor). | `diagnostics/frozen/stoich_realization_arm.FROZEN.json` |
| 4 | **Task-aware `H_z` steers but cannot rescue.** 174/176 picks changed, SA<=4 passing 100 -> 131, yet feasibility did not improve. | `diagnostics/guided_realization_delta{0.4,0.6}.json` |
| 5 | **The option was wrong, not the steering.** 20 of 22 declared dev cells have **zero** feasible endpoints anywhere in the growth basin — narrow (245 unique endpoints) or open (16,784). | `diagnostics/pool_ceiling_audit.json` |
| 6 | **Ring expansion: supported but mass-starved.** 7->8 expansion reachable in 2-3 valid primitive edits with reference path probability 1e-8 to 1e-14. Not an operator gap; a kinetic one. | `diagnostics/frozen/ring_expansion_support.FROZEN.json` |
| 7 | **PMO contains both regimes** (target-free half). 6/19 tasks concentrate >=70% of their top-10 in one sigma0 stratum; 7/19 disperse below 40%. | `diagnostics/pmo_banks_all.json` (analysis only) |

## Apps

| App | Purpose |
|---|---|
| `modal_apps/composition_mass_audit_app.py` | support -> R_theta -> Q_spec -> semantic completion, at true chain-growth states |
| `modal_apps/stoich_realization_arm_app.py` | matched 3-arm composition-exposure comparison |
| `modal_apps/guided_realization_arm_app.py` | `select_endpoint(value_fn=H_z)`; task-aware realization selection |
| `modal_apps/pool_ceiling_audit_app.py` | feasibility ceiling + selection regret over the frozen pools |
| `modal_apps/ring_expansion_support_app.py` | is winner-like expansion reachable, and at what path mass |
| `modal_apps/multiscale_atlas_app.py` | measured structural displacement + utility per lane, k=1..3 + open-ended lane |
| `modal_apps/option_basin_atlas_app.py` | superseded by `multiscale_atlas_app.py` |

## Methodological rules that came out of this

- Aggregate successor fibers **before** any claim about `R_theta`; mark-level
  element histograms measure the (element,valence) encoding, not chemistry.
- A composition probability is a sum over complete realization paths, never a
  product of per-step marginals.
- Develop on `docs/GENMOL_T4_DEV_SEEDS.json` (canonical-SMILES disjoint from the
  15 benchmark seeds), never on `GENMOL_T4_SEEDS.json`.
- `trace_ok` is the compiler's claim; `semantic_ok` must independently verify the
  graph, INCLUDING the requested electronic state.
- Open arms buy extra cheap search — always report candidate counts, unique
  canonical endpoints, kernel calls, and wall time alongside any win.

## Known-defective components, do not reuse

- `rtheta_semantic_prior` scores the SIZE of a spec's element set
  (permissiveness), not plausibility. Anti-correlated with what works.
- `COMPOSITION_CODES` has no key for an exact label like `C5N1` and silently
  falls back to `carbon_rich`. Use `parse_stoich_label` / the `stoich` quota.

## Superseded by the pivot

The exhaustive option-basin atlas is **not** the next step. The next architectural
object is generic variable-scope executable region resampling, with semantic
macros as proposal shortcuts for measured kinetic bottlenecks (finding 6).
