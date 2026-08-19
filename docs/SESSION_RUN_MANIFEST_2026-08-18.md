# Run manifest — 2026-08-18 session

Every experiment run in this session: the app, the exact invocation, where the
artifact landed, the result, and the commit. Written so a reader can re-run any
line without reconstructing it from logs.

**Two storage locations, and the distinction matters for reproduction.**
Small result JSONs are committed under `docs/`. Bulk artifacts (corpora,
embeddings, per-run records, trained heads) live on the Modal volume
`compose-v4-artifacts` under `editing_v2/r_theta_run/` and are **not in git**.
A fresh clone can re-run the apps but cannot read those results without the
volume.

---

## Frozen assets used throughout

| asset | location | note |
|---|---|---|
| R_theta checkpoint | volume `runs/run_v2_01/` | never retrained, any session |
| H24 h_phi head | volume `hphi_v2/head.pt` | sha256 `9ea51ec4…`, verified byte-identical before and after this session's training |
| H24 norm | volume `hphi_v2/norm.json` | sha256 `4f93f7ec…` |
| Task-3 oracle bundle | `artifacts/oracles/molleo_task3_v1/` | in-repo, SHA-pinned manifest |
| DRD2 oracle | `artifacts/oracles/drd2_svm_v1/` | in-repo |
| dev init sets | `artifacts/benchmarks/molleo_task3_init_v1/` | in-repo, seeds 100–107 |

---

## QED / GrIDDD line

| # | app / script | invocation | artifact | result |
|---|---|---|---|---|
| 1 | `scripts/hphi_budget_max_regression.py` | local | — | budget_max=24 reproduces frozen assembly **bitwise** (Xtr/Ytr/Btr/Mva) |
| 2 | `hphi_rollout_corpus_app.py` | `--limit 256 --trajectories 2 --horizon 40` | `hphi_rollout_corpus/train_0256x02_H40.json.gz` | 20,480 transitions (42% of frozen budget) |
| 3 | `hphi_encode_app.py` | `--corpus …H40 --shard-dir hphi_v2/embeddings_h40 --n-shards 24 --reuse-dir hphi_v2/embeddings` | 24 shards | 15,872 states → 5,869 new after reuse |
| 4 | `hphi_train_app.py` | `--budget-max 40 --emb-dirs "hphi_v2/embeddings,hphi_v2/embeddings_h40" --out-dir hphi_v2_h40` | `hphi_v2_h40/` | val Brier 0.0545 vs constant 0.0750; 342,043 train rows |
| 5 | `hphi_prospective_two_head_app.py` | `--stage extract` then `--stage score` | `docs/PROSPECTIVE_TWO_HEAD.json` | parity 1.49e-07; transition-level AUC old/new |
| 6 | `hphi_longrange_auc_app.py` | default | `docs/LONGRANGE_AUC.json` | H40 head worse at every lookahead 4–40 |
| 7 | `hphi_h40head_ab_app.py` | default (12-source panel) | volume `hphi_h40head_v10/` | decision strata 4/8 → **3/8**; H40 head not adopted |
| 8 | `hphi_monotonicity_localize_app.py` | default | `docs/MONOTONICITY_LOCALIZE.json` | violations NOT concentrated in failures; Spearman 0.996 under isotonic projection |
| 9 | `hphi_multiplicity_app.py` | default | `docs/PARTICLE_MULTIPLICITY.json` | 94.7% of particle-states already unique; ceiling 1.13× on hard |
| 10 | `hphi_shortlist_retention_app.py` | needs `docs/SHORTLIST_TASKS.json` | `docs/SHORTLIST_RETENTION.json` | 0/21 hard rescues inside top-128; median worst rank 271 |
| 11 | `hphi_h40head_ab_app.py` | `--k-start/--k-end` slices 4-6, 6-10, 10-20 | volume `hphi_recede_v5_k*/` | dev curve @1–@20 → `docs/EXTENDED_CURVE_64.json` |
| 12 | `hphi_h40head_ab_app.py` | `--valid --k 8`, then `--valid --valid-from 8 --k 12` | volume `hphi_valid128_k8/`, `hphi_valid128_k912/` | **@8 49.2%, @12 54.7%** on 128 held-out → `docs/VALID128_K8_RESULT.json`, `docs/VALID128_CURVE.json` |

**Headline:** 54.7% at 12 candidates on a prospectively held-out 128-source panel
vs GrIDDD's 45.1% at 20. Controller frozen beforehand in
`docs/AMENDMENT_VALIDATION_128.md`. Development curve reached 47/64 = 73.4% @20.

**Parity checks that gate these numbers** (re-run before trusting any of them):
- slice parity 64/64 — a k=0..1 slice reproduces banked candidate 1 exactly
- `hphi_recede_v5/005_H40.json` reproduced 4/4 identical returned SMILES

---

## MOLLEO line — all four search branches closed

| # | app | artifact | result |
|---|---|---|---|
| 13 | `molleo_env_gate_app.py` | `docs/MOLLEO_ENV_GATE.json` | pins held; all 5 oracles score; TDC QED == our RDKit QED exactly. **Superseded** — see note below |
| 14 | `molleo_rtheta_parity_app.py` | volume `molleo_rtheta_parity/` | R_theta reproduces a banked QED candidate on the TDC image |
| 15 | frozen cohort | `docs/MOLLEO_DEV_COHORT.json`, `docs/MOLLEO_DEV_LABELS.json` | seed 100, 120 molecules, 24 roots by sha256(smiles) |
| 16 | `molleo_lazy_gate_app.py` | volume `molleo_lazy/` | L1 vs L4 did not diverge; HV moved +0.5%; **0 resampling events** |
| 17 | `molleo_basin_substrate_app.py` | `docs/MOLLEO_BASIN_ANCHORS.json` | 42/96 reverse-verified routes; one-step fiber max median 0.365 |
| 18 | `molleo_fiber_census_app.py` | volume `molleo_fiber/` | **0/24 random-ZINC fibers contain JNK3 ≥ 0.3**; max 0.16 over ~16,000 successors |
| 19 | `molleo_bridge_probe_app.py` | `docs/MOLLEO_BRIDGE_PAIRS.json` | target supplied: similarity 0.21→0.51, JNK3 to 0.37, none reached 0.5 |
| 20 | `molleo_coverage_sentinel_app.py` | volume `molleo_coverage/` | 2× scaffolds, no JNK3 gain, moved *further* from actives |
| 21 | `bridge_train_app.py` | volume `bridge_v1/` | goal-conditioned reachability, zero task-oracle calls; top-10 61.5% at b=24 (51× chance) |
| 22 | `bridge_navigation_gate_app.py` | volume `bridge_nav/` | **null**: 0 exact hits both arms, Δsim +0.006, sign test 28/50, p=0.24 |

**NOTE on #13/#14 — a dependency that was later removed.** The TDC image
(PyTDC + an `rdkit.six` shim + scikit-learn pinned to 1.2.2) was built before
discovering that `artifacts/oracles/molleo_task3_v1/` already provides the same
oracles as SHA-pinned `.npz` with no TDC or sklearn at runtime. **Everything
after #14 uses the frozen bundle on the untouched base image.** The TDC apps are
kept for the record; do not reintroduce that dependency.

---

## Preregistrations written before their data

- `docs/AMENDMENT_H40_HPHI.md`
- `docs/AMENDMENT_VALIDATION_128.md`
- `docs/AMENDMENT_SHORTLIST_RETENTION.md`
- `docs/AMENDMENT_MOLLEO_LAZY_GATE.md`

## Known reproducibility gaps in this session's own work

1. `docs/SHORTLIST_TASKS.json` was generated by an inline script that read
   `/tmp/rv5`, `/tmp/k56`, `/tmp/k1120` — local caches of volume records. The
   inputs are on the volume; the assembly step is not scripted in the repo.
2. Several analyses ran as inline `python3 -c` rather than committed scripts.
   `scripts/hphi_extend_curve_read.py`, `scripts/hphi_valid128_read.py`,
   `scripts/hphi_valid128_curve.py`, `scripts/hphi_shortlist_read.py` were
   committed; the ad-hoc ones were not.
3. The kinase actives used for MOLLEO basin anchors come from a path inside an
   ephemeral session scratchpad, not from the repo. See
   `docs/REPRODUCIBILITY_HAZARDS_2026-08-19.md`.
