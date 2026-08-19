# Reproducibility hazard audit — 2026-08-19

**Scope** `modal_apps/`, `scripts/`, `src/`, `docs/` (plus `configs/` where a config
carries the same defect). **Repo state** `20d6ca4`. **Method** every claim below was
checked against the file and, where it is an existence claim, against the filesystem
at audit time. Nothing was modified; this document is the only file created.

**Standard used throughout.** The project already contains the right answer to each
hazard class. Class 1's answer is `src/compose_v4/data/durable_path.py`
(`require_durable_path`, which raises `ReapablePathError` on an OS-reaped prefix).
Class 3's answer is `artifacts/oracles/molleo_task3_v1/molleo_task3_oracle_manifest.json`
(SHA-256-pinned frozen bundle) and `scripts/build_griddd_qed_lead_manifest.py`
(upstream commit + expected file SHA-256, verified at read time). Class 4's answer is
`docs/HPHI_SMC_REFERENCE_BANKED.json` (`provenance.git_commit`, `mechanical_qualification.script`,
`artifact_sha256`). Class 5's answer is
`src/compose_v4/experiments/griddd_conditional.py:2052`, whose handler records the
error into the result object and leaves the failure in the denominator. The findings
are places that fall short of the project's own existing standard, not of an
imported one.

---

## Severity summary

| # | Class | Critical | High | Medium | Noted / intentional |
|---|---|---|---|---|---|
| 1 | Hardcoded ephemeral paths | 2 | 3 | 4 | 5 |
| 2 | Data dependencies not in the repo | 2 | 3 | 3 | 2 |
| 3 | Unpinned external dependencies | 1 | 1 | 2 | 1 |
| 4 | Results without provenance | — | 15/15 sampled | 18 orphans | 13 partial |
| 5 | Silent-failure patterns | 3 | 5 | 4 | 3 |

---

# Class 1 — Hardcoded ephemeral paths

`src/compose_v4/data/durable_path.py:42-46` defines the reapable prefixes:
`/private/tmp`, `/tmp`, `/private/var/folders`, `/var/folders`. `require_durable_path`
is imported by exactly six modules — `scripts/editing_v2_build_prepared_manifest.py:42`,
`scripts/editing_v2_freeze_v2_dataset.py:54`, `scripts/editing_v2_pack_collated_store.py:38`,
`scripts/editing_v2_consolidate_library.py:42`, `scripts/editing_v2_freeze_eval_panel.py:49`,
`src/compose_v4/data/corpus_training_library.py:68` — all in the editing_v2 lane.
Every finding below is code that hardcodes a prefix the gate exists to reject, without
calling the gate.

### CRITICAL

**`scripts/task3_offmanifold_check.py:50-52`** — the known example, confirmed.
```
KINASE_TSV = ("/private/tmp/claude-502/-private-tmp-compose-process-v2-atom-delete/"
              "4b3cfb50-af96-4de4-be72-c9b11493e0b6/scratchpad/hngfn_repo/oracle/"
              "scorer/kinase_rf/kinase.tsv")
```
A per-session scratchpad path, session UUID and all. The file is present *right now*
(6,228,796 bytes, mtime 2026-08-12) only because that UUID is the currently live
session. `find . -name "kinase*.tsv"` returns nothing — the file exists nowhere in the
repo, and the frozen oracle bundle vendored the three `.npz` payloads but **not** this
TSV. The moment the session ends, `load_kinase()` at `:96` raises `FileNotFoundError`
and the "training-set drift" half of the off-manifold analysis — the sharper of the two
questions, per the module docstring at `:24-30` — is permanently unrunnable.

**`scripts/editing_v2_local_compile.py:52-55`** — worse than a data path, because it is a
*code* path:
```
ROOT = Path("/private/tmp/compose-t1-collated-cache")
for _p in (ROOT / "src", ROOT / "tests", ROOT / "scripts"):
    ... sys.path.insert(0, str(_p))
```
`ROOT` exists at audit time. `sys.path` is mutated at import, so this script runs
`src/` from a reapable worktree rather than from the repo. This is the exact failure
mode `docs/MASTER_PLAN_EXPERIMENTS_AND_COMPARATORS.md:628-643` records — a pilot passing
from a `/private/tmp` worktree while the real run mounted a branch missing the module,
0/5,984. Here the divergence is silent: no gate compares this tree to the repo tree, and
`src/` is covered by the process-identity hash while this shadow copy is not.

### HIGH

**`scripts/hphi_extend_curve_read.py:5, 24`** — the producer of `docs/EXTENDED_CURVE_64.json`.
```
BANK = Path('/tmp/rv5/hphi_recede_v5')
...
json.load(open(BANK/f'{i:03d}_H40.json'))['arms']['restart']['candidates']
```
Note the asymmetry that makes this a real hazard rather than a caching convention:
the same file defines `get()` at `:8-12` (`modal volume get compose-v4-artifacts …`)
and uses it for `/tmp/k56` at `:32` and `/tmp/kpar` at `:47`, so those two are caches
with a re-fetch path. `BANK` has **no** `get()` fallback and is read directly. It
currently holds 128 files. When `/tmp` is reaped, `docs/EXTENDED_CURVE_64.json` — a
banked k=12 coverage result — becomes unreproducible, and the JSON itself records
nothing about where its inputs came from (see Class 4).

**`modal_apps/train_qed_frozen_residual.py:15-16`** — both **MISSING** now:
```
LOCAL_CHECKPOINT = Path("/private/tmp/pancake_checkpoint/checkpoint.recovery.pt")
LOCAL_PARTITION  = Path("/private/tmp/compose_v4_stage3_paths_manifest.pt")
```
This is a *training* app; its base checkpoint and its frozen path partition are both
gone. `docs/GENERATOR_LINEAGE_MAP.md:461` already flags this ("`/private/tmp` is wiped
on reboot; the shipped base…") and it has not been acted on.

**`configs/experiments/griddd_qed_frozen_residual_launch_v1.json:8, 44, 51`** — the
committed *launch config* for that app hardcodes the same dead paths:
`retained_checkpoint_path`, `evidence_path` (`/private/tmp/qed-step500-target090-metrics.json`,
MISSING), `training_partition_manifest`. A launch config is the record of what a run
consumed; here it points at three files, all absent.

### MEDIUM — the Lineage B checkpoint cluster

`/private/tmp/lineage_b_checkpoint/checkpoint.best_so_far.pt` is **MISSING**, and is the
hardcoded default of eight scripts:

| file:line |
|---|
| `scripts/property_dial.py:43` |
| `scripts/physchem_box.py:49` |
| `scripts/pathwise_precheck.py:39` |
| `scripts/tier1_pathwise_safety.py:48` |
| `scripts/denovo_learned_vs_uniform.py:52` |
| `scripts/composition_learned_vs_uniform.py:47` |
| `scripts/pareto_editing_hero.py:61` |
| `scripts/collect_twist_trajectories.py:87` (argparse default) |
| `scripts/griddd_reward_finetune_train.py:83` (argparse default) |

and `/private/tmp/pancake_checkpoint/checkpoint.recovery.pt` (**MISSING**) is the default of

| file:line |
|---|
| `scripts/qualify_analytic_pancake_quotient_backbone.py:31` |
| `scripts/run_griddd_analytic_zero_sidecar_smoke.py:50` |
| `scripts/griddd_valid_fiber_controller_panel.py:137` (argparse default) |

Mitigation that partly applies: `docs/GENERATOR_LINEAGE_MAP.md:24, 62` records a SHA-256
table for these checkpoints and states they also exist on the Modal volume, so the
*identity* is recoverable even though the bytes are not local. The hazard is that the
scripts encode no route to the volume copy — the default resolves to a path that cannot
be there, and the six non-argparse cases (module constants `CKPT = "…"`) cannot even be
overridden from the command line.

**`configs/pan_lung_corpus_manifest_v1.json:17`** and
**`scripts/build_compose_lipid_pretraining_inventory.py:32`**,
**`scripts/train_lnpdb_lung_oracle_baselines.py:111`** — absolute paths into
`/Users/rmaganti/Desktop/thesis_projects_ML/diffusion_project/lipid_diffusion`. These
**exist** and are not OS-reaped, so this is machine-lock rather than ephemerality: the
lipid corpus inventory is reproducible only on this laptop, and the external tree is
under no version control visible from here.

**`configs/benchmarks/griddd_qed_lead_manifest_v1.json:3`** — `"path":
"/Users/rmaganti/Documents/Codex/2026-07-14/ok-so/compose_rgm/scripts/build_griddd_qed_lead_manifest.py"`.
The manifest names its producer by absolute path into a *different checkout* of this
project. The path exists, so this reads as correct provenance while actually pointing
outside the repo whose commit the manifest is filed under.

**`scripts/verification/verify_mark_sampling.py:25-26, 61, 77`** and
**`scripts/verification/kernel_cost_profile.py:24-25, 60, 78`** — `sys.path.insert(0,
"/Users/rmaganti/compose_v2_work/src")` and `Path("/Users/rmaganti/compose_trainset_backup")`.
All exist. Arguably intentional and partly self-aware: both files carry a comment
(`verify_mark_sampling.py:29-34`) explaining that scratchpad paths were deliberately
avoided in favour of `COMPOSE_SCRATCH` + a documented `modal volume get`. The residual
defect is that the *repo root itself* is hardcoded, so a checkout at any other path
imports the wrong `src/` or fails.

### NOTED — intentional or already remediated, listed so they are not re-flagged

- `src/compose_v4/data/durable_path.py:42-45` — the reapable prefixes are the gate's
  own allowlist. Correct by construction.
- `scripts/audit_ring_restate_primitive_paths.py:15-21` — `DEFAULT_TRANSFER_ROOT = None`
  with a comment recording that this previously pointed into a dead 2026-07-14
  scratchpad. **This is the remediation pattern the Class-1 findings above should
  follow**: no default, fail loudly.
- `scripts/hphi_valid128_read.py:4`, `hphi_valid128_curve.py:4`, `hphi_shortlist_read.py:5`,
  `hphi_h40head_ab_read.py:4`, `task3_*_smoke.py`, `task3_reachable_ab.py:160`,
  `task3_phase_a_report.py:89`, `task3_mechanism_ab_report.py:94` — `/tmp/*` used as a
  **cache in front of a `modal volume get`** (`hphi_valid128_read.py:6-10`) or in front of
  regenerable run directories that `.gitignore:70-77` explicitly documents as
  "regenerable in under a minute per seed". Reapable but recoverable.
- `scripts/presentation/build_e0_exactness_slide.py:10` — `MPLCONFIGDIR` only.
- `scripts/corruption_shard_gate.py:182` — `/tmp/_gate_ckpt.pt`, a scratch file inside a gate.
- `scripts/ops/launch_detached.sh`, `poll_volume.sh`, `transfer_workspace.sh` — clean;
  every path is a parameter, and `transfer_workspace.sh:31-34` pins three expected
  content hashes.

### Out of scope but material

`diagnostics/` (not in the audited set) contains ~30 result JSONs whose recorded input
paths point into a **2026-07-14 session scratchpad that no longer exists** — e.g.
`diagnostics/griddd_valid_fiber_controller_panel12_lineageB_ceiling.json:146`,
`diagnostics/composition/scaled_edit_data_manifest.json:159`,
`diagnostics/coherence/packed_charge_policy_audit_v1_2026-07-30.json:146008`, and
~2,300 such paths in
`diagnostics/coherence/ringcore_v1_validation_semantic_cell_census_full_2026-07-30.json`.
As *records of what a past run read* these are correct provenance; as *inputs to a
rerun* they are dead. Flagged only so the distinction is on record.

---

# Class 2 — Data dependencies not in the repo

Each row was tested against the filesystem. "Modal volume" means the artifact is
recoverable from `compose-v4-artifacts` or `guacamol` — a real dependency, but not a
loss.

### CRITICAL

**`scripts/task3_offmanifold_check.py:53`** — `ZINC = Path("local_runtime/zinc250k/250k_rndm_zinc_drugs_clean_3.csv")`.
The file exists locally (22.6 MB) but `git check-ignore -v` confirms it is ignored by
`.gitignore:55` (`local_runtime/`) and `git ls-files` confirms it is **untracked**. So
this script has *both* its reference distributions outside the repo: the ZINC pool via
gitignore, the kinase training set via the session scratchpad. Neither has a recorded
SHA-256 or a fetch URL anywhere in the repo. A fresh clone can run neither of the
script's two questions.

**`scripts/hphi_extend_curve_read.py:24`** — `/tmp/rv5/hphi_recede_v5/{i:03d}_H40.json`
(128 files present, no re-fetch path). See Class 1 HIGH. This is the only
`docs/*.json`-producing script whose primary input has no volume fallback.

### HIGH

**`data/guacamol_subset_500000_seed0.smiles` — MISSING.** Referenced as
`MiningConfig.corpus_path` at `scripts/mine_edit_traces.py:130`, and named as
`corpus_id: "guacamol_subset_500000_seed0"` at `:129`. **On the Modal volume**:
`modal_apps/mine_edit_traces_app.py:537` reads `/guacamol/`, and
`scripts/transfer_workspace.sh:28` lists it in `CORPUS_FILES` for the `guacamol`
volume. Acceptable-with-note: the local script's default silently names a file that
only exists in the cloud, with no code path that fetches it.

**`diagnostics/composition/analogue_trace_pool.jsonl` — MISSING.** Read at
`scripts/build_edit_data_manifest.py:45` and defaulted at `scripts/sampling_unit_audit.py:112`.
Partly mitigated, and the mitigation is worth copying: `build_edit_data_manifest.py:47-56`
labels it `"kind": "generated dataset (NOT a committed fixture)"` and records a
`generation_command` (`PYTHONPATH=src:scripts python scripts/build_analogue_trace_pool.py`).
The residual hazard is that `build_edit_data_manifest.py:47` computes `rows = … if
pool.exists() else 0` and `_sha256(pool)` on a missing file — so the manifest can be
regenerated *right now* recording `row_count: 0`, which is a wrong number rather than a
failure. `sampling_unit_audit.py:120` by contrast calls `Path(args.pool).read_text()`
directly and crashes loudly, which is the correct behaviour.

**`local_runtime/hphi_pilot_h24.json.gz` — MISSING**, at
`modal_apps/hphi_steering_test_app.py:339-340`. Guarded by `.exists()`, so `blob` binds
to `None` rather than raising. `local_runtime/` is gitignored wholesale (`.gitignore:55`).
A guard that turns a missing input into `None` is a Class-5 pattern wearing Class-2
clothes: the steering test proceeds without the pilot blob.

### MEDIUM

- **`artifacts/griddd_qed_canonical_successor_smoke/checkpoint.pt` — MISSING**
  (`scripts/run_griddd_real_rewrite_smoke.py:38`). Doubly excluded: `.gitignore:23` (`*.pt`)
  and `.gitignore:62` (`artifacts/`). Its sibling default
  `diagnostics/canonical_successor_backbone_qualification.json` at `:34` **exists** and
  is tracked, so the smoke test half-resolves.
- **`data/agile_virtual/agile_virtual.csv` — MISSING** relative to repo root
  (`scripts/build_compose_lipid_pretraining_inventory.py:539`). It resolves under
  `external_root`, i.e. the `/Users/…/lipid_diffusion` tree from Class 1 — reproducible
  only on this machine.
- **`/private/tmp/process_v2_active8_teacher_support_benchmark.json` — MISSING**, cited by
  `docs/EDITING_V2_ACTIVE8_FAST_PATH_DECISION.md:90`, which correctly calls it "a local"
  artifact. The tracked twin `diagnostics/process_v2_active8_teacher_support_benchmark.json`
  (`scripts/benchmark_process_v2_active8_teacher_support.py:108`) is also **MISSING**, so
  the decision doc's supporting measurement exists in neither location.

### NOTED — dependency is on a Modal volume, which is the documented source of truth

- Every `modal_apps/*` constant of the form `RUN_ROOT = "/artifacts/editing_v2/r_theta_run"`
  (≈60 occurrences, e.g. `modal_apps/hphi_train_app.py:54`, `hphi_smc_app.py:50`) and
  `REMOTE_ROOT = "/root/compose"`. These are container paths, correct by design.
- `.gitignore:60-62` states the policy explicitly: "the volume is the source of truth;
  these are local working copies". The `artifacts/oracles/*` and `artifacts/h_phi_frozen_v1/*`
  bundles are the deliberate exception and are confirmed **tracked** in git
  (39 files under `artifacts/`, including all four `molleo_task3_v1` files and the four
  frozen `h_phi_*.pt` weights).

---

# Class 3 — Unpinned external dependencies

The standard is `artifacts/oracles/molleo_task3_v1/molleo_task3_oracle_manifest.json`:
`pickle_sha256` + `pickle_bytes` + `extraction_environment` (python/sklearn/rdkit/numpy)
+ `identity_evidence` naming the exact fingerprint call + an explicit
`"byte_identity_to_tdc_download": "UNVERIFIED -- host unreachable"`. Note that it
records its own honesty about what it could not verify. That is the bar.

Across `modal_apps/`, an AST scan of all `pip_install` / `apt_install` calls found the
environment otherwise **fully pinned** — `torch==2.4.0`, `numpy==1.26.4`,
`scipy==1.13.1`, `networkx==3.3`, `rdkit==2024.3.5` repeated in ~48 apps, and hoisted
into named constants at `modal_apps/audit_editing_active8_aromatic_cycle_open_impact.py:52-58`
and `modal_apps/run_editing_v2_semantic_p50_app.py:45-62` (`_IMAGE_BUILD_SPEC`, which
additionally pins gpu/cpu/memory/dtype/`CUBLAS_WORKSPACE_CONFIG`/`NVIDIA_TF32_OVERRIDE`).
Exactly two unpinned specifiers exist in the whole tree.

### CRITICAL

**`modal_apps/molleo_rtheta_parity_app.py:71`** — `.pip_install("PyTDC")`, no version.
This is the app that establishes MOLLEO/R_θ **parity**. Its oracle scores come from
`tdc.Oracle`, which downloads its payloads at runtime with no checksum on this side.
Two identical invocations months apart can resolve different PyTDC versions and
therefore different oracle pickles, and the parity verdict has no field that would
reveal it. The line below (`:72-73`) re-asserts the four numeric pins, so R_θ's kernel
is protected — the *oracle* is not.

### HIGH

**`modal_apps/molleo_env_gate_app.py:32`** — same `.pip_install("PyTDC")`. The gate is
self-aware about the risk (docstring `:1-13`: "If pip resolves numpy or scipy upward,
R_theta computes slightly different numbers … with no error to notice") and forces the
four pins back at `:34-45` with an excellent comment explaining the `scikit-learn==1.2.2`
choice. The defect is what the gate's own output shows. `docs/MOLLEO_ENV_GATE.json`
records:
```
"torch": "2.4.0+cu121", "numpy": "1.26.4", "scipy": "1.13.1",
"rdkit": "2024.03.5", "sklearn": "1.2.2", "pins_held": true, "drift": {},
"tdc": "unknown"
```
`pins_held: true` and `drift: {}` are reported for the five pinned packages while the
one *unpinned* package's version came back `"unknown"` — `modal_apps/molleo_env_gate_app.py:107`
falls back to `getattr(tdc, "__version__", "unknown")`. So the gate certifies "no drift"
for a container in which the only drift-capable component is also the only one whose
identity was not captured. Recording the resolved version does not fix the pin, but it
is the minimum: right now there is no way to reconstruct which TDC produced any banked
MOLLEO number.

### MEDIUM

- **`modal_apps/molleo_env_gate_app.py:168`** — `o = Oracle(name=name)` inside the
  objective loop. The comment at `:117-119` states that downloading the payloads "is
  part of what is being tested", so the *fetch* is intentional here. What is missing is
  a checksum on what arrived: the loop scores two molecules and records the values
  (`objective_scores`) without ever comparing the downloaded JNK3/GSK3B forests against
  the SHA-256 the project already holds in `molleo_task3_oracle_manifest.json:62, 101`.
  The pin exists in the repo and is simply not consulted at the point of use.
- **`artifacts/oracles/molleo_task3_v1/molleo_task3_oracle_manifest.json:62, 101, 160`** —
  the `found_at` / `kinase_source` fields are session-scratchpad absolute paths. For the
  two `.pkl`s this is harmless (the payloads were extracted into tracked `.npz` files and
  are SHA-pinned). For `kinase_source` at `:160` it is **not** harmless: the TSV was never
  vendored, so this line is the only surviving pointer to the Class-2 CRITICAL
  dependency and it points into a dying directory.

### NOTED

- `scripts/build_griddd_qed_lead_manifest.py` is the exemplar for upstream code:
  `JIN_COMMIT` and `GRIDDD_COMMIT` at `:23-24`, `JIN_TEST_EXPECTED_SHA256` /
  `ZINC_SOURCE_EXPECTED_SHA256` at `:26-31`, both **verified at read time** (`:113`, `:182`),
  and — unusually good — an explicit `blockers` list at `:332-338` naming five reasons the
  GrIDDD release cannot be reproduced exactly, plus `"must_not_be_labeled": "official
  GrIDDD 800"` at `:342`. `docs/workstreams/multiobjective/HANDOFF.md:171-172` likewise
  pins its HN-GFN clone to commit `90078b8ceeee…`.
- No `run_commands`, `git clone`, `curl`, `wget`, `urlretrieve`, `requests.get`,
  `hf_hub_download`, `from_pretrained` or `torch.hub` call exists anywhere in
  `modal_apps/`, `scripts/` or `src/`. The network surface is genuinely small — which is
  why the single `PyTDC` line matters so much.

---

# Class 4 — Results without provenance

`docs/` holds **64** `*.json` result files. Sampling the **15 most recently modified**:

| file | mtime | producer (by grep, not by the file) | provenance fields |
|---|---|---|---|
| `docs/MOLLEO_BRIDGE_PAIRS.json` | 08-18 21:13 | `modal_apps/molleo_bridge_probe_app.py` | none (`note`, `pairs`) |
| `docs/MOLLEO_H40_VISITED.json` | 08-18 21:05 | `modal_apps/molleo_fiber_census_app.py` | none — bare JSON list |
| `docs/MOLLEO_BASIN_ANCHORS.json` | 08-18 20:47 | `modal_apps/molleo_basin_substrate_app.py` | none |
| `docs/MOLLEO_DEV_LABELS.json` | 08-18 18:47 | ambiguous — 3 apps reference it | none |
| `docs/MOLLEO_DEV_COHORT.json` | 08-18 18:43 | `modal_apps/molleo_lazy_gate_app.py` | **partial**: `seed`, `init_sha256`, `roots_sha256` |
| `docs/MOLLEO_ENV_GATE.json` | 08-18 18:16 | `modal_apps/molleo_env_gate_app.py` | **partial**: package versions; no commit/script |
| `docs/SHORTLIST_RETENTION.json` | 08-18 15:47 | `scripts/hphi_shortlist_read.py` | none |
| `docs/SHORTLIST_TASKS.json` | 08-18 13:29 | ambiguous — app + reader both touch it | none |
| `docs/VALID128_CURVE.json` | 08-18 12:13 | `scripts/hphi_valid128_curve.py` | none |
| `docs/VALID128_K8_RESULT.json` | 08-18 11:40 | `scripts/hphi_valid128_read.py` | none |
| `docs/EXTENDED_CURVE_64.json` | 08-18 10:14 | `scripts/hphi_extend_curve_read.py` | none |
| `docs/PARTICLE_MULTIPLICITY.json` | 08-18 08:31 | `modal_apps/hphi_multiplicity_app.py` | none |
| `docs/RECEDING_HORIZON_AB_64.json` | 08-18 00:15 | **no writer found in the repo** | none |
| `docs/MONOTONICITY_LOCALIZE.json` | 08-18 00:12 | `modal_apps/hphi_monotonicity_localize_app.py` | none |
| `docs/BUDGET_SENSITIVITY.json` | 08-18 00:05 | `modal_apps/hphi_budget_sensitivity_app.py` | none |

**15 of 15 record no git commit and no producing script or app.** Two record a partial
environment or input hash. None records its inputs. The producer is recoverable for 13
of 15 only by grepping the repo for the output filename — an out-of-band step that
stops working the moment an app is renamed.

Two findings compound with earlier classes:

- **`docs/RECEDING_HORIZON_AB_64.json` is an orphan**: `grep -rn "RECEDING_HORIZON_AB_64"`
  across the entire tree (excluding `.git`) returns **zero** hits outside the file itself.
  Its near-namesake `docs/RECEDING_HORIZON_AB.json` is written by
  `modal_apps/hphi_recede_read_app.py:104`. A banked A/B result whose producer cannot be
  identified even by search.
- **`docs/EXTENDED_CURVE_64.json`, `docs/VALID128_*.json`, `docs/SHORTLIST_RETENTION.json`**
  are produced by the four `/tmp`-reading scripts from Class 1. So the results with the
  weakest provenance are exactly the ones whose inputs are least durable — and
  `EXTENDED_CURVE_64.json` is the one whose input has no volume fallback at all.

**Repo-wide**: only **13 of 64** `docs/*.json` carry any provenance-shaped top-level key,
and most of those carry only `schema`. Complete provenance appears in exactly one file:

- `docs/HPHI_SMC_REFERENCE_BANKED.json` — `artifact_sha256`, `record_sha256`,
  `provenance.git_commit` (`528292fc9d5c`), `provenance.protocol`, `provenance.output_rule`,
  `provenance.proposal_law`, and `mechanical_qualification.script`
  (`scripts/audit_smc_reference.py`). This is the template.
- `docs/HPHI_CORPUS_FROZEN.json` comes close — `sha256`, `bytes`, `artifact` path,
  `r_theta` model hash — but has no `git_commit` and no producing script.

Also noted: 18 of 64 `docs/*.json` have no discoverable writer at all
(`ARCHIVE_3ARM_RESULT`, `ARCHIVE_AB_RESULT`, `CLAUDE_GENERATOR_HANDOFF_MANIFEST_V1`,
`CLAUDE_GENERATOR_RUN_LINEAGE_MANIFEST_V2`, `CLAUDE_LIPID_HANDOFF_MANIFEST_V1`,
`HPHI_CORPUS_CENSUS`, `HPHI_CORPUS_FROZEN`, `HPHI_DEV_PANEL_64_BANKED`,
`HPHI_HORIZON_QUALIFICATION`, `HPHI_PILOT_CENSUS`, `HPHI_SMC_PROBE_BANKED`,
`HPHI_SMC_REFERENCE_BANKED`, `QUARANTINE_PRE_FREEZE_SMOKE`, `RECEDING_HORIZON_AB_64`,
`SMC_EFFICACY_PROBE_SOURCES`, `SMC_LADDER_RUNG2`, `SMC_PROBE_MATCHED_BASELINE`,
`SMC_SENTINELS`). For the manually-banked ones this is expected — which is precisely
why the in-file `provenance` block that `HPHI_SMC_REFERENCE_BANKED.json` carries is the
only thing standing between them and anonymity.

---

# Class 5 — Silent-failure patterns

An AST scan of every `except` handler in `modal_apps/`, `scripts/` and `src/` found
**304** handlers whose body only swallows (`pass` / `continue` / `break` / a default
assignment / a default return) and **zero bare `except:`**. The findings below are the
subset where the swallowed error changes a **reported number or verdict** rather than
merely skipping work. For each, the concrete consequence is stated.

### CRITICAL — a verdict can pass while the check did not run

**`modal_apps/hphi_lazy_parity_app.py:182-183`**
```
except Exception as err:  # noqa: BLE001
    fb["error"] = f"{type(err).__name__}: {err}"
```
This wraps the `_family_base` fallback comparison — the check that the lazy batch's
precomputed ring mass reproduces the eager recomputation bit-for-bit. The verdict `ok`
is accumulated at `:197-198` in a loop that **skips this key entirely** (`:192-193`:
`if t == "_family_base": continue`). `fb["error"]` is printed as a "note" at `:214-215`
and stored in the output, but never consulted. So `LEVEL 1 RAW-HEAD PARITY: PASSED` can
be printed, and `docs/LAZY_HEAD_PARITY.json` written with `ok: true`, while the
family-base-logits comparison silently never executed. The module's own failure text
(`:222-225`) says the lazy sampler "MUST NOT be used until this is bit-identical" — that
guard is exactly the one this handler can bypass.

**`modal_apps/hphi_cache_parity_app.py:57-59`**
```
try:    body = json.loads(f.read_text())
except Exception:  # noqa: BLE001
    continue
```
Inside `load()`, which builds the record dictionaries for both the cold and warm arms.
A truncated or corrupt replicate — the most likely artifact of an interrupted Modal run,
which this project has had — is dropped from its arm with no counter. The comparison
then runs over `shared = set(a) & set(b)` (`:68`) and reports parity over whatever
survived. The printed `cold N records warm M records shared K` is the only hint, and it
looks like an ordinary partial run. Loss: the corrupted records, which are the ones most
likely to disagree, are the ones removed from the parity denominator.

**`scripts/verification/admission_mask_parity.py:106-107` + `:189-190`**
```
except Exception:
    return None          # fingerprint() -> counted as `skipped`
...
if missing:
    print(f"  WARNING: {len(missing)} banked states absent from this run")
```
`fingerprint()` returns `None` when `smiles_to_molecular_graph` raises; `run()` counts it
into `skipped` (honest so far). But `main()` compares only `shared = set(base) ∩ set(got)`
(`:184`) and treats `missing` as a **warning**, then falls through to
`print("BITWISE PARITY: FAILED")` / `raise SystemExit(1)` **only** if `bad` is non-empty
(`:192-198`). Consequence: a regression that makes the parser *throw* on N banked states
causes those states to vanish from the comparison, prints a warning, and **exits 0 with
parity passing**. The one failure mode a bitwise-parity gate must not have.

### HIGH — a reported metric is computed over a silently shrunken sample

**`modal_apps/molleo_basin_substrate_app.py:329-331`**
```
for y in succ:
    try:    js.append(float(obj(y)[1]))
    except Exception:  # noqa: BLE001
        continue
```
`js` then feeds `out["onestep_jnk3_max"]`, `_mean`, `_p95` and
`out["onestep_jnk3_n_ge_0.5"]` (`:334-339`). A successor whose objective evaluation
raises is dropped from all four. `n_ge_0.5` is the **active count** — the quantity the
Task-3 investigation is built around ("Both Phase A arms found zero JNK3 actives where
~4.1 were expected", `scripts/task3_offmanifold_check.py:4-6`). A silently dropped
molecule is indistinguishable from a molecule that scored low, and drives the count in
the same direction as the hypothesis under test.

**`modal_apps/molleo_basin_substrate_app.py:325`** — the sibling handler in the same loop
drops successors whose `system.apply` raises, so `out["n_unique_successors"]` (`:336`)
undercounts while `out["fiber_size"] = len(law.marks)` (`:335`) is the full mark count.
The two are reported side by side and will be read as a ratio.

**`modal_apps/molleo_fiber_census_app.py:249`** — same shape in the census: a mark whose
`_coordinate_action`/`apply` raises never enters `best`, so the fiber is undercounted and
the R_θ probability ordering `ys` (`:252`) is computed over the survivors. Its sibling at
`:257-258` is better — it appends `float("nan")` rather than dropping — but the NaN then
flows into `np.asarray(js)` and downstream statistics unguarded.

**`scripts/ring_core_rollout_panel.py:319-321`**
```
for _ in range(24):
    try:  ... succ.add(canonical_state_key(system.apply(source, mk.rule_name, mk.action)))
    except Exception:  # noqa: BLE001
        continue
branchings.append(len(succ))
```
`branchings` is the panel's branching-factor metric. Every draw that raises silently
reduces `len(succ)`, so a model that fails on 20 of its 24 draws reports a branching
factor of 4 rather than an error. No counter, no denominator adjustment.

**`scripts/ring_core_checkpoint_selection.py:367-369`**
```
except Exception:  # noqa: BLE001
    pass
```
Wraps the alias search that builds the canonical-successor aggregation. On failure
`canon_logp` stays at `raw_logp` and `n_alias` stays 1 (`:355-356`), so that example
silently contributes an **un-aggregated** NLL to `canon_nlls` (`:372`) — the array used
to *select the checkpoint*. `alias_counts[n_alias]` records a 1, which is
indistinguishable from a genuine no-alias example. A checkpoint can be selected on a
metric that is a mixture of two different quantities.

### MEDIUM

**`modal_apps/hphi_sampler_floor_app.py:130-137`**
```
t0 = time.perf_counter()
try:    F.enumerate_ring_restate_semantic_groups(st, system=F.de_novo_rewrite_system())
except Exception:
    pass
t["enumerate_ring_restate_semantic_groups"].append(time.perf_counter() - t0)
```
The elapsed time is appended **outside** the `try`. A call that raises early contributes
a near-zero sample to that operator's cost distribution, so the reported per-call floor
for `enumerate_ring_restate_semantic_groups` is biased downward — and this is a *cost
floor* used for budgeting. It is also the only one of the six timed calls in that block
wrapped in a `try`, so the bias is specific to one row of the table.

**`scripts/composition_learned_vs_uniform.py:67-72`**
```
def safe_sa(m):
    try:    s = sascorer.calculateScore(m); return float(s) if s is not None else 10.0
    except Exception:  # noqa: BLE001
        return 10.0
```
`10.0` is the worst SA, mapped at `:76` to objective `0.0` and fed into `nondominated()`
and the learned-vs-uniform Pareto comparison. Arguably intentional as a worst-case
penalty — but the failures are not counted, so a systematic RDKit failure on one arm's
molecules is indistinguishable from that arm generating unsynthesizable molecules, which
is the exact conclusion the experiment draws. `scripts/denovo_learned_vs_uniform.py` uses
the same helper.

**`src/compose_v4/experiments/checkpoint_evaluator.py:162-173`** — `_operator_registry_hash()`
returns `None` through three nested handlers. The docstring at `:157-160` states the
intent — "a missing hash must degrade to 'unknown' rather than crash an evaluation. It is
still recorded, so a None here is visible" — so this is **deliberate and documented**. The
residual risk is worth stating anyway: an operator-registry hash is an *identity* check.
Degrading identity to `None` means a checkpoint evaluated against a silently different
operator registry produces a full set of plausible numbers, and the only signal is a null
field that a downstream reader must know to check.

**`scripts/ring_core_rollout_panel.py:223-224`** — leads that fail `smiles_to_molecular_graph`
are `continue`d out of the panel loop with no counter, so every per-lead rate in the
panel is computed over an unrecorded denominator.

### NOTED — correct handling, listed as the in-repo counter-examples

- **`src/compose_v4/experiments/griddd_conditional.py:2052-2060`** — `except Exception as
  error:` with the inline comment *"# all failures remain in the denominator"*, constructing
  a `CandidateGenerationResult(stalled=True, error=f"{type(error).__name__}: {error}")`.
  The failure is typed, recorded, and counted. `:1808-1818` does the same. **This is the
  pattern the CRITICAL and HIGH findings above should be converted to.**
- **`modal_apps/hphi_execution_parity_app.py:155-165`** — both handlers append to `bad`
  with a `reason` string, and `ok = not bad` (`:181`) makes any raised exception **fail**
  the parity verdict, with `raise SystemExit(1)` at `:215`. Correct. (Minor note: a
  raising coordinate increments `seen[fam]` but not `checked`, so the reported
  "coordinates executed on BOTH paths" correctly excludes it.)
- **`scripts/ddsbm_frozen_analysis.py:57-58`** — `_sa_scorer()` returns `None` when the
  RDKit contrib scorer is unavailable; `:120` guards with `if sa_fn is not None` and `:163`
  records `"sa_scorer_available": sa_fn is not None` into the output. Absence is reported
  rather than imputed.
- **`modal_apps/molleo_rtheta_parity_app.py:365`** — `except Exception: pass` on reading a
  prior record for resume. A corrupt record simply causes recomputation. Benign.
- **`scripts/ring_core_checkpoint_selection.py:337-339`** — increments `skipped` before
  `continue`. Counted, therefore not silent.

---

## Cross-cutting observation

The five classes are not independent here. The highest-severity items cluster on a
single chain: **`scripts/hphi_extend_curve_read.py`** reads a `/tmp` bank with no volume
fallback (Class 1 HIGH / Class 2 CRITICAL) and writes **`docs/EXTENDED_CURVE_64.json`**
with no commit, script or input record (Class 4). And **`scripts/task3_offmanifold_check.py`**
depends on a session-scratchpad TSV (Class 1 CRITICAL) whose only surviving pointer is a
`found_at` string inside the otherwise-exemplary oracle manifest (Class 3 MEDIUM), plus a
gitignored ZINC file (Class 2 CRITICAL). In both cases the repo already contains the
mechanism that would have prevented the hazard — `require_durable_path`, the SHA-pinned
bundle, the `provenance` block — and the mechanism is simply not invoked at the point of
use.
