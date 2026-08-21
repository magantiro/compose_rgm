# COMPOSE — navigation

*A newcomer's map. Written 2026-08-19. The root [`README.md`](README.md) is
older than this file and describes a superseded framing (Lineage B, the
`paper_iclr_stochastic_rewriting/` draft as **the** paper); it was left in place
because `pyproject.toml` names it. **Prefer this file.***

---

## What this project is

COMPOSE learns a **validity-closed molecular generator**: a continuous-time
Markov chain over molecular graphs whose committed states are always complete,
chemically valid, connected molecules, and whose transitions are executable
chemical rewrites. Generator Matching learns the contextual firing rates of
those rewrites (`R_θ`, goal-independent), and a learned reachability controller
(`h_φ`, goal-dependent) steers trajectories toward a target region over the
exact legal successor graph. The current critical path is the **QED / GrIDDD
editing benchmark**, where the headline is 54.7% source success at 12 returned
candidates on a prospectively held-out 128-source panel, against GrIDDD's
reported 45.1% at 20.

---

## Read these six things, in this order

| # | File | Why |
|---|---|---|
| 1 | [`docs/MASTER_PLAN_EXPERIMENTS_AND_COMPARATORS.md`](docs/MASTER_PLAN_EXPERIMENTS_AND_COMPARATORS.md) | **The governing plan.** Doctrine, resource-accounting rule, experiment map, QED protocol. |
| 2 | [`docs/REGISTRY_AUDIT_2026-08-19.md`](docs/REGISTRY_AUDIT_2026-08-19.md) | Which of the four registries you may trust, and the seven inter-document contradictions. |
| 3 | [`docs/REPRODUCIBILITY_HAZARDS_2026-08-19.md`](docs/REPRODUCIBILITY_HAZARDS_2026-08-19.md) | What will break when you try to re-run something, and why. |
| 4 | [`docs/SESSION_RUN_MANIFEST_2026-08-18.md`](docs/SESSION_RUN_MANIFEST_2026-08-18.md) | Every recent experiment: app, exact invocation, artifact, result, commit. |
| 5 | [`docs/DECISION_LOG.md`](docs/DECISION_LOG.md) | What was tried and **refuted**. Read before proposing an experiment. |
| 6 | [`docs/INDEX.md`](docs/INDEX.md) | All 300+ files under `docs/`, each with a purpose and a CURRENT / SUPERSEDED / HISTORICAL / UNKNOWN status. |

---

## Which plan governs

**`docs/MASTER_PLAN_EXPERIMENTS_AND_COMPARATORS.md`.** Lines 1–425 are the
governing section; line 426 opens `# ARCHIVED PROVENANCE — NOT GOVERNING`.

Four other documents claim to be canonical and are **not**:

| Document | What it claims | Reality |
|---|---|---|
| `docs/EXPERIMENT_PLAN.md` | "This file is the only current plan." | Overtaken. Contains zero mentions of MOLLEO and its "must-run" external list is the main thing the master plan changed. |
| `docs/ESTATE_REGISTRY.md` | "Exactly three files are current." | Its whitelist does not name the governing plan. |
| `docs/ARTIFACT_INDEX.md` | "Canonical research plans" → three HTML files | 1,247 commits behind. |
| `docs/research_plans/README.md` | "The canonical project strategy artifacts." | 2026-07-20 framing. |
| `CLAUDE.md` (auto-loads every session) | Repeats the `ESTATE_REGISTRY` answer | Outranks the others in practice, and is wrong about which plan governs. |

**One unresolved wrinkle, know it before you cite the master plan.**
`# ⭐ CURRENT AMENDMENT` (line 1141) and `# ⭐⭐ CURRENT AMENDMENT II` (line 1740)
both declare themselves governing, but both sit *below* the file's own
`NOT GOVERNING` divider at line 426. Which region wins has not been decided.

---

## Where the current results live

Small banked result JSONs are committed under `docs/`. The headline chain:

| Result | File |
|---|---|
| **Prospective 128-source validation, k=8** — coverage 49.2% | `docs/VALID128_K8_RESULT.json` |
| **Same panel at k=12** — 54.7% vs GrIDDD's reported 45.1% at k=20 | `docs/VALID128_CURVE.json` |
| Controller frozen *before* that run | `docs/AMENDMENT_VALIDATION_128.md` |
| 64-source development curve @1–@20 — 47/64 = 73.4% at 20 | `docs/EXTENDED_CURVE_64.json` |
| H24-vs-H40 receding-horizon A/B — 32/64 → 36/64 | `docs/RECEDING_HORIZON_AB_64.json` |
| SMC primitive, the provenance template for this repo | `docs/HPHI_SMC_REFERENCE_BANKED.json` |
| Exact finite-horizon control anchor | `diagnostics/exactness/editing_v2_experiment_b_exact_control.json` |
| Sealed 65-pair held-out panel — greedy 26/65 → verified rollout 40/65 | `diagnostics/editing_v2_sealed67_result.json` |

Older committed evidence is under `diagnostics/` and `results/`.
`docs/CLAIM_LEDGER.md` is the nearest thing to a claim register, but it has **no
rows** for QED, MOLLEO or the 128-source validation — do not read its silence as
absence of evidence.

---

## ⚠️ Bulk artifacts are on the Modal volume, not in git

**This is the single fact most likely to waste your first day.**

Corpora, embeddings, per-run records and trained heads live on the Modal volume
**`compose-v4-artifacts`**, under `editing_v2/r_theta_run/`. A fresh clone can
re-run the apps but **cannot read those results without the volume**.

| Asset | Where it actually is |
|---|---|
| `R_θ` checkpoint (never retrained) | volume, `runs/run_v2_01/` |
| Frozen H24 `h_φ` head + norm | volume, `hphi_v2/head.pt` (sha256 `9ea51ec4…`), `hphi_v2/norm.json` |
| Rollout corpora, embedding shards, per-run records | volume, `hphi_rollout_corpus/`, `hphi_v2/embeddings*/`, `hphi_recede_v5/` … |
| Task-3 and DRD2 oracle bundles | **in repo**, SHA-pinned: `artifacts/oracles/` |
| Frozen `h_φ` v1 weights | **in repo**: `artifacts/h_phi_frozen_v1/` |
| GuacaMol corpus | volume `guacamol` |

`.gitignore` states the policy directly: *"the volume is the source of truth;
these are local working copies."* `artifacts/oracles/*` and
`artifacts/h_phi_frozen_v1/*` are the deliberate tracked exceptions.

Pull a file with `modal volume get compose-v4-artifacts <remote path> <local>`.

---

## How to run an experiment

Experiments are Modal apps in `modal_apps/` (141 of them) driven by readers in
`scripts/` (238). The pattern, from `docs/SESSION_RUN_MANIFEST_2026-08-18.md`:

```bash
# 1. Price it first. Budget is an authorization ceiling, not a guideline.
#    Rate card: docs/MODAL_COST_MODEL.md   (CPU $0.04716/core-hour)

# 2. Launch DETACHED, from a COMMITTED tree, never from a scratchpad worktree.
scripts/ops/launch_detached.sh <worktree> modal_apps/hphi_h40head_ab_app.py run.log --valid --k 8

# 3. Poll the VOLUME, never the log. A client log stops updating the moment the
#    client dies while the detached run continues.
scripts/ops/poll_volume.sh compose-v4-artifacts editing_v2/r_theta_run <shard-name>

# 4. Read the result into a committed docs/*.json via a committed script.
python3 scripts/hphi_valid128_read.py
```

Three failure modes this project has already paid for:

- **`--detach` alone is not enough**, and never wrap a detached launch in a
  client-side `timeout` — it cancels the job. The app entrypoint should
  `drive.spawn()` rather than `.remote()`. (`scripts/ops/launch_detached.sh`)
- **Launch from the committed tree.** A pilot that passed from a `/private/tmp`
  worktree once preceded a full run that mounted a branch missing the module —
  0/5,984. (`docs/MASTER_PLAN_EXPERIMENTS_AND_COMPARATORS.md:628-643`)
- **Persist expensive deterministic artifacts as their own job.** Encode →
  persist → exit; do not hold embeddings in memory until the end of a long job.

Locally, without Modal:

```bash
export PYTHONPATH=src:scripts
export KMP_DUPLICATE_LIB_OK=TRUE            # macOS OpenMP guard
python3 -c "import sys; sys.path.insert(0,'src'); import compose_v4"
python3 -m pytest -q                        # pythonpath=src is set in pyproject.toml
```

---

## Where the paper drafts are

| Directory | Status |
|---|---|
| `paper_iclr2027/` | **The live draft — another agent is writing here now.** Untracked at time of writing. Do not edit without coordinating. |
| `paper/` | Preserved. Has its own `CLAIM_LEDGER.md` with a *different* schema from `docs/CLAIM_LEDGER.md`. |
| `paper_arxiv/` | Preserved. Carries `SCIENTIFIC_TRACEABILITY.md`, which cites `docs/CLAIM_LEDGER.md` rows by ID. Somebody compiled its PDF on 2026-08-18; no tracked file changed. |
| `paper_iclr_control_substrate/` | Preserved. |
| `paper_iclr_stochastic_rewriting/` | Preserved. Still named as "the paper" by the root `README.md`. |

All four preserved directories carry a `_STATUS.md` saying they predate the
canonical plan and must not be read as instruction. **Those notices point at
`docs/EXPERIMENT_PLAN.md` as the plan of record, which is itself now
superseded** — see `REORG_REPORT.md`. Which directory is the submission target
was still open at `docs/PROJECT_BOARD.md:176-182`; the existence of
`paper_iclr2027/` appears to answer it, but no document records the decision.

---

## Repository map

```
src/compose_v4/          the model. A process-identity SHA over rewrite/ and model/
                         gates every banked artifact — do not move files here.
modal_apps/              141 distributed experiment apps (the QED/h_phi lane is hphi_*)
scripts/                 238 drivers, readers and verification probes; scripts/ops/ for launches
docs/                    302 files — see docs/INDEX.md
diagnostics/             committed result JSONs from the editing-V2 / coherence lines
results/                 older committed result JSONs
configs/                 frozen inputs, launch configs, self-hashed contracts
artifacts/               mostly gitignored working copies; oracles/ and h_phi_frozen_v1/ ARE tracked
tests/                   pytest suite
third_party/             InversionGNN (the live external comparator)
data/, local_runtime/    corpora; largely gitignored or volume-resident
baselines/, recipes/, runs/   comparator code, training recipes, run outputs
paper*/                  see above
```

---

## Traps

Documents whose **names claim more authority than they have**:

- `docs/CURRENT_MODEL.md` — describes the pre-`run_v2_01` framing.
- `docs/PAPER1_FRAMING_AUTHORITATIVE.md` — ARCHIVED; on the plan's
  "do not read, cite or execute" list. It stays where it is: ~13 files link to it.
- `docs/EXPERIMENT_PLAN.md` — "the only current plan"; it is not.
- `docs/PROJECT_STATUS.md` — a 2026-07-19 page describing a retired checkpoint as live work.
- `docs/PROJECT_BOARD.md` — the "durable task list", 359 commits behind, whose sole
  in-flight item was refuted four hours before its last edit.

Before you trust any banked number, check
`docs/REPRODUCIBILITY_HAZARDS_2026-08-19.md`: **15 of 15** sampled recent
`docs/*.json` results record no git commit and no producing script, and
`docs/EXTENDED_CURVE_64.json`'s producer reads a `/tmp` cache with no volume
fallback.
