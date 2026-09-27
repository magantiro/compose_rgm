# Handoff for repo organization — 2026-09-27

Written for an agent about to reorganize this repository. Everything below is either
MEASURED (I ran the command in this session) or marked INFERRED. Read the two "do not
break" sections before moving anything.

---

## 1. There is one repository and 41 working copies of it

MEASURED. `git -C ~/compose_rgm_git worktree list` returns **41 entries**. All of them are
worktrees of a single repo, `https://github.com/KoshaTx/compose_rgm.git`.

    ~/compose_rgm_git            branch compose-iclr          64 GB   <- the primary checkout
    ~/compose_pmo_chain          pmo-chain-ablation-20260925  1.5 GB  <- PMO A/B/C ablation
    ~/compose_t4_nitya           t4-nitya-expansion-20260925  781 MB  <- T4 docking panel
    ~/compose_pmo_*              ~12 further PMO worktrees
    ~/compose_denovo_*, ~/compose_fragment_*, ...  more per-experiment worktrees
    /private/tmp/armc_frozen     detached at e11cd89d                 <- REAPABLE
    /private/tmp/revX, /private/tmp/compose_denovo_prior_split_..., ...  3 more REAPABLE

**The `~/compose_rgm_git` checkout is 64 GB and 12 GB of that is `diagnostics/`.** That is
where any size win is, not in code.

Not a git repo at all, and easy to miss:

    ~/compose_pmo_ablation/      140 KB, PLAIN DIRECTORY, no VCS

It holds `frozen_config_v1.json` (the PMO ablation's frozen protocol), `inputs/pmo_1k_final.json`
(**the only local copy of arm A's per-seed metrics**), the mechanism report, and `armC/`.
If you tidy directories, this one is unversioned and load-bearing. It should be brought under
git, not deleted or moved without copying first.

### Two hazards to fix while you are in here

1. **4 worktrees live under `/private/tmp`, which macOS reaps mid-session.** One of them
   (`/private/tmp/revX`) is already marked `prunable`. A worktree's objects live in the parent
   repo, so a reaped checkout loses no commits — recover with `git worktree prune` then
   `git worktree add <durable-path> <branch>`. But do not create new ones there.
2. **15 branches exist on no remote** (MEASURED via `git ls-remote --heads origin` per branch,
   not via `refs/remotes`, which only shows local tracking refs and will lie to you). They are
   single-copy on one disk. List: `denovo-fragment-quality-20260923`,
   `denovo-legal-mark-prior-pilot-20260924`, `fragment-attachment-library-20260924`,
   `fragment-chem-prior-pilot-20260924`, `fragment-decoration-breadth-v1`,
   `fragment-decoration-source-coupled-v1`, `fragment-interface-ablation-20260923`,
   `fragment-linker-broaden-v1`, `fragment-linker-novelty-v1`,
   `fragment-linker-strict-unseen-v1`, `fragment-qed-ablations-20260925`,
   `pmo-construction-prior-gate-20260924`, `pmo-existing-region-remodel-20260924`,
   `pmo-prior-memory-gate-20260924`, `qed-program-controller-smoke-20260925`.
   **Push before removing any worktree.** Worktree removal never deletes a branch, but the
   safety check must be "is this branch on origin at a matching full SHA", not "does a
   checkout exist".

---

## 2. Layout of the repo itself

MEASURED counts in `~/compose_rgm_git`:

    src/          453 python modules   29 MB   the model, chemistry kernel, controllers
    tests/        502 test files       22 MB
    scripts/      274 scripts         4.4 MB   local drivers, audits, figure builders
    modal_apps/   228 apps            6.7 MB   cloud entry points
    configs/      140 json            3.8 MB   CONTRACTS -- see section 5
    diagnostics/  533 entries          12 GB   committed results, and the size problem
    docs/                              19 MB
    paper_*/      7 directories                submitted + in-progress manuscripts
    archive/ artifacts/ baselines/ data/ experiments/ oracle/ output/ recipes/
    results/ third_party/ tmp/ tools/ upload/  + 4 COMPOSE_ICLR_2027_* directories

`CLAUDE.md` and `.claude/context/{mission,conventions,glossary,learnings}.md` auto-load every
session. **`learnings.md` is append-only and is the single most valuable file in the repo** —
it is a dated ledger of gotchas, several of which will bite a reorganization directly. Do not
rewrite or reflow it.

Authority order for docs, from `CLAUDE.md`: `AGENTS.md` (contract) → `docs/START_HERE_ICLR.md`
(status) → `docs/PAPER_TO_CURRENT_CODE.md` (paper↔code map) → `docs/CONTROLLER_LIVE.md`
(chronological ledger). **Superseded plans are kept deliberately as part of the scientific
record.** Do not infer authority from filenames like `CURRENT`, `PLAN`, `FINAL` or `HANDOFF` —
the repo explicitly warns about this, and there are many such files.

---

## 3. What I have been doing, and where it lives

Three workstreams, all currently live or just-finished. Each splits its state between **git
(code, contracts, small reductions)** and **Modal volumes (the actual run artifacts)**. The
volumes are NOT in git and are far larger than the repo.

### 3.1 T4 — constrained lead optimization, the docking benchmark

**What it is.** 15 benchmark cells (5 protein targets × 3 supplied lead molecules) × 2
similarity thresholds δ=0.4 and δ=0.6 = 30 cells, run at 3 replicate seeds = 90 cell-runs.
The objective is **QuickVina2 docking score of the best eligible molecule, lower is better**,
subject to QED ≥ 0.6, SA ≤ 4.0, Tanimoto ≥ δ to the supplied lead. 250 charged docking calls
per cell. Comparators: InVirtuoGen, GenMol, RetMol, GraphGA.

**Worktree.** `~/compose_t4_nitya`, branch `t4-nitya-expansion-20260925`.

**Code.** `modal_apps/t4_unified_controller_app.py` is the one that matters. Its function
limits are load-bearing and I got burned by them (section 6).
`src/compose_v4/experiments/t4_fiber_campaign.py` is the campaign loop and the eligibility
gate. `src/compose_v4/control/{bridge_region_law,replace_completion_law}.py` are proposal-law
repairs developed this cycle.

**Artifacts live on Modal, in two sets:**

    profile rahul-94866   compose-t4-unified-controller-{braf,fa7,5ht1b,jak2,parp1}-v1
                          the main replicate-2/3 panel, 60 cells
    profile nitya         compose-t4-unified-controller-{...}-rev-v1
                          the REVIVAL arm, 15 cells re-run after a preemption defect

**Artifact schema, which is not obvious and cost me a wrong reading:** every file on these
volumes is wrapped as `{"payload": {...}, "payload_sha256": "..."}`. You must go one level
into `payload`. A completed cell publishes `result.json` with `final_best`, `charged_calls`,
`status`, `best_smiles`, `claim_boundary`. A running cell has only `checkpoint.json` +
`round_NNN_lock.json` files; its best score is `payload.rounds[-1].best_so_far`, and its
`charged_calls` is in the checkpoint. **Completed vs partial is structural, not a convention:
`result.json` exists or it does not.**

**δ is not in the artifact.** It is in the contract (`configs/t4_unified_controller_*_{d04,d06}_*.json`,
field `delta`), resolvable by matching `payload.contract_payload_sha256`. The
`claim_boundary` prose in a completed result also states it. A past defect had a contract whose
prose said δ=0.6 while its executable `delta` was 0.4 — **diff the executable field against the
prose, never trust the prose.**

**Frozen replicate 1** is committed, at `diagnostics/T4_FROZEN_RESULT_v1.{json,md}`. The `.md`
holds the actual per-cell table; the `.json` holds metadata only. That file is authoritative
for replicate 1 and should not be regenerated.

**Current state (MEASURED, 00:33 on 2026-09-27): the panel has STOPPED at 18 of 60 cells
complete.** 27 cells are partially done (17–249 of 250 charged, 3,078 calls remaining), 15 have
no checkpoint at all. The revival arm is 14 of 15 done. Cause is the driver timeout, section 6.
Resuming needs no code change: `modal run --detach <app> --mode resume --run-id <id>
--confirm-prior-call-terminal`.

### 3.2 Docking specifically — what an organizer needs to know

- Docking runs **inside the Modal image**, not locally. The scoring pipeline is
  `obabel --gen3D` (3-D conformer) then `qvina02`.
- **`qvina02` is seeded and the box is fixed, but `obabel --gen3D` takes no seed.** MEASURED
  consequence: re-docking one molecule across runs spans **1.3 kcal/mol** (the fa7 seed
  molecule scored −7.5, −8.30, −8.8). INFERRED that the conformer is the cause — no repeated-
  conformer experiment has been run.
- Consequence for reporting: **per-cell margins under ~1.3 are not reproducible differences.**
  The right per-cell uncertainty for a 3-replicate mean is the replicate s.d. (MEASURED median
  **0.46**), and the aggregate/paired statistics are what carry the claim. I over-corrected on
  this once in-session and had to withdraw it; the correct framing is in the report.
- Seeding the conformer generator would change the docking adapter, which **every T4 contract
  pins**, including the frozen panel's. That is an owner decision, not a cleanup task.
- `reconciled_charged_calls` from round locks is a **lower bound in both directions** — locks
  are written per round index, so a redone round overwrites its own lock.

### 3.3 PMO — black-box optimization, and the A/B/C proposal ablation

**What it is.** 6 development objectives × 3 replicate seeds, 1,008 charged oracle calls per
campaign, metrics on the first 1,000 **resolved** receipts, no-prescreen. Three arms:

    A  COMPOSE, structured molecular proposals, unchanged configuration.
       PRE-EXISTING completed campaigns, REUSED. 0 new oracle calls. Never re-run.
    B  length-matched uniform legal edit chains: arm A's synthesis supplies the realized
       primitive-edit count, then the program is discarded and replaced by that many uniform
       draws from the legal Active8 fiber, recomputed after every intermediate.
    C  created-atom rebinding: the structured recipe is KEPT byte-for-byte (operation
       sequence, length, block boundaries, payloads, all source-atom bindings) and only
       operands referring to atoms CREATED by earlier operations are resampled.

**Worktree.** `~/compose_pmo_chain`, branch `pmo-chain-ablation-20260925`, pushed.

**Code, all on that branch:**

    src/compose_v4/control/pmo_legal_mark_sampler.py     exact uniform sampling over the legal
                                                         fiber without enumerating it (arm B)
    src/compose_v4/control/pmo_uniform_chain.py          arm B chain builder
    src/compose_v4/control/pmo_binding_intervention.py   arm C (frozen at commit e11cd89d)
    src/compose_v4/control/dynamic_program_synthesis_v21.py  both arm hooks live here
    scripts/pmo_reward_adaptive_canary.py                the actual campaign runner
    modal_apps/pmo_fibercontrol_targets_app.py           cloud entry point
    tools/launch_pmo_fibercontrol_targets.py             launcher
    tests/test_pmo_binding_intervention.py               15 tests, 12/12 mutations killed

**Arms are selected by environment variable**, and they are mutually exclusive and fail closed:
`PMO_UNIFORM_CHAIN=1` (arm B), `PMO_BINDING_REBIND=1` (arm C), absent = arm A byte-identical.
The app name is overridable by `PMO_FIBERCONTROL_APP`, which is how arm C got its own deployed
app without disturbing arm B's.

**Artifacts are spread over THREE Modal workspaces**, and this is the single most confusing
part of the project:

    profile nitya        -> workspace nitya        volume compose-pmo-fibercontrol
                            namespaces scored_chain_*   = ARM B (18)
    profile rahul        -> workspace KOSHA-LABS   volume compose-pmo-fibercontrol
                            namespaces scored_rebind_*  = ARM C (18)
                            volume compose-pmo-fibercontrol-replication = ARM A campaigns
    profile rahul-94866  -> workspace rahul-94866  other PMO experiments, NOT arm A

**The profile name does not match the workspace name.** `rahul` deploys to `kosha-labs`.
And `modal volume list` prints `rahul-94866` in the "created by" column for volumes on the
`rahul` profile, so the listing itself cannot distinguish workspaces. There are also **two
distinct volumes named `compose-pmo-fibercontrol`** (one per workspace) and two named
`compose-pmo-fibercontrol-replication`. Pin the profile and the creation date before
concluding an artifact is absent — I lost time concluding arm A did not exist because I
scanned the wrong workspace.

**Per-campaign artifact schema** (flat, not payload-wrapped, unlike T4): `progress.json`
(per-round snapshot — its `charged_oracle_calls` LAGS, do not use it as the charged count),
`trajectory.jsonl` (one line per round with `charged`, `best`, `top10` — the compact source for
curves), `best_molecules.jsonl`, `queries.jsonl`, `controller_state.json`, `oracle/query_*/result.json`
(the **authoritative** charged count = resolved receipts), `canary_v1.json` (written at the end:
`best_score`, `final_top10`, `auc_official` — one read per campaign), and after the worker exits
`provenance.json` with **`returncode`: 0 = COMPLETE, non-zero = FAILED, absent = still running**.
I misread `rc=0` as dead once; it means success.

**Current state (MEASURED):** arm B **16 complete, 2 failed**; arm C **17 complete, 1 failed**;
arm A complete and reused. The A/B result is finished. Durable reductions:

    compose_pmo_chain/diagnostics/pmo_abc_ablation_v1/reduction_v1.json   per-cell A/B/C
    compose_pmo_chain/diagnostics/pmo_armc_verification_v1/VERIFICATION.md arm C verification
    compose_pmo_chain/diagnostics/compose_status_report/report_20260926.md full report

### 3.4 PMO oracle assets — a live trap

Three of the 23 PMO oracles are asset-backed: `gsk3b`, `drd2`, `jnk3`. `gsk3b` and `drd2` load
a pickle from the **relative** path `oracle/<name>.pkl` **on the first CALL**, and PyTDC's
`Oracle.__call__` has a bare `except` returning `0.0`. A constructor-scoped `chdir` does not fix
it. This previously produced a campaign that charged 250 calls and recorded `best_score 0.0`
with nothing in the artifact to distinguish it from a real result.

The fix lives at `src/compose_v4/experiments/pmo_oracle_assets.py` (`AssetPinnedOracle`: pin cwd
around **every** call, and `prime()` the lazy load inside that window). The capsule is at
`diagnostics/pmo_ivg_oracle_parity/ivg_oracle_assets/` (71 MB) and is **gitignored** — it is
local-only, so a fresh clone cannot build a working image. If you reorganize `diagnostics/`,
that directory must keep its path or the app's `add_local_dir` breaks.

**A positive control runs before the first charged call** and asserts known actives score > 0.
Do not remove it; "the oracle constructs" and "`score > 0`" are both insufficient evidence.

---

## 4. Two chemistry kernels. This is not a detail.

MEASURED from the app image definitions:

    PMO production      python 3.11, rdkit 2023.9.6, PyTDC 1.1.15 (--no-deps + a rdkit.six shim)
    T4 / editing        python 3.11, rdkit 2024.3.5, numpy 1.26.4, scipy 1.13.1, networkx 3.3,
                        torch 2.4.0
    the laptop .venv    python 3.12, rdkit 2026.03.6   <- NEITHER production kernel

PyTDC 1.1.15 pins `rdkit>=2023.9.5,<2024.3.1`, so the PMO image cannot hold 2024.3.5; the two
kernels are irreconcilable in one image by construction. Local mirrors I built and used:
`~/compose_pmo_oracle_env` (PMO, 2023.9.6) and `~/compose_region_pinned_env` (T4, 2024.3.5).
**Any number computed in the laptop `.venv` needs a parity statement** — two rdkit versions can
write different canonical SMILES for the same molecule, and canonical SMILES is used as a cache
key, so a spelling difference changes the search trajectory.

---

## 5. Do not break: the contract hash chains

`configs/*.json` are **content-addressed contracts**, not settings files. A typical one carries
`payload` + `payload_sha256`, and inside the payload `runtime_inputs_sha256` and/or
`implementation_sha256` mapping file paths to their sha256. Loaders verify these and **fail
closed**.

Consequences for a reorganization, all of which have bitten this repo before:

- **Moving or renaming a file that appears in any `runtime_inputs_sha256` or
  `implementation_sha256` breaks every contract that pins it**, and the failure appears far from
  the edit (inside a container, as `input identity mismatch`).
- **Editing a pinned file changes the run identity.** `modal_apps/train_tracelet_gm.py` hashes
  itself; the T4 unified app is one of 29 entries in every T4 contract. A launcher bugfix forces
  a fresh run label — that is the guard working, not an obstacle.
- **Re-pinning has a transitive blast radius.** One past identity move required updating nine
  artifacts plus four source constants; re-pinning only the first layer left 41 tests failing.
  The method that works is a fixed-point driver over the verifier's own pin discovery, iterated
  until a round writes nothing.
- **Never re-point a pin that encodes an AUTHORIZATION** (a contract the owner approved by hash)
  to make a launcher pass. That manufactures consent. Supersede explicitly instead.
- `configs/` is NOT in the training run-identity fingerprint (`src/`, `scripts/`, `recipes/` and
  the launcher are), so committing under `configs/`, `diagnostics/`, `tests/` and `.claude/`
  does not disturb a pinned launch worktree. INFERRED from the fingerprint definition; verify
  before relying on it.

There is also a **frozen production RingCore catalog fingerprint `639ff6078c32d43c`**. Any
claim-bearing evaluation or corpus compilation must abort on drift. This laptop's `.venv`
drifts to `82fd910c...`; that is an environment fact, not a defect.

---

## 6. The two infrastructure defects currently shaping results

**T4 driver wall.** MEASURED from the app: `run_cell` is `max_containers=3, timeout=20h`;
`drive` is `max_containers=1, timeout=21h`. So each protein runs only 3 cells at a time, and
after 21 h the driver stops launching queued cells. Observed: throughput decayed 296 → 34
calls/h and then to zero; the panel halted at 18/60. **This is why T4 is not finished.** The
20 h `run_cell` timeout is per container attempt, not cumulative — preempted cells resume from
checkpoint and several have accumulated >20 h of round time — so throughput, not the cell
timeout, is the constraint. I projected a 37 h completion assuming continuous operation and
that projection was wrong because of the 21 h driver wall.

**The round-0 preemption window.** A round lock is published BEFORE the root docking call, and
the first checkpoint is written at the END of round one. A preemption inside that window leaves
an unfinished query lock with no recoverable checkpoint, and the fail-closed guard then
correctly refuses to retry. **15 of 60 panel cells are permanently unresumable this way.** The
revival arm exists to re-run them, and is 14/15 done. A round-0 checkpoint was added to narrow
the window to one docking call; it did not close it.

---

## 7. Where a reorganization can safely win

MEASURED: `diagnostics/` is **12 GB of the 64 GB** primary checkout and holds 533 entries. A
recent cleanup of 26 stale worktrees recovered 8.7 → 28 GB, so worktree sprawl is the other
big lever. Suggestions, in order of safety:

1. **Push the 15 unpushed branches**, then prune worktrees whose branch is on origin at a
   matching full SHA. Use `git ls-remote --heads origin <branch>`, not `refs/remotes`.
2. **Move the 4 `/private/tmp` worktrees to durable paths** (`git worktree prune` +
   `git worktree add`).
3. **Bring `~/compose_pmo_ablation` under version control.** It is unversioned and holds the
   only local copy of arm A's metrics.
4. **Do not delete anything from `diagnostics/` without checking whether a config pins it.**
   `grep -rl "<path>" configs/` first. Several diagnostics files are pinned by hash, and at
   least one gitignored directory (`pmo_ivg_oracle_parity/ivg_oracle_assets`) is baked into a
   Modal image by literal path.
5. `archive/`, `output/`, `tmp/`, `upload/`, `results/` and the four `COMPOSE_ICLR_2027_*`
   directories are the plausible candidates for consolidation. I have not audited them and
   cannot say which are referenced. Check before moving.

**Standing rule from the repo, and it has been paid for: never let `/private/tmp` or
`/var/folders` hold the only copy of anything.** `compose_v4.data.durable_path` refuses such a
path for a source of truth, with `COMPOSE_ALLOW_REAPABLE_PATH=1` as an explicit opt-out.

---

## 8. What is still open

- **T4: resume the stopped panel.** 3,078 charged calls remain across 27 partial cells; 15 are
  unresumable and covered by the revival arm. Needs `--mode resume --run-id <id>
  --confirm-prior-call-terminal`, one run_id per protein per δ (10 total). No code change. Do
  not resume while an original driver is alive — currently all are at 0 tasks, so it is safe.
- **PMO A/B is finished** and is the reportable ablation (16 matched pairs; Final Top-10
  −0.065, p=0.012; AUC-Top10 −0.042, p=0.020; A better on 12/16 on both).
- **PMO arm C is finished and is a secondary result** (17 cells, AUC −0.020, p≈0.18). It should
  go in an appendix, not the main text, and its effect must not be described as an additive
  fraction of the A/B effect.
- **Manuscript integration** of the A/B ablation into Section 3.4 plus an appendix subsection,
  and a cleanup pass on stale "same learned R_theta everywhere" language, the T4 Sum row over
  unequal-coverage comparator columns, and two literally broken sentences in the T4 appendix.
