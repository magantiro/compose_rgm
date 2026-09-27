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

---

# PART II — the deep file-level account (added after direct inspection)

Everything in Part II was read in this session with `cat`/`grep`/`json.load`, not recalled.
Where I still cannot verify something I say so.

---

## 9. The docking path, read line by line

**`src/compose_v4/experiments/t4_docking_adapter.py` is 70 lines and is the whole of docking.**
`dock_t4(smiles, tag, seed, *, box, cpu=1)` does exactly three subprocess calls in `/tmp/<tag>/`:

    1. obabel -:<SMILES> --gen3D -O l.mol            timeout 120   <- NO SEED ACCEPTED
    2. obabel l.mol -O l.pdbqt                       timeout  60
    3. /opt/dock/qvina02 --receptor <box.receptor> --ligand l.pdbqt --out o.pdbqt
         --center_x/_y/_z <box[0]> --size_x/_y/_z <box[1]>
         --cpu <cpu> --num_modes 10 --exhaustiveness 1 --seed <seed>
                                                     timeout 300

The score is parsed from the first line starting `REMARK VINA RESULT`, field index 3.

Three properties that matter and that I had only inferred before:

* **`--exhaustiveness 1`.** That is the minimum. QuickVina's search is a single low-effort Monte
  Carlo run per mode. With `--seed` fixed the search is deterministic *given a conformer*, so
  the unseeded `--gen3D` really is the only stochastic input — which upgrades my earlier
  inference to a well-supported one — but exhaustiveness 1 is why a different conformer moves
  the score so much. **A reorganizer must not "tidy" these flags; they are hashed.**
* **Silent `None` on failure.** `except (subprocess.SubprocessError, OSError, ValueError): return None`.
  A docking failure is indistinguishable from an unscorable molecule at this layer.
* **It refuses to overwrite scratch:** if `l.mol`/`l.pdbqt`/`o.pdbqt` already exist it raises
  `"docking scratch collision; refusing an unaccounted retry"` rather than deleting a possibly
  valuable pose. That is deliberate and is why an interrupted cell cannot silently re-dock.

### Where the receptors and the binary come from

MEASURED in `modal_apps/t4_unified_controller_app.py` lines 96-124. At **image build time** the
image `apt_install`s `openbabel` and curls both the binary and all five receptors from the MOOD
repository:

    MOOD = https://raw.githubusercontent.com/SeulLee05/MOOD/main/scorer
    /opt/dock/qvina02                      sha256 f8ac045235025e98b15fd90aae6617edfdcc125081f72a5a5315db22be1f46e0
    /opt/dock/receptors/{parp1,jak2,braf,5ht1b,fa7}.pdbqt

So **the docking environment is an external network dependency resolved at build time.** If MOOD
moves or changes those files, a rebuilt image is no longer the pinned evaluator. The contract
carries `evaluator_sha256` with the `qvina02` hash, and the app re-hashes `RECEPTOR_PATH` into
the run record, so drift is detected — but only at run time, not prevented.

### The box is per-contract data, not code

`contract["docking_box"]` is `[[center_x, center_y, center_z], [size_x, size_y, size_z]]`.
MEASURED for parp1: `[[26.413, 11.282, 27.238], [18.521, 17.479, 19.995]]`. `contract["docking_seed"]`
is a single integer for the whole contract (MEASURED 20260918 for the parp1 δ=0.6 replicate
contract). So all cells of a contract dock with the same qvina seed; the per-cell variation comes
from the conformer and from which molecules the search proposes.

### The eligibility gate

`src/compose_v4/experiments/t4_fiber_campaign.py` lines 57-62 and 147-157:

    QED_MIN, SA_MAX = 0.6, 4.0
    REPRESENTABLE_HEAVY_ATOMS = 40
    if heavy > REPRESENTABLE_HEAVY_ATOMS: reject
    if similarity < self.delta or quality < QED_MIN or access > SA_MAX: return None

Note the comparators are **non-strict** (`<`, `>`), which is load-bearing: Tanimoto is a ratio of
small integers and endpoints land exactly on δ routinely, whereas QED and SA are continuous and
never do. Changing `<` to `<=` on similarity would silently delete real endpoints.

---

## 10. The T4 Modal app, function by function

`modal_apps/t4_unified_controller_app.py` defines five functions. MEASURED decorators:

    proposal_worker   max_containers=36   timeout=1800    (30 min)
    dock_worker       max_containers=24   timeout=720     (12 min)
    run_cell          max_containers=3    timeout=20*3600 (20 h)
    drive             max_containers=1    timeout=21*3600 (21 h)
    remote_status     max_containers=1    timeout=120

**This is the whole explanation of the throughput ceiling and of the stall.** Per protein app:
only 3 cells run concurrently, each fanning out to at most 36 proposal and 24 dock containers;
and the single `drive` container stops launching queued cells after 21 h. Six cells per run
against 3 slots means half of every run is queued from the start.

### The app is parameterised by environment, one wrapper per arm

    COMPOSE_HELD_VOLUME     e.g. compose-t4-unified-controller-parp1-v1
    COMPOSE_HELD_OUTPUT     e.g. /unified_parp1
    COMPOSE_HELD_RECEPTOR_NAME   one of parp1 jak2 braf 5ht1b fa7
    COMPOSE_HELD_WRAPPER    the wrapper app file, PINNED in runtime_inputs_sha256

The wrapper must be baked into the image because `_validate_task` re-hashes every pinned entry
**inside the container**. MEASURED: 25+ wrapper files exist, `modal_apps/t4_unified_controller_*_app.py`,
including the ten `*_{d04,d06}_r23_app.py` for the replicate panel and eight `*_rev_app.py` for
the revival arm.

### The contracts

Ten replicate-panel contracts and **eight** revival contracts (there is no `5ht1b_d06_rev` or
`fa7_d06_rev` — those two cells were not lost to the round-0 window):

    configs/t4_unified_controller_{5ht1b,braf,fa7,jak2,parp1}_{d04,d06}_r23_v1.json
    configs/t4_unified_controller_{5ht1b_d04,braf_d04,braf_d06,fa7_d04,jak2_d04,jak2_d06,parp1_d04,parp1_d06}_rev_v1.json

A contract payload has **40 keys**. The ones a reorganizer must not touch:
`runtime_inputs_sha256` (**29 pinned files**, listed below), `delta`, `docking_box`,
`docking_seed`, `evaluator_sha256`, `cells`, `charged_calls_per_cell`,
`total_charged_call_ceiling`, `authorization`, `scored_launch_authorized`,
`modal_launch_authorized`, `claim_boundary`, `promotion_criteria`, `replicate_policy`.

The 29 pinned runtime inputs (MEASURED from the parp1 δ=0.6 replicate contract; the set is the
same shape for the others, with the wrapper filename differing):

    diagnostics/t4_held_target_distillation_quality_v1/parp1_checkpoint.json
    diagnostics/t4_shared_retained_rewrite_v1/checkpoint.json
    docs/GENMOL_T4_SEEDS.json
    modal_apps/t4_unified_controller_app.py
    modal_apps/t4_unified_controller_parp1_d06_r23_app.py
    src/compose_v4/control/bridge_region_law.py
    src/compose_v4/control/complete_region_program.py
    src/compose_v4/control/dynamic_program_synthesis.py
    src/compose_v4/control/fiber_control.py
    src/compose_v4/control/frozen_proposal_escalation.py
    src/compose_v4/control/progressive_structured_sampler.py
    src/compose_v4/control/protonation_aware_proposal.py
    src/compose_v4/control/protonation_restate_program.py
    src/compose_v4/control/region_law_contract.py
    src/compose_v4/control/route_distilled_goal_expert.py
    src/compose_v4/control/structural_subgoal_policy.py
    src/compose_v4/control/structural_subgoal_realizer.py
    src/compose_v4/control/t4_unified_routing.py
    src/compose_v4/control/zero_support_fallback.py
    src/compose_v4/experiments/t4_docking_adapter.py
    src/compose_v4/experiments/t4_fiber_campaign.py
    src/compose_v4/experiments/t4_integrated_route_fiber.py
    src/compose_v4/experiments/t4_support_expansion.py
    src/compose_v4/experiments/t4_unified_controller.py
    src/compose_v4/experiments/t4_unified_proposal.py
    src/compose_v4/gates/med_chem_gate.py
    src/compose_v4/rewrite/action_codec_v5.py
    src/compose_v4/rewrite/kernel.py
    src/compose_v4/rewrite/operators.py

**Two of those are under `diagnostics/`.** Any reorganization of `diagnostics/` must treat
`t4_held_target_distillation_quality_v1/` and `t4_shared_retained_rewrite_v1/` as code.

### Cell identity

`contract["cells"]` is a list of 6 dicts per contract, keys:
`cell` (e.g. `parp1_0_r2`), `controller_seed`, `replicate`, `replicate_1_controller_seed`,
`smiles` (**the supplied benchmark lead**), `source_cell`, `source_global_index`.
So `<protein>_<0..2>_r<2|3>` decodes as protein, benchmark lead index, replicate — and the
replicate-1 seed is carried alongside so a replicate can be traced to the frozen panel.

### 55+ other T4 apps exist

MEASURED: 34 `modal_apps/*.py` reference `qvina`/`obabel`, and 20+ `src/compose_v4/experiments/t4_*.py`
do. Most are superseded single-purpose arms (`t4_shared_retained_fiber_*`,
`t4_integrated_route_fiber_*`, `t4_5ht1b2_protonation_rescue_*`, `t4_nodistill_*`). The repo
keeps superseded arms deliberately. **Do not consolidate them by similarity of name** — several
are pinned by hash in their own contracts, and some produced numbers in the frozen panel.

---

## 11. The PMO app and the arm machinery, read directly

### App

`modal_apps/pmo_fibercontrol_targets_app.py`. MEASURED decorator on `run_target`:
`cpu=(1.0, 1.0)`, `timeout=24*60*60`, `retries=modal.Retries(max_retries=3)`. One task per
container, so 18 campaigns need 18 containers.

Overridable identity, which is what let arm C deploy without touching arm B:

    PMO_FIBERCONTROL_APP      default compose-pmo-fibercontrol-targets
    PMO_FIBERCONTROL_VOLUME   default compose-pmo-fibercontrol

Baked into the image with `copy=True` (so fixed at **build** time, not launch time):
`src/`, `scripts/`, the **whole** `configs/` directory, `diagnostics/parent_edit_cycles/prepared/init_20260921.json`,
`diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json`,
`modal_apps/pmo_population_v1_app.py`, and `ASSET_DIR` =
`diagnostics/pmo_ivg_oracle_parity/ivg_oracle_assets` (71 MB, **gitignored**).
Env: `PYTHONPATH=/root/compose/{src,scripts}:/root/compose`, `OMP/OPENBLAS/MKL_NUM_THREADS=1`,
and a pinned `PYTHONHASHSEED`.

Note the comment in the source: the whole `configs/` directory is baked because `load_contract`
reads a base contract the scored contract references, and "guessing the closure file-by-file
already cost one launch."

### Seed derivation — this fully explains every seed number in the tables

    def derived_seed(task, base_seed, replicate=0):
        return int(base_seed) + 1000 * TARGET_INDEX[task] + replicate

`TARGET_INDEX` is a fixed 23-entry table (deliberately not `hash()`, which is
`PYTHONHASHSEED`-salted): celecoxib_rediscovery 0, albuterol_similarity 1, mestranol_similarity 2,
thiothixene_rediscovery 3, troglitazone_rediscovery 4, median1 5, isomers_c7h8n2o2 6,
perindopril_mpo 7, gsk3b 8, jnk3 9, qed 10, amlodipine_mpo 11, fexofenadine_mpo 12,
osimertinib_mpo 13, ranolazine_mpo 14, sitagliptin_mpo 15, zaleplon_mpo 16, valsartan_smarts 17,
deco_hop 18, scaffold_hop 19, isomers_c9h10n2o2pf2cl 20, median2 21, drd2 22.

With `base_seed = 20260923` that gives, for the six ablation objectives at replicates {0,2,3}
(plus 60 for albuterol and 80 for gsk3b):

    celecoxib_rediscovery   20260923 / 20260925 / 20260926
    albuterol_similarity    20261923 / 20261925 / 20261983
    gsk3b                   20268925 / 20268926 / 20269003
    ranolazine_mpo          20274923 / 20274925 / 20274926
    scaffold_hop            20279923 / 20279925 / 20279926
    isomers_c9h10n2o2pf2cl  20280923 / 20280925 / 20280926

### The contract

`configs/pmo_population_controller_v1_scored_contract_corrected.json`, envelope
`payload_sha256 = ba9515615b341019...`. Payload keys (19): `budget`, `controller_contract`,
`controller_contract_sha256`, `corrected_source_capsule_manifest`, `corrected_worker_app`,
`corrected_worker_path`, `implementation_sha256`, `modal_launch_authorized`, `oracle`,
`oracle_calls_authorized`, `prior_launch_terminal_failure`, `promotion`, `runtime`,
`schema_version`, `scientific_question`, `scored_launch_authorized`, `status`,
`supersedes_payload_sha256`, `task_roles`, `tasks`.

`budget` MEASURED: `charged_calls_per_task 250`, `charged_calls_total 750`,
`initialization_calls_per_task 16`, `queries_per_round 16`, `max_rounds 64`,
`candidate_calls_per_task 234`, `cpu_per_worker 1`, `automatic_retries 0`, `backfill False`.
**The ablation ran at 1,008 charged calls, not 250** — the launcher passes `--budget`, and the
contract's 250 belongs to the earlier pilot. `status` is
`FROZEN_FAIL_CLOSED_PENDING_NEW_EXPLICIT_PAYLOAD_AUTHORIZATION` with both
`scored_launch_authorized` and `modal_launch_authorized` **False** — this contract is a frozen
record, and `pmo_fibercontrol_targets_app.py` does **not** consult it at run time (`CONTRACT` is
assigned at line 45 and referenced nowhere else). `load_contract` in
`src/compose_v4/experiments/pmo_population_v1.py` verifies it only on the *other* PMO app path.

`implementation_sha256` pins **8** files: `modal_apps/pmo_population_v1_app.py`,
`src/compose_v4/control/{bootstrap_pool_continuity,pmo_credit,pmo_population_controller,pmo_realization,program_campaign}.py`,
`src/compose_v4/experiments/{pmo_oracle_assets,pmo_population_v1}.py`.
**The proposal-synthesis path is NOT among them** — so `dynamic_program_synthesis_v21.py`,
`pmo_uniform_chain.py` and `pmo_binding_intervention.py` are unpinned, and arm identity rests on
the launch receipt's `git_commit` plus the deployed image.

### The three arm modules

    src/compose_v4/control/pmo_legal_mark_sampler.py   179 lines
        LegalMarkSampler(graph).draw(rng, max_attempts=512) -> (rule, action) | None
        CHEAP_RULES = atom_delete, atom_insert, bond_reorder, bond_reroute, cycle_open,
                      ring_system_restate        (enumerated exhaustively)
        LAZY_RULES  = cycle_close, atom_restate_semantic
                      (sampled by rejection from the enumerator's own tentative set)
        Exactness argument is in the module docstring: family chosen with probability
        proportional to |S_f|, element uniform within, reject on inadmissibility, so
        P(a | accepted) = 1/|A(x)|. Choosing the family UNIFORMLY would define a different law.

    src/compose_v4/control/pmo_uniform_chain.py       216 lines   ARM B
        ENV_FLAG = "PMO_UNIFORM_CHAIN"   STAGE_NAME = "uniform_legal_chain"
        chain_arm_enabled(), uniform_legal_chain(...), STATS + snapshot_stats(reset=True)

    src/compose_v4/control/pmo_binding_intervention.py 508 lines  ARM C, frozen at e11cd89d
        ENV_FLAG = "PMO_BINDING_REBIND"  STAGE_NAME = "created_atom_rebinding"
        MAX_JOINT_CANDIDATES = 256
        binding_rng(run_seed, proposal_id)      blake2b, never hash()
        created_operands(record)                excludes an atom_insert's own birth slot
        execute_rebound_program(...)            per-step resampling, executor decides admissibility
        execute_program_graph_rebound(...)      production seam; rebuilds + reschedules + replays
        binding_arm_enabled()                   raises if PMO_UNIFORM_CHAIN is also set
        STATS + snapshot_stats(reset=True)

Both arms hook the same file, `src/compose_v4/control/dynamic_program_synthesis_v21.py`:
**arm B at line 389** (`if chain_arm_enabled():` → `uniform_legal_chain` at 397, after
`source, program, binding, metadata = result` so it inherits arm A's synthesis-success filter and
its arbitration RNG), **arm C at line 433** (`if binding_arm_enabled():` →
`execute_program_graph_rebound` at 444, replacing the `execute_program_graph` call inside
`_generate_channel_pool`). Arm C's is the only call site of the rebound executor in the entire
tree.

### Launcher and receipts

`tools/launch_pmo_fibercontrol_targets.py` with `--targets --budget --rounds --queries
--base-seed --replicate --stage {smoke,scored} --receipt` plus `--uniform-chain-arm` /
`--binding-rebind-arm` (refused together). It `spawn`s, never `call`s, so targets are submitted
without waiting. The arm appears in the label and the namespace so two arms cannot share one.

Receipts, both directories committed:

    diagnostics/pmo_fibercontrol_targets_v1/        5 files, ARM B launches (20260925T2003xx)
    diagnostics/pmo_fibercontrol_targets_armc_v1/   5 files, ARM C launches (20260925T2336xx)
        including launch_scored_20260925T233700Z_replicate60_recovered.json, reconstructed
        after two same-second launches collided on the receipt path. The path is now
        arm/replicate-scoped and refuses to overwrite.

---

## 12. The reduction scripts — now in the repo, not /tmp

Every number in the reports came from scripts that lived in `/tmp`. They are promoted to
`compose_pmo_chain/scripts/reductions_20260927/` with a README giving the exact invocation and
the required `MODAL_PROFILE` per script: `arm_metrics.py`, `arm_status.py`, `t4_full.py`,
`t4_eta.py`, `armc_health.py`, `full_report.py`.

**Still ephemeral and not promoted** (probe/one-shot, listed so nobody hunts for them):
`/tmp/armc_admission.py`, `/tmp/armc_consolidated.py`, `/tmp/armc_prodpath.py`,
`/tmp/armc_mut*.py` (the mutation batteries), `/tmp/armc_selfconsistency_tests.py`,
`/tmp/find_armA.py`, `/tmp/scan_armA.py`, `/tmp/gather_arms.py`, `/tmp/t4_timeout.py`,
`/tmp/t4_table.py`, `/tmp/t4_two_col.md`, `/tmp/full_report.py` inputs. The mutation batteries
matter most if arm C is ever revised — they are reconstructable from
`diagnostics/pmo_armc_verification_v1/VERIFICATION.md`, which lists all 13 mutations and their
verdicts.

`~/compose_pmo_ablation` (unversioned, 140 KB) holds exactly:
`frozen_config_v1.json`, `CORRECTIONS_AND_BINDINGS.md`, `MECHANISM_REPORT_PART{1..5}.md`,
`armC/{FROZEN_ARM_C_V1.md,dev_panel.py,dev_panel_v1.json}`, and
`inputs/pmo_1k_{final,auc,clean,prov}.json`. **`inputs/pmo_1k_final.json` is the only local copy
of arm A's per-seed best/top10/auc** and is what every A/B table reads.

---

## 13. Things I do not know, stated as such

* I have not audited `archive/`, `output/`, `tmp/`, `upload/`, `results/`, `data/`,
  `experiments/`, `third_party/` or the four `COMPOSE_ICLR_2027_*` directories. I do not know
  what references them.
* I do not know which of the 533 `diagnostics/` entries are pinned by a config. The check is
  `grep -rl "<name>" configs/ src/ modal_apps/` before moving anything; at least two are pinned
  by T4 contracts and one gitignored directory is baked into the PMO image by literal path.
* I have not verified that `configs/` is absent from the training run-identity fingerprint;
  that came from `learnings.md`, not from my own reading of the fingerprint code.
* The MOOD receptor/binary URLs are an external dependency I have not checked for stability.
* Whether the remaining 8 revival contracts correspond exactly to the 15 no-checkpoint cells is
  unverified; I matched deltas by contract hash but did not cross-check the cell lists.
* The repo remote now reports **moved to `https://github.com/magantiro/compose_rgm.git`**.
  Pushes succeed through the redirect. I have not updated any remote URL.
