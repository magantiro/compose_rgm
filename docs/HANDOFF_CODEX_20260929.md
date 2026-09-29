# Handoff — repository state, paper mapping, and everything else worth knowing

Written 2026-09-29. Audience: an agent picking this up cold. Everything below is
either **measured this session** (stated as such) or **cited to a file you can
open**. Where I am uncertain I say so rather than guessing — several confident
claims made earlier in this session turned out to be wrong, and they are
recorded here as corrections so you do not repeat them.

---

## 1. Orientation

COMPOSE is a generative process over molecular graphs where every transition is
an executable chemical rewrite between complete, valid molecules. A
goal-independent reference law `R_θ` is learned once; task information enters
only through controllers at inference.

**Do not confuse this with the sibling repo `KoshaTx/compose`** (a discrete-
*diffusion* operator-editing framework). Same name lineage, different process.
This is the RGM/CTMC line.

Measured scale of the working tree (`t4-objective-dynamic-reset-20260916`):

```
src/compose_v4   607 files   45M     control/ 149 py, experiments/ 302 py,
                                     rewrite/ 34, data/ 63, benchmark/ 12
tests            635 files   48M     6,438 tests collected
scripts          366 files  8.7M     drivers, readers, audits, figure builders
modal_apps       288 files   11M     cloud entry points
configs          273 files  4.9M     41 of them pin runtime_inputs_sha256
docs             493 files   10M
diagnostics    14,328 files  1.6G    working artifacts, NOT published results
tools            257 files  8.1M
experiments           8 files 128K   task/paper guides (small on purpose)
```

129 local branches. All replicated to origin **except `compose-iclr`**, which is
8 commits ahead with untracked material.

---

## 2. The paper → code map (verified this session)

Current submission: ICLR 2027, *"COMPOSE: Molecular Generation and Optimization
with a Reusable Stochastic Rewrite Process"*. The source tree used at submission
is **not tracked in git** — it lives in three untracked top-level directories in
the main checkout:

```
COMPOSE_ICLR_2027_zip5_appendix_results/   newest file 2026-09-26 08:38   <- the submitted one
COMPOSE_ICLR_2027_revision4/               2026-09-26 05:55
COMPOSE_ICLR_2027_Overleaf/                2026-09-24 21:42
```

The deadline was 2026-09-26 08:49. All three are `tracked=0`. The owner has said
the paper itself does not need to be tracked.

`experiments/paper/README.md` (added this session) holds the live mapping. The
verified correspondences:

| Paper | Code |
|---|---|
| §2.1, §C.1 state space, executor `A(x)` | `src/compose_v4/chem/`, `src/compose_v4/rewrite/` |
| Table 4, eight primitive rewrites | `src/compose_v4/rewrite/operators.py` |
| §2.2, §C.2–C.3 reference law `R_θ` | `src/compose_v4/model/`, `src/compose_v4/gm/` |
| §2.3, §C.4 structured programs | `src/compose_v4/control/` |
| §2.4–2.6 controllers | `src/compose_v4/control/` |
| §2.7, §D future values / Doob / twisted SMC | `experiments/editing_v2_bridge_control.py` |
| §3.1, Table 6 reference-law eval | `scripts/emit_reference_law_data_backed.py` |
| §3.2, Table 1 fragment | `benchmark/fragment_constrained.py`, `_runner.py` |
| §3.3, Table 10 QED editing | `scripts/evaluate_qed_controlled_rollouts.py` |
| §3.4, Tables 2, 11 PMO-1K | `experiments/pmo_population_v1.py`, `modal_apps/pmo_population_v1_app.py` |
| §3.5, Table 3 T4 | `experiments/t4_fiber_campaign.py`, `modal_apps/t4_*` |
| §G.2 quality metrics | `eval/molecular_quality.py` |

**Naming correspondence, stated in the paper's §E.1 and verified against
`rewrite/`:** paper-facing `cycle_close`, `cycle_open`,
`ring_aromaticity_restate` are recorded in code as `cycle_insert`,
`cycle_attach`, `ring_system_restate`. The other five agree. All eight present.

### Runnable surface — be precise about this

Measured: `evaluate_qed_controlled_rollouts.py` and the T4 apps have a CLI.
`fragment_constrained_runner.py` and `pmo_population_v1_app.py` do **not** — the
PMO app is driven by `tools/launch_pmo_population_v1*.py`, and fragment has only
`scripts/fragment_motif_extension_pilot_v1.py` as a CLI driver. There is **no
single clean "run benchmark X" command for all four benchmarks.** If a
reproduction guide is wanted, this is the gap to close, and it should be closed
by writing the drivers rather than by describing commands that do not exist.

### Nothing runs without `R_θ`

The reference checkpoint lives on a Modal volume
(`compose-v4-artifacts : compose-v4-stage3-flexible-graft-3k-1ac6f19-v1/checkpoint.best_so_far.pt`),
not in git. The paper says so itself: *"a fresh checkout alone is insufficient to
recover the reported measurements."* Also needed externally: QuickVina2 +
receptors + boxes (T4), TDC oracle pickles for drd2/gsk3b/jnk3 (PMO, ~71 MB,
downloaded by PyTDC on first use), a Modal account.

---

## 3. THREE distinct T4 result sets — the single easiest thing to get wrong

I got this wrong twice in one session. All three exist, all three are real, none
corrects another.

| # | Artifact | Comparators | Budget | Where it appears |
|---|---|---|---|---|
| **A** | `diagnostics/T4_FROZEN_RESULT_v1.json` | InVirtuoGen only | 250 vs IVG 1,000 | **the ICLR submission, Table 3** |
| **B** | `diagnostics/t4_combined_table.json` | GenMol, RetMol, GraphGA | 500 vs GenMol 3,000 | the NeurIPS **workshop** package |
| **C** | `diagnostics/compose_status_report/report_20260926.md` | all four | 250 | a 09-26 snapshot, replicates in flight |

**A backs the paper.** Verified this session: δ=0.4 cells match the artifact
exactly; δ=0.6 equals the artifact's 14 paired cells **plus FA7 seed 1 filled at
its seed score −6.4** (COMPOSE returned nothing feasible beating the lead there;
IVG wins at −7.7; the artifact correctly records a blank). Win counts
recomputed: **10 of 15 at δ=0.4, 13 of 15 at δ=0.6 = the abstract's 23 of 30.**

**B is a different experiment.** Its reducer is `scripts/t4_combined_table.py`,
output typeset into `paper_gem_neurips2026/tables/t4_summary_row.tex`
(19W/6L/1T, coverage 26/30, mean −0.335 [−0.651, −0.019]). Its per-cell values
disagree with A by construction. Do not mix them.

**C is a partial snapshot** and says so in its own header: *"Nothing here is a
finished panel"* — T4 was 10 of 60 replicate cells complete at the time.

### A real arithmetic error in A, preserved not fixed

At δ=0.6 the artifact stores `sum_gap = -7.6`; the rows sum to **−8.5**, and two
independent routes agree with the rows (`compose_sum − ivg_sum` = −8.5;
`mean_gap × paired_cells` = −8.4994). The stored field **understates** COMPOSE's
margin by 0.9 kcal/mol. The artifact is LOCKED and was deliberately not edited.

---

## 4. PMO state

**The paper's PMO-1K table (Tables 2 and 11)** is backed by
`diagnostics/pmo_ablation_frozen_v1/inputs/pmo_1k_final.json` — 23 objectives ×
seeds × {best, top10, auc}. Verified this session:

- Five objectives carry a **fourth** seed (albuterol, celecoxib, mestranol,
  thiothixene, troglitazone). The paper reports three. Dropping the seed whose
  id ends **`983`** (the `noprescreen_v2_rediscovery` replicate) reproduces the
  published mean *and* standard deviation on all five.
- **21 of 22 objectives match exactly**, and both headline means reproduce:
  **0.563** final top-10 and **0.482** AUC-Top10.
- **JNK3 is the one gap**: artifact 0.246/0.210 against published 0.245/0.208. A
  fourth JNK3 run exists (`pmo_learned_trio_v1`, seed 20269928), so the
  committed artifact holds a different third seed than the paper used. Means
  unaffected.

**A much larger PMO campaign set is NOT reduced anywhere.** 78 scored arms
across 11 campaigns (09-23/09-24), **23 distinct tasks, 592,000 declared charged
calls** (10,000×50, 3,000×24, 5,000×4). Every one has a launch receipt carrying
`task`, `seed`, `replicate`, `budget`, arm flags, `git_commit`, `volume` and
`modal_call_id` — and **zero committed result files**. Those results exist only
on Modal volumes. Whether the volumes still exist is unchecked.

**Arms A/B/C** (a separate, later ablation) are reduced in
`diagnostics/pmo_abc_ablation_v1/reduction_v1.json`, status `PROVISIONAL` with
arm B 2/18 still running at reduction time.

---

## 5. Where artifacts live, and the preservation posture

The owner's decision this session: **the repo ships machinery, not scores.**
Untracked and gitignored (backed up to `~/compose_paper_artifacts_backup`,
252 KB, 23 files):

```
diagnostics/pmo_ablation_frozen_v1/     diagnostics/pmo_abc_ablation_v1/
diagnostics/compose_status_report/      tools/reproduce_{paper_tables,t4_table,pmo_tables}.py
tests/test_reproduction_tools.py
```

Two score artifacts **predating** this work remain tracked:
`diagnostics/T4_FROZEN_RESULT_v1.json` (in 18+ pushed commits) and
`diagnostics/t4_combined_table.json`. Untracking them changes nothing already
published, and other committed tooling reads them. Open decision.

`tools/preservation_inventory.py` (kept, works standalone) reports what git does
not protect, classifying each file as tracked / regenerable-by-a-named-recipe /
at risk, and verifies a backup against the **saved manifest** rather than a
fresh scan.

**Standing hazard, paid for repeatedly:** `/private/tmp` is reaped mid-session.
This repo has already lost a worktree that way and nearly lost a 70,301-entry
corpus. Four inputs to the 09-26 report existed only in `/tmp` until committed
this session.

---

## 6. Environments — two kernels, not interchangeable

```bash
# Core: executor, reference process, fragment, QED editing, T4
python 3.11 · rdkit 2024.3.5 · torch 2.4.0 · numpy 1.26.4 · scipy 1.13.1 · networkx 3.3

# PMO only
python 3.11 · rdkit 2023.9.6 · PyTDC 1.1.15 (--no-deps) · numpy 1.26.4 · setuptools 69.5.1
```

PMO needs its own because PyTDC 1.1.15 pins `rdkit<2024.3.1`; its image carries
one rdkit, so the PMO runtime — oracle *and* executor — runs 2023.9.6. A PMO
number computed locally under 2024.3.5 needs a parity statement.

**The laptop `.venv` is neither** — python 3.12 / rdkit 2026.03.6 / numpy 2.5.3.
A pinned env is one minute to build:
`uv venv --python 3.11 <dir>` then
`uv pip install --python <dir>/bin/python rdkit==2024.3.5 numpy==1.26.4 scipy==1.13.1 networkx==3.3 torch==2.4.0 pytest`.
torch is needed only because `t4_fiber_campaign` imports it transitively — but
it **is** needed, because without it you must transcribe the eligibility gate
rather than import it, and a transcribed gate cannot fail usefully.

Measured parity: rdkit 2023.9.6 vs 2024.3.5 is byte-identical on every transport
artifact tested. **2026.03.6 is not** — 1,472 of 37,784 off-path intermediates
(3.90%) are a *different molecule* by InChI. Canonical SMILES is used as a cache
key, so a spelling difference is not cosmetic.

---

## 7. Repo health — measured, not asserted

- **`pytest tests/` does not run to completion.** One collection error in
  `tests/test_editing_v2_semantic_t1_panel_cache.py`:
  `SemanticCapabilityCellError: semantic capability registry bindings have
  drifted`, raised from
  `data/editing_v2_semantic_capability_cells.load_semantic_capability_cell_registry`.
  The check compares pinned `classifier_implementation_sha256`,
  `action_codec_schema_version`, active families, data lanes and partition roles
  against live values. **Pre-existing**, not introduced by recent work. 6,438
  tests collect; `--ignore` that one file and the suite runs.
- **The suite has a large pre-existing failure population.** Measured this
  session, ignoring that one file: at 77% of collection (5,025 of 6,438 tests,
  run truncated by a 40-minute cap) — **4,688 passed, 231 failed, 101 errors.**
  The 101 errors match the figure recorded in `learnings.md` exactly, and 231
  failures at 77% extrapolates toward the 292 recorded there for a full run, so
  this is the documented baseline rather than new breakage. The bulk is the
  editing-V2 family cascading from the same registry drift. **Do not read a red
  suite here as something you broke** — attribute against this baseline first,
  ideally by re-running the same selection at the branch base.
  Caveat on my own measurement: the wrapper's exit code came from a trailing
  `tail`, not from pytest, so it read 0 despite the truncation. Check the
  summary line, not the exit status.
- **`ruff check src/` reports 514 findings** (314 auto-fixable). Mostly newer
  rule classes (RUF/UP) on a large legacy surface, not fresh breakage.
- `scripts/prelaunch_gate.py` does **not** pass, because it lints `src/`. It is
  *not* the gate for T4 launches — those are gated by `tools/preflight.py` plus
  the app's own `_local_task` runtime-input verification. Know which gate
  governs which path before treating red as a blocker or green as coverage.

---

## 8. The contract / hash-chain system — how not to break it

41 configs pin `runtime_inputs_sha256`. A launch validates every pinned file's
sha256 **inside the container** before running. Consequences:

- **Editing a pinned file invalidates every contract pinning it.** Re-pinning is
  a deliberate act; if the contract carries
  `status: AUTHORIZED_FOR_SCORED_LAUNCH` naming a payload hash, re-pointing it
  without the owner naming the new value **manufactures consent**. Do not.
- `src/compose_v4/experiments/production_successor_kernel.py` **must not be
  edited** — it fixes `process_identity_sha256`, which every artifact is keyed on.
- Kernel performance work is a deliberate **process revision** (re-seal the
  chain, re-run Gate 0, recompile every row), never an incidental speedup.
- Catalog drift: any claim-bearing evaluation must abort if the reconstructed
  RingCore catalog fingerprint differs from `639ff6078c32d43c`
  (`experiments/editing_gate_zero_runtime.py:114`). `neutralize_catalog_drift()`
  is a scratch helper only and must never appear in claim-bearing code.
- A stale binding that **validates** is worse than one that raises. Prefer
  live-loader comparison over a source constant; a constant-vs-constant check is
  self-consistent by construction.

---

## 9. Gotchas that have each cost real work

**Measurement discipline**
- A metric that cannot vary is not a measurement. Before reading a null, check
  the statistic can respond to the thing you changed.
- Estimate the per-draw **rate** before reporting a zero. A zero at a budget
  where the expected count is ~1 discriminates nothing.
- A comparison whose expectation is recomputed from the code under test cannot
  fail. Expectations must come from an independent source.
- Mutation-test the tests: a mutation that fails to apply must **abort**, not
  score as killed. A guard is only tested where it binds.
- Parallelism is free during implementation, **not** during measurement.

**Tooling traps**
- `modal app list` truncates the description column — grepping it returns false
  negatives. Read the task-count column, or better, check the volume artifact.
- `modal volume ls <vol> <subpath>` silently returns the **parent** listing.
  `modal volume get` collapses a directory onto one path and exits 0. Use the
  Python API (`modal.Volume.listdir`).
- Modal volumes exist under multiple **profiles**; `MODAL_PROFILE` is not
  inferable from a volume name, and a cross-profile listing reports live work as
  absent. Two volumes share the name `compose-v4-artifacts`.
- `git for-each-ref refs/remotes` enumerates *local tracking* refs — a pushed
  branch nobody fetched reads as absent. Use `git ls-remote`.
- zsh does **not** word-split unquoted expansions: `python3 $c` passes the whole
  string as one filename. This produced a false exit-2 in this very session.
- `vm_stat`'s `Pages free` is not free memory on macOS; use `memory_pressure`.
- `pkill -f <script>` misses spawn-pool children (`from multiprocessing.spawn`).
- `ppid=1` does **not** mean abandoned — a detached job and a leaked orphan look
  identical. Discriminate by CPU *and* a growing artifact.

**Scientific traps specific to this work**
- **Best-of-N is biased and the bias grows with N.** On a fixed GenMol cell set
  the summary statistic alone moved the total 10.4 kcal/mol. Fix statistic and N
  on both sides of any cross-system verdict.
- **T4 docking is not reproducible across runs**: `qvina02` is seeded but
  `obabel --gen3D` is not. One molecule scored −7.5 / −8.30 / −8.8 across three
  runs. Quote aggregates and win counts, not single-cell 0.2–0.7 margins.
- **Three PMO oracles reward leaving the drug-like manifold** — gsk3b, jnk3,
  drd2 are fingerprint predictors; measured `r(score, QED)` = −0.705 and −0.750.
  Top molecules carry hypervalent iodine and trioxide chains. Read the SMILES.
- **A zero-oracle gate ledger looks exactly like a scored run** — same schema,
  task name and `oracle_protocol`. One holds a celecoxib "best" of 0.9167
  against a real ~0.25. Detect via the committed report declaring
  `new_charged_oracle_calls == 0`, not via a folder named `dryrun`.
- **AUC is structurally depressed at short budgets.** Below one logging period
  the grid never fires and the run is a single trapezoid — exactly **50%**
  removed at any budget under `frequency`, not the naive
  `frequency/(2·budget)`. Never compare a 250- or 1,000-call AUC to a published
  10,000-call figure.
- **Built-but-inert mechanisms are a recurring failure here** — at least six
  found (region law behind an opt-in keyword, `exact_early_ring`,
  `allocation_priority`, `donor_program`, `zero_support_fallback`, a completion
  law reaching 20% of its intended path). After threading a keyword, grep
  **call sites**, and watch for adapters that *replace* a path rather than
  extend it.

---

## 10. What changed this session

On `t4-objective-dynamic-reset-20260916`, all pushed, all verified against
origin by SHA:

- Committed 42 previously-unpushed commits plus new work; added gitignore rules
  for 293 MB of redownloadable oracle pickles and cumulative round snapshots.
- Added `experiments/paper/` — the paper→code index (all links verified to
  resolve), and `experiments/{t4,pmo}/` task guides with input manifests.
- Added `tools/preservation_inventory.py` and `tools/verify_experiment_inputs.py`
  (strict: rejects unknown schemas, missing/malformed hashes, zero-verified).
- Preserved four `/tmp`-only inputs to the 09-26 report.
- Then, on the owner's direction, **untracked the score artifacts and the three
  score-reading reproducers** and reframed `experiments/paper/` around
  machinery rather than stored numbers.

**Nothing under `src/`, `modal_apps/`, `configs/`, `recipes/` or `paper_*` was
modified at any point.**

---

## 11. Open decisions, honestly labelled

1. **Is the stale default branch an issue?** `main` is from 2026-07-19 — 50 src
   files against 1,266 here, no `configs/`, no `experiments/`. It has no scores
   (consistent with the owner's wish) but is not the code behind the paper. This
   **only matters if the anonymized repository link mirrors the GitHub default
   branch.** If the snapshot was uploaded directly, `main` is irrelevant. I do
   not know which, and I previously stated this more absolutely than the
   evidence supports.
2. **The two pre-existing tracked score artifacts** (§5) — untrack or leave.
3. **The 78 unreduced PMO arms** (§4) — 592k declared calls with results only on
   Modal volumes; nobody has checked whether the volumes survive.
4. **The collection error** (§7) — one stale binding blocking `pytest tests/`.
5. **No single-command driver per benchmark** (§2) — the real gap if end-to-end
   reproducibility is the goal.

---

## 12. Where to read next

| File | Why |
|---|---|
| `AGENTS.md` | repository-wide scientific and execution contract |
| `.claude/context/learnings.md` | the long-form record; dense, dated, authoritative on gotchas |
| `docs/START_HERE_ICLR.md` | controller status (note: last substantive edit 2026-09-22) |
| `docs/PAPER_TO_CURRENT_CODE.md` | submitted-paper vs post-submission boundary |
| `experiments/paper/README.md` | the paper→code map this handoff summarizes |
| `docs/T4_REPLICATE_PANEL_BRIEF_20260925.md` | the replicate panel situation in full |

**Do not infer authority from filenames** such as `CURRENT`, `PLAN` or
`HANDOFF`. This repository deliberately preserves superseded plans as part of
the scientific record.
