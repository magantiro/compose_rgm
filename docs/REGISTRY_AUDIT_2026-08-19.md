# Registry audit — 2026-08-19

Are the four registry/index documents current? Checked claim by claim against the
filesystem and `git log` at `HEAD = 42ca8e1` (2026-08-19 01:16), branch
`codex/editing-v2-successor-fiber-fastpath`.

**This audit created exactly one file — itself. Nothing was deleted, moved,
renamed or edited.** Nothing below is a recommendation to remove anything.
"Stale" here means *the index asserts something that is no longer true*, never
"should be deleted". The `paper*/` directories and the superseded plan documents
are preserved deliberately and that is correct; where they are involved the
finding is about a *pointer* to them, not about them.

Every claim below cites a file and line number so it can be checked
independently. Where I could not establish whether something is stale or
deliberately frozen, it is filed under **UNCERTAIN** rather than guessed.

---

## Summary

| Document | Last commit | Commits since | Verdict |
|---|---|---|---|
| `docs/ESTATE_REGISTRY.md` | 2026-08-10 22:17:42 (`f413883`) | 400 | Rule intact; **the whitelist has been overtaken** by a governing plan it does not name. Two counts/facts now false. |
| `docs/ARTIFACT_INDEX.md` | 2026-07-19 04:06:08 (`f20c49d`) | 1247 | **Most stale of the four.** No dead paths, but its "Canonical" and "Current" labels are contradicted by three newer documents, and it is internally inconsistent. |
| `docs/CLAIM_LEDGER.md` | 2026-08-12 04:14:15 (`9e18557`) | 325 | Artifact paths all resolve; **cross-reference points at a superseded doc**; blockers stale; the repo's largest measured result has no row. |
| `docs/PROJECT_BOARD.md` | 2026-08-11 15:09:46 (`10630a1`) | 359 | **Two hard falsehoods** (its only in-flight item; its P1 blocker), both already false when the file was last written. |

Commands used for the dates:

```
git log -1 --format=%ci -- docs/ESTATE_REGISTRY.md      # 2026-08-10 22:17:42 -0400
git log -1 --format=%ci -- docs/ARTIFACT_INDEX.md       # 2026-07-19 04:06:08 -0400
git log -1 --format=%ci -- docs/CLAIM_LEDGER.md         # 2026-08-12 04:14:15 -0400
git log -1 --format=%ci -- docs/PROJECT_BOARD.md        # 2026-08-11 15:09:46 -0400
git rev-list --count f413883..HEAD                      # 400
```

The single most consequential finding is not in any one document — it is
**§5, contradiction C1**: three documents give three different answers to "which
plan governs", and the one that currently claims to govern
(`docs/MASTER_PLAN_EXPERIMENTS_AND_COMPARATORS.md`, last commit 2026-08-15) is
named by none of the four registries.

---

## 1 · `docs/ESTATE_REGISTRY.md`

### What it claims to index

A whitelist of currency for the whole estate: line 5, "**The rule, and it is the
whole point:** exactly three files are current", plus a map of the four `paper*/`
directories, the census of ARCHIVED-stamped documents, and one flagged unresolved
conflict.

Last commit 2026-08-10 22:17:42 (`f413883`, "docs: whitelist three current files;
preserve everything else as history"). 400 commits have landed since. The 30 most
recent commits span the h_φ/QED controller lane, the 128-source validation, and
the MOLLEO probes — none of which existed when this file was written.

### Still true (verified, do not "fix")

- **The ARCHIVED census (lines 45–52) is complete and accurate.** All eleven
  listed files exist and carry an ARCHIVED banner within their first three lines.
  A sweep of every `.md` under `docs/` and the four `paper*/` directories found no
  *other* file carrying such a banner. (Two files match the word "archived" in
  their opening lines without being stamped —
  `paper_iclr_control_substrate/EXPERIMENT_CHECKLIST.md:3` and
  `paper_iclr_stochastic_rewriting/EXPERIMENT_CHECKLIST.md:3` — that is prose,
  not a banner.)
- **Line 25, "Each carries a `_STATUS.md`"** — verified, all four present.
- **Lines 57–66, the CLAUDE.md conflict, is still unresolved and still worth
  flagging.** `CLAUDE.md:3` still reads "**Rewrite Generator Matching for
  validity-preserving molecular generation.**" and `CLAUDE.md:6-7` still reads
  "Generator Matching learns their contextual firing rates."
- **Line 24, "that decision is open and tracked in `docs/PROJECT_BOARD.md`"** —
  still open; `docs/PROJECT_BOARD.md:176-182` still carries it as P3.

### FALSE or out of date

**F1.1 — the document count. Lines 15–16:**

> `docs/` holds
> 84 markdown files accumulated across several earlier framings of the project;

Actually true today: **141** markdown files at the top level of `docs/`, **201**
including subdirectories.

```
ls -1 docs/*.md | wc -l                                  # 141
git ls-tree -r --name-only HEAD docs | grep -c '\.md$'   # 201
git ls-tree --name-only f413883 docs/ | grep -c '\.md$'  # 86  (at write time)
```

The claim was roughly right when written (86, stated as 84). Fifty-five new
top-level documents have accreted since — which is the registry's own thesis
about accretion, now measurable against it. The same "84" is repeated at
`CLAUDE.md:27`, which auto-loads every session.

**F1.2 — the filesystem-mtime claim. Lines 34–37:**

> The two most recently committed (`paper_arxiv`, `paper_iclr_stochastic_rewriting`)
> share a commit, "paper: define compose naming hierarchy". Filesystem mtimes are
> all 2026-08-10 and are **not** evidence of recent editing — they reflect a
> checkout, not authorship. Git dates are the reliable signal.

"Filesystem mtimes are all 2026-08-10" is now false. `paper_arxiv/` contains
files with mtime **2026-08-18 19:35** — `main.pdf`, `main.aux`, `main.log`,
`main.bbl`, `main.fls`, `main.fdb_latexmk`, `main.out`. These are LaTeX build
outputs, all matched by `paper_arxiv/.gitignore`, and `git status` shows no
tracked change anywhere under `paper_arxiv/`; `main.tex` itself is unmodified
(mtime 2026-08-10 00:40). The *reasoning* of the paragraph therefore still holds
— git dates remain the reliable signal — but the stated fact does not.

> **UNCERTAIN:** somebody compiled the arXiv manuscript on 2026-08-18, eight days
> after the registry declared all four directories dormant, and one day before
> this audit. Whether that is meaningful (work resuming on `paper_arxiv/`, which
> would bear on P3) or incidental (a build to look at the PDF) cannot be
> determined from the repository. I did not investigate further and make no
> claim either way.

**F1.3 — the paper table's "Last commit" column no longer reproduces. Lines 27–32.**
Run today, `git log -1 -- <dir>` returns `f413883` (2026-08-10 22:17:42) for
**all four** directories, because that commit added `_STATUS.md` to each and
stamped banners on `paper/PAPER_STATUS.md` and `paper_arxiv/COMPLETION_PLAN.md`.
The table's dates only reappear if those files are excluded from the pathspec:

| Row | Registry says | `git log -1 --` today | Excluding the banner files |
|---|---|---|---|
| `paper/` | 2026-07-27 | 2026-08-10 `f413883` | 2026-07-27 `325dff4` ✓ |
| `paper_arxiv/` | 2026-08-01 | 2026-08-10 `f413883` | 2026-08-10 `f413883` ✗ |
| `paper_iclr_control_substrate/` | 2026-07-22 | 2026-08-10 `f413883` | 2026-07-22 `814b26c` ✓ |
| `paper_iclr_stochastic_rewriting/` | 2026-08-01 | 2026-08-10 `f413883` | 2026-08-01 `b299ff4` ✓ |

`paper_arxiv/`'s stated 2026-08-01 does not reproduce by any pathspec I tried,
because `f413883` also modified `paper_arxiv/COMPLETION_PLAN.md`. This is
bookkeeping, not a scientific error — but a reader running the documented check
gets four wrong answers.

**F1.4 — the size/file-count column has drifted. Lines 29–32.**
`paper_arxiv/` is listed as "232K, 16 files"; on disk it is **1.1M, 26 files**
(17 tracked). The delta is the ignored LaTeX build from F1.2. The other three are
each one file larger than stated — the `_STATUS.md` added by the same commit that
wrote this table (`paper/` 792K/48 → 796K/49; `paper_iclr_control_substrate/`
24 → 25 files; `paper_iclr_stochastic_rewriting/` 27 → 28 files).

### Contradicted by a newer document

**F1.5 — the whitelist is contradicted by a plan that outranks it.** Line 11:

> | `docs/EXPERIMENT_PLAN.md` | the only experiment plan |

`docs/MASTER_PLAN_EXPERIMENTS_AND_COMPARATORS.md` (94 KB, last commit
**2026-08-15 14:42**, `d677bdc` — three days *newer* than anything on the
whitelist) opens at line 1:

> `# COMPOSE — CURRENT GOVERNING EXPERIMENTAL PLAN`

and lines 3–6:

> **This section is authoritative. Everything below `ARCHIVED PROVENANCE` records
> how the plan evolved but does not authorize experiments, comparators, rescues,
> budgets, or claims. Where archived text conflicts with this section, this
> section governs.**

That file appears **nowhere** in `ESTATE_REGISTRY.md` — not on the whitelist, not
in the ARCHIVED census. Under the registry's own rule ("Everything else in this
repository is history until it is linked from one of them", lines 6–7) the
currently governing plan is classified as history. This is the registry's central
failure mode: the rule was designed so it "cannot go stale" (line 18), and it
hasn't — but the *whitelist* did, and the rule gives the wrong answer as a result.

Details of the resulting three-way conflict are in §5, C1.

**F1.6 — the registry is not registered.** `ESTATE_REGISTRY.md` has zero inbound
references anywhere in `docs/`, `CLAUDE.md`, `AGENTS.md` or `README.md`; the only
pointers to it are the four `paper*/_STATUS.md` notices. `CLAUDE.md:19-24` links
only the three whitelisted files. By its own rule, `ESTATE_REGISTRY.md` is
history. The same is true of `ARTIFACT_INDEX.md` and `docs/CLAIM_LEDGER.md`.

---

## 2 · `docs/ARTIFACT_INDEX.md`

### What it claims to index

Line 3: "This index separates current model artifacts from historical
diagnostics." Three sections: "Canonical research plans" (line 5), "Current
audits" (line 13), and a "Legacy pre-quotient trajectory bundle" (line 25).

Last commit **2026-07-19 04:06:08** (`f20c49d`, "Document A100 production launch
readiness"). **1,247 commits** have landed since — it predates the canonical
experiment plan by three weeks, the `run_v2_01` training line, the frozen `R_θ`,
the DRD2 oracle, the sealed panel, the entire h_φ/QED lane and all MOLLEO work.

### Still true

**Every path it lists resolves.** All three research plans, both audits, all five
`docs/*.md` targets, the legacy trajectory directory and all three diagnostic
scripts under `../scripts/diagnostics/` exist. There are no dead links. Its
problem is labelling, not rot.

### FALSE or out of date

**F2.1 — "Canonical research plans" is contradicted by the canonical plan. Lines 5–11:**

> ## Canonical research plans
> | [`research_plans/paper1_compose_methods.html`](research_plans/paper1_compose_methods.html) | COMPOSE RGM methods-paper plan |
> | [`research_plans/paper2_compose_lipid.html`](research_plans/paper2_compose_lipid.html) | COMPOSE-Lipid translational plan |
> | [`research_plans/compose_two_paper_execution_plan.html`](research_plans/compose_two_paper_execution_plan.html) | Integrated execution and dependency plan |

All three last committed **2026-07-20**. `docs/EXPERIMENT_PLAN.md:3-5` says:

> **STATUS: CANONICAL. This supersedes every prior experimental plan, including the
> manuscript's existing experimental section and the A/B/C framing used earlier in
> development. Do not execute or cite the old plan.**

and line 14: "**This file is the only current plan.**" Two documents both use the
word "canonical" for different objects. `docs/research_plans/README.md:3` doubles
down: "These three standalone HTML documents are the canonical project strategy
artifacts."

> **UNCERTAIN:** the Paper-2 COMPOSE-Lipid program is a real question, not just a
> label problem. The string "lipid" appears **zero** times in
> `docs/EXPERIMENT_PLAN.md`, `docs/MASTER_PLAN_EXPERIMENTS_AND_COMPARATORS.md`,
> `docs/DECISION_LOG.md` and `docs/PROJECT_BOARD.md`, and no commit in the last
> 200 mentions it. That is consistent with "deliberately parked" and equally
> consistent with "quietly abandoned". I cannot tell which, and the index gives
> no signal either way.

**F2.2 — "Current audits" lists a document that describes a retired model. Line 19:**

> | [`PROJECT_STATUS.md`](PROJECT_STATUS.md) | Dated completed/in-flight/queued workstream and claim boundary |

`docs/PROJECT_STATUS.md` is titled "**# Project status — 2026-07-19**" (line 1,
last commit 2026-07-20) and its "Live override" block says, at lines 22–26:

> Work has returned to the exact step-6,250 pancake model: a
> 100-sample inference rescue is running with canonical self-Grafts removed,
> and a validity-certified early-ring schedule is locally verified before any
> retraining decision.

This is **internally contradicted by `ARTIFACT_INDEX.md` itself**, lines 29–30,
about the very same checkpoint:

> They document the checkpoint that motivated the successor quotient; they must
> not be presented as results of the corrected model.

It is also contradicted by `docs/CLAIM_LEDGER.md:43` (S2), which records that the
whole-ring growth macro that `PROJECT_STATUS.md`'s live override is entirely
about is now **disabled** — "masked to empty support, never enumerated/sampled/
taught". And `PROJECT_STATUS.md:27-28` forwards the reader to
`48_HOUR_RESULTS_TRACKER.md` for "the current decision schedule", a document last
committed 2026-07-19.

**F2.3 — "Current audits" line 20:**

> | [`CURRENT_MODEL.md`](CURRENT_MODEL.md) | Source, teacher, operator, generator, sampler, and evidence contract |

Last commit **2026-07-19 04:44:46**. Its opening sections describe
`NullSourcePrior` / `DegreeBoundedCarbonTreePrior` source selection
(`docs/CURRENT_MODEL.md:5-17`) — the pre-`run_v2_01` framing. Under
`ESTATE_REGISTRY.md`'s whitelist this is history; under `ARTIFACT_INDEX.md` it is
current. See §5, C2. The file's *name* makes this worse than an ordinary stale
pointer: `f413883`'s own commit message singles out "CURRENT_MODEL" as one of the
documents that "sound authoritative" without being current.

**F2.4 — it indexes a vanished fraction of the artifact surface.** The index
covers 3 plans, 7 audits/docs, 7 legacy PNG/JSON files and 3 scripts. The current
artifact surface includes 141 top-level `docs/*.md`, ~50 banked `docs/*.json`
result files, and 141 apps in `modal_apps/` (≈40 of them `hphi_*`). Nothing from
`run_v2_01` onward appears. Concretely, greps over `ARTIFACT_INDEX.md` return
**0** hits for each of: `MOLLEO`, `GrIDDD`, `QED`, `h_phi`, `SMC`, `bridge`,
`receding`, `sentinel`. (Its single "128" hit is "128 examples" in the teacher
audit row, line 17 — unrelated to the 128-source validation.)

---

## 3 · `docs/CLAIM_LEDGER.md`

### What it claims to index

Line 1: "# Claim ledger — Paper 1" — "Every claim the manuscript makes, its
status, and what would falsify or fill it", across five sections: formal claims
(F1–F9), support/construction claims (S1–S16), empirical claims by experiment,
barred/retracted items, and reviewer stress tests.

Last commit **2026-08-12 04:14:15** (`9e18557`, "Bank Claim 4A and close the
exact-target branch") — the same commit that last touched
`docs/EXPERIMENT_PLAN.md`. 325 commits since.

### Still true

**Every artifact path in the ledger resolves.** Spot-checked and confirmed
present: `diagnostics/coherence/program_state.json`,
`diagnostics/composition/benchmark_lead_scope_coverage.json`,
`diagnostics/production_preflight/teacher_filter_characterization.json`,
`diagnostics/conditional_smc/physchem_box.json`,
`diagnostics/reachability/frontier_recovery_cap4.json`,
`diagnostics/STALE_RESULTS_MANIFEST.json`,
`tests/fixtures/process_v2_gate_zero_t1_bound_decision.json`,
`configs/process_v2_prep_subset.json`,
`diagnostics/editing_v2_sealed67_preregistration.json`,
`diagnostics/editing_v2_sealed67_result.json`, and the long
`diagnostics/coherence/editing_cycle_open_global_equivalence_v2/result.08111e3…json`
of S14. The four 4A rows (lines 75–78) agree with `docs/EXPERIMENT_PLAN.md:665-668`.

### FALSE or out of date

**F3.1 — the framing cross-reference points at an ARCHIVED, superseded file. Line 13:**

> Cross-reference: `docs/PAPER1_FRAMING_AUTHORITATIVE.md` (framing), `numbers.tex` (macro inventory),

`docs/PAPER1_FRAMING_AUTHORITATIVE.md` carries an ARCHIVED banner, is listed in
`ESTATE_REGISTRY.md:47` among the stamped documents, and is called out by name in
`docs/EXPERIMENT_PLAN.md:21`:

> docs/PAPER1_FRAMING_AUTHORITATIVE.md      <- name claims authority; superseded

under the heading "Documents this supersedes — do not read, cite or execute"
(line 11). The ledger and the plan were last written **in the same commit**
(`9e18557`), and the ledger still routes readers to the superseded framing. The
file must stay where it is — ~13 inbound references depend on it — but the ledger
is pointing at it as live framing.

**F3.2 — §3 is indexed by a framing its own plan declares superseded.**
Lines 61–74 label rows `A`, `B`, `C`, `D`, `E`, `F`, `G`, while
`docs/EXPERIMENT_PLAN.md:3-5` supersedes "the A/B/C framing used earlier in
development". The 4A rows immediately below (lines 75–78) use the *canonical*
plan's numbering. One table, two incompatible numbering systems. See §5, C4 for
what that costs a reader.

**F3.3 — the stated blocker for row F is no longer true. Line 71:**

> | **F** | Exact finite-horizon control on the **current** operator registry | **PENDING** | Enumerable graph must be **rebuilt**: the existing anchor used the pre-RingCore operator set |

An anchor on the *current* registry exists and was committed the day before the
ledger's last edit: `diagnostics/exactness/editing_v2_experiment_b_exact_control.json`
(commit `5f24d1c`, 2026-08-11 08:40), `"run": "run_v2_01"`,
`"selected_step": 12500`, 966 states, `terminal_tilt_tv` 1.77e-16,
`support_violations` 0, `backward_residual` 0.0, plus an exact retarget from the
realized state `CC(C)C`.

> **UNCERTAIN — do not simply flip this row to MEASURED.** The artifact
> self-labels `"status": "DEVELOPMENT_RESULT_NOT_PAPER_BEARING"` and
> `"is_a_verification_slice_not_a_chemistry_benchmark": true`, on a carbon-only
> slice (`"elements": ["C"]`, `heavy_atom_cap` 6). `docs/EXPERIMENT_PLAN.md:252-287`
> requires 30–50 goals across budgets 2/4/6/8 over eight arms for Experiment 3.
> So **PENDING is probably still the right status; the reason given for it is
> false.** Only the "must be rebuilt / pre-RingCore" sentence is stale.

**F3.4 — the "non-starved checkpoint" blocker is stale across four rows.**
Lines 67, 68, 69 and 73 give as blocker "Needs a non-starved edit checkpoint"
(rows B, C, G) and equivalents. A frozen, non-starved checkpoint has existed
since Phase 0: `run_v2_01` step 12,500, identity chain recorded at
`docs/PROJECT_BOARD.md:104-109` and used by every `h_φ` experiment since. Row B
additionally says (line 67) "The only two artifacts are untracked **and** marked
INVALID by `diagnostics/STALE_RESULTS_MANIFEST.json`" — that remains a true
statement about *those two artifacts*, but it is no longer the reason the row is
open.

**F3.5 — the repo's largest measured results have no row.** Greps over
`CLAIM_LEDGER.md` return **0** for `QED`, `MOLLEO` and `128`. Missing entirely:

- the 64-source development curve, 47/64 = **73.4% at 20 candidates**
  (`docs/EXTENDED_CURVE_64.json`, commits `ebcb379` / `2e55b56`, 2026-08-18);
- the prospective 128-source validation, **49.2% at k=8** rising to **54.7% at
  k=12** (`docs/VALID128_K8_RESULT.json` `"coverage": 0.4921875`,
  `docs/VALID128_CURVE.json` `"12": 0.546875`; commits `125efc6` / `5be5d65`),
  against GrIDDD's reported **45.1% at k=20** (`docs/AMENDMENT_VALIDATION_128.md:98`);
- the H24-vs-H40 receding-horizon A/B, 32/64 → 36/64 with the seed-noise caveat
  (`docs/RECEDING_HORIZON_AB_64.json`, commit `437f52b`, corrected by `e9ba44b`);
- the whole MOLLEO probe series and its four negatives (commit `b800c6e`).

Section 4 ("Barred and retracted") has likewise not absorbed the newer negatives:
the H40-aware `h_φ` head closed on four measurements (`76675bd`), the
slack-stratified WE gate failure (`b535c05`), the forced-diverse branching null
(`7313879`), the three-arm archive reversal (`6eb963e`), or the `R_θ` top-K route
retention failure (`a097a0b`, `docs/SHORTLIST_RETENTION.json` — "0 of 21
hard-source rescue trajectories were fully contained in top-128",
`docs/AMENDMENT_MOLLEO_LAZY_GATE.md:10-11`).

**F3.6 — there are two claim ledgers and neither knows about the other.**
`paper/CLAIM_LEDGER.md` (last commit 2026-07-27, `38f020e`) uses an entirely
different schema — `C-CLOSURE`, `C-KERNEL-NORM`, `C-EXACT-DOOB`, with
`Result key` columns validated by `paper/scripts/check_claim_registry.py` against
`paper/results_registry.yaml`. `AGENTS.md:649` points at `docs/CLAIM_LEDGER.md`
("Evidence status: `docs/CLAIM_LEDGER.md` plus authoritative machine-readable
artifacts"), while `paper_arxiv/SCIENTIFIC_TRACEABILITY.md:35,38,65` cites
`docs/CLAIM_LEDGER.md` rows F5, S13 and S11 by ID. Neither ledger references the
other. `paper/` is preserved deliberately and should stay; the finding is only
that "the claim ledger" is an ambiguous reference in this repo.

---

## 4 · `docs/PROJECT_BOARD.md`

### What it claims to index

Line 3: "**This file is the durable task list. Git is the source of truth.**" —
the persistent replacement for a session task tool that was wiped twice. Sections:
In flight, Landed, Blocked/gated, On hold, Carried forward.

Last commit **2026-08-11 15:09:46** (`10630a1`, "Docs: state the C result as a
rescue rate, not a p-value"). 359 commits since. It is one of the three
whitelisted "current" files.

### FALSE or out of date

**F4.1 — its only in-flight item was already closed when the file was last written.**
Lines 16–18:

> ## In flight
>
> ### C0 — Is there a planning problem at all?

C0 returned STOP on **2026-08-11 11:11** — commit `2a67da4`, "C0 result: STOP.
Future value does not beat greedy on this task." The board's own last commit is
**15:09 the same day**, four hours later, and did not touch the section. Both of
the other whitelisted documents record the closure:

- `docs/DECISION_LOG.md:72`: "| DRD2 is rugged enough that future value beats
  greedy | **Refuted** | C0: mean fresh regret **negative at every depth**;
  sacrificial win rate 57.1% at **0.5 SE** from chance…"
- `docs/EXPERIMENT_PLAN.md:672-673`: "Superseded en route, recorded so they are
  not re-run: the C0 planning-signal probe on DRD2 was **negative**"

So the durable task board has, for eight days and 359 commits, shown one in-flight
item that is refuted, and nothing else — through the entire h_φ/QED lane, the
128-source validation and the MOLLEO series.

**F4.2 — P1's blocker is false, and was false 16 hours before the file was last written.**
Lines 156–160:

> ### P1 — Bind the code commit into `RunIdentity`
>
> The plan requires eight identity bindings per checkpoint. `RunIdentity` carries
> **seven**: initialization seed, initial model state, library, split, sampling
> law, manifest, packed store, eval panel. **The code commit is missing.**

`RunIdentity` now carries the code commit as its *first* field —
`src/compose_v4/experiments/editing_v2_r_theta_corpus_training.py:207`:

```
git_commit_sha: str | None
```

with the docstring at lines 205–206: "The commit the run was launched from. None
marks a LEGACY development checkpoint, readable for analysis and NOT eligible as
paper-bearing." Added by commit `a156207`, **2026-08-10 23:16:12**, "editing-v2:
bind the code commit into RunIdentity; schema v2, fail-closed" — with tests
(`tests/test_r_theta_corpus_training.py`, +101 lines). It is also carried into
`payload()` at line 219.

The identical false claim appears in the plan of record,
`docs/EXPERIMENT_PLAN.md:681`:

> - **Checkpoints do not bind the code commit.** `RunIdentity` currently carries initialization seed, initial model state, library, split, sampling law, manifest, packed store and eval panel hashes — seven of the eight required bindings. The code commit is missing and must be added before any paper-bearing run.

(No `src/` file was read for anything but this check, and none was modified.)

**F4.3 — "Nothing started" is false for the on-hold prep items.** Lines 188–190:

> ### H1 — The six prep items from the plan's immediate execution order
>
> Nothing started.

At least two of the six have substantial work landed:

- Item 4, "Implement the full-molecule `h_phi(x, z, b)` pipeline" (line 195) — is
  built and heavily exercised. Eight modules under `src/compose_v4/experiments/`
  (`hphi_rollout.py`, `hphi_smc.py`, `hphi_region_features.py`,
  `hphi_graph_encode.py`, `hphi_lazy_sampler.py`, `hphi_lazy_family_scores.py`,
  `hphi_lazy_helpers.py`, `hphi_we_resample.py`), ~40 `modal_apps/hphi_*_app.py`,
  and a frozen head: `docs/AMENDMENT_VALIDATION_128.md:13-15` names "the FROZEN
  H24 `h_phi` head (`hphi_v2/head.pt`, sha256 `9ea51ec4...`)".
- Item 6, "Verify MARS, DDSBM, GraphXForm, GraphGA, REINVENT environments
  **without** running full experiments" (line 197) — an environment gate has been
  run and passed for the MOLLEO objective bundle: `docs/MOLLEO_ENV_GATE.json`,
  `"all_objectives_ok": true`, `"qed_parity_ok": true`, `"max_abs_diff": 0.0`
  (commit `a097a0b`, "MOLLEO env gate passes"). DDSBM has a frozen result and a
  protocol (`docs/DDSBM_FROZEN_RESULT.json`,
  `docs/DDSBM_ENDPOINT_COMPETENCE_PROTOCOL.md`).

> **UNCERTAIN — the hold itself.** The section header at line 186 reads
> `## On hold (Rohin: "hold off on parallel work till I give you more instructions")`.
> 359 commits of scientific work have landed since. Whether the hold was lifted
> verbally, narrowed, or is being worked around is not recoverable from the
> repository. Similarly, the H1 gate at lines 204–206 — "the full-molecule bridge
> must beat greedy, hard mask, local Boltzmann and beam search on a 10-source
> development panel before any expensive external sweep" — is not referenced by
> any of the h_φ/QED banking documents I read, so I could not establish whether
> it was met, superseded by the QED preregistration, or bypassed.

**F4.4 — the Landed section's headline control result has been superseded, by the
board's own stated plan.** Lines 72–92 report Experiment C on 24 held-out
transformations, "exact recovery | 12/24 | **18/24**", closing at line 92:

> Proper source-level uncertainty comes later, from the sealed panel.

The sealed panel landed the next day and is not on the board:
`diagnostics/editing_v2_sealed67_result.json` (commit `9e18557`, 2026-08-12),
recorded at `docs/CLAIM_LEDGER.md:75` — "65 endpoint-clean held-out pairs, greedy
26/65 → verified rollout 40/65, paired **+21.5 pp [12.3, 33.5]** exact" — and at
`docs/EXPERIMENT_PLAN.md:665-668`. So the durable board's control headline is the
smaller development number the sealed panel replaced.

**F4.5 — Phase 0's frozen step disagrees with the decision log.** Lines 104–106:

> ### Phase 0 — `R_theta` frozen
>
> Frozen at **step 12,500**: reference-law NLL **5.4603 → 2.8364**, within-family
> identity improved in **all 8 families**.

`docs/DECISION_LOG.md:19-20`, committed *later* (2026-08-12 00:56, `624cdd4`):

> Reference-law-weighted NLL falls from **5.4603** at the frozen initialization to
> **2.9238** at the selected step 8,500 — 2.54 nats over two epochs.

Different selected step, different final NLL. Independent evidence favours the
board: `diagnostics/exactness/editing_v2_experiment_b_exact_control.json` records
`"selected_step": 12500`, and "12,500" appears in no other document in `docs/`.
The likely reading is that the log's number predates the epoch-3 rule the board
cites ("Preregistered epoch-3 rule applied", line 108) and was never refreshed —
but two whitelisted documents give two answers to "what is `R_θ` frozen at", and
I am flagging that rather than resolving it.

### Still true (verified)

- P3, lines 176–182, "Which of the four paper directories is the target? …
  **Unresolved — needs a decision.**" — no commit resolves it. (See F1.2 for the
  2026-08-18 `paper_arxiv/` build, which may or may not bear on it.)
- P2, lines 170–174, three independently trained seeds — no evidence of change;
  still a budget decision.
- The DRD2 oracle section, lines 134–150, is corroborated by
  `docs/DECISION_LOG.md`'s parity discussion.

---

## 5 · Contradictions **between** the registries

These matter more than any single document's staleness: each is a case where a
reader following one registry reaches a conclusion another registry forbids.

### C1 · Three documents, three answers to "which plan governs"

| Source | Line | Assertion |
|---|---|---|
| `docs/ESTATE_REGISTRY.md` | 11 | "`docs/EXPERIMENT_PLAN.md` \| the only experiment plan" |
| `docs/EXPERIMENT_PLAN.md` | 14 | "**This file is the only current plan.**" |
| `docs/ARTIFACT_INDEX.md` | 5–11 | "## Canonical research plans" → three 2026-07-20 HTML files |
| `docs/research_plans/README.md` | 3 | "These three standalone HTML documents are the canonical project strategy artifacts." |
| `docs/MASTER_PLAN_EXPERIMENTS_AND_COMPARATORS.md` | 1 | "# COMPOSE — CURRENT GOVERNING EXPERIMENTAL PLAN" |
| `docs/MASTER_PLAN_EXPERIMENTS_AND_COMPARATORS.md` | 1742 | "**This is now the governing section.** It supersedes CURRENT AMENDMENT I where they conflict, and both supersede the historical body." |

The newest of these (MASTER_PLAN, 2026-08-15) is named by **none** of the four
registries, and none of the four is named by it. `CLAUDE.md:17-29`, which
auto-loads every session and therefore outranks all of them in practice, repeats
the ESTATE_REGISTRY answer.

The practical divergence is not cosmetic. `docs/EXPERIMENT_PLAN.md` contains
**zero** occurrences of "MOLLEO" and treats the external benchmark suite through
its own "must-run" list (lines 424–441); MASTER_PLAN records at lines 447–449
that this is exactly what changed:

> Where an older document
> said something now overtaken — `docs/EXPERIMENT_PLAN.md`'s "must-run" external
> list is the main case — this records **what changed and why**

and its governing experiment map (lines 194–210) sets "**2 QED / GrIDDD**" as the
"**current critical path**" and adds "**4D MOLLEO Task 3**" — the two lanes that
account for essentially all work since 2026-08-13.

**A related wrinkle inside MASTER_PLAN itself, worth knowing before citing it.**
Since commit `71dff43` (2026-08-15 01:31, "Promote the governing plan to the
front; archive the history below it") the governing text is lines **1–425**, and
line **426** reads `# ARCHIVED PROVENANCE — NOT GOVERNING`, followed at lines
428–430 by "Everything below this line is retained verbatim as the scientific and
decision history … It does not authorize experiments or override the current
governing plan above." But `# ⭐ CURRENT AMENDMENT` sits at line **1141** and
`# ⭐⭐ CURRENT AMENDMENT II` at line **1740** — both *below* the archive line,
both declaring themselves governing. Anyone whose habit is "read the CURRENT
AMENDMENT at the end of MASTER_PLAN" is now reading text that the front of the
same file classifies as non-governing provenance. I did not attempt to determine
which reading is intended.

### C2 · "Current" vs "history" for the same two files

`docs/ARTIFACT_INDEX.md:13-20` places `PROJECT_STATUS.md` and `CURRENT_MODEL.md`
under "## Current audits". `docs/ESTATE_REGISTRY.md:5-7` places them in history
("exactly three files are current. Everything else in this repository is history
until it is linked from one of them"), and `f413883`'s commit message names
CURRENT_MODEL specifically as a document that "sound[s] authoritative" without
being current. A reader arriving via `ARTIFACT_INDEX` gets a 2026-07-19 status
page describing the step-6,250 pancake model as live work; a reader arriving via
`ESTATE_REGISTRY` is told to ignore it. Compounded by F2.2: `ARTIFACT_INDEX`
contradicts *itself* on this checkpoint at lines 29–30.

### C3 · The ledger cites a framing document the plan and the registry both retire

`docs/CLAIM_LEDGER.md:13` → `docs/PAPER1_FRAMING_AUTHORITATIVE.md`, versus
`docs/EXPERIMENT_PLAN.md:21` ("name claims authority; superseded", under "do not
read, cite or execute") and `docs/ESTATE_REGISTRY.md:47` (ARCHIVED). Ledger and
plan share a commit (`9e18557`); the conflict was introduced with both eyes open,
or overlooked in the same edit.

### C4 · "Experiment B" and "Experiment 3" denote different objects in different registries

| Registry | Label | Object | Status claimed |
|---|---|---|---|
| `docs/CLAIM_LEDGER.md:67` | exp **B** | "Learned rates beat uniform legal rewrites on transport" | **PENDING** |
| `docs/PROJECT_BOARD.md:118` | "Claim 3 — exact finite-horizon control (**Experiment B**)" | exact Doob control on the 966-state closure | **Landed** |
| `docs/CLAIM_LEDGER.md:71` | exp **F** | exact finite-horizon control on the current registry | **PENDING** |
| `docs/EXPERIMENT_PLAN.md:252` | "**Experiment 3** — Exact finite-horizon control" | 30–50 goals × budgets 2/4/6/8 × 8 arms | not started |
| `docs/MASTER_PLAN…:200` (map) | "**3A** Finite-horizon control" | verified/future-aware vs greedy | **✅ banked** |

Four documents, four labels, and the *same* underlying artifact is simultaneously
"Landed" (board), "PENDING" (ledger F), "✅ banked" (master plan 3A) and
unstarted (plan Experiment 3). Some of that is legitimate — they are scoped to
different requirements — but nothing in any registry says so, and the artifact
itself carries `"status": "DEVELOPMENT_RESULT_NOT_PAPER_BEARING"`, which no
registry mentions. This is the contradiction most likely to produce an
overclaim.

### C5 · C0: in flight vs refuted

`docs/PROJECT_BOARD.md:16-18` (in flight) vs `docs/DECISION_LOG.md:72,82` and
`docs/EXPERIMENT_PLAN.md:672-673` (negative, refuted, "recorded so they are not
re-run"). All three are whitelisted as current. Detail in F4.1.

### C6 · `R_θ`'s frozen step: 12,500 vs 8,500

`docs/PROJECT_BOARD.md:106` vs `docs/DECISION_LOG.md:20`. Two whitelisted
documents, two answers. Detail in F4.5.

### C7 · The code-commit binding: claimed missing in two current docs, present in the code

`docs/PROJECT_BOARD.md:160` and `docs/EXPERIMENT_PLAN.md:681` both say the code
commit is unbound; `src/…/editing_v2_r_theta_corpus_training.py:207` binds it, as
of `a156207` (2026-08-10 23:16). Detail in F4.2. Two of the three whitelisted
documents carry the same false blocker, so cross-checking within the whitelist
cannot catch it.

### C8 · The registries are not themselves registered

`ESTATE_REGISTRY.md`, `ARTIFACT_INDEX.md` and `docs/CLAIM_LEDGER.md` have no
inbound reference from the whitelist; `CLAUDE.md:19-24` links only
EXPERIMENT_PLAN, PROJECT_BOARD and DECISION_LOG. By the registry's own rule all
three are history — including the registry that states the rule. Meanwhile
`AGENTS.md:649` treats `docs/CLAIM_LEDGER.md` as the evidence-status document,
and the four `paper*/_STATUS.md` notices treat `ESTATE_REGISTRY.md` as the
directory map. Two governance systems, disjoint membership.

---

## 6 · Recent work none of the four registries covers

Verified against `git log`; each item is absent from all four documents (grep
counts of `MOLLEO`, `QED`, `128`, `receding`, `sentinel`, `fiber` census terms
are 0 across the four, with the single false positive noted in F2.4).

**Prospective 128-source QED validation** — preregistered `cfc0ad8` (2026-08-18
10:09, `docs/AMENDMENT_VALIDATION_128.md`), extended `e3b6f63`, banked `125efc6`
("BANK: prospective 128-source validation, 49.2% at k=8 vs GrIDDD 45.1% at k=20",
`docs/VALID128_K8_RESULT.json`: `"coverage": 0.4921875`, `"n": 128`,
`"runs": 1024`, `"work_transitions": 1167787`) and `5be5d65` (k=12 → 54.7%,
`docs/VALID128_CURVE.json`). Controller declared at
`docs/AMENDMENT_VALIDATION_128.md:13-15`: "H = 40 receding horizon,
`b_eff = min(24, b)`, N = 32 particles, the FROZEN H24 `h_phi` head".

**H40 receding-horizon candidate-efficiency curve** — `437f52b` (2026-08-17,
"Receding-horizon A/B on 64 sources: 32/64 -> 36/64, but inside the seed-noise
floor"), corrected by `e9ba44b` (2026-08-18 00:16, "Monotonicity is not
load-bearing. Correct the 64-source H40 numbers to all 64 sources."). Note
`docs/RECEDING_HORIZON_AB_64.json` carries its own supersession notice:
`"docs/RECEDING_HORIZON_AB.json reported n=60 and disagrees at k=1; it is stale."`
— the JSON artifacts are doing the currency bookkeeping the registries aren't.
Extended to 20 candidates in `docs/EXTENDED_CURVE_64.json` (`ebcb379`, `2e55b56`):
`ALL` curve `[28,32,35,36,37,40,40,41,42,43,…,47]` of 64, i.e. 43/64 at 10 and
47/64 = 73.4% at 20, with strata `reliable` 19/19, `marginal` 7/8, `hard` 21/37.
The H40-aware head itself was then **killed** on measurement: `76675bd`,
"H40-aware h_phi does not pay off. Line closed on four measurements."

**MOLLEO probe series** — `a097a0b` (env gate passes, `R_θ` top-K route retention
fails; `docs/MOLLEO_ENV_GATE.json`, `docs/SHORTLIST_RETENTION.json`), `2c39bb8`
(Dev Gate 1 preregistered, cohort frozen; `docs/AMENDMENT_MOLLEO_LAZY_GATE.md`,
`docs/MOLLEO_DEV_COHORT.json`), and `b800c6e` (2026-08-19 00:20, HEAD~1) — "four
blind probes fail, then a task-independent bridge learns long-range navigation",
adding `modal_apps/molleo_fiber_census_app.py`,
`modal_apps/molleo_coverage_sentinel_app.py`,
`modal_apps/molleo_basin_substrate_app.py`,
`modal_apps/molleo_bridge_probe_app.py`, `modal_apps/bridge_train_app.py`,
`modal_apps/bridge_navigation_gate_app.py`,
`src/compose_v4/experiments/molleo_surrogate.py`, and
`docs/MOLLEO_BASIN_ANCHORS.json`, `docs/MOLLEO_BRIDGE_PAIRS.json`,
`docs/MOLLEO_H40_VISITED.json`. Its own commit message flags a reversal worth a
ledger row: "my 'don't claim MOLLEO' call was premature — it generalised from
four experiments that all lacked the same primitive." A partially superseding
lane summary exists at `docs/MOLLEO_TASK3_LANE_SUMMARY.md:3-9` ("**Status: banked
and paused.** … No number in this lane is Task 3 performance"), last committed
2026-08-16 — i.e. before the bridge result.

**Also uncovered, for completeness:** the SMC/sampler efficiency lane
(`13baa3c` 90× lazy sampler, `678e225` execution parity 48/48, `7ab7b47`
extinction sentinel 8.0×), the WE/rare-event line closed at `b535c05`, and the
archive-controller line closed at `7313879` after `6eb963e` reversed its v1
result.

**Where the current state actually lives now**, for anyone trying to reconstruct
it without the registries: the amendment chain
`docs/AMENDMENT_{HORIZON_DIAGNOSTIC,SMC_EFFICIENCY,H40_HPHI,VALIDATION_128,SHORTLIST_RETENTION,MOLLEO_LAZY_GATE}.md`
(2026-08-17 → 2026-08-18, each "recorded before the data"), the ~50 banked
`docs/*.json` results, `docs/MASTER_PLAN_EXPERIMENTS_AND_COMPARATORS.md`
lines 1–425, and the commit messages.

---

## 7 · What I could not determine

Listed explicitly rather than guessed:

1. Whether the 2026-08-18 19:35 LaTeX build in `paper_arxiv/` signals resumed
   work on that directory (bearing on P3) or is incidental. No tracked file
   changed; `main.tex` is untouched.
2. Whether the COMPOSE-Lipid Paper-2 program that `ARTIFACT_INDEX.md:10` calls
   canonical is parked deliberately or lapsed. Zero mentions in any current plan,
   zero commits in the last 200.
3. Whether `docs/CLAIM_LEDGER.md:71` row F should change status now that a
   current-registry exact-control artifact exists — the artifact self-labels
   non-paper-bearing, and Experiment 3's full goal suite is unrun. Only the
   *reason* attached to the PENDING is demonstrably stale.
4. Whether the `PROJECT_BOARD.md:186` hold and the `:204-206` bridge gate were
   lifted, narrowed or superseded. Not recoverable from the repository.
5. Which of `MASTER_PLAN`'s two "governing" regions is intended to win — the
   promoted front matter (lines 1–425) or `CURRENT AMENDMENT II` (line 1740),
   which sits below the file's own `NOT GOVERNING` divider at line 426.
6. Whether `DECISION_LOG.md:20`'s step 8,500 is stale or scoped to an earlier
   epoch rule. External evidence favours the board's 12,500, but the log does not
   say it was superseded.

---

## 8 · How to re-verify

```
# dates and distance from HEAD
for f in docs/ESTATE_REGISTRY.md docs/ARTIFACT_INDEX.md \
         docs/CLAIM_LEDGER.md docs/PROJECT_BOARD.md; do
  echo "$f  $(git log -1 --format='%ci %h' -- $f)"
done

# F1.1  document count
ls -1 docs/*.md | wc -l
git ls-tree --name-only f413883 docs/ | grep -c '\.md$'

# F1.2  paper_arxiv mtimes
ls -la paper_arxiv/

# F1.3  paper-directory last commits
for d in paper paper_arxiv paper_iclr_control_substrate \
         paper_iclr_stochastic_rewriting; do
  echo "$d $(git log -1 --format='%ci %h' -- $d)"
done

# F4.1  C0 closed before the board's last commit
git log --format='%ci %h %s' --all | grep -i 'C0 result'
git log -1 --format='%ci' -- docs/PROJECT_BOARD.md

# F4.2  the code commit is bound
grep -n 'git_commit_sha' src/compose_v4/experiments/editing_v2_r_theta_corpus_training.py
git log -S git_commit_sha --format='%ci %h %s' \
  -- src/compose_v4/experiments/editing_v2_r_theta_corpus_training.py | tail -1

# C1  the governing plan nobody indexes
head -6 docs/MASTER_PLAN_EXPERIMENTS_AND_COMPARATORS.md
grep -n 'CURRENT AMENDMENT\|ARCHIVED PROVENANCE' \
  docs/MASTER_PLAN_EXPERIMENTS_AND_COMPARATORS.md
grep -c MOLLEO docs/EXPERIMENT_PLAN.md      # 0

# §6  recent work absent from all four
for f in docs/ESTATE_REGISTRY.md docs/ARTIFACT_INDEX.md \
         docs/CLAIM_LEDGER.md docs/PROJECT_BOARD.md; do
  echo "$f QED=$(grep -ic QED $f) MOLLEO=$(grep -ic MOLLEO $f)"
done
```
