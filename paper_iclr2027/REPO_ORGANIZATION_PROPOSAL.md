# Repository organization — the paper writer's view

**Written 2026-08-19 while assembling `paper_iclr2027/`. This is a document, not
an action. It moves nothing, renames nothing, deletes nothing, and modifies no
existing file.** Every proposal below is either a *new* file or an *append* to an
existing one, for a reason given in §0.

---

## 0 · What this is, what it is not, and the constraint that shapes every fix

Three documents already cover the ground I would otherwise re-derive, and I
build on them rather than repeating them:

| document | what it establishes | I do not repeat it |
|---|---|---|
| `docs/REGISTRY_AUDIT_2026-08-19.md` | The four registries are stale in specific, cited ways; seven inter-document contradictions (C1–C8); the governing plan is named by none of them. | I cite C1, C4 and C6 below where a *writing* task hit them. |
| `docs/REPRODUCIBILITY_HAZARDS_2026-08-19.md` | Five hazard classes; 15 of 15 sampled recent `docs/*.json` carry no git commit and no producing script; `HPHI_SMC_REFERENCE_BANKED.json` is the provenance template. | I add the *semantic* half of the same problem: those files also have no data dictionary (§4). |
| `REORG_REPORT.md` / `docs/INDEX.md` / `README_NAVIGATION.md` (all 2026-08-19) | A 303-file status index, a newcomer's map, and the finding that **nothing could safely be moved**. | I take the move ban as settled and propose no moves. |

**The constraint that shapes every fix below.** `REORG_REPORT.md` establishes
that this project cites documents *by line number*, and that the two 2026-08-19
audits are built almost entirely on such citations — so **prepending** a status
header to `docs/EXPERIMENT_PLAN.md` would invalidate 13 of them, to
`docs/PROJECT_BOARD.md` 8, to `docs/CLAIM_LEDGER.md` 6. I accept that finding
completely. Every proposal here is therefore **append-only or a new file**.
Nothing I propose can shift a line number in a cited document.

Two further constraints I hold to and restate so no later reader relaxes them:

* **`src/` is untouchable by reorganization.** Files under `src/` are covered by
  the Process-V2 identity hash — the same hash that gates banked artifacts.
  `docs/ADMISSION_MASK_OPTIMIZATION.md` records an optimization that is *exact
  and bitwise-qualified* and is still blocked from deployment for precisely this
  reason. Moving or renaming anything under `src/` would invalidate banked
  artifacts, including the frozen `R_θ` this paper's every number depends on.
  **No proposal below touches `src/`.**
* **Nothing may become harder to find or re-run.** Where a fix could plausibly
  break a path referenced by a script or a launch config, I say so and give the
  accompanying edit.

My distinct contribution is narrower than the audits': **what made the paper
hard to assemble.** That turns out to be a different question from "what is
stale" and from "what will fail to re-run", and it surfaced two things the other
audits did not (§1, §2) that have direct consequences for what may be published.

---

## 1 · The finding that matters most: the comparator table is incomplete, and the evidence of that is disappearing

**What happened while writing.** §5.2 of the paper reports the QED editing
result beside the published comparator table in `docs/GRIDDD_JIN_PROTOCOL.md`,
which lists JT-VAE, CG-VAE, GCPN and GrIDDD and nothing else. Writing the
abstract, I reached for the phrase "the strongest reported comparator". Before
using it I checked whether anything stronger was on record. Something is.

* `docs/PAPER_REFRAME_CONTROL_SUBSTRATE.md:445` carries a table headed
  **"Tier A — must-cite constrained-ZINC editors"** whose first row is a
  published graph-to-graph translation method with a **QED success rate on the
  same δ ≥ 0.4 task that is higher than both GrIDDD's 45.1 % and this paper's
  54.7 %.** That document carries an `ARCHIVED — DO NOT USE` banner and
  `docs/INDEX.md:175` classifies it **SUPERSEDED**.
* The **committed** revision of `docs/VALID128_K8_RESULT.json` — the headline
  result file — carried a `comparator` block naming the same method with the
  note *"Tier A must-cite and is higher; this is not SOTA"*. That block is
  **deleted** in the current uncommitted working-tree revision.

So the only two records of a stronger published comparator are (a) inside an
archived document a reader is told not to use and (b) inside a field that has
been removed from the live artifact. A writer following the repository's own
navigation would never see it.

**Why this is not a documentation problem.** Publishing Table 1 as the protocol
document defines it presents a comparator table from which a stronger published
row has been dropped. That is a claim-discipline failure of exactly the kind the
project's own rules exist to prevent — and it would have happened silently.

**What I did in the paper.** The abstract now says "the comparator whose
protocol we follow", never "the strongest"; Table 1 carries an explicit `\XXX`
row labelled *Further published, off-table*; the caption states the omission;
and a `% ===== GAP: MISSING COMPARATOR ROW =====` block (**G8**) records where
the evidence lives and what must be verified.

**Proposed fix.**

1. **New file `docs/COMPARATOR_TABLE_QED.md`** — one table, every published
   number on the constrained-ZINC QED δ ≥ 0.4 task the project is aware of, each
   row carrying: value, candidate budget, source cohort, similarity threshold,
   the publication it comes from, and a **verified / unverified** flag. This is
   the artifact `GRIDDD_JIN_PROTOCOL.md`'s table should have been, and it can be
   created without touching that file.
2. **Append to `docs/GRIDDD_JIN_PROTOCOL.md`** (append-only, no line numbers
   move) a short section: *"The table above is the comparator's own table and is
   not the complete published record for this task; see
   `docs/COMPARATOR_TABLE_QED.md`."*
3. **Do not delete a `comparator` block from a result JSON when refreshing it.**
   The deletion in the working-tree revision of `VALID128_K8_RESULT.json` removed
   the only live pointer to a must-cite comparator. If the block is stale, correct
   it; if it belongs elsewhere, move it to (1) *first*.

---

## 2 · The headline artifact is uncommitted, and its two revisions disagree

`git status` shows ` M docs/VALID128_K8_RESULT.json`, and the diff is
substantive, not cosmetic:

| field | committed | working tree (what the paper quotes) |
|---|---|---|
| panel size | `n_sources: 126` | `n: 128` |
| solved @ k=8 | `cumulative[7] = 62` | `cumulative[7] = 63` |
| runs | 1008 | 1024 |
| extinct runs | 761 | 776 |
| work transitions | 1,147,373 | 1,167,787 |
| `comparator` / `panel` blocks | present | **removed** |

The headline percentage survives (62/126 = 49.206 %, 63/128 = 49.219 %, both
"49.2 %") and `docs/SESSION_RUN_MANIFEST_2026-08-18.md` agrees with the
working-tree revision, so I quoted that one. But **the counts in Table 1 and the
whole inference ledger in §5.2 come from a file that is not committed.** A
co-author checking out the repository today would reproduce a paper whose table
they cannot regenerate. `docs/ARCHIVE_AB_RESULT.json` and
`docs/RECEDING_HORIZON_AB.json` are in the same state;
`modal_apps/hphi_archive_read_app.py` shows as deleted-and-uncommitted.
`REORG_REPORT.md` notes this working-tree state as pre-existing and not caused by
that work; I confirm it is still present and add that it is now **load-bearing
for a manuscript**.

**Proposed fix.** Commit or reconcile the three JSONs and resolve the deleted app
before the draft is circulated — this is a `git` action for their owner, not a
reorganization. Then, as a standing rule appended to
`docs/REPRODUCIBILITY_HAZARDS_2026-08-19.md`: *a number may enter a manuscript
only from a committed revision, and the manuscript records the commit.*

---

## 3 · Same-file supersession: three conventions, two of them opposite

This cost me more writing time than anything else, and it is a *convention*
problem rather than a staleness problem, so the audits do not catch it.

Three documents supersede themselves internally, using three different devices:

| document | device | where the governing text is | reading top-down gives you |
|---|---|---|---|
| `docs/GRIDDD_JIN_PROTOCOL.md` | ⛔ banner at the top **and** `⭐ CURRENT AMENDMENT` at the bottom | the **bottom** | the refuted Policy B, under a section still titled *"THE INFERENCE POLICY — frozen before any QED outcome exists"* |
| `docs/MASTER_PLAN_EXPERIMENTS_AND_COMPARATORS.md` | promoted front matter (lines 1–425) **and** `# ARCHIVED PROVENANCE — NOT GOVERNING` at line 426 | the **top** | the governing plan — but `⭐ CURRENT AMENDMENT` (1141) and `⭐⭐ CURRENT AMENDMENT II` (1740) sit *below* the divider and also claim to govern (`REGISTRY_AUDIT` C1) |
| `docs/AMENDMENT_INVERSIONGNN_2OBJ.md` | `ADDENDUM II — SUPERSEDES the B_lambda construction above`, with no top-of-file marker | the **bottom** | Addendum I's withdrawn top-10 % region construction |

**The same ⭐ visual convention means "read the bottom" in one file and "you are
reading non-governing provenance" in another.** I nearly wrote the refuted
one-step policy into §5.2 as the deployed controller.

Worse for a writer: `AMENDMENT_INVERSIONGNN_2OBJ.md` **changed under me during
the session** — Addendum II was added after I had first read the file. Had I
written §4 from my first read, the paper would carry the withdrawn design. There
is nothing at the top of that file to tell a returning reader that the governing
section moved.

**Proposed fix.** A one-line convention, applied by **appending** a footer, never
by prepending a header (§0):

> Append to the *end* of any document that supersedes itself internally:
> `GOVERNING SECTION: <heading>, line <n> onward. Everything above it is
> historical. Last moved: <date>.`

A footer costs no line numbers. For `AMENDMENT_INVERSIONGNN_2OBJ.md` the footer
would read `GOVERNING SECTION: ADDENDUM II, 2026-08-19`. For
`GRIDDD_JIN_PROTOCOL.md`, `GOVERNING SECTION: ⭐ CURRENT AMENDMENT`. For
`MASTER_PLAN`, the footer is the natural place to settle the C1 wrinkle the
registry audit could not resolve.

---

## 4 · Results that exist only as JSON, with no prose and no data dictionary

`REPRODUCIBILITY_HAZARDS` shows these files lack *provenance*. The writing task
exposed the other half: they also lack *semantics*.

`docs/VALID128_K8_RESULT.json` gave me `contact_runs`, `extinct_runs`,
`work_transitions`, `mean_distinct_returned` and `sources_any_contact`. I used
all five in the paper's inference ledger. **None of the five is defined
anywhere.** `grep -rn contact_runs` returns the JSON itself, its reader script,
one Modal app and one other JSON — no definition in prose. I inferred that
"extinct" means a particle run with no surviving particle from the SMC context;
that is an inference, and a reader cannot check it.

Similarly, `docs/EXTENDED_CURVE_64.json` has keys `reliable` / `marginal` /
`hard` with counts 19 / 8 / 37. Those strata are *used* in
`docs/AMENDMENT_VALIDATION_128.md`'s selection table but *defined* nowhere in the
verified set. I left them out of the paper rather than describe strata I could
not define — a real loss, since a stratified development curve is more
informative than the pooled one.

And the interpretation of the headline result lives in a document written
**before** the data (`AMENDMENT_VALIDATION_128.md`, a preregistration) plus one
line of the run manifest. There is no after-the-fact prose companion saying what
the numbers mean, which mismatches travel with them, and what they do not
license. I had to construct that mapping myself; it is now
`paper_iclr2027/README.md`'s "Every number used, with its source" table.

**Proposed fixes.**

1. **New file `docs/RESULT_KEYS.md`** — a data dictionary for the recurring keys
   in `docs/*.json`: `coverage`, `cumulative`, `runs`, `contact_runs`,
   `extinct_runs`, `work_transitions`, `mean_distinct_returned`,
   `sources_any_contact`, and the `reliable` / `marginal` / `hard` strata with
   the rule that assigns a source to each. One line each. Cheap, and it converts
   five uncheckable numbers in the paper into checkable ones.
2. **New file `docs/RESULTS_CURRENT.md`** — a short *post-hoc* companion for each
   banked headline: what it measures, the estimand, the statistical unit, the
   mismatches that must travel with it, and what it does not license. The
   preregistrations cannot serve this purpose because they are written before the
   data; the run manifest is a log, not an interpretation.
3. Adopt `docs/HPHI_SMC_REFERENCE_BANKED.json`'s `provenance` block as the schema
   for new result JSONs, as the hazards audit already recommends. My addition:
   include a `key_definitions` pointer to (1).

---

## 5 · The paper's central theorem has no paper-facing verification status

The draft's Theorem 2 (the budget-indexed *h*-transform) is the formal core.
Whether it has been *verified against the implementation* on an enumerable slice
is a question I could not answer from any current document.

* `diagnostics/exactness/editing_v2_experiment_b_exact_control.json` exists,
  reports the right quantities, and self-labels
  `"status": "DEVELOPMENT_RESULT_NOT_PAPER_BEARING"` on a carbon-only slice.
* `docs/CLAIM_LEDGER.md:71` marks the corresponding row PENDING for a reason the
  registry audit shows is stale (F3.3), while `docs/PROJECT_BOARD.md:118` calls
  the same artifact **Landed** and the master plan's map calls it **✅ banked**
  (C4). Four labels, one artifact.
* No document says whether the check has ever been run on the *current* operator
  registry.

I therefore could not quote it, and the paper carries it as gap **U1** with both
routes to closure spelled out. That is the honest outcome, but it means the
paper's central formal claim currently has no implementation evidence a reviewer
can see.

**Proposed fix.** Add one row to whichever ledger is chosen as canonical — I
recommend **appending** a short *"Formal claims and their verification status"*
section to `docs/CLAIM_LEDGER.md* rather than editing rows, so no cited line
moves — with columns: claim, artifact, registry it was verified on, paper-bearing
yes/no, and what would make it paper-bearing. C4 is unresolvable by a writer; it
needs an owner's ruling, and the ruling belongs in one place.

---

## 6 · A live contradiction with publication consequences: the second comparator

Four documents rule the multi-objective comparator **tier 4 — "not runnable as
shipped, no LICENSE, 70× budget disagreement"**:
`docs/MASTER_PLAN_EXPERIMENTS_AND_COMPARATORS.md:763`,
`docs/PARETO_COMPARATOR_MATRIX_REQUIREMENT.md:93`,
`docs/workstreams/EXTERNAL_BASELINE_INDEX.md:25`, and
`docs/COMPARATOR_ROLES_CANONICAL.md`. The licensing ruling is explicit elsewhere
in the same family: *"zero licence files anywhere in the tree. Default
all-rights-reserved: **not vendorable**, same stop that excluded OP-GFN"*
(`docs/workstreams/multiobjective/FRAMEWORK_NEIGHBOR_GRIDDD.md:225`).

Meanwhile `docs/AMENDMENT_INVERSIONGNN_2OBJ.md` (2026-08-19) commits to
**rerunning both methods**, and `third_party/InversionGNN/` has now been
vendored. `find third_party -iname '*licen*'` returns nothing: **the tree still
has no licence.**

I wrote the paper following the amendment, because it is newest and because it
argues the point explicitly ("the engineering tax of rerunning an external method
is accepted deliberately here; the alternative is an unfalsifiable comparison").
But the paper's §5.4 promises a *matched rerun* of a method that four documents
say cannot be run as shipped and cannot be vendored. **If the licensing ruling
still holds, gap G3 cannot be closed as written**, and §5.4 would have to fall
back to reporting the published number as context only — which the amendment
explicitly refuses.

**Proposed fix.** This needs an owner's decision, not a document move. Record it
in one place — I suggest **appending** to `docs/AMENDMENT_INVERSIONGNN_2OBJ.md`
a section *"Licensing and runnability — supersedes the tier-4 ruling, or does
not"* — stating which of the two rulings governs, on what basis, and what the
vendored tree's licence status is. Until that exists, `paper_iclr2027`'s G3 is
blocked on a question no experiment can answer.

---

## 7 · Smaller frictions, each with its fix

| # | friction | evidence | proposed fix |
|---|---|---|---|
| 7.1 | The single most useful document for a paper writer is named for a date. `docs/SESSION_RUN_MANIFEST_2026-08-18.md` is the only file that maps app → invocation → artifact → result → commit. Before `docs/INDEX.md` existed, nothing pointed to it. | inbound refs were 0 outside itself | Keep the dated file (its name is its provenance). **New file `docs/RUN_MANIFEST_LATEST.md`** containing one line: a pointer to the current dated manifest, updated when a new one is written. A stable name that never goes stale in content. |
| 7.2 | `docs/INVERSIONGNN_FROZEN_PROTOCOL.json` is the *realization* of the amendment's step 1 (the frozen, hash-recorded initialization bank) and the amendment does not mention it. I found it only by listing `docs/` for `invgnn|inversion`. Its only inbound reference is `modal_apps/invgnn_corpus_app.py:240`. | `grep -rn INVERSIONGNN_FROZEN_PROTOCOL` | **Append** a one-line "Realized artifacts" section to `docs/AMENDMENT_INVERSIONGNN_2OBJ.md` listing the frozen protocol JSON and the diagnostic JSON. Costs nothing, no line numbers move. |
| 7.3 | Manuscript-binding *wording* rules are scattered across three documents: the h_φ phrasing bar (`docs/HPHI_QED_PREREGISTRATION.md` §1 — "do NOT write that h_φ *restores* an originally-frozen policy"), the compute-accounting phrasing rule (`MASTER_PLAN` §I.4 — "improved performance at a measured inference-time cost, **not** greater efficiency"), and the benchmark terminology bar (`GRIDDD_JIN_PROTOCOL.md` — "this benchmark is NOT multiobjective"). All three bind the paper. Two of the three are discoverable only by already knowing they exist. | I found the compute rule at `MASTER_PLAN:1788`, inside the region the file's own front matter calls non-governing; I trusted it only because `HPHI_QED_PREREGISTRATION` §10 restates it independently | **New file `docs/MANUSCRIPT_RULES.md`** — every phrasing rule that binds a manuscript, each with a one-line quote and a pointer to the document that owns it. A writer reads one file instead of finding three by luck. |
| 7.4 | Notation is forked four ways. `A(x)` vs `𝒜(G)`; `P_θ` vs `R_θ` vs `Ptheta`; `h_b` vs `hval_k`; `T(x,a)` vs `T_a G`. No document says which is canonical, so a new draft must choose and then reconcile every transplanted proof by hand. | `paper/sections/*.tex` vs `paper_iclr_stochastic_rewriting/main.tex` vs `paper_arxiv/sections/*.tex` | **New file `paper_NOTATION.md` at the repository root** (or `docs/NOTATION_CANONICAL.md`) fixing one symbol table, with a translation column per existing draft. Do **not** edit the existing drafts — they are preserved deliberately, and `REORG_REPORT.md` shows why editing them is unsafe. |
| 7.5 | Three `references.bib` files with disjoint key sets, and the same work keyed differently across them (`kaech2025invirtuogen` / `kaech2026invirtuogen`; `campbell2022continuous` / `campbell2022ctdd`). Merging is manual and silently error-prone. | `paper/references.bib`, `paper_arxiv/references.bib`, `paper_iclr_stochastic_rewriting/references.bib` | **New file `references_master.bib`** at the root, built by union with a documented key-collision policy, plus a `KEY_MAP.md` recording which old key maps to which. Existing `.bib` files stay exactly where they are so the existing drafts keep compiling. |
| 7.6 | Two comparators are named in this paper with no citable bibliographic record anywhere in the repository (the second multi-objective comparator; CG-VAE and GCPN as inherited rows). Only an arXiv id and a GitHub commit exist. | `docs/workstreams/multiobjective/HANDOFF.md:48`, `BENCHMARK_ALIGNMENT_AUDIT.md:34` | Add verified BibTeX entries to (7.5) rather than to a draft's local `.bib`. Listed in `paper_iclr2027/README.md` under "Bibliography entries still needed" so it is not forgotten. |
| 7.7 | I had to use filesystem mtimes to establish ordering: whether `docs/INVERSIONGNN_BLAMBDA_DIAGNOSTIC.json` (01:54) preceded Addendum II (01:55), i.e. whether the diagnostic caused the withdrawal or followed it. That ordering is scientifically material — it is the difference between a preregistered rule and a post-hoc one — and it is recorded nowhere. | `ls -la docs/AMENDMENT_INVERSIONGNN_2OBJ.md docs/INVERSIONGNN_BLAMBDA_DIAGNOSTIC.json` | When an amendment is withdrawn by its own diagnostic, **append** one line to the amendment stating the diagnostic artifact, its hash, and that it preceded the withdrawal. Addendum II states the reasoning well; it does not name the artifact. |

---

## 8 · What a newcomer *writing the paper* still lacks

`README_NAVIGATION.md` (new today) answers "what is this project" and "which
plan governs" well. Four writing-specific questions remain unanswered by any
single document:

1. **Which numbers may go in a manuscript, and from which revision?** The seven
   files I was handed as the verified set are not identified as such anywhere in
   the repository. Without that list I would have had to construct it from the
   run manifest plus `git log`, and §2 shows I would have quoted an uncommitted
   file without knowing it. → `docs/RESULTS_CURRENT.md` (§4.2) plus the
   committed-revision rule (§2).
2. **What every number means.** → `docs/RESULT_KEYS.md` (§4.1).
3. **What the manuscript may not say.** → `docs/MANUSCRIPT_RULES.md` (§7.3).
4. **Which artifacts live on the Modal volume rather than in git.** The run
   manifest says it in prose and `docs/INDEX.md` now repeats it, but there is no
   list. A writer citing an artifact cannot tell whether a reader can open it. →
   add a column to `docs/RESULTS_CURRENT.md`: *in git / on volume / both*.

---

## 9 · Proposals, ranked, with the safety property of each

| rank | proposal | kind | can it break anything? |
|---|---|---|---|
| 1 | `docs/COMPARATOR_TABLE_QED.md` + append a pointer to `GRIDDD_JIN_PROTOCOL.md` (§1) | new file + append | No. Append-only; no line number moves. **Highest value: it is what stands between this draft and a published table missing a stronger row.** |
| 2 | Commit or reconcile the three modified result JSONs; resolve the deleted app (§2) | `git` action by their owner | No. Not a reorganization. |
| 3 | `docs/RESULT_KEYS.md` (§4.1) | new file | No. |
| 4 | Footer convention for self-superseding documents (§3) | append | No — footer, never header, precisely because of `REORG_REPORT.md`'s line-number finding. |
| 5 | `docs/MANUSCRIPT_RULES.md` (§7.3) | new file | No. |
| 6 | `docs/RESULTS_CURRENT.md` (§4.2, §8) | new file | No. |
| 7 | Licensing/runnability ruling appended to `AMENDMENT_INVERSIONGNN_2OBJ.md` (§6) | append, needs an **owner decision** | No, but it gates paper gap G3. |
| 8 | `docs/RUN_MANIFEST_LATEST.md` pointer (§7.1) | new file | No. Requires updating when a new manifest is written; the failure mode is a stale pointer, which is why it contains *only* a pointer. |
| 9 | Canonical notation table (§7.4) and `references_master.bib` + `KEY_MAP.md` (§7.5) | new files | No — existing drafts and their `.bib` files are untouched and keep compiling. |
| 10 | Append a "formal claims and verification status" section to `docs/CLAIM_LEDGER.md` (§5) | append, needs an **owner ruling** on C4 | No. Appending leaves all six cited line numbers in place. |

---

## 10 · What I deliberately do not propose

* **No moves, no renames, no deletions.** `REORG_REPORT.md` established that the
  move authority and the reference graph barely intersect here; I found nothing
  to overturn that.
* **No prepended status headers**, for the line-number reason in §0 — including
  on documents whose *names* actively mislead (`CURRENT_MODEL.md`,
  `PAPER1_FRAMING_AUTHORITATIVE.md`, `EXPERIMENT_PLAN.md`). `docs/INDEX.md` now
  carries their true status, which is the right place for it.
* **Nothing under `src/`.** See §0. A rename there invalidates banked artifacts
  through the Process-V2 identity hash, including the frozen `R_θ` behind every
  number in this paper.
* **No change to `paper/`, `paper_arxiv/`, `paper_iclr_control_substrate/` or
  `paper_iclr_stochastic_rewriting/`.** They are preserved deliberately and are
  the technical sourcebook this draft was transplanted from. I read them and
  wrote nothing.
* **No consolidation of the four registries.** That is `REGISTRY_AUDIT`'s and
  `docs/INDEX.md`'s territory, and duplicating it here would create a fifth
  registry — the exact failure mode both documents describe.
