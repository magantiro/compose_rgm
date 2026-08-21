# `paper_iclr2027/` — COMPOSE ICLR 2027 main text

**COMPOSE: Generator Matching for Trans-Dimensional Molecular Editing and Control**

A new draft. It does not modify, replace or supersede `paper/`, `paper_arxiv/`,
`paper_iclr_control_substrate/` or `paper_iclr_stochastic_rewriting/`, which are
preserved deliberately and were read but never written to. Nothing outside this
directory was created, moved, renamed or edited.

---

## Build

```
cd paper_iclr2027
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Verified with TeX Live 2024 (`pdfTeX 3.141592653-2.6-1.40.26`).

| check | status |
|---|---|
| compiles | yes, exit 0, no errors |
| undefined references / citations | 0 |
| overfull/underfull boxes | 0 overfull |
| **main text** | **exactly 9 pages** — `REFERENCES` begins at the top of page 10 |
| total document | 18 pages (9 main + references + appendix) |
| anonymity | `\iclrfinalcopy` commented out, so the style file substitutes *Anonymous authors / Paper under double-blind review*; no author names, no acknowledgements, no repository URLs in the main text |

### Style files — official ICLR 2027 author kit, in use

Downloaded from `https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip`.
No earlier year's style is used anywhere in this directory.

```
iclr2027_conference.sty  sha256 797deef41724e93761426ac0cbcca46279a91cc650dd1f0ce76a4f08d2098ea6
iclr2027_conference.bst  sha256 2d67552db7ed38ccfccb5957b52f95656e25c249724761d3cf5f7922ad1844c5
```

The kit states the 2027 limit as **9 pages of main text**, references unlimited,
expanding to 10 for rebuttal/camera-ready. The draft is at the limit, so any
addition must be paid for by a subtraction. See *Page budget* below.

---

## Files

```
main.tex                       preamble, \XXX and \gap macros, provenance rules
sections/abstract.tex          six moves, in order
sections/intro.tex             six paragraphs + three contribution bullets
sections/fig1.tex              Fig 1  possible / plausible / purposeful (+ goal-switch inset)
sections/process.tex           §2  the executable molecular reference process
sections/control.tex           §3  finite-horizon control
sections/fig2.tex              Fig 2  what is trained (R_theta / h_phi / F_hat-vs-h_phi)
sections/trajectories.tex      §4  controlling molecular trajectories
sections/experiments.tex       §5  experiments: correctness → competence → attribution → capabilities
sections/fig3.tex              Fig 3  standard-task competence
sections/fig4.tex              Fig 4  process-level consequences
sections/related.tex           §6  related work and positioning
sections/discussion.tex        §7  discussion, limitations, conclusion
sections/appendix.tex          figure plan, operator registry, quotient invariance, proofs,
                               controller spec, frozen-artifact audit, reported negatives,
                               capability designs, extended related work, positioning ledger,
                               reproducibility, gap register
references.bib                 24 entries, all transcribed from the existing bibliographies
```

**Figures 2 and 4 are in the appendix, not the main text.** Fig 2 is a
training-pipeline schematic and Fig 4's four panels are *all* blocked on unrun
experiments; at 9 pages they were the two least defensible uses of main-text
space. They are `\input` from `sections/fig2.tex` and `sections/fig4.tex` and
move back by relocating one `\input` line each.

---

## The two conventions this draft is built on

**1. No number appears that is not in a verified source.** The verified set is
exhaustively these seven files, and every numeric claim in the `.tex` carries a
`% source:` comment naming one of them:

```
docs/SESSION_RUN_MANIFEST_2026-08-18.md
docs/VALID128_K8_RESULT.json
docs/VALID128_CURVE.json
docs/EXTENDED_CURVE_64.json
docs/AMENDMENT_VALIDATION_128.md
docs/GRIDDD_JIN_PROTOCOL.md
docs/AMENDMENT_INVERSIONGNN_2OBJ.md
```

Grep the provenance trail with `grep -n '% source:' sections/*.tex`.

**2. Unmeasured quantities are `\XXX`, never guessed.** `\XXX` renders as a red
**XXX**, so an unmeasured cell is visible at a glance in the compiled PDF. There
are **32 XXX in the 9-page main text** and 59 in the whole document. Every one
sits inside a source-level comment block, so there are two independent ways to
find them:

```
grep -n 'GAP:' sections/*.tex        # source-level markers, with the run that closes each
grep -n 'XXX'  sections/*.tex        # every unmeasured quantity
```

Three marker classes, deliberately distinct:

* `GAP: NOT YET RUN` — the experiment has **not been executed**. Seven of these.
* `GAP: NUMBER NOT IN VERIFIED SOURCE SET` — the quantity **exists in the
  repository** but outside the seven files above, so this draft may not quote it.
  Three of these. They are promotions waiting on provenance, not experiments
  waiting on compute.
* `GAP: MISSING COMPARATOR ROW` — one of these, **G8**, and it is the most
  urgent item in this file. See the two red flags below.

---

## ⚠ Two red flags found while assembling this draft

**RED FLAG 1 — the comparator table omits a stronger published row.**
`docs/GRIDDD_JIN_PROTOCOL.md`'s published table, the only verified source for
comparator numbers, lists JT-VAE, CG-VAE, GCPN and GrIDDD and nothing else. The
project's own comparator audit records at least one further published method on
the same constrained-ZINC QED task at δ ≥ 0.4, labels it **"Tier A — must-cite"**,
and reports a QED success rate **higher than both GrIDDD's 45.1 % and this
paper's 54.7 %**. It survives in exactly two places:
`docs/PAPER_REFRAME_CONTROL_SUBSTRATE.md:445` (a SUPERSEDED / ARCHIVED document)
and the *committed* revision of `docs/VALID128_K8_RESULT.json`, whose
`comparator` block carried it with the note *"Tier A must-cite and is higher;
this is not SOTA"* — a block **deleted** in the current uncommitted working-tree
revision of that file.
Consequences already applied to this draft: the abstract no longer calls the
comparator "the strongest reported comparator" (it says "the comparator whose
protocol we follow"), Table 1 carries an explicit `\XXX` row for the missing
method, and the caption says the omission out loud. **Do not submit until G8 is
resolved**, either by adding the row or by stating in the caption why its
protocol does not match.

**RED FLAG 2 — the headline artifact is uncommitted, and its two revisions
disagree.** `git status` shows ` M docs/VALID128_K8_RESULT.json`. The committed
revision says `n_sources: 126`, `cumulative[7] = 62`, `runs: 1008`,
`extinct_runs: 761`, `work_transitions: 1147373`. The working-tree revision —
the one this draft quotes, and the one the run manifest agrees with — says
`n: 128`, `cumulative[7] = 63`, `runs: 1024`, `extinct_runs: 776`,
`work_transitions: 1167787`. The headline percentage is unaffected
(62/126 = 49.206 %, 63/128 = 49.219 %, both 49.2 %), but **the counts in Table 1
and the inference ledger in §5.2 come from a file that is not committed.**
Commit it, or reconcile the two revisions, before the draft is circulated.
`docs/ARCHIVE_AB_RESULT.json` and `docs/RECEDING_HORIZON_AB.json` are in the
same uncommitted state; neither is quoted here.

---

## Gap register — every marker, and the run that closes it

### `GAP: NOT YET RUN`

| id | where | what is unmeasured | the run that closes it |
|---|---|---|---|
| **G1** | `experiments.tex:94`, last `\method` row of Table 1 | success rate on the official 800-source cohort at k=20 | Run all 800 official sources × 20 candidates **once** with the frozen controller; bank per-source outcomes and the candidate-efficiency curve at every k from 1 to 20. One of the 800 was executed by a pre-freeze smoke test whose output was quarantined unread, so the cohort is *inferentially* unconsumed. |
| **G2** | `experiments.tex:129`, §5.3 | matched paired difference between the immediate and future-aware arms of Eq. (8); number of independent sources; both inference ledgers | With `F_psi_hat` fixed and `R_theta` frozen, run both arms from the frozen common initialization bank, same 5 preferences, same seeds, same returned-candidate budget. Report per-source paired outcomes, the paired difference with a bootstrap interval resampled over **sources**, and both inference ledgers separately and never pooled. **The paper's mechanism claim depends on this run.** Specified in `docs/AMENDMENT_INVERSIONGNN_2OBJ.md`, Addendum II, "The causal ablation this buys". |
| **G3** | `experiments.tex:153`, §5.4 + `app:multiobjtable` | APS, novelty, top-K diversity, hypervolume for the future-aware arm, the immediate arm, and the matched comparator rerun (2 objectives) | (1) train `F_psi_hat` on 10,000 random-ZINC labels under the comparator's sampling law; (2) score oracle-free `R_theta` rollouts with it and train `h_phi` to `E[g_lambda(X_b)]` per Eq. (9); (3) run **both methods** from the frozen initialization bank, same 5 preferences, same 5,000-evaluation budget; (4) report all four metrics for every arm, hypervolume under the preregistered origin reference point computed identically. Never pool the published row with the matched rerun. |
| **G4** | `experiments.tex:168`, §5.4 | the same four metrics, 4 objectives | Same procedure at 20,000 + 5,000. **Gated**: runs only if the 2-objective decision rule in `docs/AMENDMENT_INVERSIONGNN_2OBJ.md` is met. |
| **G5** | — (closed) | — | **CLOSED — held-out confirmation measured.** 40 source-disjoint held-out sources under two prefix histories, prefixes committed before objective $B$ existed; reported in §5.5 and `tab:retarget`. Primary state-reuse effect (continue from $x_3$ vs. restart from $x_0$) $+0.302$ [0.186, 0.413] potency-first and $+0.176$ [0.069, 0.283] developability-first, 35/40 and 29/40 paired wins. Artifact `diagnostics/retarget_heldout_confirmation.json`. **Outstanding: provenance promotion only** — that artifact still self-labels `DEVELOPMENT_PANEL_NOT_CONFIRMATION`, which is a registry inconsistency to repair before submission, not a reason to re-run. The originally planned "restart from the switch molecule with a fresh sampler" arm is vacuous: the controller is Markov in (molecule, goal, remaining budget) and carries no sampler state, so it is identical to continuation. The clairvoyant arm supersedes the planned static-compromise arm. |
| **G6** | `experiments.tex:182`, §5.4 + `app:capdesign` | complete-feasibility fraction, oracle calls wasted on infeasible trajectories, final quality | With a rule-closed predicate P, compare support restriction against endpoint rejection and a terminal penalty on identical sources and seeds. The feasibility fraction is 1 by construction under support restriction — an implementation check, not a finding; the informative quantity is the wasted budget. A cumulative constraint under state augmentation is a different mechanism, reported separately and never pooled. |
| **G8** | `experiments.tex:66`, Table 1 | the success rate of a further published, higher-scoring comparator on this task | Verify the value and its protocol against the published record; confirm the source cohort and similarity threshold match; promote it into the verified source set; add the row. If the protocol does **not** match, say so explicitly in the caption rather than omitting it silently. **This is a claim-discipline hazard, not a cosmetic gap** — see RED FLAG 1 above. |
| **G7** | `experiments.tex:193`, §5.4 + `app:capdesign` | hypervolume at matched total oracle budget against independent restarts; nondominated molecules per oracle call | Fan one saved prefix into several preference-conditioned continuations under Eq. (9) with the budget the prefix left; compare against independent restarts from the source lead at the **same total** oracle budget. Show the accounting: shared-prefix cost counted **once** for the fan and once per branch for the restart arm. |

### `GAP: NUMBER NOT IN VERIFIED SOURCE SET`

| id | where | what is unquoted | how to close it |
|---|---|---|---|
| **U1** | `experiments.tex:26`, §5.1 | total-variation residual of the exactly-controlled terminal law against the analytic tilt; support-violation count; backward-equation residual; the same three after a mid-trajectory goal replacement | The check **has been run** — `diagnostics/exactness/editing_v2_experiment_b_exact_control.json` exists — but that artifact is outside the seven verified sources and self-labels `DEVELOPMENT_RESULT_NOT_PAPER_BEARING` on a carbon-only slice. Either **(a)** promote it into the verified set with provenance (producing script, git commit, input hashes) and quote it, or **(b)** rebuild the enumerable slice on the current operator registry over the preregistered goal/budget grid. |
| **U2** | `appendix.tex:79` | heavy-atom cap, count of neutral (element, valence) classes, element list, fraction of benchmark leads the vocabulary admits | Transcribe from the scope-fingerprint artifact with its hash. These are declared-scope constants, not measurements. |
| **U3** | `appendix.tex:149` | controller input dimension, embedding width, hidden widths, dropout, Bellman weight, epochs, patience, validation fraction, number of jointly trained goal regions | Transcribe in one pass from `docs/HPHI_QED_PREREGISTRATION.md` §6 and cite the document plus its hash beside the table. These are design constants fixed **before** the data, not measurements; they were left `\XXX` only because that document is outside this draft's verified source set. |

---

## Every number used, with its source

All measured values, exhaustively. Each also carries a `% source:` comment at
its point of use.

| value | where used | source file / key |
|---|---|---|
| 49.2 % = 63/128 @ k=8, prospective panel | abstract, Table 1, §5.2 | `docs/VALID128_K8_RESULT.json` — `coverage` 0.4921875, `cumulative[7]` 63, `n` 128 |
| 54.7 % = 70/128 @ k=12 | abstract, Table 1, §5.2 | `docs/VALID128_CURVE.json` — `coverage["12"]` 0.546875, `cumulative[11]` 70 |
| 128-panel curve `31 40 44 48 51 56 61 63 65 67 68 70` | Table 1 (lower block) | `docs/VALID128_CURVE.json` — `cumulative`, K=12 |
| 1,024 runs · 1,167,787 committed transitions · 248 contact runs · 776 extinct · 2.58 distinct returned per run | §5.2 inference ledger | `docs/VALID128_K8_RESULT.json` — `runs`, `work_transitions`, `contact_runs`, `extinct_runs`, `mean_distinct_returned` |
| 47/64 = 73.4 % @ k=20, development panel | Table 1 | `docs/EXTENDED_CURVE_64.json` `ALL.curve[19]`=47; the percentage is stated in `docs/SESSION_RUN_MANIFEST_2026-08-18.md` |
| 64-panel curve `28 32 … 46 47` (k=1…20) | Table 1 (lower block) | `docs/EXTENDED_CURVE_64.json` — `ALL.curve`, `of` 64 |
| GrIDDD 45.1 %, JT-VAE 8.8 %, CG-VAE 4.8 %, GCPN 9.4 % (all @ k=20, 800 official) | abstract, Table 1 | `docs/GRIDDD_JIN_PROTOCOL.md` — "The published table — cited, never recomputed". **This table is incomplete; see RED FLAG 1 / G8.** |
| task definition: QED ∈ [0.7,0.8] sources, 20 candidates, Tanimoto ≥ 0.4, success = ≥1 candidate with QED ∈ [0.9,1.0] | §5.2 | `docs/GRIDDD_JIN_PROTOCOL.md` — "The protocol, taken verbatim from GrIDDD" |
| controller: H=40 receding horizon, `b_eff = min(24,b)`, 32 particles, region (QED ≥ 0.90, sim ≥ 0.40), frozen head | §5.2 | `docs/AMENDMENT_VALIDATION_128.md` — "What is selected, and on what evidence" |
| 32 × 40 = 1,280 internal molecule evaluations per returned candidate | §5.2 | `docs/AMENDMENT_VALIDATION_128.md` — "Different per-candidate compute" |
| k=8 is fewer than the comparator's 20 (conservative direction) | §5.2 | `docs/AMENDMENT_VALIDATION_128.md` |
| bitwise reproduction of the frozen training assembly | §5.1 | `docs/SESSION_RUN_MANIFEST_2026-08-18.md` row 1 |
| extraction parity 1.49 × 10⁻⁷ | §5.1 | `docs/SESSION_RUN_MANIFEST_2026-08-18.md` row 5 |
| slice parity 64/64; 4/4 identical returned SMILES | §5.1 | `docs/SESSION_RUN_MANIFEST_2026-08-18.md`, "Parity checks that gate these numbers" |
| frozen head byte-identical by hash before and after the session | §5.1, `app:audit` | `docs/SESSION_RUN_MANIFEST_2026-08-18.md`, "Frozen assets used throughout" |
| our QED oracle equals the standard benchmark implementation exactly | `app:audit` | `docs/SESSION_RUN_MANIFEST_2026-08-18.md` row 13 |
| `R_theta` reproduces a banked candidate on a different image | `app:audit` | `docs/SESSION_RUN_MANIFEST_2026-08-18.md` row 14 |
| one-step policy: 0 successes in 320 development trajectories (64 × 5) | §5.3, `app:negatives` | `docs/GRIDDD_JIN_PROTOCOL.md`, ⭐ CURRENT AMENDMENT |
| the budget-extended value head was built, measured and **not adopted** | `app:negatives` | `docs/AMENDMENT_VALIDATION_128.md`, "Why the retrained twist is NOT in this configuration"; `docs/SESSION_RUN_MANIFEST_2026-08-18.md` rows 6–8 |
| panels verified pairwise disjoint; official cohort excluded before sampling | `app:audit` | `docs/AMENDMENT_VALIDATION_128.md` |
| one official source executed pre-freeze, quarantined unread | `app:audit` | `docs/GRIDDD_JIN_PROTOCOL.md`, "Record: one stray artifact, unread" + its correction |
| 10K + 5K (2 obj), 20K + 5K (4 obj), 5 preference vectors, C = 10 | §5.4 | `docs/AMENDMENT_INVERSIONGNN_2OBJ.md`, "Pinned by the paper" |
| comparator published APS 0.841, novelty 100 %, diversity 0.768 | §5.4, `app:multiobjtable` | `docs/AMENDMENT_INVERSIONGNN_2OBJ.md`, same section |

Nothing else in the document is a number.

---

## Claim discipline, as implemented

* Every empirical claim names its **estimand**, the **matched comparison**, the
  **independent statistical unit** and its **limitation**. Table 1's caption and
  §5.2 carry all four for the headline result.
* The comparator's published rows are labelled **reported context** and are
  visually and textually separated from any COMPOSE quantity. They are never
  described as the thing a COMPOSE number beats.
* **No matched-compute claim is made anywhere.** The draft states the QED result
  as *improved task performance at a measured inference-time cost*, never as
  greater efficiency, and never offers the absence of objective-specific
  retraining as compensation for inference cost — the project's standing rule.
* The paper does **not** claim that valid intermediates cause better endpoints.
  The endpoint mechanism is stated as future-aware control, and §5.3 says
  explicitly that the ablation establishing it (**G2**) has not been run.
* The 0/320 developmental negative is labelled developmental, and its limitation
  is stated in the same paragraph.

### Method note that supersedes an older design

§4 and §5.4 follow `docs/AMENDMENT_INVERSIONGNN_2OBJ.md`
**"ADDENDUM II — SUPERSEDES the `B_lambda` construction above"** (2026-08-19):
`h_phi` is *not* trained on observed true-oracle hits, there is *no* top-10 %
region and *no* threshold. A surrogate `F_psi_hat` is trained on the same 10K
random-ZINC supervision the comparator receives, oracle-free `R_theta`
trajectories are scored with it, and the controller target is an expectation of
a positive terminal desirability, `h_phi(x,λ,b) ≈ E_{R_theta}[g_λ(X_b) | x]`
with `g_λ = exp(−L_λ/τ_λ)`. Addendum I's construction is **withdrawn** — its own
diagnostic showed the five preferences collapsing to three regions — and a
warning to that effect sits at the top of `sections/trajectories.tex`. If you
are editing that section from an older document, you are editing the withdrawn
design.

---

## Page budget

The main text is **at** the 9-page limit, not under it. Adding a sentence
requires removing one. If more room is needed, in order of least damage:

1. Move `\input{sections/fig3}` to the appendix (≈ 0.2 page). Panels (a)–(c) are
   drawable from banked data, so this loses a real figure — prefer 2–4 first.
2. Trim Table 1's caption (≈ 0.05 page).
3. Move §5.1's parity sentence wholly into `app:audit`, leaving the `U1` gap
   paragraph (≈ 0.15 page).
4. Move §6 (Related work) wholly into `app:related`, leaving a one-sentence
   pointer (≈ 0.25 page).

Conversely, when **G1–G7** land and space is needed for real results, the
appendix already holds the material that came out: `app:figplan`,
`app:capdesign`, `app:related`, `app:multiobjtable`, `app:fig2`, `app:fig4`.

---

## Bibliography entries still needed

`references.bib` contains only entries transcribed verbatim from the existing
bibliographies in `paper_iclr_stochastic_rewriting/`, `paper/` and
`paper_arxiv/`. Nothing in it was invented. Two comparators are named in the
main text without a citation because no verified bibliographic record for them
exists in the repository:

* **The second multi-objective comparator (InversionGNN).** The repository
  records only `arXiv:2503.01488`, the upstream repository
  `github.com/ivanniu/InversionGNN` and commit `cfdf1d9`
  (`docs/workstreams/multiobjective/HANDOFF.md`,
  `docs/workstreams/multiobjective/BENCHMARK_ALIGNMENT_AUDIT.md`). Author list,
  title and venue must be verified against the published record before a
  `\citep` is added. The main text currently says "the comparator" and cites
  nothing, which is honest but must be fixed before submission.
* **CG-VAE and GCPN.** Present in Table 1 as inherited rows of the comparator's
  published table. They are cited *through* that table rather than individually;
  if individual citations are wanted, add them from the published record.

---

## What was read and not touched

`paper/`, `paper_arxiv/`, `paper_iclr_control_substrate/` and
`paper_iclr_stochastic_rewriting/` were read as the technical sourcebook —
definitions, the encoding-invariance proposition and its adversarial checks, the
finite-horizon theorem and its proof, the notation, and the preserved line
*"Terminal sanitization cannot make an earlier oracle query meaningful"* (from
`paper/sections/intro.tex:11`). Their structure was **not** inherited: it
follows development chronology, and this draft follows the
possible/plausible/purposeful thesis instead. None of those files was modified.

A companion document, `REPO_ORGANIZATION_PROPOSAL.md`, records the friction
encountered while assembling this paper and proposes fixes. It is a document,
not an action: it moves nothing.
