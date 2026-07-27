# PAPER_STATUS.md

Status, evidence provenance, the fact table, and the private internal-name mapping for the manuscript in
`paper/`. This file is an authors' aid and is **not** part of the anonymous submission; it is the only
place internal engineering codenames appear.

- **Evidence commit:** `ee5a6ca` (branch `claude/control-closed-pareto-editing`); clean lineage tag
  `stage6-preflight-baseline` is an ancestor.
- **Draft state:** compiles in draft mode; main text (intro..conclusion) = **9 pages** (sentinel
  `mainbodyend` on page 9; ethics/reproducibility statements follow on page 10, excluded from the ICLR
  limit). Total **23 pages** with references + appendices.
- **Verdict:** `FULL_DRAFT_READY_FOR_RESULTS` (see bottom).

## 0. Titles and naming

Five ranked title candidates (the chosen default is in `main.tex`; "lead optimization" stays out of the
main title per the plan, in the abstract/primary section instead):
1. **Every Step Is a Molecule: Valid-Path Generator Matching for Controllable Molecular Optimization** (chosen).
2. Doob-Controlled Generator Matching on Valid Molecular Graphs.
3. Valid-Path Molecular Optimization with Exact and Dynamic Multi-Objective Steering.
4. Steering a Generator-Matched Molecular Rewrite Process: Exact, Dynamic, Pathwise Control.
5. Source-Conditioned Molecular Optimization by Doob-Transformed Valid-Path Generator Matching.

**Method name.** Public macro `\method` = `\textsc{Compose}` (the spec default). The citation audit found
no *hard* published conflict that forces a rename for a molecular-generation method of this name; the name
is isolated behind one `\newcommand` in `math_commands.tex` and can be swapped in one line if a conflict is
later confirmed.

## 1. Fact table

Per-fact classification: **PROVEN** (theorem + appendix proof), **IMPLEMENTED** (verified against code),
**MEASURED** (implementation-validation number, appendix-only), **SCOPE** (a deliberate boundary).

| Fact | Repository evidence | Mathematical status | Paper location | Verified |
|---|---|---|---|---|
| 8 production operator families; cycle_insert/cycle_attach disabled everywhere | `factorized_tracelet_rate_model.py:94` MARK_RULE_NAMES; disabled ops enumerate nothing | IMPLEMENTED | Sec. 3, App. Operator registry | yes (grep + code) |
| GM loss is Poisson-Bregman: L = Lambda - r(log Lambda + log p) | `factorized_tracelet_rate_model.py:4136-4152` | PROVEN (App. C) + IMPLEMENTED | Eq. (bregman); App. C | yes |
| Total hazard = softplus(head) * has-legal-mark (terminal -> 0, no NaN) | `factorized_tracelet_rate_model.py:3040` | IMPLEMENTED | Eq. (rate); Sec. 3 | yes |
| Canonical successor kernel P(y|x)=sum p(a|x); sums to 1 | quotient over isomorphism classes; measure sum(exp)=1 to ~1e-7 | PROVEN (Prop. 2, App. D) | Sec. 4; App. D | yes |
| Finite-horizon Doob terminal tilt is exact | `tests/test_finite_horizon_editing_doob.py` terminal-tilt TV ~8e-17 | PROVEN (Thm. 1, App. E) + MEASURED | Sec. 5; App. E/K | yes |
| Dynamic retargeting exact; early-stop != terminal law | same test: continuation TV ~9e-17; early-stop TV > 0.02 | PROVEN (Cor. 3, App. F) + MEASURED | Sec. 5 (Cor.); App. F | yes |
| Editing = fixed-step embedded jump chain (hazard discarded); de novo = timed Gillespie CTMC | controllers loop range(max_steps); `tracelet_conditional.py:1124` waiting time ~Exp(Lambda) | IMPLEMENTED | Sec. 4 (three objects) | yes |
| Broad-organic scope, 15 (element,valence) classes; charges retained + preserved | `data/organic_corpus.py` BROAD_ORGANIC_V1; `source_corruption.py` net-charge invariance | IMPLEMENTED + SCOPE | Sec. 2; App. D; Limitations | yes |
| Quotient sampler kernel == raw sampler within 1e-5 | `tests/test_pancake_graft_kernel_equality.py` | MEASURED | App. K | yes |
| Warm-start (4->15 heads) + save/reload exact + loud-fail on missing metadata | `tests/test_checkpoint_roundtrip_and_init.py` | MEASURED | App. D/K | yes |
| Held-out rollouts start at lead, 100% all-intermediate validity | `tests/test_heldout_rollout_wiring.py` | MEASURED | App. K | yes |
| Tiny-overfit converges 5.23 -> 1.14 (bound 1.0), every family's prob rises | `stage6_smoke_summary.json`; `tests/test_stage6a_coverage_overfit.py` | MEASURED | App. K | yes |
| Stereo/isotope/radical unencoded (deliberate) | `molecular_graph.py:44-47` (docstring only, grep-verified) | SCOPE | Sec. 2; Limitations | yes |
| Frozen-time clock 1-exp(-t) train / 1-exp(-(t+d)/2) sample | `factorized_mark_conditional.py:190`; `tracelet_conditional.py:1155` | IMPLEMENTED | App. D | yes |
| One executor builds traces and applies sampled marks | `kernel.py:51`; trace.py; controllers | IMPLEMENTED | Sec. 3 | yes |
| Corpus = 500k GuacaMol subset under broad-organic (~98%) vs ~48% neutral C/N/O/F | scope projection; at-scale census running | SCOPE (projection; count pending) | App. D | projection |
| Production mixture percentages | proposed only; PHASE0B_DATA_RECIPE.md | SCOPE (not locked) | App. D | pending |
| Goal-conditioned controller h_phi trained artifact | not-started (only exact DP realized) | DESIGN (pending) | Sec. 6; App. controller | pending |

## 2. Internal-name mapping (private; never in the anonymous paper)

| Internal engineering name | Public manuscript term |
|---|---|
| B / de-novo B / Lineage B | base prior (de novo carbon-tree prior) |
| B-edit / corrupted-source-prior model | edit prior (universal source-agnostic edit prior) |
| FactorizedTraceletRateModel / tracelet | the generator-matched rate model (unnamed in paper) |
| pancake / AnalyticPancakeQuotientSampler / canonical_successor_distillation | canonical successor kernel / quotient sampler |
| griddd_value_guided_smc_controller / value-guided SMC (V0/V1/V2) | source-conditioned controller / controlled editing process |
| ORGANIC_VOCABULARY / BROAD_ORGANIC_V1 | broad-organic vocabulary / declared vocabulary |
| CNOF (4-class) / CNOF-gate | four-element (C/N/O/F) base / neutral C/N/O/F filter |
| Phase 0b / Stage 6 / A1-A7 audit items | (not referenced in the paper) |
| Jin-800 / cnof_leads | the (provenance-pending) lead-optimization source set |

## 3. What is done vs pending
- **Done (in the manuscript):** all theory + proofs; the operator registry; the GM objective and kernel
  normalization; the finite-horizon Doob transform with all four corollaries; the controller design (exact
  DP realized; amortized as an approximation, no invented loss, zero METHODCHECK); the full experimental
  program with fairness fixed in advance; related work with verified citations; limitations, ethics,
  reproducibility, LLM-use, and anonymous-release checklist; the result-registry macro system; the
  CLAIM_LEDGER and RELATED_WORK_MATRIX; the adversarial review with revisions.
- **Pending (registry placeholders):** every at-scale empirical value (RQ1-RQ5 and de novo), the
  goal-conditioned controller as a trained artifact, the docking/property oracles, external baseline runs,
  the at-scale corpus census and locked mixture, and the rendered figures (Pareto fan, E0 panels).

## 4. Build
- Draft (placeholders visible): `cd paper && make paper-draft` (or `bash scripts/compile_paper.sh draft`).
- Final gate (fails while anything is pending): `make paper-final`. Currently blocks on 34 pending result
  keys + 2 pending figures, as intended.
- Output: `paper/manuscript.pdf`.

## 5. Verdict
`FULL_DRAFT_READY_FOR_RESULTS`. The manuscript is complete, compiles in draft mode within the 9-page main
limit, contains zero METHODCHECK, and asserts no empirical number ahead of its run. Dropping in the RQ
artifacts and re-running `make paper-final` promotes it to a final PDF without further prose surgery.
