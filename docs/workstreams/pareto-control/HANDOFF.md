# Workstream

- **Name:** Workstream E — target-free Pareto / preference control
- **Claim ID:** Purpose layer, third scalable control capability (alongside
  exact-target recovery and realized-state retargeting)
- **Branch:** `codex/compose-pareto-control`
- **Base commit:** `a0e680d`
- **HEAD commit:** see `handoff.json` (`commit`)
- **Working tree clean:** yes at handoff
- **Status:** `DESIGN_ONLY` for the arms and the smoke plan; **`SMOKE_HELD_IN`**
  for the completed Stage 0 census
- **Held-out data opened:** **NO.** `reserve_source_keys` was never read.
  No Modal run was launched.

# One-sentence scientific question

> Can the same frozen COMPOSE process pursue user-specified tradeoffs between
> competing objectives when there is no answer molecule at all — and is the
> objective pair we would use for that even capable of showing it?

# Claim this work can support

> The Stage 0 headroom gate, with thresholds fixed before any number existed,
> **selects `potency vs developability`** as an objective pair on which
> different preferences select genuinely different candidates from the real
> executable support — 2.342 of 5 distinct Chebyshev selections across 120
> decision states, only 10% of states unanimous, and a front of 5.90
> nondominated candidates spanning 4.77 distinct weight-selected points. Both
> fallback pairs are ruled out structurally, because source similarity is inert
> under this executor. A four-arm preference-control experiment is implemented,
> locally tested, budget-matched and parity-audited, ready for a costed held-in
> smoke that this lane did **not** launch.

# Claims this work cannot support

- No controller result of any kind. No arm has been run on real molecules.
- No claim that COMPOSE beats any external multi-objective optimizer. No
  baseline is qualified in this lane.
- No claim about depths 2–5 of a trajectory: the census measured decision states
  at depths 0 and 1 only.
- No claim about amortization. **There is no `h_phi` in this lane**, by design:
  whether future-aware preference control buys anything must be established
  before anyone builds an amortizer for it.
- No medicinal-chemistry plausibility claim for any molecule.

# Frozen inputs

Hashes are in `handoff.json`, **recomputed from disk** by
`scripts/pareto_write_handoff_manifest.py` and independently re-verified by
`scripts/pareto_instrument_gate.py` check D4. None was transcribed by hand.

| Object | Path / ID | Verified? |
|---|---|---|
| Process-V2 chemistry | process identity via gate-zero `DECISION.json` | yes — v6 vs v7 compared field by field |
| `R_theta` checkpoint | `/artifacts/.../run_v2_01/R_THETA_CHECKPOINT.pt` (Modal volume) | **not needed by the census**; needed by the smoke |
| split / panel | `diagnostics/editing_v2_matched_validation_reserve_ids.json.gz` | yes |
| sampling law | frozen `R_theta`; census reads **support only**, never a probability | yes |
| goal/oracle | `artifacts/oracles/drd2_svm_v1/*` + RDKit QED/cLogP | yes |
| goal normalizers | `diagnostics/retarget_goal_language_normalizers.json` | yes |
| reachability corroboration | `diagnostics/retarget_calibration_result_3plus3_fixed.json` | yes |

**The census does not need the frozen `R_theta` weights.** The legal mask is
structural, so the canonical successor *support* enumerates locally. The local
Active8 pairs with gate-zero **v7** while the volume pins **v6**, so this was
checked rather than assumed: `process_identity_sha256`,
`gate_zero_structural_contract_sha256`, `contracts_binding_sha256`,
`active8_completion_sha256`, `enforced_structural_clauses`,
`model_family_counts` and `executor_rule_counts` are **byte-identical**. The two
decisions differ in corpus accounting, not in the executable support.

# Protocol

Full contract in `PROTOCOL.md`. Summary:

- **Panel construction:** held-in `training_source_keys` (96,094), heavy-atom
  band [18, 38] (the frozen cohort band), seed 20260813. Two decision-state
  depths per source: the source, and a **uniformly random** legal successor —
  drawn objective-independently so no candidate pair gets a state chosen in its
  favour.
- **Arms:** `unguided` · `gen_rank@greedy` · `gen_rank@verified` · `greedy_pref`
  · `verified_pref`, plus a common-prefix branching mode for contrast P6. No
  learned `h_phi`.
- **Primary metric:** **held-in-scaled** hypervolume (scaled by the held-in p99 vector `z*`, **not bounded by 1**) over **committed endpoints only**,
  exactly five per arm per source, at matched budget.
- **Secondary metrics:** HV-AUC against **both** oracle-call conventions
  (benchmark-native = distinct molecules scored; raw = every scoring
  invocation), preference coverage, nondominated-set size, feasibility, source
  similarity, endpoint structural diversity.
- **Independent statistical unit:** the **source**. Preference branches are
  repeated measures. Paired bootstrap, 2000 resamples, seed 20260813.
- **Allowed calibration:** shortlist size, `gen_rank` trajectory count for
  budget parity, census source count, the similarity normalizer under the
  predeclared held-in-median/IQR recipe.
- **Stop rules:** all three pairs fail the gate → hand the choice of a fourth
  pair back to main, do not invent one here; `greedy_pref` already covers the
  front → report it, do not force a controller claim; `gen_rank` matches at
  matched budget → the sequential claim fails but P5/P6 may stand; all five
  preferences give one endpoint → `INVALID_INSTRUMENT` for the front claim.
- **Forbidden adaptations:** changing any frozen objective constant or the clip;
  reordering the pair list; shrinking the budget until a controller wins;
  choosing the preference grid after seeing coverage; training anything;
  opening `reserve_source_keys`.

# What was implemented

- `src/compose_v4/experiments/pareto_control.py` — four arm types, augmented
  weighted Chebyshev + weighted-sum diagnostic, three-counter cost ledger,
  hypervolume / HV-AUC / preference coverage / diversity, source-level paired
  bootstrap, common-prefix branching. The process is **injected**, so the
  control logic is testable without RDKit, the kernel, or the checkpoint.
- `scripts/pareto_tradeoff_census.py` — Stage 0 census on two instruments.
- `scripts/pareto_instrument_gate.py` — executable gate D1–D6, runnable at
  design stage with no results in existence.
- `scripts/pareto_write_handoff_manifest.py` — manifest with hashes recomputed
  from disk.
- `scripts/pareto_select_cohort.py` + `diagnostics/pareto_control_cohort.json` —
  the smoke cohort, frozen on pre-control criteria and **verifiably disjoint**
  from the census sources (the census's selection is deterministic, so the
  script reconstructs and excludes them rather than asserting disjointness).
- `scripts/analyse_pareto_control.py` — shards to contrasts. Runs the gate
  **first**; on failure it writes `INVALID_INSTRUMENT` and no numbers at all.
- `scripts/pareto_render_status.py` — `STATUS.md` is generated from the census
  artifact, so no number on it is hand-typed.
- `configs/pareto_control_protocol_v1.json` — every frozen constant, the pair
  order, the gate thresholds, the parity table and the withdrawn statistics, in
  a form a program can check.
- `modal_apps/pareto_control_app.py` — the costed held-in smoke. **Not
  launched.**
- `docs/workstreams/pareto-control/FIGURE_DESIGN.md` — the qualitative figure,
  designed and not rendered, with a binding example-selection rule.

# Tests and smoke checks

| Test | Result | Artifact |
|---|---|---|
| `pytest tests/test_pareto_control.py` | **36 passed** | toy-graph suite, no kernel needed |
| `pytest tests/test_pareto_protocol_config.py` | **9 passed** | the config and the code cannot drift apart |
| `pytest tests/test_analyse_pareto_control.py` | **8 passed** | a failing gate yields NO statistics |
| `pytest tests/test_pareto_render_status.py` | **6 passed** | an all-fail census never says "the adopted pair" |
| Chebyshev selects a point no weighted sum can | pass | proves why weighted-sum-only would understate the front |
| verified and greedy commit different actions | pass | the D2 defect, as a unit test |
| verified never worse than greedy | pass | asserts the *theorem*, so no report may present it as evidence |
| cost ledger: native < raw, kernel counted once | pass | the two conventions cannot be silently substituted |
| HV cannot be inflated by dominated points | pass | the inflation channel |
| all five branches share the identical prefix | pass | contrast P6 |
| `pytest tests/test_pareto_instrument_gate.py` | **23 passed** | proves the gate can FAIL, on all five defect shapes |
| `scripts/pareto_instrument_gate.py` (design self-test) | **PASS** after three failures it caught | see below |
| `pytest tests/ --collect-only` | 4,030 collected, no import errors | the lane integrates cleanly |
| `scripts/pareto_tradeoff_census.py` | ran to completion, held-in | `diagnostics/pareto_tradeoff_census.json` |

# Results

**These are `SMOKE_HELD_IN` census results. They are not a controller result and
there is no controller result in this lane.**

`diagnostics/pareto_tradeoff_census.json` — 60 held-in sources, **120 real-fiber
decision states, mean fiber 586, 70,286 scored moves**, plus 81,500 MMP pairs
over 5,592 states. 4,515 s local, CPU only. **No Modal run.**

**Adopted pair: `potency_vs_developability`** — the first in the predeclared
order, all five gates passed.

| gate | value | threshold | verdict |
|---|---:|---|---|
| G1 alignment | rho **−0.274** | `< +0.70` | PASS |
| G2 local tradeoff | tradeoff moves **0.512**; both directions available at **1.00 / 1.00** of states | `>= 0.20`; `>= 0.25` | PASS, both instruments agree |
| G3 no domination | binding share **0.76 / 0.24** | `<= 0.90` | PASS |
| G4 no saturation | P **0.700**, D **0.000** | `<= 0.85` | PASS |
| G5 front richness | **2.342** of 5 distinct, **0.100** unanimous | `>= 2.0`; `<= 0.50` | PASS, **`OPERATOR_SET_DEPENDENT`** |

Front geometry: mean **5.90** nondominated of ~586 candidates per state, and
**4.77** distinct front points selected across a 101-weight Chebyshev grid —
i.e. the achievable front is a curve, not a point. Quadrant split: potency-up /
developability-down **0.390**, the converse **0.122**.

| rejected pair | rho | failed |
|---|---:|---|
| `potency_vs_source_similarity` | −0.060 | G2 (`P−/S+` **0.005** < 0.05), G3 (**0.907** > 0.90), G4 (S inert) |
| `developability_vs_source_similarity` | −0.119 | G2 (`D−/S+` **0.006** < 0.05), G4 (S inert) |

## Why source similarity was rejected as a Pareto axis

**ECFP4 source similarity was rejected as a Pareto axis because legal graph
edits frequently leave the fingerprint unchanged, making the objective locally
non-discriminative.** Across 20 sources, the similarity-maximizing 6-edit
rollout ended at ECFP4 Tanimoto **exactly 1.000** to the source every time;
median movement in `z_S` was **0.000**. Because the source also starts at
maximal similarity, the improve-similarity side of the tradeoff is essentially
empty — **0.005** of moves for P-vs-S and **0.006** for D-vs-S.

**This is an interaction, not a free lunch.** The degeneracy sits between the
**executor support** and the **ECFP4 representation**: it is not a claim that
source preservation is costless in general, nor a claim about this chemistry
independent of how similarity is measured. A different fingerprint, or a
different operator inventory, could behave differently.

It does mean both fallback pairs are ruled out here, so the predeclared order
had one viable entry — a measured outcome rather than a lucky ordering.
**No search for a better similarity metric was or will be undertaken in this
lane:** the fallback order was preregistered, potency/developability passed
strongly, and hunting for a metric that manufactures a second viable pair is
exactly the post-hoc tuning the preregistration exists to prevent.

## I-A IS THE INSTRUMENT OF RECORD; I-B CORROBORATES ONLY

| | I-A (real fiber) | I-B (MMP pairs) |
|---|---:|---:|
| mean distinct selections of 5 | **2.342** | **1.656** |
| unanimous states | **0.100** | **0.412** |
| mean front size | 5.90 | 2.15 |
| tradeoff-move fraction | 0.512 | 0.535 |

I-B's 1.656 sits **below** the predeclared G5 floor of 2.0: a proxy-only census
would have read borderline where the executable support is comfortable. The two
agree on sign statistics and diverge on selection statistics, as a neighbourhood
of median degree 1 versus 586 predicts.

This is now a **project-wide rule** — *"MMP proxies are diagnostics, not
authorities"*, `docs/workstreams/PARALLEL_WORKSTREAMS_AND_HANDOFF.md` — and this
census is one of its two cited cases, alongside the movability census that
wrongly suggested DRD2 was unreachable in four edits when the real fiber climbed
**+4.42 log-odds in three**. Per that rule, MMP mining remains a cheap
diagnostic and is not evidence about executable support; any manuscript claim
resting materially on MMP-based reachability or support estimates must be
re-evidenced with exact-fiber measurement.

G5's `OPERATOR_SET_DEPENDENT` flag belongs in the paper text, not a footnote.

## The live risk in the adopted pair

**Potency headroom is real but limited: G4 reach fraction 0.700**, i.e. 14 of 20
sources reach the held-in p99 within six greedy edits, against a 0.85 rejection
threshold. It passed, but not comfortably. If the smoke shows arms clustering at
the top of the potency axis, that is this number materializing.

By contrast developability has ample continuous headroom — **0.000** of
rollouts reach the clipped ceiling (median endpoint 0.627 against 1.327) — even
though the same committed shards show the *binary* developability region reached
on **0.933** of 3-edit rollouts. Reaching the region is not exhausting the axis;
see `DECISION_LOG.md` D-007.

# Were the efficiency metrics frozen before or after the numbers? BEFORE.

**Stated plainly because it was asked plainly.** The `HV_ref` rule and the
efficiency metric definitions were committed at **`c97571d`, 2026-08-13 02:30:36
EDT**. At that moment the authorized smoke had emitted **6 arm checkpoints and
ZERO `DONE` lines**, and `DONE` is the only place `modal_apps/
pareto_control_app.py` prints a hypervolume. **No HV number existed anywhere
when the definitions landed** — checked at the time, not reconstructed
afterwards, and nothing is backdated.

Frozen in that commit:

- **Reference corner (nadir)** = held-in **p5** from the frozen scales;
  **utopia** = held-in **p99**. Both are frozen inputs, never corners read off
  observed results.
- **`HV_ref`** = the **union nondominated front across ALL methods and ALL
  arms** on the frozen evaluation set. Where only internal arms exist, the
  **fixed box normalisation** `[r, z*]` established in advance. **Never
  COMPOSE's own front.** `union_reference_front()` raises
  `ReferenceFrontError` on a single method; three tests assert it, including
  that an external method extending the front *raises* the bar.
- **Efficiency is the primary axis**, not final HV: `HV(b)`, `HV-AUC` over a
  fixed budget, **`B_90`** (`None` when never reached, never the max budget),
  and **preference region coverage**, which scores one excellent potency-heavy
  cluster at **0.2** however good its hypervolume.
- **Two axes, never mixed.** Internal: HV vs completed controlled
  **trajectories**, COMPOSE arms only, headline **`N_90`**. External: HV vs
  **unique valid canonical evaluations** and vs **oracle requests**. External
  methods are **never** plotted on the trajectory axis — HN-GFN, GraphGA and
  REINVENT do not share a trajectory object.

**The framing this lane will use.** An external method reaching slightly higher
*final* HV while COMPOSE has substantially better HV-AUC, or reaches 90% front
coverage with several times fewer oracle evaluations, is a **good** result and
will be reported as such. Nothing will be tuned to win final HV. **If COMPOSE is
worse on both, that will be reported.**

# Gate verdicts

See the table above, `diagnostics/pareto_tradeoff_census.json` → `pairs[].gate`,
and the generated `STATUS.md`. Adoption is **positional**: the first pair in the
predeclared order clearing all five gates, not the best-scoring pair. The
thresholds were committed at `d206d55`, before the census had ever been run.

# Bugs, invalid instruments, and superseded runs

**1. A G4 saturation statistic of my own with no falsifying range.**
- *What was wrong:* G4 was operationalized as "the best candidate at a state
  reaches the pooled p99". Against a fiber of width `n`, that fires with
  probability `1 - 0.99^n` = **0.9973 at n = 589**, whatever the chemistry does.
- *How detected:* the first census smoke printed `S: 1.0`; applying the
  project-wide question — what value could this take if the hypothesis were
  false? — gave the answer "essentially none".
- *Did a conclusion depend on it:* **no.** It was withdrawn while only the reach
  fractions were visible, before any per-pair gate verdict was computed.
- *Artifact status:* `INVALID_INSTRUMENT`, withdrawn; registered in
  `pareto_instrument_gate.WITHDRAWN_STATISTICS` so it cannot be reintroduced.
- *Corrective commit:* `0d2b487`. See `DECISION_LOG.md` D-007.

**2. A parity confound in my own contrast table.**
- *What was wrong:* `PROTOCOL.md` §8 claimed contrast P1 (`greedy_pref` vs
  `unguided`) varied only the controller. It varies controller **and**
  objective — an arm with no controller cannot have an objective.
- *How detected:* `scripts/pareto_instrument_gate.py` check `D5_contrast_parity`,
  run as a design-stage self-test with no results in existence.
- *Did a conclusion depend on it:* **no.** Caught before any run.
- *Artifact status:* P1 demoted to `CONTEXT_ONLY`; may not carry a headline.
- *Corrective commit:* `0d2b487`. See `DECISION_LOG.md` D-008.

**3. A gate check that forbade the experiment instead of a confound — twice.**
- *What was wrong:* D6 originally failed any hypervolume contrast whose compute
  budgets differed. That makes **P2** (lookahead vs myopic) unrunnable, because
  a lookahead controller intrinsically spends more compute — that *is* the
  mechanism. Corrected, it then demanded compute parity on **both** axes at
  once, which makes **P3/P4** unrunnable at either end of their declared
  bracket, because one kernel call yields ~600 candidates and the two axes are
  not simultaneously satisfiable.
- *How detected:* running the analysis end to end on synthetic shards, and then
  working through `generate_then_rank`'s cost ledger.
- *Did a conclusion depend on it:* **no.** Both caught before any run.
- *Now:* equal endpoint counts are required on **every** HV contrast — that is
  the actual inflation control — while compute parity is enforced only on the
  **one axis** a contrast declares, and the unclaimed axis is reported.
- *Corrective commits:* `18c01ef`, `4019c30`. See `DECISION_LOG.md` D-010, D-011.

> A check that forbids the experiment is as wrong as one that permits a
> confound, and only the log shows which kind of error was made. The tempting
> repair in both cases — loosen the tolerance until it passes — would have been
> tuning the instrument to fit the data.

**4. Budget-axis ambiguity for generate-then-rank (design finding, not a bug).**
One kernel call yields ~600 candidates, so matching `gen_rank` on native oracle
calls hands it far more kernel work than the closed-loop arms, while matching on
kernel calls starves it of molecules.

**WITHDRAWN AS A COST CLAIM, pending audit (main lane, `2cc6fc3`).** The counts are real; the *interpretation* is not established. At matched kernel budget, preference control is charged for interrogating the ~600-wide legal successor fiber at every decision, while generate-and-rank is charged only for the terminal molecules it produced. **Those are different acts.** The ratio therefore conflates *property evaluations per unit of kernel work* with *cost of optimizing a molecule*, and only the first was measured. Until the audit lands, this appears as a **raw count with this caveat attached** and never as a cost ratio.

The audit is `diagnostics/pareto_oracle_accounting_audit.json`; see
`DECISION_LOG.md` D-016. P3/P4 are declared as a bracket at both
matchings — but **only the kernel-matched end is affordable in the smoke** (the
native-matched end needs ~2,600 trajectories and ~30 h per source), so P3/P4
will be **one-sided** until main authorizes the other end. See `DECISION_LOG.md`
D-009 and D-011.

**5. A hand-transcribed constant that was simply wrong.**
The developability clip ceiling was typed as `1.32669` in `PROTOCOL.md` and
`DECISION_LOG.md`; computed (`clip - tau*log 2`) it is `1.3267132048600137`, and
a gate threshold reads it. Corrected from the computed value, and
`tests/test_pareto_protocol_config.py` now recomputes it rather than reading it,
so the config and the code cannot drift apart again.

**6. Two contrasts that were being skipped silently.**
`scripts/analyse_pareto_control.py` dropped **P5** and **P6** without a word,
because they are within-arm preference contrasts rather than arm-vs-arm ones.
A contrast that vanishes silently from a report is indistinguishable from one
that was never run. Both are now implemented, and anything else missing is
recorded in `skipped_contrasts`. Corrective commit `d442c2e`.

# The potency ceiling: a live risk to WATCH, not a reason to change anything

Two numbers passed close to their lines, and they face **different** lines:

- **G3 binding share 0.76** — potency was the binding Chebyshev term in 76% of
  (state, preference) decisions, against a rejection line of **0.90**.
- **G4 potency reach 0.700** — 14 of 20 sources reached the held-in p99 within
  six greedy edits, against a rejection line of **0.85**.

**Both passed, and both lines stay exactly where they were frozen at `d206d55`.**
Tightening the task now because a passing number feels uncomfortably close to
its line would be moving a line after seeing the census — the one thing the
preregistration exists to prevent.

What the smoke checks instead:

1. Do extreme preference weights (`w=0.1` vs `w=0.9`) produce different
   **endpoints**, not merely different one-step choices?
2. Does the potency-heavy side stop moving early?
3. Does one objective dominate most final Pareto points?
4. Do final preference selections stay differentiated?
5. Does HV stop improving because potency effectively saturates?

**If these come back badly the smoke can legitimately fail the Pareto experiment
despite Stage 0 passing.** That is an acceptable outcome and will be reported as
a failure rather than repaired.

# Known limitations

- **Census depth.** Decision states at depths 0 and 1 only. Whether the tradeoff
  structure persists at depths 2–5 is not measured; it is the first thing the
  held-in smoke will show.
- **The similarity normalizer is derived, not inherited.** `centre_S` / `s_S`
  follow the frozen normalizers' recipe (held-in median and IQR) but were
  computed in this lane. Gate statistics G1–G2 are rank/sign based and therefore
  scale-invariant, so the normalizer cannot have influenced them; G3 and G5 do
  depend on it.
- **The census reads support, not the reference law.** It never reads an
  `R_theta` probability. That is what makes it runnable without the checkpoint,
  and it also means the census says nothing about how *plausible* the tradeoff
  candidates are — only that they exist and are legal.
- **`gen_rank` cost model is projected, not measured.** Its trajectory count is
  computed from the control arms' observed ledgers at run time, but no `gen_rank`
  arm has yet run against a real fiber.
- **REQUIRED FIX BEFORE ANY RERUN — the census violates this project's own
  durability rule.** `scripts/pareto_tradeoff_census.py` held **75 minutes**
  (4,515 s) of work in memory and wrote one file at completion. A crash at
  minute 70 would have lost all of it. The workstream contract says *"every
  expensive job writes per-task durable shards before returning"*, and this job
  does not. **It survived only because the process outlived the agent that
  launched it** — the main lane verified PID 55513 still running at 98% CPU
  after this agent had stopped polling. That is luck, not durability.

  It was not changed mid-run, because editing the script that is producing an
  artifact breaks the provenance between the two. **The fix, required before the
  script is run again:** write a per-source shard after each source's
  enumeration and after each reach rollout; make the driver skip sources whose
  shard already exists; reconstruct the aggregate from shards. That is exactly
  the pattern `modal_apps/pareto_control_app.py` already implements, and it
  should not have been omitted here.
- **Local run environment depends on paths outside the repo** (Active8 root,
  gate-zero decision, materialized scorer). All three are arguments or
  environment variables with documented defaults; none is a session directory.

# Exact reproduction commands

```bash
# environment/setup -- all three inputs live outside the repo and are arguments
export COMPOSE_ACTIVE8=/Users/rmaganti/compose_trainset_backup/localprep/artifacts/editing_v2/process_v2_active8/8ecc0e5e825a15200560c58960d4f662c9ca23be785c825c9c24aa73308144bb
export COMPOSE_GATE_ZERO=/Users/rmaganti/compose_trainset_backup/localprep/artifacts/editing_v2/process_v2_gate_zero_v7/DECISION.json
export COMPOSE_MATSCORER=/Users/rmaganti/compose_trainset_backup/materialized_scorer

# unit tests -- no kernel, no checkpoint, ~0.2s
python3 -m pytest tests/test_pareto_control.py -q

# instrument gate -- runs at design stage, before any result exists
python3 scripts/pareto_instrument_gate.py

# the census -- held-in only, local, ~70 min single-threaded
OMP_NUM_THREADS=4 python3 scripts/pareto_tradeoff_census.py \
    --sources 60 --reach-sources 20 \
    --out diagnostics/pareto_tradeoff_census.json

# manifest -- hashes recomputed from disk
python3 scripts/pareto_write_handoff_manifest.py

# the held-in smoke -- AUTHORIZED by main and LAUNCHED 2026-08-13 02:20:53 EDT
# app compose-v4-pareto-control / ap-YJkZtmnhwS8i7RTWNq0Br9, state verified
# `ephemeral (detached)`, 13 tasks. NO client-side timeout wrapper.
modal run --detach modal_apps/pareto_control_app.py --sources 12
modal app list --json    # State MUST read `ephemeral (detached)`

# retrieve the durable per-source shards, then analyse
modal volume get compose-v4-artifacts \
    editing_v2/r_theta_run/pareto_control_smoke <local dir>
python3 scripts/analyse_pareto_control.py --shards <local dir> \
    --out diagnostics/pareto_control_analysis.json
```

# Durable artifacts

| Artifact | Path | Purpose |
|---|---|---|
| Protocol | `docs/workstreams/pareto-control/PROTOCOL.md` | the contract, committed before any measurement |
| Decision log | `docs/workstreams/pareto-control/DECISION_LOG.md` | including two defects caught in this lane's own instruments |
| Figure design | `docs/workstreams/pareto-control/FIGURE_DESIGN.md` | qualitative figure, not rendered |
| Census result | `diagnostics/pareto_tradeoff_census.json` | `SMOKE_HELD_IN` |
| Manifest | `docs/workstreams/pareto-control/handoff.json` | hashes recomputed from disk |

All hashes are in `handoff.json`. Nothing load-bearing lives only in a
scratchpad or `/private/tmp`.

# Files changed

```text
docs/workstreams/pareto-control/PROTOCOL.md
docs/workstreams/pareto-control/STATUS.md
docs/workstreams/pareto-control/DECISION_LOG.md
docs/workstreams/pareto-control/FIGURE_DESIGN.md
docs/workstreams/pareto-control/HANDOFF.md
docs/workstreams/pareto-control/handoff.json
configs/pareto_control_protocol_v1.json
scripts/pareto_tradeoff_census.py
scripts/pareto_instrument_gate.py
scripts/pareto_write_handoff_manifest.py
scripts/pareto_select_cohort.py
scripts/pareto_render_status.py
scripts/analyse_pareto_control.py
src/compose_v4/experiments/pareto_control.py
tests/test_pareto_control.py
tests/test_pareto_instrument_gate.py
tests/test_pareto_protocol_config.py
tests/test_analyse_pareto_control.py
tests/test_pareto_render_status.py
modal_apps/pareto_control_app.py
diagnostics/pareto_tradeoff_census.json
diagnostics/pareto_control_cohort.json
```

# Recommended next action

One bounded action only:

> **Decide whether to authorize the 12-source held-in smoke** in
> `modal_apps/pareto_control_app.py`. The cohort is already frozen at
> `diagnostics/pareto_control_cohort.json` (sha256 `adea8e5510852d69`), disjoint
> from the census sources. Held-in only, five arms, five preferences, budget 6,
> no `h_phi`, no held-out panel.
>
> **NOT LAUNCHED, and not this lane's decision.** At ~134 core-hours this is a
> step change over anything the project has run, so the authorization belongs to
> Rohin, not to an agent. Both cost points are given below so the choice is
> between two concrete options rather than one.

**Cost, two options.** ~700 kernel calls per source after the ~2x enumeration
caching the committed calibration shows. The census measured **~9 s per
enumeration locally** at `OMP_NUM_THREADS=4`, against 22 s on the calibration's
`cpu=2.0, OMP_NUM_THREADS=1`.

| variant | shortlist | per source | 12 sources, 12 containers | ~core-hours | ~cost |
|---|---:|---:|---|---:|---:|
| as specified | 8 | ~1.4 h | ~1.4 h wall | ~134 | **~$18** |
| **halved shortlist** | 4 | ~0.8 h | ~0.8 h wall | ~77 | **~$10** |

The shortlist-4 variant halves the lookahead breadth. That is a real reduction
in the `verified_pref` arm's search, not a free saving, and it should be an
explicit choice rather than a default.

A local-only alternative runs ~1.6 h **per source** sequentially, so only 1–2
sources are practical without Modal — enough to smoke the plumbing, not enough
for a source-level bootstrap.

# Actions explicitly not recommended

- Do **not** open `reserve_source_keys` or any held-out panel on this lane's
  evidence. There is no controller result yet.
- Do **not** train an `h_phi`. That is the mistake this project already made
  once; establish the effect first.
- Do **not** substitute an unclipped developability objective if a gate is
  uncomfortable. The escape hatch is closed in `PROTOCOL.md` §4 on purpose.
- Do **not** add a fourth objective pair inside this lane. If all three fail,
  the choice returns to main.
- Do **not** quote a single HV-AUC convention or a single `gen_rank` budget
  matching. Both are reporting choices that can decide the winner.
- Do **not** rerun `scripts/pareto_tradeoff_census.py` before adding per-source
  durable shards to it.

## P3 and P4 will be ONE-SIDED — this belongs in the paper, not a footnote

`gen_rank` can be budget-matched to the closed-loop arms on **one** compute axis
only. One kernel call yields ~586 candidates, so:

- **kernel-matched** (affordable, and what the smoke runs): `gen_rank` gets ~4
  trajectories against `greedy_pref`'s ~15,600 scored candidates. It is
  disadvantaged on the axis the multi-objective literature actually budgets.
- **native-oracle-matched** (not affordable): ~2,600 unguided trajectories per
  source, ~15,600 kernel calls, **~30 h per source**.

**With a ~600-successor fiber there is no single notion of matched compute, so
no arm may ever be described as globally "budget matched".** Pareto quality is
reported against **both** counters — unique valid oracle evaluations **and**
kernel / reference-process calls — and each arm's position on both is stated. A
kernel-matched generate-then-rank baseline may be vastly oracle-richer than the
closed-loop arms; that is fine, and it makes it a strong baseline **in one
resource dimension**, which is how it must be described.

The smoke produces only the kernel-matched end of the declared bracket, so
**any P3/P4 result is one-sided**. A reviewer who assumes oracle-call parity —
the standard convention — will read it as more favourable to COMPOSE than it is.
This belongs in the text where the comparison appears, not in a footnote.

# Main-session pickup checklist

- [ ] Read `PROTOCOL.md` before any result — it was committed at `d206d55`,
      before the census existed, and the git history proves the ordering.
- [ ] Verify all frozen-input hashes: `python3 scripts/pareto_instrument_gate.py`
      recomputes every one from disk (check D4).
- [ ] Confirm held-out-open status: **NO**, and no Modal run was launched.
- [ ] Reproduce one smoke: `python3 -m pytest tests/test_pareto_control.py -q`
      (0.2 s, needs no kernel).
- [ ] Inspect the two known-invalid instruments in this lane, both caught before
      any run: `DECISION_LOG.md` D-007 and D-008.
- [ ] Decide explicitly: merge, authorize the held-in smoke, or stop.
