# Workstream

- **Name:** Workstream E — target-free Pareto / preference control
- **Claim ID:** Purpose layer, third scalable control capability (alongside
  exact-target recovery and realized-state retargeting)
- **Branch:** `codex/compose-pareto-control`
- **Base commit:** `a0e680d`
- **HEAD commit:** see `handoff.json` (`commit`)
- **Working tree clean:** yes at handoff
- **Status:** `DESIGN_ONLY` for everything except the census, which is
  `SMOKE_HELD_IN`
- **Held-out data opened:** **NO.** `reserve_source_keys` was never read.
  No Modal run was launched.

# One-sentence scientific question

> Can the same frozen COMPOSE process pursue user-specified tradeoffs between
> competing objectives when there is no answer molecule at all — and is the
> objective pair we would use for that even capable of showing it?

# Claim this work can support

> The Stage 0 headroom gate, with thresholds fixed before any number existed,
> selects an objective pair on which different preferences can produce genuinely
> different optimal futures; and a four-arm preference-control experiment is
> implemented, locally tested, budget-matched and parity-audited, ready for a
> costed held-in smoke.

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
- **Primary metric:** normalized hypervolume over **committed endpoints only**,
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
- `modal_apps/pareto_control_app.py` — the costed held-in smoke. **Not
  launched.**
- `docs/workstreams/pareto-control/FIGURE_DESIGN.md` — the qualitative figure,
  designed and not rendered, with a binding example-selection rule.

# Tests and smoke checks

| Test | Result | Artifact |
|---|---|---|
| `pytest tests/test_pareto_control.py` | **36 passed** | toy-graph suite, no kernel needed |
| Chebyshev selects a point no weighted sum can | pass | proves why weighted-sum-only would understate the front |
| verified and greedy commit different actions | pass | the D2 defect, as a unit test |
| verified never worse than greedy | pass | asserts the *theorem*, so no report may present it as evidence |
| cost ledger: native < raw, kernel counted once | pass | the two conventions cannot be silently substituted |
| HV cannot be inflated by dominated points | pass | the inflation channel |
| all five branches share the identical prefix | pass | contrast P6 |
| `scripts/pareto_instrument_gate.py` (design self-test) | **PASS** after two failures it caught | see below |
| `scripts/pareto_tradeoff_census.py` | ran to completion, held-in | `diagnostics/pareto_tradeoff_census.json` |

# Results

**These are `SMOKE_HELD_IN` census results. They are not a controller result and
there is no controller result in this lane.**

See `diagnostics/pareto_tradeoff_census.json` and the summary in `STATUS.md`.

# Gate verdicts

See `diagnostics/pareto_tradeoff_census.json` → `pairs[].gate` and `STATUS.md`.
Adoption is **positional**: the first pair in the predeclared order clearing all
five gates, not the best-scoring pair.

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

**3. Budget-axis ambiguity for generate-then-rank (design finding, not a bug).**
One kernel call yields ~600 candidates, so matching `gen_rank` on native oracle
calls hands it ~600x the closed-loop arms' kernel budget, while matching on
kernel calls starves it of molecules. P3/P4 are therefore reported as a bracket
at both matchings. See `DECISION_LOG.md` D-009.

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

# the held-in smoke -- NOT RUN, requires main-lane authorization
# 1. freeze diagnostics/pareto_control_cohort.json first
# 2. modal run --detach modal_apps/pareto_control_app.py --sources 12
# 3. modal app list   # MUST show `ephemeral (detached)`
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
scripts/pareto_tradeoff_census.py
scripts/pareto_instrument_gate.py
scripts/pareto_write_handoff_manifest.py
src/compose_v4/experiments/pareto_control.py
tests/test_pareto_control.py
modal_apps/pareto_control_app.py
diagnostics/pareto_tradeoff_census.json
```

# Recommended next action

One bounded action only:

> **Authorize the 12-source held-in smoke** in
> `modal_apps/pareto_control_app.py`, after freezing
> `diagnostics/pareto_control_cohort.json` from held-in sources under the same
> outcome-independent eligibility rule the retargeting cohort used. Held-in only,
> five arms, five preferences, budget 6, no `h_phi`, no held-out panel. Costed
> below.

**Cost.** ~700 kernel calls per source after the ~2x enumeration caching the
committed calibration shows. At `cpu=8.0` with `OMP_NUM_THREADS=4` (an
enumeration measured 5.2–8 s locally versus 22 s on the calibration's
`cpu=2.0, OMP_NUM_THREADS=1`), that is **~1.4 h per source**; 12 sources in 12
containers is **~1.4 h wall and roughly $18** of CPU. Halving the shortlist from
8 to 4 gives ~0.8 h and ~$10. A local-only alternative is ~1.6 h **per source**
sequentially, so only 1–2 sources are practical without Modal.

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
