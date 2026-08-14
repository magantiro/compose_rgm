# Workstream

- **Name:** Multiobjective evidence package — external baseline qualification and fairness (Lane 5)
- **Claim ID:** Pareto block, Q1–Q4; external-comparator half only
- **Branch:** `codex/compose-multiobjective-package`
- **Base commit:** `f6146d7` (`codex/editing-v2-successor-fiber-fastpath`)
- **HEAD commit:** see `handoff.json :: content_commit`
- **Working tree clean:** yes
- **Status:** `DESIGN_ONLY`
- **Held-out data opened:** **no** — this lane opened no data at all

# One-sentence scientific question

> Which published multiobjective molecular methods can be run faithfully against
> the frozen COMPOSE Pareto task, on what task, at what cost — and which of them
> cannot be compared to COMPOSE without silently changing what is being measured?

# Claim this work can support

> The multiobjective comparison splits into two panels that must not be mixed: a
> source-conditioned panel that no currently qualified external method can enter,
> and a global-competence panel in which HN-GFN is the required external row and
> InversionGNN is conditional. The obstruction is structural and cited to
> specific lines of the authors' own code, not to a judgement about method
> quality.

# Claims this work cannot support

- Any statement about how COMPOSE performs against any external method. Nothing
  was run.
- Any cost measurement. Every figure is a published number or a projection.
- Bit-level DRD2 parity between HN-GFN's sklearn path and COMPOSE's numpy path.
  The shared model file is verified; score identity is not.
- Anything about pCoMole beyond title, authors and venue. The paper was not read.
- Any reinterpretation of the 12-source smoke. It is cited, never reworked.

# Frozen inputs

| Object | Path / ID | SHA-256 / identity | Verified? |
|---|---|---|---|
| Process-V2 chemistry | not touched by this lane | — | n/a |
| `R_theta` checkpoint | not touched by this lane | — | n/a |
| split / panel | `diagnostics/pareto_control_cohort.json` (12 held-in sources) | cited from Lane 4, not re-derived | read-only |
| sampling law | frozen COMPOSE reference process | — | read-only |
| goal/oracle | `artifacts/oracles/drd2_svm_v1/drd2_oracle_manifest.json` | source pickle `dbc473fca922c834dbaee6eaba832caaff26d4f891734078fb1af359a111100f`, 35,417,609 bytes | **yes — recomputed against HN-GFN's shipped copy** |
| objective pair and scales | `diagnostics/pareto_tradeoff_census.json :: frozen_scales` | `utopia_p99 = (2.7315048451066857, 0.555419816046901)`, `reference_p5 = (-1.2689119285385548, -1.3547257562637012)` | **yes — asserted by test** |
| HN-GFN upstream | `github.com/violet-sto/HN-GFN` | commit `90078b8ceeee3e907deeced9b096a6e395b71177` | yes |
| InversionGNN upstream | `github.com/ivanniu/InversionGNN` | commit `cfdf1d9a981ca4ce5dd7293dc38373ddf9377718` | yes |
| OP-GFN upstream | `github.com/yhangchen/OP-GFN` | commit `369e910891d3c9e20b8272d8291310fb9cc11c4d` | yes |

# Protocol

- **Panel construction:** Panel A source-conditioned (same supplied molecule,
  same edit budget, per-source fronts, source-level paired bootstrap); Panel B
  global competence (shared global freedom, shared objective budget). No
  comparison crosses panels.
- **Arms:** Panel A — COMPOSE fixed-preference control, generate-and-rank P3/P4,
  empirical-family reference ablation, preference-blind floor (all Lane 4's to
  run). Panel B — HN-GFN, InversionGNN conditionally, COMPOSE if a legitimate
  global arm exists.
- **Primary metric:** preference responsiveness (ordering, not distinct SMILES),
  then set-level front quality at the frozen `K`.
- **Secondary metrics:** HV-AUC with its mixture caveat, coverage, spread,
  diversity, envelope fidelity.
- **Independent statistical unit:** the source molecule.
- **Allowed calibration:** environment/compatibility work that restores intended
  behaviour; our evaluation and accounting adapter; applicability reporting.
- **Stop rules:** no compute without per-run approval; baseline engineering
  creep; licence stops; three integrity guards deep is a signal to stop; if
  Panel B cannot be frozen without shaping the task to suit a method, run none.
- **Forbidden adaptations:** supplying HN-GFN a Chebyshev; injecting a source
  molecule into a de novo method; training a checkpoint an author never
  released; porting sequence methods to graphs; mixing panels.

Full contract in `PROTOCOL.md`.

# What was implemented

- `src/compose_v4/experiments/multiobjective_qualification.py` — the common
  resource-axis vocabulary, the reconciliation of three lane-local counter
  schemes, the `surrogate_calls` axis, panel-admission rules, the cross-panel
  guard, the efficiency-comparability verdict, and the frozen-scale reader with
  its refusal of any reference not from the freeze.
- `tests/test_multiobjective_qualification.py` — 23 known-answer and adversarial
  tests.
- Six workstream documents in `docs/workstreams/multiobjective/`.

Nothing else. No adapter for any external method was written, because no
external method is authorized to run.

# Tests and smoke checks

| Test | Result | Artifact |
|---|---|---|
| `test_the_three_vocabularies_agree_on_the_algorithmic_axis` | PASS | — |
| `test_reconcile_keeps_the_raw_instrument_record_out_of_the_axes` | PASS | — |
| `test_a_de_novo_preference_method_is_panel_b_not_panel_a` (adversarial) | PASS | — |
| `test_a_source_conditioned_preference_method_enters_panel_a` (paired) | PASS | — |
| `test_cross_panel_comparison_raises` | PASS | — |
| `test_surrogate_asymmetry_makes_an_oracle_comparison_inadmissible` | PASS | — |
| `test_two_surrogate_free_methods_are_comparable` (adversarial pair) | PASS | — |
| `test_the_real_committed_census_matches_the_frozen_constants` | PASS | `diagnostics/pareto_tradeoff_census.json` |
| `test_a_reference_minted_from_a_front_is_refused` | PASS | — |
| full file | **23 passed** | — |

Each guard ships with a fixture in which its intended conclusion is false, per
the project rule. A guard that can only ever return one verdict has not measured
anything.

# Results

**None. This is `DESIGN_ONLY`.** No arm was run, no method was executed, no
number about COMPOSE-versus-anything was produced.

The two facts established are verifications, not results:

| Fact | Value | How verified |
|---|---|---|
| HN-GFN's DRD2 model file is COMPOSE's DRD2 source pickle | sha256 `dbc473fc…111100f`, 35,417,609 bytes | `shasum -a 256` on the clone vs the committed manifest |
| HN-GFN's published runtime | "our proposed HN-GFN costs 10 hours" on "1 Tesla V100 GPU", +33% with hindsight training | paper Appendix B.4, quoted |

# Gate verdicts

| Gate | PASS / FAIL / INCONCLUSIVE | Evidence |
|---|---|---|
| Every capability cell resolves to a primary source | PASS | file:line or paper section on every row of every matrix |
| No method admitted to Panel A without all three semantics | PASS | `panel_admission`, enforced by test |
| No efficiency axis dropped in translation | PASS | `reconcile_ledger` raises on unknown counters |
| pCoMole schema completed | **FAIL** | primary document unreachable; recorded `UNVERIFIED` |
| Panel B objective pair frozen | **INCONCLUSIVE** | open decision, deliberately left to main |

# Bugs, invalid instruments, and superseded runs

No runs were performed, so there are no invalid runs from this lane.

Two defects were found in third-party code and are recorded, not repaired:

- **InversionGNN `molecular/denovo.py:75-76`** — `model_ckpt = ""` followed by
  `torch.load(model_ckpt)`. Detected by reading the file. No conclusion depends
  on it; it blocks any run. Not corrected: supplying the missing checkpoint
  would be inventing the method's learned component.
- **InversionGNN `molecular/denovo.py:107` vs `molecular/inference_utils.py:153`**
  — six arguments passed to a four-parameter function. Detected by reading both
  lines. No conclusion depends on it. Repairing it is arguably permitted
  compatibility work; the decision is recorded but not taken.

# Known limitations

- Zero measurements. Every cost is published or projected.
- The local environment is rdkit 2025.09.6 / numpy 2.4.2 / Python 3.14, **not**
  the production pin rdkit 2024.03.5. No canonical key produced locally may be
  reused for a claim-bearing comparison. Nothing here produces one.
- Panel B's objective pair is unfrozen, and the choice is not neutral.
- Whether COMPOSE can enter Panel B at all is unresolved.
- pCoMole is unread.
- The predecessor lane's `oracle_accounting.py` and `shared_evaluator.py` live on
  `codex/compose-baseline-qualification` and were **not** cherry-picked here.
  `reconcile_ledger` maps onto their names without importing them, so the two
  branches can be merged in either order; the physical merge is main's call.

# Exact reproduction commands

```bash
# environment/setup -- nothing to install; the tests use only the project deps
cd <repo root>

# tests (the whole of Stage 1's executable content)
python3 -m pytest tests/test_multiobjective_qualification.py -q

# re-verify the DRD2 oracle identity claim (requires a clone of HN-GFN)
git clone https://github.com/violet-sto/HN-GFN.git /tmp/hngfn_verify
cd /tmp/hngfn_verify && git checkout 90078b8ceeee3e907deeced9b096a6e395b71177
shasum -a 256 oracle/scorer/drd2/clf_py36.pkl
# expect: dbc473fca922c834dbaee6eaba832caaff26d4f891734078fb1af359a111100f
# compare against artifacts/oracles/drd2_svm_v1/drd2_oracle_manifest.json
#   :: provenance.pickle_sha256

# analysis
# NONE -- no analysis was run
```

# Durable artifacts

| Artifact | Path | SHA-256 | Purpose |
|---|---|---|---|
| qualification module | `src/compose_v4/experiments/multiobjective_qualification.py` | see `handoff.json` | common axes, panel rules, frozen-scale guard |
| tests | `tests/test_multiobjective_qualification.py` | see `handoff.json` | known-answer and adversarial fixtures |
| protocol | `docs/workstreams/multiobjective/PROTOCOL.md` | see `handoff.json` | scientific contract |
| task matrix | `docs/workstreams/multiobjective/BASELINE_TASK_MATRIX.md` | see `handoff.json` | method × task, licences, costs |
| fairness audit | `docs/workstreams/multiobjective/SOURCE_CONDITIONING_AUDIT.md` | see `handoff.json` | the panel split and its evidence |
| lineage | `docs/workstreams/multiobjective/SAME_LAB_LINEAGE.md` | see `handoff.json` | four same-lab methods, one unread |

No load-bearing output exists only in a scratchpad. The cloned upstream
repositories are scratch and are not load-bearing: every claim drawn from them
carries a commit SHA so it can be re-derived.

# Files changed

```text
docs/workstreams/multiobjective/STATUS.md
docs/workstreams/multiobjective/PROTOCOL.md
docs/workstreams/multiobjective/BASELINE_TASK_MATRIX.md
docs/workstreams/multiobjective/SOURCE_CONDITIONING_AUDIT.md
docs/workstreams/multiobjective/SAME_LAB_LINEAGE.md
docs/workstreams/multiobjective/DECISION_LOG.md
docs/workstreams/multiobjective/HANDOFF.md
docs/workstreams/multiobjective/handoff.json
src/compose_v4/experiments/multiobjective_qualification.py
tests/test_multiobjective_qualification.py
```

No file belonging to Lane 4 was modified. No file outside this list was touched.

# Recommended next action

> Freeze Panel B's objective pair — recommendation `{GSK3β, JNK3}` with Panel B
> framed as a competence comparison among externals — and decide whether COMPOSE
> has a legitimate global arm. Both must be settled before any Panel B outcome
> exists. Everything else in this lane is blocked behind those two decisions.

# Actions explicitly not recommended

- Launching an HN-GFN smoke on the current authorization. It needs 10–13
  GPU-hours by the authors' own figure, and this lane is CPU-only and
  no-compute; that is a second, separate approval.
- Adding OP-GFN. Its licence forbids the derivative adapter that would be needed.
- Reconstructing InversionGNN's missing surrogate to obtain a table row.
- Implementing any same-lab method. None is approved and none is portable.
- Building a de novo COMPOSE arm to fill Panel B's COMPOSE cell.
- Quoting HV-AUC's 12W/0L as a set-level result.
- Cherry-picking this branch before Lane 4's top-up lands; nothing here depends
  on the order, but the Pareto docs will read oddly mid-repair.

# Main-session pickup checklist

- [ ] Read protocol before results.
- [ ] Verify all frozen-input hashes.
- [ ] Confirm held-out-open status. (It is `no`; no data was opened.)
- [ ] Reproduce one smoke. (`pytest tests/test_multiobjective_qualification.py`)
- [ ] Inspect known-invalid runs. (None from this lane.)
- [ ] Decide explicitly whether to merge, authorize held-out evaluation, or stop.
