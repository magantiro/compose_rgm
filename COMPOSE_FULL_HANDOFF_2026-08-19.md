# COMPOSE — complete project handoff
**Written 2026-08-19. Paste this whole file into a new chat as the first message.**
It replaces every earlier COMPOSE handoff. Where it conflicts with an older summary, this wins.

---

## 0. How to read this document

You are picking up a research project that is ~2 weeks from an ICLR submission. The project has
an unusually strict evidence culture: preregistration before every run, sealed panels, banked
JSON artifacts, and an explicit register of what was **refuted**. Three rules govern everything:

1. **Never invent a number.** Unmeasured quantities are written `XXX` in the paper and stay that way.
2. **A closed branch stays closed.** Section 8 lists them. Do not propose them as new ideas.
3. **Say which of measured / inferred / assumed a statement is.** Most bad calls in this project's
   history came from stating an inference in the voice of a measurement.

Repo: `/Users/rmaganti/compose_v2_work` (git worktree; bulk artifacts on Modal volume
`compose-v4-artifacts`). Entry point for a newcomer: `README_NAVIGATION.md`.

**IMPORTANT — this supersedes the state described in the previous ChatGPT conversation.**
That chat ended with "I'm running the 64, candidates 7–10" and listed the critical path as
*finish 20-candidate curve → settle H24 vs H40 → run the 128 → run the official 800*.
Since then: the 64 curve **completed** (47/64 = 73.4% @20), H24-vs-H40 **was settled** (H40 receding
horizon adopted for sampling; the retrained H40 *value head* was measured and **rejected**), the
**128-source prospective validation ran and is banked** (49.2% @8, 54.7% @12), the entire **MOLLEO
lane was closed** after four blind probes, and a **new external multi-objective lane against
InversionGNN** was opened and is now the live front. Do not re-plan from the old state.

---

## 1. What COMPOSE is — the thesis

> **COMPOSE learns one stochastic process over executable molecular edits and controls that frozen
> process toward changing molecular-design goals.**

The three-word spine the paper is built on:

**Possible → Plausible → Purposeful**

| layer | object | what it decides | varies by task? |
|---|---|---|---|
| **Possible** | exact executor + legal edit fiber `A(x)` | which molecular transitions are executable at all | **never** |
| **Plausible** | learned reference edit law `R_θ(y\|x)` (generator matching) | how probability should move over those transitions | **never** — one frozen model, all objectives |
| **Purposeful** | finite-horizon controller `h_φ(x,z,b)` | which plausible futures satisfy the current goal | **yes, by design** |

The single sentence for the introduction:
> COMPOSE separates what molecular transitions are executable, which executable transitions are
> plausible, and which plausible futures satisfy the current objective.

The single equation the reader must remember:

```
P_φ(y | x, z, b)  =  R_θ(y|x) · h_φ(y, z, b−1)  /  Σ_y' R_θ(y'|x) · h_φ(y', z, b−1)
```

**The key conceptual phrase, keep it verbatim:** *COMPOSE separates molecular plausibility from
molecular purpose.* `R_θ` = plausibility, `h_φ` = purpose.

**What this buys that endpoint generation does not:** every committed state is a complete molecule,
so (a) an oracle query at any step is meaningful, (b) a realized intermediate can be **retargeted**
when the objective changes, (c) constraints can be enforced **pathwise** rather than filtered at the
endpoint, (d) control can reason about **future reachability** rather than local property gain, and
(e) realized search effort becomes **reusable** (Pareto branching from saved prefixes).

**The endpoint-performance mechanism is future reachability, NOT intermediate validity.**
Valid paths are a structural property; they do not by themselves cause higher success rates. Never
write that they do.

---

## 2. The formalism (as written in the current draft, `paper_iclr2027/sections/`)

### 2.1 State space and exact support
- A **state** is a connected molecular graph over a declared vocabulary of elements, bond orders,
  formal charges and implicit hydrogens, in which every atom satisfies its valence constraint.
  `X` = such graphs with ≤ `N_max` heavy atoms = the **declared molecular state space**.
- An **edit mark** `a` is a typed rewrite rule with a match and payload, applied by a deterministic
  executor `T` under **propose–validate–commit**: per-atom valence and connectivity are revalidated
  before commit; otherwise the transition is not exposed.
- The **legal edit fiber** `A(x) = {a : a matches x and T(x,a) passes every guard}` is finite,
  state-dependent and enumerable. *Illegal chemistry is absent from the model rather than
  discouraged in its loss.*
- **Trans-dimensional:** `X = ⊔_n X_n` by heavy-atom count; every mark is a **birth** (n→n+1),
  a **death** (n→n−1) or **same-cardinality**. Births/deaths create/destroy typed graph entities
  that participate in valence, connectivity and ring perception — *not* padding flags. Within one
  budget a trajectory can grow, discover the growth was wrong, and delete.
- **Proposition (support closure):** if each family preserves valence at every atom it touches,
  leaves untouched atoms unchanged, keeps heavy-atom count ≤ `N_max`, and is admitted only when the
  successor is connected, then `T(x,a) ∈ X`. Hence every state committed by any process supported on
  `∪_x A(x)` lies in `X` — including the controlled process, which reweights existing marks and
  introduces none.

### 2.2 The reference law `R_θ`
- Rates factorize: `q_θ(a|x,t) = λ_θ(x,t) · p_θ(a|x,t)`, with `p_θ` normalized over `A(x)` and
  `λ_θ ≥ 0` vanishing where `A(x) = ∅` (so a state with no legal edit is terminal by construction).
- **Generator matching** (Holderrieth et al. 2025) is the fitting principle: real molecule pairs are
  joined by compiled programs of legal edits, each inducing a tractable conditional generator;
  regressing rates onto those under the **Poisson–Bregman divergence** recovers the marginal
  generator of the program mixture in the population limit. The teacher is **certified by executor
  replay**, not assumed.
- Editing spends a fixed budget of committed edits and discards the continuous-time clock, so the
  object everything downstream uses is the **embedded jump chain over distinct molecular successors**:

```
G_y(x) = {a ∈ A(x) : T(x,a) ≃ y}      S(x) = {y : G_y(x) ≠ ∅}
R_θ(y|x) = Σ_{a ∈ G_y(x)} p_θ(a|x)
```

- **`R_θ` is deliberately source-agnostic**: the network reads the current molecule and a model
  clock — never the source lead, objective, remaining budget, or protected mask. Source memory
  enters only via the initial condition, support constraints, and the controller.
  *This is the paper's only reuse claim.*

### 2.3 Canonical molecular successors (why aggregation is a contribution, not bookkeeping)
In 2-methylpropane `CC(C)C`, three equivalent terminal methyls give three deletion marks all
executing to propane; a model normalizing over **marks** gives propane 3× the mass of a
one-way successor — an inflation training cannot repair because it is a property of the encoding.
The fibers `{G_y(x)}` partition `A(x)`, so summing a normalized mark law over them yields a
normalized law on **distinct molecules**: `Σ_y R_θ(y|x) = 1` — **exact canonical normalization**.
Consequence: any controller that scores *molecules* (`h_φ`, hard state predicates) is invariant to
how marks split/merge within a fiber; a **mark-level** controller (e.g. top-k truncation) is **not**,
and COMPOSE uses none. Teacher supervision must aggregate the same way.

### 2.4 Finite-horizon control (the Doob h-transform)
- `h_φ(x,z,b) ≈ P_{R_θ}(reach B_z within b edits | X = x)` — a **budget-indexed** value.
- Controlled kernel: `P_φ(y|x,z,b) ∝ R_θ(y|x) · h_φ(y,z,b−1)`.
- **Theorem** establishes: normalization, **exact terminal tilt** under exact `h`, and **support
  preservation** (guidance can only redistribute probability across edits the executor already
  permits; it can never create an edge).
- **Exactness is budget-specific.** Early stopping does not inherit the terminal tilt: the marginal
  at `j < K` is an `h_{K−j}`-tilt. Stated as a remark, not hidden.
- **Retargeting corollary:** the exact `g'`-tilted law can be recomputed **from the realized switch
  state**, which is what makes mid-trajectory objective change coherent rather than heuristic.
- **Pathwise constraints:** `A_C(x) = {a ∈ A(x) : C(T_a x) = 1}` turns a **rule-closed** predicate
  into a pathwise invariant; no oracle call ever lands on an infeasible candidate. **Cumulative**
  constraints need state augmentation and are a *different mechanism*, reported separately.
- Reserve the word **exact** for: exact executor support, exact canonical normalization, exact
  dynamic programming, exact terminal tilt under exact `h`. Learned `h_φ` is **amortized/approximate**.

---

## 3. The code

### 3.1 Repository map (`/Users/rmaganti/compose_v2_work`)

```
src/compose_v4/          the model (285 .py files). A process-identity SHA over rewrite/ and
                         model/ gates every banked artifact — do not move files here.
  rewrite/               executor, operators, fibers, action codecs, tracelets, ring semantics,
                         compiler, canonical successors  (operators.py, fiber.py, kernel.py,
                         compiler.py, semantic_*.py, typed_ring_catalog.py, editing_v2_process_identity.py)
  model/                 rate models (factorized_tracelet_rate_model.py, relational_reroute_rate_model.py,
                         contextual_ring_restate_rate_model.py, segmented_successor.py, time_convention.py)
  gm/                    generator-matching objective
  policy/                controllers, incl. policy/task3/
  chem/ eval/ data/ oracles/ benchmark/ baselines/ experiments/ lipids/
modal_apps/              141 distributed experiment apps. The live QED/controller lane is hphi_*;
                         the MOLLEO lane molleo_*; the new multiobjective lane invgnn_*.
scripts/                 238 drivers/readers/probes; scripts/ops/ holds launch + poll helpers
docs/                    302 files (see docs/INDEX.md for CURRENT/SUPERSEDED status on each)
diagnostics/             committed result JSONs from editing-V2 / coherence lines
results/                 older committed result JSONs
configs/                 frozen inputs, launch configs, self-hashed contracts
artifacts/               mostly gitignored; artifacts/oracles/ and artifacts/h_phi_frozen_v1/ ARE tracked
tests/                   494 tests
third_party/             InversionGNN (the live external comparator)
paper_iclr2027/          THE LIVE PAPER DRAFT  (main.tex + sections/ + README.md gap register)
paper/, paper_arxiv/, paper_iclr_control_substrate/, paper_iclr_stochastic_rewriting/
                         preserved earlier drafts — read as technical sourcebook, never edited
```

### 3.2 The eight operator families
`cycle_insert`, `atom_insert`, `atom_restate`, `bond_reroute`, `atom_delete`, `bond_reorder`,
`ring_system_restate`, `cycle_attach`.

Scope facts that must not be overstated (from `docs/CLAIM_LEDGER.md`):
- Ring **generation** is compositional and catalog-independent (`cycle_close`/`cycle_open`).
- The legacy whole-ring **growth macro is disabled** (masked to empty support, log-prob −∞).
- A catalog-bounded whole-ring **deletion** implementation exists but is **disabled**.
- **Ring-editing evidence comes ONLY from the synthetic corruption lane.** `cycle_insert` (122,183
  transitions), `cycle_attach` (84,940) and `ring_system_restate` (16,380) are **entirely** in
  `reversible_synthetic_walk`, with zero in any data-backed lane. Scope the claim to *"recovers legal
  ring edits under synthetic perturbation"* — **never** "data-backed ring editing" or "scaffold hopping."
- `atom_restate:valence_state_change` is 220 of 1,803,032 admitted train transitions (0.012%) and is
  **reachability-redundant** (sulfoxide/sulfone reachable in two `atom_insert` steps). Retained as
  legal support; **not** claimed as a learned capability.

### 3.3 Frozen assets — the identity chain
| asset | where | hash / note |
|---|---|---|
| `R_θ` checkpoint (**never retrained, any session**) | volume `runs/run_v2_01/` | `sha256 c979cdb3…4e53de8` |
| H24 `h_φ` head | volume `hphi_v2/head.pt` | `sha256 9ea51ec4…` — verified byte-identical before/after a session that trained a different head |
| H24 norm | volume `hphi_v2/norm.json` | `sha256 4f93f7ec…` |
| Task-3 oracle bundle | **in repo** `artifacts/oracles/molleo_task3_v1/` | SHA-pinned manifest, no TDC/sklearn at runtime |
| DRD2 oracle | **in repo** `artifacts/oracles/drd2_svm_v1/` | see §5.7 for the parity story |
| frozen `h_φ` v1 weights | **in repo** `artifacts/h_phi_frozen_v1/` | |
| InversionGNN init bank (100 mols) | `docs/INVERSIONGNN_FROZEN_PROTOCOL.json` | bank `sha256 5fdb5f5e…`, ZINC pool `35e3f1a5…` |

`run_v2_01` identity chain: reserve `b580fdef6486` (15,031 reserve / 136,028 training / 151,059
total) · law `b0cc66f168f1` · manifest `e27494a46250` · store `b232a6fa069f` · gate FROZEN 9/9,
primary-metric SE 0.0237 (bound 0.05). Training cost to date ≈ **$2.6** on preemptible A10G.

⚠️ *Small discrepancy to reconcile before quoting:* `docs/DECISION_LOG.md` names **step 8,500** as the
selected checkpoint; `diagnostics/exactness/editing_v2_experiment_b_exact_control.json` records
`run_v2_01 selected_step 12500`. Check which is the shipped checkpoint before either goes in the paper.

### 3.4 Environment, pinned
Production image: **python 3.11, torch 2.4.0, numpy 1.26.4, scipy 1.13.1, networkx 3.3, rdkit 2024.3.5.**

**Fail closed on catalog drift.** Any claim-bearing evaluation or corpus compilation must abort if
the reconstructed RingCore catalog fingerprint differs from the frozen production value
`639ff6078c32d43c`. `neutralize_catalog_drift()` (which overwrites the expected fingerprint with
whatever the local env produced) is a **dev-probe helper only** — never in code producing a cited
number. This is not hypothetical: a local venv on rdkit 2026.03.4 drifted to `82fd910c`, and a
50-state parity gate found 2 states whose canonical successor inventory differed (aromaticity
perception change → same molecule, different canonical key). Nothing crashes; you get a split-brain
corpus.

### 3.5 Local commands
```bash
uv sync                          # or pip install -e ".[dev]"
export PYTHONPATH=src:scripts
export KMP_DUPLICATE_LIB_OK=TRUE # macOS OpenMP guard
export OMP_NUM_THREADS=1         # REQUIRED locally or the suite segfaults
pytest tests/                    # 494 tests
ruff check .                     # line-length 100, py310
python scripts/prelaunch_gate.py --corpus <name>   # tests+ruff+clean-tree+corpus+hashes
```
**Never launch a Modal job before `prelaunch_gate.py` is green.** SMC runs are single-process — the
multiprocessing rollout path fork-deadlocks on some macOS setups.

### 3.6 Modal runbook — three failure modes already paid for
```bash
# 1. Price it first (docs/MODAL_COST_MODEL.md; CPU $0.04716/core-hour). Budget is an authorization
#    ceiling, not a guideline.
# 2. Launch DETACHED from a COMMITTED tree, never a scratchpad worktree:
scripts/ops/launch_detached.sh <worktree> modal_apps/hphi_h40head_ab_app.py run.log --valid --k 8
# 3. Poll the VOLUME, never the log (a client log stops updating the moment the client dies):
scripts/ops/poll_volume.sh compose-v4-artifacts editing_v2/r_theta_run <shard-name>
# 4. Read results into a committed docs/*.json via a COMMITTED script:
python3 scripts/hphi_valid128_read.py
```
- `--detach` alone is **not enough**, and never wrap a detached launch in a client-side `timeout` —
  it cancels the job. App entrypoints should `drive.spawn()`, not `.remote()`.
- **Launch from the committed tree.** A pilot that passed from `/private/tmp` once preceded a full
  run that mounted a branch missing the module → **0/5,984**.
- **Persist expensive deterministic artifacts as their own job** (encode → persist → exit). Do not
  hold embeddings in memory until the end of a long job.
- `compose_v4/oracles/__init__.py` **eagerly imports joblib** — putting a new module there makes
  every consumer pay for the pan-lung oracle stack, absent from the Modal image. The DRD2 oracle
  therefore lives at `compose_v4.drd2_oracle`.
- **Cost estimates must be built from allocation × wall-time.** Modal bills **memory-time**. A probe
  quoted at ~$0.50 from kernel-call arithmetic actually cost ~$7.

### 3.7 Documents whose names claim more authority than they have
`docs/CURRENT_MODEL.md` (pre-`run_v2_01` framing) · `docs/PAPER1_FRAMING_AUTHORITATIVE.md`
(**ARCHIVED**, on the "do not read, cite or execute" list) · `docs/EXPERIMENT_PLAN.md` ("the only
current plan" — it is not) · `docs/PROJECT_STATUS.md` (2026-07-19) · `docs/PROJECT_BOARD.md`
(359 commits behind) · `docs/ESTATE_REGISTRY.md` · `docs/ARTIFACT_INDEX.md` · `CLAUDE.md` itself
(auto-loads every session and is **wrong about which plan governs**).

**The governing plan is `docs/MASTER_PLAN_EXPERIMENTS_AND_COMPARATORS.md`.** Lines 1–425 are the
governing section; line 426 opens `ARCHIVED PROVENANCE — NOT GOVERNING`. Two later blocks,
`⭐ CURRENT AMENDMENT` (line 1141) and `⭐⭐ CURRENT AMENDMENT II` (line 1740), both declare
themselves governing but sit **below** that divider — an unresolved wrinkle; know it before citing.

⚠️ **Reproducibility hazard, project-wide:** 15 of 15 sampled recent `docs/*.json` results record
**no git commit and no producing script**, and `docs/EXTENDED_CURVE_64.json`'s producer reads a
`/tmp` cache with no volume fallback. See `docs/REPRODUCIBILITY_HAZARDS_2026-08-19.md`.

---

## 4. Experimental doctrine — the rules that decide what counts as evidence

Three **non-interchangeable** kinds of evidence:

**A · External task competence.** Match the *published scientific task*, not the internal mechanics:
source cohort, objective/success criterion, returned-candidate budget, benchmark-defined oracle
budget (if any), normalization, metrics, seeds. **Each method uses its native inference procedure.**
COMPOSE is *not* forced to discard STOP, valid intermediates, future-aware control, or a
prospectively frozen sampler merely because a comparator lacks them. Do **not** impose equal internal
steps, kernel evaluations, wall-clock, FLOPs, or property evaluations unless the benchmark itself
defines that constraint.
- **GrIDDD QED:** the binding budget is **20 returned candidates/source**. It is *not* an oracle-call benchmark.
- **MOLLEO Task 3 / InversionGNN:** oracle counts *are* part of the task and therefore bind.
- A result may be described as **better benchmark performance** even at greater inference compute.
  It may **not** be described as more efficient unless separately established.

**B · Internal causal evidence.** Native COMPOSE vs COMPOSE **minus exactly that capability**, with
process, sources, objectives and relevant resources held fixed. **This is where matched compute
matters.** Examples: full vs size-fixed support; future-aware vs local control; closed-loop vs
generate-and-rank; hard support vs soft/post-hoc; endpoint-only vs pathwise; continuation vs restart.
External models are **not required** here.

**C · Process/property characterization.** Every committed state is a complete molecule; legal edit
families are actually used; variable-size edits occur; STOP/dead-end behavior. Descriptive; no
manufactured baseline.

**Resource accounting.** Log everything internally (training cost, inference wall time, kernel work,
property evaluations, committed edits, STOP step, particles/ESS). But the manuscript reports
benchmark-defined constraints plus whatever is materially necessary to interpret a claim — it does
not foreground every counter. **Training cost is never used to rhetorically offset inference cost.**

**Reuse claim, exactly.** The paper may claim **"no objective-specific retraining of the molecular
dynamics."** It may **not** claim "no objective-specific training anywhere" — controller/value
adaptation is a separate quantity and is stated explicitly. If one `h_φ` later generalizes across
goal regions, that is an *additional empirical result*, not an assumption.

**Why layer 3 (task policy) is allowed to vary.** For QED the control problem is hitting one fixed
region; for multi-objective work it is navigating *among* regions under an oracle budget. Forcing a
multi-objective optimizer to pretend it solves a fixed-target hitting problem would be worse science,
not fairer science. The discipline that keeps it honest, per task:
`disjoint development data → small principled policy family → FREEZE ONE → fresh evaluation`,
with **no outcome-driven rescue** after the freeze.

**No parallel scientific branching.** Critical path is sequential; a negative result closes **the
branch named by its preregistration** and nothing else.

---

## 5. Complete experiment history

### 5.1 `R_θ` training — established facts
- Reference-law-weighted NLL **5.4603 → 2.9238** (2.54 nats over two epochs), selected step 8,500.
- **Within-family identity NLL improved in all eight families:** cycle_insert 4.7147→0.9071 (−3.81);
  atom_insert 5.1526→3.2900 (−1.86); atom_restate 5.2047→3.3868 (−1.82); bond_reroute 3.6376→2.5431
  (−1.09); atom_delete 1.6492→1.0052 (−0.64); bond_reorder 1.5457→1.3122 (−0.23); ring_system_restate
  0.7139→0.5007 (−0.21); cycle_attach 2.5427→2.5360 (−0.01).
- **Improvement scales with training-scaffold support:** −0.22 / −0.21 / −0.38 / **−0.62** across
  bands 0 / 1–4 / 5–24 / 25+. Supportable phrasing: *transfers to unsupported chemistry, benefits
  increasingly from denser support.* **Not** supportable: strict monotonicity (the two low bands tie
  within 0.015 and inverted at the epoch-1 boundary).
- **No cross-cohort cost:** external 16-shard cohort 4.3364 → 4.5787 → 4.3817 → 4.3288, flat to
  slightly improving while in-population improved 0.20 nats.
- **Metric resolution ≈ 0.026 nats.** Adjacent evaluations closer than ~0.03 nats are **not
  interpretable** — four trend calls from consecutive points were each overturned at the next point.
- **The initialization is bit-reproducible** (0.00e+00 across three fresh containers, one after preemption).
- **The law never draws 29,600 of 136,028 rows** — whole synthetic strata whose real supply already
  meets target (`atom_delete|synthetic`, `atom_insert|synthetic` at exactly 0%). The sequence is
  fixed, so this is **permanent exclusion, not slow exposure**. Those strata degraded +0.80 to +1.79
  nats and are reported as diagnostics, never weighted into selection.

### 5.2 The QED / GrIDDD editing lane — the current headline

**Task (verbatim from GrIDDD/Jin, ZINC-250k):** 800 official sources with QED ∈ [0.7, 0.8]; **20
returned candidates per source**; a source is **solved** if ≥1 returned molecule has QED ∈ [0.9, 1.0]
**and** Tanimoto(x, x₀) ≥ 0.40.

**Native COMPOSE inference:** one candidate = one controlled trajectory `x₀ → … → x_τ` where every
`x_t` is a complete molecule; COMPOSE may prospectively **STOP** at the first state in `B_z`.
**20 controlled trajectories → 20 returned candidates.** Intermediates are **never** retrospectively
harvested to inflate the pool. STOP is an architectural capability, not extra budget.

**Controller (frozen before the validation panel was opened):** receding-horizon **twisted SMC**
realization of the central equation; goal region (QED ≥ 0.90, sim ≥ 0.40); **frozen H24 value head**;
horizon **H = 40**; budget index capped `b_eff = min(24, b)`; **32 particles**. One returned candidate
costs **32 × 40 = 1,280 internal molecule evaluations**.

**Chronology of the lane:**
1. **Policy B (one-step local tilt `π ∝ R_θ(y|x)·QED(y)`) — SEALED NEGATIVE, 0/320.**
   64 disjoint sources × 5 replicates, ~$0.18. Primary `1[QED≥0.9 ∧ sim≥0.4]` = **0/320**,
   source-clustered CI [0, 0]. Tanimoto ≥ 0.4 alone **0.6719** ✅; QED ≥ 0.9 alone **0.0063** ❌.
   Terminal QED median 0.7470 vs source median 0.7576. **0 dead ends, 0 zero-denominator fallbacks.**
   *Similarity is not the problem; optimization is.* A one-step tilt against a strong reference law
   cannot move a bounded objective. ⛔ **Never call this "COMPOSE"** — it is the **myopic-control
   ablation**. Barred responses: `QED^α`, temperatures, top-k, cap tuning.
2. **Horizon amendment.** Saturation criterion returned "no saturation ≤ 24" and therefore chose no
   horizon (decay ratio 1.00 at ≥0.85). H6 captured only 43% of 0.85-reachability, 33% of 0.90.
   Native controller = **max H24 with online first-hit STOP**. Unguided geometry: 0.80 → 32%,
   0.85 → 10.9%, **0.90 → 1.2% (rare but NONZERO)**, 0.95 absent. *The roads exist and the
   destination requires navigation.*
3. **Exact rejection sampler proved to reproduce the exact Doob kernel** (Stage A1, 6/6): exact
   because `h_φ ∈ [0,1]`; alias-correct for free because `h_φ` is a function of the canonical state;
   batching provably law-preserving by fixing proposal order and uniforms in advance.
4. **64-source development panel — CLOSED.** Region-`h_φ` showed strong navigation signal (source
   coverage 15.6% → 28.1%, 13 sources new vs unguided) while **capped rejection failed mechanically
   (94.3% cap-hit rate)**. The frozen cap-pressure criterion fired → **twisted SMC earned** (N=32,
   20 independent runs → 20 candidates). `h_φ` unchanged; only the inference realization changed.
   **Must not be described as decisively beating baselines** — paired source-level p = 0.096 and
   0.144, and the 64 was never an efficacy test.
5. **Development curve @1…@20 (`docs/EXTENDED_CURVE_64.json`):**
   `28 32 35 36 37 40 40 41 42 43 44 44 44 44 44 44 44 46 46 47` → **47/64 = 73.4% @20.**
   Strata: reliable 19/19 → 19; marginal 8 → 7; hard 37 → 21.
6. **H24 vs H40 receding horizon A/B (`docs/RECEDING_HORIZON_AB_64.json`), all 64 sources:**
   H24 `23 28 30 32`, H40 `28 32 35 36` at k=1..4. Paired: 5 sources H40-only, 1 H24-only.
   Contact 72 vs 88 of 256. Work 175,568 vs 277,243 transitions. **H40 receding horizon adopted.**
7. **The retrained H40 *value head* was built, measured and REJECTED** — four measurements:
   val Brier 0.0545 vs constant 0.0750 (trained fine), but transition-level AUC worse, **H40 head
   worse at every lookahead 4–40** (`docs/LONGRANGE_AUC.json`), and the paired A/B moved decision
   strata **4/8 → 3/8**. The frozen H24 head stayed. *This is a clean example of the freeze discipline
   working.*
8. **⭐ Prospective 128-source validation — the headline.** Panel prospectively held out, disjoint
   from train/dev **and from the official 800**, source QED ∈ [0.700, 0.800]; controller frozen
   beforehand in `docs/AMENDMENT_VALIDATION_128.md`; panel opened **once**.
   - `docs/VALID128_K8_RESULT.json`: **k=8 → 63/128 = 49.2%**; cumulative `31 40 44 48 51 56 61 63`;
     inference ledger **1,024 runs, 1,167,787 committed transitions, 248 contact runs, 776 extinct,
     2.58 distinct molecules returned per run.**
   - `docs/VALID128_CURVE.json`: **k=12 → 70/128 = 54.7%**; cumulative
     `31 40 44 48 51 56 61 63 65 67 68 70`. Not run beyond k=12.
   - Comparators (**reported context**, transcribed from GrIDDD's published table, 800 official @20):
     **GrIDDD 45.1%**, JT-VAE 8.8%, CG-VAE 4.8%, GCPN 9.4%.
   - **⚠️ VJTNN reports 60.6% on the same δ≥0.4 task** and is labeled Tier-A must-cite in the
     project's own comparator audit. It **exceeds both GrIDDD's 45.1% and COMPOSE's 54.7%.**
     *This result beats the closest per-step editor; it is NOT state of the art and must never be
     presented as such.* (VJTNN's protocol/initialization has not been audited to our standard.)
     **This is gap G8 and it is the most urgent claim-discipline item in the paper.**
   - Parity checks gating these numbers: slice parity 64/64 (a k=0..1 slice reproduces banked
     candidate 1 exactly); `hphi_recede_v5/005_H40.json` reproduced 4/4 identical returned SMILES.
9. **Still to run: the official 800 × 20, ONCE (gap G1).** One of the 800 was executed by a
   pre-freeze smoke test whose output was **quarantined unread**, so the cohort is *inferentially*
   unconsumed.

**Negative sub-results in this lane worth remembering:** shortlist retention — 0/21 hard rescues
inside top-128, median worst rank 271 (`docs/SHORTLIST_RETENTION.json`); particle multiplicity —
94.7% of particle-states already unique, ceiling 1.13× on hard sources; monotonicity violations
**not** concentrated in failures (Spearman 0.996 under isotonic projection) → monotonicity is not
load-bearing.

### 5.3 Banked internal causal results (the "COMPOSE-specific" evidence)

**(a) Finite-horizon control beats greedy — sealed 65-pair panel.**
65 endpoint-clean held-out exact-target pairs. Greedy **26/65** → verified rollout **40/65**;
paired **+21.5 pp [12.3, 33.5]**, rescuing **35.9% [21.2, 52.8]** of greedy's 39 failures.
Secondary: `h_φ` top-1 **40/65 at 35%** of continuations; similarity top-2 37/65 (79% gain retention)
at 52%; `R_θ` top-1 33/65 (50%) — **plausibility is not purpose.** Safety property held on every arm:
no arm lost a pair greedy recovers. Preregistered at
`diagnostics/editing_v2_sealed67_preregistration.json`; result at `diagnostics/editing_v2_sealed67_result.json`.
- **Do NOT headline the p-value.** McNemar one-sided p = 0.0156, but the rollout policy is
  constructed to be no worse than its greedy base, so `greedy-only = 0` is a **structural guarantee**,
  not an observation. The scientific effect is the rescue rate.
- **NOT claimed:** "`h_φ` beats similarity" (not preregistered; exact-target recovery privileges
  Tanimoto-to-the-answer, so this bed cannot settle it). **NOT claimed:** "`h_φ` reproduces full
  rollout" — it matches the *count*, not the *rescue set*.
- Earlier 24-pair version: 12/24 → 18/24, 6 of 12 missed targets recovered, none lost; best
  similarity 0.8589 → 0.9335; cost 5.2×.

**(b) When planning does NOT help — the characterization, not a failure.**
On DRD2 (C0), mean fresh regret was **negative at every depth** and sacrificial win rate 57.1% at
**0.5 SE** from chance; greedy solved 7/12 sources and outcomes were **bimodal** (1–5 edits, or never
close), leaving planning nothing to buy. Paired with (a), this *characterizes when remaining-budget
information matters* — a stronger framing than either result alone.

**(c) The amortization gap — why a learned value is not a cheap planner.**
`h_φ` trained on 14,110 labels from the exact rollout teacher does **not** retain the benefit:
rescue ranking 30.5% (2k) → 35.2% (5k) → 42.2% (10k) → 44.5% (all); harmful override 21.5% → 13.8%;
contrastive 52.3% → 65.9%. Baselines: greedy 0% by construction, random 20.8%, second-highest-similarity 33.3%.
Every metric beats baseline and improves monotonically — **but no override threshold yields net
benefit** (−10.2 states at margin 0, −0.2 at 4.0; the sole positive +0.5 at margin 6.0 overrides
9 of ~415 states, i.e. it has stopped acting).
**Structural cause, generalizes:** safe states (158) outnumber rescue-needing states (36) **4:1**, so
**precision dominates recall**. Explicit rollout never faced this because policy improvement
guaranteed it could never harm. **Not saturated** — rescue ranking still climbs at the largest
subset, so data limitation and genuine unlearnability are both consistent; this experiment cannot
separate them and the writeup must not imply otherwise.

**(d) Mid-trajectory retargeting — CLOSED, replicated on held-out.**
40 source-disjoint held-out sources × 2 histories, six arms; panel sealed at `947e297a`, prefixes
committed at `66687121` **before goal B was constructed**. All three preregistered questions replicate.

| | P-first | D-first |
|---|---|---|
| Q1 responsiveness (greedy-matched) | **+0.748** [+0.608, +0.891] 36W/0L | **+1.462** [+1.240, +1.691] 40W/0L |
| H2 sensitivity (greedy-matched) | **+0.367** [+0.251, +0.482] 37W/3L | +0.211 [+0.065, +0.361] 26W/14L, p=0.081 |
| **Q2 value of history (verified, PRIMARY)** | **+0.302** [+0.186, +0.413] 35W/5L | **+0.176** [+0.069, +0.283] 29W/11L, p=0.006 |
| Q3 **price of surprise** | **−0.252** [−0.364, −0.155] | **−0.428** [−0.580, −0.288] |

Development → confirmation: Q1 +0.716/+1.406 → +0.748/+1.462; Q2 +0.444/+0.290 → +0.302/+0.176;
Q3 −0.289/−0.481 → −0.252/−0.428. **Direction holds everywhere; Q2 shrank by ~⅓** — development
panels usually overstate magnitude, and the confirmation existed to find that out.
- **Do NOT write** "history's value is controller-dependent for D-first" — the D-first mean history
  effect is positive under **both** controllers; what differs is source-level sign consistency.
  Write: *"positive average effect under both controllers, but heterogeneous and less consistently
  positive under greedy."*
- **Do NOT write** "the goal changes at an arbitrary step." The machinery works from any realized
  state, but the scalable experiment fixes **τ=3, H=6**; invariance to switch time is **not established.**
- Wiring audit (appendix, not result): `greedy_restart` and `restart` return identical medians
  (−0.5316, −0.4283) across both histories — correct by construction.
- **Distinction from REINVENT:** REINVENT carries a learned policy + optimizer state and keeps
  updating; COMPOSE carries **the actual molecule `x_3`** and performs **zero parameter updates**.
  The state object is the difference, not the training.

**(e) Exact finite-horizon control — machine-precision verification.**
`diagnostics/exactness/editing_v2_experiment_b_exact_control.json`, enumerable slice: carbon-only,
heavy-atom cap 6, **966 states**, budget 6, target = "heavy atoms ≥ 6" (GROW), 770 target states,
reach probability 3.78e-5.
- terminal-tilt TV **1.77e-16** · support violations **0** · backward residual **0.0** ·
  max row-sum error 4.44e-16 · h-partition gap 2.03e-20
- **Retargeting from realized switch state `CC(C)C` to "< 4 heavy atoms" (SHRINK) with 3 budget left:**
  terminal-tilt TV **1.39e-17**, support violations 0, backward residual 0.0.
- **Unreachable target correctly reports h = 0** rather than being ε-patched.
- Self-labeled `DEVELOPMENT_RESULT_NOT_PAPER_BEARING` on a carbon-only slice → **this is gap U1**;
  either promote it with full provenance or rebuild on the current operator registry.

**(f) Closed-loop Pareto control (4A) and preference richness (4B/P0c)** are marked ✅ banked in the
master plan. **P0c is the full-information continuous-preference analysis / exact preference
partition — it is NOT a "Pareto ceiling"** and never an upper bound on attainable hypervolume.
*(Numbers for these live outside the paper's current verified source set; locate and re-verify before
quoting.)*

### 5.4 The MOLLEO lane — OPENED, PROBED FOUR WAYS, CLOSED

MOLLEO Task 3 (ICLR 2025): `max QED, max JNK3, min SA, min GSK3β, min DRD2`; 120 random ZINC-250k
init; **≤10,000 oracle calls**; 5 seeds; hypervolume. It replaced an earlier vague "custom MOEA" block.

All four search branches **closed**:
| probe | result |
|---|---|
| lazy verification gate (L1 vs L4) | did not diverge; HV moved +0.5%; **0 resampling events** |
| basin substrate | 42/96 reverse-verified routes; one-step fiber max median 0.365 |
| **fiber census — the decisive one** | **0 of 24 random-ZINC one-step fibers contain a JNK3 successor ≥ 0.3; max 0.16 over ~16,000 successors** |
| bridge probe (target supplied) | similarity 0.21 → 0.51, JNK3 to 0.37, **none reached 0.5** |
| coverage sentinel | 2× scaffolds, no JNK3 gain, moved *further* from actives |
| task-independent bridge training | goal-conditioned reachability, zero task-oracle calls, top-10 **61.5% at b=24 (51× chance)** |
| bridge navigation gate | **NULL**: 0 exact hits both arms, Δsim +0.006, sign test 28/50, p=0.24 |

**Why it failed and why that is not a COMPOSE indictment:** MOLLEO asks an optimizer to find a remote
kinase basin from random ZINC with **no task-specific training**. The neighborhood is genuinely flat.
**Also note:** a TDC image (PyTDC + `rdkit.six` shim + sklearn 1.2.2) was built and then made
unnecessary — `artifacts/oracles/molleo_task3_v1/` supplies the same oracles as SHA-pinned `.npz`
with no TDC/sklearn at runtime. **Do not reintroduce that dependency.**

### 5.5 The InversionGNN lane — THE LIVE FRONT (opened 2026-08-19)

**Why it fits where MOLLEO did not:** InversionGNN's protocol makes **task-specific training part of
the benchmark** — it spends oracle calls labelling ZINC and trains a property predictor before
optimizing. That licenses COMPOSE to train the multi-objective analogue of `h_φ` on the same budget:

```
InversionGNN   10K labels → F̂(x)          → gradient of predicted properties
COMPOSE        10K labels → h_φ(x,λ,b)     → future-reachability control over the exact legal fiber
                 SAME INFORMATION BUDGET. DIFFERENT INFERENCE PRINCIPLE.
```

**Pinned by the published paper:** oracle budgets **10K train + 5K optimize** (2 obj), 20K + 5K (4 obj);
the 5K is "1K per weight vector × 5 weights"; **5 preference vectors** by the spherical-coordinate
algorithm in App. D.3; **C = 10** molecules kept per generation; APS = average score of the top-100;
novelty; top-K diversity; vocabulary = 82 substructures appearing >1000× in ZINC-250K.
Reported: **APS 0.841, novelty 100%, diversity 0.768, HV 0.763 ± 0.031.**

**NOT pinned — and this is why we rerun rather than quote:**
- The **starting molecules are never specified** for the molecular experiments.
- The **hypervolume reference point is never stated** for the molecular tasks (only the synthetic
  task's (1,1)); they say they follow HN-GFN, which uses the origin — suggestive, not stated.
- The released `denovo.py` is **not** a faithful Table-3 reproduction: it hardcodes one start
  `C1=CC=CC=C1NC2=NC=CC=N2`, `population_size = 1`, and a single preference `[1,3]`, against the
  paper's C=10 and 5 weight vectors. That start is **not neutral** — on our frozen oracle it sits at
  **JNK3 = 0.100, the maximum of our entire random-ZINC dev cohort** (median 0.010), and it is an
  anilinopyrimidine (kinase hinge-binding chemotype).

**The design:** freeze a common initialization bank ourselves (deterministic, objective-blind, by
hash, sha256 recorded before either method runs — **100 molecules**, `docs/INVERSIONGNN_FROZEN_PROTOCOL.json`);
**run BOTH methods** from those starts with the same 10K labels, same 5 preferences, same 5K budget;
report published 0.841 as **reported context**, clearly labelled as a different (under-specified)
initialization — **never as the quantity our number beats.**
Preferences recovered from App. D.3 and verified by matching the paper's reported objective ratios
0.16/0.51/1.00/1.96/6.31 to 3 dp.
**PRIMARY metric: APS** (fully specified). **SECONDARY: HV under a preregistered origin reference
point**, computed identically for both arms, reported as our convention.
Scope: **2 objectives only** (JNK3 + GSK3β, both **maximised** — note MOLLEO minimised GSK3β, and the
frozen bundle stores `1 − gsk3b`, which **must be inverted**; that conversion is a parity check).
**Decision rule:** competitive or better on APS → escalate to 4 objectives at 20K+5K. Clearly worse →
**troubleshoot the multi-objective `h_φ`, do not escalate to hide a 2-objective deficit.**

**Four addenda, in order — read them as the design's evolution:**
- **Addendum I (WITHDRAWN).** Binary region `B_λ = {x : d_λ(x) ≤ q_λ}` with Chebyshev shortfall
  `d_λ(x) = max_i λ_i(1 − F_i(x))` and `q_λ` = 10th percentile over the 10K training labels only.
  Diagnostic (`docs/INVERSIONGNN_BLAMBDA_DIAGNOSTIC.json`): ~1,000 positives per preference as
  designed, 134–144 distinct sources — **but five preferences collapsed to three regions**
  (`B0 = B1`, `B3 = B4`, Jaccard 1.000) because with every objective near zero, `d_λ` is dominated by
  the larger λ component regardless of x. **Fatal second problem:** `R_θ` trajectories from random
  ZINC top out at **JNK3 = 0.22**, so a controller trained on *observed hits* can only reach regions
  the rollouts visited, while InversionGNN's GNN **extrapolates**. Not a fair contest — an artifact
  of our construction, not of COMPOSE.
- **Addendum II (GOVERNING).** Same supervision, different inference. Train a surrogate `F̂_ψ` on the
  10K labels; score **oracle-free** `R_θ` rollouts with it; train `h_φ` to an **expectation**, not a
  hitting probability:
  ```
  L_λ(x) = max_i λ_i (1 − F̂_i(x))            Chebyshev shortfall
  g_λ(x) = exp( − L_λ(x) / τ_λ )              positive terminal desirability
  τ_λ    = std of L_λ over the 10,000 TRAINING molecules (never an optimization outcome)
  h_φ(x, λ, b)  ≈  E_{R_θ}[ g_λ(X_b) | X_0 = x ]
  ```
  No binary region, no threshold, no positive examples needed; still a proper finite-horizon
  h-transform (state-independent constants cancel in the normalized kernel). The three objects
  separate cleanly: **`F̂_ψ` = what is desirable · `R_θ` = how molecules can move · `h_φ` = what is
  desirable in the future given how molecules move.**
  **The causal ablation this buys (gap G2, the paper's mechanism claim):** same 10K labels, same
  `F̂_ψ` **weights**, same frozen `R_θ`, same exact fibers, same init bank, same 5 preferences, same
  candidate budget — compare `R_θ(y|x)·g_λ(y)` (**immediate**) against
  `R_θ(y|x)·h_φ(y,λ,b−1)` (**future-aware**). *Both arms know the identical predicted landscape;
  if future-aware wins there is almost nowhere for the explanation to hide.*
- **Addendum III — three corrections to how this must be described.**
  1. The supervision corpus is **matched, not identical**: say *"InversionGNN-matched random-ZINC
     supervision, with the common evaluation initialization bank explicitly held out."* An earlier
     commit called it "InversionGNN's own construction" — **false**; their routine shuffles ZINC with
     no such exclusion. The exclusion is better hygiene **and** a deviation; both are stated.
  2. **"Same surrogate" means two different things.** Within-COMPOSE attribution uses literally the
     same `F̂_ψ` weights (airtight). COMPOSE-vs-published does **not** share a predicted landscape and
     must never claim to.
  3. **The mechanism, stated correctly.** 10K random-ZINC labels reach JNK3 max 0.400 (2 above 0.3,
     none above 0.5) while benchmark outputs reach APS ~0.84 — **every method here depends on
     extrapolation beyond its label range.** Never write "COMPOSE discovers high-activity regions
     from trajectory evidence." Write: **property learning says where value may be; molecular-process
     control says how to get there.** Extrapolation quality is a *shared dependency*, which is
     precisely why the claim rests on the propagation step.
- **Addendum IV — authorised Gate-2 numerical correction (not a redesign).** `g = exp(−L/τ)` produced
  numerically degenerate targets, so the computational representative became
  `g̃_λ(x) = exp(−(L_λ(x) − L*_λ)/τ_λ)` with `L*_λ` = mean of `L_λ` over the frozen training
  supervision. For fixed λ this is a positive x-independent constant → **normalized controlled kernel
  unchanged** (verified: `log g̃ − log g` constant to **7.1e-15** across 600 states × 5 preferences;
  normalized successor probabilities agree to **1.8e-15**). This is the softmax-max-subtraction
  manoeuvre, not a tuning knob.
  **Self-correction recorded:** the targets had *not* "collapsed to zero" — 0 of 600 were exactly
  zero, they ranged 1e-14 to 1e-26, and float32's min normal is 1.2e-38. **No underflow.** A 4-decimal
  print rendered ~1e-20 as `0.0000`. The real failure is dynamic-range collapse relative to the
  network's output scale. **Watch:** centering leaves a heavy right tail (per-preference median ~0.77,
  max 634 / 953 / 23005), so rank correlation is reported alongside R²; if the tail dominates, that is
  a finding to rule on, **not** something to silently transform away.

**Gate sequence, fixed:** (1) does `F̂_ψ` train sensibly? (2) does `h_φ` pass held-out
Bellman/future-value diagnostics? (3) on identical held-out decision fibers, does `h_φ` rank decisions
better than immediate `F̂_ψ`? (4) **only then** spend the 5K optimization budget.
**If 2 or 3 fails, troubleshoot the VALUE-LEARNING IMPLEMENTATION** — not the corpus, not `R_θ`, not
the operator set, not the benchmark, not the preference family.

**Latest banked result (commit `0a219a0`, today) — `docs/INVGNN_LOOKAHEAD_TRUTH.json`:**
an oracle-free high-Monte-Carlo probe of the **true** future value under `R_θ` (24 held-out states ×
16 candidates × 32 rollouts = 120 decisions, scored by the frozen surrogate, zero oracle calls):

| lookahead b | Spearman (median) vs immediate score | top-1 **flip** rate | median realized gain |
|---|---|---|---|
| **4** | 0.518 | **68.3%** (82/120) | **+19.3%** (abs +0.186) |
| **8** | 0.347 | **84.2%** (101/120) | **+21.4%** (abs +0.223) |

*Reading:* true future value disagrees with immediate scoring on 68–84% of within-fiber decisions,
worth ~+20% median realized gain. **This is a property of the TRUE value, not of a trained
controller** — it establishes that the nonmyopic gap is real and worth chasing; it is **not** the
benchmark claim (G2 still must be run).

Assets already built: `docs/INVERSIONGNN_FPSI.pt`, `docs/INVERSIONGNN_HPHI.pt`,
`docs/INVERSIONGNN_SUPERVISION_10K.json.gz`. The Addendum-I corpus `invgnn_v1/corpus_10k.json.gz` is
**quarantined as development evidence** (its 10,000 oracle calls are development spend, **not** charged
to the benchmark).

### 5.6 Structural limits and refuted hypotheses — DO NOT RE-TEST

| hypothesis | verdict | evidence |
|---|---|---|
| The compact successor fiber is the object to optimize | **wrong target** | partitions discarded by the training loop; only teacher fibers are read |
| Larger batches speed training | **refuted** | batch 128 is 3× worse than 32 |
| Vectorized backward pass gives a large speedup | **refuted** | 2%, not 20× |
| The model memorizes slot permutations | **refuted** | scorer equivariant to 1e-6 |
| The gap is a representation problem | **refuted** | provenance correlation ρ = +0.762 |
| Validation is a different chemical cohort | **refuted at role level** | 85.5% scaffold coverage — but see the correction in §5.7 |
| Held-out error grows with analogue-series depth | **refuted, opposite direction** | error *falls* 3.87 → 2.42 with depth, surviving fixed-zero-support (n=5,863) and fixed-molecule-size (n=6,992) controls |
| Most-oversampled strata overfit first | **refuted** | the three at the 3.0× ceiling improved most |
| `atom_restate` suffered capability collapse | **refuted** | its *identity* improved 1.82 nats; the joint gate was reading family-head reallocation |
| Cross-cohort performance is degrading | **refuted** | third and fourth readings returned to baseline |
| DRD2 is rugged enough that future value beats greedy | **refuted** | see §5.3(b) |
| Historical traces can be reconstructed from the corpus | **refuted, structurally** | corpora are samples of `(state, edit)` pairs across capability cells, **not paths**: 29,928 forked states in train; strict funnel reached **0** on both partitions |
| The `linker_positional_topology_analogue` lane implies two-cut/topology mining | **refuted** | it is a **post-hoc router label**; zero repo hits for `two_cut`/`double_cut`/`n_cuts` |

**Structural limit — one-cut MMP cannot produce diverse multi-step panels.** Over 972 nominated pairs:
160 `bond_reroute` + 146 `atom_restate` + 17 `bond_reorder` = **323 direct compiles**, against
**exactly 323 paths of length 1**. Everything of length ≥2 is `delete_insert_fallback`. **There is no
middle.** Therefore *every 4–6 step transformation this machinery can produce is a delete-insert
fragment swap by construction.* Consequences, all observed: `ring_delta` is **0 across all 91**
accepted pairs (the one-cut iterator skips ring bonds and requires an acyclic variable fragment);
the 86% `REFERENCE_DIP` rate is a **compiler artifact** (delete-then-insert mechanically dips).
The 91 pairs remain valid as a **held-out multi-step recovery cohort** but support **no non-locality
claim**. Provenance to claim: *held-out endpoints plus unseen supervised transitions* — **not**
"the model never saw any intermediate" (38 of 91 have an intermediate appearing elsewhere as a
training source; described, not excluded).

### 5.7 Corrections already made, and harness defects already paid for

**Claims made and then found wrong:**
- Scaffold coverage is **58.5%** of panel entries, **not 85.5%** — the 85.5% counted train-*role*
  sources across 30 shards; the model trains on the compiled library (106,759 of 564,316). This
  *reversed a retraction*: chemical novelty **is** substantial from the model's point of view.
- Support-matching does **not** flatter the number: an earlier note claimed 2.62 → 1.94, conflating
  the matched view with the supported-scaffold view. Training weights on the same per-band means give
  **2.74** — matching moves the number **+0.12, slightly worse**. The re-freeze buys an interpretable
  number, not a better one.
- **"Depth 0" meant acyclic, not shallow** — all 361 depth-0 entries had an empty Murcko scaffold and
  `if s:` treated the empty string as no-scaffold. A category error at the exact end of the curve the
  hypothesis was about.
- **The family floor was never enforced** — clamping to [floor, cap] then renormalizing divides by the
  sum, so a family pinned *at* the floor lands under it (claimed 0.05, delivered 0.0497). Replaced
  with water-filling.
- **The joint NLL is the wrong quantity for a capability gate** — it moves with family-head
  reallocation the model is entitled to perform (largest identity movement 0.062 nats against family
  movement 0.317). The gate now reads **identity**, over two consecutive evaluations.

**Harness defects that cost money or hid results (all fixed):**
- **No evaluation ran on the final step** — one epoch is 4,251 steps against an interval of 500, so
  runs ended with their last evaluation 251 steps stale. Step 4,000's weights are **permanently lost**.
- The eval line logged everything **except** the number that selects.
- Re-collating per batch cost **98.17 s per 32 examples** (97.2% of step time) → packed store, GPU
  data-wait fell to 0.93%.
- Throughput charged eval+checkpoint time (false WARN at every eval); the alarm watched only
  panel-native and missed a 9.2% deployment regression; the per-family breakdown was computed and discarded.
- The carve was defined over the wrong set (151,078 vs 151,059); four byte-identical duplicate entry ids.
- Cost estimates counted work, not billed resources (see §3.6).

**DRD2 oracle provenance (a model for how to port a benchmark oracle):** the classic SVM ships as a
Python-3.6 sklearn pickle; it is opened **once** by `scripts/drd2_oracle_extract.py` and the runtime
thereafter evaluates frozen arrays in numpy. Two things had to be *reproduced*, not assumed:
**libsvm's Wu-Lin-Weng coupling** (short-circuiting to the Platt sigmoid left a **1.7e-3** discrepancy
— enough to move molecules sitting on the 0.5 threshold), and **the Platt orientation** (an inverted
oracle returns plausible probabilities while rewarding the wrong molecules; both orientations are
scored and the rejected one is *asserted to fail* parity, so agreement is evidence rather than luck).
Parity **2.19e-14** on probabilities, **1.35e-13** on decision values over 200 molecules.
**The task uses two different fingerprints:** activity = count-based FCFP6 (radius 3,
`useFeatures=True`, folded by modulo); similarity constraint = ECFP4 bits (radius 2, 2048).
Conflating them silently redefines the benchmark.

---

## 6. Where the project stands right now (2026-08-19) and the critical path

**Last commits:** `0a219a0` *"The nonmyopic gap is real: lookahead under R_theta ranks decisions
differently from immediate score"* ← today's InversionGNN lookahead-truth probe.
`7cfadb8` *"Restore the comparator caveat silently dropped from the headline artifact"* ← the VJTNN
caveat, restored.
Working tree: `docs/ARCHIVE_AB_RESULT.json` and `docs/RECEDING_HORIZON_AB.json` modified;
`README_NAVIGATION.md`, `REORG_REPORT.md`, `docs/INDEX.md`, `paper_iclr2027/` **untracked**.

| lane | status |
|---|---|
| `R_θ` reference process | **frozen, never retrained.** Done. |
| QED / GrIDDD editing | 64 dev **closed** (73.4% @20); 128 prospective **banked** (49.2% @8, 54.7% @12); **official 800 × 20 NOT RUN (G1)** |
| Finite-horizon control vs greedy | **banked** (+21.5 pp, 35.9% rescue) |
| Mid-trajectory retargeting | **banked and CLOSED** (held-out replication) |
| Exact Doob verification | **banked**, but development-labelled → **U1** |
| MOLLEO Task 3 | **CLOSED** — four probes, flat fibers |
| **InversionGNN 2-obj (JNK3+GSK3β)** | **LIVE.** Protocol frozen, init bank frozen, 10K supervision built, `F̂_ψ` and `h_φ` trained through Gate-2 numerical correction, lookahead-truth probe banked. **Gate 3 next, then the 5K optimization run (G2/G3).** |
| InversionGNN 4-obj | **gated** on the 2-obj decision rule (G4) |
| Hard support (Block 5) | designed, after controller freeze |
| Pathwise constraints (Block 6) | designed, n=48 frozen (G6) |
| Pareto fan / map reuse (G7) | only once a useful realized map exists |
| Paper `paper_iclr2027/` | compiles clean at **exactly 9 pages**; **32 XXX in main text, 59 total** |

**Critical path, in order:**
1. **InversionGNN Gate 3** — on identical held-out decision fibers, does trained `h_φ` rank decisions
   better than immediate `F̂_ψ`? (If not: fix the value-learning implementation. Nothing else.)
2. **G2 — the matched immediate-vs-future-aware run.** *The paper's mechanism claim depends on this.*
3. **G3 — the 2-objective head-to-head** (both methods, frozen init bank, 5 preferences, 5K budget;
   APS/novelty/diversity/HV for all arms; never pool the published row with the matched rerun).
4. **G1 — the official 800 × 20, once**, with the frozen controller; bank the full k=1…20 curve.
5. **G8 — resolve the VJTNN row** before anything is circulated.
6. Then, in order: G4 (4-obj, gated) → G5 (retargeting at scale) → G6 (pathwise) → G7 (Pareto fan).

---

## 7. THE PAPER

### 7.1 Identity
- **Draft:** `paper_iclr2027/` — `main.tex` + `sections/*.tex`. **The live draft; another agent has
  been writing here.** `paper/`, `paper_arxiv/`, `paper_iclr_control_substrate/`,
  `paper_iclr_stochastic_rewriting/` are **preserved and never edited** — read them as the technical
  sourcebook (definitions, the encoding-invariance proposition and its adversarial checks, the
  finite-horizon theorem and its proof, the notation).
- **Working title:** *COMPOSE: Generator Matching for Trans-Dimensional Molecular Editing and Control*.
  Acronym: **CO**ntrollable **M**olecular **P**rocess **O**ver **S**tochastic **E**dits.
  Alternative under consideration: *Every Step Is a Molecule: Executable Stochastic Control for
  Molecular Design* (memorable; puts novelty before application).
- **Venue:** ICLR 2027 author kit (official style files, sha256 recorded in `paper_iclr2027/README.md`).
  **9 pages main text**, references unlimited, → 10 for rebuttal/camera-ready. The draft is **at** the
  limit: adding a sentence requires removing one.
- Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main`. Verified TeX Live 2024.
  Anonymity: `\iclrfinalcopy` commented out; no author names, no acknowledgements, no repo URLs.

### 7.2 The two conventions the draft is built on — keep them
1. **No number appears that is not in a verified source.** The verified set is exactly seven files,
   and every numeric claim carries a `% source:` comment naming one:
   `docs/SESSION_RUN_MANIFEST_2026-08-18.md` · `docs/VALID128_K8_RESULT.json` ·
   `docs/VALID128_CURVE.json` · `docs/EXTENDED_CURVE_64.json` · `docs/AMENDMENT_VALIDATION_128.md` ·
   `docs/GRIDDD_JIN_PROTOCOL.md` · `docs/AMENDMENT_INVERSIONGNN_2OBJ.md`.
   Audit with `grep -n '% source:' sections/*.tex`.
2. **Unmeasured quantities are `\XXX`, never guessed** — renders red in the PDF. Three marker classes:
   `GAP: NOT YET RUN` (7) · `GAP: NUMBER NOT IN VERIFIED SOURCE SET` (3) · `GAP: MISSING COMPARATOR ROW` (1).
   Audit with `grep -n 'GAP:' sections/*.tex` and `grep -n 'XXX' sections/*.tex`.

### 7.3 Structural doctrine — how to write it (this is the writing brief)

**Write from the conceptual spine, not by editing the old draft line by line.** The old drafts follow
development chronology; this one follows the possible/plausible/purposeful thesis.

**Models to emulate, and what to take from each:**
- **pCoMole — task-first compression.** Its abstract begins with the *engineering setting* (editing
  known biomolecules under multiple objectives and hard constraints), names the missing conjunction of
  capabilities, introduces one mathematical construction, explains one practical approximation, closes
  with application settings. It does **not** begin by teaching the reader discrete flow matching.
  → **Start COMPOSE with iterative lead optimization, not with generator matching.** The reader should
  understand *why an executable trajectory matters* before seeing a Markov kernel.
  pCoMole is also a critical scientific neighbour: it *guides a pretrained variable-length Edit Flow*
  toward preferences with terminal feasibility, approximating its h-function by short rollouts.
  COMPOSE **defines and learns the underlying graph-valued process itself** over complete molecular
  states, canonicalizes many-to-one rewrites, controls a trans-dimensional successor kernel, supports
  dynamic continuation, and can impose rule-closed constraints pathwise. **The paper must never sound
  like "pCoMole, but on small-molecule graphs."**
- **AReUReDi — one nearest ancestor, one missing capability, one algorithm.** It names ReDi as the
  base, states exactly what ReDi lacks, then introduces the mechanisms that close the gap; its
  contribution bullets map 1:1 onto algorithm, theory, experiments.
  → COMPOSE's version: *existing learned generators lack executable molecular states; existing
  graph-edit methods lack a learned reference process and principled future control. COMPOSE occupies
  that missing intersection.* Then name **only** the machinery required: exact executable support,
  learned canonical successor law, finite-horizon control of that law. **Do not introduce seven
  independent "innovations."**
- **PepTune — separate the base generator from the guidance.** Base model → multi-objective guidance →
  task classifiers → experiments that first validate the base, then the guided design. Contribution
  bullets correspond 1:1 to method sections; experiment subsections carry **declarative titles**.
  → COMPOSE's analogue: **Executable reference process → Finite-horizon controller → Molecular-design
  tasks.** Do not interleave executor details, value learning, benchmark protocol, and retargeting in
  one section.
- **MadSBM — contribution formatting.** No isolated "Contributions." heading; the final introduction
  paragraph flows into *"Our main contributions are fourfold:"* followed by the list. Use that.

### 7.4 Abstract — the six moves, in order
1. **Problem.** Molecular lead optimization is iterative, but most learned generators expose only a
   final molecule, while graph-edit optimizers lack a learned distribution over molecular trajectories.
2. **Process.** COMPOSE: a generator-matched stochastic process on an exact trans-dimensional rewrite
   graph of connected, valence-admissible molecules.
3. **Control.** A budget-conditioned h-transform reweights this frozen process toward user-specified
   objective regions **without creating transitions outside its legal support**.
4. **Exact vs scalable.** Exact values → exact terminal reweighting on enumerable state spaces; an
   amortized goal-conditioned value function → closed-loop control at scale.
5. **Principal empirical results.** Only the two or three strongest numbers.
6. **Distinctive consequence.** Because every committed state is a molecule, the same process can be
   retargeted mid-trajectory and constrained throughout the path rather than only at its endpoint.

**No operator list. No carbon-tree prior. No mention of all eight experiments. No claim that valid
paths themselves produce better endpoints.** Target ~180–210 words with exactly two hard numbers.
The empirical sentence should ideally mirror the architecture:
*"Under identical task supervision and molecular support, future-aware control improves over local
property guidance, while closed-loop reuse of reached molecular states further improves Pareto
hypervolume at the same oracle budget."* (`learn the process → reason over its futures → reuse it closed-loop`)

⚠️ **The abstract currently in `sections/abstract.tex` was supplied verbatim by the author and asserts
three things the results section marks `XXX`** — multi-objective competence (G3/G4), the future-aware
improvement (G2), and Pareto-front reuse (G7). *An abstract asserting what the results mark unmeasured
is the one inconsistency a reviewer is guaranteed to find.* It also carries three terminology
deviations flagged in §7.8. **Reconcile before submission.**

### 7.5 Introduction — six paragraphs
1. **Molecular-design reality.** Open with the workflow, not the model: lead optimization begins from a
   useful molecule and proceeds through revisable decisions; candidates are evaluated as they are made;
   priorities change; structural restrictions may need to hold throughout.
   **Preserve this sentence:** *"Terminal sanitization cannot make an earlier oracle query meaningful."*
2. **The missing intersection.** Endpoint generative models learn rich distributions but generally do
   not expose complete molecular intermediates; graph-edit optimizers expose molecules throughout but
   use heuristic proposals and endpoint/local scoring. *What is missing is a learned molecular process
   whose states are all molecules and whose trajectory distribution can be controlled principledly.*
   Do not list ten model families here.
3. **Possible / plausible / purposeful.** `A(x) → R_θ(y|x) → R^h_θ(y|x,z,b)`, in prose before notation.
   **Figure 1 appears here.**
4. **Why future reachability matters.** The one equation. Then: *a move can be useful because of what
   it makes reachable later, even when it does not improve the current property score.* This is the
   endpoint-performance hypothesis.
5. **Evidence map, two tiers.** *Conventional competence:* source-conditioned QED editing; JNK3/GSK3β
   multi-objective. *Process-specific consequences:* exact terminal reweighting where enumerable;
   future-aware vs local decisions; mid-trajectory retargeting; pathwise constraints; closed-loop
   Pareto control. No protocol details here.
6. **Contributions**, flowing out of paragraph 5 as *"Our main contributions are fourfold:"*
   (the draft currently uses four; the earlier brief argued for three — either is defensible, but
   **the benchmark itself is never a contribution**):
   1. a generator-matched stochastic process over legal, trans-dimensional molecular edits between
      complete molecular graphs;
   2. molecular design as **finite-horizon control** of that learned process — a budget-indexed Doob
      h-transform with support preservation and exact terminal reweighting under exact future values;
   3. a **goal- and budget-conditioned future-value controller** steering the same frozen reference
      process across single- and multi-objective tasks without retraining the generator;
   4. evaluation on established editing and multi-objective benchmarks **plus** the process-level
      tests (future-aware edit selection, adaptive Pareto exploration, mid-trajectory retargeting,
      pathwise constraints).

### 7.6 Main-text organization

**§2 An executable molecular reference process** — 2.1 exact transition support (state space, `A(x)`,
executor, closure, trans-dimensionality; **one** proposition; families → appendix); 2.2 learning which
legal edits are plausible (generator matching, arrive quickly at `R_θ` over canonical successors,
source independence); 2.3 canonical molecular successors (one worked example — 2-methylpropane → why
aggregation is necessary; *control should not favour a successor merely because symmetry gives it more
syntactic edit descriptions*). No number appears in this section.

**§3 Finite-horizon control** — 3.1 exact h-transform (backward recursion + controlled kernel;
theorem establishes normalization, exact terminal tilt, support preservation).
**Immediately follow the theorem with its operational consequence:** *exact guidance can redistribute
probability only across molecular edits the executable process already permits.* Never make the reader
infer why a theorem matters. 3.2 goal-conditioned value learning (future-hit / future-value Monte-Carlo
supervision, Bellman consistency, continuous preference conditioning, source-level leakage prevention;
architecture and optimizer settings → appendix). 3.3 **immediate prediction vs future value** — state
the distinction `F̂_ψ(x)` = what is good now vs `h_φ(x,z,b)` = what can lead somewhere good later, in
the *method* section, so the causal ablation is set up before the reader sees the result.

**§4 Controlling molecular-design trajectories** — 4.1 objective regions and preferences (Chebyshev,
fixed training-quantile rule); 4.2 dynamic retargeting (one compact corollary); 4.3 pathwise
constraints (rule-closed via support restriction **vs** cumulative via augmented state — different
mechanisms); 4.4 Pareto exploration (archive construction, optional branching from saved intermediates).
Do not let the Pareto-fan algorithm become load-bearing for the primary benchmark.
⚠️ §4 and §5.4 must follow **Addendum II** — if you are editing from an older document with `B_λ`
regions and a 10% threshold, **you are editing the withdrawn design.**

**§5 Experiments** — the order is **Correctness → Competence → Attribution → Capabilities**, with
declarative subsection titles:
- 5.1 *The implemented process reproduces its own frozen artifacts* — closure, canonical-kernel parity,
  exact-Doob terminal-law TV, support violations, `h_φ` Bellman residual/calibration.
  **Validity is not a competitive result; it is implementation/theory validation.**
- 5.2 *A controller frozen before the panel was opened transfers to prospectively held-out sources* —
  the GrIDDD/Jin QED table + candidate curve + size-fixed ablation.
- 5.3 *Future reachability improves decisions beyond local property prediction* — **the most important
  experiment in the paper.** Hold fixed: task labels, `R_θ`, exact legal fiber, starting molecules,
  preferences, candidate budget. Compare `R_θ + F̂_ψ` against `R_θ + h_φ`. Include the banked
  greedy-vs-lookahead rescue evidence as an independent manifestation of the same phenomenon.
- 5.4 *A frozen molecular process supports distinct Pareto preferences* / *Realized molecular states
  retain value after an objective switch* / *Support restriction enforces molecular constraints
  throughout generation* — multi-objective competence, retargeting, pathwise, closed-loop Pareto.
  For retargeting report **retained prior progress, cost/steps after the switch, and final achievement**
  — not only final performance. Use one real trajectory figure selected by a **deterministic rule**,
  not by visual appeal. For pathwise, the informative quantity is **wasted budget**, not the
  feasibility fraction (which is 1 by construction).

**§6 Related work — three paragraphs, each closing on the COMPOSE distinction, no bold run-in labels:**
1. molecular generation/editing → **GrIDDD** (insertion/deletion in discrete graph diffusion;
   size-adaptive; strong molecular-optimization performance) → *COMPOSE does not claim molecular
   editing, variable-size generation, or validity preservation are new; its distinction is defining an
   explicit state-dependent rewrite space and learning a goal-independent law over it, so the
   transition process itself becomes the controlled object.*
2. generator matching / Edit Flows → **MadSBM** (closest mathematical relative: controlled CTMC on an
   amino-acid edit graph — but its learned control *constructs* the transport rather than adapting an
   already-learned process) → **Value Matching** (lightweight value steering a frozen flow model
   without fine-tuning) and **pCoMole** (Pareto-constrained molecule editing with discrete flows) →
   *COMPOSE separates the stages.*
3. multi-objective molecular design → **InversionGNN** (multi-property predictor + gradient-based
   Pareto search; our numerical comparator), **PepTune** (masked discrete diffusion + tree search),
   **AReUReDi** (Tchebycheff scalarization + annealed Metropolis-Hastings), **Routing by Reaching**
   (composing pretrained GFlowNets at inference time) → *COMPOSE places multi-objective control on a
   reusable transition process, so future reachability and already-reached states participate in
   closed-loop Pareto exploration.*
   **No fourth "stochastic control" paragraph** — Doob / Feynman-Kac / Schrödinger citations live in
   the preliminaries and §5. Graph-GRPO is held for the extended related work unless we compare directly.
   **MOLLEO is out of the central narrative.**

**§7 Discussion / limitations / conclusion.**

### 7.7 Figures
- **Fig 1 — possible / plausible / purposeful.** Three panels: exact legal successors from a molecule;
  `R_θ` assigning different probability to them; `h_φ` changing the edge probabilities toward a target
  region. Small inset: a goal switch at an intermediate molecule. **No operator families, no benchmark numbers.**
- **Fig 2 — what is trained.** Row 1: goal-free molecular transitions → `R_θ`. Row 2: `R_θ` trajectories
  + task labels → `h_φ(x,z,b)`. Include the contrast `F̂_ψ(x)` vs `h_φ(x,z,b)`. *This figure may do
  more for reviewer understanding than another molecular gallery.*
- **Fig 3 — standard-task competence.** GrIDDD success curve + final comparison; JNK/GSK front or
  APS/HV; size-fixed ablation as a small panel.
- **Fig 4 — why the process view matters.** Future-aware vs local; dynamic retargeting; pathwise
  constraints; (qualified) Pareto-fan trajectory.
- *Current state:* Figs 2 and 4 are **in the appendix** — Fig 2 is a schematic and all four of Fig 4's
  panels are blocked on unrun experiments; at 9 pages they were the least defensible use of main-text
  space. Each moves back by relocating one `\input` line.

### 7.8 Terminology contract — non-negotiable

| use | avoid |
|---|---|
| executable molecular process / **stochastic molecular edit process** / **reference edit law** | "molecular dynamics model" (invites a physical-kinetics reading) |
| **connected and valence-admissible within the declared chemical state space** | "chemically valid" unqualified |
| declared molecular state space | "all of chemical space" |
| learned future value; **amortized / approximate** `h_φ` | "exact controller" when `h_φ` is learned |
| **exact h-transform** (reserved: exact executor support, exact canonical normalization, exact DP, exact terminal tilt under exact h) | "exact learned guidance" |
| source-conditioned optimization | "conditional generation" without defining the conditioning |
| canonical molecular successor | "action alias" without explanation |
| task-trained controller | "task-agnostic system" when `h_φ` used task labels |
| executable edit | "synthesis step" |
| computational property oracle | "biological truth" |
| **one reusable executable molecular process** | "one reusable molecular world" (great for a talk, unsafe as a formal claim) |

### 7.9 Claim–evidence discipline

| claim | required evidence | forbidden interpretation |
|---|---|---|
| every committed state lies in the declared molecular space | closure proof + trajectory audit | experimentally synthesizable |
| `R_θ` learns useful molecular transport | support-matched likelihood / trajectory baselines | physical molecular kinetics |
| future-aware control improves decisions | **same-label immediate predictor vs `h_φ`** | valid intermediates alone cause the improvement |
| trans-dimensionality matters | full vs **identical** size-fixed support | insertion/deletion is always necessary |
| one process supports multiple tasks | `R_θ` fixed across QED and MOO | no task-specific controller training |
| retargeting preserves useful progress | continuation vs restart | globally optimal replanning |
| path constraints are exact | rule-closed support restriction | arbitrary history-dependent constraints without state augmentation |
| COMPOSE is benchmark-competitive | exact inherited protocols | matched compute / sample efficiency unless explicitly matched |

Every empirical claim names its **estimand**, the **matched comparison**, the **independent statistical
unit**, and its **limitation**. Comparator rows are labelled **reported context**, visually and
textually separated, and are **never** described as the thing a COMPOSE number beats.
**No matched-compute claim is made anywhere:** the QED result is *improved task performance at a
measured inference-time cost* — never greater efficiency, and the absence of objective-specific
retraining is never offered as compensation.

### 7.10 Sentence-level style
One claim per paragraph. First sentence says what the paragraph establishes; last says why it matters.
Define the object in prose before the equation. After every theorem, an operational consequence. After
every experiment, the scientific interpretation **and** the limitation. Use *we define / we derive / we
evaluate*, never *we leverage*. Adjectives only when followed by evidence. Avoid *novel, powerful,
robust, general* unless precisely bounded. **Never use internal experiment names** (P0c, K41, 4A, L4,
policy versions). **Never narrate the chronology of failed branches in the main paper.** Do not write
"we are explicit about status" — that belongs to development drafts. Do not mention a result twice
unless the second occurrence changes the interpretation.

### 7.11 The gap register — every `XXX` and the run that closes it

**`GAP: NOT YET RUN`**
| id | where | unmeasured | closing run |
|---|---|---|---|
| **G1** | Table 1, last COMPOSE row | official 800-source success at k=20 | run all 800 × 20 **once** with the frozen controller; bank per-source outcomes + curve at every k |
| **G2** | §5.3 | matched paired difference between immediate and future-aware arms; n sources; both inference ledgers | with `F̂_ψ` fixed and `R_θ` frozen, run both arms from the frozen init bank, same 5 preferences/seeds/budget; bootstrap **over sources**; never pool the two ledgers. **The paper's mechanism claim depends on this.** |
| **G3** | §5.4 | APS, novelty, top-K diversity, HV for future-aware, immediate, and the matched comparator rerun (2 obj) | the four-step procedure in Addendum II; never pool the published row with the matched rerun |
| **G4** | §5.4 | same four metrics, 4 objectives at 20K+5K | **gated** on the 2-obj decision rule |
| **G5** | §5.4 | post-switch cost vs restart, retention, n paired sources | four arms on identical sources/seeds: continue under the corollary; restart from source lead; restart from switch molecule with fresh sampler; static compromise goal |
| **G6** | §5.4 | complete-feasibility fraction, wasted oracle calls, final quality | rule-closed predicate: support restriction vs endpoint rejection vs terminal penalty. Feasibility = 1 by construction (an implementation check); **the informative quantity is wasted budget** |
| **G7** | §5.4 | HV at matched total oracle budget vs independent restarts; nondominated molecules per oracle call | fan one saved prefix into several preference-conditioned continuations; **show the accounting** — shared-prefix cost counted once for the fan, once per branch for restarts |
| **G8** | Table 1 | the success rate of a further published, higher-scoring comparator (**VJTNN, 60.6%**) | verify value + protocol against the published record; confirm cohort and similarity threshold match; promote into the verified source set; **add the row**. If the protocol does not match, say so **in the caption** rather than omitting it silently |

**`GAP: NUMBER NOT IN VERIFIED SOURCE SET`**
| id | where | unquoted | how to close |
|---|---|---|---|
| **U1** | §5.1 | exact-control TV residual, support violations, backward residual, and the same three after retargeting | the artifact **exists** (`diagnostics/exactness/…exact_control.json`) but is outside the verified set and self-labels development-only on a carbon slice → promote with provenance, **or** rebuild on the current registry over the preregistered goal/budget grid |
| **U2** | appendix | heavy-atom cap, count of neutral (element, valence) classes, element list, fraction of benchmark leads admitted | transcribe from the scope-fingerprint artifact with its hash (declared-scope constants, not measurements) |
| **U3** | appendix | controller input dim, embedding width, hidden widths, dropout, Bellman weight, epochs, patience, validation fraction, n jointly trained goal regions | transcribe in one pass from `docs/HPHI_QED_PREREGISTRATION.md` §6, citing the document + hash (design constants fixed **before** the data) |

### 7.12 Two red flags in the current draft
1. **The comparator table omits a stronger published row (G8).** `docs/GRIDDD_JIN_PROTOCOL.md`'s
   published table lists only JT-VAE / CG-VAE / GCPN / GrIDDD. **VJTNN's 60.6%** survives in exactly
   two places: `docs/PAPER_REFRAME_CONTROL_SUBSTRATE.md:445` (a SUPERSEDED doc) and the *committed*
   revision of `docs/VALID128_K8_RESULT.json`, whose comparator block was **deleted** in one working-tree
   revision and has since been restored (commit `7cfadb8`). Mitigations already applied: the abstract no
   longer says "the strongest reported comparator" (it says "the comparator whose protocol we follow");
   Table 1 carries an explicit `\XXX` row; the caption says the omission out loud.
   **Do not submit until G8 is resolved.**
2. **The headline artifact was uncommitted and its two revisions disagreed.** Committed: `n_sources 126`,
   `cumulative[7] 62`, `runs 1008`. Working tree: `n 128`, `cumulative[7] 63`, `runs 1024`. The
   percentage is unaffected (49.206% vs 49.219%, both 49.2%) but **Table 1's counts and the §5.2
   inference ledger come from the working-tree revision.** Confirm it is committed before circulating.

### 7.13 Bibliography gaps
`references.bib` (24 entries) contains only entries transcribed from existing bibliographies — nothing
invented. Two comparators are named without citation:
- **InversionGNN** — the repo records only `arXiv:2503.01488`, `github.com/ivanniu/InversionGNN`,
  commit `cfdf1d9`. **Author list, title and venue must be verified against the published record**
  before adding a `\citep`.
- **CG-VAE and GCPN** — cited *through* GrIDDD's published table; add individual citations if wanted.

### 7.14 Page budget — order of least damage if space is needed
1. Move `\input{sections/fig3}` to the appendix (≈0.2 page) — loses a real figure, prefer 2–4 first.
2. Trim Table 1's caption (≈0.05 page).
3. Move §5.1's parity sentence into the appendix audit, leaving the U1 gap paragraph (≈0.15 page).
4. Move §6 (Related work) wholly to the appendix, leaving a one-sentence pointer (≈0.25 page).
When G1–G7 land and space is needed for real results, the appendix already holds what came out
(`app:figplan`, `app:capdesign`, `app:related`, `app:multiobjtable`, `app:fig2`, `app:fig4`).

### 7.15 Reviewer stress tests and the prepared answers
| attack | answer |
|---|---|
| "Generator matching with handcrafted edits." | It is a new Markov-process design; the evidence is competitive generation, learned-vs-uniform transport, trans-dimensional and topological ablations, and state-level control. |
| "Insertion/deletion already exists." | Credited explicitly (jump diffusion, **GrIDDD**, **Edit Flows**). The claim is the richer typed rewrite language + molecular closure + topology change + generator-matched event law + quotient kernel + dynamic control. |
| "Validity by construction is tautological." | **Agreed** — it is a support property. The content is what it buys: pathwise constraints, mid-trajectory intervention, branching, zero wasted invalid oracle calls. State it once, then *use* it. |
| "Doob is classical." | The contribution is the learned chemistry-native kernel on which exact, retargetable, support-preserving control becomes operationally meaningful. |
| "Too many ideas." | One causal chain; each experiment tests one link. |
| **"Trans-dimensionality is just padded-slot bookkeeping."** | The slot array is a coordinate system; the semantic state is the active graph, and births create typed entities participating in valence, connectivity, ring perception and canonicalization. **The measurement is the full-vs-size-fixed ablation, and it does not exist yet — this is the weakest point in the draft.** |

### 7.16 The framing paragraph to build the submission around
> COMPOSE treats molecular design as control of a learned executable process. An exact graph-rewrite
> system defines which molecular transitions are **possible**; a source-agnostic reference model `R_θ`
> learns which legal transitions are **plausible**; and a budget-conditioned controller `h_φ` favors the
> futures that satisfy the current goal — **purpose**. Every committed state is a connected,
> valence-admissible molecule within the declared chemical space, so properties and constraints remain
> meaningful throughout generation.
>
> We first ask whether this formulation is useful on standard molecular-design tasks, under the
> established GrIDDD source-conditioned QED protocol and the JNK3/GSK3β multi-objective benchmark,
> using the same candidate or task-information budgets as the comparison methods. These are
> **competence checks**: they establish that a controlled executable process can produce strong
> optimized molecules and Pareto fronts.
>
> We then isolate what becomes possible *because* generation is a controlled molecular trajectory.
> Future reachability can favor an edit that is locally unimpressive but opens a better downstream
> route; a realized intermediate can be recontrolled when the objective changes; and rule-closed
> constraints can be enforced over every committed molecule rather than filtered at the endpoint. The
> central claim is therefore not that COMPOSE is another molecular optimizer: it learns one reusable
> executable molecular process and changes the **controller** — not the molecular prior — when the
> purpose changes.

**The one thought the paper should leave:** *Molecular generation need not be repeated endpoint
sampling; it can be control of one learned executable molecular process.*

---

## 8. Things that are CLOSED — do not reopen

These do **not** become future experiment ideas merely because they appear in the archive:
the **DDSBM manuscript branch** · **Policy B** as the final QED controller · the **exact-target
branch** · the **retargeting branch** (banked and complete) · **P3/P4** · **P0c** · **K41** (failed
breadth criterion; it changed the search itself and was therefore never a retention approximation to an
upper bound) · **MOLLEO Task 3** and all four of its search branches · arbitrary new scalarizations ·
extra baseline ports · **any post-outcome controller rescue knob** (`QED^α`, temperatures, top-k,
shortlist sizes, cap tuning).

A negative result closes **the branch named by its preregistration**. It does not retroactively erase
banked results, and it does not authorize an unrelated rescue.

**Also barred:** reintroducing the TDC/sklearn image; `neutralize_catalog_drift()` in claim-bearing
code; quoting BARRED artifacts (early data-starved RingCore checkpoints; pre-Active8 preflight
reachability/round-trip/path-cost numbers; pre-RingCore exact-control artifacts; `conditional_smc`
physchem box — two mutually inconsistent versions on disk; the matched-budget frontier-recovery
reading, retracted as tuning-dependent). **Retracted numbers that must never reappear:** the "6.3M"
parameter count, the "≈560 oracle calls" figure, and the developmental compiler/training figures
(99,738 / 49,869 of 50,000 / 84.09→28.14 / 90.2%).

---

## 9. One-paragraph orientation, if you read nothing else

COMPOSE learns a single frozen, goal-independent stochastic process `R_θ` over **executable molecular
edits** — every state a connected, valence-admissible molecule, every transition enumerated by an exact
executor — and then supplies *purpose* at inference through a budget-indexed Doob h-transform
`P_φ(y|x,z,b) ∝ R_θ(y|x)·h_φ(y,z,b−1)` that can only reweight edits the executor already permits.
Exactness is verified to machine precision on a 966-state enumerable slice, including exact
recontrol after a mid-trajectory goal switch. On the standard GrIDDD/Jin QED editing benchmark a
controller frozen before the panel was opened reaches **49.2% at 8 candidates and 54.7% at 12** on a
prospectively held-out 128-source panel, against GrIDDD's reported **45.1% at 20** (and VJTNN's
**60.6%**, which is higher than both — so this beats the closest per-step editor but is **not** SOTA);
the official 800-source run has not yet been executed. Internally, verified future-aware control beats
greedy by **+21.5 pp**, rescuing **35.9%** of greedy's failures, and a realized molecule retains
positive value after an objective switch under held-out replication. The live front is a matched
2-objective head-to-head against InversionGNN in which both methods receive the same 10K oracle labels
and diverge only in what they build from them — a property gradient versus future-reachability control
— which is the experiment the paper's mechanism claim depends on.
