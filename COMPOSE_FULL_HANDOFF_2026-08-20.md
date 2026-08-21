# COMPOSE — complete project handoff, v2
**Written 2026-08-20. Supersedes `COMPOSE_FULL_HANDOFF_2026-08-19.md` entirely.**
Paste this whole file as the first message of a new chat. Where it conflicts with any earlier
summary or handoff, **this wins**.

---

## 0. How to read this, and what changed in the last 24 hours

You are picking up a research project ~2 weeks from an ICLR 2027 submission. Three standing rules:

1. **Never invent a number.** Unmeasured quantities are `\XXX` in the paper and stay that way.
2. **A closed branch stays closed** (§9). Do not re-propose it as a new idea.
3. **Say which of measured / inferred / assumed a statement is.** Most bad calls in this project's
   history came from stating an inference in the voice of a measurement.

Repo: `/Users/rmaganti/compose_v2_work`. Paper: `paper_iclr2027/`. Bulk artifacts on Modal volume
`compose-v4-artifacts`.

### ⚠️ THE FIVE THINGS THAT CHANGED SINCE THE LAST HANDOFF — read before anything else

1. **GENERATOR MATCHING IS NO LONGER THE FRAMING.** This is the single biggest correction. `R_θ` is
   **not** trained by the generator-matching population objective. It is trained by the
   **productive-successor likelihood**, and the exit rate is **never optimized**. GM was deleted from
   the Preliminaries on 2026-08-20 and demoted to a *stated relationship* in §4.2 + `app:gm`. The
   paper **title changed accordingly** (§7.1). Any sentence anywhere that says "COMPOSE is trained by
   generator matching" is now **wrong**. Details in §2.3 and §7.4.
2. **A new headline experiment exists: §5.1, the reference-law experiment.** `R_θ` cuts
   canonical-successor NLL from **5.37 → 3.90 nats** against an empirical-family baseline on 3,545
   held-out transitions, and **86.7%** of that gain is within-state successor choice rather than
   global edit-family frequency. This is the first experiment in the paper and it is fully banked.
3. **The QED table now carries a diversity column**, and COMPOSE wins it against GrIDDD:
   **0.393 vs 0.283** (but **HierG2G reports 0.477**, higher than ours — see §5.4).
4. **The main text is now ~14 pages against a hard 9-page limit.** This is the most urgent
   mechanical problem in the paper. Discussion currently begins on p.14.
5. **The InversionGNN `h_φ` Gate-2 diagnosis came back weak.** The learned value model
   (Spearman 0.484) is **worse than a trivial persistence baseline** (0.845); only an exploratory
   log-target arm (0.862) edges past it. §5.6 — this is the live blocker.

---

## 1. What COMPOSE is — the thesis (unchanged, and still correct)

> **COMPOSE learns one stochastic process over executable molecular edits and controls that frozen
> process toward changing molecular-design goals.**

**Possible → Plausible → Purposeful**

| layer | object | what it decides | varies by task? |
|---|---|---|---|
| **Possible** | exact executor + legal edit fiber `A(x)` | which molecular transitions are executable at all | **never** |
| **Plausible** | learned reference law `R_θ(y\|x)` over canonical successors | how probability should distribute over those transitions | **never** — one frozen model, all objectives |
| **Purposeful** | finite-horizon controller `h_φ(x,z,b)` | which plausible futures serve the current goal | **yes, by design** |

The sentence for the introduction:
> COMPOSE separates what molecular transitions are executable, which executable transitions are
> plausible, and which plausible futures satisfy the current objective.

The one equation to remember (`eq:central`, §4.5):
```
P_φ(y | x, z, b)  =  R_θ(y|x) · h_φ(y, z, b−1)  /  Σ_{y'∈S(x)} R_θ(y'|x) · h_φ(y', z, b−1)
```

**Keep verbatim:** *COMPOSE separates molecular plausibility from molecular purpose.*
`R_θ` = plausibility, `h_φ` = purpose.

**What it buys that endpoint generation does not:** every committed state is a complete molecule, so
(a) an oracle query at any step is meaningful, (b) a realized intermediate can be **retargeted** when
the objective changes, (c) constraints can be enforced **pathwise** rather than filtered at the
endpoint, (d) control can reason about **future reachability** rather than local property gain, and
(e) realized search effort becomes reusable.

**The endpoint-performance mechanism is future reachability, NOT intermediate validity.** Valid paths
are a structural property; they do not by themselves cause higher success. Never write that they do.
The Discussion says this explicitly: *"Variable-size editing, stochastic molecular processes, and
h-transforms are not individually new; the distinction is that COMPOSE brings them together in a
common molecular transition process whose executable support and learned plausibility remain separate
from task-specific purpose."*

---

## 2. The formalism, exactly as the current draft states it

### 2.1 State space and exact support (§4.1)
- A **state** is a connected molecular graph over a declared vocabulary of elements, bond orders,
  formal charges and implicit hydrogens in which every atom satisfies its valence constraint;
  `X` = such graphs with ≤ `n_max` heavy atoms = the **declared molecular state space**.
- An **edit mark** `a` is a typed rewrite rule with a match and payload, applied by a deterministic
  executor `T` under **propose–validate–commit**.
- **Legal edit fiber** `A(x) = {a : a matches x and T(x,a) passes every guard}` — finite,
  state-dependent, enumerable. *Illegal chemistry is absent from the model rather than discouraged
  in its loss.*
- **Trans-dimensional:** `X = ⊔_n X_n`; every mark is a birth, a death, or same-cardinality. Births
  and deaths create/destroy typed graph entities participating in valence, connectivity and ring
  perception — **not** padding flags.
- **Proposition (Closure of executable support):** if each admitted edit preserves declared valence
  at every atom it modifies, leaves untouched atoms unchanged, respects `n_max`, and commits only
  when the successor is connected, then `T(x,a) ∈ X`. Hence every state reached by any process
  supported on `{A(x)}` remains in `X`.

### 2.2 The learned reference law (§4.2) — READ THIS, IT CHANGED
- The model assigns a normalized law over **legal edit marks**:
  `p_θ(a|x,t)`, `Σ_{a∈A(x)} p_θ(a|x,t) = 1`.
- **Hierarchical:** `p_θ(a|x,t) = p_θ(k|x,t) · p_θ(o,η|k,x,t)` where `k` = edit family, `o` = matched
  molecular operands, `η` = typed payload. A shared molecular encoder feeds family-specific scorers;
  normalization is restricted to legal candidates.
- **Source-agnostic:** the reference model receives neither the source lead nor the objective,
  remaining budget, or task constraints. Those enter only through initialization, support
  restrictions, and the controller. *This transition law represents molecular plausibility
  independently of molecular purpose.*
- **Supervision:** molecular examples are compiled into executable edit programs; every stored
  teacher transition is **replayed through the same executor used at inference**. The program may
  depend on both endpoints, but neither endpoint nor program reaches the model at inference.
- **⭐ THE TRAINING OBJECTIVE — the productive-successor likelihood:**
  ```
  L_ref(θ) = − [ Σ_i w_i log R_{θ,t_i}(y*_i | x_i) ] / [ Σ_i w_i ]
  ```
  over productive teacher transitions `x_i → y*_i`, with `w_i` the registered training weight.
  **Terminal rows do not enter.** Training learns which committed molecular successor follows a state
  under the declared editing measure; **it does not calibrate continuous-time holding rates.**

### 2.3 The exact relationship to generator matching (§4.2 + `app:gm`) — the answer to "is this still GM?"
**No — GM is a connection, not the training framework.** The appendix now derives it precisely.
Write the productive rate as `Q⁺(y|x,t) = Λ(x,t)·P(y|x,t)`. For the Bregman divergence generated by
`φ(u) = u log u − u`:
```
D_φ(Q⁺*, Q⁺_θ) =  Λ*·KL(P* ‖ R_θ)          ← conditional-successor block
               + (Λ* log(Λ*/Λ_θ) − Λ* + Λ_θ) ← productive-hazard block
```
The blocks are **variationally separable**. COMPOSE optimizes an empirical version of the
**conditional-successor block**, and differs from the GM population objective in **exactly two**
respects:
1. **The hazard block is never optimized.** The compiled supervision carries no rate magnitudes —
   every stored transition is registered with **unit teacher rate**, entering training only as a
   nonterminality indicator. Under that convention `Λ*` is constant and its weight is absorbed into a
   θ-independent positive constant. Fixed-budget editing consumes only the embedded successor kernel,
   on which constant rescaling of `Q⁺` has no effect.
2. **The expectation is under a different measure.** GM integrates the per-state divergence against
   the state–time marginal of the conditional-process mixture. COMPOSE draws rows under a **declared
   editing measure deliberately weighted across operator families**, so a family's exposure reflects
   the reference law we mean to estimate rather than the incidental corpus composition; **some strata
   are assigned zero draw**. No importance correction is applied, and on zero-draw strata none could be.

**Therefore:** the decomposition holds pointwise in `(x,t)` and `L_ref` *is* the conditional-successor
block of it, **but the training functional is not the GM population objective.** Describe it as
*"a decomposition that identifies which factor `R_θ` estimates,"* never as an objective identity.
The Preliminaries comment forbids reinstating GM as a preliminary: *"Giving it a subsection told the
reader it is the training framework, which the shipped objective is not. Do not reinstate it here."*

### 2.4 Canonical molecular successors (§4.3)
- Successor fiber `G_y(x) = {a ∈ A(x) : T(x,a) ≃ y}`; pushforward
  `p̄_{θ,t}(y|x) = Σ_{a∈G_y(x)} p_θ(a|x,t)`.
- Worked example: deleting any of three symmetry-equivalent terminal methyls in 2-methylpropane
  `CC(C)C` gives propane. Treating those as distinct choices would favour propane merely because it
  has several syntactic realizations; **COMPOSE exposes propane once with their combined mass.**
- Conditioning on committing an edit gives the molecular reference kernel over distinct successors:
  ```
  R_{θ,t}(y|x) = 1{y ∈ S(x)} · p̄_{θ,t}(y|x) / Σ_{y'∈S(x)} p̄_{θ,t}(y'|x)
  ```
  `S(x) = ∅` ⇒ terminal. Iterating defines the molecular reference process.
- **Proposition (Canonical normalization and refinement invariance):** `Σ_y R_θ(y|x) = 1`, and
  splitting/merging marks within a fiber leaves `R_θ` unchanged when aggregate mass is preserved.
  Hence **any controller defined on canonical successors is invariant to refinements of the internal
  edit representation** — and a procedure that truncates or reweights *individual marks* before
  aggregation need not be. COMPOSE performs control **after** canonicalization.

### 2.5 Finite-horizon control (§4.4–§4.8)
- **§4.4 Doob:** backward value `h_b`, controlled kernel `R^{z,b}_θ`. **Theorem:** wherever backward
  values are positive the controlled kernels are normalized, **introduce no transitions outside the
  support of `R_θ`**, and after `K` controlled edits realize the terminal tilt. Proof by telescoping
  path-measure argument (`app:proofs`). **Early stopping does not inherit the tilt** — the marginal at
  `j < K` is an `h_{K−j}`-tilt, which is also why continuation ≠ specifying `z'` from the first edit.
- **§4.5 `h_φ`:** `h_φ(x,z,b) ≈ h_b(x,z) = E_{R_θ}[g_z(X_b) | X_0 = x]`. Monte-Carlo target
  `ĥ_M(x,z,b) = (1/M) Σ_m g_z(X^{(m)}_b)`. Accommodates both event-valued objectives (indicator ⇒
  finite-budget reachability) and soft desirability (⇒ expected future value). **Train/val partitioned
  at the source level.** Bellman consistency `h_b(x,z) = Σ_y R_θ(y|x) h_{b−1}(y,z)` is used as an
  auxiliary training signal via adjacent-budget pairs. Approximation error can misallocate probability
  among legal successors **but cannot introduce a transition outside `R_θ`'s support**; exact terminal
  reweighting remains a property of the exact transform and is **not claimed for learned `h_φ`**.
- **`F̂_ψ` vs `h_φ`:** `F̂_ψ(x)` scores the current molecule; `h_φ(x,z,b)` scores its reachable
  futures. **Property learning estimates where value may lie; future-value learning propagates that
  value through the molecular process.**
- **§4.6 Sampling:** direct normalized evaluation; an **equivalent rejection construction** when
  `h_φ` admits a finite envelope (for reachability-valued controllers `h_φ ∈ [0,1]` the value *is* the
  acceptance probability); and **twisted SMC** when desired futures carry little reference mass —
  particles propagate under frozen `R_θ` with `h_φ` entering **only through Feynman–Kac weights**.
  *SMC changes the numerical realization of control, leaving the reference process and value function
  unchanged.* **Prospective STOP:** `τ = min{t : X_t ∈ B_z}`; intermediate molecules are **not**
  retrospectively collected — **one controlled trajectory produces one returned candidate**.
- **§4.7 Preferences:** weighted Chebyshev shortfall `L_λ(x) = max_i λ_i(1 − F̂_{ψ,i}(x))` (the max
  penalizes the most limiting objective, so one property cannot arbitrarily compensate another);
  terminal desirability `g_λ(x) = exp(−L_λ(x)/τ_λ)` with `τ_λ = Std_{x∼D_train}[L_λ(x)]` fixed from
  the property-supervision set **before** optimization. Implementation uses the centered form
  `g̃_λ = exp(−(L_λ − L*_λ)/τ_λ)`, which multiplies all desirabilities by the same positive constant
  and leaves every normalized controlled transition unchanged. **We make no separate Pareto-optimality
  guarantee from the scalarization alone.**
- **§4.8 Retargeting + pathwise:** `P_φ(y|x_τ,z',b) ∝ R_θ(y|x_τ)·h_φ(y,z',b−1)`; **Corollary (exact
  continuation after a goal change)**. Pathwise: `A_C(x) = {a ∈ A(x) : C(T(x,a)) = 1}` and rebuild
  `R^C_θ` by the same canonicalization — *an approximate value model may allocate probability poorly
  among feasible transitions, but cannot violate a constraint enforced through the support itself.*
  Rule-closed predicates (protected scaffold, forbidden substructure, size/charge window, similarity
  floor) work directly; **cumulative** requirements need state augmentation
  `X̃_k = (X_k, ξ_k)`, `ξ_{k+1} = Φ(ξ_k, X_{k+1})` — a **different mechanism, reported separately.**

**Reserve "exact"** for: exact executor support, exact canonical normalization, exact dynamic
programming, exact terminal tilt under exact `h`. Learned `h_φ` is **amortized / approximate**.

---

## 3. The code

### 3.1 Repository map
```
src/compose_v4/          the model (285 .py). A process-identity SHA over rewrite/ and model/
                         gates every banked artifact — do not move files here.
  rewrite/               executor, operators, fibers, action codecs, tracelets, ring semantics,
                         compiler, canonical successors
  model/                 rate models (factorized_tracelet, relational_reroute,
                         contextual_ring_restate, segmented_successor, time_convention)
  gm/ policy/ chem/ eval/ data/ oracles/ benchmark/ baselines/ experiments/ lipids/
modal_apps/              144 distributed experiment apps: hphi_* (QED/controller lane),
                         molleo_* (closed), invgnn_* (live multi-objective lane),
                         experiment1_reference_law_app.py, h_phi_pair_trace_app.py
scripts/                 242 drivers/readers/probes; scripts/ops/ for launches
docs/                    300+ files — see docs/INDEX.md for CURRENT/SUPERSEDED status
diagnostics/             committed result JSONs from the editing-V2 / coherence lines
configs/ artifacts/ tests/ (494 tests) third_party/ (InversionGNN)
paper_iclr2027/          THE LIVE DRAFT
paper/, paper_arxiv/, paper_iclr_control_substrate/, paper_iclr_stochastic_rewriting/
                         preserved earlier drafts — read as sourcebook, never edited
```

### 3.2 The eight operator families
`cycle_insert`, `atom_insert`, `atom_restate`, `bond_reroute`, `atom_delete`, `bond_reorder`,
`ring_system_restate`, `cycle_attach`.

Scope facts that must not be overstated:
- Ring **generation** is compositional and catalog-independent (`cycle_close`/`cycle_open`).
- The legacy whole-ring **growth macro is disabled** (masked to empty support, log-prob −∞); a
  catalog-bounded whole-ring **deletion** exists but is **disabled**.
- **Ring-editing evidence comes ONLY from the synthetic corruption lane.** `cycle_insert` (122,183
  transitions), `cycle_attach` (84,940), `ring_system_restate` (16,380) are **entirely** in
  `reversible_synthetic_walk`; zero in any data-backed lane. The four data-backed lanes are
  `linker_positional_topology_analogue`, `observed_local_analogue`, `operator_aware_real_endpoint`,
  `real_endpoint_multistep_path`. Scope the claim to *"recovers legal ring edits under synthetic
  perturbation"* — never "data-backed ring editing" or "scaffold hopping."
  **This is now handled in the paper by reporting the data-backed-only subset separately (§5.1).**
- `atom_restate:valence_state_change` is 220 of 1,803,032 admitted train transitions (0.012%) and is
  **reachability-redundant**. Retained as legal support; **not** claimed as a learned capability.

### 3.3 Frozen assets — the identity chain
| asset | where | hash / note |
|---|---|---|
| `R_θ` checkpoint (**never retrained**) | volume `runs/run_v2_01/` | `sha256 c979cdb3…4e53de8`; **selected step 12,500** (see note) |
| frozen law | — | `sha256 b0cc66f168f1cf3e7a4b5bacfb1138059005251a10561d6961c3ea1c5dd7ac73` |
| H24 `h_φ` head | volume `hphi_v2/head.pt` | `sha256 9ea51ec4…` — verified byte-identical across a session that trained a different head |
| H24 norm | volume `hphi_v2/norm.json` | `sha256 4f93f7ec…` |
| reserve (held-out) ids | `diagnostics/editing_v2_matched_validation_reserve_ids.json.gz` | `sha256 ba9270fa…` |
| split precedence | `diagnostics/editing_v2_split_precedence_resolution.json` | `sha256 f96ea79d…` |
| Task-3 oracle bundle | **in repo** `artifacts/oracles/molleo_task3_v1/` | SHA-pinned, no TDC/sklearn at runtime |
| DRD2 oracle | **in repo** `artifacts/oracles/drd2_svm_v1/` | see §5.9 |
| InversionGNN init bank (100 mols) | `docs/INVERSIONGNN_FROZEN_PROTOCOL.json` | bank `sha256 5fdb5f5e…`, ZINC pool `35e3f1a5…` |

**Checkpoint-step note, now resolved in favour of 12,500.** `docs/DECISION_LOG.md` names step 8,500
as the selected checkpoint from the training curve; both the exact-control anchor **and** the
paper-bearing reference-law artifact record `run_v2_01 selected_step 12500`. The two *paper-bearing*
artifacts agree, so **12,500 is what the shipped results were produced with**. The DECISION_LOG line
describes the training-selection gate at an earlier point. Say "step 12,500" in the paper and fix the
DECISION_LOG wording rather than the other way round.
`run_v2_01` chain: reserve `b580fdef6486` (15,031 reserve / 136,028 training / 151,059 total) · law
`b0cc66f168f1` · manifest `e27494a46250` · store `b232a6fa069f` · gate FROZEN 9/9, primary-metric
SE 0.0237. Training cost ≈ **$2.6** on preemptible A10G.

### 3.4 Environment, pinned
**python 3.11, torch 2.4.0, numpy 1.26.4, scipy 1.13.1, networkx 3.3, rdkit 2024.3.5.**
**Fail closed on catalog drift:** any claim-bearing evaluation or corpus compilation must abort if the
reconstructed RingCore catalog fingerprint differs from `639ff6078c32d43c`.
`neutralize_catalog_drift()` is a **dev-probe helper only** — never in code producing a cited number.
Not hypothetical: rdkit 2026.03.4 drifted to `82fd910c`, and a 50-state parity gate found 2 states
whose canonical successor inventory differed (aromaticity perception → same molecule, different
canonical key). Nothing crashes; you get a split-brain corpus.

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

### 3.6 Modal runbook — failure modes already paid for
```bash
# 1. Price it first (docs/MODAL_COST_MODEL.md; CPU $0.04716/core-hour). Budget = authorization ceiling.
# 2. Launch DETACHED from a COMMITTED tree, never a scratchpad worktree:
scripts/ops/launch_detached.sh <worktree> modal_apps/hphi_h40head_ab_app.py run.log --valid --k 8
# 3. Poll the VOLUME, never the log (a client log stops the moment the client dies):
scripts/ops/poll_volume.sh compose-v4-artifacts editing_v2/r_theta_run <shard-name>
# 4. Read results into a committed docs/*.json via a COMMITTED script.
```
- `--detach` alone is **not enough**; never wrap a detached launch in a client-side `timeout` — it
  cancels the job. Entrypoints should `drive.spawn()`, not `.remote()`.
- **Launch from the committed tree.** A pilot that passed from `/private/tmp` preceded a full run that
  mounted a branch missing the module → **0/5,984**.
- **Persist expensive deterministic artifacts as their own job** (encode → persist → exit).
- `compose_v4/oracles/__init__.py` **eagerly imports joblib**; the DRD2 oracle therefore lives at
  `compose_v4.drd2_oracle`.
- **Cost estimates must be allocation × wall-time.** Modal bills memory-time. A probe quoted at ~$0.50
  actually cost ~$7.

### 3.7 Documents whose names claim more authority than they have
`docs/CURRENT_MODEL.md` · `docs/PAPER1_FRAMING_AUTHORITATIVE.md` (**ARCHIVED**, "do not read, cite or
execute") · `docs/EXPERIMENT_PLAN.md` ("the only current plan" — it is not) · `docs/PROJECT_STATUS.md`
· `docs/PROJECT_BOARD.md` · `docs/ESTATE_REGISTRY.md` · `docs/ARTIFACT_INDEX.md` · `CLAUDE.md` itself
(auto-loads and is **wrong about which plan governs**) · **and now `paper_iclr2027/README.md`**, which
still carries the *old paper title* and claims 9 main-text pages / 18 total (it is 14 and 33).

**The governing plan is `docs/MASTER_PLAN_EXPERIMENTS_AND_COMPARATORS.md`**, lines 1–425. Line 426
opens `ARCHIVED PROVENANCE — NOT GOVERNING`. Two later blocks (`⭐ CURRENT AMENDMENT` at 1141,
`⭐⭐ CURRENT AMENDMENT II` at 1740) declare themselves governing but sit **below** that divider — an
unresolved wrinkle; know it before citing.

⚠️ **Project-wide reproducibility hazard:** 15 of 15 sampled recent `docs/*.json` results record **no
git commit and no producing script**, and `docs/EXTENDED_CURVE_64.json`'s producer reads a `/tmp`
cache with no volume fallback. See `docs/REPRODUCIBILITY_HAZARDS_2026-08-19.md`.
**The counter-example to imitate** is `diagnostics/editing_v2_experiment1_reference_law_paper.json`,
which binds producer app, git commit + date, clean-tree flag, every input SHA, and an
`EXACT_TO_TOLERANCE` reproduction verdict against the development artifact. **New paper-bearing
artifacts should follow that schema.**

---

## 4. Experimental doctrine — what counts as evidence

Three **non-interchangeable** kinds:

**A · External task competence.** Match the *published scientific task*, not internal mechanics:
source cohort, objective/success criterion, returned-candidate budget, benchmark-defined oracle budget
if any, normalization, metrics, seeds. **Each method uses its native inference procedure.** COMPOSE is
not forced to discard STOP, valid intermediates, future-aware control, or a prospectively frozen
sampler because a comparator lacks them. Do **not** impose equal internal steps, kernel evaluations,
wall-clock, FLOPs, or property evaluations unless the benchmark defines that constraint.
- **GrIDDD QED:** the binding budget is **20 returned candidates/source**; not an oracle-call benchmark.
- **InversionGNN:** oracle counts **are** part of the task and bind.
- A result may be **better benchmark performance** at greater inference compute. It may **not** be
  called more efficient unless separately established.

**B · Internal causal evidence.** Native COMPOSE vs COMPOSE **minus exactly that capability**, with
process, sources, objectives and relevant resources fixed. **This is where matched compute matters.**

**C · Process/property characterization.** Descriptive; no manufactured baseline.

**Resource accounting.** Log everything internally; report benchmark-defined constraints plus what is
materially necessary to interpret a claim. **Training cost never rhetorically offsets inference cost.**

**Reuse claim, exactly.** The paper may claim **"no objective-specific retraining of the molecular
reference process."** It may **not** claim "no objective-specific training anywhere" — controller
adaptation is separate and stated explicitly.

**Why the task policy is allowed to vary.** For QED the problem is hitting one fixed region; for
multi-objective work it is navigating *among* regions under an oracle budget. Forcing a multi-objective
optimizer to pretend it solves a fixed-target hitting problem would be worse science, not fairer
science. Discipline per task: `disjoint development data → small principled policy family → FREEZE ONE
→ fresh evaluation`, with **no outcome-driven rescue** after the freeze.

---

## 5. Complete experiment history

### 5.1 ⭐ NEW — Experiment 1: the reference law learns state-dependent preferences (§5.1, BANKED)
**Artifact:** `diagnostics/editing_v2_experiment1_reference_law_paper.json` (status
`PAPER_BEARING`; producer `modal_apps/experiment1_reference_law_app.py`, git commit
`eee0a0eabf6530d50a9cfa72ed781f88199cc196`, clean tree).

**Question:** does `R_θ` learn state-dependent transition structure beyond corpus-wide edit-family
frequencies? **Design:** 3,545 held-out transitions, 2,513 sources with partitions, minimum cell 20.
**Source disjointness VERIFIED by direct set intersection** of 10,653 reserve source keys against
96,094 training source keys: **empty**.

**The five-arm NLL ladder (nats, lower better):**
| arm | overall | data-backed families only |
|---|---|---|
| uniform over canonical successors | 6.2537 | 6.2172 |
| **empirical family frequencies** (baseline) | **5.3666** | **5.5477** |
| learned family, uniform identity | 5.1723 | 5.7922 |
| empirical family, learned identity | 4.0992 | 4.2462 |
| **`R_θ` (full)** | **3.9049** | **4.4907** |

- **Headline:** 5.37 → 3.90 nats, **improvement 1.4616**.
- **Decomposition:** improvement from empirical→identity-only is **1.2673**, i.e. **86.7%** of the
  total gain is recovered by keeping empirical family frequencies and learning only *which successor
  is preferred within the current state*. **Most of the learned signal is not which kinds of edits
  occur globally, but which executable edit is appropriate for the molecule at hand.**
- **Robustness to the synthetic-ring-lane problem (§3.2):** restricted to families with data-backed
  supervision (`atom_delete, atom_insert, atom_restate, bond_reorder, bond_reroute`; 2,744 scored
  transitions, 7 capability cells), `R_θ` retains a **1.0569-nat** advantage over empirical family
  frequencies. *The stronger gains on synthetically supervised ring families are therefore not
  required for the claim.* Derived artifact: `diagnostics/editing_v2_reference_law_data_backed.json`,
  partitioned by `diagnostics/editing_v2_family_lane_provenance.json` with the explicit rule
  `synthetic_only(F) iff n_F(reversible_synthetic_walk) > 0 and Σ_{data-backed lanes} n_F = 0`.
- **Reproduction:** deterministic re-score of the frozen development experiment,
  `max_abs_delta 6.4e-10`, verdict `EXACT_TO_TOLERANCE`. Same scorer, same frozen checkpoint, same
  frozen reserve; the original developmental artifact is retained unmodified.
- **Stated limit in the paper:** *"We use `R_θ` throughout as a goal-independent reference law over
  fixed executable support, without interpreting its probabilities as physical kinetics or synthetic-
  route likelihood."*

### 5.2 `R_θ` training — established facts (from `docs/DECISION_LOG.md`)
- Reference-law-weighted NLL **5.4603 → 2.9238** (2.54 nats over two epochs).
- **Within-family identity NLL improved in all eight families:** cycle_insert 4.7147→0.9071 (−3.81);
  atom_insert 5.1526→3.2900 (−1.86); atom_restate 5.2047→3.3868 (−1.82); bond_reroute 3.6376→2.5431
  (−1.09); atom_delete 1.6492→1.0052 (−0.64); bond_reorder 1.5457→1.3122 (−0.23); ring_system_restate
  0.7139→0.5007 (−0.21); cycle_attach 2.5427→2.5360 (−0.01).
- **Improvement scales with scaffold support:** −0.22 / −0.21 / −0.38 / **−0.62** across bands
  0 / 1–4 / 5–24 / 25+. Supportable: *transfers to unsupported chemistry, benefits increasingly from
  denser support.* **Not** supportable: strict monotonicity (the two low bands tie within 0.015).
- **No cross-cohort cost:** external 16-shard cohort 4.3364 → 4.5787 → 4.3817 → 4.3288.
- **Metric resolution ≈ 0.026 nats.** Adjacent evaluations closer than ~0.03 nats are **not
  interpretable** — four trend calls from consecutive points were each overturned next point.
- **The initialization is bit-reproducible** (0.00e+00 across three fresh containers).
- **The law never draws 29,600 of 136,028 rows** — whole synthetic strata whose real supply already
  meets target. The sequence is fixed, so this is **permanent exclusion, not slow exposure**. (This is
  also the second GM deviation in §2.3: *some strata are assigned zero draw.*)

### 5.3 Future reachability improves multi-step editing (§5.2, BANKED)
**Sealed 65-pair panel.** 65 endpoint-clean held-out source–target pairs, sources disjoint from
development, each admitting a certified executable edit path within budget. Both policies share the
same frozen `R_θ`, target, edit budget, and candidate successors.
- **Local (greedy, most-similar-to-target): 26/65. Future-aware rollout: 40/65.**
  Rescues **14 of the 39** local failures. **+21.5 percentage points, [12.3, 33.5]**.
- **`h_φ` as a prioritizer:** retains the same aggregate recovery using only **34.6%** of the
  continuation evaluations required by all-candidate lookahead. Reported as a **prioritization
  diagnostic, not a standalone evaluation of `h_φ`** — the two procedures do not recover exactly the
  same targets, and exact-target similarity uses information unavailable in ordinary property
  optimization. (Related banked secondary arms: similarity top-2 37/65 at 52% of continuations;
  `R_θ` top-1 33/65 — **plausibility is not purpose.** Safety property held on every arm: no arm lost
  a pair greedy recovers.)
- **Do NOT headline the p-value.** McNemar one-sided p = 0.0156, but the rollout policy is constructed
  to be no worse than its greedy base, so `greedy-only = 0` is a **structural guarantee**. The
  scientific effect is the rescue rate.
- **NOT claimed:** "`h_φ` beats similarity" (not preregistered; this bed cannot settle it).
  **NOT claimed:** "`h_φ` reproduces full rollout" — it matches the *count*, not the *rescue set*.
- **Figure 2(b) example, deterministic:** `diagnostics/editing_v2_pair10_trace.json`. Selection rule:
  *lowest sealed pair index among the 14 rescues at the median certified path length.*
  Source `CCC(C)NC(=O)c1ccc(CSCc2ccccc2)o1` → target `Cc1ccc(CSCc2ccc(C(N)=O)o2)cc1`, 5 verified
  steps. Greedy: not recovered, best similarity 0.7778, 27 universe candidates seen, 0 overrides.
  Full: **recovered (1.0)** with **1 override** and 20 continuation evaluations.
  `reproduces_banked_outcome: true`.
- **Companion characterization — when planning does NOT help.** On DRD2 (C0) mean fresh regret was
  **negative at every depth**, sacrificial win rate 57.1% at **0.5 SE** from chance, greedy solved
  7/12 sources, outcomes **bimodal** (1–5 edits or never close) — a locally easy landscape leaves
  planning nothing to buy. Paired with the above this *characterizes when remaining-budget
  information matters*, a stronger framing than either result alone.
- **The amortization gap** (why a learned value is not a cheap planner): `h_φ` trained on 14,110
  labels from the exact rollout teacher does **not** retain the benefit. Rescue ranking 30.5% (2k) →
  35.2% (5k) → 42.2% (10k) → 44.5% (all); harmful override 21.5% → 13.8%; contrastive 52.3% → 65.9%.
  Baselines — rescue: greedy 0% by construction, random 20.8%, second-highest-similarity 33.3%.
  Every metric beats baseline and improves monotonically, **but no override threshold yields net
  benefit** (−10.2 states at margin 0; the sole positive +0.5 at margin 6.0 overrides 9 of ~415 states,
  i.e. it has stopped acting). **Structural cause, generalizes:** safe states (158) outnumber
  rescue-needing states (36) **4:1**, so **precision dominates recall**; explicit rollout never faced
  this because policy improvement guaranteed it could never harm. **Not saturated** — rescue ranking
  still climbs at the largest subset, so data limitation and genuine unlearnability are both
  consistent; this experiment cannot separate them.

### 5.4 Source-conditioned QED editing (§5.3, BANKED — headline benchmark)
**Task (verbatim from GrIDDD/Jin, ZINC-250k):** 800 official sources with QED ∈ [0.7, 0.8]; **20
returned candidates/source**; solved if ≥1 returned molecule has QED ≥ 0.9 **and** Morgan Tanimoto
≥ 0.40. **One controlled trajectory → one returned candidate**; prospective STOP at first entry into
the region; trajectories that never reach it **remain failures rather than being resampled**.

**Controller, frozen before the panel was opened** (`docs/AMENDMENT_VALIDATION_128.md`):
receding-horizon **twisted SMC**, goal region (QED ≥ 0.90, sim ≥ 0.40), **frozen H24 value head**,
horizon **H = 40**, budget index capped `b_eff = min(24, b)`, **32 particles**. One returned candidate
costs **32 × 40 = 1,280 internal molecule evaluations**.

**Results:**
| panel | k | solved | note |
|---|---|---|---|
| 64 development (consumed) | 20 | **47/64 = 73.4%** | curve `28 32 35 36 37 40 40 41 42 43 44 44 44 44 44 44 44 46 46 47` |
| **128 prospective held-out** | **8** | **63/128 = 49.2%** | ledger: 1,024 runs, 1,167,787 committed transitions, 248 contact runs, 776 extinct, 2.58 distinct returned/run |
| **128 prospective held-out** | **12** | **70/128 = 54.7%** | curve `31 40 44 48 51 56 61 63 65 67 68 70`; not run beyond k=12 |
| 800 official | 20 | **NOT RUN (G1)** | one source executed by a pre-freeze smoke test, output **quarantined unread**, so the cohort is inferentially unconsumed |

**Published comparators (reported context, 800 official @ k=20):** JT-VAE 8.8%, CG-VAE 4.8%,
GCPN 9.4%, **GrIDDD 45.1%**.
**⚠️ VJTNN reports 60.6%** on the same δ≥0.4 task, Tier-A must-cite, **higher than both GrIDDD and
COMPOSE**. *This beats the closest per-step editor; it is NOT state of the art and must never be
presented as such.* Its protocol/initialization has not been audited to our standard. **This is gap G8.**

**⭐ NEW — diversity** (`docs/VALID128_DIVERSITY.json`, computed with **Jin et al. ICLR 2019's released
`scripts/diversity.py`, audited 2026-08-20**: admits candidates with sim ≥ 0.4 and prop ≥ 0.9,
deduplicates via `set()`, **excludes** sources with zero valid molecules, sources with exactly one
contribute 0.0; similarity = Morgan r=2, nBits=2048, useChirality=False):
- **COMPOSE 0.3929**, sd 0.2568, 95% CI **[0.3313, 0.4519]**; averaged over 70 sources, 19 contributing
  zero, 51 with ≥2 distinct; **4.34 distinct molecules per solved source**.
- Comparators: **GCPN 0.216 · GrIDDD 0.283 · VSeq2Seq 0.331 · VJTNN 0.373 · HierG2G 0.477.**
- So COMPOSE beats GrIDDD and VJTNN on diversity but **HierG2G (0.477) is higher.** Do not claim
  best-in-class diversity. GrIDDD reports diversity **without specifying its evaluator
  implementation** — flagged with a dagger in the table.

**Particle-count ablation** (`docs/HPHI_N_SWEEP.json`, seeds paired): N=32 → 19 successes / 42,502
transitions; N=8 → 7 / 11,091 (6 retained, 13 lost, 1 gained); N=4 → 3 / 5,944 (3 retained, 16 lost,
0 gained). Control quality depends strongly and monotonically on particle count.

**Still TODO in this section — the size-fixed ablation** (nested Block 1B/C). Run identical frozen
controller, `R_θ`, sources, candidate allowance and randomness as the 128-source panel, removing every
successor with `Δn ≠ 0`. **GrIDDD runs the analogous ablation and reports 45.1% → 33.8%.** A positive
gap shows trans-dimensional support contributes to benchmark performance rather than merely enlarging
the formal state space. **This is the answer to the "trans-dimensionality is just padded-slot
bookkeeping" reviewer attack, and it is the weakest point in the draft until it exists.**

### 5.5 Mid-trajectory retargeting (§5.5, BANKED and CLOSED)
40 source-disjoint held-out molecules × 2 histories (potency-first, developability-first), six arms;
panel sealed at `947e297a`, prefixes committed at `66687121` **before goal B was constructed**. Each
source is edited three steps before the joint objective `B` is specified, so the prefix cannot
anticipate the goal; after the switch, continue-`x₃` and restart-`x₀` receive the same `B`, the same
three-edit remaining budget, and the same frozen process.

| | potency-first | developability-first |
|---|---|---|
| **State reuse (PRIMARY): continue x₃ vs restart x₀** | **+0.302** [0.186, 0.413] 35W/5L | **+0.176** [0.069, 0.283] 29W/11L, p=0.006 |
| Retargeting: new vs old objective | +0.748 [0.608, 0.891] 36W/0L | +1.462 [1.240, 1.691] 40W/0L |
| Sensitivity (greedy-matched) | +0.367 [0.251, 0.482] 37W/3L | +0.211 [0.065, 0.361] 26W/14L, p=0.081 |
| **Price of surprise** (gap to clairvoyant) | **−0.252** [−0.364, −0.155] | **−0.428** [−0.580, −0.288] |

Development → confirmation: +0.716/+1.406 → +0.748/+1.462; +0.444/+0.290 → +0.302/+0.176;
−0.289/−0.481 → −0.252/−0.428. **Direction holds everywhere; the primary shrank by ~⅓** — development
panels usually overstate magnitude, and the confirmation existed to find that out.
- The paper states the benefit **is not universal across sources**, so state reuse is an *empirical
  advantage rather than a consequence of the controller construction*.
- **Do NOT write** "history's value is controller-dependent for D-first" — the mean effect is positive
  under **both** controllers; what differs is source-level sign consistency.
- **Do NOT write** "the goal changes at an arbitrary step." The machinery works from any realized
  state, but the experiment fixes **τ=3, H=6**; invariance to switch time is **not established.**
- Wiring audit (appendix): `greedy_restart` and `restart` return identical medians (−0.5316, −0.4283)
  across both histories — correct by construction, since restarting from `x₀` erases the history.
- **Distinction from REINVENT:** REINVENT carries a learned policy + optimizer state and keeps
  updating; COMPOSE carries **the actual molecule `x₃`** and performs **zero parameter updates.**

### 5.6 ⭐ LIVE FRONT — the matched 2-objective InversionGNN comparison (§5.4, ALL XXX)
**Why it fits where MOLLEO did not:** InversionGNN's protocol makes **task-specific training part of
the benchmark** — it spends oracle calls labelling ZINC and trains a property predictor before
optimizing. That licenses COMPOSE to train the multi-objective `h_φ` on the same budget:
```
InversionGNN   10K labels → F̂(x)        → gradient of predicted properties
COMPOSE        10K labels → h_φ(x,λ,b)   → future-reachability control over the exact legal fiber
                 SAME INFORMATION BUDGET. DIFFERENT INFERENCE PRINCIPLE.
```
**Pinned by the published paper:** **10K train + 5K optimize** (2 obj), 20K + 5K (4 obj); the 5K is
"1K per weight vector × 5 weights"; **5 preference vectors** by App. D.3's spherical-coordinate
algorithm; **C = 10** kept per generation; APS = mean of top-100; novelty; top-K diversity; vocabulary
= 82 substructures appearing >1000× in ZINC-250K. Reported: **APS 0.841, novelty 100%, diversity
0.768, HV 0.763 ± 0.031.**

**NOT pinned — why we rerun rather than quote:** the **starting molecules are never specified**; the
**HV reference point is never stated** for the molecular tasks (they follow HN-GFN, which uses the
origin — suggestive, not stated); and the released `denovo.py` is **not** a faithful Table-3
reproduction (hardcodes one start `C1=CC=CC=C1NC2=NC=CC=N2`, `population_size = 1`, single preference
`[1,3]`, against the paper's C=10 and 5 weight vectors). **That start is not neutral** — on our frozen
oracle it sits at **JNK3 = 0.100, the maximum of our entire random-ZINC dev cohort** (median 0.010),
and it is an anilinopyrimidine (kinase hinge-binding chemotype).

**Design:** freeze a common objective-blind init bank by hash (**100 molecules**, sha256 recorded
before either method runs); run **BOTH** methods from those starts with the same 10K labels, 5
preferences, 5K budget; report published 0.841 as **reported context**, never as the quantity our
number beats. Preferences recovered from App. D.3 and **verified by matching the paper's reported
objective ratios 0.16/0.51/1.00/1.96/6.31 to 3 dp.** PRIMARY metric **APS**; secondary HV under a
**preregistered origin reference point** computed identically for both arms.
**Scope: 2 objectives only** (JNK3 + GSK3β, both **maximised** — MOLLEO minimised GSK3β and the frozen
bundle stores `1 − gsk3b`, which **must be inverted**; that conversion is a parity check).
**Decision rule:** competitive or better on APS → escalate to 4 objectives. Clearly worse →
**troubleshoot the multi-objective `h_φ`; do not escalate to hide a 2-objective deficit.**

**The four addenda, in order:**
- **Addendum I — WITHDRAWN.** Binary region `B_λ` with `q_λ` = 10th percentile of the Chebyshev
  shortfall over the 10K labels. Its own diagnostic killed it: ~1,000 positives per preference as
  designed, but **five preferences collapsed to three regions** (`B0=B1`, `B3=B4`, Jaccard 1.000)
  because with every objective near zero `d_λ` is dominated by the larger λ component. **Fatal second
  problem:** `R_θ` trajectories from random ZINC top out at **JNK3 = 0.22**, so a controller trained
  on *observed hits* can only reach regions the rollouts visited, while InversionGNN's GNN
  **extrapolates**. Not a fair contest — an artifact of our construction, not of COMPOSE.
- **Addendum II — GOVERNING.** Same supervision, different inference. Train `F̂_ψ` on the 10K labels;
  score **oracle-free** `R_θ` rollouts with it; train `h_φ` to an **expectation**, not a hitting
  probability: `L_λ = max_i λ_i(1 − F̂_i)`, `g_λ = exp(−L_λ/τ_λ)`, `τ_λ` = std of `L_λ` over the
  10,000 **training** molecules, `h_φ(x,λ,b) ≈ E_{R_θ}[g_λ(X_b) | X_0 = x]`. No region, no threshold,
  no positive examples needed; still a proper finite-horizon h-transform.
  **`F̂_ψ` = what is desirable · `R_θ` = how molecules can move · `h_φ` = what is desirable in the
  future given how molecules move.**
  **The causal ablation this buys (G2 — the paper's mechanism claim):** same 10K labels, same `F̂_ψ`
  **weights**, same frozen `R_θ`, same exact fibers, same init bank, same 5 preferences, same budget;
  compare `R_θ(y|x)·g_λ(y)` (immediate) vs `R_θ(y|x)·h_φ(y,λ,b−1)` (future-aware). *Both arms know the
  identical predicted landscape; if future-aware wins there is almost nowhere for the explanation to
  hide.*
- **Addendum III — three corrections to how this is described.** (1) The supervision corpus is
  **matched, not identical**: say *"InversionGNN-matched random-ZINC supervision, with the common
  evaluation initialization bank explicitly held out."* An earlier commit called it "InversionGNN's
  own construction" — **false**; their routine shuffles ZINC with no such exclusion. The exclusion is
  better hygiene **and** a deviation; both are stated. (2) **"Same surrogate" means two different
  things** — within-COMPOSE attribution uses literally the same `F̂_ψ` weights (airtight);
  COMPOSE-vs-published does **not** share a predicted landscape and must never claim to. (3) **The
  mechanism, stated correctly:** 10K random-ZINC labels reach JNK3 max 0.400 (2 above 0.3, none above
  0.5) while benchmark outputs reach APS ~0.84 — **every method here depends on extrapolation beyond
  its label range.** Never write "COMPOSE discovers high-activity regions from trajectory evidence."
  Write: **property learning says where value may be; molecular-process control says how to get
  there.** Extrapolation quality is a *shared dependency*, which is precisely why the claim rests on
  the propagation step.
- **Addendum IV — authorised Gate-2 numerical correction (not a redesign).** `g = exp(−L/τ)` produced
  degenerate targets, so the computational representative became
  `g̃_λ = exp(−(L_λ − L*_λ)/τ_λ)` with `L*_λ` the mean of `L_λ` over the frozen training supervision.
  For fixed λ this is a positive x-independent constant → normalized controlled kernel unchanged
  (verified: `log g̃ − log g` constant to **7.1e-15** across 600 states × 5 preferences; normalized
  successor probabilities agree to **1.8e-15**). Softmax-max-subtraction, not a tuning knob.
  **Self-correction recorded:** the targets had *not* "collapsed to zero" — 0 of 600 were exactly zero,
  they ranged 1e-14 to 1e-26, float32's min normal is 1.2e-38. **No underflow.** A 4-decimal print
  rendered ~1e-20 as `0.0000`. The real failure is dynamic-range collapse relative to the network's
  output scale. **Watch:** centering leaves a heavy right tail (per-preference median ~0.77, max
  634 / 953 / 23005), so rank correlation is reported alongside R²; if the tail dominates, that is a
  finding to rule on, **not** something to silently transform away.

**Gate sequence, fixed:** (1) does `F̂_ψ` train sensibly? (2) does `h_φ` pass held-out Bellman /
future-value diagnostics? (3) on identical held-out decision fibers, does `h_φ` rank decisions better
than immediate `F̂_ψ`? (4) **only then** spend the 5K optimization budget.
**If 2 or 3 fails, troubleshoot the VALUE-LEARNING IMPLEMENTATION** — not the corpus, not `R_θ`, not
the operator set, not the benchmark, not the preference family.

**⚠️ CURRENT STATUS — Gate 2 is WEAK.** `diagnostics/invgnn_gate2_diagnosis.json`, 600 states,
budgets 1–8, split 424 train / 86 val / 90 test, **test rows only**, Spearman primary (R² reported
alongside because the target's tail makes it fragile):
| arm | Spearman | R² |
|---|---|---|
| **1 persistence** (trivial baseline) | **0.8455** | 0.7712 |
| 2 emb | 0.2584 | 0.0237 |
| **3 emb+base** (the real arm) | **0.4838** | 0.3057 |
| 4 emb+base/log **[exploratory]** | 0.8617 | 0.6833 |
**The learned value model is far worse than a trivial persistence baseline.** Only the exploratory
log-target arm edges past it, and the artifact states plainly *"Arm 4 is exploratory and is not a
substitute for arm 3."* Persistence degrades gracefully with budget (0.924 at b=1 → 0.782 at b=8)
while arm 3 is flat-to-rising (0.427 → 0.496), which is itself diagnostic. **This is the live blocker
on G2/G3.** Per the fixed gate rule, the response is to fix the value-learning implementation.

**Banked supporting probe — the nonmyopic gap is real** (`docs/INVGNN_LOOKAHEAD_TRUTH.json`, commit
`0a219a0`). Oracle-free high-Monte-Carlo probe of the **true** future value under `R_θ`: 24 held-out
states × 16 candidates × 32 rollouts = 120 decisions, scored by the frozen surrogate, zero oracle calls.
| lookahead b | Spearman vs immediate | top-1 **flip** rate | median realized gain |
|---|---|---|---|
| 4 | 0.518 | **68.3%** (82/120) | **+19.3%** (abs +0.186) |
| 8 | 0.347 | **84.2%** (101/120) | **+21.4%** (abs +0.223) |
**This is a property of the TRUE value, not of a trained controller.** It establishes the gap is worth
chasing; it is **not** the benchmark claim.

Assets: `docs/INVERSIONGNN_FPSI.pt`, `docs/INVERSIONGNN_HPHI.pt`,
`docs/INVERSIONGNN_SUPERVISION_10K.json.gz`, `docs/INVGNN_HPHI_TRAIN_STATES.json`,
`docs/INVGNN_FIBER_MOLECULES.json`; apps `invgnn_corpus_app.py`, `invgnn_hphi_corpus_app.py`,
`invgnn_hphi_encode_app.py`, `invgnn_lookahead_truth_app.py`, `invgnn_mc_sweep_app.py`; scripts
`invgnn_train_hphi.py`, `invgnn_gate2_diagnosis.py`. The Addendum-I corpus
`invgnn_v1/corpus_10k.json.gz` is **quarantined as development evidence** (its 10,000 oracle calls are
development spend, **not** charged to the benchmark).

### 5.7 Pathwise constraints (§5.6, DEVELOPMENT VALUES IN PLACE — confirmation NOT opened)
**Design:** a fixed cLogP corridor `C` while optimizing an independent DRD2 objective over six edits.
Endpoint-only control requires only the returned molecule to lie in `C`; COMPOSE's pathwise controller
removes successors outside `C` **before** a transition is sampled. Same `R_θ`, sources, objective,
horizon, budget. **Because zero pathwise violations follow from support restriction by construction,
the empirical question is not whether the mask works, but whether endpoint-only optimization actually
hides meaningful violations** — i.e. **the informative quantity is prevalence and wasted budget, not
the feasibility fraction.**

**⚠️ The section is currently written in the SUCCESS branch of a confirmation that has not been
opened.** Numbers render from `\Pw*` macros in `main.tex` holding **development (n=24)** values from
the verified/future-aware arm: 14/24 (**58.3%**) of endpoint-feasible trajectories leave `C` at least
once; excursions extend a median **0.69** cLogP units beyond the admissible region and persist **four
of six** committed edits; terminal DRD2 utility changes by only **−0.023 [−0.182, 0.134]**.
**Development did NOT pass confirmation** — it is a different, already-inspected sample, and the cLogP
corridor was **chosen after a feasibility screen**. Never write or imply otherwise.
**Frozen confirmation:** n=48, corridor `C = [2.3689, 4.4522]`, **intersection-union** on
(i) lower CI bound on hidden-path prevalence > 1/3, and (ii) lower CI bound on utility difference
> −0.25 IQR (−0.20 sensitivity). Branch rules, fixed in advance:
- **both pass** → keep the title and prose, swap macros, set `\PwEvalPhrase` to "On the 48-source
  held-out evaluation", delete the `\DEVELOPMENTALPATHWISE` sentinel.
- **utility fails** → retitle *"Endpoint feasibility can hide pathwise molecular constraint
  violations"*; the phenomenon becomes the result. Delete paragraph 4 and the "retaining comparable
  endpoint utility" claim with it.
- **prevalence fails** → the section loses its motivation; **compress or demote it rather than
  rescuing it with secondary analyses.**
**Do not change the corridor, either threshold, the controller, or the analysis after seeing the
confirmation. Do not rewrite the question after the answer.**
**`main.tex` says `scripts/check_submission.sh` FAILS while `\DEVELOPMENTALPATHWISE` is defined —
but that script DOES NOT EXIST YET.** Write it, or the guard is decorative.

### 5.8 Exact finite-horizon control — machine-precision verification (BANKED, provenance gap U1)
`diagnostics/exactness/editing_v2_experiment_b_exact_control.json`. Enumerable slice: carbon-only,
heavy-atom cap 6, **966 states**, budget 6, target "heavy atoms ≥ 6" (GROW), 770 target states,
reach probability 3.78e-5.
- terminal-tilt TV **1.77e-16** · support violations **0** · backward residual **0.0** · max row-sum
  error 4.44e-16 · h-partition gap 2.03e-20
- **Retargeting from realized switch state `CC(C)C` to "< 4 heavy atoms" (SHRINK) with 3 budget left:**
  terminal-tilt TV **1.39e-17**, support violations 0, backward residual 0.0.
- **An unreachable target correctly reports h = 0** rather than being ε-patched.
- Self-labels `DEVELOPMENT_RESULT_NOT_PAPER_BEARING` on a carbon-only slice → **gap U1**: either
  promote with full provenance (following the Experiment-1 schema) or rebuild on the current registry.

### 5.9 The MOLLEO lane — OPENED, PROBED FOUR WAYS, CLOSED
MOLLEO Task 3 (ICLR 2025): `max QED, max JNK3, min SA, min GSK3β, min DRD2`; 120 random ZINC init;
≤10,000 oracle calls; 5 seeds; hypervolume.
| probe | result |
|---|---|
| lazy verification gate (L1 vs L4) | did not diverge; HV +0.5%; **0 resampling events** |
| basin substrate | 42/96 reverse-verified routes; one-step fiber max median 0.365 |
| **fiber census — decisive** | **0 of 24 random-ZINC one-step fibers contain a JNK3 successor ≥ 0.3; max 0.16 over ~16,000 successors** |
| bridge probe (target supplied) | similarity 0.21 → 0.51, JNK3 to 0.37, **none reached 0.5** |
| coverage sentinel | 2× scaffolds, no JNK3 gain, moved *further* from actives |
| task-independent bridge training | goal-conditioned reachability, zero task-oracle calls, top-10 **61.5% at b=24 (51× chance)** |
| bridge navigation gate | **NULL**: 0 exact hits both arms, Δsim +0.006, sign test 28/50, p=0.24 |
**Why it failed and why that is not a COMPOSE indictment:** MOLLEO asks an optimizer to find a remote
kinase basin from random ZINC with **no task-specific training**; the neighbourhood is genuinely flat.
**Do not reintroduce the TDC image** (PyTDC + `rdkit.six` shim + sklearn 1.2.2) — the frozen
`artifacts/oracles/molleo_task3_v1/` bundle supplies the same oracles as SHA-pinned `.npz`.

### 5.10 Refuted hypotheses — DO NOT RE-TEST
| hypothesis | verdict | evidence |
|---|---|---|
| The compact successor fiber is the object to optimize | **wrong target** | partitions discarded by the training loop |
| Larger batches speed training | **refuted** | batch 128 is 3× worse than 32 |
| Vectorized backward pass gives a large speedup | **refuted** | 2%, not 20× |
| The model memorizes slot permutations | **refuted** | scorer equivariant to 1e-6 |
| The gap is a representation problem | **refuted** | provenance correlation ρ = +0.762 |
| Held-out error grows with analogue-series depth | **refuted, opposite** | error *falls* 3.87 → 2.42 with depth, surviving fixed-zero-support (n=5,863) and fixed-molecule-size (n=6,992) controls |
| Most-oversampled strata overfit first | **refuted** | the three at the 3.0× ceiling improved most |
| `atom_restate` suffered capability collapse | **refuted** | its *identity* improved 1.82 nats; the joint gate was reading family-head reallocation |
| Cross-cohort performance is degrading | **refuted** | third and fourth readings returned to baseline |
| DRD2 is rugged enough that future value beats greedy | **refuted** | §5.3 companion |
| Historical traces can be reconstructed from the corpus | **refuted, structurally** | corpora are `(state, edit)` samples, **not paths**: 29,928 forked states in train; strict funnel reached **0** |
| The `linker_positional_topology_analogue` lane implies two-cut/topology mining | **refuted** | post-hoc router label; zero repo hits for `two_cut`/`double_cut`/`n_cuts` |
| Policy B (one-step local tilt `π ∝ R_θ·QED`) can optimize QED | **refuted, 0/320** | see below |

**Policy B — the sealed developmental negative, 0/320.** 64 disjoint sources × 5 replicates.
Primary `1[QED≥0.9 ∧ sim≥0.4]` = **0/320**, source-clustered CI [0,0]. Tanimoto ≥ 0.4 alone
**0.6719** ✅; QED ≥ 0.9 alone **0.0063** ❌. Terminal QED median 0.7470 vs source 0.7576. **0 dead
ends, 0 zero-denominator fallbacks.** *Similarity is not the problem; optimization is.* A one-step
tilt against a strong reference law cannot move a bounded objective. ⛔ **Never call this "COMPOSE"** —
it is the **myopic-control ablation**. Barred responses: `QED^α`, temperatures, top-k, cap tuning.

**Horizon amendment.** The saturation criterion returned "no saturation ≤ 24" and therefore chose no
horizon (decay ratio 1.00 at ≥0.85). H6 captured only 43% of 0.85-reachability, 33% of 0.90. Unguided
geometry: 0.80 → 32%, 0.85 → 10.9%, **0.90 → 1.2% (rare but NONZERO)**, 0.95 absent.
*The roads exist and the destination requires navigation.*

**Structural limit — one-cut MMP cannot produce diverse multi-step panels.** Over 972 nominated pairs:
160 `bond_reroute` + 146 `atom_restate` + 17 `bond_reorder` = **323 direct compiles**, against
**exactly 323 paths of length 1**. Everything of length ≥2 is `delete_insert_fallback`. **There is no
middle** — so *every 4–6 step transformation this machinery can produce is a delete-insert fragment
swap by construction.* Consequences: `ring_delta` is **0 across all 91** accepted pairs; the 86%
`REFERENCE_DIP` rate is a **compiler artifact**. The 91 pairs remain a valid held-out multi-step
recovery cohort but support **no non-locality claim**. Provenance to claim: *held-out endpoints plus
unseen supervised transitions* — **not** "the model never saw any intermediate" (38 of 91 have an
intermediate appearing elsewhere as a training source).

### 5.11 Corrections and harness defects already paid for
**Claims made and then found wrong:** scaffold coverage is **58.5%**, not 85.5% (the 85.5% counted
train-*role* sources; the model trains on the compiled library, 106,759 of 564,316) — this *reversed a
retraction*, chemical novelty **is** substantial · support-matching does **not** flatter the number
(2.74 vs 2.62, i.e. **+0.12, slightly worse**) · **"depth 0" meant acyclic, not shallow** (all 361
depth-0 entries had an empty Murcko scaffold and `if s:` treated the empty string as no-scaffold) ·
**the family floor was never enforced** (clamp-then-renormalize puts a floor-pinned family *under* the
floor: claimed 0.05, delivered 0.0497; replaced with water-filling) · **the joint NLL is the wrong
capability gate** (it moves with family-head reallocation the model is entitled to perform; the gate
now reads **identity**, over two consecutive evaluations).

**Harness defects (all fixed):** **no evaluation ran on the final step** (one epoch = 4,251 steps
against an interval of 500, so runs ended with their last evaluation 251 steps stale; step 4,000's
weights are **permanently lost**) · the eval line logged everything **except** the selecting number ·
re-collating per batch cost **98.17 s per 32 examples** (97.2% of step time) → packed store, GPU
data-wait fell to 0.93% · throughput charged eval+checkpoint time · the carve was defined over the
wrong set · four byte-identical duplicate entry ids · cost estimates counted work, not billed
resources.

**DRD2 oracle provenance — the model for porting a benchmark oracle.** The classic SVM ships as a
Python-3.6 sklearn pickle, opened **once** by `scripts/drd2_oracle_extract.py`; the runtime thereafter
evaluates frozen arrays in numpy. Two things had to be *reproduced*: **libsvm's Wu-Lin-Weng coupling**
(short-circuiting to the Platt sigmoid left a **1.7e-3** discrepancy — enough to move molecules on the
0.5 threshold) and **the Platt orientation** (an inverted oracle returns plausible probabilities while
rewarding the wrong molecules; both orientations are scored and the rejected one is *asserted to fail*
parity, so agreement is evidence rather than luck). Parity **2.19e-14** on probabilities, **1.35e-13**
on decision values over 200 molecules. **The task uses two different fingerprints:** activity =
count-based FCFP6 (r=3, `useFeatures=True`, folded by modulo); similarity = ECFP4 bits (r=2, 2048).
Conflating them silently redefines the benchmark.

---

## 6. Where the project stands right now (2026-08-20)

**Last commit:** `0a219a0` *"The nonmyopic gap is real: lookahead under R_theta ranks decisions
differently from immediate score."* **Everything from 2026-08-19 12:00 onward is UNCOMMITTED** —
8 new artifacts, 4 new scripts, 3 new modal apps, and the whole of `paper_iclr2027/`. Commit soon;
the last handoff already flagged uncommitted headline artifacts as a hazard and it recurred.

| lane | status |
|---|---|
| `R_θ` reference process | **frozen, never retrained.** Done. |
| **§5.1 reference-law experiment** | **✅ BANKED, paper-bearing, provenance-complete** |
| **§5.2 future reachability (sealed 65)** | **✅ BANKED** + deterministic Fig-2(b) trace |
| §5.3 QED / GrIDDD editing | 128 prospective **banked** (49.2% @8, 54.7% @12) + **diversity 0.393**; **official 800 × 20 NOT RUN (G1)**; **size-fixed ablation NOT RUN** |
| §5.4 InversionGNN 2-obj | **LIVE, BLOCKED at Gate 2** — learned `h_φ` below the persistence baseline |
| §5.5 retargeting | **✅ BANKED and CLOSED** (held-out replication) |
| §5.6 pathwise | development n=24 in place; **frozen n=48 confirmation NOT OPENED** |
| Exact Doob verification | banked, development-labelled → **U1** |
| InversionGNN 4-obj | **gated** on the 2-obj decision rule (G4) |
| Pareto fan / map reuse (G7) | only once a useful realized map exists |
| MOLLEO | **CLOSED** |
| **Paper** | compiles; **main text ≈14 pages against a hard 9-page limit**; 33 pages total; 26 `\XXX`; **all four figures are placeholder boxes** |

**Critical path, in order:**
1. **Fix the InversionGNN `h_φ` value-learning implementation** so Gate 3 can be attempted. The
   diagnosis points at target representation (the log-target arm is the only one that beats
   persistence) — but arm 4 is exploratory and cannot be adopted by fiat. Whatever is adopted must be
   fixed *before* seeing G2/G3 outcomes.
2. **G2** — the matched immediate-vs-future-aware run. *The paper's mechanism claim depends on it.*
3. **G3** — the 2-objective head-to-head (both methods, frozen init bank, 5 preferences, 5K budget).
4. **G1** — the official 800 × 20, **once**, with the frozen controller; bank the full k=1…20 curve.
5. **The size-fixed ablation** — the answer to the strongest reviewer attack.
6. **Open the n=48 pathwise confirmation** and apply whichever branch rule fires.
7. **G8** — resolve the VJTNN row before anything is circulated.
8. **Cut the main text from ~14 pages to 9.**
9. Then: G4 (gated) → G5 (retargeting at scale) → G7 (Pareto fan).

---

## 7. THE PAPER

### 7.1 Identity — note the title changed
- **Draft:** `paper_iclr2027/` — `main.tex` + `sections/*.tex`. The preserved drafts (`paper/`,
  `paper_arxiv/`, `paper_iclr_control_substrate/`, `paper_iclr_stochastic_rewriting/`) are the
  technical sourcebook and are **never edited**.
- **⭐ CURRENT TITLE (changed 2026-08-20, marked "fixed by the owner. Do not alter."):**
  > **COMPOSE: Generative Stochastic Rewriting for Trans-Dimensional Molecular Design and Control**

  The previous title said *"Generator Matching for Trans-Dimensional Molecular Editing and Control"* —
  it was changed because generator matching is **not** the training framework (§2.3). Any document
  still carrying the old title (including `paper_iclr2027/README.md`) is stale.
  Acronym: **CO**ntrollable **M**olecular **P**rocess **O**ver **S**tochastic **E**dits.
  Typeset at 15.9/18.8pt rather than the style's 17.28pt so the title makes two clean lines without
  hyphenating "Con-trol".
- **Venue:** official ICLR 2027 kit (sha256 recorded in `main.tex`). **9 pages main text**, references
  unlimited, → 10 for rebuttal/camera-ready. `\iclrfinalcopy` **stays commented out**; the style file
  substitutes the anonymous block itself.
- `\raggedbottom` is set deliberately: ICLR's `\flushbottom` dumped an entire page's shortfall into a
  lone section heading (44pt above / 30pt below "Discussion and limitations" against an 8pt paragraph
  gap). **Re-check once real figures replace the placeholder boxes.**

### 7.2 Current structure (as `main.tex` inputs it)
```
Abstract
§1 Introduction                         (p.1)   + Figure 1 (possible/plausible/purposeful)
§2 Related work                         (p.2)
§3 Preliminaries                        (p.3)   3.1 Markov chains · 3.2 finite-horizon Doob control
§4 Methods                              (p.4)
   4.1 Executable molecular transitions (p.4)   4.2 Learned molecular reference process (p.5)
   4.3 Canonical molecular successors   (p.6)   4.4 Finite-horizon molecular control (p.7)
   4.5 Goal- and budget-conditioned future value (p.7)
   4.6 Controlled molecular sampling    (p.8)   4.7 Multi-objective preference control (p.9)
   4.8 Retargeting and pathwise constraints (p.10)
§5 Experiments                          (p.10)
   5.1 COMPOSE learns state-dependent preferences among executable molecular edits
   5.2 COMPOSE uses future reachability to improve multi-step molecular editing
   5.3 COMPOSE performs strongly on standard source-conditioned molecular editing
   5.4 Future-aware control improves Pareto exploration under competing objectives
   5.5 COMPOSE's realized molecular states preserve progress when objectives change
   5.6 COMPOSE prevents constraint violations hidden by endpoint-only molecular optimization
§6 Discussion and limitations           (p.14)
Appendix A–G (see 7.7)
```
**Note the deliberate divergence from the earlier plan:** Related Work sits at **§2**, before
Preliminaries and Methods — not at §6. An earlier brief argued for method-first; the draft went the
other way and `related.tex` was revised again on 2026-08-20. **Treat §2 placement as the current
decision.** `sections/trajectories.tex` was **absorbed into 4.7–4.8** and is retained but no longer
`\input`.

### 7.3 The two conventions — keep them
1. **No number appears that is not in a verified source.** The verified set is exactly seven files
   (`SESSION_RUN_MANIFEST_2026-08-18.md`, `VALID128_K8_RESULT.json`, `VALID128_CURVE.json`,
   `EXTENDED_CURVE_64.json`, `AMENDMENT_VALIDATION_128.md`, `GRIDDD_JIN_PROTOCOL.md`,
   `AMENDMENT_INVERSIONGNN_2OBJ.md`). Every numeric claim carries a `% source:` comment.
   ⚠️ **This list is now out of date** — §5.1's numbers come from
   `diagnostics/editing_v2_experiment1_reference_law_paper.json` and the diversity figure from
   `docs/VALID128_DIVERSITY.json`. **Add both to the verified set in `main.tex` and annotate their
   uses**, or the convention is being violated silently.
2. **Unmeasured quantities are `\XXX`, never guessed** (renders red). Marker classes:
   `GAP: NOT YET RUN` · `GAP: NUMBER NOT IN VERIFIED SOURCE SET` · `GAP: MISSING COMPARATOR ROW`.
   Current inventory: **26 `\XXX`** — abstract 1, experiments 10, appendix 12, fig3 3;
   **7 `GAP:` blocks** — abstract 1, intro 1, appendix 5.
   Audit: `grep -n 'GAP:' sections/*.tex` and `grep -n 'XXX' sections/*.tex`.

### 7.4 ⭐ The writing doctrine — decided over the last two sessions

**Models to emulate, and what each contributes:**
- **pCoMole — task-first compression.** Begin with the engineering setting, name the missing
  conjunction of capabilities, introduce one construction, one practical approximation, close with
  applications. Do **not** open by teaching the reader the underlying formalism. pCoMole is also a
  scientific neighbour: it *guides a pretrained variable-length Edit Flow* toward preferences with
  terminal feasibility, approximating its h-function by short rollouts. **The paper must never sound
  like "pCoMole, but on small-molecule graphs."**
- **AReUReDi — one nearest ancestor, one missing capability, one algorithm.** Name the base, state
  exactly what it lacks, introduce only the machinery that closes the gap. **Do not introduce seven
  independent "innovations."**
- **PepTune — separate base generator from guidance**, with contribution bullets mapping 1:1 to
  method sections and **declarative experiment titles**.
- **MadSBM — contribution formatting.** No standalone "Contributions." heading; the final
  introduction paragraph flows into *"Our main contributions are fourfold:"*. **✅ ALREADY APPLIED.**
- **MOG-DFM — capability-centered result headings.** Its headings read like *"MOG-DFM generates
  peptide binders under five property guidance"* — the **capability**, not the mechanism, not an
  abstract label. **This is the template for every §5 subsection title**, and §5.1/5.2/5.3/5.5/5.6
  already follow it.

**⭐ THE SECTION TEMPLATE — established while rewriting §5.2, apply to every experiment subsection:**
```
1. ABILITY   — what COMPOSE uniquely enables, and the structural reason it can (one paragraph,
               opening sentence states the capability, not the problem)
2. TEST      — how we isolate it: the panel, what is held fixed, what differs between arms
3. RESULT    — the numbers, then the mechanism, then a deterministic illustrative example
4. SCALE     — how the capability is made usable at molecular scale, with its honest caveat
5. BRIDGE    — one sentence handing off to the next subsection
```
The old §5.2 opened *"a locally attractive edit can be a poor multi-step decision…"* — problem-first.
The revised version opens *"COMPOSE can evaluate a molecular edit by the futures it leaves
reachable…"* — **capability-first**, then connects
`canonical successor → realized complete molecule → continue R_θ from it → evaluate remaining-budget
reachability`, which is the real conceptual payoff of the representation. The section is no longer
"lookahead beats greedy"; it shows **why COMPOSE's molecular process makes meaningful lookahead
possible in the first place.**
**Vocabulary discipline within a section:** pick one term and repeat it. Use **"future reachability"**
for the capability; "future-aware" may describe the controller. Do not alternate among
future-aware / lookahead / continuation value / nonmyopic.

**Related work — settled form.** Three ordinary prose paragraphs, **no bold run-in labels, no
mini-subsections**; the topic sentence does the organizational work; each paragraph **closes on the
COMPOSE distinction**. Target ~375–425 words: longer than MadSBM's unusually compact section because
COMPOSE genuinely spans three reviewer-relevant literatures, short enough not to read as a survey.
1. molecular generation/editing → **GrIDDD** → *COMPOSE instead learns a goal-independent stochastic
   law over an explicit, state-dependent molecular rewrite space, making the transition process itself
   the object of subsequent control.*
2. stochastic-process learning and post-training control → generator matching, **Edit Flows**,
   **MadSBM** (closest relative; its learned control *constructs* the transport rather than adapting
   an already-learned process), **Value Matching**, **pCoMole** →
   **⭐ the concession sentence added 2026-08-20, keep it:** *"Future-aware generative control is
   therefore not unique to COMPOSE: value-guided flow adaptation and pCoMole likewise use learned or
   rollout-estimated future value. Our distinction is the substrate on which that control operates.
   It is a learned canonical transition law over executable molecular graphs, which is what allows
   realized molecular states to be reused, retargeted, and constrained."* Conceding priority on
   future-aware control and relocating the claim to the substrate is **stronger and more defensible**
   than implying the control idea is ours.
3. multi-objective molecular design → **InversionGNN**, **PepTune**, **AReUReDi**, **Routing by
   Reaching** → *COMPOSE places multi-objective control on a reusable molecular transition process,
   so future reachability and already-reached molecular states can participate directly in Pareto
   exploration.*
**No fourth "stochastic control" paragraph** — Doob / Feynman–Kac / Schrödinger citations live in the
Preliminaries and §4. Graph-GRPO is held for the extended related work unless we compare directly.
**MOLLEO is out of the narrative.**

### 7.5 Terminology contract — non-negotiable
| use | avoid |
|---|---|
| **reference edit law** / **learned molecular reference process** / stochastic molecular edit process | "molecular dynamics model" (invites physical kinetics) — ✅ already removed from the abstract |
| **complete molecular graphs**; connected and valence-admissible within the declared state space | "chemically valid" unqualified — ✅ already removed from the abstract |
| **one reusable executable molecular process** | **"one reusable molecular world"** — ⚠️ **STILL IN THE ABSTRACT'S LAST SENTENCE**, retained as author wording. Great for a talk; unsafe as a formal claim |
| learned future value; **amortized / approximate** `h_φ` | "exact controller" when `h_φ` is learned |
| **exact h-transform** (reserved for: exact executor support, exact canonical normalization, exact DP, exact terminal tilt under exact `h`) | "exact learned guidance" |
| **productive-successor likelihood** | "trained by generator matching" — **now factually wrong** |
| source-conditioned optimization | "conditional generation" without defining the conditioning |
| canonical molecular successor | "action alias" without explanation |
| task-trained controller | "task-agnostic system" when `h_φ` used task labels |
| executable edit | "synthesis step" |
| computational property oracle | "biological truth" |

### 7.6 Claim–evidence discipline
| claim | required evidence | forbidden interpretation |
|---|---|---|
| every committed state lies in the declared molecular space | closure proof + trajectory audit | experimentally synthesizable |
| `R_θ` learns state-dependent preference | **support-matched** empirical-family baseline on held-out transitions (§5.1) | physical kinetics or synthetic-route likelihood |
| future reachability improves decisions | same-information immediate vs future-aware, same fibers | valid intermediates alone cause the improvement |
| trans-dimensionality matters | full vs **identical** size-fixed support | insertion/deletion is always necessary |
| one process supports multiple tasks | `R_θ` fixed across QED and MOO | no task-specific controller training |
| retargeting preserves progress | continuation vs restart, paired | globally optimal replanning |
| path constraints are exact | rule-closed support restriction | arbitrary history-dependent constraints without state augmentation |
| benchmark competitiveness | exact inherited protocols | matched compute / sample efficiency unless explicitly matched |

Every empirical claim names its **estimand**, the **matched comparison**, the **independent
statistical unit**, and its **limitation**. Comparator rows are **reported context**, visually and
textually separated, **never** described as what a COMPOSE number beats. **No matched-compute claim
anywhere:** the QED result is *improved task performance at a measured inference-time cost* — never
greater efficiency, and the absence of objective-specific retraining is never offered as
compensation. The current §5.3 says this explicitly: *"Internal inference costs are not matched across
methods, since the benchmark constrains the number of returned molecules rather than the successor
evaluations performed to produce them."*

### 7.7 The appendix — structure, and the open style complaint
```
A  Molecular state space and executable edits   (state space · rewrite operators · canonical successors)
B  Theoretical results  (executable-support closure · canonical successor representation ·
                         ⭐ RELATION TO GENERATOR MATCHING · finite-horizon control · retargeting & pathwise support)
C  Data and transition supervision  (molecular data · transition construction · provenance & splits)
D  Model and training details  (reference edit law · property models · QED future-value model ·
                                multi-objective future-value model · training-pipeline schematic)
E  Sampling and control  (exact finite-horizon control · molecular-scale sampling · multi-objective ·
                          retargeting & pathwise)
F  Experimental details  (F.1 reference-law · F.2 future-reachability · F.3 source-conditioned editing ·
                          F.4 multi-objective · F.5 retargeting · F.6 pathwise)
G  Benchmark and baseline details  (QED benchmark · published record · two-objective benchmark)
H  Reproducibility  (software & environment · randomization & statistics · compute)
```
**⚠️ OPEN STYLE COMPLAINT, not yet resolved.** The appendix uses a **bolded one-sentence paragraph**
pattern (a bold lead-in sentence followed by a short paragraph) throughout, and the owner's judgment
is that this reads nothing like the appendices in the reference papers, which run as **continuous
technical prose under plain subsection headings** — a heading, then paragraphs that carry their own
topic sentences without typographic scaffolding. The instruction was to go through **section by
section, piecemeal, in a principled way**, converting bold-lead fragments into ordinary prose and
letting the subsection heading do the work the bolding was doing. Two related complaints: the
subsection titles were judged **too choppy** and too numerous, and the overall effect is fragmentary.
**This pass has not been done.** It is a pure-prose task requiring no new measurements, and the
appendix has since grown to 77 KB, so it is now larger than when the complaint was raised.

### 7.8 Figures — all four are placeholder boxes
| figure | content | status |
|---|---|---|
| **Fig 1** `fig:ppp` | possible / plausible / purposeful acting on **one** successor set: (a) `x → A(x)` with 5–6 legal successors including cardinality- and topology-changing edits; (b) *the same* successors, arrow width `= R_θ(y|x)`, learned and goal-blind; (c) *the same* successors reweighted by goal `z` and budget `b`, `P_φ ∝ R_θ·h_φ`, with 2–3 future paths at right; tiny inset `z_A → x₃ → z_B` foreshadowing retargeting | **`\fbox` placeholder.** Polish this first: it should carry conceptual load the prose then need not |
| **Fig 2** `fig:mechanism` | (a) reference-law NLL ladder · (b) the pair-10 rescue trace · (c) 26/65 → 40/65 · (d) 34.6% continuation cost | placeholder |
| **Fig 3** `fig:capabilities` | (a) QED success + diversity · (b) size-fixed ablation / Pareto · (c) retargeting · (d) pathwise excursions | placeholder, 3 `\XXX` |
| **Fig 4** | reserved | appendix |
**Design brief for Fig 1:** the reference point is PepTune's overview figure — a professional,
multi-panel schematic with a clear left-to-right conceptual flow. **Do not imitate its visual
identity.** COMPOSE needs its own palette, its own panel geometry and its own typographic treatment so
the figure reads as native rather than derived. Draft images were generated in the previous session
and were not adopted; the design intent is *that level of polish, a distinct look.*

### 7.9 The gap register
**`GAP: NOT YET RUN`**
| id | unmeasured | closing run |
|---|---|---|
| **G1** | official 800-source success at k=20 | run all 800 × 20 **once** with the frozen controller; bank per-source outcomes + the curve at every k. Then replace the starred COMPOSE row in Table 1 in place, drop the `$^{*}$` clause, **keep the diversity column**, and rewrite the interpretation to what the result actually supports rather than mechanically asserting competitiveness |
| **G2** | matched paired difference between immediate and future-aware arms; n sources; both inference ledgers | `F̂_ψ` fixed, `R_θ` frozen, both arms from the frozen init bank, same preferences/seeds/budget; bootstrap **over sources**; never pool the two ledgers |
| **G3** | APS, HV, novelty, top-K diversity for future-aware, immediate, and the matched comparator rerun | the Addendum-II procedure; **never pool the published row with the matched rerun** |
| **G4** | the same four metrics at 4 objectives, 20K+5K | **gated** on the 2-obj decision rule |
| **G5** | retargeting at scale: post-switch cost vs restart, retention, n paired sources | four arms on identical sources/seeds |
| **G6** | pathwise n=48 confirmation | open the frozen panel; apply the branch rule that fires (§5.7) |
| **G7** | HV at matched total oracle budget vs independent restarts | fan one saved prefix into preference-conditioned continuations; **show the accounting** — shared prefix counted once for the fan, once per branch for restarts |
| **G8** | the further published, higher-scoring comparator (**VJTNN, 60.6%**) | verify value + protocol; confirm cohort and similarity threshold match; promote into the verified source set; **add the row**. If the protocol does not match, say so **in the caption** rather than omitting it silently |
| **—** | **size-fixed ablation** (no id yet; it is a `%% TODO` block in `experiments.tex`) | identical controller/`R_θ`/sources/allowance/randomness, remove every successor with `Δn ≠ 0`. GrIDDD's own analogous ablation: 45.1% → 33.8% |

**`GAP: NUMBER NOT IN VERIFIED SOURCE SET`**
| id | unquoted | how to close |
|---|---|---|
| **U1** | exact-control TV residual, support violations, backward residual, and the same three after retargeting | promote `diagnostics/exactness/…exact_control.json` with provenance **following the Experiment-1 schema**, or rebuild on the current registry |
| **U2** | heavy-atom cap, count of neutral (element, valence) classes, element list, fraction of benchmark leads admitted | transcribe from the scope-fingerprint artifact with its hash — declared-scope constants, not measurements |
| **U3** | controller input dim, embedding width, hidden widths, dropout, Bellman weight, epochs, patience, validation fraction, n jointly trained goal regions | transcribe from `docs/HPHI_QED_PREREGISTRATION.md` §6 with the document hash — design constants fixed **before** the data |

### 7.10 Open critiques raised but NOT yet applied
1. **The introduction still opens at the wrong level.** It begins *"Generative modeling has become a
   central approach to molecular design…"* and walks through model families. The agreed opening is
   **iterative molecular lead optimization as a workflow** — a useful molecule, successive revisable
   decisions, evaluations made *during* the trajectory, priorities that change, requirements that must
   hold throughout — and only then the missing intersection. **This is the largest remaining
   writing-level gap.** Paragraph 2 has already been enriched with exactly the right material
   (potency-vs-solubility trade-offs, new measurements introducing requirements, constraints that must
   hold throughout); the fix is largely **promoting that material into paragraph 1** and demoting the
   model-family survey. Preserve the line *"Terminal sanitization cannot make an earlier oracle query
   meaningful"* from `paper/sections/intro.tex:11` if it can be worked back in.
2. **The abstract still asserts three unmeasured results** — multi-objective competence "under fixed
   oracle budgets" (G3/G4), the future-aware improvement (G2), and Pareto-front reuse (G7). *An
   abstract asserting what the results mark unmeasured is the one inconsistency a reviewer is
   guaranteed to find.*
3. **The abstract's opening sentence pair has an agreed replacement that was never applied.** Current:
   *"…rather than a reusable representation of molecular transitions that can be learned once and
   controlled across changing design goals."* Agreed hybrid — smallest edit that makes the
   distinctive object concrete:
   > *"Yet molecular generation and optimization are typically built around fixed endpoint
   > distributions or task objectives, rather than a reusable, trans-dimensional process over complete
   > molecules that can grow, shrink, and restructure molecular graphs while keeping every committed
   > state admissible, and then be controlled across changing design goals."*

   The point: "reusable" must immediately mean something tangible — trans-dimensional ⇒ molecules can
   grow and shrink; over complete molecules ⇒ states are molecules, not masked/corrupted
   intermediates; every committed state admissible ⇒ validity is trajectory-wide, not an endpoint
   property; learned once and controlled ⇒ the reusable-process point survives.
4. **"One reusable molecular world"** still closes the abstract (§7.5).
5. **The appendix prose pass** (§7.7).
6. **All four figures are placeholders** (§7.8).
7. **The main text is ~5 pages over** (§7.11).
8. **`scripts/check_submission.sh` does not exist** though `main.tex` claims it gates the
   developmental pathwise macros.
9. **`paper_iclr2027/README.md` is stale** — old title, "9 pages / 18 total", and a gap register that
   predates §5.1, the diversity column and the restructure.

### 7.11 Page budget — now the binding mechanical constraint
Main text runs to **p.14** against a **9-page** limit. Roughly five pages must come out. In order of
least damage:
1. **§4 is 6.5 pages (pp.4–10) and is the obvious source.** 4.6 (sampling) and 4.7 (preferences)
   restate the central equation in three forms — normalized, rejection, SMC — and much of that belongs
   in `app:sampling`, which already exists.
2. Move §3 Preliminaries almost wholly to the appendix, leaving the two equations inline.
3. Move Related Work to the appendix with a one-sentence pointer (~0.25 page) — but this fights the
   deliberate §2 placement, so prefer 1–2 first.
4. Compress §5.1's third paragraph (the data-backed robustness check) into a footnote pointer.
5. Trim table captions.
**Do not shorten the `GAP` blocks to save space** — `experiments.tex` marks them load-bearing.
When G1–G7 land and space is needed for real results, the appendix already holds what came out.

### 7.12 Bibliography
41 entries. Newly added since the last handoff: `liu2018cgvae`, `you2018gcpn`, `jin2020hierg2g`,
`kim2025ddsbm`, `kajino2019mhg`, `guo2022grammar`, `madeira2024construct`, `sharma2024prodigy`,
`ruizbotella2026cocograph`, `koziarski2024rgfn`, `danos2015thermodynamic`, `behr2021rewriting`,
`wu2023twisted`, `skreta2025feynman`, `nisonoff2025guidance`, `chen2025mogdfm`, `gao2022pmo`,
`qin2025defog`, `austin2021d3pm`, `campbell2022ctdd`, `campbell2023jump`.
**Still unverified:** `niu2025inversiongnn` — the repo records only `arXiv:2503.01488`,
`github.com/ivanniu/InversionGNN`, commit `cfdf1d9`. **Author list, title and venue must be checked
against the published record.** Same for the 2026 entries (`pcomole2026`, `goel2026madsbm`,
`jensen2026valuematching`, `yoon2026routing`, `ruizbotella2026cocograph`) — confirm venue and
publication status before submission.

### 7.13 Reviewer stress tests and the prepared answers
| attack | answer |
|---|---|
| "It's just generator matching with handcrafted edits." | **It is not generator matching.** `R_θ` is fitted by the productive-successor likelihood; the hazard block is never optimized and the measure is a declared family-weighted editing measure with zero-draw strata. `app:gm` gives the exact decomposition and the two deviations. |
| "Insertion/deletion already exists." | Credited explicitly (GrIDDD, Edit Flows, jump diffusion). The claim is the typed rewrite language + molecular closure + topology change + canonical-successor kernel + finite-horizon control on that kernel. |
| "Validity by construction is tautological." | **Agreed** — it is a support property. The content is what it buys: pathwise constraints, mid-trajectory intervention, branching, zero wasted invalid oracle calls. State once, then *use* it. Validity is never reported as a metric. |
| "Doob is classical." | The contribution is the learned chemistry-native kernel on which exact, retargetable, support-preserving control becomes operationally meaningful. |
| "Future-aware control isn't new." | **Conceded in Related Work.** The distinction is the substrate: a learned canonical transition law over executable molecular graphs, which is what makes realized states reusable, retargetable and constrainable. |
| "Too many ideas." | One causal chain; each experiment tests one link — and §5 now runs in exactly that order. |
| **"Trans-dimensionality is just padded-slot bookkeeping."** | The slot array is a coordinate system; the semantic state is the active graph, and births create typed entities participating in valence, connectivity, ring perception and canonicalization. **The measurement is the size-fixed ablation and it does not exist yet — still the weakest point in the draft.** |

### 7.14 The framing paragraph to build the submission around
> COMPOSE treats molecular design as control of a learned executable process. An exact graph-rewrite
> system defines which molecular transitions are **possible**; a source-agnostic reference law `R_θ`
> learns which legal transitions are **plausible**; and a budget-conditioned controller `h_φ` favors
> the futures that serve the current goal — **purpose**. Every committed state is a complete molecular
> graph within the declared state space, so properties and constraints remain meaningful throughout
> generation.
>
> We first ask whether this formulation is useful on standard molecular-design tasks, under the
> established GrIDDD source-conditioned QED protocol and the JNK3/GSK3β multi-objective benchmark.
> These are **competence checks**.
>
> We then isolate what becomes possible *because* generation is a controlled molecular trajectory.
> Future reachability can favor an edit that is locally unimpressive but opens a better downstream
> route; a realized intermediate can be recontrolled when the objective changes; and rule-closed
> constraints can be enforced over every committed molecule rather than filtered at the endpoint. The
> central claim is not that COMPOSE is another molecular optimizer: it learns one reusable executable
> molecular process and changes the **controller** — not the molecular prior — when the purpose changes.

**The one thought the paper should leave:** *Molecular generation need not be repeated endpoint
sampling; it can be control of one learned executable molecular process.*

---

## 8. Sentence-level style
One claim per paragraph. First sentence says what the paragraph establishes; last says why it matters.
Define the object in prose before the equation. After every theorem, an operational consequence. After
every experiment, the scientific interpretation **and** the limitation. Use *we define / we derive /
we evaluate*, never *we leverage*. Adjectives only when followed by evidence. Avoid *novel, powerful,
robust, general* unless precisely bounded. **Never use internal experiment names** (P0c, K41, 4A, L4,
policy versions). **Never narrate the chronology of failed branches in the main paper.** Do not write
"we are explicit about status." Do not mention a result twice unless the second occurrence changes the
interpretation. **No bold run-in labels in Related Work; no bolded one-sentence paragraphs in the
appendix** (§7.7).

---

## 9. Things that are CLOSED — do not reopen
The **DDSBM manuscript branch** · **Policy B** as the final QED controller · the **exact-target
branch** · the **retargeting branch** (banked and complete) · **P3/P4** · **P0c** (the
full-information continuous-preference analysis — **not** a "Pareto ceiling") · **K41** ·
**MOLLEO Task 3** and all four of its search branches · arbitrary new scalarizations · extra baseline
ports · **any post-outcome controller rescue knob** (`QED^α`, temperatures, top-k, shortlist sizes,
cap tuning) · **generator matching as the stated training framework** (§2.3) · **Addendum I's `B_λ`
region construction** (withdrawn — if you are editing from a document with a 10% threshold, you are
editing the withdrawn design).

A negative result closes **the branch named by its preregistration**. It does not erase banked results
and does not authorize an unrelated rescue.

**Also barred:** reintroducing the TDC/sklearn image · `neutralize_catalog_drift()` in claim-bearing
code · quoting BARRED artifacts (early data-starved RingCore checkpoints; pre-Active8 preflight
reachability/round-trip/path-cost numbers; pre-RingCore exact-control artifacts; the `conditional_smc`
physchem box; the retracted matched-budget frontier-recovery reading). **Retracted numbers that must
never reappear:** the "6.3M" parameter count, "≈560 oracle calls", and the developmental
compiler/training figures (99,738 / 49,869 of 50,000 / 84.09→28.14 / 90.2%).

---

## 10. One-paragraph orientation, if you read nothing else
COMPOSE learns a single frozen, goal-independent stochastic process `R_θ` over **executable molecular
edits** — every state a complete molecular graph, every transition enumerated by an exact executor —
fitted by a **productive-successor likelihood** over canonical successors (**not** by generator
matching, which is a stated relationship, not the objective), and then supplies *purpose* at inference
through a budget-indexed Doob h-transform `P_φ(y|x,z,b) ∝ R_θ(y|x)·h_φ(y,z,b−1)` that can only
reweight edits the executor already permits. The learned law beats an empirical-family baseline by
**1.46 nats** on held-out transitions, **86.7%** of that from within-state successor choice.
Exactness is verified to machine precision on a 966-state enumerable slice, including exact recontrol
after a mid-trajectory goal switch. On the standard GrIDDD/Jin QED benchmark a controller frozen
before the panel was opened reaches **49.2% at k=8 and 54.7% at k=12** on a prospectively held-out
128-source panel with **0.393 diversity**, against GrIDDD's reported **45.1% at k=20 / 0.283** (VJTNN
reports 60.6% success and HierG2G 0.477 diversity, so COMPOSE beats the closest per-step editor but is
**not** SOTA on either axis); the official 800-source run has not been executed. Verified future-aware
rollout beats greedy by **+21.5 pp**, rescuing 14 of 39 failures, and `h_φ` reproduces that recovery at
**34.6%** of the continuation cost. A realized molecule retains positive value after an objective
switch under held-out replication. The live front is a matched 2-objective head-to-head against
InversionGNN where both methods get the same 10K oracle labels and diverge only in what they build
from them — **currently blocked because the multi-objective `h_φ` does not beat a trivial persistence
baseline.** The paper is drafted end to end, compiles, and is about five pages over the limit with all
four figures still placeholders.
