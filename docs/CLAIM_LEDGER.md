# Claim ledger — Paper 1

Every claim the manuscript makes, its status, and what would falsify or fill it. Status values:

- **PROVEN** — a theorem/proposition with a proof in the appendix.
- **BY CONSTRUCTION** — a support property enforced by the executor/enumerator; auditable, not learned.
- **MEASURED** — backed by a committed artifact under `diagnostics/` or `results/`.
- **COMPUTED** — produced by a versioned deterministic analysis with hashed inputs and implementation.
- **PENDING** — stated in the manuscript as `\XXX`; no artifact exists.
- **BARRED** — an artifact exists but may not be cited (data-starved, superseded, or inconsistent).
- **RETRACTED** — previously asserted, no traceable source, now a placeholder.

Cross-reference: `docs/PAPER1_FRAMING_AUTHORITATIVE.md` (framing), `numbers.tex` (macro inventory),
`diagnostics/coherence/program_state.json` (empirical status).

---

## 1. Formal claims

| # | Claim | Status | Location | Notes |
|---|---|---|---|---|
| F1 | Every committed state lies in the declared molecular state space | **PROVEN** | `prop:closure` (closure half) | Induction on committed events; same executor for teacher and sampler |
| F2 | The abstract untyped graph basis changes cardinality and can add or remove non-tree edges compositionally | **PROVEN (structural scope only)** | `prop:closure` (reachability half) | This is not mutual reachability of the molecular strata. Valence-feasible routes between particular typed molecules and finite-budget path cost require executor replay |
| F3 | The population minimizer of the Poisson–Bregman objective is the marginal aggregated rate | **PROVEN** | `prop:gm` | Strict convexity of `u − E[Y] log u` |
| F4 | An exactly-fitted model transports `π₀ → p_data` | **PROVEN** | `prop:gm` (endpoint half) | Requires exact fit on occupied states; not a claim about a finite network |
| F5 | Encoding invariance: successor-preserving, rate-preserving re-encodings leave the process and every fiber-constant controller unchanged | **PROVEN (with stated conditions)** | `prop:encoding`, `def:reencoding` | Adversarial check in `app:proofs`; **fails** for mark-level controllers — counterexample included |
| F6 | No controller in the control law can create support | **PROVEN** | `prop:control-closure` | Consequence: guidance cannot leave the valid molecules |
| F7 | A rule-closed predicate becomes a pathwise invariant; no oracle call lands on an infeasible candidate | **PROVEN** | `prop:control-closure` (second half) | Depends on `P` being decidable on a complete molecule |
| F8 | The finite-horizon Doob transform realizes the exact terminal tilt, and exact continuation after a mid-trajectory objective change | **PROVEN** | `thm:finite-doob` | Telescoping-`h` path-measure argument |
| F9 | Exactness is budget-specific; early stopping does not inherit the terminal tilt | **PROVEN** | `rem:exact-scope` | Marginal at `j < K` is an `h_{K−j}`-tilt |

**Deliberately not claimed:** exactness at scale · that a finite network attains the population minimizer ·
that teacher programs are optimal paths · that graph validity implies synthetic feasibility ·
path-dependent (non-terminal) rewards without a Feynman–Kac potential.

---

## 2. Support / construction claims (auditable, model-independent)

| # | Claim | Status | Source |
|---|---|---|---|
| S1 | Ring **generation** is compositional and catalog-independent (`cycle_close` / `cycle_open`) | **BY CONSTRUCTION** | `PROGRAM_CONTRACT.md` §4; model enumerators |
| S2 | Legacy whole-ring growth macro is **disabled** — masked to empty support, never enumerated/sampled/taught | **BY CONSTRUCTION** | `enable_ring_grow_macro=False`; grow family log-prob `−∞` |
| S3 | A catalog-bounded whole-ring deletion implementation exists but contributes no legal marks to Active8 | **BY CONSTRUCTION, DISABLED** | Active8 capability contract; `enable_ring_system_delete=False` |
| S4 | Ring aromaticity restatement is catalog-independent | **BY CONSTRUCTION** | `enumerate_ring_system_restate_actions` |
| S5 | Internal slot names never appear publicly | **BY CONSTRUCTION** | Deviation **D1**; verified 0 grep hits in `main.tex` / `numbers.tex` |
| S6 | Legacy preflight reports 14/14 ring-topology classes reachable | **BARRED FROM CURRENT CLAIMS** | Pre-Active8 support artifact; current reachability values remain `\XXX` pending registered replay |
| S7 | Legacy preflight reports 1,346/1,346 ring-edge round trips | **BARRED FROM CURRENT CLAIMS** | Pre-Active8 support artifact; not evidence for the current operator registry |
| S8 | Legacy preflight reports ring path-cost summaries | **BARRED FROM CURRENT CLAIMS** | Pre-Active8 task and budget selection; current path-cost values remain `\XXX` |
| S9 | Legacy primitive transport reports exact endpoint recovery on a developmental panel | **BARRED FROM CURRENT CLAIMS** | Developmental source/coupling artifact; experiment B requires the frozen Active8 corpus and checkpoint |
| S10 | Legacy whole-ring transport and teacher-fiber audits exist | **BARRED FROM CURRENT CLAIMS** | Whole-ring operators are disabled in Active8 and the audits do not establish current successor-level capability |
| S11 | Broad vocabulary admits 800/800 benchmark leads vs 33.0% for neutral C/N/O/F | **COMPUTED** | `diagnostics/composition/benchmark_lead_scope_coverage.json` (schema v2, provenance-complete support census) |
| S12 | The legacy broad-organic corpus census reports 466,483/500,000 retained, 216,183 MMP pairs, and 10,096/10,096 replayed | **BARRED FROM EDITING-V2 CLAIMS** | Valid historical data-lineage diagnostic, but not the frozen five-lane editing-V2 corpus |
| S13 | The completed 16,000-step diagnostic run used 661,105 admitted training traces and wrote 32 snapshots | **MEASURED DIAGNOSTIC, NOT MODEL SELECTION** | Frozen run manifest for `compose-v4-ringcore-v1-scientific-a7546e2-v1`; canonical-successor checkpoint selection remains pending |
| S14 | The component-factored aromatic cycle-open resolver matches an independent complete global mixed-integer oracle on the declared validation slice | **COMPUTED, VALIDATION ONLY** | `diagnostics/coherence/editing_cycle_open_global_equivalence_v2/result.08111e3173e0a262f0fe37ad9bf59283a4578571a59e21fad1904a0b0fb0cf5b.json`: 19,311/19,311 complete equivalent edge comparisons across 1,707 exact sources, zero mismatches, zero overflows. This is fixed-coordinate semantic evidence, not slot-relabeling, multistep, production-integration, or learned-capability evidence |
| S15 | Valence-state change is retained as legal support and is NOT claimed as a learned capability | **SCOPE, MEASURED** | `atom_restate:valence_state_change` holds 220 of 1,803,032 admitted train transitions (0.012%) while multivalent substrate is present in 94.7% of corpus molecules, so it is under-represented relative to opportunity. It is nevertheless **reachability-redundant**: bounded search over the production fiber reaches the sulfoxide `CS(C)=O` from `CSC` and the sulfone `CS(C)(=O)=O` from the sulfoxide, each in two `atom_insert` steps, so sulfur oxidation needs no valence restate. The operator shortens paths; it does not enlarge the reachable set. Training it to an artificial rate would misstate the learned firing law that experiment B reports |
| S16 | Ring-editing capability is evidenced ONLY from the synthetic corruption lane | **SCOPE, MEASURED** | Counts from the frozen Gate-0 decision `tests/fixtures/process_v2_gate_zero_t1_bound_decision.json` (`decision_sha256` 613259c4..., PASS, 0 violations, 1,803,032 train transitions); lane attribution from the metadata scan of all 270 train `TASK_SUMMARY.json` (2026-08-06): `cycle_insert` 122,183 transitions in 102 chunks, `cycle_attach` 84,940 in 102 and `ring_system_restate` 16,380 in 29 are ALL in `reversible_synthetic_walk`, with zero in any of the four data-backed lanes. Lanes are near family-pure and chunks are lane-homogeneous. In the pinned distribution-matched training subset (`configs/process_v2_prep_subset.json`) that lane is 23,100 of 145,756 transitions (15.8%) against 84.2% data-backed. Master plan §3 item 2 warns corrupt-and-reconstruct data carries identity bias, so ring evidence is scoped to "recovers legal ring edits under synthetic perturbation" and must NOT be reported as data-backed ring editing or scaffold hopping. `bond_reroute` and `atom_restate` each appear in two lanes and so do have a real-chemistry source; the ring operators do not. Lifting the scope requires a data-backed ring-transformation lane (MMP ring transformations / scaffold hops), which does not exist |

---

## 3. Empirical claims — status by experiment

| exp | claim | status | blocker |
|---|---|---|---|
| **A** | COMPOSE is a credible unconditional generator (validity, FCD, coverage, novelty, diversity, scaffold/size marginals, non-memorization) | **PENDING** | **No de-novo checkpoint exists.** The editing checkpoint has `denovo_weight=0` and an uncalibrated hazard (~2× teacher) and cannot carry the timed-CTMC claim |
| **A** | Matched baseline rows (SMILES AR, DiGress, DeFoG, GrIDDD, Morph, motif, valid editor) | **PENDING** | No runs; two citations unresolved |
| **B** | Learned rates beat uniform legal rewrites on transport | **PENDING** | Needs a non-starved edit checkpoint. The only two artifacts are untracked **and** marked INVALID by `diagnostics/STALE_RESULTS_MANIFEST.json` |
| **C** | Trans-dimensional adaptation: smaller/larger targets, grow-then-delete in one trajectory, held-out size bins, no-insert / no-delete ablations | **PENDING** | Needs a non-starved edit checkpoint. **This is the single most attackable claim in the paper** and is currently unmeasured |
| **D** | Topological adaptation and any eligible primitive/catalog/hybrid ring ablation | **PENDING** | Current Active8 reachability, finite-budget replay, and learned adaptation have not been measured |
| **E** | Canonical-quotient invariance: aggregate mass matches sampling, re-encoding leaves controlled law fixed, mark-level controller moves | **PENDING** | Mechanism test — runnable on *any* checkpoint, so this is the **cheapest** claim to convert from PENDING to MEASURED |
| **F** | Exact finite-horizon control on the **current** operator registry | **PENDING** | Enumerable graph must be **rebuilt**: the existing anchor used the pre-RingCore operator set |
| **F** | Pre-RingCore exact-control artifacts | **BARRED** | Superseded registry and task selections; the revised manuscript reports no number from them |
| **G** | Same-base controller table; Pareto/HV-AUC/IGD+; dynamic switching; Pareto fan; pathwise constraints; held-out oracle | **PENDING** | Needs a frozen non-starved edit checkpoint |
| **G** | Base-B mechanism artifacts | **BARRED** | Superseded CNOF base; the revised manuscript reports no number from them |

---

## 4. Barred and retracted

| item | disposition | reason |
|---|---|---|
| Early data-starved RingCore checkpoints (`bounded-e38210b`, `schedcheck-48af1bd`, `cont1500-48af1bd`, `full-6428901`, `corrected-0d2d4fe`) | **BARRED** | `DATA_STARVED_BASELINE`: 6,922 unique records vs 363,456 pool rows, ~28 passes, train −24% / held-out +8% |
| "Lower-frequency family recovery" from the step-1500 continuation | **BARRED** | Top-3/mass artifact; canonical-successor likelihood degrades 500→2000 |
| "Training improves held-out edit quality past step 500" | **BARRED** | Contradicted by the overfitting signature |
| `diagnostics/production_preflight/teacher_filter_characterization.json` (2.04% removal) | **BARRED** | Tagged `NON_SCIENTIFIC_PREFLIGHT` |
| `diagnostics/conditional_smc/physchem_box.json` | **BARRED** | Two mutually inconsistent versions on disk (tracked vs working tree) |
| `diagnostics/reachability/frontier_recovery_cap4.json` | **BARRED** | The matched-budget frontier-recovery reading was **retracted** as tuning-dependent (learnings 2026-07-22). The **barrier** diagnostic survives and is reported |
| `diagnostics/composition/composition_learned_vs_uniform*.json` | **BARRED** | Untracked **and** marked INVALID |
| Parameter count "6.3M" | **RETRACTED** | No script counts parameters; no checkpoint field; config-dependent |
| Oracle-hungry "≈560 calls" | **RETRACTED** | Prose-only provenance; committed per-lead means span 458–587 |
| Developmental compiler/training figures (99,738 / 49,869 of 50,000 / 84.09→28.14 / 90.2%) | **RETRACTED** | No artifact in repo history; the 90.2% value belongs to a different run |
| "median lead reaches 90% of gain within 13 calls" | **CORRECTED** | 13 is the **median-curve** crossing; per-lead median is 11.5, mean 35.17 |
| "≈3.8 of 20 candidates" | **CORRECTED** | The 19% rate is backed; the "of 20" denominator is not |

---

## 5. Reviewer stress tests and the prepared response

| attack | response | where it lands in the manuscript |
|---|---|---|
| "Generator Matching with handcrafted edits." | It is a new Markov-process design; the evidence is competitive generation, learned-vs-uniform transport, trans-dimensional and topological ablations, and state-level control. | §1 causal chain; experiments A–D |
| "Insertion/deletion already exists." | Credited explicitly (jump diffusion, GrIDDD, Edit Flows); the claim is the richer typed rewrite language + molecular closure + topology change + generator-matched event law + quotient kernel + dynamic control. | §1 novelty boundary; `RELATED_WORK_MATRIX.md` |
| "Validity by construction is tautological." | Agreed — it is a support property. The content is what it buys: pathwise constraints, mid-trajectory intervention, branching, zero wasted invalid oracle calls. | `prop:control-closure`; stated once in §1 and then *used* |
| "Doob is classical." | The contribution is the learned chemistry-native kernel on which exact, retargetable, support-preserving control becomes operationally meaningful. | §6; `thm:finite-doob` |
| "Too many ideas." | One causal chain; each experiment tests one link. | §1; §8 question structure |
| "Only molecular." | RGM is stated abstractly; broader reach is outlook only. | §3 opening |
| **"Trans-dimensionality is just padded-slot bookkeeping."** | The slot array is a coordinate system; the semantic state is the active graph, and births create typed entities participating in valence, connectivity, ring perception, and canonicalization. **Experiment C is the measurement, and it does not exist yet.** | §4.1; experiment C — **the weakest point in the current draft** |
