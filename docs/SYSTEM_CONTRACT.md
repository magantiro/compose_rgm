# SYSTEM_CONTRACT.md

The exact mathematical + software process the code **currently implements**, reconstructed from source
(not docstrings — those are treated as hypotheses and verified). Every claim cites `file:line`. Written
for the correctness audit preceding the universal-edit-prior training run.

Scope: the **factorized tracelet** model family (`FactorizedTraceletRateModel`) — the de-novo base "B"
and its editing fine-tune "B-edit". This is the model the Modal trainer builds and the editing
controllers sample. (The older successor-fiber `TraceletRateModel` is the semantic reference, not the
trained/sampled path.)

---

## 1. Molecular state representation

A committed state is a fixed-slot padded `MolecularGraph` (`src/compose_v4/chem/molecular_graph.py`,
padded via `chem/state.py:pad_molecular_graph`). Per state, four parallel arrays over `n_slots`:

- `atom_types[i] ∈ {0..11}` — element index (`ELEMENTS`, `molecular_graph.py:77`): `0=null (NULL_IDX)`,
  `1..10 = B,C,N,O,F,P,S,Cl,Br,I`, `11=SCAR (SCAR_IDX)` an inert deletion marker (not a real element).
- `bonds[i,j] ∈ {0..4}` — bond class (`BOND_NULL/SINGLE/DOUBLE/TRIPLE/AROMATIC`, `:326`). **Stored states
  are Kekulé** — `smiles_to_molecular_graph` calls `Chem.Kekulize(clearAromaticFlags=True)` (`:615`), so a
  live state carries integer orders 1–3; aromaticity is *perceived at read-out*, never a stored class-4.
- `formal_charges[i] ∈ {-2..2}` (`FORMAL_CHARGES`, `:310`; `NEUTRAL_CHARGE_IDX=2`).
- `implicit_h_counts[i] ∈ {0..4}` (`H_COUNT_CLASSES=5`, `:317`). H is stored, not explicit atoms.

Valence invariant (`:19`): `Σ_j order(bonds[i,j]) + implicit_h[i] − formal_charge[i] ∈ ALLOWED_VALENCES[elem]`.

**Deliberately NOT encoded** (documented limitation, `molecular_graph.py:44-47`, verified by grep — the
only hits are the docstring): stereochemistry (E/Z, R/S), isotopes, radicals, hybridization. Round-trips
lose stereo *by design*; this is a representation choice, not a bug.

## 2. Definition of a valid state

`chem/state.py:is_valid_state` + `is_connected_or_null` (connected or all-null) + per-atom valence
(`molecular_graph.py:per_atom_valence_check`), and RDKit sanitizability on the round-tripped SMILES
(`is_rdkit_valid`, `:762`; `molecular_graph_to_smiles` returns `None` on sanitize failure, never raises).
Every committed state in a trajectory is a complete, connected, graph-valid molecule.

## 3. Vocabulary (two distinct spaces — do not conflate)

- **Raw element embedding vocab** — size `M=12` incl. `null`+`SCAR`; the model's `atom_embedding`
  (`factorized_tracelet_rate_model.py:1885`). Index 2 = carbon here.
- **`(element,valence)` head class vocab** (`AtomVocabulary`, `molecular_graph.py:222`) — the space the
  prediction heads output over. **CNOF** = 4 classes `(C,4)(N,3)(O,2)(F,1)` (`:259`); **ORGANIC** = 15
  classes (CNOF + S2/S4/S6, P3/P5, Cl1, Br1, I1/I3/I5, B3) (`:262`). Index 0 = `(C,4)` here — a *different*
  ordering than the raw vocab; any cross-index use must translate.
- Ring-atom element vocab: CNOF (4) / ORGANIC (6, adds S,P) — `:218`.
- **Charge is representable in the graph (5 classes) but NOT a head dimension** — heads emit neutral
  `(element,valence)` only; `class_index(element, bond_sum, H, formal_charge)` returns `None` for a class
  it can't represent. Charged atoms are carried but **edit-protected** in the corruption
  (`rewrite/source_corruption.py`); charge/protonation edits are out of scope by construction.
- **Special tokens**: `PAD = NULL_IDX = 0` (atoms), `BOND_NULL = 0`. `SCAR = 11`. There is **no MASK / UNK /
  STAY / STOP** token; self-loops are excluded from the fiber (`factorized_fiber.py:119`), termination is
  by horizon (§11).
- Single source of truth: `ELEMENTS → ELEMENT_TO_IDX/IDX_TO_ELEMENT` and the shared `AtomVocabulary`
  object; no duplicated hard-coded index maps (verified by grep).

## 4. The legal action set A(x) — marks / operator families

The model trains and samples over **`MARK_RULE_NAMES` = 10 families** (`factorized_tracelet_rate_model.py:94`):
`atom_insert, atom_delete, atom_restate, bond_reorder, bond_reroute, cycle_insert, cycle_attach,
ring_system_grow, ring_system_delete, ring_system_restate`.

- **No standalone `bond_insert`/`bond_delete`** family — chain growth folds into `atom_insert`-with-neighbor
  (the grow head carries the bond order); ring bonds go through `ring_system_*`.
- `cycle_insert`/`cycle_attach` are **legacy null-prior ops, dead** for B and B-edit (no candidates enumerated).
- A mark is a typed `(rule_name, action)`; the action is an operator dataclass (`rewrite/operators.py`,
  `rewrite/tracelets.py`).
- The **legal fiber** at `x` is `enumerate_factorized_cnof_fiber(x, allow_bond_reroute=…)`
  (`factorized_fiber.py:93`) — the *slot-quotiented* neutral C/N/O/F action fiber; it executes every
  candidate through the real executor and drops self-loops (`successor_key == current_key`). The
  editing-only families are gated by three enumeration flags into `prepare_factorized_mark_batch`:
  `compute_ring_restates` (de-aromatization), `compute_cyclic_graft` (pendant relocation on cyclic
  molecules), `compute_ring_opening` (decoration-preserving clean ring delete).

**Capability gating (audit-critical).** Which editing families are legal at sampling MUST equal what the
checkpoint was trained with. Those three flags are sourced from the model's `enable_ring_restates /
enable_cyclic_graft / enable_ring_opening` (set at load from checkpoint metadata,
`evaluate_tracelet_rollouts.py:191`). The raw model sampler passes them (`factorized_tracelet_rate_model.py:3105`);
the pancake sampler now passes ring-restate + ring-opening (`canonical_successor_distillation.py:322,:434`),
with **cyclic graft gated off + a fail-fast guard** pending the quotient derivation (§7 / AUDIT_REPORT).

## 5. The executor T(x,a) = y

`RewriteSystem.apply(state, rule_name, action)` (`rewrite/kernel.py:51`): validates action type +
`is_valid_state(source)` + `rule.validate`, executes `rule.execute`, re-checks `is_valid_state(successor)`
and constraints. The system is `de_novo_rewrite_system() = default_rewrite_system + connected_successor_constraint`
(`kernel.py:188`). **The same executor object builds training traces** (`rewrite/trace.py:execute_trace`;
`experiments/analogue_prior.py:23,49`; `rewrite/source_corruption.py`) **and applies sampled marks**
(`experiments/tracelet_conditional.py:1195`; controller scripts `griddd_*:225`, `pareto_editing_hero.py:103`).
Executor validity is a strict **superset** of dense-mask legality (a mark can be executor-valid yet outside
the factorized mask); teacher marks must therefore come from the factorized fiber the mask represents.

## 6. Model output semantics — total exit rate + conditional mark distribution

`forward_mark_batch` (`:3016`) returns a `FactorizedMarkPrediction` with two objects (NOT interchangeable):

- **`total_hazard` = Λ_θ(x) = softplus(total_hazard_head(x)) · has_legal_mark** (`:3040`) — a **rate**
  (units: events per unit operational time), zeroed when no legal mark exists (terminal, no NaN fallback).
- **`selected_mark_log_probability` = log p_θ(mark | x)** for the batch's teacher mark (`:3030`), built as
  `family_log_prob[fam] + action_score − action_log_z[fam]` (`_selected_mark_log_probability`, `:3750`),
  i.e. `log[ p(family) · p(action | family) ]` where `family_log_prob = log_softmax(family_logits)` (`:3029`)
  and `action_log_z` is the within-family masked-logsumexp normaliser.

Per-mark **head outputs are unnormalized scores** (`delete_head`, `restate_head`, `reorder_head`,
`graft_head`, …, `:1939-2017`); they become a categorical only after `log_softmax`/`softmax`. The marked
rate for a mark is therefore `q_θ(x → mark) = Λ_θ(x) · p_θ(mark | x)`.

**Successor aggregation.** For **graft (`bond_reroute`)** the teacher/model score is a **logsumexp over the
canonical-successor group** (`graft_successor_groups`, `:3866`) — supervision is at the *molecular
successor* level, summing the marked rates of all slot-presentations of one graft. For the other families
the score is a single canonical coordinate. **Verified (measure agent):** when non-graft families place
multiple coordinates on one canonical successor (deleting either of two symmetric methyls), each scores its
*own* logit against a per-coordinate normalizer, so the coordinate probabilities **sum to the correct
successor rate — no over-count**. Graft alone attaches the full group mass to every member coordinate (by
design: GM supervises the molecular successor), which is exactly why graft — and only graft — is drawn
one-representative-per-group at sampling.

## 7. State-level generator (CTMC)

`q_θ(x,y,t) = Σ_{a : T(x,a)=y} Λ_θ(x,t)·p_θ(a|x,t)`, `q_θ(x,x,t) = −Σ_{y≠x} q_θ(x,y,t)`. Off-diagonal rates
are `≥0` (softplus·prob); the diagonal is the negative total exit rate. `selected_mark_rate` (`:679`) is
exactly `total_hazard · exp(selected_mark_log_probability)`.

**Measure the loss declares** (the algebraic-correctness hinge, §8): `p_θ(·|x)` is a proper distribution
over the declared space `{non-graft coordinates} ∪ {graft successor groups}`, so `Σ_marks Λ_θ·p_θ = Λ_θ`.
**Verified (measure agent):** `Σ exp(selected_mark_log_probability) = 1` to ~1e-7 on every molecule tested,
and the raw sampler `sample_rewrite_mark` draws successors with that same `p_θ` (bootstrap-TV PASS, incl.
neopentane's 12-fold graft group; `unseen=0`).

**Cyclic-graft quotient (RESOLVED).** The pancake sampler's graft quotient corrects the de-novo *tree*-graft
over-count via a raw-vs-quotient "survival" ratio whose raw mask is tree-gated (`:690,:714`). Cyclic graft
is **already quotiented by the general canonical successor key** (self-grafts dropped at enumeration, aliases
grouped) — the measure training normalizes graft over — so on a cyclic lead the empty raw partition falls
back to the quotient partition, giving `survival[graft]=1` and finite rates
(`canonical_successor_distillation.py`, cyclic-graft fix; verified `rate == Σ successor-group rates`, inert
on trees). **Caveat (A7, pre-existing, owner decision):** the *tree*-graft path still uses `Zr+survival`,
which does not equal training's `Zq` — a measure mismatch that does not affect B-edit editing (tree graft
never fires on cyclic leads) but should be decided before relying on the pancake sampler in a de-novo/tree
regime. De-novo generation itself uses the raw `Zq`-consistent sampler, not the pancake.

## 8. The generator-matching training target

`factorized_mark_bregman_loss` (`:4136`), per example:

```
teacher_rate r = record.path.operational_jump_rate(progress)          # factorized_mark_conditional.py:207
per_example = Λ_θ − r·( log Λ_θ + log p_θ(selected) )   if r > 0       # :4146-4151
            = Λ_θ                                        if r = 0 (terminal)
loss = mean( per_example · importance_weight )                         # :4152
```

This is the **Poisson / generalized-KL Bregman divergence** between the model's marked rate and the teacher's
single supervised marked rate: `Σ_marks[ q_θ(mark) − q_teacher(mark)·log q_θ(mark) ]` with one nonzero teacher
mark of rate `r`, using `Σ_marks q_θ(mark)=Λ_θ`. Term map: `Λ_θ`=`total_hazard` (the `Σq_θ` term, softplus,
`:3040`); `log p_θ(selected)`=`selected_mark_log_probability` (`:3030`); `r`=`teacher_rates`
(`factorized_mark_conditional.py:207`). An **illegal teacher mark → `-inf` selected log-prob → non-finite
loss** (`:3789`); this is the guard that caught the earlier charge/ring-site mask bugs. Terminal states drive
`Λ_θ → 0`.

## 9. Trajectory orientation & time

- Source `x_0` → target `x_T`. Training predicts the **forward** committed transition at the current state.
- **Frozen-time clock**: the network input time is `1 − exp(−operational_time)` for training
  (`factorized_mark_conditional.py:190`) and `1 − exp(−(t+Δ)/2)` for sampling
  (`tracelet_conditional.py:1155`, `model/time_convention.py`). Domain is `[0,1)`; larger operational time →
  frozen time → 1.
- The universal edit prior is source-agnostic in the *base* transition model: it observes `(current state x,
  time t)` only. The immutable original lead does not enter the base model; it enters the **controller**
  (`h_φ(t,x; x_src,z,b,m)`) — NOT YET IMPLEMENTED as a trained artifact.
- Editing seeds from the **provided lead** (`griddd_*:415`, `pareto_editing_hero.py:206`); the carbon-tree
  prior (`chem/source_prior.py:76`) is **unreachable** from any editing entry point (verified) — de-novo only.

## 10. The sampling law — TWO processes, one generator

- **De-novo evaluation = timed Gillespie CTMC.** `sample_tracelet_ancestral` (`tracelet_conditional.py:1124`):
  waiting time `~ Exp(Λ)` (`:1167` `rng.exponential(1/total_hazard)`), continuous `operational_time`
  advanced to `operational_horizon=7.0` in piecewise-constant `time_step=0.1` intervals (rates frozen within
  an interval), truncated correctly at the horizon; `Λ≤1e-12 → terminal`. Hazard governs timing → "CTMC" is
  literally accurate here.
- **Editing = fixed-step embedded jump chain.** The controllers (`griddd_value_guided_smc_controller.py:206`,
  `pareto_editing_hero.py:139`) loop `for step in range(max_steps)`, draw one mark from `p_θ(·|x, frozen_t)`,
  apply it, and advance a bookkeeping clock by a fixed `time_step`. **The hazard magnitude Λ is discarded**;
  only `action is None` (terminal) is read. This is the embedded jump chain of the learned CTMC over a fixed
  edit budget — a valid way to consume the generator, but it is **not** a timed CTMC. Papers must say
  "embedded jump chain over a fixed edit budget" for editing and reserve "CTMC/Gillespie" for de-novo.
- The mark distribution both processes use is `p_θ(mark)=p(family)·p(action|family)` — sampled family-then-
  action (`:3159-3193`), matching the loss factorization.

## 11. Stopping rule

No STAY/STOP action exists. A trajectory ends when: (a) the productive exit rate is `≤1e-12` (absorbing
terminal, `<TERMINAL>`, `:3194`, `canonical_successor_distillation.py:630`); or (b) a fixed budget is hit
(`max_events`/`operational_horizon` de-novo; `max_steps`/`horizon` editing). Self-loops are never committed
(excluded from the fiber). A fully-masked state yields `Λ=0` and terminates — never a NaN.

---

## Appendix A — finite-horizon Doob control for the editing embedded chain

Because editing is a fixed-`K`-step embedded chain `P_k(x,y)` (§10), exact terminal-reward steering is the
**finite-horizon Doob transform** (this is the object the controller approximates; NOT the continuous-time
generator transform):

```
h_K(x) = g(x)                                   # terminal desirability
h_k(x) = Σ_y P_k(x,y) h_{k+1}(y)                # backward value recursion
P_k^g(x,y) = P_k(x,y) · h_{k+1}(y) / h_k(x)     # controlled transition
```

Then on an enumerable state space `Pr^g(X_K=y | X_0=x_0) = Pr(X_K=y | X_0=x_0)·g(y) / h_0(x_0)` (exact
terminal reweighting), verifiable within Monte-Carlo error. Dynamic steering = replace `g` at an intermediate
step and apply the exact continuation transform over the remaining budget. Exact Doob steering **is**
available for editing — on the embedded chain, not the timed generator. **[Enumerable verification: task #35.]**

---

*Pending items (`[...]` above) are under adversarial verification (three Fable-5 agents in flight); this
contract will be finalized with their evidence in AUDIT_REPORT.md.*
