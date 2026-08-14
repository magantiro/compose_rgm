# Five questions — internal three-arm hard-constraint comparison

**Handoff artifact for Lane 2.** Lane 2 **builds** this experiment; Lane 6
supplies the framing and does not design or run it.

**Status: `DESIGN_ONLY`.** Recorded **before any implementation**, as
`docs/COMPARATOR_ROLES_CANONICAL.md` requires. Nothing here has been run.

**Experiment:** post-hoc filtering vs soft guidance vs exact support restriction.
**Role:** `MATCHED_CAUSAL_CONTROL` — **the primary comparison for the constraints
block.**

---

## 1. Question — what methodological axis?

> **Hard support restriction versus soft guidance versus endpoint rejection**, as
> three ways of imposing the *same* constraint on the *same* pretrained process.

*Not* "which method makes better molecules." The axis is **where in the
generative process the constraint acts**:

| arm | where the constraint acts | what the process sees |
|---|---|---|
| **post-hoc filter** | **after** generation | nothing — the process is unconstrained and infeasible outputs are discarded |
| **soft guidance** | **during**, as a penalty on the law | a reweighted distribution over the **full** legal support |
| **exact support restriction** | **during**, on the support itself | `F_C(x)` — infeasible successors are not candidates |

All three impose the identical predicate `C`. Only the **mechanism** differs.
That is what makes it a causal control rather than a benchmark.

## 2. Framework counterfactual

The nearest alternative generative abstraction that changes *this* axis is
**projection-based constrained diffusion** — CDD, PRODIGY, ConStruct.

**It cannot supply a number.** `EXTERNAL_HARD_CONSTRAINT_AUDIT.md` §4: all three
are de novo, none is source-conditioned, and a full-text search of PRODIGY and
ConStruct for `inpaint | scaffold | lead optim | editing | source molecule |
given molecule` returns zero relevant hits. CDD has no released code.

**So the framework counterfactual is discharged conceptually**, by the trilemma
(`CONSTRAINTS_SECTION_TRILEMMA.md`), and **the causal weight falls entirely on
this internal experiment.** That is the reason it is the primary comparison and
not a supporting ablation.

## 3. Matched causal control — can it be isolated more cleanly internally?

**Yes, and that is exactly what this is.** Hold fixed: `R_θ`; the source panel;
the objective; the controller class; the horizon; the legal executor; the oracle
budget; and the resource accounting. Vary **only** the constraint mechanism.

Three implementation requirements Lane 6's Stage-0 audit established, which Lane
2 should not have to rediscover:

1. **The mask must filter the enumerated successor list and renormalize.**
   `RewriteSystem(constraints=)` **raises**, and
   `production_successor_kernel.canonical_successor_result:557-567` converts any
   executor exception into a fatal `ProductionSuccessorKernelError`. A predicate
   installed there crashes the kernel on the first violating mark instead of
   filtering it — and blinds a guard that exists to catch genuine model/executor
   mask disagreement. See `CONSTRAINT_SEMANTICS.md` §5.
2. **Renormalization is mandatory.** `validate_successor_batch:297-302` rejects a
   batch whose successor probabilities do not sum to 1.
   `SuccessorBatch.is_terminal` already represents the mask-empty case.
3. **The hard arm is a `SUPPORT_ABLATION`, not `LAW_ONLY`**
   (`successor_kernel.py:308`), and the mask is the single preregistered
   support-differing field. The **soft-guidance arm is `LAW_ONLY`** — same
   support, reweighted law. Registering either wrongly would misstate what the
   experiment controls.

`src/compose_v4/experiments/constraints_hard_mask.py` implements the filter and
renormalization with its adversarial fixtures; reuse or discard it as Lane 2
prefers, but the three requirements above hold either way.

## 4. Competence comparator — at most one or two

**GraphXForm** and **Prompt-MolOpt^P**, and only where their native benchmark
matches. Both are `TASK_COMPETENCE`: they establish that COMPOSE performs
credibly on ordinary molecular design. **Neither defines the novelty claim**, and
GraphXForm's hard-constraint cell is `N/A` — it has no user-specified
preservation and an add-only action space.

**Do not add a third.** Do not let either substitute for arm 3 above.

## 5. Falsifier — what kills the claim?

The claim is that **exact support restriction is materially better than soft
guidance or post-hoc rejection at imposing the same hard constraint.** It dies
if any of these holds:

| falsifier | why it kills the claim |
|---|---|
| **soft guidance matches hard support on feasible performance** | named as a live falsifier in the canonical policy. If a penalty gets you the same feasible yield and quality, the exactness is not buying anything |
| **post-hoc filtering matches at equal or lower total cost** | the "generate and discard" strategy is simpler; if it ties on the resource frontier that matters, exact support is unjustified complexity |
| **the mask strangles the fiber** | retention below the frozen floor, or frequent mask-empty states — the constraint is infeasible rather than useful |
| **the constraint is vacuous** | unconstrained control rarely violates `C`, so there is nothing to enforce. **This is the shape that already killed the Bemis–Murcko scaffold**, and ConStruct's Appendix G.1 planarity result is the same failure in the published literature |
| **feasible outputs are trivial** | the constrained arm returns molecules that satisfy `C` by not editing anything interesting |

### Two guard rails on how the result may be reported

**The hard arm's 100% constraint satisfaction is true by construction and is
never an empirical result.** Record it as a construction check, without a
denominator. This is the project sign-guarantee rule, which has caught seven
prior instances; the mathematical reason and the adversarial fixture are in
`CONSTRAINT_SEMANTICS.md` §7 and `tests/test_hard_scaffold_constraint.py`.

**The measurement is what the *unconstrained* and *soft* arms do, and what the
constraint costs.**

**State the denominator on every satisfaction rate.** The soft arm's satisfaction
is a genuine measurement, and if its feasible-output count falls while its
violation rate improves, that is the **same** trade CDD's table hides — validity
895 → 353 at τ = 3.0, with violations reported over valid molecules only. We
should not reproduce in our own table the thing we criticized in theirs.

---

## Efficiency-axis discipline

Any efficiency comparison between these arms **names its axis**. The three
counters are never merged:

- `unique_valid_canonical_evaluations` — benchmark-native, PMO-comparable
- `oracle_requests` — algorithmic demand, including duplicates and rejects
- `evaluator_calls` — expensive executions after caching

Plus kernel calls and complete trajectories. Post-hoc filtering will look cheap
on one axis and expensive on another; that is the point of the experiment, and
selecting the flattering axis would void it.

## What Lane 6 is NOT handing over

No panel, no arms implementation, no cost model, no launch plan. The scaffold
protocol that would have supplied those is **`SUPERSEDED`**, and its protected
object failed its feasibility census. **The Bemis–Murcko branch stays killed —
no "smaller scaffold" rescue**, per the canonical policy's frozen decisions.

Lane 6's contribution is the framing above, the injection-point requirements in
§3, and the reporting guard rails in §5.

---

## Open decisions Lane 2 must make — named, not papered over

This handoff is **not** self-sufficient. Four things are genuinely undecided, and
a builder would otherwise have to come back and ask. Each is Lane 2's to settle,
but the constraints on settling them are recorded here so the answer does not
have to be renegotiated.

### O1 — The soft-guidance arm has no definition yet. **This is the biggest gap.**

The arm is named in §1 but its functional form is not specified. It needs:

- a **reweighting of `R_θ` over the full legal support** — the natural family is
  `R_θ(y|x) · exp(−λ · penalty(y))` renormalized, with `penalty(y) = 0` when
  `C(y) = 1`, but the exact form is Lane 2's call;
- registration as **`LAW_ONLY`** (`successor_kernel.py:308`) — same support,
  different law. This is what makes it the correct middle arm: it differs from
  the hard arm in *mechanism*, not in support.

**Binding constraint on λ.** The project rule *"no hyperparameter search for a
favourable operating point"* applies with full force. **Pre-register ONE λ from a
stated requirement** — a target violation rate, a target KL from the unguided
law, a structural argument. Running `λ ∈ {0.1, 1, 10, 100}` and reporting the
best is choosing the result and then naming a configuration for it. If the curve
is genuinely the object of study, say so in advance and report the whole curve.

A λ chosen to make soft guidance lose would be as invalid as one chosen to make
it win, and it is the more likely error here.

### O2 — Which predicate `C`?

Two options, and they are **not** equivalent in evidential value:

| option | provenance | risk |
|---|---|---|
| **CDD's `SA(y) ≤ τ`**, τ ∈ {3.0, 3.5, 4.0, 4.5} | **exogenous** — fixed by other authors before they had heard of us | **none of the outcome-selection kind** |
| Lane 2's own frozen corridor | endogenous | the predicate was chosen against our own corpus |

**Recommend the exogenous one.** An externally fixed threshold structurally
cannot repeat the defect that killed the Bemis–Murcko branch — a constraint
selected, however innocently, because it leaves enough room to act. The three
verbatim-reproduction conditions are in `BASELINE_TASK_MATRIX.md` §3a; the one
easy to miss is **pin our own RDKit version and state it**, because CDD pins none
and SA is not version-portable.

If none of the four thresholds is workable on our fiber, **that is a reportable
result**, not licence to pick a fifth.

### O3 — Gate thresholds

Reuse Lane 2's own frozen Stage-A2 criteria verbatim — V3 ≥ 20 events, V4a pooled
median retention ≥ 0.10, V4b mask-empty ≤ 0.05, V5a source spread > 1/3, V5b
single-source share ≤ 0.50. **Do not mint new ones**, and do not widen them
because a chosen predicate fails.

### O4 — Panel, horizon, budget, cost model

Entirely Lane 2's. Lane 6's cost anchor (~170 s per arm-source, from Lane 2's own
Stage A shard) is in the superseded `PROTOCOL.md` §11 if it is useful, but it was
measured for a different predicate and should be re-derived.

---

## Self-assessment of this handoff

**Sufficient without further questions:** the methodological axis; why the
framework counterfactual is discharged conceptually; the three injection-point
requirements; the arm-registration vocabulary (`SUPPORT_ABLATION` vs `LAW_ONLY`);
the five falsifiers; the construction-check rule; the denominator rule; the
three-counter efficiency discipline.

**Requires a Lane 2 decision before implementation:** O1–O4 above. **O1 is
load-bearing** — the soft-guidance arm is one of the three arms and one of the
named falsifiers (*"soft guidance matches hard support on feasible
performance"*), so the experiment cannot be built without it, and a badly chosen
λ would void the comparison rather than merely weaken it.
