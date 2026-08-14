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

Lane 2 chooses the constraint predicate from its own frozen work. Lane 6's
contribution is the framing above, the injection-point requirements in §3, and
the reporting guard rails in §5.
