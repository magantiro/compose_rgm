# Held-in feasibility census — externally defined SA constraint

**Status: `DESIGN_ONLY`.** Nothing run. No Modal, no GPU, nothing installed.

**This document fixes the τ-selection rule BEFORE any measurement exists.** It is
committed ahead of the census data on purpose: the git history is the evidence
that the threshold was not chosen from the result it has to clear.

---

## 1. Is the predicate instantiable exactly as published?

> **Yes. Verbatim, with zero new scoring code.**

| element | published | ours |
|---|---|---|
| predicate | `SA(y) ≤ τ` | identical |
| thresholds | **τ ∈ {3.0, 3.5, 4.0, 4.5}** — CDD §5.2, verbatim | identical, all four |
| scorer | RDKit contrib SA_Score (Ertl & Schuffenhauer) | `src/compose_v4/eval/molecular_quality.py:13` already does `from rdkit.Contrib.SA_Score import sascorer` and exposes `"sa_score"` |
| direction | `≤` | identical |

Verified locally: `sascorer.calculateScore` returns 1.0000 for benzene, 1.5800
for aspirin, 1.9803 for ethanol. The tool works and needs nothing installed.

**Why this predicate is admissible where a smaller endogenous core was not.** The
thresholds were published before COMPOSE existed, by authors optimizing a
different process on a different dataset for their own reasons. **They cannot
have been selected because they leave our fiber enough room to act.** That is
structurally what the Bemis–Murcko branch lacked, and it is the entire basis for
this constraint being allowed where a rescue scaffold is not.

### One thing they left unpinned, and we must not inherit it

CDD **pins no RDKit version** and never names `sascorer.py`. SA values shift with
the fragment-contribution tables, so a τ = 3.0 boundary is **not
version-portable**. We pin our own version and state it. Local work here is RDKit
**2025.09.6**; the production pin is **2024.03.5**, and any claim-bearing run uses
Lane 3's pinned shared evaluator. **The census must report SA under the pinned
version**, and a version-sensitivity check on a sample of sources is cheap
insurance.

## 2. τ-selection rule — FIXED HERE, BEFORE ANY DATA

> **Primary: τ = 3.0. All four reported as a predeclared constraint-strength
> curve.**

**Why τ = 3.0 is the designated primary, on CDD's authority and not ours.** CDD's
Figure 4 caption defines its headline novelty metric at that threshold —
*"novel molecules must be valid and have no violation (τ ≤ 3.0)"*. That is the
closest thing to a designated primary in the paper, and it is theirs.

**Why reporting all four is not threshold shopping.** The project rule permits it
explicitly: *"Where a curve is genuinely the object of study, say so in advance
and report the whole curve, not its maximum."* For a **feasibility census** the
curve *is* the object — how retained support degrades as the constraint tightens
is the thing we need to know. It is declared in advance, here, and the whole
curve is reported regardless of shape.

### The honest part: τ = 3.0 is not uniformly conservative

It is the tightest of the four, and tightness **cuts both ways across our gates**:

| gate | effect of tightening τ |
|---|---|
| **nonvacuity (V3)** | **easier to pass** — more successors violate, so the constraint bites harder |
| **retained support (V4a ≥ 0.10)** | **harder to pass** — the mask removes more |
| **mask-empty (V4b ≤ 0.05)** | **harder to pass** |
| **headroom** | **harder** — less room to improve inside the feasible set |

So τ = 3.0 is conservative on the three gates whose failure would be
embarrassing, and generous on the one gate a critic would most suspect us of
gaming. That asymmetry is the right way round, but it is a **consequence** of
adopting CDD's threshold, not the reason for adopting it.

**Barred:** reporting τ = 4.5 as the headline because retention looked better
there. If the primary fails and a looser threshold passes, that is reported as
*"the constraint is feasible only at τ ≥ X"* — a finding about constraint
strength, not a rescue.

## 3. Eligibility — starting-state applicability only

A source is eligible iff:

1. **`SA(x_0) ≤ τ`** — the source already satisfies the predicate;
2. it is parseable and within the frozen heavy-atom band **[18, 38]** (reused);
3. the objective has measurable headroom at `x_0`.

**No criterion may reference a controller outcome.** No source is dropped because
its constrained trajectory behaves awkwardly. Applied identically to both arms.

### What criterion 1 makes this experiment, stated plainly

Requiring `SA(x_0) ≤ τ` makes the constraint a **maintenance** constraint —
*do not make the molecule harder to synthesize than it already is* — rather than
a **repair** constraint. This is deliberate:

- it is the realistic medicinal-chemistry framing;
- it makes the two arms comparable, since a source starting infeasible would give
  the post-hoc arm nothing to return;
- it is a property of `x_0` alone, so it cannot encode an outcome.

**The excluded fraction is reported as source applicability, not hidden.**

## 4. The objective must NOT be SA

> **If the objective is SA, the experiment is degenerate** — the constraint and
> the objective become the same quantity, and "optimize SA subject to SA ≤ τ" is
> not a constrained-optimization question.

Use an **independent** frozen objective (the existing potency/QED/cLogP goal
language). The scientific question then has its real shape:

> **maximize property P subject to staying synthesizable.**

This is also where nonvacuity most plausibly comes from: there is a genuine
chemical prior that potency optimization **increases molecular complexity**, and
complexity is what SA measures. A constraint that fights the objective is a
constraint that bites. The census measures whether it does; it does not assume it.

## 5. What the census measures

Per source, per τ, on held-in sources only.

| # | quantity | gate |
|---|---|---|
| 1 | **violation prevalence** — does unconstrained COMPOSE actually exceed τ? | **V3 ≥ 20 events** |
| 2 | **retained support** `\|F_C(x)\|/\|F(x)\|` | **V4a pooled median ≥ 0.10** |
| 3 | **mask-empty rate** | **V4b ≤ 0.05** |
| 4 | **source spread** of nonvacuous opportunity | **V5a > 1/3**; V5b single-source share ≤ 0.50 |
| 5 | **optimization headroom inside `F_C`** — is property improvement available under the mask, or does it remove everything useful? | qualitative; feeds the kill rule |
| 6 | **recoverability** — see §5a | descriptive, not a gate |

**All thresholds reused verbatim from Lane 2's frozen Stage-A2 criteria. None is
minted here. None may be widened because a τ fails.**

**Do not filter to sources that already show both satisfying and violating moves.**
Nonvacuity is *measured*, never selected. Selecting it would make the downstream
experiment true by design — the exact defect this project has caught seven times.

### 5a. One cheap extra measurement, because SA may behave unlike the scaffold

Lane 2 established that **structural** motif destruction is **absorbing**:
`recovery_rate = 0.0` over 19 breaking rollouts, which is why endpoint validity
implied path validity and the trajectory claim died for scaffolds.

**SA is a continuous score and is plausibly recoverable** — a molecule pushed
above τ by adding a complex fragment can come back down by removing it. If so,
**endpoint-only filtering would NOT imply path validity for this predicate**, and
Lane 2's G1 question is alive again for SA where it was dead for scaffolds.

This does **not** change the two-arm design. It is nearly free to collect
alongside measurement 1, and it tells the project whether a separate pathwise
claim is available for this constraint. Record `broke_and_recovered / broke` and
`endpoint_valid_path_invalid`. **Descriptive only** — no claim is made from it in
this lane.

## 6. Kill rule

> **If the census fails, the constraint dies. No rescue predicate. No threshold
> shopping.**

Fail conditions, any of which kills it:

- violation prevalence below V3 at the primary τ → **vacuous**;
- pooled median retention < 0.10 → **the mask strangles the fiber**;
- mask-empty > 0.05 → **infeasible**;
- source spread ≤ 1/3 → the opportunity is concentrated in a few sources;
- no property improvement available inside `F_C` → **the mask removes everything
  useful**.

If it fails at τ = 3.0 but passes at looser thresholds, the reportable result is
*"feasible only at τ ≥ X"*. That is a constraint-strength finding and it is
published as one. **It is not a licence to move the primary.**

## 7. The two-arm experiment this gates

> **optimize normally, then filter violations afterward** **vs** **exact
> support-constrained COMPOSE**

Same `R_θ`, source panel, objective, controller, horizon, budget, accounting.

**The soft-guidance arm is dropped.** No external method supplies a natively
calibrated guidance prescription for this predicate, so λ would be ours to
choose — and a λ chosen to make soft guidance lose is as invalid as one chosen to
make it win. Given who would be writing it, the former is the likelier error. **A
two-arm causal experiment is stronger than a three-arm one with an arbitrary
middle arm.**

If an external method later supplies a native, externally calibrated guidance
prescription that transfers without us choosing λ, the arm may be reinstated.

**Primary outcome: feasible property improvement at fixed resources.**

**Not** constraint satisfaction. The hard arm is at 100% by construction; that is
a construction check recorded without a denominator and **remains barred as a
result** (`CONSTRAINT_SEMANTICS.md` §7).

## 8. Blocker — the census needs the kernel, and Gate-0 blocks it locally

Measurements 1–6 all require the exact successor fiber, hence the production
kernel, hence the authenticated source chain.

**It fails locally, reproducibly, and I did not route around it:**

```
$ python3 docs/workstreams/constraints-hard/probes/gate0_local_authentication_repro.py \
      --local-runtime <local_runtime>
BLOCKED (expected): ProcessV2T1PanelError
  the Gate-0 PASS names another Active8/process input
```

`source_index_sha256` (`editing_v2_process_v2_gate_zero.py:593`) hashes a body
whose first field is the **absolute Active8 mount path**, so byte-identical
content mounted locally authenticates against the wrong string
(`docs/PARETO_PARITY_ENVIRONMENT_STATUS.md`). There is **no partial local path**:
the model is built *from* the authenticated source
(`retarget_intervention_app.py:185`), so no source means no model means no fiber.

**Consequence: the fiber census must run where the tree is mounted at the path
the digest names — on Modal — and that needs the lead's per-run authorization.
Nothing has been launched.**

**What runs locally without it:** the model-free half — SA distribution and
source applicability per τ. See `SA_CENSUS_APPLICABILITY.md`.

## 9. Costed plan

Basis: Lane 2's Stage A shard (`diagnostics/pathwise_constraints_smoke.json`),
the only measured cost for fiber enumeration under a mask —
**51.0 kernel calls** and **852.6 s** mean per source for 5 arms plus census.

| item | value |
|---|---|
| census scope | fiber enumeration along unconstrained rollouts, 24 held-in sources |
| **τ values** | **4** — but the fiber is enumerated **once per state** and the mask applied four times, so τ costs ~nothing extra |
| per-source | ≈ 200 s (≈ 1 arm-equivalent + masking) |
| **serial CPU** | **≈ 1.3 CPU-hours** |
| fan-out | 24 containers × 2.0 CPU |
| **wall** | **≈ 10–15 min** |
| uncertainty | **0.7×–2.5×** — Lane 2's per-source range alone spans 1.9× |

**The four τ values are nearly free**, which is what makes reporting the whole
curve the natural design rather than an indulgence.

Two-arm experiment, if the census passes: ≈ 2 arms × 24 sources × ~170 s ≈
**2.3 CPU-hours**, ≈ 10 min wall. **Cost is not the reason to hesitate.**

### Durability requirements before any launch

1. **Per-arm partial checkpointing** — `PARALLEL_WORKSTREAMS_AND_HANDOFF.md`
   records this as **UNBLOCKED and BLOCKING** for Stage B; this app ships it from
   the start.
2. Durable per-task shards written before return; aggregate reconstructible.
3. **`--detach` on any Modal run** — a server-side `drive()` does not survive
   closing the laptop. Check the State column before calling a run safe.
4. No session-specific paths.

## 10. Five questions — recorded before implementation

Per `docs/COMPARATOR_ROLES_CANONICAL.md`.

**1. Question — methodological axis.** **Hard support restriction vs endpoint
rejection**, imposing the *same externally defined* predicate on the *same*
pretrained process. Not "which makes better molecules". The axis is **where in
the process the constraint acts**: on the support during generation, or on the
output set afterward.

**2. Framework counterfactual.** Projection-based constrained diffusion — CDD,
PRODIGY, ConStruct. **Discharged conceptually**, via the trilemma: all three are
de novo, none is source-conditioned, CDD has no released code, and no native
common protocol exists (`EXTERNAL_HARD_CONSTRAINT_AUDIT.md` §4). **The causal
weight therefore falls entirely on this internal experiment**, which is why it is
primary rather than a supporting ablation.

**3. Matched causal control.** Yes — this *is* it. Hold `R_θ`, sources,
objective, controller class, horizon, executor, budget and accounting fixed; vary
only where the constraint acts. Three implementation requirements from the
Stage-0 audit, so they are not rediscovered:

- the mask **filters the enumerated successor list and renormalizes**;
  `RewriteSystem(constraints=)` **raises**, and
  `canonical_successor_result:557-567` turns any executor exception into a fatal
  error, so a predicate installed there crashes the kernel instead of filtering;
- renormalization is **mandatory** (`validate_successor_batch:297-302`);
  `SuccessorBatch.is_terminal` already covers mask-empty;
- the hard arm registers as **`SUPPORT_ABLATION`**, not `LAW_ONLY`
  (`successor_kernel.py:308`), with the mask as the single preregistered
  support-differing field.

**4. Competence comparator.** At most one or two `TASK_COMPETENCE` rows
(GraphXForm, Prompt-MolOpt^P), only where their native benchmark matches.
Neither defines the novelty claim; neither substitutes for the internal arms.

**5. Falsifier.** The claim is that **exact support restriction beats
optimize-then-filter at imposing the same hard constraint**. It dies if:

| falsifier | why it kills the claim |
|---|---|
| **post-hoc filtering matches at equal or lower total cost** | generate-and-discard is simpler; a tie on the resource frontier makes exact support unjustified complexity |
| **the mask strangles the fiber** | retention < 0.10 or frequent mask-empty |
| **the constraint is vacuous** | unconstrained control rarely exceeds τ — the shape that killed Bemis–Murcko |
| **feasible outputs are trivial** | the constrained arm returns molecules satisfying `C` by not editing anything interesting |
| **no headroom inside `F_C`** | improvement is only available outside the feasible set |

### Reporting guard rails

- The hard arm's 100% satisfaction is **definitional** — construction check, no
  denominator, never a result.
- **The measurement is what the unconstrained arm does and what the constraint
  costs.**
- **Every satisfaction rate carries its denominator.** If our feasible-output
  count falls while the violation rate improves, that is the same trade CDD's
  table hides (validity 895 → 353 at τ = 3.0). We do not reproduce in our own
  table the thing we criticized in theirs.
- Efficiency claims **name their axis**: `unique_valid_canonical_evaluations`,
  `oracle_requests`, `evaluator_calls` — never merged — plus kernel calls and
  complete trajectories.
