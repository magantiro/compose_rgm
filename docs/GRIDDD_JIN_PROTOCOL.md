# GrIDDD / Jin ZINC-250k constrained editing — protocol

**FROZEN before any COMPOSE outcome exists.** Tier-1 under
`AMENDMENT_PUBLISHED_NUMBER_FIRST.md`: **COMPOSE runs alone; GrIDDD's table is
cited as reported.** Authorized by `AMENDMENT_EDITFLOWS_GRIDDD_HEADLINE.md`.

## The question

> Can a frozen COMPOSE reference process be pointed at **two chemically
> different** inference-time objectives and perform competitive
> similarity-constrained molecular editing on an established external benchmark?

`R_θ` is frozen. **Nothing is retrained between the two tasks** — only the
inference-time objective changes. That modularity *is* the claim.

## THE INFERENCE POLICY — frozen before any QED outcome exists

COMPOSE's banked controller is greedy and deterministic; GrIDDD samples 20
candidates. The gap is closed by sampling from a **stochastic recontrol of the
frozen process**:

```
pi(y|x) = R_theta(y|x) * QED(y) / sum_z R_theta(z|x) * QED(z)      z in F(x)
```

applied in **receding fashion at each of the six committed edits**.

**Name it honestly: `stochastic one-step COMPOSE control` — a receding one-step
QED tilt of the frozen process.** It is an *exact* one-step terminal tilt
(`h_0(y) = QED(y)`, `P^g(x,y) ∝ R_θ(x,y) g(y)`), **not** the exact `H=6` Doob
controller, and must never be described as such.

Properties that made this the choice:

- **full legal successor fiber** — no support truncation
- frozen learned `R_θ` remains the reference law
- **parameter-free** — no temperature, no `β`, no beam width, no shortlist size
- stochastic, so 20 independent outputs are native
- cheap enough to run all 800 authoritative sources

### Why NOT the rollout `ĥ`, recorded with the measurement

The finite-horizon rollout `ĥ_{b-1}(y) = QED(greedy landing from y)` over the
full fiber was specified first and **measured infeasible**, at 12.8 s per kernel
call (measured, P0c source 000: 60 expansions in 766 s) and a median fiber of
606:

| `ĥ` variant | kernel calls | cost | wall @ 80 |
|---|---:|---:|---:|
| rollout, full fiber | 145,440,000 | $24,241 | **0.7 years** |
| rollout, 4/2/2 shortlist | 2,016,000 | $336 | 89 h |
| **one-step tilt (adopted)** | **96,000** | **$16** | ~4 h |

**~1,500× more kernel work** than the adopted policy. It was not casually
abandoned.

The 8-candidate shortlist variant was **rejected on scientific grounds, not
cost**: the 4/2/2 composition is an algorithmic hyperparameter designed for
*argmax search*, never as a probability-support approximation, and candidate
inclusion is already influenced by the quantity being optimized. Sampling over
it would let a reviewer ask whether the result came from COMPOSE's process or
from the shortlist construction — ambiguity we would have paid $336 to buy.

### This experiment does NOT carry the finite-horizon claim

One experiment, one question. GrIDDD asks only whether the frozen process,
recontrolled at inference, solves a standard editing task competitively.
Finite-horizon reachability is proven by exact-target/verified control; closed-
loop vs generate-and-rank by P3/P4. **GrIDDD is not asked to prove external
competence AND stochastic sampling AND planning simultaneously.**

### Frozen sampling details

1. **20 independent rollouts**, each reset to the original source, same frozen
   horizon. No continuation from another candidate.
2. **Terminal endpoints only.** No intermediate molecule enters the candidate
   set.
3. **Frozen seed manifest**, committed before outcomes:
   `seed = uint64(sha256(protocol_version + task + canonical_source + replicate)[:8])`.
   **Never Python's process-salted `hash()`.**
4. **Replicates 0–19, duplicates consume attempts.** No resampling until 20
   unique molecules appear.
5. **One panel. No seed shopping.** Uncertainty by bootstrapping the 800
   source-level outcomes, never by raising the output budget.
6. **Similarity is evaluation-only** — `≥0.4` is not part of `π`.
7. **No success-triggered early stopping.**
8. **Failures remain failures** — dead ends, invalid, duplicate, and
   below-threshold candidates stay in the denominator.
9. **Zero-denominator fallback, preregistered:** if `Σ R_θ·QED == 0`, fall back
   to `R_θ` itself. Not a constant invented after it happens.

### The size-fixed ablation

**Identical policy and identical seeds.** The only intervention: successors
changing heavy-atom count are removed **before** normalization.

## ⬇️ AMENDED — penalized LogP is PARKED, not failed

The two-task plan is **withdrawn**. QED is a bounded non-negative terminal
potential, which is exactly what makes the parameter-free law above possible.
Penalized logP is unbounded real and would need `g_β(y) = exp(β·f(y))` — **a new
controller hyperparameter introduced solely to obtain a second table row.**

> The published benchmark contains additional objectives; the primary
> comparison uses QED because it admits a parameter-free stochastic terminal
> potential under the frozen COMPOSE controller. Extending to unbounded
> utilities requires a separately specified utility-to-potential map and is
> outside this benchmark.

If a **general** utility-to-positive-potential calibration is ever developed for
the method, logP returns naturally. It must not be invented to fill Table 3.

### (superseded) the original two-task rationale

QED alone would demonstrate competence. QED + LogP would have demonstrated modularity,
at almost no extra conceptual cost: same benchmark, same data, same evaluation
machinery, same 800×20 structure. The two properties are chemically different
and both are **cheap deterministic RDKit calculations**.

**DRD2 is excluded, deliberately.** It uses a *learned activity oracle*, where
the budget asymmetry between GrIDDD's 20 sampled candidates and COMPOSE's
closed-loop successor inspection becomes a serious fairness problem. QED and
LogP avoid that entirely. **This exclusion is frozen now, before any result.**

## Terminology — this benchmark is NOT multiobjective

GrIDDD's suite is **three separate constrained single-objective editing tasks**,
each subject to a similarity floor. It is *not* simultaneous multi-objective
optimization and must never be described as such.

- **"general-purpose editor pointed at different objectives"** → yes, this is
  that demonstration.
- **"Pareto / several objectives at once"** → **not this benchmark.** That is
  our preference-continuum work, which GrIDDD does not adjudicate.

## The protocol, taken verbatim from GrIDDD

| | **LogP** | **QED** |
|---|---|---|
| sources | **800** lowest-logP test molecules | **800** test molecules with QED ∈ **[0.7, 0.8]** |
| candidates per source | **20** | **20** |
| similarity | Tanimoto **≥ 0.4** and **≥ 0.6** (reported separately) | Tanimoto **≥ 0.4** |
| reported quantity | **best improvement** under each threshold | **success rate**: ≥1 candidate reaching QED ∈ **[0.9, 1.0]** |

Source selection, fingerprints, property calculations, candidate count and
aggregation must be **identical** or this is not a tier-1 comparison.

## The published table — cited, never recomputed

GrIDDD, **NeurIPS 2025 poster**. These are their reported values.

| method | LogP ≥0.4 | LogP ≥0.6 | QED success |
|---|---:|---:|---:|
| JT-VAE | 1.03 | 0.28 | 8.8 % |
| CG-VAE | 0.61 | 0.25 | 4.8 % |
| GCPN | 2.49 | 0.79 | 9.4 % |
| **GrIDDD** | **2.70** | **1.33** | **45.1 %** |
| **COMPOSE** | *ours* | *ours* | *ours* |

**JT-VAE, CG-VAE and GCPN are inherited benchmark context, not chosen
comparators.** They come free with the benchmark. Do **not** spend paper energy
on them individually, and do **not** go hunting newer optimizers merely because
they appear here. **GrIDDD is the modern row that matters.**

## The mandatory caveat — task performance, not oracle efficiency

> **This establishes TASK PERFORMANCE on an identical protocol. It is NOT a
> matched computational or oracle budget.**

GrIDDD samples 20 candidates after conditioning; COMPOSE's closed-loop
controller may inspect far more successors internally. **Both resource ledgers
are reported separately and never pooled**, exactly as in P3/P4. Claiming
matched oracle calls here would be false.

## What counts as a strong result

**We do not need to beat GrIDDD.** The bar is credible competitive performance
on an externally defined task with **zero objective-specific parameter updates**
across two chemically different properties, while every intermediate state stays
valid.

If COMPOSE wins: strong. If comparable: still strong for train-once/control-later.
If materially worse: **report it, identify which axis — improvement magnitude or
similarity retention — is responsible, and do not modify the objective.**

## Precedent — why this table is sufficient

GrIDDD's own accepted evaluation is disciplined, not a zoo: DiGress and
FreeGress for the nearby generative regime, a **three-row** JT-VAE/CG-VAE/GCPN
table for standard editing, **one** newer appendix comparator (EDM-SyCo, with
its protocol difference explicitly acknowledged), and then internal ablations
carrying the actual contribution.

Our structure matches: **Edit Flows** for the methodological neighbor, **GrIDDD**
for native molecular numbers, **matched internal experiments** for the mechanism.
A 12-model suite would look *less* like the accepted precedent, not more.

## Barred

Retraining or reimplementing GrIDDD, JT-VAE, CG-VAE or GCPN · adding baselines
to lengthen the table · DRD2 · describing this benchmark as multiobjective ·
claiming matched oracle budgets · tuning the objective after seeing a result.

## Record: one stray artifact, unread

`griddd_qed/full_0004.json.gz` exists on the volume. It is the 4-source
mechanical smoke that completed before its stop took effect, and it contains
**official test sources 0–3**.

**It has not been read and will not be.** It is left in place rather than
deleted — removing data to tidy a record is worse than labelling it.

It consumes nothing. Seeds are frozen and deterministic, so those four sources
will produce **bit-identical** endpoints whenever the official run happens;
reading it now versus reading the official run later is the same information.
The file is superseded by `full_0800x20.json.gz` when that exists, and must
never be reported.

### Correction to the record above

Only **one** source completed, not four — the stop caught the other three
mid-flight. Established by reading the record **count only**; no endpoint, QED,
similarity or qualification value was inspected.

So precisely: **1 of 800 official sources was executed, 799 were never run, and
the official set is INFERENTIALLY UNCONSUMED — not literally never executed.**
Quarantined at `docs/QUARANTINE_PRE_FREEZE_SMOKE.json`, sha256
`8545840839389b7e6156ecdce1a250d075f6f808fe6a0c3fd06fdd6f88294840`, marked
`PRE_FREEZE_SMOKE_DO_NOT_USE`. Never read, never merged; rerun from scratch
with all 800 once the policy freezes.

## How development policies are judged — frozen before reading the dev panel

**Primary development endpoint: the exact per-trajectory benchmark event.**

```
1[ QED(y) >= 0.9  AND  Tanimoto(y, x0) >= 0.4 ]
```

over all **320** trajectories (64 sources × 5 replicates), analysed with
**source clustering**, and **paired on (source, seed)** when two policies are
compared. Identical seeds make that pairing exact.

**Secondary, always shown SEPARATELY: terminal QED and Tanimoto.** Reporting
them apart is what stops a policy from "improving" by raising QED while
quietly destroying similarity.

> **No combined scalar score is invented.** Not `QED − λ·(1−sim)`, not a
> weighted product. The benchmark event is a conjunction; keep it one.

**The best-of-5 source success rate is DESCRIPTIVE ONLY.** It may be reported;
policy selection must not be driven by 64 noisy binary source outcomes when
320 paired per-trajectory observations are available.

### The path to the official set

```
64 x 5 dev  →  iterate principled variants  →  choose ONE
            →  128 x 20 FRESH disjoint validation panel
            →  freeze  →  official 800 x 20, ONCE
```

The second panel must be **freshly drawn and disjoint from both the first panel
and the official sources**. That is what licenses aggressive learning from
these 64 molecules without overfitting the final controller to them. The
eligible pool is **66,696**, so untouched development material is not scarce.

**Barred during iteration:** `QED^α` sweeps, temperatures, arbitrary shortlist
sizes, or any knob tuned until the benchmark-shaped metric rises. Permitted:
controller variants statable independently of their result.

## ⭐ REVISION — the anytime stopping time is the headline, not fixed-H6

**Earlier framing was over-anchored on mimicking GrIDDD's inference.** The
correct principle:

> **Match the task and the number of returned candidates. Do NOT force different
> methods to share internal trajectory semantics.**

The published task says: per source, generate **20 candidates**; the source
succeeds if **at least one** has QED ≥ 0.9 and Tanimoto ≥ 0.4. Nothing in that
definition requires another method's candidate to be the state after a fixed
number of *its own* internal operations.

### The native COMPOSE policy — an online stopping time

```
τ_z = min{ t ≤ H : g_z(X_t) = 1 }          return X_τ if reached, else X_H
```

One source → one stochastic trajectory → **one returned molecule**, max six
edits, stopping as soon as the predeclared region is reached. Run it 20 times →
**20 candidates. Not 120.**

**This is a policy, not retrospective selection**, and the distinction is the
whole argument:

| | |
|---|---|
| ⛔ **questionable** | run six edits, inspect all six afterwards, return whichever scored best |
| ✅ **clean** | at `x_t`, observe the predeclared goal is satisfied, execute `STOP` |

The clean version never consults a future state to decide whether an earlier one
was better. **First qualifying state. Period.** *"Step 3 had QED 0.889 but was
prettier"* is barred, and so is any use of later states to re-rank earlier ones.

**`STOP` belongs in an executable molecular controller.** Why would a
lead-optimization algorithm be required to keep modifying a molecule that
already satisfies the requested specification?

### Why this is exploiting the representation, not gaming it

GrIDDD's internal states are noisy diffusion intermediates — part of its
denoising computation, not candidate molecules. **COMPOSE was engineered so that
every committed state is already a complete candidate.** Discarding a successful
intermediate purely because another method cannot expose one would remove a real
COMPOSE advantage to make the algorithms look superficially alike.

### The three-row presentation — the objection becomes evidence

| method | inference | returned/source | QED success |
|---|---|---|---|
| **GrIDDD** | property-conditioned diffusion | 20 | **45.1 % reported** |
| **COMPOSE — fixed H6** *(ablation)* | region `h_φ`, forced six edits | 20 | — |
| **COMPOSE — native anytime** *(headline)* | region `h_φ`, STOP on first hit, max H6 | 20 | — |

**The native number is the primary COMPOSE result.** Fixed-H6 is the
conservative ablation answering *how much comes specifically from anytime
execution?*

If anytime ≫ fixed-H6, the conclusion is **not** "COMPOSE peeked at
intermediates." It is: *because COMPOSE evolves through evaluable molecules, it
can terminate once the specification is satisfied; forcing it to continue
editing destroys successful leads.* **That is a result about the architecture.**
If they are close, stopping did not matter here. Either outcome is informative.

### The only wording that changes

> same Jin/GrIDDD source cohort, same QED/similarity success criterion, same
> allowance of **20 returned candidates per source**; **inference algorithms
> differ natively by method.**

**Not** "identical inference protocol."

### Consequence for `h_φ`

The claim-bearing native controller targets **first-passage** reachability:

```
h_hit_b(x,z) = P_{R_θ}( ∃ t ≤ b : X_t ∈ B_z | X_0 = x )     with h = 1 on B_z
```

More natural for optimization than *"what is the probability I am inside the
region exactly six edits from now?"* Equivalently, terminal-Doob on an
**absorbed** chain.

**The corpus stores both**, so no rerun is needed: `terminal_success`,
`ever_hit`, and the **first-hit step**. Fixed-horizon terminal remains the
matched ablation target.

### What the census must therefore report, per region

- terminal-at-H6 prevalence
- **hit-by-H6 prevalence**
- distribution of **first-hit step**
- **fraction of hits subsequently LOST by H6**

That last number is the scientifically interesting one: if many trajectories
reach `0.90/0.40` and then fall back out, forced fixed-H6 semantics are actively
wasting COMPOSE's strongest structural feature.
