# Pathwise control — held-out confirmation, preregistration

**Status: DESIGN ONLY. NOT LAUNCHED. NO HELD-OUT SOURCE HAS BEEN OPENED.**

Everything below — the prevalence criterion, the noninferiority margin, the
sample size, the exclusions, the arms, and all three outcome consequences — is
fixed here, before any confirmation source exists. Nothing may be revised after
a held-out number is visible.

Stage B is the development record: `docs/PATHWISE_STAGE_B_RESULT.md`, Lane 2
branch `codex/compose-pathwise-constraints` at `bcc4a40`.

## Why this experiment exists

Stage B found a large, falsifiable phenomenon — endpoint-only control returned
endpoint-acceptable trajectories that had left the allowed cLogP corridor on
16/24 sources under greedy and 14/24 under verified control — and a terminal
cost whose sign the data did not resolve.

The development wording was corrected before this design was written. An
interval spanning zero is not evidence of equivalence. `Δ^G`'s interval reaches
−0.377 held-in IQR units, which would be a material potency penalty. So the
confirmation asks two questions, not one, and the second is a **noninferiority**
question against a margin fixed in advance rather than a null of zero
difference.

The corridor family was selected *after* the three-family feasibility screen.
That is what keeps Stage B developmental, and it is the specific thing a fresh
held-out panel repairs.

---

## P1 — Is hidden-path behaviour practically prevalent?

**Estimand, source-level, unchanged from Stage B:**

```
p_hidden = P( x_H ∈ C  ∧  ∃ t < H : x_t ∉ C )
```

Same frozen corridor `C = [p25, p75]` of held-in cLogP = `[2.3689, 4.4522]`,
same `H = 6`, same potency objective, same endpoint-only semantics. The source
is the independent unit; interval is a source-clustered bootstrap, 95%.

**Primary arm: `endpoint_verified`.** It is the complete COMPOSE controller
under endpoint-only enforcement, so it is the configuration the paper actually
claims about. `endpoint_greedy` is sensitivity and is reported alongside, never
substituted for the primary.

### The threshold is reused, not invented

Stage A2 fixed a source-spread feasibility criterion **before Stage B ran**:

```
V5a_source_spread : fraction of sources with ≥1 event > 0.3333   (observed 0.6667, PASS)
```

That is this project's existing, principled notion of "broad enough to matter",
and it was frozen before any control experiment. **P1 reuses `1/3` verbatim.** A
threshold derived from the 14/24 result would be a threshold chosen to be
cleared.

**One declared strengthening.** A2 applied `1/3` to the point estimate. A
confirmation must clear it with uncertainty attached, so P1 requires

```
lower bound of the 95% source-clustered bootstrap CI on p_hidden  >  1/3
```

This is strictly harder than A2's form. It is stated here as a change, in the
conservative direction, rather than presented as the same rule.

---

## P2 — What terminal price do we pay?

**Estimand:**

```
Δ^V = U_pathwise^V − U_endpoint^V           (paired, per source)
```

**Test:** noninferiority at a preregistered margin δ. P2 passes iff

```
lower bound of the 95% source-clustered bootstrap CI on mean Δ^V  >  −δ
```

### δ = 0.25 held-in DRD2 IQR units — a TOLERANCE, and only that

**Two earlier justifications have been withdrawn.** "A quarter of the held-in
IQR" is an arbitrary fraction of a spread and carries no scientific meaning.
"Smaller margins are unaffordable" is a planning fact and may never serve as a
justification. The third attempt — a medicinal-chemistry fold-change convention
— is withdrawn below for a subtler reason.

What survives is the raw-scale *interpretation*, which is worth having even
though it is not a justification. All quantities below were fixed before Stage B.

**The scale is a log-odds.** `src/compose_v4/drd2_oracle.py` emits
`margin = A·d + B = logit P(active)` under Platt scaling, with the orientation
pinned by a parity test against the original estimator rather than inferred. The
goal threshold is `POTENCY_THRESHOLD = 0.5`, i.e. **margin 0 is exactly
P(active) = 0.5**. `U` divides that by the frozen held-in IQR of 2.6161 log-odds.

So a margin in `U` units converts to a **scale-free odds ratio**, which is the
right invariant on a logit scale:

| δ (U) | log-odds | **odds ratio** | at the P=0.5 threshold | drop |
|---:|---:|---:|---:|---:|
| 0.10 | 0.262 | 1.30× | 0.500 → 0.435 | 6.5 pp |
| 0.15 | 0.392 | 1.48× | 0.500 → 0.403 | 9.7 pp |
| 0.20 | 0.523 | 1.69× | 0.500 → 0.372 | 12.8 pp |
| **0.25** | **0.654** | **1.92×** | **0.500 → 0.342** | **15.8 pp** |
| 0.30 | 0.785 | 2.19× | 0.500 → 0.313 | 18.7 pp |

#### δ is a TOLERANCE on a measurement scale, not a biological margin

An earlier draft justified δ = 0.25 as "just under the two-fold line medicinal
chemistry conventionally treats as the threshold of experimental significance."
**That justification is withdrawn.**

The two-fold convention is real — experimental IC50/EC50 measurements commonly
carry two- to three-fold reproducibility limits, and assay-specific minimum
significant ratios are estimated from replicates. But **our quantity is not an
experimental IC50.** It is the odds emitted by a classifier. Importing an
experimental fold-change convention onto a QSAR classifier's output does not
solve the scientific-margin problem; it relabels it, and it would give the
number more biological authority than the oracle can support.

**So δ = 0.25 is retained, and what it licenses is narrowed.** It is a
prespecified **tolerance on the frozen held-in DRD2 score scale** — nothing more.
The odds-ratio table above is retained as the honest *interpretation* of that
tolerance's size, not as its justification.

**Barred claim:**
> ~~"pathwise control removes hidden excursions with no meaningful potency
> loss"~~ — this asserts biological negligibility the oracle cannot license.

**Licensed claim, if P2 clears:**
> Terminal performance under pathwise enforcement remained **within a
> prespecified tolerance of 0.25 held-in DRD2-score IQR** of endpoint-only
> control.

**One limitation still stated rather than buried.** At the success threshold the
tolerance is not small: a molecule sitting exactly at P = 0.5 would be pushed to
P = 0.342, from passing to clearly failing. The logit is steepest at 0.5, so
this is the worst case rather than the typical one, but it is real.

#### What must be reported together — all four, always

1. **Continuous `Δ^V` with its 95% CI.** The estimate itself, never replaced by
   a pass/fail verdict.
2. **The primary δ = 0.25 tolerance result.**
3. **The δ = 0.20 sensitivity — MANDATORY, not optional.** It costs no
   additional outcomes to compute and is materially stricter.
4. **The number of sources near the activity threshold**, where the tolerance
   could change pass/fail status.

**The escalation of language is governed by the sensitivity, not by the
primary:**

| result | what may be said |
|---|---|
| **both 0.25 and 0.20 clear** | the cost may be described as **small** |
| **only 0.25 clears** | report exactly that. Do **not** translate it into biological noninferiority |
| **neither clears** | the cost is reported as measured |

**Why 0.20 is a mandatory sensitivity rather than the primary.** Measured, not
asserted: making δ = 0.20 primary would drop joint power at n = 48 from 0.892 to
0.788 and require **n ≈ 64** to recover it — a 33% larger panel. Running it as a
sensitivity on the same panel costs nothing and is materially stricter, so the
n = 48 design is **not** redesigned over this. The price is recorded so the
choice is visible; it is not the reason for the choice.

### Disclosure, so the margin can be audited rather than trusted

Stage B's development values, against this margin:

| Stage B contrast | 95% CI | clears −0.25? |
|---|---|---|
| `Δ^V` verified (the primary form) | [−0.182, +0.134] | **yes** |
| `Δ^G` greedy (sensitivity) | [−0.377, +0.099] | **no** |

Disclosed plainly: the development primary would have cleared this margin, and
the development sensitivity arm would have **failed** it. δ = 0.25 is therefore
not a margin reverse-engineered to pass — one of the two Stage B arms already
fails it — and it is not a margin so strict that the experiment cannot succeed.
Stage B's role in setting δ is **none**; its role is to supply the variance used
to power the panel, which is the next section.

---

## Sample size: n = 48 held-out sources

Powered by nonparametric simulation, resampling whole sources from the Stage B
plug-in population so that each synthetic source carries **both** its hidden-path
indicator and its `Δ^V` value. That preserves the observed correlation between
the two criteria (−0.319) instead of assuming independence, and it avoids a
normal approximation that `Δ^V` plainly violates — the distribution has five
exact ties and heavy tails.

Planning values, all from Stage B and used only here:

```
p_hidden = 0.5833        (endpoint_verified — the LOWER of the two arms, so the
                          primary is powered conservatively)
mean Δ^V = −0.0228,  sd = 0.3973,  n = 24
```

`scripts/pathwise_heldout_power.py`, 6,000 synthetic panels × 2,000 bootstrap
resamples each, δ = 0.25:

| n | P1 | P2 | **joint** |
|---:|---:|---:|---:|
| 24 (Stage B's size) | 0.599 | 0.798 | 0.438 |
| 36 | 0.804 | 0.924 | 0.731 |
| 40 | 0.890 | 0.948 | 0.838 |
| 44 | 0.927 | 0.964 | 0.891 |
| **48** | **0.912** | **0.976** | **0.888** |
| 52 | 0.951 | 0.984 | 0.935 |
| 56 | 0.975 | 0.989 | 0.964 |

**n = 48**, joint power ≈ 0.89.

Three things this table says that matter more than the number:

- **The binding constraint is P1, not P2.** The panel is sized by the prevalence
  criterion. P2 is comfortable at every n above 36.
- **Copying Stage B's n = 24 would have been badly underpowered** — joint 0.438,
  a coin flip. So would picking 40 because it matched the retargeting panel.
- **P1 is not monotone in n** (0.927 at 44, 0.912 at 48). That is a genuine
  lattice effect, not simulation noise: the bootstrap percentile of a binomial
  count is a step function, so the criterion flips at integer counts. 44 and 48
  are indistinguishable in power; **48 is chosen as the more conservative of the
  two**, and the non-monotonicity is disclosed rather than smoothed away.

Supply is not a constraint: **4,893** held-out reserve sources pass the size and
corridor filters after registry exclusion, so n was chosen on power alone.

P1 and P2 form an **intersection–union pair**: the main-figure verdict requires
both, so each is tested at full α with no multiplicity correction, and the panel
is sized on the *joint* rate rather than on either criterion alone.

### The margin choice, and what it costs — P2 at n = 48

| δ | P2 |
|---:|---:|
| 0.10 | 0.291 |
| 0.15 | 0.617 |
| 0.20 | 0.874 |
| **0.25** | **0.976** |
| 0.30 | 0.996 |

This table is a **planning** artifact. It records what each margin costs; it does
**not** justify the choice among them. The justification is the odds-ratio
argument above.

### P2 correctly fails if the true cost is worse than Stage B measured

| true mean cost | P2 at n = 48, δ = 0.25 |
|---:|---:|
| 0.000 (no cost at all) | 0.990 |
| −0.023 (Stage B verified) | 0.976 |
| −0.050 | 0.939 |
| **−0.105 (Stage B *greedy*)** | **0.723** |

The last row is the point of the whole design. If the true terminal penalty is
the one the greedy arm showed, P2 fails more than a quarter of the time and the
"no meaningful loss" claim does not become available. The test can lose.

---

## The direction of Δ is expected-negative but NOT guaranteed

Required by the project-wide sign-guarantee rule, and stated before any result.

The pathwise arm's feasible set is a subset of the endpoint-only arm's at steps
`0 … H−2`, and identical at step `H−1` — `_enforce_here` differs in nothing
else. Under an *exact* optimizer of the same terminal objective, `Δ ≤ 0` would
hold by construction and a two-sided test of "is there a difference" would be a
test against a false null, which this project has already caught five times.

It is **not** a guarantee here, because both controllers are heuristic: a
restricted greedy can escape a myopic trap that the unrestricted greedy walks
into. Stage B confirms this empirically — `pathwise_verified` won on 9 of 24
sources.

Two consequences, both binding:

1. Noninferiority is the correct frame precisely *because* the null of zero is
   not the interesting hypothesis. This is why P2 is not a sign test.
2. **A positive point estimate must not be reported as pathwise superiority.**
   If `Δ^V > 0` on the held-out panel, that is heuristic-controller slack, and
   it is reported as "the constrained controller was not worse", never as
   "constraining the path improves potency".

---

## Panel, exclusions, arms

**Pool.** The held-out matched validation reserve (`reserve_source_keys`,
10,653). Stage A, A2 and B all ran on **held-in** sources, so this is a genuinely
fresh population, not a re-split.

**Selection order — registry first, then applicability, then seeded shuffle.**
Identical in structure to `scripts/retarget_seal_heldout_panel.py`, for the
reason recorded there: excluding after the fact means discovering the problem
after the fact.

1. **Claim-bearing registry exclusion** — every reserve molecule already touched
   by a claim-bearing evaluation: exact-target sealed and development panels
   (sources *and* targets), the 30 retargeting calibration sources, and the 40
   retargeting held-out sources. Zero-overlap assertions against each, plus the
   held-in training universe, are checks on the filter and not the mechanism.
2. **x₀-only eligibility, reused verbatim from
   `scripts/pathwise_select_stage_b_panel.py`** — parses; heavy atoms in
   `[18, 38]`; `cLogP(x₀)` inside the frozen corridor; DRD2 margin at step zero
   `< 0` (not already potent).
3. Seeded shuffle, take the first 48. Seed distinct from A/A2/B.

**Excursion-blind, and this is the load-bearing property.** Nothing about
whether a molecule previously produced an excursion, and nothing about
post-prefix headroom, enters eligibility. Selecting on excursion propensity
would make P1 true by construction. `test_stage_b_eligibility_is_excursion_blind`
pins the signature; the confirmation selector must be pinned the same way.

**No source may be replaced** because its trajectory behaves awkwardly or its
outcome is inconvenient.

### Arms — the identical 2×2, plus one declared descriptive arm

```
                     greedy              verified
  endpoint-only      endpoint_greedy     endpoint_verified     <- mask at step H−1 only
  pathwise           pathwise_greedy     pathwise_verified     <- mask at every step
```

Both primary contrasts (P1 and P2) live in the **verified** column. The greedy
column is sensitivity for both, reported always, substituted never.

`unconstrained_potency` is carried as a **declared non-claim-bearing descriptive
arm**, as in Stage B, solely to reproduce the "how many endpoints land in the
corridor at all" statistic. It enters no contrast and no verdict.

### Handling rules, fixed now

- **Terminal-infeasible endpoint arm** (the controller reached the last step and
  no successor satisfied the constraint — Stage B saw one). Such a source has no
  legitimate comparator utility, so it **leaves P2's conditioning set** and the
  count is reported. It **stays in P1's denominator as a zero**, since there is
  no acceptable endpoint and therefore no hidden path. Both directions are
  conservative for the headline.
- **Support-tightness.** The frozen 0.10 median-retention threshold, and the
  predeclared sensitivity analysis excluding tight sources, are reused verbatim.
- **Mask integrity.** Zero violations on pathwise arms is definitional. Recorded
  as `mask_integrity: PASS`, never with a denominator, never as a finding.

---

## Predeclared consequences

| held-out outcome | consequence |
|---|---|
| P1 clears 1/3 **and** P2 clears −δ | **main Figure 6.** The bolder statement — pathwise control removes hidden excursions without a practically meaningful loss in terminal potency — becomes available with evidential backing |
| P1 clears, P2 does not | secondary, or a main-text paragraph. The prevalence finding stands; the cost is reported as measured, including if it is large |
| P1 does not clear | supplement. The pillar is not forced, and no rescue experiment is run |

`incomparable` has no meaning here — this is not a two-axis frontier verdict.
Each criterion passes or does not.

## Barred

- Any threshold, margin, or estimand revised after a held-out number is visible.
- Reporting P2 as a two-sided test against zero, or as evidence of equivalence
  from an interval that merely spans zero.
- Reporting a positive `Δ` as pathwise superiority (see the sign section).
- Substituting the greedy sensitivity arm for the verified primary in either
  question, in either direction.
- Reporting prevalence without cost, or cost without prevalence.
- Replacing a source, or extending the panel, after opening it.
- Running a rescue experiment if P1 fails.
