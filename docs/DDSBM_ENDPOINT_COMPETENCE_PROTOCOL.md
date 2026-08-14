# DDSBM ZINC logP 2→4 — endpoint-competence protocol

**Status: FROZEN. Everything below fixed before any COMPOSE outcome exists.**
Tier-1 under `docs/AMENDMENT_PUBLISHED_NUMBER_FIRST.md`: **COMPOSE runs alone and
DDSBM's Table 1 is cited as reported.**

## The question

> Can a frozen COMPOSE reference process be given a completely new, externally
> specified endpoint objective at inference time and perform credible
> source-conditioned molecular transformation?

DDSBM's task was **nowhere involved in learning `R_θ`**, which is what makes this
a test of *train once, control for a new job* rather than a benchmark score.

## The objective — a TRANSPORT adapter, not a point target

DDSBM's ZINC experiment is **distribution transport**: source logP constructed
around 2, target around 4, **same variance 0.5**, scored by Wasserstein-1 between
the generated and target **marginals**, with QED and SA measured for preservation
relative to each source.

```
τ(x₀) = logP(x₀) + 2
u(y; x₀) = − | logP(y) − τ(x₀) |
```

**Why not `−|logP − 4|`.** A point objective at 4 encourages collapse toward the
target mode, which would be a bad adapter for the very `W₁` we are asked to
reproduce. The benchmark prescribes **equal-variance distributions separated by
exactly +2 in mean**, so translating each source's logP by +2 is the natural
monotone transport between the prescribed marginals — inferred from the published
task definition, not chosen after seeing COMPOSE performance.

**Verified against the data before launch:**

| | mean | sd |
|---|---:|---:|
| source logP | 2.015 | 0.499 |
| τ = logP + 2 | 4.015 | 0.499 |
| DDSBM prescribes | 2 → 4 | 0.5 |

The shift reproduces the target marginal in **both moments**; a point target
would drive sd → 0 against a target spread of 0.5.

**Absolute error, not squared**, because the load-bearing metric is `W₁`, an
`L₁` transport metric.

**BARRED: using the CSV's randomly paired target molecule as per-source goal
information.** DDSBM builds a random initial coupling *for training*; its
scientific task is transport between marginals. Handing COMPOSE the paired test
target's logP would inject per-source information that is not intrinsic to the
transport problem.

## Frozen parameters — no sweep

| parameter | value | why fixed this way |
|---|---|---|
| **horizon** | **H = 6** | COMPOSE's established native horizon. The benchmark has no edit-count notion, so H comes from our framework **before** results rather than being optimised against their table |
| **controller** | **greedy closed-loop only** | P3 established greedy state feedback as the main mechanism, at ~22× lower cost. A boring competence experiment does not deploy the expensive controller to chase a benchmark |
| **`R_θ`** | frozen | no objective-specific parameter updates |

## Representability gate — PASSED before launch

**5,984 / 5,984 = 100.00% representable.** Zero exclusions: none unparseable,
none over the 48-slot canonical limit, none rejected by the executor.

DDSBM itself filtered ZINC for representability in its graph encoding, ending at
23,936 / 5,984. COMPOSE covers that published test set **entirely**, so this is a
**true tier-1 comparison** and no downgrade applies. Had exclusions been
necessary, source-only rules would have been frozen before generation and the
comparison downgraded rather than presented as head-to-head.

## Metrics — no cherry-picking

Report **all** of: validity, uniqueness, novelty, **logP `W₁`**, QED MAD, SA MAD,
**NSPDK**, **FCD**.

NSPDK and FCD stay in because DDSBM uses them specifically to assess whether the
generated distribution resembles the target — not merely whether logP moved.
Dropping them would reduce the comparison to the one axis we optimised.

**DDSBM's NLL remains excluded**: defined through their own reference process, so
quoting it would score each method against a different yardstick.

**Metric dependencies must not block generation.** Generate and persist the 5,984
endpoints **once**; compute RDKit metrics immediately; add NSPDK and FCD from the
**same saved endpoints** once their packages are qualified. **Never re-run
COMPOSE because a metric package was missing.**

## Uncertainty

Greedy COMPOSE is deterministic under the frozen tie rule, so **there is no
scientific reason to manufacture three "seeds."** Bootstrap over the 5,984
sources for our interval; quote DDSBM's published three-training-run mean ± SD
**separately**, and never pool the two.

### DDSBM's published ZINC values, for the comparison table

`W₁ = 0.139` · QED MAD `= 0.120` · SA MAD `= 0.402` · NSPDK `= 7.30e−4` ·
FCD `= 0.833`

## Scope

**No additional endpoint-optimization benchmark is authorized** unless this
protocol loses tier-1 status through a material representability mismatch. It has
not.
