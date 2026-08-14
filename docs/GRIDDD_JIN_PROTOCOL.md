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

## Why two tasks and not one

QED alone would demonstrate competence. **QED + LogP demonstrates modularity**,
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
