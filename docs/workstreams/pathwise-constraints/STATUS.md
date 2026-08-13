# Workstream C — Pathwise Constraints: Status

- **Status:** Stage B **DESIGNED AND COMMITTED — NOT LAUNCHED**
- **What is running:** **NOTHING.** No Modal run is open.
- **Held-out data opened:** **NO.** No held-out confirmation is designed.

## Stage B — frozen, awaiting launch authorisation

**Question:** when terminally acceptable trajectories can pass through
forbidden intermediate states, what is the cost and benefit of enforcing the
constraint throughout molecular evolution?

**Task:** increase DRD2 potency subject to `2.3689 ≤ cLogP(x_t) ≤ 4.4522` for
all `t`, `H=6`. Objective frozen from the retargeting lane; no objective search.

**2×2, one code path:**

| | greedy | verified |
|---|---|---|
| enforce at `t=H` | `endpoint_greedy` | `endpoint_verified` |
| enforce at every `t` | `pathwise_greedy` | `pathwise_verified` |

plus `unconstrained_potency`, **descriptive only**. No fifth causal arm.

**Estimands:**

- **PRIMARY** hidden-path RATE `P(x_H ∈ C AND ∃t<H: x_t ∉ C)` — denominator is
  every eligible source, cannot collapse.
- **SECONDARY** hidden-path FRACTION `P(∃t<H: x_t ∉ C | x_H ∈ C)` — intuitive,
  but **controller-dependent denominator**, always reported with it.
- Both reported **separately for greedy and verified**, never pooled.
- **Terminal cost** `Δ^G`, `Δ^V` under controller parity; sign is free, and
  three outcomes are declared informative in advance — none is a failure.

**Panel:** 24 new held-in sources, `panel_sha256 2d993508…`, seed 20260815,
disjoint from all 48 previously used sources. Eligibility is **excursion-blind**.

**Support-tight predeclared:** median retention < 0.10 (the existing viability
number). All 24 stay in the primary ITT; sensitivity excluding them is
secondary; the threshold is frozen.

**Cost: ≈12–15 container-hours**, ~45 min wall — **9–11× the A2 spend**, the
largest run in this lane. Source-sharded and resumable, so it can be authorised
in halves.

## Prior stages

| Stage | Verdict | State |
|---|---|---|
| Ring-system stage A | **G1 FAIL** | CLOSED, permanently, not revised |
| Three-family reversibility census | **NO FAMILY PASSES** | CLOSED |
| Stage A2 — cLogP corridor prevalence | **PASS (5/5)** | complete |
| **Stage B — corridor-constrained potency** | — | **committed, not launched** |

## Stage A2 verdict — PASS

| # | Criterion | Threshold | Observed | |
|---|---|---:|---:|:--:|
| **V3** | event yield | ≥ 20 | **25** | PASS |
| **V4a** | median support retention | ≥ 0.10 | **0.573** | PASS |
| **V4b** | mask-empty fraction | ≤ 0.05 | **0.0038** (1/261) | PASS |
| **V5a** | source spread | ≥ 0.333 | **0.667** (8/12) | PASS |
| **V5b** | largest single-source share | ≤ 0.50 | **0.24** | PASS |

**Source spread is the headline: 8 of 12 sources produced at least one
endpoint-valid / path-invalid event**, source-clustered bootstrap 95% CI
**[0.417, 0.917]** — the criterion holds at the lower bound, not just at the
point estimate.

Trajectory level (72 trajectories, **12 observations**): violation incidence
0.597 (43/72); return rate among violators **0.581** (25/43).

Excursions are substantial, not boundary grazes: median max depth **0.693**
logP, median duration **4.0 of 6 steps** outside the corridor — real budget
spent in a forbidden region, invisible to endpoint-only filtering.

**V4 was evaluable for the first time.** The corridor mask keeps a median
**57.3%** of legal successors and emptied the support at exactly 1 of 261
states.

## Current position

| Stage | Verdict | State |
|---|---|---|
| Ring-system stage A | **G1 FAIL** | CLOSED, permanently, not revised |
| Three-family reversibility census | **NO FAMILY PASSES** | CLOSED |
| **Stage A2 — cLogP corridor prevalence** | **PASS (5/5)** | complete |

### Honest framing, carried in the protocol and emitted in the analysis JSON

> Family B was selected for follow-up AFTER the three-family feasibility census
> because it alone exhibited the intended reversible-excursion mechanism. Stage
> A2 is developmental follow-up, not independent confirmation of the
> phenomenon.

What A2 adds beyond stage A: prevalence on **new** sources, spread across
**distinct molecules**, and **V4 mask viability** — none of which existed
before. What it does not add: independent evidence that cLogP corridors are
special, since B was chosen for showing the effect.

### Cost

12 sources, **≈1.33 container-hours** against a 2.1 h estimate (63%). 21.75
kernel calls/source mean (estimate 35), 400 s/source mean. Circuit breaker
never approached. Launched `--detach`, verified `ephemeral (detached)`, 12/12
shards committed, 0 void.

---

## Closed: reversibility census — no family passes

## Reversibility census (2026-08-13, no new compute)

Families, thresholds, rank order and pass criteria were sealed in commit
`bf14d53` **before** the measuring script was written.

| Family | violators | returned | return rate | events | V1 | V2 | V3 | verdict | failure mode |
|---|---:|---:|---:|---:|:--:|:--:|:--:|---|---|
| **A** undesired motif | 1/35 | 1 | 1.000 | **1** | ✓ | ✓ | ✗ | FAIL | underpowered |
| **B** cLogP corridor | 7/21 | 6 | **0.857** | **6** | ✓ | ✓ | ✗ | FAIL | underpowered |
| **C** size corridor | 0/42 | 0 | 0.000 | 0 | ✗ | ✗ | ✗ | FAIL | **vacuous** |

Criteria: V1 non-vacuous · V2 return rate ≥ 0.10 · V3 ≥ 20 endpoint-valid /
path-invalid events · V4 mask leaves room (**not evaluable from shards**).

**The failure modes differ and that difference is material.** C is vacuous —
heavy-atom count never left the frozen band, so it is dead for a scientific
reason, like the ring system. A and B are **reversible** (return rates 1.000
and 0.857, both far above the 0.10 floor) and fail only on **event count**,
which is a function of the 42-trajectory pool inherited from the 6-source smoke
panel. At the observed rates, V3 would need ~**701** trajectories for A and
~**71** for B.

Family B's excursions are **not boundary noise**: median depth 0.700 logP units
= 33.6% of the corridor width, 0 of 7 below the 0.1-unit noise threshold. Real
departures that came back.

**Per the frozen precedence and stop rules, this is a FAIL and the lane stops.**
Whether B's underpowered result justifies a larger pool is the lead's call, not
this lane's — see `HANDOFF.md`.

---

- **Status (ring-system stage A):** `SMOKE_HELD_IN` — **G1 FAILED**, closed
- **Branch:** `codex/compose-pathwise-constraints`
- **Base commit:** `04f1c46`
- **What is running:** **NOTHING.** Stage A finished; stage B was never launched.
- **Last completed gate:** G0 PASS · **G1 FAIL** · G2 PASS · G3 PASS · G4 PASS
- **Held-out data opened:** **NO.** Only `training_source_keys` was touched.

## The result, in one line

**Motif destruction is absorbing under the frozen kernel at H=6.** Of 42
rollouts on the unconstrained support, 19 broke the protected motif and
**0 recovered by the end**. Endpoint validity therefore *implies* path validity
here, so endpoint-only filtering is sufficient **by dynamics, not by luck** —
and the premise this lane rests on is empirically false in this regime.

## Gate table

| Gate | Verdict | Evidence |
|---|---|---|
| G0 mask integrity | **PASS** | 0 leaks (bug detector only — not a finding) |
| **G1 constraint non-vacuous** | **FAIL** | `endpoint_valid_path_invalid` = **0/6** sources and **0/19** endpoint-valid rollouts; threshold 0.10 |
| G2 mask leaves room to act | **PASS** | removes mean 22.6% of successors (median 5.5%, max 99.9%); 0/42 states left with empty support |
| G3 feasible paths can improve | **PASS** | `pathwise_greedy` improves worst margin by mean +3.10 over source |
| G4 distinguishable | **PASS** | landings differ 6/6; mean 4.3 distinct landings per source |
| G5 planning subclaim | **NOT ASSESSED** | stage B not authorised and not run |

G2 passing while G1 fails is the precise finding: the mask **is** a real
restriction on the support, but that restriction never changes what
endpoint-only filtering returns, because no trajectory ever comes back.

## Per the pre-committed anti-tuning rule

The motif rule was **not** changed. G1 failing is a reportable result about the
frozen kernel's legal support — its edit operators can open a labeled ring
system but effectively cannot reconstruct one within 6 edits — not a prompt to
re-roll the motif until an arm wins.

## One genuine positive, reported with its caveat

`endpoint_only` **failed to return anything** on 2/6 sources (all 6 of its
rollouts ended motif-invalid), while `pathwise_greedy` and
`pathwise_stochastic` succeeded 6/6. This has a free sign — nothing in the
construction forced those failures. But it is a *different* claim from the one
the lane was built to test, and n = 6 with 2 events is an existence proof, not
a rate.

Where `endpoint_only` did return something, the guarantee was **nearly free**:
paired delta mean +0.012, median −0.002 (n = 4).

## Cost actually incurred

| | Estimated | Actual |
|---|---|---|
| kernel calls / source | 85 (range 65–110) | **51** (43–58) |
| seconds / source | ~1 475 | **853** (620–1 163) |
| container-hours | 2.5 (range 1.9–3.1) | **≈ 1.4** |

Well under budget; the 360-call circuit breaker was never approached.

## Next action

**Await launch authorisation for Stage B.** The protocol, panel, runner,
analyser and 22 stage-B tests are committed. Nothing is running.

Because Stage B is 9–11× the A2 spend, the lead may prefer to authorise it in
halves: `--start 0 --sources 12`, inspect, then `--start 12 --sources 12`. The
driver skips committed shards, so the second half costs nothing extra and an
outage costs nothing.

Two facts from A2 that shaped this design and are already handled:

1. **Retention heterogeneity** (0.048–0.917) is handled by the predeclared
   support-tight classification with ITT plus frozen-threshold sensitivity.
2. **Absorption has not disappeared** — a third of A2 sources produced no
   excursion. The primary estimand's denominator is every eligible source
   precisely so that this cannot inflate the headline.
