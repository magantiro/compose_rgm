# Workstream C — Pathwise Constraints: Status

- **Status:** Stage B **EXECUTED, 24/24 sources, instrument healthy** (`SMOKE_HELD_IN`)
- **What is running:** **NOTHING.** All batches finished.
- **Held-out data opened:** **NO.** No held-out confirmation is designed.

> **DEVELOPMENTAL, not a headline claim.** The cLogP family emerged from the
> feasibility screen rather than being specified in advance. Stage B would need
> its own held-out confirmation before it can be stated as a result. Recorded
> in `PROTOCOL.md` before the run.

## Stage B results — 24 held-in sources

### Primary: hidden-path RATE `P(x_H ∈ C AND ∃t<H: x_t ∉ C)`

| arm | rate | 95% CI (source-clustered) |
|---|---:|---|
| `endpoint_greedy` | **16/24 = 0.667** | [0.458, 0.833] |
| `endpoint_verified` | **14/24 = 0.583** | [0.375, 0.792] |

**Two-thirds of sources produce an endpoint-only trajectory that lands inside
the corridor after passing through a state the corridor forbids.** This is the
estimand that had genuine room to be zero — the ring motif returned exactly
that — and it is not zero here.

### Secondary: conditional fraction, with denominators

| arm | fraction | denominator |
|---|---:|---|
| `endpoint_greedy` | 16/24 = 0.667 | all 24 delivered |
| `endpoint_verified` | 14/23 = 0.609 | 23 delivered |

The denominators are nearly full, so the small-denominator pathology the
primary estimand guards against **did not materialise here**. The guard was
still correct to have.

### Terminal cost, controller parity (sign free)

| | mean | median | 95% CI |
|---|---:|---:|---|
| `Δ^G` = pathwise − endpoint, greedy | −0.105 | **0.000** | [−0.377, +0.099] |
| `Δ^V` = pathwise − endpoint, verified | −0.023 | **0.000** | [−0.182, +0.134] |

**Both CIs include zero and both medians are exactly zero** — on most sources
the guarantee costs no potency at all; the negative means are carried by a few
sources (`Δ^G` min −2.49). This is the first of the three predeclared outcomes:
little or no potency cost.

### Support viability

- **1 of 24 sources support-tight** (4.2%), index 1.
- Per-source median retention: min 0.084, **median 0.798**, max 0.922 — much
  tighter spread than A2's 0.048–0.917.
- **Zero sources had any mask-empty state.**
- Predeclared sensitivity excluding the tight source: rates 0.652 / 0.565,
  `Δ^G` −0.121, `Δ^V` −0.041. **Conclusions unchanged.**

### Future-aware — guaranteed sign, magnitude only

Effect size mean +0.661, median +0.559, CI [0.428, 0.944]; top-1 disagreement
2.25 of 6 steps. **Binary headroom: 0 rescued over a denominator of 0
`pathwise_greedy` failures — a CEILING, not a null.** The subclaim closes as
the retargeting lane closed its own.

### Barred from the results table

`pathwise_greedy` and `pathwise_verified` recorded **0** intermediate
violations. That is the construction, checked only as a bug detector
(`mask_integrity: PASS`, 0 leaks, 0 void shards).

## Cost

**17.41 container-hours** against a 12–15 estimate (16% over). 182 kernel calls
per source (estimate 120–150), median 44 min per source. Circuit-breaker margin
46.5%, never hit.

## Prior stages

| Stage | Verdict | State |
|---|---|---|
| Ring-system stage A | **G1 FAIL** | CLOSED, permanently, not revised |
| Three-family reversibility census | **NO FAMILY PASSES** | CLOSED |
| Stage A2 — corridor prevalence | **PASS (5/5)** | complete |
| **Stage B — corridor-constrained potency** | **executed** | complete, developmental |

---

## Stage B design as frozen

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

**None from this lane. Stopped as instructed.** Stage B is complete and the
main lane decides what follows. No held-out confirmation is designed, and
designing one is explicitly not this lane's call.

## Operational note worth carrying to other lanes

The batch-1 client was killed by the harness at 10/12 and **`--detach` did not
save it** — the app went to `stopped`, exactly as this lane was warned. Modal's
own message explains why: *"running a local entrypoint in detached mode only
keeps the last triggered Modal function alive after the parent process has been
killed or disconnected."*

**The resumable driver did save it.** Relaunching skipped the 10 committed
shards and dispatched only 2, so the outage cost 2 sources of recompute instead
of 12. The fix for the client itself is to orphan it —
`nohup … & disown` from a script that exits immediately, so no process-group
signal from the harness reaches it. Both later launches survived.
