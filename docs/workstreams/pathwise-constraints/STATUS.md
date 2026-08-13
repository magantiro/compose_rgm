# Workstream C — Pathwise Constraints: Status

- **Status:** `SMOKE_HELD_IN` — ring-system G1 FAILED (closed); reversibility census **NO FAMILY PASSES**; lane complete
- **Salvage-lane verdict:** none of the three predeclared families clears the frozen criteria. **Pathwise constraints leave the main paper.** The ring-system negative goes to the appendix. No fourth predicate was searched.

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

**None from this lane.** Both gates failed under frozen criteria. Handing back
to main. This lane recommends **not** paying for stage B and **not** searching
a fourth predicate; the one open question — whether family B's underpowered
result warrants a larger trajectory pool — is explicitly the lead's to decide,
and is costed in `HANDOFF.md`.
