# Workstream C — Pathwise Constraints: Status

- **Status:** `SMOKE_HELD_IN` — stage A complete, **G1 FAILED**, lane stopped per stop rule
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

**None from this lane.** Handing back to main with the frozen protocol and a
FAILED premise gate. See `HANDOFF.md` for the three options the lead may pick
between; this lane recommends **not** paying for stage B.
