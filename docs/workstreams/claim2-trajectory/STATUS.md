# Claim 2 — trajectory characterization — STATUS

**Status:** `DEVELOPMENT` — Step 1 (cycle attribution) and Step 2 (36-source
held-in run) both complete.

**Decision-rule branch: KEEP AND REFRAME.** Do not reopen `R_theta`.
**Frontier result: INCOMPARABLE**, both axes resolved, at n=36.

**Branch:** `codex/compose-claim2-trajectory`
**Held-out data opened:** **NO.** The matched reserve is untouched.
**What is running:** nothing.

---

## The central causal question — answered

**The training reference process is itself locally reversible.** Across 151,059
teacher transitions and 144,870 distinct directed edges, 53,118 are mutual:

> **mutual-edge fraction = 0.7333**

The preregistered thresholds, frozen at `5c81818` before the census existed,
were ≥ 0.33 → `inherited`, ≤ 0.10 → `model_specific`. This is not near the
line. `R_theta` reproducing local reversibility is **faithful modelling of the
process it was trained on**, not a compositional pathology. It corroborates the
corpus construction, which lists `reversible_synthetic_walk` among its lanes.

## 36-source held-in development result

| arm | mobility | fidelity | net edits | cancelled | 2-cycles | disp/net | reverse-edge P |
|---|---:|---:|---:|---:|---:|---:|---:|
| `r_theta` | 0.5515 | **0.9676** | 4.25 | **29.2%** | 0.90 | 0.1477 | **0.1630** |
| `uniform_canonical` | **0.7755** | 0.8310 | 5.97 | 0.5% | 0.01 | 0.1286 | 0.0242 |
| `empirical_family` | 0.7635 | 0.9213 | 5.81 | 3.2% | 0.08 | 0.1351 | 0.0535 |

Paired over sources, both comparisons return **`incomparable`**:

| comparison | mobility | fidelity |
|---|---|---|
| vs `uniform_canonical` | −0.2439 [−0.3085, −0.1762] **resolved** | +0.1366 [+0.0718, +0.2199] **resolved** |
| vs `empirical_family` | −0.2457 [−0.3093, −0.1819] **resolved** | +0.0463 [+0.0046, +0.0972] **resolved** |

`R_theta` buys chemical-envelope retention with structural displacement. That
is a declared, publishable outcome and is not scalarized away.

## Where the cycling lives

Cancellation is almost entirely **immediate toggling**, not wandering: 0.90
two-cycles per trajectory against 0.26 longer revisits. It concentrates in the
families the operator algebra makes cheapest to undo — `cycle_attach` 0.43,
`ring_system_restate` 0.33, `bond_reorder` 0.23 — while `atom_delete` sits at
0.01. `R_theta` puts **0.163** of its mass on the reverse edge, against 0.024
for uniform and 0.054 for empirical-family.

**No operator collapse.** All three arms committed all 8 active families.

## Instrument gates (n=36)

0 kernel cross-check disagreements · 0 degenerate states of 1,025 · 0
enumeration failures · 0 truncated trajectories · measured **14.47 s** median
per enumeration.

## Two things that did NOT replicate, recorded rather than buried

1. **The n=8 "moves further per net edit" finding did not survive.** At 8
   sources displacement per net edit was 0.213 vs 0.120/0.131; at 36 it is
   0.1477 vs 0.1286/0.1351 and the paired interval **includes zero**
   (+0.0194 [−0.0051, +0.0459]). It is reported as an unresolved diagnostic,
   never as a verdict axis. The 20-source floor earned its keep.
2. **Cancellation fell from 43.8% to 29.2%** between the 8- and 36-source
   panels — a reminder of how unstable an 8-source point estimate is.

## Decision rule

**REOPEN `R_theta`** required all three of: cycling model-specific, `R_theta`
worse on **both** net mobility and fidelity, and controlled trajectories also
cyclic. **None of the three holds.**

**KEEP AND REFRAME** holds: reversibility is inherited (0.733), fidelity is
materially and resolvedly better, controlled trajectories show no cancellation,
and net mobility remains useful (0.55 median endpoint distance, 4.25 net edits
of 6). The honest framing is **learned local reference dynamics**, not
goal-directed transport — which is exactly the layer the paper needs `R_theta`
to occupy, with purpose supplied by control.

## Next action

**None without authorization.** Reserve stays closed. The bounded next step
would be main-lane review of this handoff and a decision on whether the
matched-reserve confirmation (~27 container-hours) is warranted given that the
frontier lands on `incomparable` rather than `dominates`.
