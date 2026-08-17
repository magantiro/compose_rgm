# Amendment — SMC efficiency study

**Amends preregistration section 13**, which freezes N = 32 particles, freezes
the terminal output rule, and states that no particle-count sweep is permitted.
This amendment is recorded BEFORE the study runs and before any of its data are
read.

## Why the original freeze existed, and why this is not a violation of its intent

Section 13 froze N = 32 as "the single pre-existing operating point" and barred a
sweep. The purpose is clear from its context: to stop a controller from being
tuned for PERFORMANCE against the development panel, which would inflate the
reported success rate through selection.

This study asks the opposite question. The endpoint is the **smallest**
configuration that does not lose success relative to N = 32 — never the
best-performing configuration. A configuration that scores *higher* than N = 32
is not thereby selected; it is treated as noise on 64 binary outcomes, exactly as
the protocol's own warning about best-of-N source rates requires. The direction
of the search is what keeps this outside the failure mode the freeze protects
against.

## What motivated it

Measured on the banked cohort (`docs/EARLY_STOP_PROBE.json`, 130 records):

    successful records                27 / 130
    transitions in failed records     79k / 91k  = 87%

Eighty-seven percent of all particle-transition compute is spent on runs where
no particle ever enters the region. A fixed N = 32 by H = 24 budget is paid in
full to establish that a source is unreachable.

This also bounds what whole-run termination can do. Terminating at first target
hit is EXACT — `success <=> some particle reached B`, confirmed on all 130
records, because the terminal potential `h_0(x) = 1[x in B]` drives every
non-region particle to zero weight — but it saves only **4.9%** of total
compute, because a failed run has nothing to stop at. The particle count is the
only lever that cuts failures and successes alike.

## What is amended

1. **Particle count.** N may take values other than 32 in a declared study.
2. **Terminal output rule.** Whole-run termination at first target hit is
   permitted. It preserves the binary benchmark event exactly and changes only
   WHICH molecule is returned, and therefore the secondary terminal QED and
   similarity, which are reported separately in any case.
3. **Early abandonment.** Terminating a run whose particle mass has collapsed is
   permitted as a declared arm. Unlike (2) this is **NOT exact** — it will drop
   runs that would have succeeded — and must be reported as a measured
   success/compute tradeoff, never folded into a headline rate.

## Design, fixed before the data are read

- **Panel:** all 64 development sources. Not a subset, so no panel selection.
- **Baseline:** the existing N = 32 replicate 0 records. Same sources, same
  seeds, already banked — the comparison is paired by construction.
- **Arms:** N = 16, 8, 4 at replicate 0. Seeds are identical to the baseline.
- **Primary readout:** source success count per arm against 19/64 at N = 32.
- **Decision rule:** choose the smallest N whose success count is not
  meaningfully below N = 32 on the paired comparison. Any chosen configuration
  is then confirmed on the full 64 sources before use.
- **Not touched:** horizon, region, twist, proposal law, resampling rule,
  chemistry, and the official 800 sources, which remain unconsumed.

## Recording requirements

Every non-frozen run announces its N in the container log, records it in both
the record and the provenance, and is REFUSED into the frozen cohort's output
directory. An amended arm therefore cannot be mistaken for a
frozen-operating-point result, and cannot be silently mixed into the ladder.

## What this amendment does NOT license

It does not license reporting an amended configuration as the frozen controller,
choosing a configuration because it scored higher, running the official 800 under
any configuration that has not been confirmed on the 64-source panel first, or
treating the secondary terminal metrics as comparable across arms with different
output rules.
