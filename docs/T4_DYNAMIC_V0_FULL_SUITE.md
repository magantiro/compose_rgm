# Frozen Dynamic-v0 T4 comparator

## Scientific identity

- **Problem:** measure the benchmark-wide performance of the previously measured,
  route-free Dynamic-v0 COMPOSE search law.
- **Primary output:** a complete strict-eligible docking best-so-far curve for each
  T4 source cell.
- **Claim under test:** the generic Dynamic-v0 program synthesizer can optimize all
  released T4 sources without a stored complete-route bank.
- **Setting:** all 15 released source cells, first at strict similarity above 0.6,
  then at strict similarity above 0.4, with 1,000 charged calls per cell.
- **Baselines:** released IVG cell results and later learned Dynamic COMPOSE versions,
  joined only after the run.
- **Support:** exact supported molecular graphs with at most 40 active heavy atoms,
  one-to-three generic modules, 32 primitives and eight blocks. Stereochemistry and
  formal-charge edits remain outside the editing support.

## Frozen delta-0.6 wave

The first wave uses the exact historical Dynamic-v0 proposer and the first
predeclared controller/docking seed pair, `20260913` and `1701`. It loads zero
program-bank rows and starts with an empty complete-route archive. Every completed
novel endpoint must pass exact execution, strict similarity above 0.6, QED above
0.6, SA below 4, and the 40-heavy-atom support before docking.

Each cell owns an independent 1,000-call ledger and checkpoint. All 15 cells run in
separate single-CPU Modal workers, with at most 15 concurrent containers. There is
no competitive-plateau stop, retry, confirmation call, GPU, result-dependent
replacement, or target-specific routing. Genuine exhaustion of eligible proposal
support remains a reported terminal outcome.

This controller was developed with public T4 information. The wave measures
benchmark-trained performance, not held-out generalization. The delta-0.4 wave is
locked only after every delta-0.6 unit is durable and changes only the threshold and
its oracle-domain identity.

## Required artifacts

The run records the sealed contract, zero-oracle preflight, launch receipt, every
candidate lock and charged query receipt, per-call best-so-far curve, round summary,
checkpoint including random state, failures, and final cell outcome. Live reporting
uses the durable artifacts rather than worker log inference.

## Zero-oracle preflight revision

The first remote preflight failed before launch and made zero oracle calls. It
ran each cold-start proposal batch twice and required identical batch hashes.
That assertion was not valid for the frozen controller because a batch stops at
the first of 16 candidates, 128 attempts or 45 seconds of proposal work. Two
executions can therefore stop after different attempted-proposal counts at the
wall-clock boundary while using the same controller and random seed.

Preflight v2 preserves that failure and the historical 45-second bound. It
executes one cold-start batch per cell, requires a nonempty attempt ledger, and
exactly recompiles and replays every emitted candidate. It also verifies the
sealed empty route bank, controller identity, input hashes and zero-oracle
boundary. It does not change any scored-run setting.
