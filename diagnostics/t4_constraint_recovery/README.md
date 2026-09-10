# Constraint-recovery diagnostic, 2026-09-09

## Outcome

The graded intermediate guide changed the sampled search, but did not produce
an eligible offspring on this development root. All 90 option attempts completed
and replayed successfully. No new docking calls or training were performed.

| Arm | Complete / attempted | Distinct eligible offspring | Recovery transitions / complete attempts from ineligible parents | Proposal time | Total worker time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Post-hoc | 30/30 | 0 | 0/18 | 526.8 s | 663.2 s |
| Hard terminal guide | 30/30 | 0 | 0/18 | 527.0 s | 664.0 s |
| Graded recovery guide | 30/30 | 0 | 0/18 | 740.9 s | 947.4 s |

The three workers ran concurrently. Each arm produced 30 distinct complete-option
molecules. Zero eligible offspring also means zero novel eligible or eligible
ring-adding offspring. The exact eligible starting parent occupied one active
beam slot throughout, but is not counted as an offspring, recovery, or new
observation. Its saved observed docking score remains -9.7; this experiment
measured no new docking score.

## What this establishes

Computed from the locked attempts and decisions:

- Hard terminal guidance was flat at all four retention decisions. Its candidate
  set was identical to post-hoc sampling.
- Graded guidance was nonflat at all four decisions. It shared 18 candidates with
  either other arm and produced 12 different candidates. The first-slot KL values
  against the empirical endpoint pool were 0.296, 0.827, 0.828 and 0.797, below
  the unchanged kappa=1 limit. The exploration floor remained active.
- The smallest constraint violation was unchanged across arms. The nearest
  candidate had QED 0.713460, SA 3.370458 and original-seed similarity 0.397059,
  below the required 0.4. It was never retained in any beam and received no
  exact-state continuation attempt. No benchmark-only violation-to-zero transition
  occurred either, so the null result is not solely a final medchem-screen effect.
- Region scopes remained variable. Intended release medians were 0.1724 in the
  first two arms and 0.1207 in the graded arm. Realized per-option coherent-change
  medians were 0.0690 in all three; these are distinct quantities, not evidence
  that a large requested region was substantially rewritten.
- Generic, growth, deletion, ring opening, rebuilding, and fused/pendant
  construction options were attempted. Exact option counts, structural changes,
  canonical identities and all selection decisions are retained in the artifacts.

Interpretation: keeping the incumbent and replacing a flat immediate score with
a graded immediate score is insufficient in this bounded comparison. The graded
heuristic is not a learned recovery probability or an exact finite-horizon
Doob transform. A missing successful continuation here does not prove that none
exists in the executable process.

## Decision and limits

Do not dock this empty eligible pool or tune the score weights using these
outcomes. The next proposed diagnostic is to examine short legal continuations
from generated near-feasible states, separating recovery to the known incumbent
from new eligible products. That would distinguish repair availability from
failure to retain and explore a recoverable state. It has not been run or
authorized by this diagnostic's contract.

This is one inspected warm development parent, one paired random seed and at
most four completed options per chain. It establishes neither general failure
of the controller nor competitive T4 performance. The benchmark endpoint gate,
frozen reference, executor, region/option priors and kappa were unchanged. No IVG
winner target entered generation. The same-generator post-hoc arm is retained.

## Evidence and reproduction

- Recipe: [task contract](../../configs/t4_constraint_recovery.json) and
  [design and launch decision](../../docs/T4_CONSTRAINT_RECOVERY.md).
- Generation revision: `f3834cd7b9a862b12e47433b5602a2c5da1583f4`.
- Run: `2dc9ba02d3e63016b666b9b1479177f0d508f77d71178b7031ef65258de9e69c`.
- [Launch receipt](attempt_1/launch.json) binds all three spawned calls, resources,
  exact code, inputs and zero-oracle authorization.
- [Machine-readable reduction](summary.json) binds physical input hashes,
  configurations, seed, source, software, work counts, timings and reducer identity.
  It preserves zero, failed and empty outcomes rather than omitting them.
- Per-case `generation_lock.json`, `scored_lock.json`, `result.json`,
  `runtime_gate.json` and `law_cache_inventory.json` are saved in `attempt_1/`.
  Exact law caches and executor receipts remain on the Modal volume at the
  locations recorded in the launch receipt.
- [Verification record](verification.json): 10 focused tests passed on the exact
  clean generation revision, strict preflight passed, touched-code lint passed.
  The legacy app retains 17 unchanged lint findings. No full repository suite
  was run, and no repository-wide milestone is claimed complete.

`fetch.py` only downloads existing durable artifacts, never launches or retries
scientific work. `summarize.py` verifies contract/input identity, sealed candidate
locks, root preservation, replay receipts, eligibility, paired first-level
attempts, probability normalization/floor/KL, and the recorded guidance scores.
With RDKit 2024.03.5 and the recorded chemistry environment:

```sh
PYTHONPATH=src:. python diagnostics/t4_constraint_recovery/summarize.py
```

Two successive reductions at the same analysis revision were byte-identical.
The summary includes analysis revision and environment, so reproducing it under
another revision or platform intentionally changes provenance fields.
