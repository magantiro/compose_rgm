# Validity-fiber diagnostic — what state-dependent executability conditioning buys

**Mechanism diagnostic, not a benchmark arm.** Zero oracle calls, zero docking, zero Modal (0, 0, 0). Measured under the pinned kernel: python 3.11.13, rdkit 2024.03.5, numpy 1.26.4.

15 T4 held-target seeds at delta=0.6, 48 slots, horizon 3, 256 programs per state per arm, identical seeds across arms. Serial single process, because a loaded machine corrupts wall_seconds.

## The two arms

- **Arm F** — the shipped fiber-conditioned sampler, through the real `expand` / `synthesize_dynamic_program` / executor / `Fiber.check`. Not transcribed; the harness only wraps them to observe a funnel the return value does not expose.

- **Arm U** — the SAME program space with the state-dependent executability conditioning removed at the draw site, then execute-and-reject. The separation rule is structural: the production enumerators are `tuple(action for <coords> in <DECLARED RANGE> if <ADMISSION PREDICATE>)`; the for-clause is the space, the if-clause is the conditioning. Arm U keeps the `for`, drops the `if`, and lets the production executor refuse. Each removed predicate is recorded in the artifact with its production file:line and an argument for why it is executability rather than space.

An equivalence self-check (52 checks, 0 mismatches) requires the switchable copies to reproduce the production helpers exactly at `conditioned=True` under identical seeds, so a copy that drifts fails instead of silently measuring a different sampler.

## Aggregate

| | arm F (fiber-conditioned) | arm U (unconditioned + reject) |
|---|---|---|
| draws | 3840 | 3840 |
| executable fraction | 1.0000 | 0.9997 |
| **valid intermediate fraction** | **1.0000** | **1.0000** |
| committed intermediate states | 19430 | 20479 |
| invalid intermediate states | 0 | 0 |
| unique eligible endpoints | **409** | **369** |
| states with ANY eligible endpoint (of 15) | **11** | **7** |
| wasted proposals | 0 | 1 |
| module compile attempts | 7617 | 14190 |
| module compile refusals | 382 | 6975 |
| **executor refusal rate** | **0.0502** | **0.4915** |

## What this does and does not show

**It does NOT show that conditioning buys validity.** Both arms report a valid-intermediate fraction of exactly 1.0000 over 39,909 committed intermediate states, with ZERO invalid states in either. That is the validity closure working as designed and it is a property of the EXECUTOR, which refuses an illegal rewrite rather than committing it — so removing the draw-time conditioning cannot manufacture an invalid molecule. Any claim that fiber conditioning is what keeps intermediates valid is refuted here.

**What it buys is search efficiency and coverage.** Unconditioned sampling is refused by the executor on 49.2% of module compile attempts against 5.0% conditioned — **9.8x** — and spends 1.86x the compile attempts to return 10% FEWER unique eligible endpoints, covering 7 of 15 states instead of 11.

**And the gap is where the benchmark lives.** Per state:

| idx | target | heavy | F eligible | U eligible | F refusal | U refusal |
|---|---|---|---|---|---|---|
| 0 | parp1 | 19 | 81 | 87 | 0.002 | 0.452 |
| 1 | parp1 | 16 | 37 | 35 | 0.018 | 0.436 |
| 2 | parp1 | 17 | 15 | 15 | 0.011 | 0.453 |
| 3 | fa7 | 32 | 0 | 0 | 0.016 | 0.485 |
| 4 | fa7 | 29 | 8 | 0 | 0.006 | 0.415 |
| 5 | fa7 | 35 | 0 | 0 | 0.041 | 0.528 |
| 6 | 5ht1b | 39 | 11 | 0 | 0.070 | 0.577 |
| 7 | 5ht1b | 16 | 24 | 20 | 0.046 | 0.443 |
| 8 | 5ht1b | 31 | 0 | 0 | 0.130 | 0.501 |
| 9 | braf | 40 | 0 | 0 | 0.118 | 0.489 |
| 10 | braf | 39 | 8 | 0 | 0.068 | 0.499 |
| 11 | braf | 37 | 3 | 0 | 0.086 | 0.514 |
| 12 | jak2 | 22 | 133 | 113 | 0.026 | 0.478 |
| 13 | jak2 | 22 | 68 | 93 | 0.010 | 0.480 |
| 14 | jak2 | 27 | 21 | 6 | 0.060 | 0.547 |

On the small seeds (parp1 16-19 heavy atoms, jak2 22) the two arms are comparable and arm U is sometimes ahead — brute force works when the state is permissive. On the large constrained seeds the difference is categorical: **states 4, 6, 10 and 11 (29, 39, 39 and 37 heavy atoms) yield 8, 11, 8 and 3 eligible endpoints conditioned and ZERO unconditioned.** Arm F's refusal rate also climbs with molecule size (0.002 at 19 heavy atoms to 0.130 at 31) while arm U's sits near 0.45-0.58 regardless, which is what a state-blind proposal distribution looks like.

So the conditioning is not a constant-factor speedup; it is what makes the crowded, heavily-substituted drug-like states searchable at all — and those are exactly the T4 leads.

## Caveat worth stating

Arm U's WALL time is lower (2023s against 3223s) because a refused module is cheap — the executor rejects it before the instantiate-and-gate work a surviving module pays for. Wall time is therefore the wrong cost axis here; compile attempts per eligible endpoint is the right one, and on that axis arm F is 2.1x more efficient.

