# QED development cadence — the standard from 2026-08-15

## The slow SMC reference is FINISHED FOREVER

It was a one-time, deliberately serial reference for a mathematically delicate
new primitive. It **had** to reach the `b = 0` terminal boundary, because that is
where the all-zero-weight extinction bug lived — a 3-step smoke test could not
have found it. It caught two real molecular-integration bugs and is banked at
`docs/HPHI_SMC_REFERENCE_BANKED.json`.

**We never pay that two-hour tax again for this SMC implementation.**

* change only the controller on top of SMC → **no new long reference**
* abandon SMC for branch/search → **no SMC reference at all**
* genuinely new delicate mathematics → **the cheapest test capable of
  falsifying it**, which is usually an exact enumerable toy, not molecules

## The steady-state loop

```
new controller idea
   -> 8-16 representative sources x 1 attempt   ~$0.17-0.69, target 10-20 min
   -> 64 x 1 broad gate                          ~$2.76   FINALIST ONLY
   -> 128 x 20  ->  800 x 20                     final-scale evidence, never debugging
```

**A new idea never earns a 64-source or multi-hour run until it shows signal on
the fast panel.** Long correctness runs are allowed only when a genuinely new
mathematical semantic cannot be tested on an exact toy or a short molecular
construction.

## Why the current runtime is not intrinsic

After carried-`h`, one particle transition is ~5.45 s, of which **79.1% is
"encode"**. That path calls `_one_state_batch` → `prepare_factorized_mark_batch`,
558 lines building admission and action masks, to obtain a graph embedding —
while `_encode_batch` provably reads **only 10 graph-structure fields and never
touches a mark-space tensor** (verified by AST).

That is implementation waste, not the cost of resolving executable chemistry.

## Engineering order after Round 1 (one focused pass, not a science detour)

1. **Graph-only `h_φ` encoding path.** Build only the fields the encoder
   consumes. **Gate: bitwise-identical embeddings AND `h_φ` values** across all
   736 banked reference transitions plus a broader molecular sample. Not exact →
   discard.
2. **Re-profile.** If per-transition falls toward ~1-2 s, the loop is cheap.
3. **Particle-level parallelism for WALL CLOCK.** The 32 particles are
   independent until ESS/resampling. Use fixed per-particle RNG substreams so
   scheduling cannot change the algorithm — then serial and parallel implement
   the *same* contract and can be compared exactly.

## The bigger algorithmic lever — only if still needed

Twenty independent 32-particle searches from the same source, discarding every
useful state between candidate slots, is conceptually wasteful. The benchmark
requires **20 returned candidates**, not 20 amnesiac searches. A source-level
executable archive that retains promising intermediates, fibers and value
information could reduce **total work**, not merely parallelize it — and is more
COMPOSE-native.

**Order:** establish SMC is strong across 64 → make it fast with exact
engineering → only then, if 20-output scaling is still costly or coverage
plateaus, build shared-state multi-output search.

## The standard we hold ourselves to

| inference overhead vs comparator | verdict |
|---|---|
| 5-10x with a major gain + unique capabilities | defensible |
| 10-30x with a very strong result | defensible, but discuss the tradeoff |
| **100x for a marginal gain** | **a bad final algorithm, even if the benchmark permits it** |

"Native inference is allowed" is not cover for failing to engineer the thing
properly. We are aiming for genuinely good, not merely defensible.
