# Validity-fiber diagnostic — what state-dependent executability conditioning buys

**Mechanism diagnostic, NOT a benchmark arm.** Zero oracle calls, zero docking, zero Modal (0, 0, 0, asserted in the artifact). Pinned kernel: python 3.11.13, rdkit 2024.03.5, numpy 1.26.4, scipy 1.13.1.

All 15 T4 held-target seeds at delta=0.6 (heavy atoms 16-40, five targets), 48 slots, horizon 3, 256 programs per state per arm, identical seeds across arms, serial single process (a loaded machine corrupts wall_seconds). 7,680 draws total.

## The two arms, and the rule that separates them

- **Arm F** — the shipped fiber-conditioned sampler through the real `expand` / `synthesize_dynamic_program` / executor / `Fiber.check`. Not transcribed: the harness wraps them only to observe a funnel that `expand`'s return value does not expose.

- **Arm U** — the SAME program space with state-dependent executability conditioning removed at the draw site, then execute-and-reject through the same production executor. The separation rule is structural and was read off the enumerators, not invented: they are written as `tuple(action for <coords> in <DECLARED RANGE> if <ADMISSION PREDICATE>)`, so the `for` clause is the SPACE and every `if` that consults graph structure to decide executability is the CONDITIONING. Arm U keeps the `for` and drops the `if`. Every removed predicate is recorded in the artifact with its production file:line.

An equivalence self-check (52 checks, 0 mismatches) requires the two switchable copies to reproduce the production helpers byte-for-byte at `conditioned=True` under identical seeds, so a copy that drifts fails loudly instead of silently measuring a different sampler.

## FINDING 1 — the three headline metrics are SATURATED BY CONSTRUCTION

| | arm F | arm U |
|---|---|---|
| executable fraction | 1.0000 | 0.9997 (1 draw of 3840) |
| valid intermediate fraction | 1.0000 (0 invalid of 19,430) | 1.0000 (0 invalid of 20,479) |
| wasted proposals | 0 | 1 |

These three cannot discriminate the arms, and the reason is architectural: `synthesize_dynamic_program` does not draw a program and then execute it — **the program IS the record of what executed.** `compile_generic_module` calls `execute_program` inside the draw, and every action passes `editing_v2_semantic_rewrite_system()` (`whole_ring_plan.py:70-90`), which raises `InvalidRewrite` on any invalid state. Both arms inherit validity closure from the EXECUTOR, not from the conditioning.

So the validity-closure claim is confirmed empirically at scale — **0 invalid intermediates in 39,909 committed states**, each decoded and re-parsed through RDKit — but it is NOT what the fiber conditioning buys. Any claim that draw-time conditioning is what keeps intermediates valid is refuted here.

## FINDING 2 — what it does buy: rejection cost, 9.8x

| | compile attempts | refusals | refusal rate | attempts per accepted module |
|---|---|---|---|---|
| arm F | 7,617 | 382 | **0.0502** | **1.05** |
| arm U | 14,190 | 6,975 | **0.4915** | **1.94** |

Per-family refusal rates locate the effect precisely on the families whose draw is a uniform pick from an enumerated legal-event fiber: `ring_system_restate` 1.000, `bond_reroute` 0.991, `cycle_open` 0.936, `cycle_close` 0.875, `heteroatom_substitute` 0.640. The families whose conditioning could NOT be removed barely move (`append_ring` 0.189 vs 0.170, `fuse_ring` 0.154 vs 0.095, `carbonyl_insert` 0.172 vs 0.164) — an internal control showing the effect tracks the patches rather than something else.

**Instrument cross-check.** The harness's refusal count was compared against production's own `module_failure_counts`: arm F **382 = 382 exactly**; arm U 6,975 vs 6,962, the 13-refusal gap being exactly the one draw where all 13 families refused and no metadata existed. Two independent instruments agree.

## FINDING 3 — eligible-endpoint yield, and where the gap is categorical

| | unique eligible | cells with any eligible |
|---|---|---|
| arm F | **409** | **11 / 15** |
| arm U | **369** | **7 / 15** |

The aggregate UNDERSTATES it, because it is dominated by a few high-yield cells. Cell-level W/T/L for arm F is **8/5/2**, and the decisive asymmetry is **4 cells where arm F yields >0 and arm U yields exactly 0, and 0 cells the other way**.

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

**No zero is reported without its expected count.** The artifact's `zero_discrimination` block gives, for each arm-U zero, the count expected at arm F's rate and an approximate Poisson P(0): idx6 (11 expected, 1.7e-05) and idx4 / idx10 (8 expected, 3.4e-04) are DISCRIMINATING; **idx11 (3 expected, p=0.050) is WEAK**; and idx3 / idx5 / idx8 / idx9 are **BOTH_ZERO and therefore undiscriminating** — those are the known delta=0.6 exhaustion cells, and their zeros are a property of the cell, not of either draw law.

Arm F's own refusal rate is not saturated either: it runs 0.002 (idx0, 19 heavy atoms) to 0.130 (idx8, 31), rising with molecule size, so it is a live instrument rather than a flat one.

## WHAT WAS FALSIFIED

1. **That conditioning would show in executable fraction or intermediate validity.** It does not and cannot — both are executor-guaranteed. Checked that both metrics were CAPABLE of moving before the null was read: arm U's executable fraction did move (0.9997) and arm F's refusal rate moves 65-fold across cells.

2. **That unconditioned sampling is uniformly worse.** False on permissive cells: arm U BEAT arm F on idx0 (87 vs 81) and idx13 (93 vs 68) and tied idx2. Where the gate admits a large set, blind draw-and-reject finds it fine. The conditioning earns its keep only where the admissible set is thin — which is the crowded, heavily-substituted drug-like states T4 actually asks about.

3. **That conditioning is the cheaper path.** The opposite: arm F is **1.59x SLOWER per draw** (0.839 s vs 0.527 s), because enumerating the admitted fiber — notably the semantic cycle-close and atom-restate admission masks — costs more than executing a bad draw and discarding it. Conditioning buys HIT RATE and pays for it in ENUMERATION.

## CAVEATS THAT MUST TRAVEL WITH THESE NUMBERS

- **Arm U is only PARTIALLY unconditioned — 10 of 13 families — so every gap above is a LOWER BOUND on what full fiber conditioning is worth.**

- **One conditioning point is INSEPARABLE, and that is itself an architectural finding.** `rewrite/tracelet_fiber.py:353-387 _ring_system_restate_candidates`: `RingSystemRestate` carries an arbitrary-length tuple of `BondOrderChange`, so there is no bounded coordinate range to draw from. The candidate set is CONSTRUCTED — perceive bridges, take ring-only connected components, enumerate maximum-cardinality matchings (`:371-386`) to build coherent Kekule alternations. The matching does not filter a declared space, it manufactures the parameter. An 'unconditioned' draw over all subsets of all bond pairs x orders is a combinatorially different and astronomically larger space, which would break the same-space requirement. Arm U therefore has this family decline the state (refusal rate exactly 1.000), which is what production does when a fiber is empty. Two further points (`ring_program.py:188-216 construction_branches` for append/fuse_ring, and the inline `carbonyl_insert` anchor filter plus capacity clamps) were retained for scope, not because they are inseparable.

- **This is a matched-DRAW comparison, not matched-COMPUTE.** At equal wall clock arm U would get ~1.59x more draws; that arm was NOT run, so 409 vs 369 is a statement about 256 draws each. The four zero-cells are the load-bearing result and are less exposed (arm U yields 0 per draw there, so more draws of the same law is not obviously a fix) — but arm U at ~407 draws was not measured, so those zeros are NOT claimed to be budget-proof.

- **`valid_intermediate_fraction` is measured at program-graph block boundaries** (`trace["states"]`, source excluded), not per primitive edit; the artifact records `primitive_edits` alongside so the granularity is visible.

