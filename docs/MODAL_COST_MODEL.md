# Modal cost model — VERIFIED, use this and nothing else

**Runtime class verified: standard Modal Functions.** `@app.function`, no Sandbox,
no region pin, no non-preemptible flag. Confirmed by inspection, not assumed.

| resource | rate | per hour |
|---|---|---|
| CPU | $0.0000131 / core-second | **$0.04716 / core-hour** |
| memory | $0.00000222 / GiB-second | **$0.007992 / GiB-hour** |

**Modal bills the GREATER of requested and actual usage**, so an oversized request
is a standing cost even if the process never touches it.

## Errors this file exists to prevent

1. **Memory was never priced at all.** Every estimate before 2026-08-15 counted
   CPU only. At 1 CPU + 16 GiB, memory is **73%** of the container cost.
2. **Then the CPU rate was "corrected" to $0.137/core-hour** — the
   Sandbox/Notebook rate, 3× too high — and memory to $0.024. Both wrong.
   The original $0.047 was right. This produced a $370/day and $800/run panic
   that was withdrawn.
3. **Cost was once divided by concurrency.** Total work is
   `slots x duration`, independent of packing. **Concurrency changes wall clock,
   not cost.**
4. **Memory requests were inherited by copy.** 32 GiB was legitimate for the
   trainer, which held ~1M x 1055 matrices. Every app derived from it inherited
   that figure; the SMC container needs ~1.5-2 GiB.

## Required before any launch expected to exceed $5

- verified rate class (Function vs Sandbox) and any region / non-preemptible multiplier
- measured duration distribution, not the mean alone — **wall clock follows the
  slowest unit, cost follows the total**
- measured peak RSS, with the memory request set from observed p90-95 plus headroom
- CPU and memory requests stated explicitly
- total slot-hours and the resulting maximum spend
- app tagged so actual billed cost can be reconciled afterwards

## Reference figures (1 CPU, 3 GiB, graph-only encode pending)

| run | cost |
|---|---|
| 4-source smoke | $0.17 |
| 16-source dev panel | $0.69 |
| 64-source broad gate | $2.76 |
| 128x20 validation | $110 (vs $672 at 16 GiB and today's encode) |

**Iteration happens on the smoke and dev panels. 64 is a finalist gate. 128/800
are escalation runs, never debugging tools.**
