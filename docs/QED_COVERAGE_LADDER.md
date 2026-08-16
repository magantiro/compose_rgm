# QED coverage ladder — add attempts only to UNSOLVED sources

## Why not another full panel

The benchmark metric is **source success**: a source counts once, no matter how
many of its 20 candidates qualify. So re-running attempts on already-solved
sources buys nothing. Development should spend attempts exactly where the
uncertainty is.

## The ladder

Candidate seeds are preassigned `1..20` and deterministic, so attempts are
append-only and any prefix is a valid experiment.

```
C(1) -> C(2) -> C(4) -> C(8) -> ...
```

where `C(k)` is source coverage after the first `k` attempts. After each rung,
**run the next candidate only for sources still unsolved.** Continue only while
coverage is still climbing meaningfully.

Because seeds are fixed and every slot is independently persisted, `C(4)`
computed this way is **identical** to having run 4 slots on every source — it
just skips work whose outcome cannot change the metric.

## The question this is designed to answer

The probe showed coverage is limited by **which sources are reachable at all**,
not by attempt count:

| | |
|---|---|
| pooled slot hit rate, 4 slots | 39.1% |
| independence predicts at 4 slots | 86.2% |
| **actually observed** | **68.8%** |

Projecting to 20 attempts spans **69%-99%** depending entirely on how the five
0/4 sources are treated — and **four samples cannot distinguish "unreachable"
from "hard"**: a source with true `p = 0.15` shows 0/4 about 52% of the time.

**The whole uncertainty lives in the unsolved sources.** The ladder puts every
additional attempt there.

## Stop rule

| observation | action |
|---|---|
| coverage climbs toward or past 45.1% within a few slots | engineer the production version, then 128 |
| coverage plateaus with sources still untouched | stop buying independent SMC attempts; open the fast controller loop (value-guided branch/search, or a shared source-level archive) |
| broad matched advantage disappears | stop scaling SMC immediately |

## Panels

* **12-16 source balanced panel** (easy/medium/hard from the 64) — daily
  controller comparisons, every arm gets the same slots. Target 10-20 minutes.
* **64-source cohort** — broad coverage gate for finalists, not debugging.
* **128 x 20 / 800 x 20** — final-scale evidence only.

**No more multi-hour controller-development runs.**
