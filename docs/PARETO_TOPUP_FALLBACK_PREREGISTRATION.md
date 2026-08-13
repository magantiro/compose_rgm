# Generate-and-rank top-up — fallback rule, frozen BEFORE source 000's verdict

**Written at 16:46 EDT on 2026-08-13, with the volume verified EMPTY** — no
`000_verified_pref.json`, no partial. Source 000's status was unknown when every
rule below was fixed, and the probe is blinded to scientific outcomes in any
case.

## Why a fallback is needed, and why it is not moving the goalposts

Source 000 is an **extreme engineering case, and this was known in advance** from
a statistic that predates the probe — the committed smoke's realized kernel yield
per generate-and-rank trajectory:

| src | traj | kernel | yield |
|---:|---:|---:|---:|
| **0** | 61 | 60 | **0.98** |
| 8 | 60 | 125 | 2.08 |
| 3 | 67 | 141 | 2.10 |
| 7 | 59 | 128 | 2.17 |
| 2 | 66 | 145 | 2.20 |
| **5** | 75 | 188 | **2.51** |
| 10 | 76 | 198 | 2.61 |
| 1 | 65 | 173 | 2.66 |
| 6 | 62 | 176 | 2.84 |
| 11 | 74 | 215 | 2.91 |
| 9 | 71 | 214 | 3.01 |
| 4 | 62 | 242 | 3.90 |

Median 2.556. **Source 000 is the worst of twelve by a wide margin**, at 38% of
the median and well below the next lowest. Its unguided trajectories from `x_0`
collide in the enumeration cache far more than any other source's.

## The rule

| source 000 verdict | consequence |
|---|---|
| **`MATCHED`** | proceed with the remaining 11, each against **its own** committed target |
| **`MATCHING_UNREACHABLE`** | that status is **permanent for source 000**. Run **one** median-yield source as a second engineering feasibility probe. If it matches, proceed with the rest of the cohort; source 000 contributes a **resource frontier** rather than a kernel-matched scalar |

## The median-yield fallback source, selected now: **source 5**

Selected from the pre-existing kernel-yield statistic alone, with no reference to
any P3/P4 or hypervolume outcome, before source 000's verdict existed.

**The selection needed exact arithmetic to be honest.** Yields are ratios of
integers, and two sources are **exactly tied** at the smallest deviation from the
median:

```
median            = 14569/5700 = 2.555964912...
smallest |dev|    =   281/5700 = 0.049298246...
exactly tied      = {5, 10}
declared tiebreak = lowest source index  ->  SOURCE 5
```

In floating point the two deviations print identically to six decimals and the
sort order flipped between runs depending on how the arithmetic was arranged — a
tiebreak decided invisibly, below printed precision. That is worse than an
arbitrary rule, because it looks deterministic and is not. Resolved with
`fractions.Fraction`, so the tie is **explicit** and the declared rule actually
fires.

**Source 5:** yield 2.5067, committed `verified_pref` target **450** kernel calls.

### Disclosure, made now rather than discovered later

Source 5's committed set-level HV contrast is **+0.1382** (verified over greedy)
— a source where COMPOSE won. The other tied candidate, source 10, is **−0.073**
— one of the two sources where COMPOSE lost.

Recorded explicitly because a reviewer will check: the tiebreak is *lowest
index*, declared before the tie was known to exist, and it happens to select the
favourable member of the pair. **The rule was not chosen to produce that.** Had
the tiebreak been "highest index" — equally defensible a priori — it would have
selected source 10. If that asymmetry is judged uncomfortable, the correct
response is to **probe both 5 and 10**, not to switch the rule; and since the
probe is blinded and measures only kernel feasibility, neither choice can move a
scientific outcome.

## What `MATCHING_UNREACHABLE` on source 000 would and would not license

**Would not license:**

> ~~"generate-and-rank cannot be kernel matched."~~

**Would license, and only this:**

> Kernel matching is unreachable for **source 000** under the frozen
> generate-and-rank process.

**And a correction to my own earlier reasoning, recorded so it is not repeated.**
I wrote that because 000 is the worst source, "if 000 can spend the budget, the
other eleven almost certainly can." That is reasonable *operational* intuition
and it is **not a scientific inference**. Different sources have different
reachable-state graphs and can saturate differently. Every source's status is
**run, never inferred from its initial yield.**

## The structural reading, if UNREACHABLE occurs anywhere

It would reveal something genuine rather than something broken. Generate-and-rank
repeatedly restarts from `x_0`, so it revisits the same reachable local states
and eventually gains almost no new kernel work. COMPOSE's controlled lookahead
walks many different successor states along evolving trajectories. **The two
computational structures are simply different.**

They must not be forced to look identical by disabling the cache, moving the
start state, or redesigning generate-and-rank. That is why the frozen report is

```
HV vs kernel work     HV vs objective queries     HV vs completed trajectories
```

**The matched point is useful where it exists; the resource frontier is more
fundamental.**
