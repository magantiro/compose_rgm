# Scope lock for experimental closeout, and the kernel-cost blocker

**Canonical. Nothing is cut; the package is organised, not reduced.**

## The locked package — six pieces, no additions

| # | piece | state |
|---|---|---|
| 1 | executable-valid reference dynamics and **edit expressivity** | reported *from* other panels; no standalone experiment |
| 2 | **DDSBM** conventional endpoint competence, frozen `R_θ` | protocol frozen, gate passed, **DEFERRED on cost** — see below |
| 3 | exact-target control + mid-trajectory retargeting | **banked and closed** |
| 4 | Pareto: P3/P4 → P0c → **exactly one** earned front-construction branch → fresh final panel | P3/P4 banked; P0c running |
| 5 | **SA hard-support**, if its frozen census passes | designed; blocked on the Gate-0 mount defect |
| 6 | frozen **cLogP pathwise** confirmation, after checkpoint qualification | frozen; blocked on my checkpoint regression |

**Still queued, not removed — the navigation capstone.** After the final Pareto
controller produces a usable state/front map: can a newly requested tradeoff be
reached more effectively by **selecting and continuing from an already-realized
mapped or intermediate state** than by restarting from `x₀`? **Do not run it
before the map exists** — and note the bar is higher than the banked retargeting
result, which already established that an intermediate state retains value after
a goal change. What is new here is *choosing which* mapped state to reuse.

**Barred additions:** new objective pairs, new scalarization families, new
constraint classes, architecture changes, baseline zoos.

> **One experiment = one question. One primary contrast. One stop rule. No rescue
> unless preregistered.** A negative result closes a branch; a positive one is
> banked and we move on.

---

## Why DDSBM was stopped, and what it exposes

Launched 01:41, stopped 01:51. **Actual spend ~$4–5**, because most containers
were still in the ~8-minute Active8 authentication walk and no checkpoint had
fired. Nothing was lost.

**Projected full cost: ~2 hours wall, ~390 core-hours, roughly $50–80.** 24
containers × 250 sources × 6 kernel calls × ~4.5–7 s. For a single
reviewer-defense table that is a bad trade.

### The cost is the kernel, and the kernel has never been profiled

This is the point that matters beyond DDSBM. **Every remaining run pays the same
per-enumeration cost:**

- the fresh final Pareto panel
- the SA hard-support experiment
- the cLogP n=48 confirmation
- DDSBM, if it is ever run

Measured earlier: **the kernel is ~80% of wall time** (~393 enumerations ≈ 46 min
against ~12 min of scoring). Within one enumeration, canonicalization is only
**0.3–0.6 s of ~7 s**. **The remaining ~6.5 s has never been decomposed** between
`runtime.apply` (graph rewrite) and the neural forward pass.

That is an unmeasured 80% of the cost of the entire remaining program.

### The one thing worth doing before any further large run

**A bounded kernel profile.** One enumeration, line-profiled, answering only:
how much of ~7 s is the model forward, `runtime.apply`, canonicalization, and
Python overhead?

- **If the forward dominates**, batching it across marks is straightforward and
  semantics-preserving — potentially a large, safe speedup.
- **If `runtime.apply` dominates**, that is a systems project we probably do not
  need for this paper, and we would know to stop looking.

Cost: minutes. It gates whether the remaining program costs hundreds of dollars
or tens. **It must run on Modal**, because the Gate-0 mount-path defect blocks
local kernel access — the same blocker as the SA census, so one fix unblocks both.

**Standing rule still applies:** no optimization ships into a claim-bearing run
unless it reproduces trajectories, actions and accounting exactly. The scorer
batching episode is why — it failed that gate and was rejected.

### DDSBM's status

**DEFERRED, not cancelled.** Protocol frozen, representability gate passed
5,984/5,984, app written and persistence-complete. It can launch unchanged the
moment the kernel cost is understood — or be dropped, if the profile shows the
cost is irreducible and the reviewer question is judged not worth $80.

**The full 5,984 remains necessary if it runs at all:** `W₁`, FCD and NSPDK are
distributional, and FCD is badly biased at small N, so a subset would not be
comparable to DDSBM's published full-test numbers and would silently demote the
comparison out of tier 1.
