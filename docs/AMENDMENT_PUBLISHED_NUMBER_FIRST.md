# Amendment after Lane 5/6 and P3 feasibility — published-number-first

**Project-wide. Binds Lane 2, Lane 3, Lane 5, Lane 6 and the main lane.** Extends
`docs/COMPARATOR_ROLES_CANONICAL.md`; that role taxonomy is unchanged and remains
in force. Nothing here reopens a frozen decision.

---

## 1. The published-number-first policy

> **Use published baseline results whenever we can reproduce the exact published
> evaluation protocol on COMPOSE. Rerun an external method only when a direct
> comparison genuinely requires it.**

Not *"every baseline in our table must be rerun by us."* **Actual baseline
execution is the exception, justified by the scientific question — never the
default reflex.**

| tier | condition | action |
|---|---|---|
| **1 — exact alignment** | task, oracle, query budget, metric, normalization/reference point, validity/canonicalization, seed aggregation and applicability rules all align | **run COMPOSE only**; cite baseline numbers from the original papers, labelled **"reported"**, never "our rerun" |
| **2 — partial alignment** | some protocol details differ | published values may appear as **contextual literature numbers**, never as head-to-head statistical comparison |
| **3 — rerun** | the question is on our custom task **and** a published method natively supports it through a thin adapter | authorize a rerun, of **the smallest necessary set** |
| **4 — do not reconstruct** | code broken, checkpoints absent, licenses restrictive, or adapting requires method invention | **cite and discuss.** Do not spend a week rebuilding someone else's method |

**Before implementing or rerunning any external baseline, audit whether COMPOSE
can instead be run under that baseline's published protocol.** Do not rerun a
method merely to reproduce its published table.

### Why this rule earned its place today

Lane 5 found that citing HN-GFN and InversionGNN as if their reported numbers
were interchangeable would be **scientifically wrong**: the two papers attribute
**70,000 versus 1,000** oracle calls to HN-GFN and use incompatible oracle
representations (1024-bit ECFP4 against its own forests versus TDC's 2048-bit).
The correct response was neither to cite them anyway nor to spend 10 V100-hours
reproducing everything — it was to establish that the comparison does not exist
yet.

Lane 6 found the same shape for CDD: the useful object is the **published
constraint definition**, not a reconstruction of an implementation whose repo is
a 59-byte placeholder README.

### Where actual runs are still expected

- **General source-conditioned editing** — if DDSBM or GraphXForm have no
  published results on our exact sources, a native rerun is meaningful. Smallest
  necessary set.
- **The custom COMPOSE Pareto task** — HN-GFN's global de-novo front cannot sit
  beside our per-source fronts. Either run an external method natively on the
  exact task if it is easy and faithful, or state that no direct comparison
  exists. **Our P3 matched generate-and-rank contrast is more informative for the
  COMPOSE-specific claim anyway.**
- **Retargeting / exact-target / pathwise** — custom capability experiments. **Do
  not bend published methods into unnatural variants to manufacture a
  leaderboard.** Internal matched controls are the correct evidence.

---

## 2. P3/P4 — the fair matcher is operational. Continue exactly as frozen

Source 000 **`MATCHED`**: the committed 368-kernel-call target reached at 369,
after **1,025** generate-and-rank trajectories — **16.8× more search** than the
broken open-loop matcher gave it. The fair comparison exists.

Let the remaining eleven finish under their own committed targets. **Do not alter
the matcher or any COMPOSE artifact.**

**The accounting defect is bookkeeping, not science.** The new top-up
implementation scores each candidate once, so the old correction would subtract a
trajectory again and falsely report **zero** algorithmic oracle demand. Make
`algorithmic_oracle_requests` implementation-aware and derive it **offline**.
**Do not rerun claim-bearing trajectories for an accounting-only correction** —
the matcher targets kernel calls, so this does not touch the generation
experiment.

## 3. Global MOO Panel B — `BLOCKED_ON_QUERY_EFFICIENT_COMPOSE`

**Not `BASELINE_UNAVAILABLE`.** The distinction matters: the blocker is ours.

Pause external baseline execution and dependency archaeology. Do not pick a
different benchmark hoping something aligns — full-fiber COMPOSE spends **~2,187
oracle evaluations per preference trajectory on its cheapest guided arm and
~53,977 on its primary**, against a benchmark class living at 10³–10⁴. **No
choice of benchmark fixes a 5–50× budget-class mismatch.**

## 4. `R_θ` shortlisting — upgraded to `CONDITIONAL_LOAD_BEARING_FOR_ORACLE_BENCHMARKS`

Previously "optional scaling upside after Pareto works." Revised.

It is **still not needed** to establish the core scientific capability — that one
frozen process can be recontrolled across preferences. But if the paper also
wants to claim *COMPOSE is a credible ordinary multiobjective optimizer under
contemporary oracle-budget evaluation*, some form of query-efficient control is
now **load-bearing**.

**Ordering is unchanged.** Finish repaired P3/P4 → finish the full-fiber larger
Pareto development sufficient to establish capability → **then** run exactly one
preregistered efficiency experiment:

```
exact legal canonical fiber → R_θ-prioritized shortlist → expensive objective evaluation → same control
```

**One operating point, chosen before outcomes from a target query reduction
(~10×). No K sweep.** Primary result: **HV retained per oracle query retained**.
*90% of full-fiber front quality at 10% of objective queries* would close the
architecture — legality → plausibility → purpose — and make the conventional MOO
comparison feasible.

**If shortlisting fails badly that is informative too:** COMPOSE remains a rich
control framework, and we do not market it as oracle-efficient MOO.

## 5. Hard constraints — reuse CDD's task, not CDD

**Do not rebuild CDD. Do not invent another endogenous scaffold.**

CDD's published `SA(y) ≤ τ` predicate and thresholds may be reused as an
**externally defined task** if reproduced verbatim — valuable precisely because
the constraint definition is then independent of COMPOSE, so we are not shopping
our own corpus for a constraint that happens to work. **CDD itself remains
contextual, non-head-to-head**, since its released implementation and scoring
protocol are not reproducible enough.

The internal causal comparison is unchanged: **post-hoc filtering vs soft
guidance vs exact hard-support restriction**, same `R_θ`, objective, sources and
budget. Load-bearing outcomes are feasible objective gain, feasible-return yield,
retained support and resource use. **Constraint satisfaction of the hard arm is
definitional.**

**Present the trilemma, not a leaderboard.** Methods trade among broad constraint
classes, guarantees by construction, and avoiding explicit support enumeration:
CDD attains zero reported violations but can sacrifice valid yield; ConStruct
obtains structural guarantees for a restricted family; COMPOSE applies arbitrary
evaluable predicates to an enumerable executable fiber and pays enumeration cost.
That is far better than *"COMPOSE 87%, CDD 82%."*

## 6. Unchanged and closed

Bemis–Murcko remains a closed negative — **no smaller-core rescue**. C0 and the
oracle batch-size defect remain closed and superseded — **no requalification
absent an actual action- or outcome-changing defect.** The role taxonomy stands.

---

## Why two negative qualification reports were a good day

They removed two pieces of likely-wasted work: expensive external MOO
reproduction that would not have been interpretable, and reconstruction of
hard-constraint baselines whose published setup is not reproducible.

And source 000 matching is a major positive: **P3/P4 will be genuinely fair, not
merely less unfair.**

The only substantive addition to the roadmap is that query-efficient plausibility
shortlisting matters more than we thought — **not to rescue the Pareto science,
but because it is required for the separate claim that COMPOSE can compete in
ordinary oracle-budget multiobjective optimization.**
