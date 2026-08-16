# MOLLEO Task 3 lane — summary and handoff

**Status: banked and paused.** Two mechanism families closed with diagnoses; the
benchmark harness, oracles and infrastructure are complete and reusable. No
official five-seed run has been made — the official initialization sets are still
sealed.

Everything below is development-only unless it says otherwise. No number in this
lane is Task 3 performance.

---

## The two things a future reader most needs

### 1. Task 3's entire 10,000-call budget buys about three chance encounters with a JNK3 active

Measured over 80,000 random ZINC-250k molecules through the frozen oracle bundle:

| threshold | rate in random ZINC | draws for one by chance |
|---|---|---|
| jnk3 ≥ 0.2 | 0.164 % | ~600 |
| jnk3 ≥ 0.3 | **0.040 %** | **~2,500** |
| jnk3 ≥ 0.5 | 0.005 % | ~20,000 |

So the task is **reachability-limited, not search-limited**: the difficulty is
entering the active basin at all, not choosing well among candidates once inside
it. The QED lane reached the same characterisation independently on a different
task.

This retro-explains most of what the lane measured: why hypervolume correlates
0.992 with best-JNK3-found, why Graph-GA is bimodal, why Graph-GA does not
reliably beat uniform random sampling, and why a surrogate fit on an official
random-120 start is blind — that start contains ~0.05 actives in expectation.

It also bounds experiment design. Any experiment whose primary metric is a binary
activity threshold needs thousands of calls per arm to have power. A 300-call
rung cannot see the effect regardless of the policy under test.
`diagnostics/task3_active_base_rate_BENCHMARK_PROPERTY.json`

### 2. Fixed scalarization beat every archive-aware variant we tried

The plain unweighted sum of the five transformed objectives — MOLLEO's own
scalarization — is the best proven selection rule on this task at these budgets,
and is the control going forward. Three archive-aware alternatives were built and
all three lost or failed to engage. Details in *What was closed*.

---

## What was established, and is reusable

### The frozen five-oracle bundle
`artifacts/oracles/molleo_task3_v1/` · `src/compose_v4/benchmark/oracles/`

Opened once under a pinned scikit-learn 1.2.2; the runtime imports no sklearn and
unpickles nothing. Evaluates in plain numpy.

**The correction that mattered most in this lane.** The obvious local pickles
(`kinase_rf/{jnk3,gsk3b}.pkl`) are **1024-feature** forests — the HN-GFN/MARS
variant. MOLLEO calls TDC, which builds
`GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)` and scores a **2048-feature**
forest. Two incompatible models circulate for the same two tasks and the wrong one
produces a plausible, wrong oracle rather than an error. The extractor now refuses
any other width.

Parity is kept as **two separate claims**, and conflating them is how a wrong
oracle passes:

- *Implementation parity* (fatal if it fails): jnk3/gsk3b bit-identical to
  sklearn's `predict_proba` over 400 molecules, drd2 4.4e-14, QED and SA
  bit-identical to RDKit's own reference implementations.
- *Version drift* (measured, not assumed): RDKit's stereo perception changed
  between 2023.09 and 2025.09, moving SA's `log10(n_stereocentres + 1)` term.
  0/5000 ZINC molecules move; 37/4253 kinase actives move (0.87 %), max 0.66 raw
  SA.

Orientation is pinned against **known actives**, not agreement on inactives —
actives separate from ZINC background by +0.82 (jnk3) and +0.84 (gsk3b), so an
inverted objective fails loudly.

**Provenance is "probable, parity pending."** The pickles match TDC's contract on
every checkable structural detail and sit at TDC's own relative path, but
byte-identity to TDC's hosted files is unverified — Harvard Dataverse answered 156
requests with an AWS WAF challenge. Settling it takes a browser download of
datafiles 4170293 and 4170295 and two sha256 comparisons.

### The strict meter, and the audit of the released implementation
`src/compose_v4/benchmark/molleo_task3.py`

One novel canonical molecule receiving the five-objective vector = one budget
unit. Canonicalisation is part of the rule, reproduced from their `score_smi`.

**The released code evaluates far more molecules than it charges.** Measured by
running their repo (`fd138a7`) with the objectives stubbed and every evaluator
recorded — `select_pareto_front` calls the five `tdc.Oracle` objects directly,
bypassing the metered path:

| budget | metered | evaluated (unique) | ratio |
|---|---|---|---|
| 300 | 319 | 814 | 2.55× |
| 500 | 519 | 1,786 | 3.44× |
| 1,000 | 1,015 | 4,804 | 4.73× |
| 2,000 | 1,337 | 7,256 | 5.43× |

The ratio **grows with budget** (marginal 9.3× by generation 104). Two further
leaks compound it: `clean_buffer()` makes the budget guard per-generation so it
never fires, and survivors are re-evaluated each generation uncharged.

Consequence: our harness is **strict-10k**, not "exact MOLLEO Task 3", and a
strict-10k number must not be placed beside their published figures as if the
protocols matched. Strict-10k is closer to the paper's *stated* semantics
(objective evaluation dominates cost; methods compared at equal call budgets)
than reproducing the released path would be.

### The hypervolume implementation
The paper's Eq. 5 — volume of the union of hyperrectangles from the origin in
normalised [0,1]^n. Implemented as a deterministic Sobol integration, so two
policies are measured against the identical sample points and a result is
recomputable from an archive alone. **Qualified against the definition** by exact
inclusion–exclusion: worst absolute error 3.5e-05, where the differences acted on
are of order 0.09.

What Eq. 5 does not fix is which *set* the front is taken over; we use everything
evaluated (monotone, recomputable from the ledger, most generous to baselines).
**Published-number comparability is therefore PENDING**, and protocol identity is
not something to claim.

### The durability layer
`src/compose_v4/benchmark/run_store.py` · `task3_run.py`

Append-only ledger (the budget is *derived* from it, never restored from a saved
counter), atomic checkpoints with a previous-generation fallback, and a sha256 of
the ledger prefix linking them. Exactly one corruption is forgiven — a truncated
final line, which is what a process killed mid-write leaves.

**Proven three times against real failures**, not simulated ones: a Modal
preemption mid-run (replacement resumed to 300 contiguous records), and two of my
own operational errors that killed 2-hour runs at ~75 % completion. Nothing was
lost in any of them.

### Baselines, banked
Eight development seeds, both policies from identical frozen initialization sets,
full 10,000 calls:

- graph-ga mean HV 0.4195 (sd 0.1842); random-zinc 0.3331 (sd 0.0593)
- paired +0.0864, **4/8 seeds**, Wilcoxon p = 0.312

**Graph-GA does not reliably beat drawing 10,000 random drug-like molecules.**
corr(HV, best JNK3) = 0.992; Graph-GA is bimodal — four seeds latch onto an
active scaffold, four never do, and in the second group it finishes below uniform
sampling. `diagnostics/task3_dev_baselines.json`

### The surrogate/acquisition kind-mismatch finding
`diagnostics/task3_surrogate_acquisition_kind_mismatch.json`

**General, and the most transferable thing this lane produced.** If an
acquisition's value is zero unless the prediction exceeds the incumbent frontier
(marginal hypervolume, probability of improvement, EI-over-max), then *any*
convex-combination estimator is mathematically incapable of firing it — a k-NN
mean, kernel regression, a bagged average, attention-weighted retrieval. None can
exceed the largest label they average. Nothing errors; the acquisition returns
near-zero for everything and degrades into its tie-break.

**One-line detector:** the fraction of predictions exceeding the incumbent front
on at least one axis. Near zero means no amount of tuning the acquisition will
help.

Measured here at zero cost: 100 % of predicted vectors interior to the archive
box, 0.9 % of candidates with any predicted gain, predicted max JNK3 0.539 against
an archive max of 0.720. Note that *shrinkage is the wrong diagnosis* — variance
was inflated on two axes. The decisive quantity is the ceiling, not the spread.

### Infrastructure worth reusing
- **Initialization sets** sealed before any policy existed; `official_init_set()`
  raises unless passed `official=True`. Development sets are disjoint by
  construction.
- **R_θ expansion cost**, measured on the volume: **12.5 s median, ~613 legal
  successors each**. This pushes policy design toward *expand once, choose among
  hundreds* rather than many cheap walks.
- **Fiber cache** with a determinism gate (exact fiber equality on successors
  *and* R_θ probabilities, passed on three seeds).
- **The Active8 lineage guard hashes an absolute filesystem path.** The local
  corpus is content-identical; it fails on *location*, not content. Run where the
  volume is mounted. Recorded in `scripts/pareto_local_runtime.py`.

---

## What was closed, and why

### Archive-aware purpose allocation — CLOSED
`diagnostics/task3_archive_acquisition_family_CLOSED.json`

| # | mechanism | outcome | cost to find out |
|---|---|---|---|
| 1 | dominance-shaped aspiration | never engaged — practically unreachable | two charged experiments |
| 2 | marginal-HVI over a k-NN mean | mathematically inexpressible | **zero** (offline smoke) |
| 3 | reachable-HVI with optimism | fully engaged, lost 3/3 by 3.4× | $0.094 |

The third is decisive *because* it fired: 100 % divergence (the arms differed on
every decision) and still well behind. That makes the loss a result about the
acquisition, not about its plumbing — which the first two could never have told
us.

Untested hypothesis for why: marginal HVI is greedy and myopic, buying isolated
extremes, while a scalar sum buys well-rounded molecules — and since the archive
is also the source of start states, isolated extremes may degrade future
expansion. Consistent with the adaptive arm's collapsing front counts (28, 13, 9
against 35, 42, 53); not established.

### Phase A, exploration vs fixed scalarization — CLOSED at this budget
`diagnostics/task3_phase_a_discovery.json`

At equal achieved budget (1840 / 2036 / 2160 calls, both arms truncated to the
smaller), from an official-protocol random-120 start on development seeds:

- actives by search: **0, 0, 0** for both arms
- expected by chance 2.06 per arm; difference 0 against a Poisson sd of 2.03

**A hypothesis of mine was refuted here.** I predicted the control found nothing
because it re-expands one state on ~98 % of decisions and so buys fewer effective
independent draws. Novelty-exploration spreads far more (31/40 distinct starts
against 1/40, 0.84 scaffolds per pick against 0.21) and found exactly as many:
zero. It also moved best-JNK3 less and had lower early HV on every seed.
Concentration was not what suppressed discovery.

### The off-manifold explanation — REFUTED in its explanatory form
`diagnostics/task3_offmanifold_check.json`

| measurement | ZINC | successors | gap |
|---|---|---|---|
| nearest-neighbour to ZINC | 0.406 | 0.294 | +0.112 |
| to the JNK3 training set | 0.345 | 0.286 | +0.060 |
| **to the JNK3 actives** | 0.250 | 0.236 | **+0.014** |

R_θ successors *do* drift off the ZINC manifold (a 28 % relative drop), but they
are **not** further from the JNK3 actives, and their closest approach is nearer
than ZINC's (0.694 vs 0.633). Drift that does not move search away from the
active region cannot explain a failure to reach it.

**And the anomaly it was invoked to explain was probably not real.** The p = 0.017
came from a Poisson calculation assuming independent draws; edit-based search
violates that badly, since its calls are correlated neighbours of a few states.
With a realistic effective sample size the expected count falls below one and
observing zero is unremarkable.

**Scope, because the headline sentence invites over-reading.** This concerns Task
3's *fitted* oracles (RFs and an SVM over Morgan bits) and edit-based search under
them. It is **not** about COMPOSE or R_θ generally, and does not transfer to the
QED lane — QED and Tanimoto are *computed* properties, defined for any valid
molecule, so they cannot go off-manifold and no analogous failure mode exists.

---

## Open questions

1. **Why edit-based search underperforms uniform random sampling on this oracle.**
   No established mechanism, and the anomaly may not need one. Worth revisiting
   only with an effective-sample-size-aware power calculation.
2. **Whether Task 3 is the right flagship external MOO benchmark**, given it is
   reachability-limited and its whole budget buys ~3 chance actives.
3. **Phase A at genuinely large budget** — the question is real; the rung sizes
   tried could not see it.
4. **TDC byte-identity** — a browser download settles it in two minutes.
5. **MOLLEO LLM variants** — the released `GPT4.py` has three syntax errors and
   needs a paid key. Not attempted.
6. **RDKit version pinning** for the SA drift (0 % on ZINC, 0.87 % on exotic
   cohorts).

---

## What someone resuming needs to know

**Run where the volume is mounted.** The lineage guard hashes an absolute path;
a local mirror is content-identical but cannot satisfy it. Do not modify the
lineage system to accommodate a mirror.

**`modal run --detach`, and never wrapped in `timeout`.** The check that it worked
is the State column reading `ephemeral (detached)` after the client exits. I lost
two multi-hour runs to this — once by omitting `--detach`, once by wrapping it in
a `timeout` that SIGTERMed the client. Both recovered only because of the ledger.

**Cost model** (verified): $0.04716/core-hour, $0.007992/GiB-hour. Memory is
billed on the greater of requested and actual — measured peak for R_θ expansion is
**3.9 GiB**, so request 5–6 GiB, not 8. `ru_maxrss` is kilobytes on Linux.

**The official sets are sealed** and `official_init_set()` raises without
`official=True`. Development seeds are 100–107.

**A failure pattern worth carrying to other lanes:** *sorted or grouped data
sampled by position*. It produces numbers rather than errors. In this lane, taking
the first 8,000 rows of `kinase.tsv` as "the kinase training set" yielded 2,668
GSK3β actives and 5,332 GSK3β inactives with zero JNK3 molecules — answering a
question about the wrong target while appearing to answer the right one. The QED
lane hit the same class twice on the same day.

**Test suite:** 103 tests across `tests/test_molleo_task3_*.py` and
`tests/test_task3_*.py`.

---

## Total spend

About $2.50 of Modal across the whole lane. Every mechanism iteration after the
fiber cache was free — offline smokes and local runs against cached fibers. The
most valuable single result (the kind-mismatch finding) cost nothing, because the
smoke checks whether a mechanism *engages* separately from whether it *wins*.
