# The final fairness matrix — method × task

**Artifact status: `DESIGN_ONLY`.** Frozen before any COMPOSE-versus-baseline
outcome. Tasks are defined in `CONVENTIONAL_SUITE.md`; capability evidence is in
`comparator_registry_v3.json`.

Three verdicts, and **`inappropriate` is a first-class answer**:

| verdict | meaning |
|---|---|
| **native** | the method's own scientific object matches the task; run it with published defaults and report the deviation list as empty |
| **adapted** | runnable only through a declared modification that changes the method's behaviour; **every adaptation is listed in the cell** and reported with the result |
| **inappropriate** | the task is outside the method's scientific object. Report `N/A`. **Do not run it.** An honest `N/A` is worth more than a strained comparison a reviewer will dismantle |

`N/A` is never reported as a COMPOSE win.

---

## The matrix

| method | T1 single-property | T2 multi-property | T3 similarity-constrained | T4 topology/cardinality | T5 PMO aggregate |
|---|---|---|---|---|---|
| **COMPOSE** | native | native | native | native | adapted¹ |
| **GraphGA** | adapted² | **inappropriate**³ | adapted⁴ | adapted² | **native** |
| **REINVENT 4** | adapted⁵ | adapted⁵ | adapted⁶ | **inappropriate**⁷ | native (2.0-era via PMO) |
| **MARS** | adapted⁸ | adapted⁸ | adapted⁸ | adapted⁸ | **inappropriate**⁹ |
| **GraphXForm** | adapted¹⁰ | adapted¹⁰ | adapted¹⁰ | **inappropriate**¹¹ | **inappropriate**¹² |
| **DDSBM** | **inappropriate**¹³ | **inappropriate**¹³ | **inappropriate**¹³ | **inappropriate**¹³ | **inappropriate**¹³ |
| **HN-GFN** | **inappropriate**¹⁴ | **inappropriate**¹⁴ | **inappropriate**¹⁴ | **inappropriate**¹⁴ | **inappropriate**¹⁴ |

Six of thirty-five cells are `native`. That is the honest shape of this
comparison and it should be stated rather than hidden behind a full table.

---

## Cell notes

**¹ COMPOSE on T5.** PMO's tasks are de-novo with no designated source; COMPOSE
is source-conditioned. Running it requires a seeding convention, which is a
declared adaptation. Its numbers are comparable to PMO's leaderboard only under
`unique_valid_canonical_evaluations`.

**² GraphGA on T1/T4.** *Adaptations:* (a) `crossover.average_size` and
`crossover.size_stdev` must be set — they are undocumented required globals and
impose a soft Gaussian size prior on offspring; we set them from the panel, not
from upstream's ZINC values (39.15/3.50), and report both. **Measured:** the
choice moved mean endpoint heavy-atom count from 14.65 to 16.25 on a 5-source
held-in smoke. (b) A wall-clock guard is required: `reproduce()` loops unbounded
and failed crossovers cost no oracle calls, so an oracle budget does not bound
runtime. (c) The source is not preserved — **measured** median nearest-seed ECFP4
Tanimoto **0.236** on held-in sources, matching the paper's own reported 0.27.

**³ GraphGA on T2 — `inappropriate`, and this was measured, not assumed.**
GB-GA's roulette selection computes `score/sum(scores)` and feeds it to
`np.random.choice(p=...)`, so the objective must be non-negative; upstream's own
objective clamps for exactly this reason (`logP_max` returns `max(0.0, score)`).
The T2 developability margin is **negative for the entire initial population** —
that is why headroom exists — so upstream's clamp maps every individual to 0.0
and `calculate_normalized_fitness` raises `ZeroDivisionError: division by zero`.
Verified in `diagnostics/baselines/graph_ga_held_in_smoke.json`
(`upstream_clamp_probe`).
A constant shift makes it *run*, but it changes the fitness **ratios** that
roulette-wheel selection acts on, i.e. it changes the search, on a task where the
scoping already says the comparison is not claim-bearing. **Do not run it. Report
`N/A` with this reason.**

**⁴ GraphGA on T3.** The similarity floor is a constraint and GraphGA has no
constraint mechanism, so it can only be folded into the scoring function — which
changes the task from "optimize subject to a constraint" to "optimize a different
objective". Declared adaptation; report the folded form explicitly.

**⁵ REINVENT 4 on T1/T2.** *Adaptations:* a COMPOSE objective as a custom scoring
component; Mol2Mol input for source conditioning, which does **not** preserve the
source ("the scaffold can change within the limits of the given similarity");
per-objective RL fine-tuning, whose oracle cost is the entire budget and must be
reported. Its native caching means its raw counts are not comparable to MARS's
without naming the counter.

**⁶ REINVENT 4 on T3.** LibInvent/LinkInvent preserve a supplied fragment **by
construction**, which is the closest native fit to a similarity constraint in the
whole registry — but they need a scaffold with explicit attachment points, not an
arbitrary source molecule. `MatchingSubstructure` is a **soft 2× penalty**
(`0.5 * (1.0 + match)`), not a constraint. Declared adaptation either way.

**⁷ REINVENT 4 on T4 — `inappropriate`.** Generation is autoregressive over SMILES
tokens; there are no intermediate molecular states and no notion of an edit
relative to a source, so "changed the ring count by at least 1 relative to the
source" has no referent inside the method. It can be *measured* post hoc on the
endpoint, but the task is about the edit, not the endpoint.

**⁸ MARS on T1–T4.** *Adaptations:* (a) objective shim — the shipped DRD2 scorer
is commented out of its own imports and its model file is not shipped, so the
oracle must come from our side regardless; (b) chain-count reduction — MARS's
designed regime is ~10⁶ molecule scorings per run and a matched budget of ~10³
runs it three orders of magnitude below the setting its online-trained proposal
needs, so **both a matched-budget row and a native-regime row are required**;
(c) a trajectory logger, since MARS keeps no history; (d) a shared-semantics
oracle cache, since MARS has none and rescores its current molecule whenever a
proposal is invalid; (e) a one-line `rdkit.six` fix in `sa_scorer.py`.
**Environment: FEASIBLE-WITH-WORK, verified** — python **3.11 only** (DGL
publishes no wheel for 3.14), torch 2.1.2 + torchdata 0.7.1 + DGL 2.2.0 from
`data.dgl.ai`; `Set2Set` and `number_of_edges()` both still work; `BasicEditor`
instantiates at 2,618,026 parameters.
For T3, MARS has no constraint mechanism, so the similarity floor is a rejection
filter we add — a modification, reported as such.

**⁹ MARS on T5 — `inappropriate`.** PMO's protocol is a 10,000-call budget with
AUC top-10. MARS's native regime is ~10⁶ scorings and it has no cache; placing it
in PMO's table means running it at 1% of its designed budget and reporting the
result as its performance. If a PMO-style MARS number is wanted, take it from the
literature and label it context-only.

**¹⁰ GraphXForm on T1–T3.** *Adaptations:* per-objective fine-tuning (head-only,
but gradient updates); `start_from_smiles` for source conditioning — this one is
genuinely native and is the cleanest source-conditioned entry point in the
registry; atom-vocabulary alignment to the COMPOSE element set; trajectory export
at action level 0, the only points where a complete connected molecule exists.
**Environment: FEASIBLE, verified** — `torch_scatter` compiled successfully
against CPU torch on this machine (the expected blocker did not occur),
`MoleculeTransformer` instantiates at 31,542,218 parameters, checkpoint URL
returns 200 with Content-Length 347,364,546 (331 MiB).
For T3 the similarity floor cannot be masked — there is no SMARTS or substructure
matching anywhere in the released code — so it is a terminal filter.

**¹¹ GraphXForm on T4 — `inappropriate` for the deletion arm.** The action space
is strictly `AddAtom`, `AddBond`, `DontChange`; the paper defers removal to
future work. A task requiring a cardinality **decrease** is unreachable, not
merely hard. Report as a scope result, never as a loss.

**¹² GraphXForm on T5 — `inappropriate`.** The paper explicitly declines an
oracle-call budget ("executed until convergence or until a maximum wall-clock
time of eight hours"), arguing its objectives are cheap. Forcing PMO's
10,000-call budget runs it outside its designed protocol.

**¹³ DDSBM — `inappropriate` on everything.** Three independent reasons, any one
sufficient. (a) **No LICENSE file exists** — verified by exhaustive search;
all-rights-reserved by default, which is a legal blocker, not a technical one.
(b) No checkpoints are released and training is 6 IMF iterations × 300 epochs on
four GPUs. (c) It makes **zero oracle calls at sampling time**, so an
oracle-budget comparison has no shared axis at all. Its native object is a bridge
between two distributions, not source-conditioned optimization. The only honest
use is a native-protocol reproduction, and that is gated on the license.
*(Its dummy-atom birth/death semantics were verified for the related-work matrix
and confirmed: `"X"` is an ordinary, non-absorbing class in a uniform transition
matrix, so C→X and X→C both have positive probability.)*

**¹⁴ HN-GFN — `inappropriate` on this suite, and that is not a demotion.** It
generates from scratch over a 105-fragment vocabulary and **cannot be conditioned
on a source molecule at all**, so it cannot enter T1–T4, which are all
source-conditioned. Its true-oracle budget is spent against a **surrogate** inside
a Bayesian-optimization loop, so a budget comparison on T5 would be invalid in
both directions. Its correct home is the Pareto experiment — see below.

---

## HN-GFN and Lane 4 — DECIDED 2026-08-13: it runs native

HN-GFN belongs to the **Pareto/preference experiment** (branch
`codex/compose-pareto-control`), not to this conventional suite. Its adapter
should fit whatever preference interface that lane freezes. One mismatch must be
settled before an adapter is written:

> **Lane 4 has frozen a Chebyshev scalarization. HN-GFN is natively a LINEAR
> WEIGHTED SUM.**

Verified in source: `main.py` lines 226–232 compute
`raw_reward = (weights*score).sum()` under `--scalar WeightedSum`, which is the
argparse **default** (`main.py:54`). The MOBO entrypoint ignores `--scalar`
entirely and selects by `--acq_fn`, whose default `UCB` is also linear
(`proxy/proxy.py:198-199`, with the author's own comment
`# weighted_sum scalarization`).

Two further details that matter for a faithful comparison:

- HN-GFN's opt-in `Tchebycheff` branch is **not** classical Chebyshev. It is an
  augmented max-min, `min_i(w_i·r_i) + 0.1·Σ_i(w_i·r_i)`, with **no ideal point
  and no absolute deviation**, and a hardcoded augmentation coefficient of 0.1.
  Classical weighted Tchebycheff is `max_i(w_i·|z*_i − r_i|)`, minimized.
  Selecting it does not reproduce a standard Chebyshev either.
- The preference vector conditions **only the output heads** via a hypernetwork
  (`model_pred_hyper.py`); under the default `version='v4'` the message-passing
  trunk is preference-independent.

### The decision — binding

> **HN-GFN runs under its native published preference conditioning and linear
> scalarization. We do NOT supply a Chebyshev to match Lane 4's.**

Modifying a competitor's internal formulation to match ours is not fairness; it
is changing the competitor, and a reviewer will read it that way. HN-GFN's
official optional augmented-Tchebycheff mode may appear as a **clearly labelled
secondary configuration**, never as a replacement for its default.

### Compare in a common OUTCOME space, not a common scalarization

The fair question is *"given the same objectives and the same evaluation budget,
what Pareto set does each method produce?"* — not *"can every algorithm be made
to optimize our internal scalarization?"*. So the comparison is on outcomes both
methods genuinely produce:

| reported quantity | why it is common ground |
|---|---|
| hypervolume | defined on the objective vectors, independent of how either method scalarized to get there |
| Pareto coverage | ditto |
| nondominated-set quality | ditto |
| preference coverage | how much of the requested tradeoff space each method actually reaches |
| oracle usage under **all three counters** | `unique_valid_canonical_evaluations`, `oracle_requests`, `evaluator_calls` |

Objectives and budget are matched; the scalarization is not, and is reported.

> **Binding reporting rule.** Linear scalarization **cannot recover concave
> regions of a Pareto front**; Chebyshev can. Therefore **any coverage
> difference in a concave region is a METHOD PROPERTY of linear scalarization
> and must be reported as such — never as a COMPOSE win.** If COMPOSE covers a
> concave region that HN-GFN misses, the correct sentence names the cause:
> weighted-sum scalarization cannot reach it by construction.

The surrogate asymmetry still applies on top of this: HN-GFN's true-oracle budget
is spent against a learned proxy inside a Bayesian-optimization loop, so the
oracle-usage row must disclose the surrogate rather than compare raw totals.

---

## What this matrix refuses to do

- Force every baseline into every task. Twenty-nine of thirty-five cells are not
  `native`, and six are `inappropriate` for reasons no amount of engineering
  fixes.
- Report an `inappropriate` cell as a COMPOSE win.
- Run T2 for GraphGA to produce a number, when the method's own selection
  operator cannot accept the objective.
