# Workstream D — External-baseline qualification: scientific protocol

**Artifact status: `DESIGN_ONLY`.** Nothing in this document is a result. It is
the contract under which external comparators may later be run.

---

## Claim this workstream can support

> For each external method, a *verified* statement of which COMPOSE claim it is
> a valid comparator for, under what matched-budget definition, and where the
> comparison stops being fair.

That is a **methodological** claim about comparability. It is deliberately not a
claim about which method wins.

## Claims this workstream cannot support

- No claim that COMPOSE beats any baseline. No sweep was run and none is
  authorised here.
- No claim that a baseline is weak. Absence of a native capability is a scope
  fact, not a quality judgement, and several of these methods are stronger than
  COMPOSE on their own native task.
- No claim from published numbers. Literature numbers enter this registry only
  as `context_only` and may never be placed in a COMPOSE comparison table
  without full protocol matching (§ Fairness contract).

---

## Which COMPOSE claim each comparison is against

From `docs/EXPERIMENT_PLAN.md`, the four claims:

| id | claim | can an external baseline test it? |
|---|---|---|
| C1 | learned reference beats legal/random/empirical proposals | **No.** C1 is a likelihood statement about `R_theta` on the COMPOSE executable support. An external method does not share that support or that training target, so it cannot be an arm. Externals are `N/A` here by construction, not by choice. |
| C2 | the reference process produces healthy executable trajectories | **No**, for the same reason. Internal-arm territory (Workstream B). |
| C3 | bridge control is genuinely future-aware | **No.** C3 is an internal comparison against greedy / Boltzmann / SMC / exact Doob on one frozen kernel. |
| C4 | the architecture buys capabilities static optimizers lack | **Yes — and only here.** This is the entire reason external baselines exist in this program. |

C4 decomposes into four *separately qualifiable* sub-capabilities, and a method
may qualify for one and not the others:

| id | sub-capability | COMPOSE status |
|---|---|---|
| C4a | static source-conditioned optimization under a matched oracle budget | competitive result wanted, not a novelty claim |
| C4b | exact-target recovery from a source under a finite edit budget | **done, sealed, closed** (greedy 26/65, verified rollout 40/65, learned top-1 + verification 40/65 at ~35% of continuation evaluations) |
| C4c | same-prefix dynamic retargeting — goal changes mid-trajectory, continue from the realized molecule | in flight |
| C4d | pathwise constraints — a motif preserved at every intermediate state | planned |

**The frozen-process property is the actual claim.** `R_theta` is not retrained
between C4a, C4b, C4c and C4d. A baseline that can do C4c or C4d only by
retraining per objective does not refute the claim; a baseline that can do them
*without* retraining does. The registry must record retraining requirements
precisely for exactly this reason.

**"Future-aware control helps" is scoped, not universal.** It is established on
C4b (greedy 26/65 → verified rollout 40/65, sealed) and *measured absent* on an
easy target-free property goal (greedy 28/30 = verified 28/30, binary headroom 0,
gate CLOSED). This lane must not design a comparison that reads as testing the
universal version. The full scoping, with artifacts, is in the registry's
`compose_claim_scoping` block and in `FAIRNESS_CONTRACT.md` §0a; it is the reason
the C4a table is classified as a competitiveness sanity check rather than a
claim-bearing test.

---

## The manuscript rule — QUALIFY BROADLY, PRESENT NARROWLY

Binding, and it governs everything below.

Keep qualifying every method: that is what protects us from reviewer surprises
and lets the paper choose intelligently later. But **a method does not enter the
paper because its adapter works. Sunk engineering effort is not a reason for a
baseline to occupy a figure.** The final comparator set is chosen by which
scientific question each method answers, and by nothing else.

### Reviewer-facing hierarchy

| tier | method | the question it answers |
|---|---|---|
| **PRIMARY** | **GraphXForm** | modern learned graph editing/generation |
| **PRIMARY** | **DDSBM** | modern stochastic graph transformation — blocked on licence, `CONTEXT_ONLY` until that resolves |
| **PRIMARY** | **HN-GFN** | Pareto / preference control, coordinated with Lane 4 |
| **ANCHOR** | **GraphGA or REINVENT 4** — *one* of them, whichever fits the task | conventional goal-directed optimization |
| **DEMOTED** | **MARS** | "why isn't this just a graph-editing optimizer?" — it is sequential graph editing, structurally the closest method to COMPOSE |

**MARS is not the method that carries a claim of modern generative
competitiveness.** Beating a 2021 method proves nothing about the state of the
art. Include it where its sequential-edit semantics make the comparison
scientifically useful; otherwise it belongs in a supplementary table. Its
correctness work is finished anyway, because the scikit-learn silent-corruption
finding proved why that work mattered.

### Selection rules

1. Select a baseline because it is **strong, contemporary and relevant** — never
   because COMPOSE is likely to beat it.
2. Freeze the task-specific comparator set **before** viewing any
   COMPOSE-versus-baseline outcome.
3. Give every method its **native algorithmic machinery**; compare in common
   **outcome** and **accounting** space.

## The success criterion — state it before any number exists

> **COMPOSE does not need to win the standard optimization table.** The table
> answers whether COMPOSE remains a credible molecular-design method despite
> being built for richer process-level control. GraphXForm best on one
> conventional task and REINVENT best on another, with COMPOSE competitive
> across the set, is a perfectly good outcome — and more credible than a table
> where we somehow win everything.
>
> What **would** be a problem is COMPOSE substantially worse than every strong
> modern method on **every** conventional task. That is a different thing from
> not ranking first.

### Barred claim

**"Outperforms state of the art" is barred from every artifact this lane
produces**, along with its paraphrases. We will not have a broad apples-to-apples
sweep and the claim is not needed. The sanctioned framing is:

> COMPOSE achieves competitive molecular optimization while enabling forms of
> stateful, inference-time control that conventional goal-directed generators do
> not naturally represent.

Enforced by `validate_qualification_registry`, which rejects a registry
containing the barred string.

## Evidence standard

Binding, and the reason this lane exists.

1. Every capability cell in the registry carries `verdict` ∈
   `{YES, NO, PARTIAL, UNVERIFIED}` **and** an `evidence` field containing a
   primary source: the paper (with section), the official repository (with file
   or module), or the official release page.
2. `UNVERIFIED` is a legitimate and preferred value. "I believe X supports Y" is
   not admissible. A cell with no primary source is `UNVERIFIED`, and an
   `UNVERIFIED` cell may not be used to justify either a COMPOSE novelty claim
   or a baseline exclusion.
3. Secondary sources (blog posts, survey papers, third-party reimplementations)
   may be recorded as `corroborating` but never as the sole evidence.
4. A capability the official code *could be modified to* support is **not** a
   native capability. The registry distinguishes `native` from
   `adapter_required` from `not_possible_without_method_change`.

## Classification rule

Assigned per (method, COMPOSE sub-capability) pair, then aggregated to a method
verdict.

- **`MUST_RUN`** — the method's *native* scientific object matches the COMPOSE
  sub-capability, a faithful adapter exists or is cheap, and omitting it would
  leave an obvious reviewer question unanswered.
- **`CONDITIONAL`** — comparable in principle, but gated on a specific,
  pre-declared technical condition (checkpoint availability, adapter fidelity,
  CPU feasibility, oracle-parity check). The gate is written down *before* any
  run, and failing the gate demotes to `CONTEXT_ONLY` rather than licensing a
  distorted adaptation.
- **`CONTEXT_ONLY`** — cited in related work and possibly reproduced on its own
  native task, but never placed in a matched COMPOSE table. This is the correct
  status for a method whose native object is different, and it is **not** a
  demotion.

`N/A` is preferable to a distorted adaptation. This rule is inherited verbatim
from `docs/workstreams/PARALLEL_WORKSTREAMS_AND_HANDOFF.md` § Workstream D.

---

## Fairness contract — what "matched budget" means

Wall time is never the primary matched quantity: hardware differs, and several
of these methods are GPU-native while COMPOSE evaluation here is CPU-only.

### Primary matched quantity: **property-oracle calls, counted three ways**

**FROZEN 2026-08-13.** Every run logs all three counters, none substitutes for
another, and every table names the counter it reports:

- `unique_valid_canonical_evaluations` — distinct valid canonical molecules
  evaluated. The **benchmark-native** number and the only one comparable to
  published PMO results.
- `oracle_requests` — every scoring request the algorithm makes, including
  duplicates, rejects and invalids. **Algorithmic demand, not CPU.**
- `evaluator_calls` — expensive oracle executions after caching. **Real work.**

> **Conceptual invariant: caching may reduce evaluator work, but it cannot erase
> wasteful algorithmic requests.**

For a literal-compute number, report wall and core time separately. Full contract
in `FAIRNESS_CONTRACT.md` §0; implementation in
`src/compose_v4/experiments/oracle_accounting.py`.

An *oracle call* is one evaluation of the frozen objective on one molecule.
Counted for every method:

- every candidate scored, including candidates that are subsequently rejected;
- every MCMC proposal, including rejected proposals;
- every particle in a particle method;
- every molecule scored during prescreening, shortlisting or reranking;
- every molecule scored during **objective-specific training or fine-tuning**,
  when that training is performed per task. This is the single most
  frequently-fudged line in the literature and it must not be fudged here.

Not counted as oracle calls, but reported separately:

- neural-network forward passes (`model_calls`);
- legal-successor enumerations / kernel calls (`kernel_calls`);
- cached repeats. Caching is **allowed for every method** and a cache hit is not
  a new oracle call. The cache must be shared-semantics: identical canonical
  SMILES ⇒ identical score.

Invalid or unscorable molecules count as **failed oracle proposals**, not as
free actions.

### Secondary matched quantity: **edit budget**

Only meaningful for methods whose native object is a sequence of molecule→molecule
edits. Where a method has no such notion (a from-scratch generator), the edit
budget is `N/A` and the comparison must be run on oracle calls alone, with the
`N/A` printed in the table rather than silently omitted.

### Reported, never matched: wall time and hardware

Reported as `wall_seconds` + `accelerator_class`. Never the headline metric.

### Where the comparison is *not* fair, stated up front

- **Support differs.** COMPOSE proposes only canonical successors of the frozen
  rewrite kernel. MARS proposes fragment-level edits from its own vocabulary;
  GraphGA proposes crossover/mutation products; a from-scratch generator
  proposes anything in its decoder's range. Equal support must never be claimed.
- **Start point differs.** Where a method cannot be initialised from an exact
  supplied molecule, a source-conditioned comparison is invalid regardless of
  budget matching.
- **Retraining cost is not commensurable with search cost.** A method requiring
  per-objective fine-tuning pays an oracle cost that must be *shown*, not
  amortised away; but a reviewer may reasonably object to either including or
  excluding it. Report both totals — with and without the training-phase oracle
  calls — and let the reader choose.
- **A method that has no notion of an intermediate molecule gets `N/A`** on
  every pathwise metric. It does not get a synthesised trajectory.

---

## Forbidden adaptations

Binding. Any of these invalidates a comparison.

1. Changing a COMPOSE task, goal language, panel or budget to make a baseline
   runnable.
2. Giving a baseline a different oracle, a different similarity definition or a
   different validity filter from COMPOSE.
3. Constructing a "trajectory" for a method that does not natively expose one
   (e.g. treating a GA lineage or a partially-decoded SMILES prefix as a
   sequence of intermediate molecules) and then scoring pathwise metrics on it.
4. Running a baseline with its objective-specific training omitted, and then
   reporting its oracle count as if it were budget-matched.
5. Excluding a baseline because preliminary results are strong.
6. Reporting literature numbers in a COMPOSE table without full protocol match.
7. Tuning a baseline less than COMPOSE. Where a baseline has documented default
   hyperparameters for a comparable task, use the published defaults and say so;
   where it does not, the deviation is recorded in the registry
   `deviations_from_official_protocol` field.

---

## Allowed calibration

- Choosing the 3–5 held-in smoke sources. These come from held-in data only and
  never from the sealed panel or from any confirmatory reserve.
- Choosing per-method container sizes and timeouts.
- Fixing dependency versions to make an environment build at all, recorded in
  `baselines/<method>/environment.lock` with the deviation noted.

## Not allowed

- Any held-out or sealed panel contact. This lane has opened none and must not.
- Any final sweep.
- Any change to `R_theta`, the rewrite kernel, the goal language or the oracles.

---

## Stop rules

Stop and report rather than continue if:

1. A method's official code cannot be built CPU-only within the container budget
   → record `adapter_status: blocked_cpu` and demote to `CONDITIONAL` with the
   gate "GPU authorisation".
2. A required checkpoint is unavailable or unlicensed → `CONTEXT_ONLY`, with the
   provenance gap recorded.
3. A method turns out to natively support a COMPOSE sub-capability → **stop and
   escalate immediately**. This changes what the paper may claim and is more
   valuable than any run.
4. A faithful adapter cannot be written without touching a forbidden adaptation
   → `CONTEXT_ONLY` with the reason, in preference to a distorted run.

---

## Smoke definition

A *held-in smoke* under this protocol is: **3–5 held-in source molecules, one
objective, the smallest budget at which the method's own loop executes end to
end.** Its only purpose is to prove the adapter runs, that oracle accounting is
wired, and to measure per-source cost for the eventual costed sweep.

A smoke result is `SMOKE_HELD_IN` and may never be cited as evidence for or
against any claim. It is an instrument check.

---

## Deliverables and their canonical paths

| artifact | path | status |
|---|---|---|
| qualification registry (authoritative) | `docs/workstreams/baseline-qualification/comparator_registry_v3.json` | `DESIGN_ONLY` |
| markdown rendering (generated) | `docs/workstreams/baseline-qualification/COMPARATOR_MATRIX.md` | generated — do not hand-edit |
| renderer | `scripts/render_comparator_registry.py` | code |
| schema validator | `src/compose_v4/experiments/baseline_qualification.py` | code |
| claim-scoped comparator plan (existing) | `configs/comparator_registry_v3.json` | updated status fields only |
