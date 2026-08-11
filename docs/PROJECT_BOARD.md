# COMPOSE — Project Board

**This file is the durable task list. Git is the source of truth.**

The in-session task tool has been wiped twice at session boundaries, losing 52
tracked items both times. Nothing important was actually lost — it lived in
commit messages and `diagnostics/` artifacts — but the board itself did not
survive. It lives here now. The session task tool may mirror active work; this
file is what persists.

Plan of record: [`docs/EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md).
History and established findings: [`docs/DECISION_LOG.md`](DECISION_LOG.md).

---

## In flight

### C0 — Is there a planning problem at all?

`modal_apps/experiment_c0_planning_signal_app.py`. Gates the full six-arm
controller pilot **before** any `h_phi` is trained.

Task, preregistered: DRD2 `< 0.05` → `>= 0.5`, ECFP4 Tanimoto to the source
`>= 0.4`, six productive edits, held-out reserve sources.

The deciding statistic is **not** global rank correlation — across ~600
successors, immediate and future value agree on hundreds of obviously bad edits
while disagreeing on the few that matter. What decides it:

| statistic | question |
|---|---|
| top-1 disagreement | does MC value pick a different edit than greedy? |
| fresh regret | `h(MC-best) - h(greedy-best)` on independent rollouts |
| **sacrifice-to-win** | is the MC edit *worse now* and *better later*? |

**Self-calibrating null.** Among disagreements where the MC action is
immediately worse, "no signal" predicts fresh regret is positive ~**50%** of the
time — the evaluation sample is independent of the selecting one, so its sign is
a coin flip. Near 50% is noise at any disagreement count; well above 50% is the
phenomenon. **STOP** at chance; **GO** clearly above it; **INCONCLUSIVE** means
deepen C0, not run full C anyway.

Reported as a **ladder over depths 1/2/3 on the same states**. Depth 3 gates the
idea; the ladder is what makes it convincing, since one depth alone cannot
separate "planning helps" from "this horizon happened to look good". It is free:
a depth-3 rollout contains its depth-1 and depth-2 prefixes.

Structural conservatisms: selection and evaluation use independent rollout
samples (argmax over noisy estimates is biased upward, toward "planning helps"),
and only ~8 of ~500 successors are MC-evaluated, so measured discordance
**understates** the true signal.

**Not a benchmark result.** Planning is scored in margin units (logit
`P(active)`), monotone with the benchmark probability so greedy is unchanged, but
monotonicity does not survive taking expectations — `E[max margin]` and
`E[max P]` are different objectives. Benchmark outcomes are Experiment C's job.

Two costing lessons, both recorded in the decision log: the first quote (~$0.50)
counted kernel calls and ignored that Modal bills **memory-time**; and the
per-call cost was profiled rather than assumed. 41% of an enumeration is the
marked law, 59% is applying and canonicalising every mark
(`molecular_graph_to_smiles`: 25,581 calls for one state). Memoizing that would
be the biggest win but it is inside the Process-V2 content hash, so instead
rollouts sample a single mark — exact, since `P(canonical y)` is the sum of mark
probabilities reaching `y`. Verified at 0 support mismatch and 0.000e+00
virtual-mass gap. ~$1.50–3.00 for 12 sources, ~45 min.

---

## Landed

### Experiment C — future-aware control rescues half of greedy's failures

On 24 held-out, **known-reachable** 4–6 step transformations, with the strict
override rule (the planner may override greedy only on strictly better
remaining-budget value; ties keep greedy):

| | greedy | rollout lookahead |
|---|---|---|
| exact recovery | 12/24 | **18/24** |
| mean best similarity | 0.8589 | 0.9335 |
| kernel cost | 71 calls | 368 marginal (**5.2×**) |

**The claim to make is the rescue rate: 6 of the 12 targets greedy failed to
recover were rescued by future-aware control — 50%.** Zero went the other way.

**The claim NOT to make is the p-value.** McNemar gives one-sided p = 0.0156,
but the comparison is *asymmetric by construction*: the rollout policy is built
to be no worse than its greedy base under the value used for improvement, so
`greedy-only = 0` is guaranteed rather than observed. Quoting the p-value as the
headline would dress up a structural guarantee as an empirical finding. Proper
source-level uncertainty comes later, from the sealed panel.

The planner is **surgical**: 11 overrides across 72 decisions (15%), with 215
tied candidates deferring to greedy. It is not behaving differently everywhere
and getting lucky.

Also not supported: any horizon trend (+3 / 0 / +2 at 4/5/6 steps), and any
mechanism story — the panel is uniformly delete-insert fragment swaps.

**This is what earns `h_phi` its place**: an expensive computation
(`V_greedy(x, z, b)`, 5.2× cost) that is *already demonstrated useful*.

### Phase 0 — `R_theta` frozen

Frozen at **step 12,500**: reference-law NLL **5.4603 → 2.8364**, within-family
identity improved in **all 8 families**. Preregistered epoch-3 rule applied.
Identity chain: reserve `b580fdef6486` / law `b0cc66f168f1` / manifest
`e27494a46250` / store `b232a6fa069f`; freeze gate FROZEN 9/9.

### Claim 1 — learning beats corpus-level operator frequencies

`R_theta` **1.46 nats** below the empirical-family baseline, which is the strong
form: it gets the correct global operator frequencies, exact legal support,
state-dependent availability, and alias aggregation. Development result on the
step-12,500 reference checkpoint.

### Claim 3 — exact finite-horizon control (Experiment B)

`diagnostics/exactness/editing_v2_experiment_b_exact_control.json`. Closed slice,
realised switch state, and a non-degenerate retarget objective **in one run** —
the three had previously only been demonstrated separately.

| | |
|---|---|
| slice | 966 states, `stop_reason=closed` |
| switch state | `CC(C)C`, on-trajectory |
| retarget | SHRINK to <4 heavy atoms; proper subset, guard-enforced |
| primary tilt TV | 1.77e-16 |
| retarget tilt TV | 1.39e-17 |
| support violations | 0 · backward residual 0.0 |
| `open_caveat` | null |

### DRD2 oracle — frozen, parity-verified

`artifacts/oracles/drd2_svm_v1/`. The classic benchmark SVM (Olivecrona
REINVENT; the model behind the published VJTNN/GrIDDD numbers), extracted from
its Python-3.6 pickle **once** and reimplemented in numpy. The runtime imports no
sklearn and unpickles nothing.

Parity vs the original estimator: **2.19e-14** on probabilities, **1.35e-13** on
decision values. Reproducing `predict_proba` required reproducing libsvm's
Wu-Lin-Weng coupling, not just the Platt sigmoid — the exact fixed point of that
iteration *is* the sigmoid, but libsvm stops at `max_error < 0.005/k`, and
short-circuiting it left a 1.7e-3 discrepancy that would decide molecules sitting
on the 0.5 success threshold.

Sanity checks a constant scorer would fail: source panel max **0.0482** (which
independently confirms it is the `<0.05` set the benchmark uses), active panel
**300/300** above 0.5.

---

## Blocked / gated

### P1 — Bind the code commit into `RunIdentity`

The plan requires eight identity bindings per checkpoint. `RunIdentity` carries
**seven**: initialization seed, initial model state, library, split, sampling
law, manifest, packed store, eval panel. **The code commit is missing.**

Matters most for the three paper seeds, which must be shown to differ *only* in
seed. `RunIdentity` is frozen/slots and verified on resume, so adding a field
changes the identity digest and will refuse resumes of existing checkpoints —
do this at the **seed-training boundary**, not mid-run, and decide deliberately
whether `run_v2_01` checkpoints are migrated or marked development-only.

Blocks: any paper-bearing run. Does not block the development run.

### P2 — Three independently trained seeds

Required by the plan; the present run is development only. **This is a budget
item, not a scheduling one** — three seeds at ~3 epochs each is several times
the current cap. Needs a budget decision before it can be scheduled.

### P3 — Which of the four paper directories is the target?

`paper/`, `paper_arxiv/`, `paper_iclr_control_substrate/`,
`paper_iclr_stochastic_rewriting/` all exist. `CLAUDE.md` points at
`paper_iclr_stochastic_rewriting/` as "the paper". The canonical plan implies a
substantial rewrite; it is not recorded which directory it targets. **Unresolved
— needs a decision.**

---

## On hold (Rohin: "hold off on parallel work till I give you more instructions")

### H1 — The six prep items from the plan's immediate execution order

Nothing started.

1. Replace the manuscript's claim/RQ list with the four-claim hierarchy.
2. Build the reference-model baseline evaluator (Exp 1B — eight internal baselines).
3. Freeze the exact-control confirmatory goal suite (Exp 3 — 30–50 goals, budgets 2/4/6/8, on the 966-state closure with an explicit cemetery state).
4. Implement the full-molecule `h_phi(x, z, b)` pipeline (Exp 4, `R_theta` frozen; structured continuous goals, not one-hot tasks; broad smooth goals first).
5. Wrap greedy / hard-mask / Boltzmann / beam / SMC behind one interface.
6. Verify MARS, DDSBM, GraphXForm, GraphGA, REINVENT environments **without** running full experiments.

**Check before building item 2:** a previously completed task was "Prove learning
matters: model vs uniform vs empirical-family", so part of the baseline evaluator
may already exist. That check was the first thing interrupted and was never done.

**Gate:** the full-molecule bridge must beat greedy, hard mask, local Boltzmann
and beam search on a 10-source development panel before any expensive external
sweep.

---

## Carried forward from the wiped board

Open items that predate the current plan and were never closed. Recorded so they
are not silently lost; several may no longer be relevant under the new framing.

| Was | Item | Status under the new plan |
|---|---|---|
| #24 | Compare teacher vs model drift, matched by lane and family | Possibly subsumed by Exp 1A family/lane reporting |
| #28 | Source ring-changing molecular pairs (new mining criterion) | Relevant to Exp 5 Track B's ring/topology task |
| #29 | Nested / annealed bridge control for rare conjunctions | Relevant to Exp 3; plan says rare conjunctions must **not** define the first controller |
| #32 | Report fan-out progress unordered | Housekeeping |
| #46 | Resume the old pilot to 1 epoch to confirm underexposure | **Obsolete** — superseded by the run_v2_01 result |
