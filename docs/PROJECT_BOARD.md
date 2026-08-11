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

### P0 — Finish epoch 3, apply the preregistered rule, freeze `R_theta`

Run `run_v2_01`, steps 8,502 → 12,753, on preemptible A10G.

Identity chain: reserve `b580fdef6486` / law `b0cc66f168f1` / manifest
`e27494a46250` / store `b232a6fa069f`; freeze gate FROZEN 9/9.

Entering epoch 3: selected step 8,500 at reference-law NLL **2.9238**, from an
initialization value of **5.4603**. Within-family identity improved in **all 8
families** over two epochs.

Preregistered rule — `diagnostics/editing_v2_epoch3_preregistration.json`:

| epoch-3 gain | action |
|---|---|
| < ~0.05 nats | **stop and freeze** |
| 0.05 – 0.10 | stop and freeze unless the curve is clearly still steep |
| > 0.10 | consider epoch 4 **only** on a fresh explicit decision |

Read the **epoch-level** result. Effective resolution of this metric is ~0.026
nats, so adjacent 500-step points are not interpretable.

Budget: ~$1.38 of the $4 cap remained at launch; epoch 3 costs ~$1.10. **Epoch 4
is not affordable under the current cap.** One preemption already occurred near
step 9,750 and recovered cleanly; each costs ~$0.10.

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
