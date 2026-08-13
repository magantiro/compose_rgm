# The conventional optimization suite — FROZEN 2026-08-13

**Artifact status: `DESIGN_ONLY`.** Frozen **before** any COMPOSE-versus-baseline
outcome was seen. No COMPOSE arm has been run on any task below, and no baseline
sweep has been run.

Five tasks, not twenty. PMO has 23 tasks; running them all would buy breadth we
cannot defend and cost budget we do not have. The set below spans the four
regimes a reviewer will ask about — single-property, multi-property,
similarity-constrained/source-conditioned, and one standard PMO aggregate — and
stops.

---

## Freeze rules

1. **Frozen before outcomes.** Nothing here may be changed after a
   COMPOSE-versus-baseline number exists. A task that turns out to favour or
   disfavour COMPOSE is reported, not replaced.
2. **Every task declares all nine fields** below. A task missing one is not
   runnable.
3. **Both budget counters are always logged** (`FAIRNESS_CONTRACT.md` §0).
   `unique_valid_canonical_evaluations` is the budget-binding counter for every
   task here, because that is PMO's convention and the only one comparable to
   the literature. `oracle_requests` is logged alongside and is what efficiency
   claims use.
4. **T5 is the only PMO-comparable task**, and only under PMO's own oracle,
   budget and metric. The other four are COMPOSE-native tasks and their numbers
   may never be placed beside a PMO leaderboard.

## The nine frozen fields

`oracle` · `starting-state semantics` · `budget` · `success metric` ·
`benchmark-native count` · `oracle_requests count` · `seeds` ·
`trained/retrained per objective` · `applicable methods`

---

## T1 — single-property, target-free, source-conditioned

| field | value |
|---|---|
| oracle | DRD2 activity **log-odds**, `src/compose_v4/drd2_oracle.py` (sha256 `318cc812b3a5c591ee4bcb8c2cd2da004ff5b147856c9b0375e7fe81a3bf1a44`); numpy-only, cannot drift with the environment |
| starting state | **one designated source molecule per task instance**, from the held-out matched reserve; every method starts from that exact molecule |
| budget | 1,000 `unique_valid_canonical_evaluations` per source |
| success metric | best DRD2 log-odds reached; secondary: AUC of best-so-far vs budget |
| benchmark-native count | binding |
| `oracle_requests` count | logged, reported, never budget-binding |
| seeds | 3 per source |
| trained per objective | COMPOSE **no** (`R_theta` frozen); GraphGA **no**; MARS online only; REINVENT **yes** (RL); GraphXForm **yes** (fine-tune) |
| applicable | see `FAIRNESS_MATRIX.md` |

Never scored in `P(active)`: the pool sits at median 0.004 where probability is
saturated and a probability delta understates real movement.

## T2 — multi-property, target-free, source-conditioned

| field | value |
|---|---|
| oracle | the frozen developability goal — QED floor and cLogP box in held-in IQR units, `diagnostics/retarget_goal_language_normalizers.json` (sha256 `187d1ccc60b00c858f0f54f729a99ea91e06d33a913287886c80080889a76cc8`) |
| starting state | one designated source per instance, as T1 |
| budget | 1,000 `unique_valid_canonical_evaluations` |
| success metric | binary in-region success; secondary: continuous margin |
| benchmark-native count | binding |
| `oracle_requests` count | logged |
| seeds | 3 per source |
| trained per objective | as T1 |
| applicable | see matrix |

> **Scoping, binding.** The main lane's held-in calibration measured greedy
> 28/30 and verified 28/30 on this goal, binary headroom **0**, gate CLOSED. T2
> is therefore a **competitiveness sanity check**, not a test of future-aware
> control. A baseline winning T2 refutes nothing COMPOSE claims.
>
> **The margin objective is negative for most starting molecules**, which is not
> incidental: it is why headroom exists. GB-GA's roulette selection cannot
> accept that (measured — see T2 note in `FAIRNESS_MATRIX.md`).

## T3 — similarity-constrained lead optimization

| field | value |
|---|---|
| oracle | DRD2 log-odds **subject to** ECFP4 (Morgan r=2, 2048 bit) Tanimoto to the source ≥ 0.4 |
| starting state | one designated source per instance; the constraint is defined **relative to that source** |
| budget | 1,000 `unique_valid_canonical_evaluations` |
| success metric | fraction of sources with a molecule satisfying **both** the similarity floor and a preregistered potency threshold |
| benchmark-native count | binding |
| `oracle_requests` count | logged; candidates failing the similarity floor **still count** — they are failed proposals, not free actions |
| seeds | 3 per source |
| trained per objective | as T1 |
| applicable | **this is the task that separates the methods.** See matrix. |

The similarity floor is a **constraint, not a scoring term**. Folding it into the
objective is a recorded deviation, and for GraphGA it is the only way to express
it at all.

## T4 — topology / cardinality-changing edit

| field | value |
|---|---|
| oracle | developability (T2's oracle) with a required change in ring count **or** heavy-atom count of at least 1 relative to the source |
| starting state | one designated source per instance |
| budget | 1,000 `unique_valid_canonical_evaluations` |
| success metric | fraction reaching the objective **while** satisfying the structural change requirement |
| benchmark-native count | binding |
| `oracle_requests` count | logged |
| seeds | 3 per source |
| trained per objective | as T1 |
| applicable | **GraphXForm is `inappropriate` for the deletion arm** — its action space is strictly constructive and cannot remove an atom or bond |

T4 exists because it is the one conventional task where a support difference,
not a control difference, decides the outcome. It must be reported as a scope
result.

## T5 — PMO aggregate (the only literature-comparable task)

| field | value |
|---|---|
| oracle | PMO's own, unmodified, via `wenhao-gao/mol_opt`. **Not** the COMPOSE oracles |
| starting state | PMO's own semantics — de-novo, no designated source |
| budget | **10,000 unique valid canonical SMILES**, PMO's standard |
| success metric | **AUC top-10**, PMO's standard |
| benchmark-native count | binding, and identical to PMO's `Oracle.score_smi` |
| `oracle_requests` count | logged for our arms; PMO's published numbers have no such column and must not be given one |
| seeds | PMO's standard |
| trained per objective | as declared by each method |
| applicable | GraphGA, REINVENT (**2.0-era via PMO**), COMPOSE |

**Two hard rules for T5.** (a) PMO's `main/reinvent` is a REINVENT 2.0-era
reimplementation; a T5 row may cite PMO numbers **or** claim a REINVENT 4
capability, never both. (b) PMO tunes σ for its own environment and warns to
re-tune when the environment changes; whichever we do is a recorded deviation.

---

## Preregistered task subset

Per `docs/EXPERIMENT_PLAN.md` the PMO subset is **DRD2, GSK3β, JNK3** plus three
official GuacaMol multi-property objectives. T5 uses that subset. QED is a smoke
task, never a headline.

## What is deliberately excluded

- The other ~17 PMO tasks. Breadth we cannot defend at this budget.
- Any task chosen because it makes greedy fail. Binding anti-tuning rule.
- Unconditional distribution learning. Not what COMPOSE claims.
- Any task requiring a GPU baseline, until GPU is authorised and costed.
