# Amendment: scaffold_hop is a terminal-value problem before it is a lookahead problem

Status: measured 2026-08-22. All numbers from persisted artifacts on
`compose-v4-artifacts`. Companion to `AMENDMENT_PMO_PRESCREEN_AUDIT.md`.

## The result stack

| stage | scaffold_hop top-10 | source |
|---|---|---|
| clean, objective-blind | 0.4733 | `pmo_pilot` @500 |
| oracle-prescreened initialization | 0.5424 | `pmo_matched_init` @500 |
| surrogate-guided search, 250 counted calls | 0.5663 | `pmo_control`, best arm |
| oracle-greedy, TRUE oracle on every successor | 0.6285 | `pmo_ceiling` |
| GenMol published AUC-top10 | **0.628** | `docs/genmol_pmo_targets.json` |

Three separate interventions -- task-specific initialization, dense learned
guidance, and online adaptation -- move the number from 0.4733 to 0.5663. The
target is 0.628.

## What was eliminated, and by what evidence

**Initialization is not the answer.** The full 249,455-molecule oracle prescreen
yields a top-10 of 0.5194 and a top-100 of 0.4965. Handing COMPOSE the best 100
molecules in ZINC250k under the task oracle itself moves @500 from 0.4733 to
0.5424, i.e. +0.069. See `AMENDMENT_PMO_PRESCREEN_AUDIT.md`.

**Dense learned guidance is not the answer either, and the reason is specific.**
An MLP fitted to 50,000 prescreen labels reaches Spearman **0.9587** on a random
held-out ZINC split, with 55x top-1% enrichment. Evaluated on 18,031
COMPOSE-generated successors carrying true oracle values, it collapses:

| slice | n | Spearman | top-10% recall |
|---|---|---|---|
| ZINC held-out, random split | 10,000 | 0.9587 | 0.55 |
| all COMPOSE successors | 18,031 | 0.3313 | 0.119 |
| below corpus max (0.5261) | 6,663 | 0.3543 | 0.123 |
| **above corpus max** | **11,368** | **0.0688** | 0.085 |

Within a fiber -- which is what the controller actually consumes each round --
median Spearman is 0.3844 and the **median rank of the true-best successor is
85.5**. The surrogate essentially never surfaces the right molecule.

**A random-split score on the training corpus is not evidence that a surrogate
can guide search.** The only informative validation set is model-generated
molecules with true labels, stratified above the training-label maximum. That
check must gate every future surrogate before it is wired into a controller.

## The controller sweep

Five arms, one flag apart, two seeds, 250 counted calls each, everything else
identical (frozen R_theta, N_LINEAGE=12, PER_ROUND=20, 400-state cap,
`OracleMeter` PER_MOLECULE, counted-only official `top_auc`).

| arm | top-10 counted | AUC@10k | above corpus max |
|---|---|---|---|
| online_thompson | 0.5663 ± 0.0032 | 0.5634 | 182 |
| online_greedy | 0.5651 ± 0.0011 | 0.5622 | 196 |
| online_ucb | 0.5633 ± 0.0002 | 0.5605 | 195 |
| no_surrogate | 0.5560 ± 0.0034 | 0.5530 | 53 |
| frozen_greedy | 0.5533 ± 0.0011 | 0.5506 | 32 |

Two facts worth keeping. Online adaptation is worth **+0.0102**, reproducible and
small. And `frozen_greedy` scores BELOW `no_surrogate` on both seeds: a frozen
surrogate with 0.9587 corpus Spearman is worse than no guidance at all once
search leaves the corpus. Acquisition rule barely matters by comparison -- the
three online arms span 0.0030 -- whereas whether the model updates at all is the
whole effect.

## The corpus maximum is not a barrier

11,368 of 18,031 COMPOSE-generated successors (63%) score above 0.5261, the
maximum over all 249,455 ZINC250k molecules. Crossing the corpus ceiling is
routine for COMPOSE's edits, so `n_above_corpus_max` measures activity, not
progress. The real barrier is 0.63 to 0.628.

## Consequence for future-value control

The plan of record was to reuse the fitted surrogate as terminal desirability,
g(x) = f_hat(x), and test h_b(x) = E_{R_theta}[g(X_b) | X_0 = x] against
immediate guidance on a matched budget.

**That experiment is currently untestable, and running it would produce a null
result that says nothing about lookahead.** f_hat has Spearman 0.0688 above the
corpus maximum, which is precisely the region h propagates through. An
expectation over a terminal value that is noise there is noise.

The gap decomposes into two independent problems:

    0.5663 -> 0.6285   learned g versus a perfect g. A TERMINAL VALUE problem.
    0.6285 ~ 0.628    perfect g under greedy selection versus GenMol.
                       This is the genuine case for lookahead, and it survives.

The second is unaffected by the first and remains the strongest argument for
h-control we have: with perfect current-state information on every enumerated
successor, greedy selection converges at 0.6285 and the beam collapses
(mean equals best from depth 5 onward). Perfect local information is provably
insufficient. But the first problem gates any test of the second.

## What is NOT established

- **That 0.628 is reachable in COMPOSE's operator support.** Nothing run so far
  bounds this. The oracle-greedy diagnostic is greedy: it prunes sacrificial
  routes by construction, so its 0.6285 is a statement about greedy search, not
  about reachability. It must never be cited as a support ceiling.
- **That beam collapse alone caused the remaining gap.** Plausible, unmeasured.
- **That a better terminal value would close 0.56 to 0.63.** It is the obvious
  hypothesis and it is untested.

## Open routes, in the order the evidence supports

1. **Fix the terminal value.** The 18,031 true-oracle-scored COMPOSE successors
   are persisted at `pmo_ceiling/ood_scaffold_hop.npz` and would train a
   surrogate on exactly the distribution search visits. **Caveat that must be
   stated wherever this is used: those labels come from off-budget oracle calls
   beyond the 249,455-label prescreen, so a lane trained on them is no longer
   information-matched to GenMol.** It is a legitimate development route and an
   illegitimate reporting lane, and the two must not be conflated.
2. **Bound the support.** Determine whether 0.628-class chemistry is reachable
   under the edit operators at all, by a search that is not greedy. Until this
   is known, effort spent on inference could be spent on an unreachable target.
3. **Then, and only then, test h** with a terminal value that is trustworthy in
   the region it propagates through.

## Frozen

jnk3 and osimertinib_mpo are regression tasks, not development tasks, as of this
amendment. Their matched-lane results at 500 counted calls, under the official
counted-only `top_auc(..., finish=True)` normalised by 10,000:

| task | ours @500 | GenMol @10k | bound status |
|---|---|---|---|
| jnk3 | 0.7471 | 0.906 | provable: primed max 0.68 < top-10 mean 0.7530 with best 0.76, so no primed molecule can be in the top 10 |
| osimertinib_mpo | 0.8540 | 0.876 | NOT yet provable: primed max 0.8288 would require the other nine to average 0.8628 <= 0.8717, which is feasible. Recompute from the counted-only ledger. |
