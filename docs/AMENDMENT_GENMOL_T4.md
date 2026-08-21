# Amendment — GenMol Table 4 (goal-directed lead optimization)

**Opened 2026-08-20.** External-competence benchmark. The question is narrow:

> Can the same frozen learned stochastic molecular process serve as a strong lead
> optimizer when handed a completely different purpose?

Not a future-aware-control claim. That is Jin's job (§5.2, §5.3). T4's job is that
the substrate is a good reusable generative process.

---

## The task, adopted verbatim from GenMol §5.4 / Appendix D.6

Five targets (parp1, fa7, braf, jak2 from DUD-E; 5ht1b from ChEMBL), three seed
molecules each, published in `NVIDIA-BioNeMo/genmol` at
`scripts/exps/lead/docking/actives.csv`. Per (target, seed): **10 iterations x 100
generations = 1,000 evaluated molecules**, following Wang et al. 2023. Constraints
on the RETURNED molecule: `QED >= 0.6`, raw `SA <= 4`, `sim(x, x0) >= delta` for
`delta` in {0.4, 0.6}. Score is the docking score of the best feasible lead.
Docking is QuickVina 2 with the MOOD receptors and boxes.

30 cells. GenMol solves 26/30 (14/15 at delta=0.4, 12/15 at delta=0.6).

---

## GATE 0 — docking parity. PASSED 2026-08-20.

Reproduced all 15 published seed docking scores with our own stack (`qvina02` and
receptors from MOOD, boxes from GenMol's `docking.py`, `exhaustiveness=1`,
`num_modes=10`, obabel `--gen3D`), 5 replicates each.

    mean |delta| vs published   0.34 kcal/mol
    median |delta|              0.24
    systematic bias            -0.02          (none)
    replicate spread           0.70 median, 1.47 mean, 9.30 max
    exact reproductions        4/15

Agreement is inside the metric's own noise and unbiased. The comparison is
legitimate. Artifact: `diagnostics/genmol_t4_gate0.json`.

**Consequence that binds later claims.** Per-molecule docking noise at
`exhaustiveness=1` is ~0.7 kcal/mol median. Several margins in Table 4 are
smaller than that. A sub-1 kcal/mol win on an individual cell is not a win. What
is defensible is aggregate performance across cells, and filling cells where the
comparator reports a dash, which is a feasibility question and immune to docking
noise.

## GATE 0.5b — reachability. PASSED 2026-08-20. CLOSED.

Does the frozen `R_theta` contain routes into the feasible region, and how long
are they? No docking; nothing here could be tuned toward the reported metric.

    depth curve   H<=4: 19/30   H<=8: 23/30   H<=12: 24/30   H<=16: 24/30
    delta=0.4     15/15   (GenMol 14/15)
    delta=0.6      9/15   (GenMol 12/15)

Saturates by depth 12. **The useful edit horizon is 8-12.**

Both fa7 cells that GenMol dashes are reachable (depth 4 and depth 10). All six
misses are at `delta=0.6`, and four are within 0.03 QED of threshold.

That deficit was measured while solving a **strictly harder problem than the
benchmark**: similarity was enforced at every step, though T4 requires it only of
the returned molecule. See the decision below.

Artifact: `diagnostics/genmol_t4_reachability_b.json`. Superseded 0.5a, whose beam
ranked by QED alone and silently dropped the similarity filter when it emptied.

**This diagnostic is closed.** No further beams, no weighting variants, no
beam-width rescues.

---

## FROZEN DECISIONS

**1. Start with the simplest COMPOSE-native algorithm. No learned `h_phi`, no
twisted SMC, and NOT FK-SMC by default.**
Level 1 below is a local canonical-fiber sweep with continuation from realized
states. FK-SMC is a legitimate modern tool but is Level 3: escalate to it only if
development exhibits the population problem it solves. Twisted SMC needs a
trustworthy `h(x,b)`, and the multi-objective `h_phi` is the project's live weak
point, losing to a persistence baseline and failing its within-fiber gate. Making
an external competence claim depend on an unresolved internal one is the coupling
to avoid.

This separates the two experiments cleanly:

    Jin  = R_theta + h_phi + twisted SMC  -> future-aware CONTROL
    T4   = R_theta, training-free         -> the process is a good SUBSTRATE

If the simple version competes, that is stronger evidence for the reusable-process
thesis than needing another trained controller.

**2. Similarity is an ENDPOINT constraint in the primary run.**
T4 requires `sim(x, x0) >= delta` of the returned molecule only. The primary
external comparison matches the published task; we do not impose unrelated
internal constraints on ourselves. Pathwise similarity becomes a separate COMPOSE
ablation, not the headline number.

**3. THREE-WAY CONSTRAINT SEPARATION. The only hard wall is executability.**

    hard at EVERY step      valence, connectivity, declared molecular state
                            space, executable rewrite support. That is COMPOSE
                            itself and nothing else.
    search GUIDANCE         QED, SA, similarity-to-seed, docking. These may rise
                            AND FALL along a trajectory.
    required of RETURNED    QED >= 0.6, SA <= 4, sim(x,x0) >= delta.
    candidates only         Among those, maximise docking.

The feasibility distance

    v(x) = max{ (0.6 - QED(x))_+ / 0.6 ,
                (SA(x) - 4)_+   / 4   ,
                (delta - sim(x,x0))_+ / delta }

is a NAVIGATION SIGNAL, NOT A SUPPORT MASK. `v(x) > 0` states are permitted and
often necessary: nine of the fifteen seeds start with `v > 0`, so masking on `v`
would forbid leaving the seed. A trajectory may dip below the similarity floor
and climb back, or lose QED to gain access to better downstream chemistry, and
still return a feasible lead. Only the returned molecule is judged.

Pathwise similarity was imposed ONLY in Gate 0.5b, deliberately, to ask the
stricter question of whether routes exist without ever leaving the similarity
neighbourhood. All six of its misses were at delta=0.6, which is evidence that
forbidding temporary detours costs reachable endpoints. It was a diagnostic, and
it does not carry into the inference policy.

**4. Development seeds are DISJOINT from the 15 benchmark seeds.**
Policy development uses other DUD-E/ChEMBL actives for the same targets. Choosing
horizon, particle count and potential shape on the benchmark seeds, having already
seen which cells GenMol dashes, is outcome-driven tuning. Freeze the policy on
development molecules, then run the 30 published cells ONCE.

**5. Development degrees of freedom, deliberately small.**
Rounds and candidates docked per round, subject to their product being exactly
the evaluation budget; number of parents swept per round; and ONE selection rule
chosen from a small predeclared family. Trajectory depth is already measured at
8-12 by Gate 0.5b and is not re-searched. Nothing else. In particular the
selection rule is chosen once on development molecules and frozen; it is not
re-tuned per target or per delta.

---

## ESCALATION LADDER. Start at Level 1. Escalate only for a diagnosed reason.

**Level 1 — the algorithm to build now.**

    enumerate the canonical fiber S(x)  ->  R_theta-weighted sweep
      ->  soft progress/docking selection  ->  continue from realized states

Every successor gives R_theta(y|x), QED, SA and similarity BEFORE any docking
call, so COMPOSE sees the complete legal local menu for free and spends the
expensive evaluation only where it chooses to. Winners become the next round's
actual molecular states, so edits already made are not discarded. This is
realized-state reuse arising from the optimisation problem itself rather than
from a constructed retargeting experiment. Experiment 1 supports the premise:
most of R_theta's gain over the empirical-family baseline is in identifying which
successor suits the current molecule, which is exactly this decision.

**If Level 1 works, STOP.** That is the strongest scientific outcome: the frozen
process is a good substrate without elaborate control.

**Level 2 — only if development shows a specific myopia failure.** Short explicit
R_theta continuation rollouts from tied candidates, scored on QED/SA/similarity
only. No learned model. The 65-pair result shows this can matter: explicit future
reachability rescued 14 of 39 local failures. Escalate for that diagnosed reason,
not because lookahead is interesting.

**Level 3 — only if population collapse or rare-event search is the problem.**
FK-SMC with ESS-triggered resampling.

**Level 4 — not for T4.** Learned h_phi / twisted SMC.

Retargeting is excluded outright: nothing about the goal changes during T4, so
including it would be gimmickry.

## BARRED

Training a T4-specific `h_phi`. Twisted SMC. Reopening beam widths or the
reachability diagnostic. Tuning for fa7 because its GenMol dashes are now known.
Target-specific logic, in particular for braf. Any modification to `R_theta`.
Post-result rescue of any kind: whatever the 30 cells return is banked.

## Claim boundary

"braf is a systematic weakness" is NOT established. Five braf misses under a
depth-6 beam meant only that a shallow search found no route; four of six current
misses sit within 0.03 QED of threshold. A cell not reached is a cell this search
did not find a route to, which is weaker than unreachable.

The bar is not 30/30. Broadly competitive docking, respectable feasible coverage,
and genuinely filling one or more of the comparator's missing cells is already
useful external evidence. Beating aggregate performance would be a strong result.
Current status: **promising, scientifically clean, not yet demonstrated
competitive.** Spend to date ~$1.00.
