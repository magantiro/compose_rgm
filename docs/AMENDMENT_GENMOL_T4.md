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

## FROZEN CONFIGURATION — recorded 2026-08-23, before the official run

The policy below is frozen. After the 30 official cells are inspected, no
scientific change may be made: not the controller, archive rule, feasibility
definition, docking configuration, or any hyperparameter. Infrastructure
failures may be resumed or retried; scientific behaviour does not move.

    arm                 immediate      (Level 1. future_h is dead on measurement:
                                        107 h/round, and sampled rollouts are only
                                        1.7x cheaper, not the 22x first claimed.)
    archive             multi-lineage Pareto-on-shortfall PLUS elitism on v
    lineages            N_LINEAGE = 8
    horizon             4
    rounds x per_round  50 x 20 = 1,000 evaluated molecules per cell
    docking             QuickVina2, exhaustiveness 1, num_modes 10,
                        DOCK_WORKERS = 4 concurrent, cpu_per_dock = 1
    containers          cpu = 4, resume enabled, persist every round

### The bug the dev confirmation caught

Archive admission was `top-32 by R_theta(y|x) * g_terminal(y)`. A molecule that is
nearly feasible but that R_theta rates an improbable edit fell outside that cut
and was never stored. The signature is `min_v` RISING between rounds, which can
only mean the incumbent was lost:

    fa7    0.2415 -> 0.146 -> 0.0635 -> 0.0635 -> 0.0635   (frozen; found 0.0207
                                                            at round 6 and lost it)
    parp1  0.2154 -> 0.1049 -> 0.0432 -> 0.0432 -> 0.0517  (rose)

Best-v candidates are now admitted unconditionally, independent of R_theta mass.
Result on the same dev cells, same budget:

    jak2  (control)  155 -> 187 feasible, best -9.9      no regression
    parp1            0   -> 161 feasible, best -12.2     feasible from round 2
    fa7              0   -> 13  feasible, best -9.1      feasible from round 6

Both cells that killed the original Level-1 optimizer are solved by one archive
fix. Not lookahead, not a learned controller. The search was already finding the
chemistry; the bookkeeping was deleting it. This is why the escalation ladder was
not climbed: the diagnosed failure was archive admission, not myopia.

### Execution gates, all passed

    integration    40 calls -> 40 feasible docked, order and accounting identical
                   to serial; docking 59.8s -> 32.1s inside the controller
    docking bias   signed mean D(cpu=1) - D(cpu=4) = -0.045 kcal/mol, which is
                   6.4% of the Gate-0 median replicate spread of 0.70. No shift.
    resume         killed after a persisted docking batch, then restarted:
                   20/20 molecules retained, 0 lost, 40 docked == 40 calls
                   (no double charge), rounds 1 -> 2, feasible 20 -> 40.

`DOCK_WORKERS = 4` is measured, not chosen: w=4 gave 2.52x, w=8 gave 2.07x, and
w=16 collapsed to 0.48x while silently returning results for only 4 of 20
molecules. More workers loses dockings without raising an error.

### Reporting constraints that bind the official table

Gate 0 measured per-molecule docking noise at ~0.70 kcal/mol median, 9.30 max.
**An individual cell margin under 1 kcal/mol is not a result.** What is
defensible is cells solved, cells filled where the comparator reports a dash, and
aggregate behaviour. The first pass is ONE replicate; GenMol reports one figure
per cell, but any claim about a single cell's score must carry the noise caveat.

fa7 is the fragile cell: it solved with 13 feasible molecules, crossing at round 6
of 10 at a 200-call budget. The official 1,000-call budget gives it far more room
after crossing, but it is the cell to watch.

## The horizon finding: why six controllers plateaued

Recorded 2026-08-24. Comparator tables are extracted artifacts, never hand-typed:
`docs/genmol_t4_targets.json` (GenMol Table 4) and
`docs/invirtuogen_t4_targets.json` (InVirtuoGen lead table, 28/30 solved).

### The published winners are 8-15 executable edits away

InVirtuoGen released the molecules behind its T4 runs. Comparing them structurally
to the benchmark seeds -- zero docking calls -- gives the distance directly, since
each executable edit changes the heavy-atom count by at most one:

| cell | dHeavy | dRings | min possible edits |
|---|---|---|---|
| 5ht1b s7 d=0.4 | **+15** | +3 | >= 15 |
| parp1 s0 d=0.4 | **+14** | +3 | >= 14 |
| braf s9 d=0.6 | -14 | -1 | >= 14 |
| braf s11 d=0.6 | -10 | -1 | >= 10 |
| braf s10 d=0.6 | -8 | 0 | >= 8 |

Every controller in this amendment searched **1-4 edits** from the seed, with macro
proposals reaching 8. The chemistry that scores -13 to -14 sits at 14-15. Six arms
were re-ranking candidates in a neighbourhood that does not contain the answer.

**This is the single explanation that covers all of them.** vecelite, dockelite,
macro at two budgets, softreward, and the lightweight adapter each moved scores by
+-0.5 and traded one cell class for another, because all of them redistribute a
fixed local candidate supply.

### Growth is much harder than deletion

A beam search over legal successors, guided by similarity to the winner minus a
penalty on remaining heavy-atom distance, run to a per-cell depth exceeding the
structurally required minimum:

| cell | direction | depth | best similarity to a winner | exact hit |
|---|---|---|---|---|
| braf s10 d=0.6 | -8 atoms | 16/16 | **0.8750** | no |
| braf s11 d=0.6 | -10 atoms | 18/18 | **0.8750** | no |
| braf s9 d=0.6 | -14 atoms | 22/22 | **0.8644** | no |
| parp1 s0 d=0.4 | **+14 atoms** | 22/22 | **0.5500** | no |
| 5ht1b s7 d=0.4 | **+15 atoms** | 23/23 | **0.5098** | no |

Same algorithm, same budget, same guidance: deletion-direction targets are
approached to 0.87, growth-direction targets stall below 0.55. Removing structure
is easy in this state space; **building it -- +14 atoms and +3 rings -- is hard.**
The two cells where we are ~4 kcal/mol behind are precisely the growth cells.

**Evidence boundary.** No exact hit was found, and this is a bounded heuristic
beam: a miss means "not found at width 16, depth >= the required minimum", NOT
that the molecule lies outside COMPOSE's executable state space. Claiming a
support limitation would need an explicit operator incompatibility or exhaustive
bounded search. The *contrast* between directions is the robust result; the
absolute misses are not.

### Two audit-design errors worth not repeating

The first version capped depth at 12. Three of five targets require >= 14 edits,
so those misses were **guaranteed before the search began** and carried no
information. Depth is now derived per cell from |dHeavy| plus headroom.

The first version also ranked purely by Tanimoto similarity. A molecule can look
similar while having no route to the target's size, so the beam circled near the
seed. Ranking is now `sim - 0.02 * |heavy gap|`.

### Competitor molecules: development only

The published winners were used to measure required horizons and the
growth/deletion asymmetry. They must NOT appear anywhere in a reported
optimizer: no target-similarity guidance, no seeding, no reward term. The
production controller sees only docking, QED, SA, and similarity-to-seed, and
starts from the benchmark seed. Using competitor outputs to choose a horizon
range is calibration; using them inside inference would be leakage.

### Consequence

The indicated algorithm is long-horizon trajectory optimization with credit
assigned across the path: an offspring is `x_0 -> ... -> x_L` with every
intermediate executable, only `x_L` docked, and the endpoint reward applied to
every transition that produced it. Synthetic check: with a payoff reachable only
through unrewarded intermediate states, endpoint-only credit learns the bridge at
+0.16 while path-level credit learns it at +1.64 -- a 10x difference on exactly
the structure `+3 rings via unrewarded growth steps` presents.

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

---

## BOOKKEEPING CORRECTIONS 2026-08-21

**The delta-n audit is not a T4 gate.** It is a Methods/executor audit supporting
the Sec. 4.1 claim that each admitted edit changes heavy-atom count by at most
one. It is listed here only because it ran in the same window. The T4 substrate
evidence is: docking parity (Gate 0), reachability and edit-depth regime
(Gate 0.5b), endpoint-only similarity behaving correctly in smoke, and a
cost-valid CPU configuration.

**The official cohort is NOT prospectively untouched.** An earlier note in this
file said the policy would be frozen "before any benchmark seed is touched."
That is false and is retracted. The published 15 seeds were used for docking
calibration (Gate 0) and reachability diagnostics (Gate 0.5b), and those results
informed the horizon choice and the decision not to impose pathwise similarity.

The accurate description, to be used in the paper:

> The published benchmark cohort was used for protocol calibration and
> task-geometry diagnostics; the optimization policy itself was developed on a
> separate, seed-disjoint DUD-E/ChEMBL set and frozen before COMPOSE
> docking-optimization results on the official 30 cells were generated.

The 30-cell run is a faithful external benchmark evaluation with disjoint
optimizer development. It is NOT a prospectively untouched held-out panel in the
sense of the Jin 128-source panel, and must not be described as one.

---

## FROZEN LEVEL-1 POLICY CONTRACT. Frozen 2026-08-21, before the dev sweep.

Nothing below may be changed because one target looks bad. Any change voids the
dev sweep and requires a new amendment.

**Algorithm.** realized molecule -> enumerate executable canonical fiber ->
R_theta-weighted soft selection -> dock selected endpoints -> continue from
promising realized molecules.

    fiber enumeration   enumerate_factorized_marked_law at TIME_POINT=0.5,
                        top APPLY_CAP=300 marks by R_theta probability per
                        parent, applied and canonicalized
    parents per round   3, ranked by (v>0, ds if feasible-and-docked else 0.0, v)
                        so feasible-and-well-docked states are extended first
    rounds              10
    dockings per round  20   (dev)      100  (official)
    evaluation budget   200 per cell (dev), 1000 per run (official)
    max edit depth      10, one committed edit per round from a realized parent
    selection rule      "tilt": sample k without replacement with weights
                        w = R_theta(y|x) * exp(-v(y)/TAU_V), TAU_V = 0.1,
                        clipped at 1e-30 and renormalized
    feasibility         ENDPOINT-ONLY, evaluated on the candidate itself:
                        v = max( (0.6-QED)+/0.6, (SA-4)+/4, (delta-sim)+/delta )
                        feasible iff v == 0. sim is Morgan/Tanimoto to the SEED.
                        Intermediates are NOT masked on QED/SA/similarity.
    hard constraints    only COMPOSE's native executable-state guards
    deduplication       by canonical SMILES. A candidate is skipped if it equals
                        its parent, was already docked in this run, or is already
                        in this round's candidate set.
    no feasible cand.   the round still docks its k picks; tilt biases toward
                        low v but does not require v==0. Infeasible molecules can
                        never become the returned best. If the fiber is empty or
                        every candidate fails property evaluation, the run stops
                        early and is banked as-is.
    returned molecule   argmin docking score over archive entries with v==0 and a
                        docking score. If none, the cell returns no molecule and
                        is recorded as a failure, not resampled.
    RNG                 numpy default_rng(20260820 + cell_index), one stream per
                        cell, fixed before launch
    docking             qvina02, MOOD receptors, GenMol boxes, exhaustiveness=1,
                        num_modes=10, obabel --gen3D, cpu=1 (Gate-0 validated)

**Diagnostics recorded per cell**, independent unit = dev molecule:
feasibility-acquisition curve, best feasible docking score vs evaluation count,
docking improvement vs the starting lead, count and distinctness of feasible
molecules, realized lineage and edit depth of the returned molecule.

**Post-sweep decision hierarchy. Precommitted; no thrashing.**

1. Level 1 behaves sensibly across the dev set -> FREEZE IT. No "let us see if
   SMC gets another 0.2 kcal/mol." Authorize the official 30-cell run.
2. A specifically diagnosed myopia failure -> and only then, test Level 2
   (short explicit R_theta continuation lookahead) on dev molecules. Requires
   evidence that locally disfavored edits systematically have better downstream
   outcomes, not merely that performance is mediocre.
3. Level 1 generally poor -> kill or demote T4. No rescue via h_phi, SMC,
   weight sweeps, or target-specific rules.
