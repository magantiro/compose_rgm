# Campaign lessons

Exported 2026-09-05 from the agent's persistent memory, which lives
outside this repository and would otherwise not survive a handoff.
Each entry is a fact or rule that cost real time to learn.

## amortized-h-still-pays-executor-calls
*The fitted committor is only amortized if successors can be featurized without applying them; today h_model applies every admissible action.*

The point of fitting a structural committor was to replace expensive tree
lookahead with one cheap network evaluation per successor. As integrated in
`region_rewrite.propose`, the `h_model` branch calls `apply_fn` on EVERY
admissible successor to build its features, so it still pays one executor call
per admissible action per step: measured 1280 calls per trial against 11 for the
undirected base kernel (2026-09-01, 45 held-out regions).

The network eval is cheap. The featurization is not. Until successors can be
scored from the action and current state without materialising the successor,
the amortization claim is not supported and h_phi is not cheaper than the
lookahead it was meant to replace.

**Why it matters:** any "amortized vs lookahead" runtime claim is wrong until
this is fixed, and matched-budget arm comparisons are distorted by it
([[matched-budget-needs-one-binding-resource]]).

## benchmark-targets-from-artifact
*Never hand-copy baseline benchmark numbers; ingest the published table into a checked artifact with a sum assertion and cite only that.*

Baseline comparison numbers must be extracted programmatically from the source
table into a machine-readable artifact with an assertion that reproduces a
published aggregate. Every downstream doc, table, and analysis cites that
artifact. Never type a competitor's number into prose or code from memory or
from a chat message.

**Why:** on 2026-08-22 an entire night of COMPOSE PMO development ran against
GenMol targets of jnk3 0.688 and scaffold_hop 0.936. The real published values
are **jnk3 0.906 and scaffold_hop 0.628** — the figures in use had been read
across misaligned columns of the results table. The consequences were not
cosmetic: scaffold_hop was treated as a mysterious 0.56→0.94 chasm and made the
sole development task, when COMPOSE's oracle-greedy diagnostic had already hit
0.6285, i.e. parity. Meanwhile jnk3 was reported as a likely win when it was in
fact the largest deficit (0.7471 vs 0.906). Hours went into the wrong task and a
false positive was reported to the user.

**How to apply:** for COMPOSE/PMO the artifact is
`docs/genmol_pmo_targets.json`, extracted from arXiv:2501.06158v3 and asserted to
contain 23 tasks summing to 18.362. The optimization target is the aggregate
sum, so per-task deficits must be ranked before choosing what to develop —
whichever row is currently in front of you is not evidence it is the costly one.
Pairs with [[microbenchmark-before-launch]]: measure, do not assume, and that
applies to the target as much as to the runtime.

## bridge-handoff-is-proposal-not-grammar
*Region-replacement sentinels -- pendant 12/12, both splitting-region cases 0/12, but the failure is an undirected proposal, not missing primitives.*

Measured 2026-08-31, `diagnostics/region_sentinels.json`,
`modal_apps/region_sentinels_app.py`, real executor, CPU.

**pendant 12/12 OK.** 3 steps, conditional_path_logq -3.82, changed fraction
0.04, coherence 1.00, valid endpoint. Frozen context, restricted support,
lineage, canonicalization and explicit FAIL all work end to end.

**Both splitting-region ("bridge") cases 0/12**, every failure
`no_connectivity_preserving_handoff`. But the traces say the grammar is fine:

| | bridge_free | bridge_saturated |
|---|---|---|
| admissible actions/step | mean 66 (6-105) | mean 103 (69-205) |
| families sampled | bond_reorder 126, restate 37, delete 15, **insert 11** | insert 75, delete 57, restate 19, cycle_close 6, cycle_open 4 |

66-103 admissible actions at EVERY step, including atom_insert, cycle_close and
bond_reroute -- exactly the primitives make-before-break needs. In bridge_free
`bond_reorder` ate 126 of 192 steps because it carries the most R_theta mass.

**Verdict: the primitive grammar supports connectivity-preserving handoff; an
R_theta-proportional undirected walk does not find it.** Proposal design, not
operator support. Fix is a grow_new kernel biased toward the far terminal, kept
likelihood-evaluable as an explicit density.

**Two of my own bugs produced earlier, WRONG verdicts -- do not cite those runs:**
`region_sentinels_run1_contaminated.json` (admissibility let terminal-terminal
bond ops through, surfacing later as context_mutated) and
`run2_guard_bug.json` (the fix over-corrected: an atom_insert on a single
terminal has reach {terminal} after its new slot is removed, so the
context-only guard refused the OPENING MOVE of make-before-break and produced
12/12 handoff failures). Guard now applies only when the action creates nothing.

**Naming:** this "bridge" is graph-theoretic (removal splits the preserved
context), NOT a bridged bicyclic ring. The dev panel has 0 bridgehead and 0
spiro atoms, matching zero in the IVG winners; on 6 dev cells only 31 of 181
splitting regions are pure linkers, 150 are ring segments. Rename to `splitting`
before it reaches the paper. See [[t4-ring-type-coverage]].

## committor-needs-goal-seeded-replay
*Fitted Bellman for the structural committor fails on uniform-ish replay -- 5 terminal successors in ~500k edges; the replay set must be seeded with known successful trajectories.*

Measured 2026-08-31, `diagnostics/committor_bellman.json`,
`modal_apps/committor_bellman_app.py`. 48 units, 5,758 replay states, each with
its full canonical successor law stored (~100 successors), slowest unit 2044s.

**Passive Monte Carlo first (`committor_train_app.py`): 48,588 samples, positive
rate 0.0000 in ALL 24 strata** (2 cases x 2 terminals x b=1..6). 320 base-kernel
rollouts of 14 steps reached establishment zero times, in a case where the
directed proposal reaches it 12/12 in 2-3 steps. That is the rare-event regime,
quantified.

**Fitted Bellman then failed for a different reason: no terminals in the replay
set.** terminal STATES = 0. Terminal SUCCESSORS = **5**, out of roughly 500,000
successor edges. At b=1 the exact backup has only 5 nonzero targets and a mean
of 0.0000; from b=2 the report shows "nonzero=5758" with max 0.003, which is the
network SMEARING a near-zero signal across every state, not propagation. Do not
read nonzero-count as evidence of learning without checking the magnitude.

**Cause: discovery was breadth-oriented, not goal-oriented.** BFS with
per_family=3, depth 3, beam 12 samples the neighbourhood of x0 but does not
retain goal-reaching paths, and the beam truncates the rare branches that reach
a terminal. Base-kernel rollouts contribute nothing by construction (see the
0/320 result). So the replay set described the space around the start, not the
events the committor is defined by.

**Fix: seed the replay set with KNOWN successful trajectories.** The reachability
probes already produce them -- depth-3 `cycle_open -> atom_delete -> atom_delete`
for prune, depth-2 `cycle_open -> cycle_close` for saturated -- and the directed
proposal reaches establishment 12/12. Store those paths and their local
neighbourhoods, so terminals and their predecessors are present and Bellman has
something to propagate backward. Discovery may be goal-directed; the TARGET is
still the R_M backup, so the committor being learned is unchanged.

**How to apply:** before any committor fit, assert the replay set contains a
non-trivial count of terminal successors AND terminal states. The app already
prints `n_terminal_successors`; treat a single-digit value as a stop condition,
not a small number. See [[greedy-potentials-cannot-cross-plateaus]].

## compose-canonical-experiment-plan
*MASTER_PLAN_EXPERIMENTS_AND_COMPARATORS.md now governs, not EXPERIMENT_PLAN.md; and the governing doc contradicts itself about where its own governing section is.*

**Corrected 2026-08-19.** This memory previously said `docs/EXPERIMENT_PLAN.md`
is the only current plan. That is no longer true.

`docs/MASTER_PLAN_EXPERIMENTS_AND_COMPARATORS.md` (last commit 2026-08-15,
newer than EXPERIMENT_PLAN's 2026-08-12) opens
`# COMPOSE — CURRENT GOVERNING EXPERIMENTAL PLAN` and states at ~line 447 that
`EXPERIMENT_PLAN.md`'s "must-run" external list is **now overtaken**. Its map
sets QED/GrIDDD as the current critical path plus MOLLEO Task 3;
`EXPERIMENT_PLAN.md` never mentions MOLLEO at all.

**The governing document contradicts itself, so read it carefully:**
- line 3 says everything below `ARCHIVED PROVENANCE` is not governing
- line 426 is `# ARCHIVED PROVENANCE — NOT GOVERNING`
- but `⭐ CURRENT AMENDMENT` (1141) and `⭐⭐ CURRENT AMENDMENT II` (1740) sit
  **below** that divider, and 1742 reads "This is now the governing section."

Resolution in force: the amendments govern (see
[[current-amendment-pointer]]), i.e. Amendment II is the operative text despite
sitting under the archived divider.

`docs/ESTATE_REGISTRY.md:11` still claims EXPERIMENT_PLAN is "the only
experiment plan" — stale. `ESTATE_REGISTRY`, `ARTIFACT_INDEX`, `CLAIM_LEDGER`
and `PROJECT_BOARD` are all 325-1247 commits behind and contain verified
falsehoods; see `docs/REGISTRY_AUDIT_2026-08-19.md` before trusting any of them.
Superseded docs and the `paper*/` directories are preserved DELIBERATELY.

## compose-scope-lock
*"Scope lock in force until the 64-source QED result lands — what is authorized, what is barred, and the rule that kills runtime-estimation pilots."*

**In force from 2026-08-15 until the frozen 64-source QED dev panel result
lands.** The next QED information anyone cares about is the preregistered
64-source result — **not another timing number**.

> **Stop doing pilots whose only remaining purpose is runtime estimation.** If a
> run is not required by the frozen protocol or a mechanical launch gate, stop it
> and launch the real experiment unchanged.

1. **QED** — no more pilots, timing probes, caching work, batching work, sampler
   modifications, or controller diagnostics outside the frozen panel. The only
   next scientific action is to read the frozen gates and select rejection vs
   SMC vs controller-failure.
2. **MOLLEO** — benchmark **infrastructure only**: finish/freeze the five Task-3
   oracles, verify unique-molecule counting against the released implementation,
   resolve the unmetered `select_pareto_front()` issue, harden the 10k meter,
   keep the harness passing end-to-end with a non-COMPOSE dummy policy.
   **Do not develop, compare, or tune any COMPOSE five-objective policy.**
3. **Hard support / pathwise** — no claim-bearing runs, no new controller work.
   Shared infrastructure stays parked. **Do not polish it.**
4. **Closed branches stay closed** — exact target, retargeting, P3/P4, P0c, K41,
   Policy B rescue, DDSBM. No new probes.
5. **When the 64 lands** — apply the preregistered decision rule **once**. Do not
   ideate before reading the gate. SMC earned → abandon rejection optimization,
   implement/qualify SMC. Rejection survives → qualify only semantics-preserving
   production improvements. **Freeze the surviving path BEFORE the 128.**
6. **Use only diagnostics already in the preregistration** to separate
   value-model signal from sampler failure. **Do not invent new diagnostic
   experiments after seeing the panel.**

**Scientific priority:** completing the remaining claim-bearing experiments, not
elegant or fast implementations. **Engineering work is authorized only when it
unblocks the next frozen experiment.**

```
64 QED gate → 128 → 800 + size-fixed
   ‖ (parallel, benchmark contract only)
   MOLLEO oracles/meter → develop+freeze MOLLEO policy → official MOLLEO
                                                       → hard support + pathwise
```

Anything that does not move one of those boxes must justify its existence.
Related: [[hphi-amendment-wording-rule]], [[modal-budget-is-a-hard-ceiling]].

## composition-is-controller-not-rtheta
*Authoritative audit at true mid-chain growth states -- R_theta favours nitrogen; the frozen carbon_rich spec is the bottleneck, and stoich already fixes it.*

Measured 2026-08-30, `diagnostics/composition_mass_audit.json`,
`modal_apps/composition_mass_audit_app.py`. 15 T4 seeds x 4 arms, **313 real
decision states**, CPU only. All mass is canonical-successor mass after fiber
aggregation (see [[measure-successors-not-marks]]).

**Stratification is the whole result.** First-insertion states (tip=None) and
mid-chain states disagree about nitrogen:

| state class | n | C support/R_e/ratio | N support/R_e/ratio | O ratio |
|---|---|---|---|---|
| first (tip=None) | 60 | .076/.371/**4.91** | .076/.049/**0.64** | 2.13 |
| **mid-chain** | 253 | .088/.284/**3.22** | .088/.193/**2.19** | 7.22 |

So R_theta *upweights* N 2.19x over support at the states where ring composition
is actually chosen. An earlier probe reporting "R_theta suppresses N ~7x" used
tip=None states only and was wrong for the states that matter.

**The bottleneck is the frozen spec.** At the SAME mid-chain states:

| spec | median candidates | median share of the step's R_theta mass |
|---|---|---|
| `carbon_rich` (frozen) | 3 | **0.284** |
| `mixed` | 8 | **0.955** |

`carbon_rich` discards ~2/3 of the reference mass, and it is exactly the N/O
mass. Conditional odds are healthy too: with a live quota, R_theta puts median
**0.517** (IQR .388-.803) of step-compatible mass on continuations that can still
satisfy the residual stoichiometry.

**Heteroatom rings build.** trace_ok == semantic_ok on every arm; **zero** cases
of "compiler says OK, graph disagrees", so the earlier 3/14 discrepancy does not
reproduce on the `build_ring_system_exact` + `stoich` path. Completion, aromatic
required: carbon_rich 15/15, C5N1 11/15, C4N2 11/15, 5-ring C4N1 **8/15** --
the 5-ring failures are all `UNSAT@aromatise`, i.e. the ring forms saturated and
electronic completion fails.

**Verdict: Case C, controller-specification failure.** Not support, not
reference allocation. `stoich` (running quota) already exists and works.

**How to apply:** the fix is to let the controller request stoichiometry rather
than widening `COMPOSITION_CODES`. Watch the 5-ring aromatic path separately --
`UNSAT@aromatise` is a real, distinct defect. Verifier caution: my first version
did not check the requested electronic state and scored 3 saturated pyrrolidines
as aromatic successes; `semantic_ok` must include `state_ok`.
See [[t4-is-refinement-limited]].

## connectivity-potential-works-prune-missing
*Directed splitting sentinel -- 12/12 vs 0/12 against the matched control, handoff in 2-3 steps; but the old region is never removed because no prune potential was implemented.*

Measured 2026-08-31, frozen kernel (beta=6, W=64, eps=0.1, horizon 16, 12
trials), `diagnostics/region_sentinels.json`. Matched ablation at identical
horizon and kernel budget.

| case | directed | undirected |
|---|---|---|
| pendant | **12/12 OK**, old_region_removed **True** | n/a |
| splitting_free (termH 1,1) | **12/12 OK**, first_handoff_step **2-3** | **0/12**, handoff never |
| splitting_saturated (termH 0,0) | 0/12, new_material **0** | 0/12, new_material **0** |

**The ablation is the result.** On splitting_free the connectivity potential turns
an impossible search into one that succeeds in 2-3 steps of a 16-step horizon,
against 0/12 for the R_theta-proportional control. Same support, same budget.
This supports: the primitive process already contains the global restructuring
path; task-independent proposal geometry is what makes it computationally
accessible.

**But complete replacement FAILED.** old_region_removed=False,
n_old_atoms_surviving=2. Cause: I implemented Phi_conn for growth and applied it
in BOTH phases. There is no psi_prune rewarding safe elimination of the
superseded region, so prune_old is an undirected walk that spends its steps
doing nothing useful. psi_prune was in the design and was never built -- this is
an unimplemented spec item, not a tuning question.

**Saturated terminals: no new material was EVER created, in either arm.** With
both terminals at zero free hydrogen the executor offers no atom_insert on them,
so grow_new cannot start. Note the dense term rewards |new material|, which is
exactly what a saturated terminal cannot produce; a reroute/cycle-based handoff
would show delta-deficit only at the merge instant, giving a flat gradient again.
So this is partly support and partly a potential that cannot see the one route
available.

**Metric caution:** `n_handoff_reached` counts attempts whose stage is not
`no_connectivity_preserving_handoff`, so other failures inflate it -- undirected
saturated showed 2 that were `apply_failed`. Trust `first_handoff_step`, which
is None when handoff never held. Rename the metric before reuse.

**How to apply:** by the preregistered strict criterion (new connection THEN
removal of the old region) directed splitting does NOT fully qualify, so the
0.4-0.8 scale gate is not yet unlocked. See
[[bridge-handoff-is-proposal-not-grammar]].

## current-amendment-pointer
*"current amendment" means the ⭐ CURRENT AMENDMENT section at the bottom of docs/MASTER_PLAN_EXPERIMENTS_AND_COMPARATORS.md — read that section, not the blocks above it.*

When Rohin says **"current amendment"**, read the `⭐ CURRENT AMENDMENT` section
at the **bottom** of
`/Users/rmaganti/compose_v2_work/docs/MASTER_PLAN_EXPERIMENTS_AND_COMPARATORS.md`.

**Why:** the master plan grew to ~840 lines of accumulated history. Rather than
rewriting it and losing the record of what changed and why, the convention is
that everything above the amendment is the **historical record**, and the
amendment **governs where they conflict**. It holds the self-contained
experiment/comparator table, current experiment statuses, and the live external
benchmark choices.

**How to apply:** go to that section first for current state. Do not quote a
block from the body as current without checking the amendment for a
supersession — several blocks there are explicitly overtaken (DDSBM removed
from the manuscript, GrIDDD promoted, the K41 branch closed). When new rulings
land, extend or replace the amendment rather than editing the body, so the
"reference only this" property holds. Related: [[compose-canonical-experiment-plan]].

## dedup-keeps-survivor-intact
*Approval to delete a duplicate covers the deletion only; the surviving copy stays byte-identical, and prior state is read from the archive, never recalled.*

When two near-identical blocks exist and the user approves removing one, remove
that block and change **nothing** on the survivor. Near-identical is not
identical: diff the two first and report every field that differs, then let the
user pick. Merging the deleted copy's "better" version of a differing field onto
the survivor is an unrequested substitution, even though it looks like salvage.

**Why:** FORGE had two copies of `tab:ugi-common-benchmark`, data identical but
captions different -- copy A `[h!]` 48 words (rendered Table 2, p6), copy B
`[htbp]` 124 words (rendered Table 3, p7). The user approved deleting the
duplicate. I deleted copy B's block but moved its long caption onto copy A, so
Table 2 gained a caption it never had. The user: "table 2 has an insane caption
right i thought we were going with the other one." Then, asked what had been
there before, I asserted "two lines" from recollection -- correct by luck, but
the user was right to stop me: "look into it right before you deleted it."

**How to apply:** state the prior state only after reading it out of
`upload/versions/vN/`, and show the reconstruction (line numbers, placement
specifier, word count per copy). Every applied change gets a `diff` against the
previous version in the report, so an unrequested edit cannot hide inside an
approved one. See [[specified-edits-only]].

## ess-fallback-disables-the-controller
*adaptive_tilt returns the untilted base whenever base ESS <= target, so guidance is off exactly when the base kernel is peaked.*

`region_rewrite.adaptive_tilt` bisects the tilt temperature T so that
`q ∝ R_M · ĥ^(1/T)` hits a target ESS. It opens with

    if ess_frac(q_at(hi)) <= target_ess:   # even the base kernel is peaked
        return q_at(hi), hi, ...

so when the base proposal's own ESS is already at or below the target it
returns the UNTILTED base. Measured on 16 held-out regions (2026-09-02): base
ESS 0.061 against `target_ess=0.3`, so the branch fired at all 79 steps, T
pinned at its bound 1000, `ĥ^(1/1000) ≈ 1`, and the guided proposal equalled
the base proposal to four decimals.

Written as an ESS safety measure, it disables the controller in exactly the
regime that motivates it: a peaked base kernel is when guidance matters most.

**The committor itself is fine.** `h_max_progress / h_max_all = 1.000` median --
the progress successor is the top-scored candidate at essentially every step, at
h = 0.85-0.88.

**RESOLVED 2026-09-02** by `kl_tilt`: a base-relative trust region,
`q_eta ∝ R_M · h^eta` with the largest `eta` such that `KL(q_eta ‖ R_M) <= kappa`.
Always attainable, since `eta=0` recovers `R_M` at `KL=0`. Progress mass went
0.00004 -> 0.09359 at kappa=1 from the SAME committor. With guidance actually
applied, 16 particles completed 7/16 held-out regions where the base kernel
completed 0/16. `propose(tilt="kl", kappa=...)` selects it.

**How to apply:** any guided-vs-base result must first check the reported tilt
temperature and ESS. If T sits at the bisection bound, guidance was off and the
comparison is R_M against R_M plus overhead -- which is what the 0/128 single
attempts and 0/16 population on this campaign actually measured. Fixing it is a
design choice: apply a fixed modest tilt when the target is unreachable, define
the target RELATIVE to base ESS, or bound KL instead of ESS.
Related: [[matched-budget-needs-one-binding-resource]].

## exact-target-branch-closed
*"Exact-target recovery work is finished and deliberately closed as of 2026-08-12; next value is retargeting, pathwise constraints, and goals with no target molecule."*

As of 2026-08-12, after the sealed 65-pair result (greedy 40% → verified rollout
62%, h_phi top-1 also 62% at ~35% of continuations), Rohin closed the
exact-target-recovery branch. Explicitly ruled out: tuning h_phi further,
sweeping more K, expanding to ~500 successors, building a pairwise h_phi, or
manufacturing another target-recovery panel.

Next, in priority order: (1) dynamic retargeting from a realized intermediate,
(2) pathwise constraints holding throughout a trajectory rather than at the
endpoint, (3) a goal with no known target molecule.

**Why:** exact-target recovery is the one goal with a privileged non-learned
heuristic — Tanimoto-to-the-answer — so it cannot settle whether learned
desirability beats similarity, and further percentage points there are worth
less than any capability a static optimizer actually lacks.

**How to apply:** do not propose exact-target follow-ups. Two phrasings are
barred as unsupported: "h_phi beats similarity" (not preregistered) and "h_phi
reproduces full rollout" (matches the count, not the rescue set). See
[[compose-canonical-experiment-plan]] §4A.

## greedy-potentials-cannot-cross-plateaus
*Both region-replacement failures are one bug -- greedy shaping cannot take an enabling move whose own step gain is zero; cycle_open is the move in both cases.*

Measured 2026-08-31, `diagnostics/executor_reachability.json`,
`modal_apps/executor_reachability_app.py`. Family-stratified BFS (quota 4 per
family, depth<=5, beam 40), so absence of a path could not be confused with low
R_theta probability. Diagnostic only, not the deployed sampler.

| probe | start | path found | verdict |
|---|---|---|---|
| PRUNE, splitting_free | d_old = 0 | depth 3: **cycle_open -> atom_delete -> atom_delete** | PATH EXISTS |
| PLATEAU, splitting_saturated | d_old = 4 | depth 2: **cycle_open -> cycle_close** | PATH EXISTS |

**Both are proposal problems, not operator problems, and it is the same defect.**
`cycle_open` is the enabling move in both. Its own one-step gain is zero or
negative under the relevant potential -- it removes no old atom (so Phi_prune is
flat) and it does not by itself reduce d_old (so Phi_establish is flat) -- yet it
unlocks the goal on the very next step. Greedy single-step shaping cannot cross
that plateau, no matter how large beta is.

That explains BOTH standing failures at once:
- prune left 2 old atoms in place even at a structural budget of 5, because the
  interior atoms only become deletable AFTER the ring is opened;
- saturated never reached handoff, because rewiring needs open-then-close and the
  open step looks worthless.

**Correction to my earlier reading.** From `min_d_old_reachable` in [3,4] I
leaned toward calling saturated an operator-support limit. That was wrong: no
single-step progress exists, but a 2-step path does. The caveat I attached
mattered more than the headline.

**How to apply:** the fix is lookahead in the shaping, not bigger beta, not more
horizon, not new operators. A k-step backup, Phi_k(x) = max over paths of length
<= k of Phi(endpoint), keeps the proposal likelihood-evaluable and is the same
shape as the h-transform the project already uses. k=2 suffices for both
measured cases. Do NOT hand-add "prefer cycle_open" -- that is the benchmark
hacking this whole line exists to avoid. See
[[connectivity-potential-works-prune-missing]] and
[[bridge-handoff-is-proposal-not-grammar]].

## h40-hphi-line-closed
*The budget-extended (H40-aware) h_phi retraining line was tested and closed on 2026-08-18; what was measured and what not to re-propose.*

Training a budget-extended `h_phi` (one-hot widened 25 -> 41 slots, trained on
`train_0256x02_H40`) was tested end to end and did NOT pay off. Four
independent measurements agreed:

- unguided pooled-region AUC: H40 head worse at every lookahead 4..40
- run-level prospective AUC on banked marginal/hard runs: no statistical
  support for the long-range gain (P(new>old) ~= 0.61, 95% CI [-0.44, +0.80]);
  the only well-supported effect was a SHORT-range regression at lookahead 4
- paired H40 controller A/B, 12-source panel: decision strata 4/8 -> 3/8 sources
- monotonicity: meaningful `h_{b+1} < h_b` violations on 28.9% of states vs
  4.7% for the frozen H24 head

Do not re-propose "train h_phi on a longer-horizon corpus" without a new reason.
A full-size H40 corpus was priced at ~$8-9 and deliberately NOT bought, because
the binding constraint on the measurement is ~8 hit trajectories, not training
rows.

**Refuted along the way:** the hypothesis that `h_phi` ignores its budget input.
It does not -- sweeping b moves the logit about as much as changing the molecule
(ratio 0.93 old, 1.23 new). Flat AUC across lookaheads is not evidence of
budget-insensitivity.

**Still open, and the one real lead:** BOTH heads violate `h_{b+1} >= h_b`, which
is definitional for a hitting probability rather than a fit target. Enforcing it
architecturally is a distinct proposal from "more data".

The frozen H24 head stays the controller. See [[compose-scope-lock]] and
[[exact-target-branch-closed]].

## heldout-sentinel-regions-are-nearly-all-unreachable
*1/52 sentinel splitting regions are short-horizon reachable, so that pool cannot answer any guided-vs-unguided question.*

> **BUDGET CAVEAT.** These rates mean "completable within B=5 primitive
> edits", not "rewriteable". Region-scale rewrites may need 20-40 edits.
> See [[reachable-means-budget-not-rewriteable]].


> **PROVISIONAL (2026-09-01).** The numbers here were produced with law caches
> keyed on `canonical_state_key`, which collides across states differing only in
> slot layout (measured 9/65 keys). On a collision the cache returned another
> state's marks, actions and probabilities, which were then applied to the
> current state. The direction of the resulting bias is NOT known -- do not
> assume it was pessimistic. Caches are now keyed on exact slot layout; treat
> every quantitative proposal / reachability / committor figure below as
> provisional until re-derived. See [[law-cache-slot-key]].


Screening EVERY eligible splitting region on the six sentinel molecules
(dv[:6]) with the frozen search (depth 5, beam 40, per-family 4) gives
**1/52 reachable, 0/20 saturated**, and the single hit only at depth 5.
Training molecules (dv[6:22]) gave 22/204 = 10.8% on the identical procedure.

Consequences, both measured rather than assumed:
- The plateau-ranking benchmark is undefined here: with one reachable region
  there is no contrast to rank, and `pairs=0` measures the absence of a ranking
  problem, not committor quality.
- Any R_M vs R_M*h_phi comparison on this pool returns 0 vs 0 regardless of the
  controller, which is the same trap as [[hz-guidance-cannot-rescue-t4-growth]].

**How to apply:** before running a guided-vs-unguided comparison, screen the
pool and report reachable/eligible first. If it is near zero, change the pool --
do not run the comparison and report the null. The 22 dev seeds are now fully
consumed (6 sentinel + 16 training); a powered held-out pool needs fresh
molecules, and drawing them from PMO/MOLLEO banks adds distribution shift that
must be stated.

## hphi-amendment-wording-rule
*Region-h_φ is a pre-outcome amendment consistent with the pre-existing finite-horizon controller — never call it a restoration of an originally-frozen policy.*

The GrIDDD/QED claim-bearing controller was amended from Policy B (refuted
0/320 on the disjoint 64-source dev panel) to region-`h_φ`, frozen 2026-08-14
**before any official Jin test outcome existed**. Frozen in the ⭐ CURRENT
AMENDMENT at the bottom of `docs/GRIDDD_JIN_PROTOCOL.md`, cross-referenced as
section J of the master plan.

**The barred phrasing:** do **not** write that region-`h_φ` "restores the
originally specified policy," or that compute cost was the only thing ever
separating them. `GRIDDD_JIN_PROTOCOL.md` has a section explicitly titled *"Why
NOT the rollout `ĥ`"* — the protocol *declined* it. Claiming restoration
overstates the case and is refutable straight from our own document.

**The approved phrasing, verbatim:** region-`h_φ` is a **pre-outcome amendment
consistent with the pre-existing finite-horizon COMPOSE controller**, motivated
by a developmental failure of the cheap one-step approximation.

**How to apply:** the validity rests only on the amendment predating the
official evaluation — that is the whole argument, and it is enough. Policy B
stays in the manuscript as the failed developmental approximation rather than
disappearing. Until `h_φ` is mature it is the only scientific lane moving:
pathwise, hard-support and MOLLEO may be built against an abstract controller
interface, but no claim-bearing controller-dependent result may be produced,
and the native five-objective search policy is not frozen. Related:
[[current-amendment-pointer]], [[exact-target-branch-closed]].

## hz-guidance-cannot-rescue-t4-growth
*Task-aware H_z measurably improves realization selection but cannot make ring growth feasible on T4; 16 of 22 dev seeds have zero QED headroom for any ring addition.*

Measured 2026-08-30 on the DECLARED dev panel (`docs/GENMOL_T4_DEV_SEEDS.json`,
22 DUD-E actives, canonical-SMILES disjoint from the 15 GenMol benchmark seeds).
22 cells x 3 arms x K=8, both deltas. `diagnostics/guided_realization_delta*.json`.

**H_z works as a selector.** It changed 174/176 picks vs unguided and moved every
axis the right way (delta=0.6): median QED 0.323 -> 0.353, median SA 3.83 ->
3.63, SA<=4 passing **100 -> 131**.

**It does not convert into feasibility.**

| delta | narrow_frozen | free_unguided | free_guided |
|---|---|---|---|
| 0.6 | **14**/176 | 8 | 7 |
| 0.4 | **14**/176 | 11 | 12 |

**The stratification is the whole story.** EVERY feasible endpoint in EVERY arm
comes from the 6 seeds with seed QED >= 0.6. The other 16 seeds produce
**0/128 feasible in all three arms** at both deltas.

So on 16 of 22 dev seeds the ring-growth macro is structurally wrong and no
realization control -- guided or not -- repairs it. On the 6 seeds where growth
is viable, all-carbon aromatic rings remain the most feasible choice, because
heteroatom rings cost SA and similarity (SA<=4 passing: narrow 168, open 100).

**Why this matters architecturally:** the correct controller action on those 16
seeds is *do not invoke this macro at all*. That is an OPTION-SELECTION decision,
not a realization decision. This experiment is therefore evidence for the option
layer, not against the realization layer -- H_z did its job within the option it
was given. Confirms [[t4-seed-qed-headroom]] on an independent declared panel.

**How to apply:** do not price docking off this -- guided did not restore
feasibility over unguided, which was the preregistered gate. Report under the
approved phrasing in [[realization-claim-limits]]. See
[[realization-exposure-is-not-enough]].

## law-cache-slot-key
*Cache marked laws on exact slot layout, never on canonical_state_key -- actions carry slot coordinates.*

Never key a marked-law or successor cache on `canonical_state_key`. It is a
graph-isomorphism key, so two states differing only in which slots hold which
atoms share it, while mark actions carry slot COORDINATES. A hit then returns
another state's marks, actions and probabilities, and applying index j performs
a different edit than the model scored: some actions raise and are silently
dropped, the rest carry mis-assigned probability. Nothing fails loudly, because
the resulting molecules are still valid.

Measured 2026-09-01: 9 of 65 canonical keys covered more than one slot layout
during ordinary held-out exploration, and a canonically-keyed successor memo
returned 381 wrong successors out of 909 on a single region -- with ZERO feature
mismatches, which is exactly why it hid.

**How to apply:** key on `(atom_types.tobytes(), bonds.tobytes())`. Canonical
identity remains correct for state-discovery dedup ("have I recorded this
state"), so do not blanket-replace it. Before adopting any successor-scoring
optimisation, run an equivalence suite that compares cached against fresh on
mark rule names, actions and probabilities bitwise, and that reports how many
compared states sat on a colliding key -- otherwise the suite passes without
ever exercising the bug. Related: [[measure-successors-not-marks]].

## local-to-global-gate-passed
*Every scope band produces coherent rewrites; cost rises monotonically with scope; arbitrary splitting regions are executable ~1% of the time.*

General variable-scope gate, 2026-09-02. 239 regions SAMPLED by
released-fraction band across every interface class (not selected on
reachability), one particle each, frozen controller at kappa=1.

    band       n   ok  rate  med steps  med calls  med sec  med r_coh
    0.0-0.2   48   19   40%          3        475       60       0.07
    0.2-0.4   48   12   25%         11       1989      108       0.08
    0.4-0.6   47   13   28%         18       4214      190       0.08
    0.6-0.8   48   14   29%         21       6963      256       0.10
    0.8-1.01  48   22   46%         27      11459      448       0.15

**Cost by scope is the number the population controller needs:** a global
proposal costs ~7.5x a local one in wall time and ~24x in executor calls. Budget
many cheap local proposals and few expensive global ones.

**Scale holds sub-proportionally.** Median r_coherent 0.07 -> 0.15, max 0.30,
while r_change runs 0.36-0.65 against r_coherent 0.21-0.30 -- large-scope edits
are DISTRIBUTED, not one contiguous block. Ring systems do change (-2, -1, +1),
so it is topology restructuring.

**Interface class dominates success:** pendant 75%, segment 74%, multi 20%,
splitting **1/91**. Set against [[structural-execution-qualified]] (15/16
establishment on splitting regions selected BECAUSE a rewrite existed), the pair
says: arbitrary splitting regions are almost never executable, and when one is,
the controller finds it. Q(M|x,z) should prefer pendant/segment and treat
splitting as rare-but-valuable rather than as a default scope.

Rates are lower bounds at one particle. Mechanism work is DONE -- next is the
unified population controller, then T4/PMO.

## matched-budget-needs-one-binding-resource
*A two-arm comparison must be bound by ONE resource; a call budget plus a wall-clock valve silently binds each arm differently.*

When comparing two proposal arms at "matched budget", pick a single binding
resource and let only that one stop both arms. Giving each arm both a call
budget and a wall-clock valve means whichever binds first differs per arm, and
the comparison is invalid in a way the summary numbers hide.

**Why:** on the R_M vs R_M*h_phi held-out run I set budget_calls=4000 AND a
1500s valve. The call budget stopped the tilt arm after ~4 trials per region in
95s; the valve stopped the base arm at 25 min having used only ~1100 of its
4000 calls. The tilt got 4.5x the executor calls, 1/16 the wall-clock and 1/26
the restarts. Both arms scored 0 and it looked like a clean null.

**How to apply:** state the binding resource in the app docstring, assert only
one limit is active, and always report per-arm trials, calls, calls/trial and
whether the limit fired. If the two arms have wildly different per-trial costs
(here 1280 vs 11 executor calls), that ratio is itself the finding -- see
[[amortized-h-still-pays-executor-calls]]. Related: [[microbenchmark-before-launch]].

## measure-successors-not-marks
*Any claim about what R_theta prefers must be computed on canonical successors after fiber aggregation, never on raw edit marks.*

The scientific object is the canonical successor law

    R_theta(y|x) = sum_{a in G_y(x)} p_theta(a|x) / Z,   G_y(x) = {a : T(x,a) ~= y}

**Never report mark-level counts or mark-level probability mass as if it were a
statement about chemistry.** The action space is over (element, valence)
classes, so S has three valences, I three, P two, C one. Mark multiplicity
therefore differs per element *by encoding*, and any element histogram over raw
marks measures the encoding, not the process.

This makes the docstring figure in `match_growth_descriptors` -- "carbon is only
7.5-7.9% of legal atom_insert actions" -- implementation trivia. It is a correct
statement about draw efficiency and a correct reason to select on descriptor
rather than by rejection. It is **not** a neutral support probability, and it
must not be used as the denominator in a bias argument.

The quantity to report at a growth state, conditional on making an insertion:

    R_e(x) = sum_{y in S_insert,e(x)} R_theta(y|x) / sum_{y in S_insert(x)} R_theta(y|x)

**Why:** aggregating G_y(x) is the whole reason COMPOSE's control layer is
alias-invariant. Measuring marks re-imports the multiplicity the construction
exists to remove, and it would have produced a carbon-bias number that was an
artifact of valence-class counting.

**Related trap, same session:** a composition probability like P(C4N2 | x) is
NOT the product of per-step marginal element masses. It is a sum over complete
executable realization paths, state-dependent at every step, with quota
constraints reshaping support after each insertion and canonical aliases
coalescing paths. Use exact enumeration/DP over (state, remaining quota) or a
qualified estimator. See [[t4-is-refinement-limited]].

## microbenchmark-before-launch
*Every new expensive loop gets a 1%-scale timing microbenchmark before the real launch; never size a run from a guess.*

Before launching any new expensive loop, run it at ~1% scale first — 20 candidates,
or 1,000 prescreen molecules — measure, extrapolate, and only then launch the real
thing. State the extrapolation out loud when reporting the launch.

**Why:** on 2026-08-21 two multi-hour surprises came from sizing runs by estimate
instead of measurement. A scaffold continuation probe was sized at 46,080 fiber
enumerations and could not finish a single state in 10 minutes after I told the
user it would take 30–60 minutes. A T4 `future_h` arm ran 2h45m across three cells
and persisted zero results, because per-round cost was never measured. Both were
caught only by looking at what had not appeared, which wastes both money against
[[modal-budget-is-a-hard-ceiling]] and the user's time.

**How to apply:** treat "I estimate this takes N minutes" as unusable unless a
measurement backs it. When a prior run exists, derive the rate from its recorded
seconds-per-unit rather than re-guessing. Pair this with
[[persist-expensive-deterministic-artifacts]] — measure first, then stream and
persist per unit so a slow run is still distinguishable from a hung one.

## modal-budget-is-a-hard-ceiling
*"The stated Modal dollar cap is a hard authorization ceiling, not a warn-me threshold — stop at it and ask for an explicit amendment before spending past it."*

When a Modal spend cap is stated ("i want a modal cap right of 2 dollars"),
treat it as a **hard authorization ceiling**. Estimate cost *before* launching,
and if a run would exceed the cap, do not launch it — ask for an explicit
budget amendment first.

**Why:** the `h_φ` encode was launched at ~$4.39 against a stated $2 cap. The
sum is scientifically trivial; the problem is that silently exceeding our own
frozen compute control is inconsistent with the discipline applied everywhere
else in this project, and it invites exactly the "they relax their own rules
when inconvenient" reading we are trying to be immune to. The user's rule:
*"don't retroactively pretend it was under cap."*

**How to apply:** price the run first and say the number out loud before
launching. If over cap, stop and request an amendment to a specific figure. If
a cap is exceeded, record the exception explicitly rather than letting it pass
unremarked. Cost estimates have been badly wrong before in both directions
(encode predicted ~21 min / $1.35, actual 70 min / $4.39), so quote measured
per-container rates, not guesses. Related:
[[persist-expensive-deterministic-artifacts]].

**Amended 2026-08-23.** The user has said explicitly that Modal spend is not the
binding constraint for the COMPOSE benchmark push and that exceeding a previously
stated figure is acceptable. Do NOT halt work to ask for re-authorisation at every
threshold. What still matters, and what actually failed on 2026-08-23, is
**tracking**: the running total went from a stated ~$5 to ~$34 across ~15 T4 jobs
without a single re-tally, so the overrun was discovered by accident rather than
reported. Keep pricing runs before launching and keep a running total in the
report, but treat the number as information for the user rather than a gate.

## modal-detach-required
*"Modal `--detach` only protects the last triggered function; durable shards plus volume-based polling is what actually survives a client death."*

Four separate failures, each of which cost real time before the mechanism was
understood:

1. **`.map()` stalls** when the client stops iterating → run the fan-out inside
   an on-Modal `drive()` function.
2. **The app dies with the client** unless launched with `modal run --detach`.
3. **`--detach` is NOT sufficient.** Modal's own message explains why: detached
   mode only keeps the **last triggered function** alive past a client kill. It
   did not survive a client-side DNS failure (twice), a harness reaping the
   client, or `timeout 300 modal run --detach …` — which cancels the job
   outright. Never wrap a detached launch in a client-side timeout.
4. **Log-based progress checks lie** once the client dies. The log stops
   updating while the detached run continues, so a waiter can report a stall
   that isn't happening or miss the finish entirely.

**Why:** what actually protects work is (a) a durable per-task shard committed
as each task completes, (b) a driver that skips already-committed shards on
relaunch, (c) per-arm checkpointing inside tasks over ~20 minutes, and (d)
polling the **volume**, never the log, for progress and completion. That
combination turned a total client loss into a 2-source loss instead of 12, and
survived three network outages with zero completed work lost.

**How to apply:** launch long runs orphaned from the client —
`nohup … & disown` from a wrapper that exits immediately — in addition to
`--detach`. Check progress with `scripts/lane_status.py`, which reads the volume
and flags stalls and duplicate apps. Never tell Rohin it is safe to walk away on
the strength of the flag alone; confirm shards are accumulating.
See [[compose-canonical-experiment-plan]].

## modal-launch-from-committed-tree
*"Modal launches must come from the committed working tree, never the scratchpad worktree; a pilot that passes from a different directory proves nothing about the full run."*

Always launch Modal apps from `/Users/rmaganti/compose_v2_work` (the committed
working tree), and before launching verify that every module the app imports is
committed **on the branch being mounted** — not merely present on disk somewhere.

**Why:** the DDSBM full run (5,984 sources) failed on *every* source with
`ModuleNotFoundError: compose_v4.experiments.pareto_control`, after a 16-source
pilot had passed 16/16 on the same code path. The pilot was launched from the
`/private/tmp` scratchpad worktree, which had the module; the full run mounted
the branch's `src/`, which did not. The file was committed only on
`codex/compose-pareto-control`. Every failure burned ~160 s of container init.

**How to apply:** before any Modal launch, parse the app's `compose_v4` imports
and confirm each resolves in the mounted tree, then *actually import the chain*
rather than only checking that files exist. Treat a green pilot as evidence only
if it ran from the same directory the full run will mount. See
[[scratchpad-is-not-durable-storage]].

## persist-every-iteration-not-on-completion
*Write partial state after every iteration; a unit that persists only when finished loses everything if it stops one step short.*

Any long-running unit must write its state after EVERY iteration, not when the
whole unit completes. Persisting on completion means a unit stopped one step
short of the end contributes nothing, even though nearly all of its work is
already computed.

**Why:** the population-search A/B reached 176 of 180 iterations and still lost
three whole cells. Those four missing iterations were spread across three
different cells, each one or two short of its fifteenth, and `search_cell` only
wrote its JSON after the final iteration -- so roughly 93% of each cell's work
was discarded. The screen, collection and scale-gate apps in the same campaign
all wrote per unit as they went; the loop simply did not inherit that pattern.

**How to apply:** write the record every iteration with `complete: False`, and
set `complete: True` on the final write, so a reader can tell partial from
finished. Pair it with a `resume` entrypoint that reruns only named units, so an
interrupted run costs the missing units rather than the whole thing. Related:
[[persist-expensive-deterministic-artifacts]] (same rule at job scale) and
[[modal-detach-required]].

## persist-expensive-deterministic-artifacts
*"Persist expensive deterministic intermediates to the Modal volume the moment they exist, in a job separate from whatever consumes them — never at the end of a long job."*

If an intermediate artifact is **expensive** and **deterministic**, write it to
the Modal volume **as soon as it exists**, in a step decoupled from whatever
consumes it. Never leave it in container memory until the end of a long job.

**Why:** the `h_φ` trainer encoded 39,663 corpus states with frozen `R_θ` —
4,206 s wall × 80 containers = **93.5 core-hours ≈ $4.39** — then continued into
training in the same function. Training had a separate defect (no
best-checkpoint tracking, overfitting from epoch 0), so the run had to be
stopped, and the embeddings were destroyed with it. They were then re-encoded
from scratch, because the caching code was added *after* the run that could have
populated the cache.

**The summary-statistic variant, which is easier to miss.** On 2026-08-22 three
separate jobs computed expensive per-item results and persisted only an aggregate:
a PMO prescreen scored 249,455 molecules and saved the top 100, discarding 249,355
labels; an oracle-greedy diagnostic scored 18,031 COMPOSE-generated successors and
recorded `successors_scored: 18031` while keeping not one (molecule, value) pair.
Both sets were needed days-later — the labels to fit a surrogate, the pairs as the
only out-of-distribution validation set that existed — and both required a full
re-run. Writing a count instead of the data reads as "persisted" in review and is
not. **If a loop computes a value per item, persist the per-item values, not just
the statistic you happen to need right now.**

**How to apply:** structure it as `encode → persist → exit`, then a separate
`train` that loads the cache. Anything derived from a frozen checkpoint over a
frozen corpus can never change, so it should be written once and reused forever.
Corollary: before stopping a long-running job, check whether it is holding
expensive completed work that has not been flushed. Related:
[[modal-launch-from-committed-tree]].

## qed-benchmark-53-closed-items
*"Paper 5.3 QED benchmark: diversity is settled and closed, the output contract is frozen, and only three items remain."*

**As of 2026-08-20.** Diversity on the QED editing benchmark is **closed**. Do
not recompute it, re-audit it, or research it further. The settled value is
**0.393 [0.331, 0.452]** on the 128-source prospective panel, computed with Jin's
*released* `scripts/diversity.py`, not with the definition in the paper's prose.

The rule that makes the number is easy to get wrong and cost two wrong answers
before the audit: a source with **zero** valid molecules is **excluded** from the
average, but a source with **exactly one** enters as **0.0**. Nineteen of our 70
solved sources are singletons, which is the entire gap between 0.393 and the
0.539 you get by averaging only multi-molecule sources. Fingerprint is Morgan
r=2, 2048 bits, `useChirality=False`.

**The output contract is frozen** and the 800-run must honour it unchanged: each
source gets exactly K slots, every slot returns exactly one molecule, and a slot
whose trajectory never enters the goal region returns the **unmodified canonical
source**, which always fails. No slot gets a second attempt.

**Remaining 5.3 queue, in order:** freeze the official 800x20 contract, run the
size-fixed ablation under the already-frozen matched design (GrIDDD's comparable
ablation drops 45.1% to 33.8%), then run the official 800 once. **Do not add more
baselines or more metrics.**

Bar the claim that COMPOSE beats GrIDDD until the official 800 adjudicates it;
the cohorts differ. Related: [[compose-scope-lock]].

## reachable-means-budget-not-rewriteable
*"Reachable" means completable within B primitive edits, not rewriteable; never quote a reachability rate as a property of molecules.*

Every COMPOSE "reachability" number means exactly one thing: a complete rewrite
was found by the frozen bounded search within **B primitive edits** under
frozen-context rules. With B=5 that is a narrow statement about the budget, not
about the molecule or the operator set. A scaffold rewrite may legitimately need
20-40 primitive edits; calling that region "unreachable" at B=5 is like calling
a city unreachable because you allowed five turns.

B=5 was a fair UNIT TEST for future-aware plateau reasoning, because the known
paths (`cycle_open` -> `cycle_close`, `cycle_open` -> delete -> delete) are 2-3
edits long. It drifted into a development gate, and I then reported "0.7% of
regions reachable" as if it described molecular rewriteability. It does not.

**How to apply:** always say "completable within B=n", never "reachable", when
quoting these rates. The rewrite allowance must scale with region scope --
B(M) from |V_M|, |E_M|, |boundary M| -- because region scope is variable by
design. And do not require arbitrary regions to succeed: the outer controller's
job is choosing which region is worth rewriting, the execution controller's job
is rewriting a chosen one. Some selected regions are simply bad choices.
Related: [[heldout-sentinel-regions-are-nearly-all-unreachable]],
[[structural-terminals-are-rare-across-regions]].

## realization-claim-limits
*What the guided-realization experiment may and may not claim; the anchor-level prior is not a full path density, and the open arms buy extra cheap search.*

Two constraints on reporting `modal_apps/guided_realization_arm_app.py`.

**1. Do not call it exact R_theta x H_z path control.** The `prior` passed to
`select_endpoint` is ANCHOR-LEVEL R_theta insertion mass -- the mass of
atom_insert marks at each attachment site -- not the probability of the whole
multi-edit ring-building realization. So the selection is

    P(xi) ∝ (anchor mass) x H_z(y_xi)

which is a task-aware realization SELECTION rule, not

    P(xi | x,z) ∝ P_Rtheta(xi | x) H_z(y_xi)

over the complete path. **Approved phrasing:** "task-aware endpoint guidance over
a generic executable realization space improves realization selection."
**Barred:** "exact finite-horizon R_theta-twisted macro control" and any
phrasing implying the full realization density was used. The exact version needs
the complete path/realization likelihood -- which is also what the eventual
importance-corrected sampler needs (see [[measure-successors-not-marks]] for the
companion aggregation rule).

**2. Equal K endpoints matches the EXPENSIVE budget, not total computation.**
The open arms pool ~900 realizations across 35 specs; the frozen carbon-only arm
sees ~8-11. All arms advance the same K canonical endpoints and no docking is
spent, so there is no oracle confound. But the open arms do materially more
CHEAP CPU search. Always report alongside the result: candidate realizations
enumerated, unique canonical endpoints, kernel/scoring calls, wall time. Never
imply identical total computation. The honest claim is "improves
oracle-efficient optimization by using a richer cheap search over executable
chemistry."

**How to apply:** primary endpoint is conversion to T4-FEASIBLE outputs at
matched K. Do not price docking unless guided open composition restores or
improves feasibility over unguided open composition -- do not dock a controller
that cannot clear the cheap gates. See [[realization-exposure-is-not-enough]].

## realization-exposure-is-not-enough
*Exposing a ring option's admissible compositions produces the intended chemistry but does not improve T4 feasibility at matched budget, and hurts at delta=0.6.*

Measured 2026-08-30, `diagnostics/stoich_realization_arm.json`,
`modal_apps/stoich_realization_arm_app.py`. 6 dev seeds x 3 arms x K=8, **48
endpoints per arm, exactly matched**, CPU, no docking. Generic stoichiometry
space (35 specs) enumerated from declared RING_ELEMENTS = C,N,O,S,P. No IVG
composition referenced; no per-cell tuning.

**Realization: the mechanism works.**

| arm | endpoints with a hetero ring |
|---|---|
| narrow_frozen (`carbon_rich`) | **0 / 48** |
| free_composition (declared elements, R_theta picks) | **29 / 48** |
| stoich_enumerated (uniform over 35) | 9 / 48 |

**Downstream: no gain, and a real loss under the tight similarity floor.**

| gate | narrow | free | stoich |
|---|---|---|---|
| feasible, delta=0.4 | 16 | 17 | 15 |
| feasible, **delta=0.6** | **10** | **3** | **3** |
| QED>=0.6 | 16 | 17 | 16 |
| sim>=0.6 | 37 | **27** | 28 |

The binding gate is SIMILARITY, not QED: heteroatom substitution moves the
molecule further from the seed, so it falls through the 0.6 floor. QED pass is
flat across arms, so composition does not buy QED either.

**Conclusion:** exposing the admissible realizations is NECESSARY (0 -> 29
hetero rings) but NOT SUFFICIENT. This arm ran the realization selector with
H = identity, i.e. P(xi) proportional to R_theta(xi) alone. The measured claim
"same R_theta + same structural search + better realization control => better
optimization" is **not supported** with an unguided selector. The next test is
the same arm with the downstream value term, P(xi) ∝ R_theta(xi)·H_z(y_xi).

**Design flaw in my own experiment, to fix before rerunning:** the dev set was
`seeds[:6]` = parp1_s0-s2 + fa7_s3-s5. Four of six start below the QED gate
(0.438, 0.284, 0.186, 0.156) and the fa7 seeds are exactly the ones
[[t4-seed-qed-headroom]] says growth macros structurally cannot help. Pick dev
cells by QED headroom, not by list order. See
[[composition-is-controller-not-rtheta]].

## ring-macro-controller-interface
*The controller sees ring SEMANTICS (topology/size/composition/state); attachment site stays an internal realization — settled by a mass-matched ablation, not preference.*

Settled 2026-08-26 on 5ht1b_s7_d0.6.

**Ring construction is strongly useful.** Ring-containing trajectories averaged
-8.96 against -7.75 without, over 33 docked molecules — above the 0.70 kcal/mol
docking noise floor.

**Per-attachment-site CEM credit is not.** With total initial ring mass matched
at 0.24, six site-visible actions and one opaque action performed identically:
best -11.3 vs -11.2, mean -8.46 vs -8.57, two 28-heavy molecules each. An
8-replicate isomer test agreed: meta -10.62 vs para -10.36 at sd ~1.2.

**Why:** an apparent -9.6 -> -11.4 "win" from site-parameterisation was a
CONFOUND — replacing one ring action (mass 1/20 = 0.05) with six (6/25 = 0.24)
silently raised ring sampling ~5x. Always mass-match before attributing a gain
to a representation change. A single-shot docking difference on 28-heavy
flexible molecules (sd ~1.2) is not evidence of anything; the same para
terphenyl scored -7.60 and -11.30 on two draws.

**How to apply:** ACTION_SPACE carries `ring:{topology}/{size}/{C}/{state}`.
The compiler still enumerates every admissible realization WITH its predicted
endpoint molecule, and `select_realization()` implements
P(xi) ∝ R_theta(xi)·H(xi) with H identity — so a future contextual
h_phi(y_xi, z, b) can score candidate futures directly. Do not split scarce
oracle observations across unrelated global site tokens. Partial pooling
(`groups=`, shrink 0.25) is retained and tested for when sites return.
Topology is frozen at linked + fused; see [[t4-ring-type-coverage]].

## single-axis-parent-selection
*Selecting search parents by one objective climbs a direction that destroys the others; rank by the actual per-point contribution to the reported metric instead.*

Measured 2026-08-26 on MOLLEO Task 3 (strict-10k), and it repeats a mechanism
already seen on GenMol T4.

Selecting parents by `jnk3` alone produced a large, highly significant gain on
that axis -- paired vs random ZINC over 16 seeds: **+0.168 +- 0.120, 15/16 wins,
t = 5.45** -- and NO gain at all on the reported metric: hypervolume
+0.010 +- 0.064, t = 0.62.

The decomposition shows why. ~51% of hypervolume comes from ONE molecule, so its
full 5-vector decides the score:

| arm | best-jnk3 point | volume |
|---|---|---|
| random | jnk3 0.49, qed 0.779, sa 0.87 (its best-volume point TOO) | 0.322 |
| ours | jnk3 0.54, **qed 0.196**, sa 0.621 | 0.042 |

Random finds molecules that are active AND drug-like; single-axis selection
found active-but-not-drug-like ones. The ring macro raises jnk3 by adding
mass, which is exactly what destroys QED -- the same mechanism that drove
QED 0.904 -> 0.710 -> 0.392 when stacking rings on T4.

**Why:** a single-objective parent key climbs a coordinate whose ascent
direction degrades the rest. The gain is real and worthless.

**How to apply:** rank parents by the per-point contribution to the metric
actually being reported (here the product of all five objectives), not by the
sub-objective that looks like the bottleneck. And never present a significant
sub-objective effect as if it were a benchmark result -- they are different
claims. See [[t4-ring-type-coverage]], [[ring-macro-controller-interface]].

## specified-edits-only
*When given an explicit edit map, apply only its named edits; page-fitting and other judgment trims are the user's to make, not mine.*

When the user hands over an explicit edit plan (KEEP / CUT / MOVE / REPLACE, a
redline, a surgical map), apply **only** the edits it names. Do not reword,
condense, or delete anything else — not to hit a page limit, not to tighten
prose. If the specified edits alone do not meet a hard constraint (page count,
word count), say so, quantify the gap, and name the biggest lever. The user
trims from there.

**Why:** on the GEM workshop fork the map said most retained prose stays
verbatim. I applied its edits, then invented my own rewordings to force a
6-page body down to 5 — rewriting sentences the map had explicitly said to
keep. The user caught it: "if the agent didn't specify the surgical edits then
don't do it, i can trim myself." My trims also obscured the real finding: the
6th page was an *undrawn figure placeholder*, not prose, so cutting prose was
solving the wrong problem.

**How to apply:** keep the two categories visibly separate in the artifact
itself. Header comments in each edited file should state which map instruction
produced each change, and any departure from the map gets its own labelled
block explaining why. Additions count too, not just trims — pulling a sentence
back in from the source document is also an unrequested edit. See
[[verified-numbers-only]] for the companion rule on numbers.

## structural-execution-qualified
*Guided population completes 7/16 held-out regions where base R_M completes 0; structural execution is frozen and splitting is closed.*

Final splitting qualification, 2026-09-02. 16 known-reachable held-out regions
on molecules disjoint from the committor's training set, 16 particles per
(region, arm), matched horizon / ceiling / regions / seeds:

    R_M        256 attempts   2 establishment    0 complete    0/16 regions
    R_M·h_phi  256 attempts 142 establishment   26 complete    7/16 regions

Establishment in 15/16 regions at 4-15 particles of 16, completions in 7
distinct regions -- not one lucky case. N=16 came from the measured handoff mass
(0.0766) via 1-(1-m)^N and kappa=1.0 was declared, both before the run.

**Structural execution is qualified and frozen.** Splitting-specific development
is closed: no more sentinels, horizon studies, saturated campaigns or committor
variants. Next is the general variable-scope local→global region gate across
substituent, linker, ring, fused/linked and scaffold regions, then the unified
population controller, then T4/PMO.

**The result is CONDITIONAL execution efficiency, not coverage.** Regions were
selected because a rewrite is known to exist. Coverage -- how often a useful
region can be found at all -- is a separate question for the general gate; see
[[reachable-means-budget-not-rewriteable]].

Every earlier guided-vs-base zero in this campaign was an artifact of
[[ess-fallback-disables-the-controller]] and must not be cited.

## structural-terminals-are-rare-across-regions
*With a probe-matched goal search, only 2 of 48 arbitrary splitting regions yield any terminal path, and 0 of 15 saturated ones -- the difficulty is general, not sentinel-specific.*

> **BUDGET CAVEAT.** These rates mean "completable within B=5 primitive
> edits", not "rewriteable". Region-scale rewrites may need 20-40 edits.
> See [[reachable-means-budget-not-rewriteable]].


> **PROVISIONAL (2026-09-01).** The numbers here were produced with law caches
> keyed on `canonical_state_key`, which collides across states differing only in
> slot layout (measured 9/65 keys). On a collision the cache returned another
> state's marks, actions and probabilities, which were then applied to the
> current state. The direction of the resulting bias is NOT known -- do not
> assume it was pessimistic. Caches are now keyed on exact slot layout; treat
> every quantitative proposal / reachability / committor figure below as
> provisional until re-derived. See [[law-cache-slot-key]].


Measured 2026-08-31, `diagnostics/committor_gate.json`,
`modal_apps/committor_bellman_app.py`. 16 TRAINING molecules disjoint from the
sentinel set (dv[6:22] vs dv[:6]), 48 splitting regions, goal-directed search
matched to the reachability probe (depth 5, beam 40, R_theta-ordered frontier,
family-stratified, diversity by committing each search to a different opening
family).

**The search fix worked:** terminal successors went 5 -> **1037**, terminal
states 0 -> 13. The earlier collector shuffled the frontier before truncating to
16, which destroyed family stratification at the frontier level and gave
paths=0 on every unit.

**But terminals are genuinely rare across arbitrary regions:**

| | regions | yielded a path |
|---|---|---|
| splitting_free | 33 | 2 |
| splitting_saturated | 15 | **0** |
| total | 48 | **2** |

All 13 terminal states come from ONE molecule. Gate failed on
`distinct_molecules_with_terminals = 1 < 3` for free, and on zero terminals for
saturated -- correctly, since a committor fitted here would learn one molecule's
path motif.

**The saturated result is the substantive one.** It is not a property of the
chosen sentinel: 15 independent saturated splitting regions, on molecules the
sentinel never touches, produced no establishment path within depth 5. Combined
with the sentinel's own 0/12 -- and later 0/50 on the training screen and 0/20
on the held-out screen -- saturated splitting is very rarely SHORT-HORIZON
reachable on arbitrarily sampled regions.

**This is NOT an operator-set impossibility, and must never be recorded as one.**
A reachability probe previously exhibited a concrete saturated plateau path
(`cycle_open` then `cycle_close`). So the executor can do saturated splitting;
what is rare is a short-horizon path from an ARBITRARILY CHOSEN saturated
region. For many such regions the right move is to select a different
boundary/region, or allow a longer structural horizon -- not to conclude the
operators cannot express it. See [[greedy-potentials-cannot-cross-plateaus]].

Slowest unit 2847s at state_cap 60 -- collection is expensive because each
retained state stores its whole canonical successor law.

**How to apply:** do not fit a committor on this replay set. Either deepen the
search budget for terminals (and measure whether depth >5 finds saturated paths
at all), or accept that the training distribution must be built from regions
PRE-SCREENED for reachability, which changes what the model generalises over
and must be stated. See [[committor-needs-goal-seeded-replay]].

## t4-growth-option-is-the-wrong-move
*Pool-ceiling audit -- 20 of 22 T4 dev cells contain ZERO feasible ring-growth realizations, so the failure is broad option selection, not narrow steering.*

Measured 2026-08-30, `diagnostics/pool_ceiling_audit.json`,
`modal_apps/pool_ceiling_audit_app.py`. Deterministic re-enumeration of the SAME
pools as the frozen guided-realization run (declared dev panel, 22 cells,
pendant 6-ring aromatic, 35 stoichiometries, max_realizations=64). CPU only.

**The adjudicating number: 20 of 22 dev cells have ZERO endpoints clearing all
three T4 gates -- in EITHER pool, at EITHER delta.** Narrow contributes 245
unique canonical endpoints across the panel, open contributes 16,784, and on 20
cells not one of them is feasible.

Only two cells have any: `parp1_dev16` (seed QED 0.900) and `parp1_dev17`
(0.866). Everything from seed QED 0.72 downward yields nothing.

**So H_z is innocent.** It had nothing to select on 20/22 cells. The broad
decision "build a ring here" is what is wrong, and no realization control fixes
it. Correct controller output on those cells is *do not invoke this macro*.

**On the 2 cells where feasibility is achievable, guided does NOT beat
unguided** (lift over pool base rate):

| cell | delta | narrow | free_guided | free_unguided |
|---|---|---|---|---|
| dev16 | 0.6 | 8/8 | 3/8 (1.91x) | 3/8 (1.91x) |
| dev17 | 0.6 | 6/8 | 4/8 (1.75x) | 5/8 (2.18x) |

Residual regret is real there: dev16 at delta=0.6 has 141 feasible endpoints in
the open pool and guided took 3/8, where a constraint-first selector would take
8/8. But n=2 cells / 16 picks -- do not over-read it.

**Note the open space is not useless:** where the option IS appropriate, the best
feasible candidate is a HETEROATOM ring -- dev16 delta=0.4 best is C4N2
(QED 0.752), dev17 best is C4O2 (QED 0.726). The realization layer earns its
place once the option is right.

**Cost accounting** (closing the gap the frozen run left): unique canonical
endpoints 245 vs 16,784; realize() calls per cell 1 vs 35; kernel enumerations 1
per cell; median 17 s/cell.

**How to apply:** build the option-selection layer next, not a fancier H_z. If a
constraint-first realization selector is added, it is justified only by the 2
high-QED cells. See [[hz-guidance-cannot-rescue-t4-growth]],
[[t4-seed-qed-headroom]].

## t4-is-refinement-limited
*Stage-1 sigma-stratum audit says T4 failures are within-region refinement, not region entry; the committor-exit option is aimed at a capability we already have.*

Measured 2026-08-30 from `diagnostics/ivg_winners.json` + `diagnostics/t4_best_molecules.json`
+ `docs/GENMOL_T4_SEEDS.json`, over the **3 cells** where our best-molecule
artifact and the IVG winner artifact overlap. Script:
`$CLAUDE_JOB_DIR/tmp/sigma_audit5.py`.

**Both methods leave the seed's coarse stratum on the growth cells.** IVG grows
MORE than us there (+14.6 and +12.4 mean heavy atoms vs our +13 and +10). So the
gap is not region entry, and a generic structural-exit option addresses a
capability T4 does not lack.

The matched pair that settles it, `5ht1b_s7_d0.4` -- identical ring topology,
count, sizes and aromaticity, differing in ONE ring's composition:

| | rings | ds |
|---|---|---|
| seed | 6a:C6, 6s:C4N2 | |
| ours | 6a:C6, 6a:C6, 6a:C6, 6s:C4N2, **6s:C6** | -12.0 |
| IVG | 6a:C6, 6a:C6, 6a:C6, 6s:C4N2, **6s:C4N2** | -13.0 |

Aggregate over added rings: **all-carbon share OURS 83% vs IVG 51%** -- an
independent slice reproducing [[t4-ring-type-coverage]]'s 89% figure.

Second missing operation, distinct from composition: **ring EXPANSION of an
existing ring.** 4 of 5 parp1_s0_d0.4 winners carry `8s:C6N2`, an expansion of
the seed's own `7s:C5N2`. We never do this; we only ADD rings.

**Why:** this is the preregistered "within-region refinement failure" branch, so
the instrument is composition control in ring construction plus an
expansion/contraction operator -- NOT the committor exit, which is the most
novel piece of the proposed option framework and the one T4 cannot justify.

**How to apply:** n=3 cells, and the single-pair 1.0 kcal/mol delta is inside
docking noise (sd ~1.2) -- the distributional 83%/51% claim is the load-bearing
one. Before acting, widen IVG winner coverage beyond 5 cells
(`diagnostics/ivg_molecules/`) and run the same audit on PMO and MOLLEO, which
may well be entry-limited where T4 is not. Two wrong baselines were used before
the right one: `genmol_t4_ivg.json` `best_ds` is OUR reproduction of IVG
(-8.7 on 5ht1b_s7_d0.4), not IVG's winners (-13.0). See
[[benchmark-targets-from-artifact]].

## t4-ring-type-coverage
*Measured ring-type audit of the 25 IVG T4 winners, normalised to rings ADDED beyond each seed; sets the macro roadmap by evidence.*

Measured 2026-08-26 from `diagnostics/ivg_winners.json`. Count rings the winners
ADD beyond their own seed -- raw ring counts overstate gaps badly, because most
heteroaromatic rings are inherited from the seed, not built.

Of 43 added rings:

| added | ring | status |
|---|---|---|
| 16 (37%) | 6-aromatic all-C **linked** | built + frozen |
| 5 (12%) | 6-aromatic all-C **fused** | open |
| 12 (28%) | 5/6 **saturated N** rings (piperidine, pyrrolidine, piperazine, morpholine) | needs composition stoichiometry |
| 6 (14%) | 7-, 8-, 9-rings | not built, low priority |
| 2 | 6-saturated all-C fused | built |

**Why:** the largest single category winners add is exactly the linked
6-aromatic carbocycle, which is why BUILD_RING_SYSTEM(linked,6,C,aromatic) was
the right first macro. Spiro and bridged appear in ZERO of the 25 winners,
confirmed independently by RDKit CalcNumSpiroAtoms and CalcNumBridgeheadAtoms
both summing to zero -- do not spend qualification time there.

The 28% saturated-N gap is the SAME gap as the QED ceiling: stacking three
all-carbon aromatic rings drives QED 0.904 -> 0.710 -> 0.392, while IVG holds
QED 0.62-0.72 using amine-bearing saturated rings. `composition` is currently
an allowed-element SET, so it cannot request "6-ring, exactly one N".

**How to apply:** rank macro work by added-ring share, not by raw ring counts
and not by intuition. See [[t4-seed-qed-headroom]].

## t4-seed-qed-headroom
*9 of the 15 T4 seeds start BELOW the QED>=0.6 gate, so growth macros cannot succeed there at all; applicability is set by seed headroom, not by the macro.*

Measured 2026-08-26 over all 15 T4 seeds. Seed QED, before any edit:

- Above the gate (6): parp1_s0 0.888, parp1_s1 0.758, 5ht1b_s7 0.767,
  5ht1b_s8 0.716, jak2_s12 0.725, jak2_s13 0.712
- Below the gate (9): parp1_s2 0.438, fa7_s3 0.284, fa7_s4 0.186, fa7_s5 0.156,
  5ht1b_s6 0.438, braf_s9 0.235, braf_s10 0.346, braf_s11 0.255, jak2_s14 0.482

5ht1b_s8 additionally starts at SA 4.69, already past the SA<=4 gate.

**Why:** T4 constrains the RETURNED molecule. On a 37-40 heavy-atom, logP 5.2-6.8
seed (braf, fa7), adding an aromatic carbocycle moves QED further from 0.6, so
BUILD_RING_SYSTEM is structurally the wrong operator there no matter how well it
is implemented. Those cells require QED *repair* under a similarity floor -- a
different macro. Do not read a 0/8 feasibility on braf/fa7 as a macro defect:
the construction succeeded 120/120 mechanically, and similarity was 0.53-0.90
throughout. QED was the only binding constraint.

**How to apply:** Before proposing a growth macro for a cell, check the seed's
QED and SA first. Growth macros are for the small QED-rich seeds. The three
highest-deficit cells (5ht1b_s7 both delta, parp1_s0_d0.6) all sit in that
group, which is why the macro is worth pursuing at all. See
[[compose-scope-lock]] and [[qed-benchmark-53-closed-items]].

## t4-sentinel-results-2026-08-26
*The frozen semantic ring controller's measured T4 results on two cells, and the topology generalization that makes them a finding rather than a tuned cell.*

Measured 2026-08-26. Statistic is the comparator's: mean over 3 INDEPENDENT
runs of each run's best lead satisfying the PUBLISHED criterion only
(QED>=0.6, SA<=4, Tanimoto>=delta, parseable). Our stricter med-chem screen is
secondary; all six leads passed it anyway.

SEVEN cells, all at ~100 oracle calls against the comparator's 1000
(GenMol: num_gen=100 x num_iter=10; InVirtuoGen: --max_oracle_calls 1000).

| cell | ours | prior COMPOSE | GenMol | vs GenMol |
|---|---|---|---|---|
| parp1_s1 d0.6 | -10.90 +- 0.36 | -9.40 | -9.70 | **+1.20 BEATS** |
| braf_s10 d0.6 | -10.80 (40 calls) | none | -9.70 | **+1.10 BEATS** |
| parp1_s0 d0.6 | **-11.37 +- 0.12** | -8.80 | -10.40 | **+0.97 BEATS** |
| jak2_s12 d0.6 | -8.93 +- 0.26 | -8.30 | -8.60 | **+0.33 BEATS** |
| parp1_s0 d0.4 | -10.73 +- 0.12 | -10.50 | -10.60 | **+0.13 BEATS** |
| 5ht1b_s7 d0.6 | **-11.10 +- 0.16** | -8.80 | -12.00 | -0.90 |
| 5ht1b_s7 d0.4 | -11.23 +- 0.17 | -10.20 | -12.30 | -1.07 |

**5 of 7 beat GenMol at ~10% of its oracle budget.** All 21 leads pass the
secondary med-chem screen. Both losses are the same seed (5ht1b_s7).

**The generalization is the point.** Same frozen controller, no per-cell config:
- 5ht1b: linked mass 0.29-0.36, fused 0.07-0.16 -> linked terphenyl / naphthalene
- parp1: linked 0.20-0.22, fused **0.16-0.28** -> three fused tetralins per lead

It chose linked where published winners are linked, fused where they are fused,
from docking feedback alone. Artifacts: `diagnostics/t4_sentinel{A,B}_*.json`.

**UNVERIFIED, gates the paper:** GenMol's PER-RUN oracle budget. Ours is 100
unique docking calls per run. If theirs is 1000, this is a strong result at a
tenth the budget but NOT a same-budget table entry. Verify before Table 2.

braf_s10 d0.6 needs the feasibility bootstrap -- its seed is QED 0.346 against
the 0.6 gate, so 10 rounds gave 0 feasible / 0 docked. A cheap probe found 113
feasible endpoints at depth 5-6, so the set is reachable.
See [[ring-macro-controller-interface]], [[t4-ring-type-coverage]].
