# Decision log — Workstream E, target-free Pareto / preference control

Every material decision, the evidence available *before* it, the alternatives
rejected, whether it changes a frozen object, and the commit that contains it.

---

## D-001 · 2026-08-13 · Lane opened on `codex/compose-pareto-control` from `a0e680d`

**Decision.** Base the lane on the main-lane HEAD `a0e680d` ("Add greedy-restart
completion arm; audit the recovery-fraction denominator") rather than on any
parallel lane branch, so the frozen goal language, the DRD2 oracle artifacts and
the committed retargeting calibration are all present and hash-identical to
main.

**Evidence before the decision.** `git log` on
`codex/editing-v2-successor-fiber-fastpath`; the four frozen-input hashes in
`PROTOCOL.md` §3 recomputed on this tree.

**Alternatives rejected.** Branching from `codex/compose-pathwise-constraints`
(would inherit an unrelated lane's uncommitted design surface).

**Changes a frozen object:** no.

---

## D-002 · 2026-08-13 · The census is run LOCALLY, not on Modal

**Decision.** Stage 0 runs entirely on this machine: RDKit + the frozen DRD2 SVM
+ one-cut MMP mining. No Modal run is launched.

**Evidence before the decision.**
- `artifacts/oracles/drd2_svm_v1/` is committed and loads in 0.5 s on CPU;
  `oracle.margin_many([smiles])` returns log-odds directly.
- `mine_one_cut_pairs` over the full held-in pool of 96,094 molecules costs
  **51.6 s** and yields **81,500** pairs, 30,976 molecules with >= 2 neighbours.
- The `R_theta` checkpoint is **not** in the repo — it lives on the Modal
  artifact volume at `/artifacts/editing_v2/r_theta_run/runs/run_v2_01/`, and
  `enumerate_factorized_marked_law(model, state, t)` needs the model even to
  produce the legal mask. So true successor-fiber enumeration is the *only*
  part of the census that cannot run locally.

**Consequence, recorded before results.** C3 and C4 are measured on the MMP
proxy, whose bias direction is known and one-sided (median neighbour degree 1
versus a ~500-wide real fiber, so it *understates* tradeoff availability and
distinguishability). `PROTOCOL.md` §6.2 therefore licenses a PASS on G2/G5 from
the proxy alone but only a `PROVISIONAL_REJECT_PENDING_FIBER` on a FAIL.

**Alternatives rejected.** Launching a Modal fiber census before the cheap local
one — it would cost hours to learn something the local instrument can rule in
for free, and the lane contract requires a costed plan to main first.

**Changes a frozen object:** no.

**Superseded in part by D-002b**, below, before any census ran.

---

## D-002b · 2026-08-13 · The REAL successor fiber is enumerable locally; it becomes the primary instrument

**Decision.** Census C2–C4 on the real model-gated canonical successor support
(instrument I-A), with the MMP proxy retained as a cross-check (I-B). Taken
**before** any census statistic was computed.

**Evidence before the decision.**
- `canonical_successor_result(model, state, 0.5)` returns **334 and 339**
  canonical successors for two drug-sized held-in molecules at **5.2–5.7 s per
  state** on this machine, after a 63 s runtime build. That is the real
  ~250–600-wide fiber, not a proxy.
- The legal mask is structural — it comes from the model's action tables, not
  its weights — so the frozen `R_theta` checkpoint (which is only on the Modal
  volume) is **not needed to enumerate support**. The census reads support and
  properties only and never reads a probability.
- The local Active8 pairs with gate-zero **v7** while the volume's `RUN_PATHS`
  pins **v6**, which `scripts/verification/kernel_cost_profile.py` warns about.
  This was **checked, not assumed**: `process_identity_sha256`,
  `gate_zero_structural_contract_sha256`, `contracts_binding_sha256`,
  `active8_completion_sha256`, `enforced_structural_clauses`,
  `model_family_counts` and `executor_rule_counts` are **byte-identical**
  between the two decisions. They differ in corpus accounting, not in the
  executable support.

**Why this matters for the verdict, not just for precision.** Under D-002 alone,
a G2/G5 failure could only ever have been `PROVISIONAL_REJECT_PENDING_FIBER`,
because the MMP proxy's bias runs one way. With I-A the failure direction
becomes conclusive too, so the gate can actually reject a pair rather than defer
it. A gate that can only pass is not a gate.

**Alternatives rejected.** Keeping the MMP proxy alone (cheaper, but leaves
every rejection provisional); using the purely combinatorial
`enumerate_action_fiber` (78 successors under `editing_v2_semantic` — a
different and much narrower support than the model-gated one, so it would have
censused a set the controller does not actually see).

**Changes a frozen object:** no. Changes this lane's census instrument, before
the census ran.

---

## D-003 · 2026-08-13 · Objective-pair order predeclared and committed BEFORE the census ran

**Decision.** The order is (1) potency vs developability, (2) potency vs source
similarity, (3) developability vs source similarity. The first pair passing all
five gates is adopted. No reordering after results.

**Evidence before the decision.** Only the mission contract and the retargeting
design document. Explicitly **not** consulted: any controller performance, any
census output.

**Alternatives rejected.** Choosing the pair after seeing which one makes the
controller look best — the failure mode `RETARGETING_SAME_PREFIX_DESIGN.md`
names as "engineered".

**Changes a frozen object:** no. Commit: this one, before the census commit.

---

## D-004 · 2026-08-13 · The unclipped-developability escape hatch is closed in advance

**Decision.** `z_D` uses the frozen clipped soft-min (`MARGIN_CLIP = 1.5`). If
O-D fails the saturation gate G4, this lane may **not** substitute an unclipped
soft-min or QED alone.

**Evidence before the decision.** The committed calibration
`retarget_calibration_result_3plus3_fixed.json` reports developability
`{at_switch: 5, greedy: 28, verified: 28}` of 30 sources in **three** edits.
That number was visible before the gate threshold was written, so G4's 0.85
ceiling is *not* a post-hoc threshold — it is a threshold chosen knowing the
committed evidence and stated anyway, which is the honest ordering: the reader
can see that O-D was expected to be at risk on G4 and that the threshold was
not moved to rescue it.

**Alternatives rejected.** Silently widening the objective until pair 1 passes.
That is exactly how "29/30 vs 28/30" became a reported result instead of a
recognised ceiling.

**Changes a frozen object:** no — it forbids a change.

---

## D-005 · 2026-08-13 · `gen_rank` is instantiated at two budget levels

**Decision.** `gen_rank@greedy` and `gen_rank@verified` are separate arm
instances.

**Evidence before the decision.** Observed cost from the committed calibration:
26 kernel calls per source for a greedy-style 3+3 protocol with an 8-candidate
lookahead, at ~22 s per kernel call. A verified 6-edit, 5-preference protocol
projects to ~600 kernel calls per source versus ~26 for greedy. One `gen_rank`
instance cannot hold native-oracle-call parity against two arms that differ ~23x
in cost.

**Alternatives rejected.** A single `gen_rank` at one budget — it would make
either P3 or P4 vary controller *and* budget, the exact parity defect the main
lane's audit caught.

**Changes a frozen object:** no.

---

## D-007 · 2026-08-13 · A G4 statistic of MY OWN had no falsifying range; withdrawn before any pair verdict

**Decision.** Withdraw `best candidate at a state reaches the pooled p99` as the
G4 saturation statistic and replace it with **single-objective greedy rollouts
at the full 6-edit budget**.

**Evidence before the decision.** The first census smoke printed
`G4 reach fractions {'P': 0.133, 'D': 0.933, 'S': 1.0}`. The `S: 1.0` prompted
the project-wide question — *what value could this take if the hypothesis were
false?* — and the answer is: essentially none. With a fiber of width `n`,
`P(max of n draws >= pooled p99) = 1 - 0.99^n`, which is **0.9973 at n = 589**.
The measured fiber width was 589–613. The statistic was going to read ~1.0
whatever the chemistry did.

This is defect shape number six of the same family, and the first one this
project caught in its own instrument *before* the run rather than after.

**The replacement, and why it is two-sided.** Reach fraction is now the fraction
of held-in sources whose 6-edit single-objective greedy rollout attains the
objective's ceiling, where the ceiling is analytic for O-D (the clipped soft-min
cannot exceed `1.326713`) and the held-in pool p99 for O-P. Both can plainly come
out anywhere in [0, 1]. O-S has no reachable ceiling at all — `T = 1` requires
zero edits — so it is tested for **inertness** instead: an objective that six
real edits cannot move is not controllable and degenerates the pair. Inertness
can equally well come out false.

**Alternatives rejected.** Keeping the statistic and caveating it in prose. The
project has five recorded instances of exactly that not working.

**Changes a frozen object:** yes — it changes the G4 operationalization declared
in `PROTOCOL.md` §6.3/§6.4. Timing matters and is verifiable in the history: the
change was made when only the *reach fractions* had been seen, before any
per-pair gate verdict was computed or printed. The G4 **threshold** (0.85) was
not touched.

**Consequence worth recording.** Under the corrected statistic, developability
does **not** saturate as a continuous Pareto axis, even though the committed
retargeting calibration reports greedy reaching the binary developability region
on 28/30 sources in three edits. Reaching the region is not the same as
exhausting the axis: the region boundary sits far below the clip ceiling. The
old binary reading would have rejected pair 1 for the wrong reason.

---

## D-008 · 2026-08-13 · The executable gate rejected MY OWN parity table

**Decision.** Demote contrast P1 (`greedy_pref` vs `unguided`) from PRIMARY to
`CONTEXT_ONLY`, and record in `PROTOCOL.md` §8 that it varies **controller and
objective**.

**Evidence before the decision.** `scripts/pareto_instrument_gate.py`, run as a
design-stage self-test with no results in existence, failed check
`D5_contrast_parity` and printed
`P1: varies ["controller", "objective"], intended "objective"`. The protocol
table had claimed P1 varied only the controller. It was wrong: an arm with no
controller cannot have an objective either, so the two are inseparable in that
contrast.

This is the same shape as the confound the main lane's audit found in
`verified_retarget` vs `continue_A`, which inflated its effect by roughly 11%.

**Alternatives rejected.** Deleting P1 (the unguided floor is genuinely
informative); redefining `unguided` to carry an objective (it would stop being
unguided). The parity-clean forms of the same question already exist as P3
(closed vs open loop, same objective) and P5 (objective only), so nothing is
lost by demoting P1.

**Changes a frozen object:** yes — `PROTOCOL.md` §8. Before any run.

---

## D-009 · 2026-08-13 · The two budget axes differ ~600x, so P3/P4 are reported as a bracket

**Decision.** Run `gen_rank` at **both** matchings — kernel-matched and
native-oracle-matched — and report P3/P4 as a bracket rather than a number.

**Evidence before the decision.** One kernel call yields ~600 candidate
molecules (measured: fiber width 589–613). So matching `gen_rank` on native
oracle calls hands it ~600x the closed-loop arms' kernel budget, which is the
hypervolume inflation channel wearing a benchmark convention's clothes; matching
on kernel calls gives it far fewer distinct molecules than the closed-loop arms
see, which understates it on the axis the multi-objective literature budgets.

**Alternatives rejected.** Picking one axis and calling it fair. Whichever is
chosen decides the winner, which makes the choice a result rather than a method.

**Changes a frozen object:** extends `PROTOCOL.md` §7.2. Before any run.

---

## D-010 · 2026-08-13 · The gate forbade the experiment; the gate was wrong, not the experiment

**Decision.** Split D6 into two checks. **Equal endpoint counts** are required
on every hypervolume contrast. **Compute parity** is enforced only on contrasts
that claim it (P3, P4) and is *reported* elsewhere (P2), with the flag
`COMPUTE_ASYMMETRIC_BY_DESIGN_REPORT_THE_RATIO`.

**Evidence before the decision.** Running the analysis pipeline end to end on
synthetic shards, the gate failed `D6_hv_budget_matched` on **P2** — with
`verified_pref` at ~300 kernel calls against `greedy_pref` at ~26, a ratio of
11.5. Under the original rule, verified-vs-greedy could never be reported at
all.

**Why the gate was wrong.** Compute is not one of the four parity dimensions.
`budget` in the parity table means the **edit** budget `H = 6`, which P2 holds
exactly. A lookahead controller intrinsically spends more compute than a myopic
one — that *is* the mechanism — and throttling it to greedy's compute would
delete what is being tested. The hypervolume inflation risk P2 actually carries
is that the more expensive arm contributes more *points*, and that is closed by
the equal-endpoint-count rule, which still applies.

**Alternatives rejected.** Dropping P2 (it is the future-awareness question, the
whole reason the verified arm exists); throttling `verified_pref` (deletes the
mechanism); leaving the gate failing and overriding it by hand (an override that
becomes routine is not a gate).

**Changes a frozen object:** yes — `PROTOCOL.md` §7.3 and §9. Before any run.
Recorded rather than silently corrected, because **a check that forbids the
experiment is as wrong as one that permits a confound**, and only the log shows
which kind of error was made.

---

## D-011 · 2026-08-13 · The same mistake again, one level down: D6 is now axis-aware

**Decision.** `budget_parity_claimed: bool` becomes `budget_parity_axis: "kernel"
| "native" | None`. Only the claimed axis is enforced; the unclaimed one is
reported. P3/P4 claim `kernel`.

**Evidence before the decision.** Working through `generate_then_rank`'s ledger:
a kernel-matched `gen_rank` runs ~4 trajectories and scores ~4 endpoints, while
`greedy_pref` scores ~15,600 candidates. Under D-010's rule P3 claimed compute
parity outright, so the gate would have failed it on the **native** axis — and
a native-matched `gen_rank` would fail on the **kernel** axis. Both ends of the
declared bracket would have been rejected, leaving P3/P4 unreportable.

**Why this is the same error as D-010.** D-010 fixed a gate that forbade P2 by
conflating compute with the edit budget. The fix conflated something else: that
a contrast claiming compute parity claims it on *both* axes. With ~600
candidates per kernel call the two axes are not simultaneously satisfiable, so
demanding both is demanding the impossible. A gate that cannot be satisfied is
indistinguishable from a gate that is never run.

**Alternatives rejected.** Loosening the tolerances until both axes pass (that
is tuning the check to fit the data, at the level of the instrument); dropping
D6 for P3/P4 entirely (it still has to enforce equal endpoint counts, which is
the actual inflation control).

**Cost consequence, recorded for main.** Only the **kernel-matched** end of the
bracket is affordable in the smoke. The native-matched end needs ~2,600 unguided
trajectories per source, ~15,600 kernel calls, ~30 h per source. It is costed
and deferred, not quietly dropped — and until it is run, P3/P4 are one-sided.

**Changes a frozen object:** yes — `PROTOCOL.md` §7.3 and §9. Before any run.

---

## D-012 · 2026-08-13 · Census verdict — pair 1 adopted, and it is the FIRST that passed

*(Recorded AFTER the census ran. `diagnostics/pareto_tradeoff_census.json`,
sha256 `5b1fc96b4ceb3a75…`, 60 held-in sources / 120 real-fiber decision states /
mean fiber 586, plus 81,500 MMP pairs. 4,515 s local, CPU only, no Modal.)*

Deliberately a separate commit from D-001..D-005, so the git history shows the
predeclaration preceded the measurement rather than merely claiming it: the gate
thresholds and the pair order landed at `d206d55`, before
`scripts/pareto_tradeoff_census.py` had ever been run.

**Adopted: `potency_vs_developability`.** All five gates pass. It is the first
pair in the predeclared order, so the positional rule and the outcome agree —
which is worth stating precisely *because* they agree: had pair 2 or 3 scored
better on some statistic, the rule would still have taken pair 1.

| gate | value | threshold | verdict |
|---|---:|---|---|
| G1 alignment | rho **−0.274** | `< +0.70` | PASS |
| G2 local tradeoff | tradeoff moves **0.512**; both directions available at **100%** of states | `>= 0.20`, `>= 0.25` | PASS, **both instruments agree** |
| G3 no domination | binding share **0.76 / 0.24** | `<= 0.90` | PASS |
| G4 no saturation | P **0.700**, D **0.000** | `<= 0.85` | PASS |
| G5 front richness | **2.34** of 5 distinct, **10%** unanimous | `>= 2.0`, `<= 0.50` | PASS, **`OPERATOR_SET_DEPENDENT`** |

### Three things a reader should not miss

**1. G5 carries a caveat, and the paper must carry it too.** I-A passes (2.34 of
5 distinct Chebyshev selections) but I-B fails (1.66). Per §6.2 that is
`OPERATOR_SET_DEPENDENT`: preference distinguishability is present in the
*executable support* and not visible in real one-cut analogue pairs. Both
instruments were declared in advance precisely so this could be seen rather than
assumed, and the direction is the expected one — I-B's median neighbourhood is
degree 1 against I-A's 586.

**2. D-007's correction changed the verdict, and the committed shards show it.**
The same artifact reports developability reaching the *binary region* on
**0.933** of committed 3-edit rollouts but the *clipped continuous ceiling* on
**0.000** — median endpoint 0.627 against a ceiling of 1.327. Reaching the
region is not exhausting the axis. Under the withdrawn statistic, or under the
binary reading, pair 1 would have been rejected for the wrong reason.

**3. Potency has real but limited headroom, and this is the live risk.** P's
reach fraction is **0.700** — 14 of 20 sources reach the held-in p99 within six
greedy edits, against a 0.85 rejection threshold. It passed, but not
comfortably. If the smoke shows arms clustering at the top of the potency axis,
that is this number materializing, not a surprise.

### Why pairs 2 and 3 failed, mechanically

Both fail G4 because **ECFP4 source similarity is locally non-discriminative**:
legal graph edits frequently leave the fingerprint unchanged, so across 20
sources the similarity-maximizing 6-edit rollout ended at ECFP4 Tanimoto
**exactly 1.000** to the source, every time. This is an **interaction between
the executor support and the ECFP4 representation** — not a claim that source
preservation is costless in general, and not a claim about this chemistry
independent of how similarity is measured.

The same fact explains their G2 failures. The source *starts* at maximal
similarity, so a move can only hold or reduce it — the "other objective down,
similarity **up**" quadrant is nearly empty: **0.005** of moves for P-vs-S and
**0.006** for D-vs-S, against a `>= 0.05` floor. A Pareto axis you cannot
improve is not an axis.

Pair 2 additionally fails G3 at **0.907**: potency is the binding Chebyshev term
in 90.7% of decisions, just over the 0.90 line. That is the asymmetry
`RETARGETING_SAME_PREFIX_DESIGN.md` already documented — potency is 4–5x harder
in normalized units — showing up as goal domination, exactly what G3 exists to
catch.

**No fallback pair was needed, and none was invented.** The predeclared order
was not consulted after the fact.

> **What a reviewer should take from this, stated plainly.** The similarity-inert
> finding means the predeclared order effectively had **one viable entry** — both
> fallback pairs are killed by construction, not by a marginal statistic. That is
> a discovery, not a convenience. It was not knowable before the census: the
> obvious prior is that "stay close to the starting molecule" trades against any
> potency or property objective, and under this executor-plus-fingerprint pairing
> it does not. Adopting the first pair in the order is therefore the *only*
> outcome the gate could have produced, and the honest way to present it is that
> we measured why, not that we got lucky with the ordering.
>
> **No search for a better similarity metric will be undertaken in this lane.**
> The fallback order was preregistered and potency/developability passed
> strongly; hunting for a metric that manufactures a second viable pair is the
> post-hoc tuning the preregistration exists to prevent.

---

## D-013 · 2026-08-13 · I-A is the instrument of record; I-B is corroborative only

**Decision.** State explicitly, in `PROTOCOL.md` and `HANDOFF.md`, that the real
model-gated successor fiber (**I-A**) is the instrument of record for this
census, and that one-cut matched pairs (**I-B**) corroborate but never decide.

**Evidence, from the completed census.** On the adopted pair the two instruments
disagree by a wide margin on exactly the statistic that matters most:

| | I-A (real fiber) | I-B (MMP pairs) |
|---|---:|---:|
| decision states | 120 | 5,592 |
| moves scored | 70,286 | 22,865 |
| **mean distinct selections of 5** | **2.342** | **1.656** |
| **unanimous states** | **0.100** | **0.412** |
| mean front size | 5.90 | 2.15 |
| distinct front points, 101-weight grid | 4.77 | 2.07 |
| tradeoff-move fraction | 0.512 | 0.535 |

**A proxy-only census would have read borderline where the executable support is
comfortable.** I-B's 1.656 sits *below* the predeclared G5 floor of 2.0 and its
41.2% unanimous rate approaches the 50% ceiling; I-A clears both. Note that the
two agree closely on the *sign* statistics — tradeoff-move fraction 0.512 vs
0.535 — and diverge on the *selection* statistics, which is exactly what a
neighbourhood of median degree 1 versus 586 predicts.

**This project has already been misled by a matched-pair proxy once.** The
movability census read single-edit DRD2 gains off 972 MMP pairs and implied the
potency threshold was out of reach in four edits; the real fiber then climbed
**+4.42 log-odds in three**. `RETARGETING_SAME_PREFIX_DESIGN.md` records that
the matched-pair pessimism was wrong and the position-independent best-of-N
reading was right. The same asymmetry appears here on a different statistic, in
the same direction.

**Consequence for the paper.** G5 on the adopted pair is flagged
`OPERATOR_SET_DEPENDENT`: preference distinguishability is a property of the
executable support and is *not* visible in real one-cut analogue pairs. That
caveat belongs in the text, not in a footnote.

**Alternatives rejected.** Dropping I-B as uninformative — its disagreement is
itself the finding, and running both instruments was declared in advance
precisely so that a disagreement could be seen rather than assumed.

**Changes a frozen object:** no. It makes §6.2's existing precedence rule
explicit rather than implicit.

---

## D-014 · 2026-08-13 · Held-in smoke AUTHORIZED by main and launched; shortlist stays at K=8

**Decision.** Launch the 12-source held-in smoke. **Shortlist K = 8 unchanged**
(4 immediate / 2 reference / 2 random).

**Authorization.** Main lane, after the 60-source census passed. The ~$18 was
authorized explicitly; the ~$9 shortlist-4 saving was declined on the grounds
that it is a cost consideration and not a scientific one, and that deliberately
weakening the controller is the wrong move immediately after establishing that
the task has genuine geometry. K was frozen in `PROTOCOL.md` §7.2, so it stays
frozen — this lane did not get to choose either way, which is the point of
freezing it.

**Launch, verified rather than assumed.**

| | |
|---|---|
| app | `compose-v4-pareto-control` |
| app id | `ap-YJkZtmnhwS8i7RTWNq0Br9` |
| **state** | **`ephemeral (detached)`** — confirmed in `modal app list` |
| tasks | 13 (12 sources + 1 on-Modal driver) |
| launched | 2026-08-13 02:20:53 EDT |
| command | `modal run --detach modal_apps/pareto_control_app.py --sources 12` |
| profile | `rahul-94866`, CPU only |
| cohort | `adea8e5510852d69`, 12 held-in sources |
| pair | `potency_vs_developability` |

**No client-side timeout wrapper**, deliberately: the census's single-write
defect and two prior `--detach` losses to client-side DNS failures are the
reason. Durability here is structural — one durable shard per source, a
checkpoint committed after **every arm**, a driver that skips sources whose
shard already exists, and the fan-out running on Modal so a client disconnect
cannot stall it.

**What is NOT in this run.** No `h_phi`. No ablation arms. No held-out panel. No
development panel. The frozen protocol's five arms plus the P6 prefix-branching
mode, and nothing else.

**Changes a frozen object:** no.

---

## D-015 · 2026-08-13 · This lane is now the definitive host of the R_theta ablation

**Decision.** Record that the registered `R_theta`-versus-empirical-family
ablation is hosted **here**, and that it is staged **after** Pareto control is
validated — not folded into this smoke.

**Evidence before the decision.** The preregistration made Pareto the ablation
host *conditional on the geometry gate passing*. It passed: 2.342 of 5 distinct
selections across 120 real-fiber decision states, 10% unanimous, a front of 5.90
nondominated candidates spanning 4.77 distinct weight-selected points.

**Binding consequence.** The host **may not be moved later because some
conventional benchmark makes `R_theta` look better.** That is the whole function
of a conditional preregistration: the condition was measured, so the host is
settled by the measurement rather than by which venue flatters the result.

**Staging.** Validate preference control first on the 12-source smoke, then run
the registered ablation. Arms are not multiplied in this smoke, because the
frozen protocol does not require them here.

**Changes a frozen object:** no — it discharges a conditional one.

---

## D-016 · 2026-08-13 · The oracle-demand ratio is WITHDRAWN as a cost claim; the counter is sound

**Decision.** Withdraw the oracle-demand ratio as a **cost** claim. Report raw
counts with the caveat attached. `diagnostics/pareto_oracle_accounting_audit.json`,
run on committed shards with **no new kernel work**.

**What was wrong with the claim, not the counts.** At matched kernel budget,
preference control is charged for **interrogating the whole legal successor
fiber at every decision**; generate-and-rank is charged only for **the terminal
molecules it produced**. Those are different acts. The ratio conflated
*property evaluations per unit of kernel work* with *cost of optimizing a
molecule*, and only the first was measured. I compounded it by reporting a
round "~3,000x" as though it characterised efficiency.

**The counter is sound — this was the thing to establish.**

| arm | requests | unique evaluations | cache hit | kernel | eval/kernel |
|---|---:|---:|---:|---:|---:|
| `greedy_pref` | 17,726 | 10,352 | 0.476 | 16 | **596.6** |
| `verified_pref` | 273,854 | 174,294 | 0.364 | 398 | **437.9** |
| `gen_rank@greedy` | 6 | 2 | 0.500 | 11 | **0.2** |
| `unguided` | 5 | 5 | 0.000 | 20 | 0.2 |

`greedy_pref`'s **596.6** evaluations per enumeration sits essentially on the
census's mean distinct fiber width of **586**. The increments track distinct
candidate molecules; there is no counter bug. `verified_pref`'s 437.9 is lower
because lookahead rollouts revisit states whose candidates are already cached.

### The four separations

1. **Batch or invocation? SEPARATE INVOCATIONS.** `objective_vector()` calls
   `oracle.margin_many([key])` with a list of **exactly one** molecule. Every
   evaluation is its own scorer invocation, so the counter is **not** hiding
   vectorisation. The counts are honest — and the implementation is leaving
   batching on the table, at ~10k one-molecule SVM calls per source for
   `greedy_pref` alone.
2. **DRD2 versus cheap descriptors? NOT SEPARABLE AS INSTRUMENTED.**
   `objective_vector()` computes DRD2, QED and cLogP together on every
   evaluation, so `N_drd2 == N_descriptor == N_all_objective` **by
   construction**. Today the biologically meaningful count happens to equal the
   total — but only because nothing short-circuits. Any future version that
   skipped DRD2 when developability binds would make them diverge while the
   counter kept reporting one number. **Separating the increment sites is a
   prerequisite for any cost claim.**
3. **Requests versus post-cache? SEPARABLE AND REPORTED.** Roughly half of
   `greedy_pref`'s requests resolve from cache (17,726 → 10,352), so the two
   tell materially different efficiency stories, exactly as Lane 3 found on real
   baselines.
4. **Marks versus distinct successors? PARTIALLY ANSWERED.**
   `canonical_successor_result()` returns `batch.successors` **after** alias
   collapse, so the census's 586 is already a distinct-canonical-successor
   count, not a marked-action count. The pre-collapse marked-action count is
   **not recorded anywhere**, and establishing the collapse ratio needs one
   instrumented enumeration — new kernel work, **not done here**.

### The finding this exposes, stated as a weakness

Exhaustive preference control **genuinely scores essentially every distinct
legal successor at every edit** — 596.6 of a 586-wide fiber. Against a real
black-box oracle that is a **genuine computational weakness of the exhaustive
implementation**, not an artifact of accounting.

It is also a conspicuous place where COMPOSE's own structure should help: it is
strange for the scalable algorithm to score all ~600 legal moves equally when a
learned reference distribution is already available, and goal-aware
shortlisting is the same shape that already worked in exact-target recovery.
**That follow-up is NOT authorised, is not designed here, and does not touch
this smoke.** It is recorded so the observation is not lost.

**Changes a frozen object:** no. Withdraws an interpretation, changes no metric,
no threshold and no arm.

---

## D-017 · 2026-08-13 · Post-smoke plan recorded BEFORE results; nothing authorised

**Decision.** Commit `POST_SMOKE_PLAN.md` while **1 of 12 sources** had finished
and no aggregate existed. Implement none of it. Touch nothing in the running
experiment.

**Why now rather than after.** The decision tree in §5 says what happens if
Pareto control turns out weak. Writing that down *after* seeing the numbers
would be worthless — the value is entirely in it being fixed first. Everything
else in the document rides along for the same reason.

**The four items.**

1. **Batch the scorer.** `oracle.margin_many([key])` on a one-element list is an
   implementation defect. Batching is an **exact vectorisation**, gated by six
   conditions: identical candidate set, identical potency scores to tolerance,
   identical selected action, identical complete trajectory, identical `N_drd2`,
   and a lower batch count. **If any of the first five differ it is a change of
   computation wearing a performance label and it does not land.**
2. **Split the counters** into `N_drd2_requests`, `N_drd2_unique`,
   `N_drd2_evaluator_calls`, `N_descriptor_requests`, `N_scorer_batches` — even
   though several coincide today. It makes future accounting *incapable* of
   conflating them.
3. **Do not measure the pre-alias mark count.** ~586 distinct canonical
   successors is the quantity relevant to expensive scoring; the collapse ratio
   is a kernel-efficiency curiosity and does not change the oracle-demand
   conclusion. **Closed, not deferred** — no kernel instrumentation will be added
   for a number we do not need.
4. **Do not headline the raw ratio.** "Exhaustive control uses vastly more
   objective information at matched kernel computation" is correct. *"COMPOSE
   needs 3,000x more oracle evaluations for the same Pareto quality"* is not the
   same statement, and **we have not seen Pareto quality yet.**

**The completed smoke stays reconstructible.** Nothing short-circuited in the
code that produced these shards — `objective_vector()` computed DRD2, QED and
cLogP together on every evaluation, unconditionally — so
`N_drd2 = N_descriptor = N_all_objective = native_oracle_calls` exactly, and the
split counters can be back-filled from the existing artifact without rerunning
anything. That is why item 2 can be a pure instrumentation change rather than a
re-run.

**Why the counters are NOT split today.** Editing `pareto_control.py` now would
break the provenance between the shards and the code that produced them — the
same reason the census script was not fixed mid-run. Instrumentation lands after
the artifact is closed.

**The finding that survives the fix.** Even perfectly batched, the algorithm
still requests ~586 x (13-23) distinct potency evaluations per source. If
potency were a docking workflow or a wet-lab assay, batching would save
**nothing conceptually**. Exhaustive full-fiber preference control is genuinely
objective-query-intensive, and that is an algorithmic property rather than an
implementation artifact.

**Binding, and worth naming because it is the tempting move.** **NO K-SEARCH.**
Running `K in {4, 8, 16, 32, 64, 128}` and reporting the attractive point is a
selection step hidden in the write-up. One operating point is pre-registered from
a desired reduction in oracle demand, and the report gives fraction of HV
retained against fraction of potency evaluations retained.

**Changes a frozen object:** no. Nothing implemented, nothing authorised, no
metric, threshold or arm touched.

