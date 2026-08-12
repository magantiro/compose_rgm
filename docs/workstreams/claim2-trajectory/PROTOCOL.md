# Claim 2 — trajectory characterization — PROTOCOL

**Status:** `DESIGN_ONLY`. Frozen before any trajectory exists.
Machine-readable twin: `configs/claim2_trajectory_protocol_v1.json`.

---

## Claim

> The executor defines what *can* happen; `R_theta` organizes those legal edits
> into coherent molecular trajectories that achieve nontrivial structural
> movement without the pathological drift, cycling, or operator collapse
> induced by unlearned same-support transport laws.

## Non-claims

- Not physical dynamics. Not synthetic route plausibility. These are **design
  dynamics** over executable edits.
- Not medicinal plausibility. The descriptor envelope is a distributional
  sanity check.
- Not "fewer reversals is better". Reversibility may be part of the reference
  law. The claim is about the **absence of pathological domination**.
- Not a distribution-learning result. This is source-conditioned transport.
- Not another likelihood experiment. Experiment 1 already established that
  `R_theta` beats uniform legal rewriting (1.46 nats) and the empirical-family
  law. This asks what that preference **produces when iterated**.

---

## Frozen inputs

| Object | Path | Identity |
|---|---|---|
| `R_theta` | `/artifacts/editing_v2/r_theta_run/runs/run_v2_01/R_THETA_CHECKPOINT.pt` | `diagnostics/editing_v2_r_theta_FROZEN.json`, run `run_v2_01`, step 12,500, identity `426ac15e7f4cc68c…`, state `c977ee3fe0cfdcaf…` |
| Empirical family law | `diagnostics/editing_v2_sampling_law_v2.json` | `realized_coefficients.by_family`, frozen sha `b0cc66f168f1cf3e…` |
| Process identity | `configs/editing_v2_process_v2_capability_cells.json` | `0c938177a34819e6…`, 8 active families, 22 registry cells, max 40 active atoms |
| Split | `diagnostics/editing_v2_matched_validation_reserve_ids.json.gz` | 96,094 held-in / 10,653 reserve sources |
| Descriptor envelope | `diagnostics/claim2_descriptor_envelope.json` | `fb369161b303a49c…`, calibrated on all 96,094 held-in molecules |
| Development panel | `diagnostics/claim2_trajectory_development_panel.json` | `a4282229742eff2e…`, 36 held-in sources |

Nothing in this lane retrains, fine-tunes, reselects or edits any of these.

---

## Arms — one support, three laws

Comparison type is `law_only` in the sense of
`successor_kernel.assert_arms_comparable`: **identical executable support at
every state, only the probability assignment differs.**

| Arm | Law |
|---|---|
| `r_theta` | `P_theta(y \| x)`, conditioned on taking a productive jump |
| `uniform_canonical` | `1 / \|N+(x)\|` over distinct successors after alias collapse |
| `empirical_family` | `sum_{k reaching y} qhat(k) / \|N_k(x)\|` |

`qhat` is the frozen realized training family law, restricted to families with a
legal productive successor at `x` and renormalized there. That is the **strong**
form of the unlearned baseline: it gets the correct global operator
frequencies, the exact legal support, state-dependent family availability and
alias aggregation. What it does not get is learned molecular context. The
definition is not invented here — it is the frozen definition in
`configs/comparator_registry_v3.json` and the one Experiment 1 used, reused so
the two experiments cannot silently disagree about what the baseline is.

Two details that would quietly break the comparison if got wrong, and are
tested:

- Uniform is over **distinct successors**, not over marks. Uniform-over-marks
  would reward whichever molecule has the most syntactic encodings — a fact
  about the action codec, not about chemistry.
- The empirical-family law **sums** across families reaching one successor.
  Taking a single "primary" family instead yields a sub-probability and biases
  every downstream statistic. The sum is what makes it normalized:
  `sum_y sum_{k reaching y} qhat(k)/|N_k(x)| = sum_k qhat(k) = 1`.

**The optional fourth arm** (empirical family + learned identity) is **not
implemented**. It needs a within-family score the trajectory row does not
carry, and the workstream permits it only if already cheap.

---

## Panels

**Development (held-in, opened).** 36 sources, 4 scaffold-support bands × 3
heavy-atom size bands × 3 sources. Selection is mechanical and
outcome-independent: no property, no rollout, nothing tunable after a result.

Heavy atoms are restricted to **[12, 34]**. The upper bound is load-bearing:
the frozen process caps active atoms at 40, so a 38-atom source would hit the
ceiling inside six edits and its suppressed growth would be indistinguishable
from a learned preference against growing. Six edits of headroom removes that
confound from the mobility axis.

Support band = number of distinct retained held-in training source molecules
sharing the RDKit Murcko scaffold, banded `0 / 1–4 / 5–24 / 25+` — recomputed
from the committed reserve-ids file rather than copied, so it cannot drift from
the matched-validation carve. A held-in molecule does not support itself.

Acyclic molecules have an **empty** Murcko scaffold and form one explicit
class, so they concentrate in the densest band. The panel records ring status
per source so that confound is visible; 35 of 36 development sources contain a
ring.

**Confirmatory (matched reserve, NOT opened).** Definition only:

```bash
python scripts/claim2_select_trajectory_panel.py --pool reserve --count 96 \
  --seed 20260812 --out diagnostics/claim2_trajectory_confirmatory_panel.json \
  --i-am-authorized-to-open-the-matched-reserve '<authorizing agent>'
```

96 sources, 8 per (support band × size band). Source-disjoint from:

| Prior consumer | Evidence | Reserve sources burned |
|---|---|---|
| exact-target development + sealed67 | `diagnostics/editing_v2_controller_panel_seal.json`, **both source and target endpoints** of all 91 pairs | 164 |
| planning-signal probe | `diagnostics/editing_v2_experiment_c0_planning_signal.json` | 2 |
| retargeting sources | `diagnostics/retarget_calibration_cohort.json` declares held-in only | 0 |
| pathwise-constraint sources | `configs/benchmarks/cnof_leads.json`, disjoint benchmark lineage | 0 |
| `h_phi` teacher data | `diagnostics/editing_v2_teacher_validation_carve.json`, held-in | 0 |
| **global final test** | `diagnostics/editing_v2_final_test_integrity_audit.json`: `exact_overlap.with_matched_reserve = 0` | disjoint by construction |

Fresh reserve pool after exclusions: **10,487** of 10,653. The burn is
band-skewed — 5.5% of band `25+` versus 0.7% of band `0`, because the existing
panel was mined for compilable multi-step paths.

Targets are excluded as well as sources: the sealed67 amendment dropped two
pairs precisely because a target had become another panel's source.

**The reserve path is gated in code, not by convention.** `--pool reserve`
fails at argument parsing without a written authorization string, which is then
recorded verbatim in the artifact. Materializing the panel is the act that
opens it.

---

## Rollout

- Horizon **H = 6** productive edits, identical for every arm.
- **3 seeds** per source (2 in the smoke). Seeds are repeated measures, not
  independent examples.
- Model time 0.5, canonical slots 48, matching every other COMPOSE experiment.
- Every accepted canonical successor is committed **as sampled**. No
  controller, objective, oracle or goal enters this lane.

**Common random numbers.** All three arms consume the identical uniform variate
at the identical step, over the identical sorted successor order. Where two
laws agree they take the same action, so a divergence in realized trajectories
is attributable to the laws rather than to the sampler. This is a declared
variance reduction, not a discovered one.

**H = 8** may be compared **once**, on held-in smoke only, and only if H = 6
fails to expose structural movement. It is not a tuning knob.

---

## Primary result — the mobility–fidelity frontier

No single metric defines "good trajectories". The result is a two-dimensional
Pareto statement with a decision rule fixed before any data:

- **Mobility** `M` — median over sources of the per-source mean endpoint ECFP4
  Tanimoto **distance** from the source.
- **Fidelity** `F` — mean over sources of the per-source fraction of committed
  intermediate states inside the held-in descriptor envelope.

`frontier_verdict` returns exactly one of:

| Verdict | Meaning |
|---|---|
| `dominates` | better or equal on both axes, strictly better and **resolved** on at least one |
| `dominated_by` | the same with the arms exchanged |
| `incomparable` | resolved movement in opposite directions — mobility bought with fidelity, or the reverse |
| `unresolved` | neither axis separated; the panel is underpowered |

**All four are real outcomes.** `incomparable` is publishable and must not be
scalarized away after seeing the point estimates. `dominated_by` is reachable
and is directly tested in `tests/test_claim2_trajectory_analysis.py`.

An axis contributes a direction **only** if the paired bootstrap 95% interval
over **sources** excludes zero. The source molecule is the independent
statistical unit; seeds are averaged within source first, or the interval is
computed against a denominator the experiment does not have.

**No verdict is emitted below 20 sources**, whatever the point estimates show.
A smoke measures cost and instrument sanity; it does not decide a claim.

### The envelope ceiling

Held-in self-retention of the envelope is **0.9639**, not 1.0. A coordinate-wise
[0.5%, 99.5%] box over nine descriptors necessarily excludes part of its own
calibration sample. That is the **ceiling for every arm**, and retention below
1.0 is not drift.

---

## Secondary metrics

Structural: heavy-atom change, ring-**system** change (fused rings sharing ≥ 2
atoms are one system — ring count alone cannot separate "grew a second isolated
ring" from "fused onto an existing one"), mean support size.

Health: unique-state fraction, immediate-reversal rate, two-cycle rate,
any-state revisit rate, early dead-end rate.

Operators: family entropy under fractional attribution (a successor reached by
several families contributes `1/|families|` to each — fixed before any result,
with raw family sets kept in every shard so an alternative attribution needs no
rerun), family coverage count, multi-family trajectory fraction, `family:table`
cell coverage.

Diversity: source-conditioned endpoint uniqueness across seeds.

Drift: standardized L2 descriptor displacement from source versus edit count.

All of the above are reported separately across support bands `0 / 1–4 / 5–24 /
25+`. The desired story is not "support does not matter"; it is an honest
characterization of how mobility and fidelity change as training support
weakens.

---

## Instrument gates

This project has been bitten three times by statistics whose sign was fixed
before any data existed: an action selected by `argmax V_G` then scored by
`V_G`; a "verified" arm that was secretly greedy; a sign test against a null
that policy improvement makes false. Four gates, all executable:

**1. The falsifiability gate.** Every metric in `METRIC_REGISTRY` must answer:
*what value could this take if the hypothesis were false?*
`validate_metric_registry` raises on any metric whose `falsifying_observation`
is empty, and the analysis script calls it before reading a shard. Prose in a
protocol does not stop a fixed-sign metric reaching a results table; an
import-time check does.

**2. The arm-divergence gate.** Pairwise total variation between the three laws
is recorded at **every** state. Where `|N+(x)| = 1` the arms coincide by
construction and the state carries no information about which law produced it.
A run whose arms never diverge is `INVALID_INSTRUMENT`, not a null result.

**3. The kernel cross-check gate.** `enumerate_successor_row` reaches the
learned law by a different production route than `canonical_successor_result`,
which every other COMPOSE experiment consumes. Each source verifies the two
against each other on its source state. Any disagreement marks the run
`INVALID_INSTRUMENT` and the launcher refuses to proceed.

**4. The power gate.** No Pareto verdict below 20 sources.

### Banned statistics

These may never be reported as evidence for Claim 2. Each is tautological on
this instrument:

| Statistic | Why it is not a measurement |
|---|---|
| `E_R[log R]` vs `E_U[log R]` | Gibbs' inequality. Sampling from `R_theta` then scoring with `R_theta` cannot come out the other way. |
| probability of the chosen successor | The successor was chosen **by** that probability. Selector and scorer are one object. |
| support size by arm | Identical across arms by construction. A difference is a bug, not a result. |
| validity rate by arm | 100% for every arm, imposed by the executor. Carries no information about what was learned. |

The list is carried in `BANNED_STATISTICS` and copied into every analysis
artifact, so a later reader cannot reintroduce one without contradicting the
output file.

---

## Stop rules

Stop and **report**, do not retune:

- all arms nearly identical after the single held-in horizon check;
- `R_theta` collapses to one operator family, or cycles pathologically;
- descriptor metrics reveal broad off-distribution drift;
- the suite cannot distinguish mobility from trivial growth or shrinkage —
  which is why heavy-atom and ring-system change are reported beside Tanimoto
  distance;
- kernel cross-check disagreement → `INVALID_INSTRUMENT`, fix before analysing.

A negative characterization is a real result.

---

## Allowed calibration

- The descriptor envelope, on held-in molecules, before any trajectory. **Done.**
- The development panel, mechanically, from held-in sources. **Done.**
- One held-in H = 6 vs H = 8 comparison, only if H = 6 fails to expose movement.

## Forbidden adaptations

- Retraining, fine-tuning or reselecting `R_theta`.
- Changing objective, architecture, corpus, operator set or process identity.
- Refitting the envelope after seeing trajectories.
- Adding, removing or reweighting an arm after seeing results.
- Introducing a scalarization of the two axes after seeing the estimates.
- Opening the matched reserve without written main-workstream authorization.
- Any use of the global final test.

---

## Cost

**Measured basis.** Per-enumeration cost from committed artifacts, restricted to
tasks with ≥ 20 kernel calls so container startup is amortized:

| Artifact | Tasks | Median s/call |
|---|---:|---:|
| `editing_v2_h_phi_teacher_train_labels.json` | 280 | 13.8 |
| `editing_v2_experiment_c0_planning_signal_preregistered.json` | 12 | 8.6 |
| `editing_v2_sealed67_result.json` | 27 | 14.8 |
| `retarget_calibration_result.json` | 30 | 20.6 |

Pooled median ≈ **14 s/call**, p90 ≈ **35 s/call**.

**This lane's overhead.** `enumerate_successor_row` runs
`compile_state_successor_map` *plus* `forward_compiled_successor_partitions` —
one mark-execution pass and **two** model forwards, versus
`canonical_successor_result`'s one pass and one forward. Assumed 1.2–1.5×. The
smoke **measures** it: every shard records `seconds_per_kernel_call`.

**Enumeration count.** The source state is enumerated once and shared by every
arm and seed; each trajectory then contributes at most one new state per step.
Worst case per source is `1 + 3·seeds·(H−1) + 1`, the last term being the
cross-check.

| Run | Sources | Seeds | Worst-case enum/source | Expected container-hours | Worst case |
|---|---:|---:|---:|---:|---:|
| **Smoke (next action)** | 8 | 2 | 32 | **1.5** | **3.8** |
| Full held-in development | 36 | 3 | 47 | 10.1 | 25 |
| Confirmatory matched reserve | 96 | 3 | 47 | 27 | 67 |

CPU only, 2 CPU per container, `max_containers=40`, no GPU anywhere in the app.
A per-task `kernel_budget` of 40 stops a pathological source from spending the
run; the launcher refuses to start if the worst case exceeds the budget, so
trajectories cannot be silently truncated on exactly the sources that need the
support most.

Only the smoke is being requested.

---

## Operational contract for the eventual run

Two failure modes the main lane hit today, folded in here so this lane cannot
repeat them.

**1. `modal run` uses a different interpreter, and it has no RDKit.** A
`@app.local_entrypoint()` must therefore never import RDKit or do molecule
work. This lane already has the right shape and it is now enforced by test:

- panel selection is a normal `python3` script
  (`scripts/claim2_select_trajectory_panel.py`) writing a committed artifact
  under `diagnostics/`;
- that artifact is mounted into the image with `add_local_file`;
- the entrypoint only reads JSON and builds task dicts.

`tests/test_claim2_rollout_resumability.py` asserts the entrypoint body and the
module scope reference neither `rdkit` nor `compose_v4`. This is better than
repairing the interpreter, because it makes the panel auditable and byte-identical
across reruns rather than recomputed at launch.

**2. Launch detached, and make a relaunch cheap.** Three independent layers,
because the first two are not sufficient:

- the server-side `drive()` fan-out stops `.map()` stalling when the client
  goes away — but it does **not** keep the app alive if the client dies;
- `modal run --detach` does, verified by `modal app list` showing
  `ephemeral (detached)` — but even that did not save the main lane's run
  through a client-side DNS failure *at launch*;
- so every task commits its own shard to the volume as it completes, and
  `drive()` skips sources whose committed shard matches the current task
  identity. A relaunch after an outage costs only the sources that had not
  finished.

Reuse is **exact or not at all**. A shard is skipped only if its source,
horizon, seed set, kernel budget, panel hash and frozen-family-law hash all
match, and only if it did not end with an exhausted budget. Reusing a shard
produced under a different panel or horizon would mix two measurements into one
table with nothing downstream able to detect it, which is worse than
recomputing. All eight rejection cases are unit-tested.
