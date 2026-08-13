# Claim 2 — trajectory characterization — DECISION LOG

Every material decision, the evidence available **before** it, what was
rejected, and whether it changes a frozen object. None of the decisions below
changes a frozen object; all are recorded on branch
`codex/compose-claim2-trajectory`.

---

## 2026-08-12 — Base the branch on the main lane head, not the assigned worktree

**Evidence before:** the assigned worktree sat at `1ac6f19`, **915 commits**
behind `codex/editing-v2-successor-fiber-fastpath` (`04f1c46`). The workstream
contract `docs/workstreams/PARALLEL_WORKSTREAMS_AND_HANDOFF.md` and
`docs/EXPERIMENT_PLAN.md` do not exist at `1ac6f19`. `1ac6f19` is a strict
ancestor of `04f1c46`.

**Decision:** create `codex/compose-claim2-trajectory` at `04f1c46`.

**Rejected:** working at `1ac6f19` and cherry-picking the contract. It would
have produced a lane whose frozen-input hashes could not be verified against
the main branch, failing handoff acceptance gate 3.

**Changes a frozen object:** no.

---

## 2026-08-12 — Reuse the existing baseline definitions rather than restate them

**Evidence before:** Experiment 1 is complete and its five arms
(`uniform_canonical`, `empirical_family`, `learned_family_uniform_id`,
`empirical_family_learned_id`, `r_theta`) are implemented inline in
`modal_apps/experiment1_reference_law_app.py:185-191`. The empirical-family
definition is separately frozen in `configs/comparator_registry_v3.json` under
`frozen_definitions.state_independent_empirical_family_prior` with
`frozen_before_seeing_its_results: true`. `UniformSuccessorKernel` already
exists in `successor_kernel.py`.

**Decision:** `empirical_family_law` implements the committed five-step frozen
definition verbatim, and a unit test reproduces it by hand from the frozen `q`
coefficients. Uniform is the same "distinct successors, not marks" rule as
`UniformSuccessorKernel`.

**Rejected:** writing a fresh empirical-family law for the trajectory setting.
Two definitions of one baseline across two experiments is how Experiment 1 and
Experiment 2 end up disagreeing about what "unlearned" means.

**Changes a frozen object:** no.

---

## 2026-08-12 — Compose two production functions instead of duplicating aggregation or paying twice

**Evidence before:** `canonical_successor_result` returns probabilities but
discards the mark-to-successor map, so it cannot say which operator families
reach a given molecule — exactly what the empirical-family arm needs.
`compile_state_successor_map` keeps that attribution but carries no model
score. Mark execution and canonicalization dominate the cost of the whole
experiment (measured 8.6–20.6 s per enumeration).

**Decision:** `enumerate_successor_row` calls `compile_state_successor_map`
(one mark-execution pass, family attribution retained) then
`forward_compiled_successor_partitions` (scores those static groups). One
execution pass, two model forwards.

**Rejected:**
- calling `canonical_successor_result` **and** `compile_state_successor_map` —
  doubles the dominant cost;
- copying the ~50-line aggregation out of `production_successor_kernel` — a
  second, private definition of the pushforward, which the module docstring of
  `successor_kernel.py` explicitly forbids;
- adding a field to `SuccessorKernelDiagnostics` — touches a module every other
  lane consumes, for one lane's convenience.

**Residual risk and its mitigation:** this reaches the learned law by a
different route than the kernel every other experiment uses. Every source
therefore cross-checks the two against each other on its source state
(`cross_check_against_production_kernel`), and any disagreement marks the run
`INVALID_INSTRUMENT` before analysis.

**Changes a frozen object:** no. Both functions are called, neither is edited.

---

## 2026-08-12 — Emit trajectories on Modal, compute every metric locally

**Evidence before:** canonical state keys **are** canonical SMILES
(`rewrite/kernel.py:100`), so every metric in the suite is computable from the
committed shard with RDKit alone. Enumeration is the only expensive step.

**Decision:** the Modal app emits state sequences, family sets, support sizes
and per-state arm divergences. `scripts/analyse_claim2_trajectories.py`
computes mobility, health, envelope retention and the frontier locally.

**Consequence:** a metric bug costs a local rerun, not a container-hour. The
whole shard → frontier path is exercised by 13 local tests on synthetic shards
built from real molecules.

**Changes a frozen object:** no.

---

## 2026-08-12 — Calibrate the descriptor envelope before anything else exists

**Evidence before:** the workstream contract requires calibration on held-in
molecules before confirmatory trajectories are examined. Nothing had been
rolled out.

**Decision:** fit the coordinate-wise [0.5%, 99.5%] box and the standardization
on **all 96,094** held-in sources, commit it with its hash, and only then build
anything that produces a trajectory.

**Measured, and load-bearing:** held-in **self**-retention is **0.9639**. A
nine-coordinate box at those quantiles necessarily excludes part of its own
calibration sample. That is the ceiling for every arm; without it, an arm
scoring 0.95 would look like drift.

**Rejected:** a multivariate ellipsoid or Mahalanobis envelope. Harder to
interpret, and the contract asks for simple reproducible descriptors.

**Changes a frozen object:** no. Creates one.

---

## 2026-08-12 — Cap panel sources at 34 heavy atoms

**Evidence before:** the frozen process scope caps active atoms at **40**
(`configs/editing_v2_process_v2_capability_cells.json`,
`scope.maximum_active_atoms`). The nearest precedent, the retarget calibration
cohort, used [18, 38].

**Decision:** [12, 34], leaving six edits of growth headroom below the ceiling.

**Reasoning:** at 38 atoms a six-edit trajectory can hit the process ceiling.
Growth suppressed by the ceiling is indistinguishable from a learned preference
against growing, and mobility is the primary axis. This removes the confound
rather than measuring through it.

**Rejected:** matching the retarget cohort's [18, 38] for consistency.
Consistency with a different experiment's constraint is not a reason to
contaminate this one's primary measurement.

**Changes a frozen object:** no.

---

## 2026-08-12 — Common random numbers across arms

**Evidence before:** the three arms share a support and a sorted key order at
every state.

**Decision:** one uniform variate stream per seed, consumed identically by all
three arms via inverse CDF.

**Reasoning:** where two laws agree they take the same action, so realized
trajectory differences are attributable to the laws rather than to the sampler.
Declared here rather than discovered in analysis.

**Changes a frozen object:** no.

---

## 2026-08-12 — Fractional family attribution, with raw sets retained

**Evidence before:** a canonical successor reachable through several operator
families has genuinely ambiguous attribution. Alias multiplicities are
routinely > 1.

**Decision:** each committed step contributes `1/|families|` to every family
reaching it. Every shard also stores the raw family set per step.

**Rejected:** picking a "primary" family by lexicographic order — arbitrary and
would silently bias family entropy.

**Consequence:** an alternative attribution rule can be evaluated after the
fact without a rerun.

**Changes a frozen object:** no.

---

## 2026-08-12 — Frontier verdict has four outcomes, and none below 20 sources

**Evidence before:** none. Fixed before any trajectory existed, deliberately.

**Decision:** `dominates` / `dominated_by` / `incomparable` / `unresolved`, with
an axis contributing a direction only if its paired bootstrap interval over
sources excludes zero; and no verdict at all below 20 sources.

**Reasoning:** this is the instrument-discipline requirement made concrete. Ask
of the headline statistic: *what value could it take if the hypothesis were
false?* Three of the four outcomes contradict the claim, and
`test_pipeline_detects_a_cycling_arm_through_the_health_metrics` demonstrates
the pipeline actually returning `dominated_by`/`incomparable` against
`R_theta`. `incomparable` is recorded as publishable so it cannot be resolved
later by inventing a scalarization.

The 20-source floor exists because the smoke is 8 sources and a point estimate
on 8 sources will otherwise be read as a direction.

**Changes a frozen object:** no.

---

## 2026-08-12 — Ban four statistics explicitly, in code

**Evidence before:** three prior defects in this project all had the same
shape — a statistic whose sign was fixed before any data existed.

**Decision:** `BANNED_STATISTICS` names four with the reason each is
unfalsifiable; `validate_metric_registry` refuses to admit any of them; the
list is copied into every analysis artifact.

The sharpest is `E_R[log R] >= E_U[log R]`: it is Gibbs' inequality, so
"`R_theta` trajectories are more likely under `R_theta`" is arithmetic, not
evidence. It would have been an easy and natural-looking headline.

**Changes a frozen object:** no.

---

## 2026-08-12 — Gate the held-out pool in code, not in prose

**Evidence before:** the brief forbids opening any held-out panel; the
workstream contract requires a source-disjoint confirmatory panel definition.

**Decision:** one selection script with `--pool {training,reserve}`. The reserve
path fails at argument parsing without a written authorization string, which is
then recorded verbatim in the artifact. The confirmatory panel is committed as
a **command and a rule**, never as a list of molecules.

**Reasoning:** materializing the panel is the act that opens it. A convention
in a document does not stop an accidental `--pool reserve`; a required argument
does. Both gate directions are unit-tested.

**Changes a frozen object:** no. The reserve has not been read.

---

## 2026-08-13 — Cycle attribution: the reversibility is INHERITED

**Evidence before:** the 8-source smoke showed `R_theta` revisit 0.281 and
reversal 0.300 against ~0.01 for empirical-family, with lower displacement.
Nothing about its cause.

**Decision:** freeze the decomposition and the reading thresholds first
(`5c81818`), then measure.

**Result:** the training corpus has a **mutual-edge fraction of 0.7333** over
144,870 distinct directed teacher transitions. The preregistered `inherited`
threshold was 0.33 and `model_specific` was 0.10. `R_theta` learned a locally
reversible reference process; it did not invent the reversibility.

**Why this was worth doing before touching the model:** the same measurement
supported opposite conclusions with opposite consequences, and the corpus
census separated them decisively. Acting on the 8-source signal without it
would have meant reopening a frozen model to fix faithful behaviour.

**Changes a frozen object:** no. It is the reason not to.

---

## 2026-08-13 — Report the controlled-arm comparison as context, not evidence

**Evidence before:** controlled arms showed ~0% cancellation against
`R_theta`'s 29–44%.

**Decision:** record it, and record beside it that **the comparison has a
near-fixed sign**. The controlled arms are deterministic argmax-utility
policies, monotone non-decreasing in utility, so an immediate two-cycle
requires an exact tie and is close to structurally excluded.

**Rejected:** presenting "goal control suppresses backtracking" as a clean
finding. It is largely definitional. The falsifiable comparison is `r_theta`
versus `empirical_family` — both stochastic samplers over identical support —
where 29.2% against 3.2% could have come out either way.

**Changes a frozen object:** no.

---

## 2026-08-13 — Net mobility is a diagnostic, never a verdict axis

**Evidence before:** at n=8, displacement per net edit favoured `R_theta`
0.213 against 0.120 and 0.131 — the most flattering number in the lane.

**Decision:** report it beside the frontier as `diagnostic_only_axes`; leave
the verdict on the two axes declared before any data existed.

**Vindicated at n=36:** the effect **did not replicate as resolved** —
+0.0194 [−0.0051, +0.0459], interval spanning zero. Had it been promoted into
the decision rule on the strength of the 8-source panel, the lane would have
reported a mobility win that the larger panel does not support.

**Changes a frozen object:** no. It preserves one.

---

## 2026-08-13 — A short client timeout cancelled a detached run

**Evidence before:** the main lane's warning that `--detach` did not survive a
client-side failure twice.

**What happened:** the 36-source launch was wrapped in `timeout 300`. When the
timeout killed the client, cancellation signals propagated to the workers
despite `--detach`, and the app stopped with 22 of 36 sources complete.

**Why it cost nothing:** the resumable driver skipped all 22 committed shards
on relaunch and started only 14 tasks. Roughly six container-hours of work was
preserved by a feature added before it was needed.

**Decision:** never wrap `modal run --detach` in a short client-side timeout;
launch it in the background and poll `modal app list` instead
(`scripts/claim2_wait_for_app.py`).

**Changes a frozen object:** no.

---

## 2026-08-12 — Adopt the main lane's two operational findings before launching

**Evidence before:** the main lane hit both failures today, in a live run.

**Finding 1 — `modal run` uses a different interpreter and it lacks RDKit**, so
a `@app.local_entrypoint()` doing molecule work dies at launch.

**Decision:** keep panel selection in a normal `python3` script writing a
committed artifact, mount it with `add_local_file`, and let the entrypoint only
read JSON. This lane already had that shape; it is now **enforced by test** —
`test_the_local_entrypoint_never_imports_rdkit` and
`test_module_scope_imports_stay_light_enough_for_modals_interpreter` fail if
`rdkit` or `compose_v4` appears at module scope or in the entrypoint body.

**Rejected:** installing RDKit into the launch interpreter. Computing the panel
at launch would make it a function of whatever machine launched, where the
committed artifact makes it auditable and byte-identical across reruns.

**Finding 2 — server-side fan-out and `--detach` are both insufficient alone.**
The `drive()` fan-out stops `.map()` stalling when the client goes away but does
not keep the app alive; `--detach` does, but did not save the main lane's run
through a client-side DNS failure *at launch*.

**Decision:** three layers. Fan-out, `--detach` (documented in the command and
in the launcher's own banner, verified via `modal app list` showing
`ephemeral (detached)`), and **per-source shard commits with a resumable
driver** that skips sources whose committed shard matches the current task.

**Reuse is exact or not at all.** A shard is skipped only if source, horizon,
seed set, kernel budget, panel hash and frozen-family-law hash all match, and
only if it did not end with an exhausted budget. Reusing a shard from a
different panel or horizon would mix two measurements into one table with
nothing downstream able to detect it — worse than recomputing. All eight
rejection cases are unit-tested in
`tests/test_claim2_rollout_resumability.py`.

**Changes a frozen object:** no.

---

## 2026-08-12 — Make `handoff.json` self-verifying after fabricating a digest

**Evidence before:** none — this was a defect found in my own work.

**What happened:** while writing `handoff.json` I recorded the envelope's
content hash as `fb369161b303a49c50f9cf9de7a83ec26…`. The first 16 hex
characters were real; **the rest was invented**. A spot check against
`diagnostics/claim2_descriptor_envelope.json` caught it.

**Why it matters more than a typo:** handoff acceptance gates 2 and 3 are
"`handoff.json` validates" and "all frozen-input hashes match". A wrong digest
is worse than a missing one, because it looks verified. Nothing downstream
depended on it — no run had occurred — but the same habit applied to a frozen
input would have produced a lane that appeared hash-bound and was not.

**Decision:** `tests/test_claim2_handoff_manifest.py` recomputes **every**
digest in the manifest from the file it describes, checks the inner content
hashes of the envelope and panel against their own artifacts, checks the
declared `R_theta` identity against the decision record, asserts the held-out
status is self-consistent, and asserts the smoke command matches the app's real
entrypoint with a budget covering the declared worst case.

**Rejected:** fixing the digit and moving on. The failure mode was
transcription, and transcription recurs.

**Changes a frozen object:** no.

---

## 2026-08-12 — Discrepancies recorded, not silently reconciled

1. **Capability cells: 22, not 19.** `docs/EXPERIMENT_PLAN.md` says "19
   capability cells"; `configs/editing_v2_process_v2_capability_cells.json`
   `family_contexts` enumerates **22** across the 8 active families. The
   registry is cited as measured. Not reconciled here — it is not this lane's
   object.
2. **Directory name.** The lead specified
   `docs/workstreams/claim2-trajectory/`; the workstream contract suggests
   `docs/workstreams/claim2_trajectory/`. The lead's explicit instruction wins;
   noted so a reader looking for the underscore form finds this entry.
3. **Cell coverage is `family:table`, not the registry cell.** The shards record
   `family:table` pairs, a free by-product of the enumerated marks. Mapping to
   the 22 registry capability cells needs the semantic cell classifier per mark
   and is **not implemented**. Reported as what it is; no claim of registry-cell
   coverage is made.
4. **Cost quoted from a different code path.** The measured 8.6–20.6 s/call
   comes from `canonical_successor_result` callers. This lane's row builder
   does one extra model forward, so the quote carries an explicit 1.2–1.5×
   assumption that the smoke is designed to measure rather than assume.
