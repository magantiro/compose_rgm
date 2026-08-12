# Workstream

- **Name:** D — External-baseline qualification and adapter readiness
- **Claim ID:** C4 (all four sub-capabilities C4a–C4d). This lane makes **no**
  claim about C1, C2 or C3 — an external method does not share the COMPOSE
  executable support or training target and cannot be an arm there.
- **Branch:** `codex/compose-baseline-qualification`
- **Base commit:** `04f1c46` ("Add parallel workstream plan and agent handoff template")
- **Content commit:** `handoff.json` field `content_commit` — the commit holding
  every artifact below. This file deliberately does not repeat the hash: naming
  it here would make the file that records the commit part of the commit it
  records. Branch HEAD is one commit ahead of `content_commit` and differs from
  it only by `handoff.json`, which must be written after the content commit in
  order to hash it.
- **Working tree clean:** yes
- **Status:** `DESIGN_ONLY`
- **Held-out data opened:** **no**

# One-sentence scientific question

> For each external method, which COMPOSE Claim-4 sub-capability is it a valid
> comparator for, under what definition of matched budget, and where does the
> comparison stop being fair?

# Claim this work can support

> A verified, per-cell-sourced statement of comparability: which methods may
> enter which COMPOSE table, what "matched budget" means for each, and which
> COMPOSE Claim-4 capabilities a baseline already does natively.

# Claims this work cannot support

- Any claim that COMPOSE beats or loses to any baseline. No sweep was run.
- Any claim that a baseline is weak. Absence of a capability is a scope fact.
- Any claim from published numbers. PMO's leaderboard is recorded as context and
  is comparable only under PMO's own oracle-counting convention.
- Any claim about C1, C2 or C3. Externals are `N/A` there by construction.

# Frozen inputs

| Object | Path / ID | SHA-256 / identity | Verified? |
|---|---|---|---|
| Process-V2 chemistry | RingCore catalog fingerprint | `639ff6078c32d43c` (production, from `CLAUDE.md`) | not exercised — this lane ran no chemistry |
| `R_theta` checkpoint | — | — | not exercised |
| split / panel | — | — | **none opened** |
| sampling law | — | — | not exercised |
| goal/oracle | `src/compose_v4/drd2_oracle.py` | `318cc812b3a5c591ee4bcb8c2cd2da004ff5b147856c9b0375e7fe81a3bf1a44` | yes — named as the shared oracle in every adapter design |
| goal/oracle | `diagnostics/retarget_goal_language_normalizers.json` | `187d1ccc60b00c858f0f54f729a99ea91e06d33a913287886c80080889a76cc8` | yes — named as the frozen goal language |
| other | `configs/comparator_registry_v1.json` (schema v2) and its validator | untouched | yes |

This lane consumed no frozen model or panel. It names them so the adapters bind
to the right objects later.

# Protocol

- **Panel construction:** none carved. Smoke sources are specified as 3–5
  held-in molecules and are not selected here.
- **Arms:** not applicable — this lane qualifies comparators, it does not run them.
- **Primary metric:** none. The output is a capability matrix, not a measurement.
- **Secondary metrics:** none.
- **Independent statistical unit:** not applicable.
- **Allowed calibration:** choice of container sizes, timeouts, and dependency
  pins needed to make an environment build, each recorded as a deviation.
- **Stop rules:** (1) CPU-only build failure → `adapter_status: blocked_cpu`,
  demote to `CONDITIONAL` gated on GPU authorisation; (2) missing or unlicensed
  checkpoint → `CONTEXT_ONLY`; (3) **a method turns out to natively support a
  COMPOSE sub-capability → stop and escalate immediately** — this fired, see
  below; (4) no faithful adapter without a forbidden adaptation → `CONTEXT_ONLY`.
- **Forbidden adaptations:** listed in `PROTOCOL.md`. The two that bit here:
  synthesising a trajectory for a method that has none (GraphGA lineage,
  REINVENT SMILES prefixes), and omitting a method's objective-specific training
  from its oracle count.

# What was implemented

- `src/compose_v4/experiments/baseline_qualification.py` — registry schema and
  validator. Enforces that every capability cell carries a primary source or is
  `UNVERIFIED`; that a `YES` on a novelty-bearing axis is escalated; that a
  `PARTIAL` appears in the near-miss table; that every escalation states what
  COMPOSE still has; and that an aggregate verdict is never softer than the
  sub-capability rows it summarises.
- `docs/workstreams/baseline-qualification/comparator_registry_v3.json` — the
  qualification registry, six methods, six capability axes each, all sourced.
- `scripts/render_comparator_registry.py` — renders `COMPARATOR_MATRIX.md` from
  the JSON so the two cannot drift; `--check` verifies.
- `tests/test_baseline_qualification.py` — 10 tests, including one asserting the
  rendered markdown is current and one asserting `configs/comparator_registry_v3.json`
  agrees with the qualification verdicts.
- `configs/comparator_registry_v3.json` — external rows updated to carry the
  verdict, the adapter status and a `qualification_record` pointer. HN-GFN, which
  had been dropped from this file although the schema-v2 registry listed it, is
  restored. **No capability facts were duplicated into this file.**
- `baselines/<method>/README.md` and `environment.lock` for all six methods.
  Every lock is declarative and marked `UNRESOLVED` or `BLOCKED`; nothing was
  installed.
- `PROTOCOL.md`, `FAIRNESS_CONTRACT.md`, `DECISION_LOG.md`, `STATUS.md`.

# Tests and smoke checks

| Test | Result | Artifact |
|---|---|---|
| `tests/test_baseline_qualification.py` (10 tests) | PASS | — |
| `tests/test_comparator_registry.py` (14 tests, pre-existing) | PASS — unaffected | — |
| `scripts/render_comparator_registry.py --check` | PASS (in-sync) | `COMPARATOR_MATRIX.md` |
| Held-in smoke, any method | **NOT RUN** — categorically forbidden by the lane's instructions | — |

# Results

**These are qualification verdicts, not measurements. No comparator was run.**

| Metric | Arm / condition | Value | Uncertainty / denominator |
|---|---|---:|---|
| methods qualified | MARS, GraphXForm, DDSBM, HN-GFN, GraphGA, REINVENT | 6 | 6 of 6 priority methods |
| capability cells sourced to a primary source | all methods × 6 axes | 36 / 36 | 0 cells left `UNVERIFIED` |
| COMPOSE capabilities a baseline does natively | escalated | 5 entries across 3 methods | see below |
| near misses recorded | — | 12 | — |
| `MUST_RUN` | MARS, GraphXForm, GraphGA, REINVENT | 4 | — |
| `CONDITIONAL` | HN-GFN | 1 | gate: GPU + working BoTorch pin |
| `CONTEXT_ONLY` | DDSBM | 1 | blocker: no license, no checkpoints |
| projected smoke compute | four runnable methods | 4.4–6.9 CPU-core-hours | projection, never measured |

# Gate verdicts

| Gate | PASS / FAIL / INCONCLUSIVE | Evidence |
|---|---|---|
| Every capability claim carries a primary source | **PASS** | validator enforces it; 36/36 cells sourced |
| No COMPOSE task, panel, goal or budget altered | **PASS** | no COMPOSE file other than `configs/comparator_registry_v3.json` status fields was modified |
| No held-out or sealed data opened | **PASS** | no panel loaded; no diagnostics read beyond hashing two frozen inputs |
| No Modal run launched, no heavy dependency installed | **PASS** | no `modal` invocation; no install |
| A baseline natively does a COMPOSE claim | **FIRED — escalated** | REINVENT 4 staged learning; HN-GFN preference conditioning; GraphXForm per-step masking |
| C4a comparison is claim-bearing | **FAIL — reclassified as a competitiveness sanity check** | the main lane's held-in calibration measured future-aware control adding nothing on an easy target-free property goal (greedy 28/30 = verified 28/30, headroom 0, gate CLOSED); PMO's DRD2/GSK3β/JNK3 tasks are that regime |
| Any baseline is applicable to C4b, the regime where future-aware control *is* established | **FAIL — all six are `N/A`** | GraphXForm cannot delete; MARS has no remaining-budget concept; GraphGA does not preserve the source; REINVENT and HN-GFN have no intermediate molecular states; DDSBM has no oracle in the loop. **Report `N/A`, never as a COMPOSE win.** |
| GraphXForm qualifies as a native pathwise baseline | **FAIL** | shipped substructure constraints are a terminal `-inf` filter; no SMARTS anywhere in the repo. It instantiates the *endpoint-only filtering* arm instead. |
| DDSBM qualifies as a matched comparator | **FAIL** | no LICENSE file (all rights reserved), no checkpoints, intermediates are not molecules, zero oracle calls at sampling |

# Bugs, invalid instruments, and superseded runs

No run was performed, so no result is invalid. Three **upstream** defects that
would have invalidated a run had one been attempted:

1. **HN-GFN does not import on a modern stack.** `proxy/regression.py` imports
   `AnalyticMultiOutputObjective` and `IdentityAnalyticMultiOutputObjective`,
   both removed from current BoTorch. Upstream issue #1, opened 2024-03-05,
   unanswered, repo untouched since 2023-12-24. Also `np.bool` / `np.int` usages
   that fail on NumPy ≥ 1.24. Status: upstream defect, recorded in
   `baselines/hn_gfn/environment.lock`.
2. **GraphXForm silently continues from a previous run's output.** The upstream
   README states `data/generated_molecules.pickle` is merged rather than
   overwritten, so an undeleted file makes the next run continue from it. This is
   a cross-run contamination hazard and is listed as an adapter work item.
3. **MARS's adapted proposal is never persisted.** `sampler.py` calls `train(...)`
   without `save_dir`, so the only `torch.save` in `common/train.py` is
   unreachable and `--editor_dir` is effectively dead. Any "warm restart" arm
   claiming to carry MARS's proposal across a switch would be measuring a freshly
   initialised network. Recorded so nobody builds that arm by accident.

# Known limitations

- **OpenReview was unreachable.** Forums for MARS (`kHSu4ebxFXY`), DDSBM
  (`tQyh0gnfqW`), HN-GFN (`uoG1fLIK2s`) and PMO (`yCZRdI0Y7G`) return HTTP 403
  behind a bot challenge. No reviewer discussion was read for any method. Venues
  were confirmed from proceedings pages and paper footers instead.
- **Every cost figure is a projection.** Nothing was executed, so no runtime was
  measured. The CPU-throughput assumptions for GraphXForm and MARS are the
  weakest numbers in the packet.
- **The Modal CPU rate used for the dollar estimate is an assumption**, not a
  figure verified from this repository.
- **Checkpoint hashes are absent** for GraphXForm (347 MB, LRZ) and the REINVENT
  priors (Zenodo). Both were verified reachable but neither was downloaded, so
  neither is hashed. Hash on first fetch before any claim-bearing run.
- **REINVENT 4 has no bib entry.** `paper_iclr_stochastic_rewriting/references.bib`
  has `gao2022pmo` but nothing for Loeffler et al. 2024. `gao2022pmo` must **not**
  be used as the citation for a REINVENT 4 capability.
- **`docs/RELATED_WORK_MATRIX.md` was not edited** despite two cells being in
  tension with what was verified. See `DECISION_LOG.md` — the asymmetry in the
  reasoning is deliberate and should be reviewed.

# Exact reproduction commands

```bash
# environment/setup — nothing external is installed; this uses the repo venv only
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE

# validate the registry and check the rendering is in sync
python scripts/render_comparator_registry.py --check

# regenerate the markdown after any registry edit
python scripts/render_comparator_registry.py

# tests
pytest tests/test_baseline_qualification.py -q
pytest tests/test_comparator_registry.py -q      # pre-existing, must stay green

# smoke — NONE. No comparator smoke was run and none is authorised by this lane.
```

# Durable artifacts

SHA-256 for every path below is recorded in `handoff.json` (`durable_artifacts`),
computed at the moment that file was written. They are not duplicated here so the
two cannot disagree.

| Artifact | Path | SHA-256 | Purpose |
|---|---|---|---|
| qualification registry | `docs/workstreams/baseline-qualification/comparator_registry_v3.json` | see `handoff.json` | authoritative capability facts |
| rendered matrix | `docs/workstreams/baseline-qualification/COMPARATOR_MATRIX.md` | see `handoff.json` | generated; do not hand-edit |
| protocol | `docs/workstreams/baseline-qualification/PROTOCOL.md` | see `handoff.json` | evidence standard, classification rule, forbidden adaptations |
| fairness contracts | `docs/workstreams/baseline-qualification/FAIRNESS_CONTRACT.md` | see `handoff.json` | per-MUST_RUN-method matched-budget definitions |
| decision log | `docs/workstreams/baseline-qualification/DECISION_LOG.md` | see `handoff.json` | every material decision and its alternatives |
| status | `docs/workstreams/baseline-qualification/STATUS.md` | see `handoff.json` | one-screen state |
| claim registry | `configs/comparator_registry_v3.json` | see `handoff.json` | verdict + pointer only |
| validator | `src/compose_v4/experiments/baseline_qualification.py` | see `handoff.json` | schema invariants |
| renderer | `scripts/render_comparator_registry.py` | see `handoff.json` | JSON → markdown |
| tests | `tests/test_baseline_qualification.py` | see `handoff.json` | 10 tests |
| per-method packets | `baselines/{mars,graphxform,ddsbm,hn_gfn,graph_ga,reinvent}/` | see `handoff.json` | README + environment.lock |

# Files changed

```text
baselines/ddsbm/README.md
baselines/ddsbm/environment.lock
baselines/graph_ga/README.md
baselines/graph_ga/environment.lock
baselines/graphxform/README.md
baselines/graphxform/environment.lock
baselines/hn_gfn/README.md
baselines/hn_gfn/environment.lock
baselines/mars/README.md
baselines/mars/environment.lock
baselines/reinvent/README.md
baselines/reinvent/environment.lock
configs/comparator_registry_v3.json
docs/workstreams/baseline-qualification/COMPARATOR_MATRIX.md
docs/workstreams/baseline-qualification/DECISION_LOG.md
docs/workstreams/baseline-qualification/FAIRNESS_CONTRACT.md
docs/workstreams/baseline-qualification/HANDOFF.md
docs/workstreams/baseline-qualification/PROTOCOL.md
docs/workstreams/baseline-qualification/STATUS.md
docs/workstreams/baseline-qualification/comparator_registry_v3.json
docs/workstreams/baseline-qualification/handoff.json
scripts/render_comparator_registry.py
src/compose_v4/experiments/baseline_qualification.py
tests/test_baseline_qualification.py
```

# Recommended next action

One bounded action only:

> **Build and smoke the GraphGA adapter on 3–5 held-in sources** — wrap the
> frozen COMPOSE goal language and `src/compose_v4/drd2_oracle.py` as a
> `scoring_function`, count oracle calls under **one explicitly declared
> convention**, and record the per-source cost. Projected 0.1 CPU-core-hours,
> HIGH confidence, MIT-licensed, RDKit-only, and it can share the COMPOSE rdkit
> pin exactly. It is the cheapest end-to-end proof that the oracle shim and the
> counting convention work, and §0 of `FAIRNESS_CONTRACT.md` shows that
> convention must be fixed before any other comparator runs.

# Actions explicitly not recommended

- **Do not build the MARS or GraphXForm adapter first.** Both are 10–40× the
  compute and carry the environment risk; neither tests anything GraphGA does not
  test more cheaply about the shared plumbing.
- **Do not attempt DDSBM.** Resolve the license question with the authors first;
  no amount of engineering fixes an absent LICENSE file.
- **Do not run any baseline before the oracle-counting convention is declared.**
  A run under an undeclared convention has to be repeated.
- **Do not read the C4a static-optimization table as a test of future-aware
  control.** It is a competitiveness sanity check. PMO's DRD2/GSK3β/JNK3 tasks
  are the easy target-free regime where COMPOSE has *measured* that future-aware
  control adds nothing (greedy 28/30 = verified 28/30, gate CLOSED). A baseline
  winning there refutes nothing; a COMPOSE win there supports only "the substrate
  plus ordinary control is competitive".
- **Do not report "no baseline is applicable to C4b" as a COMPOSE win.** It is a
  scope fact and belongs in an `N/A` cell.
- **Do not pick a comparison task because it makes greedy fail.** The anti-tuning
  rule in `docs/RETARGETING_SAME_PREFIX_DESIGN.md` binds baseline selection too.
- **Do not weaken the C4c claim to "COMPOSE changes objective mid-run".**
  REINVENT 4 does that natively. Claim the conjunction — unanticipated goal,
  realized molecular history preserved, finite remaining budget, frozen reference
  process — which `docs/RETARGETING_SAME_PREFIX_DESIGN.md` already states
  correctly.
- **Do not present GraphXForm as a native pathwise-motif baseline.** Its shipped
  substructure mechanism is a terminal filter.
- **Do not edit `docs/RELATED_WORK_MATRIX.md` on this branch.** Two cells need
  review; the manuscript workstream owns that file.

# Main-session pickup checklist

- [ ] Read protocol before results.
- [ ] Verify all frozen-input hashes.
- [ ] Confirm held-out-open status.
- [ ] Reproduce one smoke. *(None exists — reproduce
      `python scripts/render_comparator_registry.py --check` instead, and note in
      the acceptance record that this lane is `DESIGN_ONLY` by instruction.)*
- [ ] Inspect known-invalid runs. *(None — three upstream defects are listed
      instead.)*
- [ ] Decide explicitly whether to merge, authorize held-out evaluation, or stop.
- [ ] **Decide the oracle-counting convention** (`FAIRNESS_CONTRACT.md` §0). This
      blocks every comparator run.
- [ ] **Review the two `RELATED_WORK_MATRIX.md` cells** flagged in
      `DECISION_LOG.md`.
