# COMPOSE Process V2 atom deletion: implementation report

**Branch** `codex/editing-v2-process-v2-atom-delete` · **base** `bc05c6e` ·
**head** `761bfe1` plus the round-two integration repair described in §4 ·
**worktree** `/private/tmp/compose-process-v2-atom-delete`

Handoff: `docs/HANDOFF_PROCESS_V2_CONNECTED_NONLEAF_DELETE.md` ·
Decision: `docs/EDITING_PROCESS_V2_DECISION.md` ·
Contracts: `configs/editing_v2_semantic_process_v2.json`,
`configs/editing_gate_zero_semantic_model_process_v2.json`

This report **supersedes the round-one version of this file in full**. Round one
was rejected on independent review. §1 states exactly what was wrong. Every
number below was re-measured after the correction; none is carried over.

This report grants no authority. No Modal job, V2 proof scan, Active8
materialization, Gate 0, T1, P50, or training run was launched at any point in
either round.

## How to read the evidence here

Three categories are kept apart, because conflating them is how the round-one
report overclaimed:

* **Measured** — a command was run in this worktree and its output is quoted or
  summarized. Every count, hash, and pass/fail figure below is measured unless
  marked otherwise.
* **Bounded observation** — measured, but only over a stated finite panel (800
  Jin-QED leads; a 19-molecule charged panel; an 18-molecule batch panel). It
  says nothing about states outside that panel.
* **Proposed downstream behaviour** — what a future run *would* do. Nothing in
  this category has been observed, because no downstream run exists.

## 1. What was rejected, and what is approved now

### The rejected round-one reading

Round one read "preserve the existing root, singleton, and leaf capabilities" as
preserving the legacy *admission set*, and implemented Process V2 as a disjoint
union:

```
V2_admitted = V1_dense_mask | connected_nonleaf_admitted
```

The round-one report stated this explicitly — "`atom_delete` keeps its V1 rule
for root, singleton, and leaf slots bit-for-bit… the two sets are disjoint". That
exempted every inherited candidate from the authoritative charge policy and
thereby **preserved a legacy defect**. The counterexample was already measured
and had been used to *justify* the exemption rather than to refute it; that is
the failure mode worth recording, not just its symptom.

The defect is concrete. For `C[N+](C)(C)CC(=O)[O-]` the legacy dense rule admits
slots `[0, 2, 3, 6, 7]`, and the production runtime **refuses four of those
five** with `InvalidRewrite`. The learned mask was assigning probability to
transitions the executor cannot perform.

### The approved correction (verbatim)

> Apply the authoritative charge policy to every Process V2 atom-delete
> candidate, including inherited root, singleton, and leaf candidates. Preserve
> the general root/singleton/leaf capabilities only when the unchanged executor,
> connectivity, charge, support, and canonicalization predicates all pass.
> Exclude newly introduced connected-nonleaf candidates incident to a SCAR
> pending a separate SCAR semantic decision. This invalidates the current
> candidate Process V2 contract and identity before any downstream run.

### What that changed

**One admission authority**, `src/compose_v4/rewrite/process_v2_atom_delete.py`,
decides every candidate. The two candidate sources survive **only as diagnostic
labels** selecting which *additional* gates apply; neither exempts a slot from a
common gate.

| Gate | Applies to | Predicate |
|---|---|---|
| 1 | both sources | slot is a real element (`is_element`) |
| 2 | both sources | unchanged executor `is_valid_atom_delete` |
| 3 | both sources | successor connected or null |
| 4 | both sources | authoritative charge policy preserved |
| 5 | both sources | successor within declared support (≤ `MAX_ACTIVE_ATOMS`) |
| 6 | both sources | successor canonicalizable |
| 7 | connected-nonleaf only | non-aromatic |
| 8 | connected-nonleaf only | not an articulation point |
| 9 | connected-nonleaf only | not SCAR-incident |

The forward guard now asserts **equality** between the scored delete mask and the
admission mask, not containment. Containment would accept a mask unioned with the
legacy dense rule — precisely the exemption this version removes — so it could
not have detected the defect it exists to guard. §6 records the case where that
choice paid for itself.

The mode string moved from `process_v2_connected_nonleaf_atom_delete_v1` to
`process_v2_uniform_gated_atom_delete_v2`; the old name falsely implied inherited
candidates were unfiltered and is recorded as removed.

### Measured breadth of the correction

Bounded observation, 800 Jin-QED leads at 40 production slots:

```
inherited root/singleton/leaf candidates : 3319
  admitted                               : 3194
  charge_policy_violated                 :  125   (3.77%, across 105 of 800 molecules)
V1-admitted-but-V2-rejected, by code     : {charge_policy_violated: 125}
connected_nonleaf candidates             : {successor_disconnected: 7290,
                                            aromatic_atom: 5076,
                                            ADMITTED: 2321,
                                            charge_policy_violated: 243}
```

On that panel the executor, connectivity, declared-support and canonicalizability
predicates excluded **zero** further inherited candidates — the charge policy was
the whole gap. These figures were reproduced independently by the adversarial
reviewer and matched exactly.

**Process V2 is neither a superset nor a subset of the legacy rule.** Over the
frozen literal fixture it loses 6 slots and gains 37.

## 2. Identities

### Current

| Artifact | Value |
|---|---|
| V1 process identity | `6c4721f0dd37132aae657e7aa5f1bfc01cef270662f228171c4587eb7dd48491` |
| V2 process identity | `0c938177a34819e6e828920c1f66e240c6eb251fe7c9ea6cfe6757829dceb2dd` |
| V1 semantic process contract | `f928f6adaf22ba7520dd28839655c93bc523ce77317b32050d3dd1b02bbcf288` (frozen, unchanged from base) |
| V2 semantic process contract | `5e317e65cfbff0a8c8770765d1248925d0e068f4e2dd7f8a4cf474b85955ee1e` |
| V1 Gate-0 model/process contract | `db4e5cc5d2bb5efee97bcde767edaf85f41e238252def5b21832b6bbe398a3ed` |
| V2 Gate-0 model/process contract | `971a124a3d790e6d5435be5651d9d3f324ea7407cfd0e82944290ffaad138d3a` |
| V1 operator capability fingerprint | `d246bc88d8440d31` |
| V2 operator capability fingerprint | `d79ffe8ef65f3fb3` |

### Rejected and superseded — **not** current

| Value | Status |
|---|---|
| `9fde14b59fc6bfb7be7aaf83564658a9a6758f479d9fd94c134206e84873319b` | **REJECTED** round-one candidate V2 identity. It produced **no** downstream artifact: no payload, receipt, cache, checkpoint, Gate-0, T1, or P50 object was ever built under it. Recorded so it can never be mistaken for a current or superseded production identity. |
| `process_v2_connected_nonleaf_atom_delete_v1` | **REJECTED** round-one mode name. Recorded as a removed name only; nothing may alias it. |
| `6b98ee21ef8b853deda9fa56a2963178208ecc893a397fb4aa412629fc2414d7` | **SUPERSEDED** V1 identity, measured at base `bc05c6e`. Historical V1 migration payloads were built under it and remain readable under V1 via a pinned identity. |

The V1 identity moved because this work edits sources the V1 identity hashes
(`factorized_tracelet_rate_model.py`, `factorized_mark_conditional.py`,
`editing_v2_process_identity.py`). That is the invalidation mechanism working,
not a change to V1 semantics — §5 gives byte-level evidence that V1 *behaviour*
is unchanged.

## 3. The P50 chain — one statement, not two

The round-one report contained contradictory statements about whether the P50
chain was in scope. It is, and the resolution is:

**The P50 chain artifacts were re-pinned. Re-pinning them is not a Gate-0 run and
confers no authority.** They are re-pinned because they transitively
content-address the V1 process identity, which moved. Every P50-chain edit is a
hash-pointer substitution, verified as such in §5.

Precedent was checked rather than assumed: `git log -p` on `d5d3016`, an
analogous earlier process change, re-pinned the Gate-0 model/process contract,
the capability-cell registry, the Active8 decision runtime, the Gate-0 structural
contract, and the `FROZEN_CONTRACT_SHA256` source constant. Round one read that
as excluding P50. **That reading was wrong**: the P50 recipe policy
content-addresses the cell-role policy, so it moves whenever the cell-role policy
moves. It is included here.

Measured evidence produced under a superseded identity stays invalid either way,
which is why re-pinning cannot launder it into authority.

## 4. Complete file inventory

**52 files, +14,677 / −591** relative to `bc05c6e` (committed plus working tree).

Commits on the branch, oldest first: `42756ce`, `37f607c`, `9464076`, `60e5d9a`,
`655717f`, `e8bc288`, `067d8da`, `c506d70`, `85169b1`, `5368c46` (round one);
`4c6e4e2`, `3fb5e53`, `1bc1dfd`, `761bfe1` (round-two worker patches). The
round-two integration repair described in §6 is uncommitted at the time of
writing and will land as additive commits on this same branch; no history was
rewritten and nothing from round one was discarded.

### New — 18

| Path | Role |
|---|---|
| `src/compose_v4/rewrite/process_v2_atom_delete.py` | **the single admission authority** |
| `src/compose_v4/data/editing_process_v2_rebind.py` | V1→V2 corpus rebind; integrity vs support separated |
| `src/compose_v4/data/editing_process_v2_admitted_source.py` | fail-closed admitted-source adapter |
| `scripts/verify_process_v2_hash_chain.py` | read-only transitive re-pin verifier |
| `configs/editing_v2_semantic_process_v2.json` | V2 semantic process contract |
| `configs/editing_gate_zero_semantic_model_process_v2.json` | V2 Gate-0 model/process contract |
| `docs/EDITING_PROCESS_V2_DECISION.md` | the frozen prospective decision |
| `docs/PROCESS_V2_IMPLEMENTATION_REPORT.md` | this report |
| `tests/test_process_v2_atom_delete.py` | resolver behaviour |
| `tests/test_process_v2_atom_delete_gates.py` | per-gate witnesses, frozen literal fixtures |
| `tests/test_process_v2_atom_delete_mask.py` | independent-oracle mask comparison |
| `tests/test_process_v2_atom_delete_teacher.py` | teacher scorability, both directions |
| `tests/test_process_v2_atom_delete_invariance.py` | representation invariance |
| `tests/test_process_v2_runtime_checkpoint.py` | runtime/checkpoint threading |
| `tests/test_editing_process_v2_identity.py` | identity and contract |
| `tests/test_editing_process_v2_rebind.py` | rebind, incl. sharding equivalence |
| `tests/test_editing_process_v2_admitted_source.py` | adapter |
| `tests/test_verify_process_v2_hash_chain.py` | the verifier's own blind spots |

### Modified — 34

**Model / rewrite / data (5):** `model/factorized_tracelet_rate_model.py` (mask,
mode, capability, equality guard) · `rewrite/editing_v2_process_identity.py`
(coexisting V1/V2 identities) · `rewrite/trace_shard_v3.py` ·
`data/semantic_packed_trace_store.py` · `data/editing_corpus_contract.py`

**Experiments (15):** `editing_gate_zero_semantic_contract.py` (V2 Gate-0
variant) · `editing_v2_scientific_identity.py` · `editing_v2_semantic_runtime.py` ·
`factorized_mark_conditional.py` (capability keywords derived from
`dataclasses.fields`) · `editing_v2_semantic_t1_capacity_runner.py` (**defect 1
fix**) · `production_successor_kernel.py` ·
`canonical_successor_distillation.py` · `editing_successor_trainer.py` ·
`factorized_successor_training.py` · `successor_micro_overfit.py` ·
`editing_gate_zero_runtime.py` · `editing_training_gate.py` *(pin only)* ·
`editing_v2_semantic_gate_zero.py` *(pin only)* ·
`editing_v2_semantic_t1_capacity_policy.py` *(pin only)* ·
`editing_v2_semantic_t1_decision.py` *(pin only)*

**Scripts / apps (2):** `scripts/evaluate_tracelet_rollouts.py` (fail-closed
guard) · `modal_apps/materialize_editing_v2_semantic_p50_validation_baseline_app.py`

**Configs — hash pointers only (9):**
`editing_gate_zero_semantic_model_process_v1.json` · `editing_training_v2_gate.json` ·
`editing_v2_semantic_active8_decision_runtime_v1.json` ·
`editing_v2_semantic_capability_cells_v1.json` ·
`editing_v2_semantic_development_cell_roles_v1.json` ·
`editing_v2_semantic_gate_zero_structural_v1.json` ·
`editing_v2_semantic_p50_recipe_policy_v1.json` ·
`editing_v2_semantic_t1_capacity_policy_v1.json` ·
`editing_v2_semantic_t1_panel_policy_v1.json`

**Tests (4):** `test_editing_v2_semantic_t1_capacity_runner.py` ·
`test_materialize_editing_v2_semantic_t1_panel_cache_app.py` ·
`test_editing_v2_semantic_t1_panel_cache.py` · `test_teacher_in_candidates.py`

**Context (1):** `.claude/context/learnings.md` — **+50/−0 in round one, +40/−0 in
round two, purely additive.** This is not an unrelated change: the entries record
this round's durable lessons, including an explicit correction to the round-one
entry's own re-pin method, which is the method that produced defect 3 in §6.
`.claude/` is treated as code by this repository's stated convention, so the
entry ships with the change that makes it true.

## 5. Verification

Environment for every run:
`KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src`.

### Builder and checkpoint paths can construct Process V2

This was the blocking requirement. Round two found that
`editing_gate_zero_semantic_contract.py` called `semantic_model_identity()`
unconditionally, so **no Gate-0-shaped path could build a Process-V2 model at
all** — while every focused test still passed. Fixed by adding the V2 contract
variant, then verified *by construction* through the real
`build_semantic_scratch_runtime`:

```
V1: semantic_editing_v2_v1  legacy_acyclic_atom_delete_v1            fp d246bc88d8440d31
V2: semantic_editing_v2_v2  process_v2_uniform_gated_atom_delete_v2  fp d79ffe8ef65f3fb3
```

One constant was neutralized to do this: `build_production_ringcore_catalog`
reconstructs `82fd910cafe2eeb7` against a frozen `639ff6078c32d43c` on this
machine. That drift is **pre-existing and environmental** — reproduced identically
on base `bc05c6e`, whose catalog inputs this branch never touches. It is stated
rather than hidden, and it is not allowed to stand in for a real failure.

### Test counts

| Run | Result |
|---|---|
| Focused Process-V2 set (17 files) | **267 passed, 1 failed, 1 skipped** |
| The single failure | `test_full_partition_scorer_covers_exact_semantic_editing_v2_active8` — the pre-existing catalog drift, **identical failure verified on base `bc05c6e`** |
| Full suite, base `bc05c6e` | 14 failed, 2279 passed, 1 skipped, 15 errors |
| Full suite, head | see §5.1 |

The 15 collection errors are a missing PyYAML in this venv, identical on base and
head.

### Equivalence and invariance

| Check | Result |
|---|---|
| Legacy V1 batch vs base | **43/43 shared fields byte-identical** over an 18-molecule panel; the 2 head-only fields take inert legacy values (`legacy_acyclic_atom_delete_v1`, `None`) |
| Other seven Active8 operators | `git diff bc05c6e..HEAD` **empty** for `kernel.py`, `operators.py`, `factorized_fiber.py`, `tracelet_fiber.py`, `fiber.py`, `ring_system_fiber.py`, `aromaticity.py`, `state.py`, `molecular_graph.py`, `charge_policy.py`, `source_corruption.py`, `configs/editing_v2_semantic_process_v1.json` |
| Base-vs-head model export | **401 shared arrays, 0 differences** across 3 model configurations × 8 drug-like leads — every `state_dict` tensor, every batch field, every forward output, and the capability fingerprint |
| With V2 enabled | only `atom_delete` moves; **10/10** other action tables identical; all logits identical |
| Direct vs multiworker | proof rows **identical row-for-row**; aggregate census totals identical; exactly 8 provenance fields differ, pinned as an absolute literal set |
| Deterministic regeneration | 3/3 contracts: `rebuild == rebuild == committed` |
| Config re-pin | every difference a 64-hex → 64-hex pointer; **zero** policy, threshold, count or cell-definition changes |
| Representation invariance | 200 leads × 3 randomized non-canonical SMILES: **600 comparisons, 0 mismatches** |
| Charge, adversarial panel | 19 hand-built states (charged leaf; charged singleton `[NH4+]`; neutral adjacent to a charged centre; nitro; N-oxide; zwitterionic glycine; the carboxylate protonation trap `CC(=O)[O-]`; charged aromatic): **0 mask/runtime disagreements in either direction** |
| Mixed V1/V2 | every mixed combination raises with a distinct message; the feature cache key includes the delete mode, so a V1 and a V2 collator cannot share a cached mask; **no silent combination found** |
| Hash chain | `status: AGREES`, **0 stale literals** (138 agree, 7 lineage) |
| Pinned Ruff 0.15.22 | 36/36 touched files clean; whole repo **64 = 64** vs base, **none** from a touched file |
| `git diff --check` | clean |

### Mutation testing

17 mutations were applied to shadow copies of the source tree and the focused
suites re-run against each. **15 were killed.** The two that reproduce the
round-one defect matter most:

| Mutation | Result |
|---|---|
| Charge gate exempts the INHERITED source (**the round-one defect**) | KILLED — 13 failed |
| Collator re-unions the V1 dense mask (**the round-one shape**) | KILLED — 6 failed |
| Charge gate removed entirely | KILLED — 13 failed |
| V2 mask zeroed | KILLED — 20 failed, 2 errors |
| Candidate source always `inherited` | KILLED — 17 failed |
| SCAR gate removed / widened to both sources | KILLED — 3 failed / 1 failed |
| Aromatic gate removed | KILLED — 6 failed |
| Real-element, executor, connectivity, support, `MAX_ACTIVE_ATOMS`, invalid-source, forward equality guard | each KILLED |
| **Gate 6 (canonicalizability) downgraded to a swallow** | **SURVIVED** |
| **Gate 8 (articulation) removed** | **SURVIVED** |

Both survivals are reported rather than papered over, and both are now documented
in the module docstring as gates no test can currently fail on:

* **Gate 8** is unreachable by construction: removing a cut vertex from a
  connected graph always disconnects it, so gate 3 fires first. Verified on every
  one of 7,290+ such slots. The `articulation_point` diagnostic is still
  populated, so the exclusion stays observable.
* **Gate 6** is unreachable behind gate 2: `is_valid_atom_delete` requires the
  same `molecular_graph_to_smiles` round-trip that `canonical_state_key` would
  fail on. Round one's docstring **claimed a test witness for it that does not
  exist**. That claim is retracted; the docstring now states it is untested.

A methodological control matters here. Appending a no-op comment to
`process_v2_atom_delete.py` fails 4 tests, because that file is content-hashed by
the process identity. Those 4 are *content-hash* tests, not semantic ones — so a
mutation whose only failures are those 4 is a **survival**, not a kill. Without
that control the two survivals above would have been misread as kills.

### 5.1 Full-suite base/head comparison

Same invocation on both sides:
`python -m pytest tests/ -q --continue-on-collection-errors`.

```
base bc05c6e : 14 failed, 2279 passed, 1 skipped, 15 errors in 862.92s
head         : 14 failed, 2470 passed, 2 skipped, 15 errors in 879.11s
```

The comparison is over failure **sets**, not counts, because equal counts can
hide an equal-sized swap:

```
failures only on head (regressions) : none
failures only on base (fixed here)  : none
collection errors base vs head      : IDENTICAL
```

The two failure sets are the same 14 tests, in
`test_ringcore_validation_panel_builder.py` (6),
`test_ring_core_identity_gate.py` (2),
`test_aromatic_cycle_open_global_oracle.py` (2), and one each in
`test_semantic_active8_decision_modal_surface.py`,
`test_ring_core_zero_mixture.py`, `test_reference_successor_kernel.py`, and
`test_editing_v2_semantic_t1_capacity_runner.py`. The last is the RingCore
catalog drift described above. **+191 tests pass** on head, which is the new
Process-V2 coverage.

This is the *second* full-suite run. The first showed **16 regressions**, all in
`tests/test_editing_v2_semantic_t1_decision.py`. They were real, were traced to a
stale pin left by my own re-pin driver, and were fixed (§6, defects 3–5). That
sequence is the evidence for the claim in §6 that a green chain verifier is not
sufficient on its own.

## 6. Defects found and fixed in round two

Ordered by severity. Every one was found by a check, not by inspection.

1. **The Process-V2 T1 capacity path could not run at all.** *(found by
   adversarial review; confirmed independently before acting)*
   `_index_factorized_batch` re-indexes `atom_delete_mask` and three sibling
   admission masks but never `atom_delete_admission_mask`, so
   `dataclasses.replace` carried the full-batch tensor through:
   `index [2,0] → delete (2,40), admission (3,40)`. The forward **equality**
   guard caught it, so it failed *closed* rather than scoring selected rows
   against another row's fiber — the design choice of equality over containment
   paid for itself here. Two reasons it survived, both mine: the existing
   selected-vs-direct test compared a **hand-enumerated field list** (now derived
   from `dataclasses.fields`, so a future field is covered the day it is added),
   and its model fixture is V1, where the field is `None` and both sides compare
   vacuously (a V2 fixture and test were added). Mutation-proven: reverting the
   one-line fix yields `assert 3 == 2`.

2. **No builder could construct Process V2** (§5). A whole class of paths was
   structurally V1-only while every focused test passed.

3. **A stale pin that validated silently, introduced by my own round-one
   re-pin.** At base `bc05c6e` the T1 capacity policy config and both its
   consumers agreed at `2186dbd9…`; round-one commit `85169b1` moved the config
   to `2bb8e951…` and left both consumers behind, because that sweep skipped
   files already touched in the same pass. It failed 19 tests. Corrected rule:
   never exclude an already-touched file; exclude only the self-occurrence.

4. **Three ways an automated re-pin corrupts an artifact**, each caught by the
   focused suites *after* the chain verifier reported `AGREES`:
   (a) `process_identity_sha256` is a top-level `*_sha256` field that is **not** a
   self-hash, so recomputing "every `*_sha256`" destroys the pin — identify the
   self-hash field on the pre-mutation payload, where the equation still holds;
   (b) `*file_sha256` means the target's **physical** hash and `*semantic_sha256`
   its **self**-hash, so a role fallback wrote the self-hash into `file_sha256`,
   which every loader checking both rejects;
   (c) a value produced and superseded **inside one run** was never committed, so
   `git show` cannot see it and the pin holding it addresses nothing.
   The lesson: **chain-verifies-green is necessary but not sufficient.**

5. **Two blind spots in the chain verifier itself**, both of which had let a real
   stale pin read as green. It scanned only JSON for pointer edges, so pins
   written as Python dict constants (`EXPECTED_T1_CAPACITY_POLICY`) and as sibling
   module constants (`CAPACITY_POLICY_RELATIVE_PATH` +
   `CAPACITY_POLICY_{FILE_,}SHA256`) were invisible; and it derived chain
   membership from value matching, so an artifact whose pin resolved to
   **nothing** — the worst case, not a benign one — fell out of the chain and had
   its finding downgraded from failure to warning. Both closed, with
   `tests/test_verify_process_v2_hash_chain.py` carrying a built-in negative
   control that calls `_chain_report` with and without the edges and asserts
   opposite outcomes.

6. **Silent coercion in the admission authority.** `int(action.v)` ran before the
   type check, so `AtomDelete(1.9)` and `AtomDelete(True)` were truncated to slot
   1 and **admitted**. Now refused with `slot = -1`, so the report cannot be
   mistaken for a decision about slot 1. `np.int64` still works, because
   `np.flatnonzero` produces it throughout this codebase and rejecting it would
   be a different defect.

7. **A false integrity contract.** `_prove_record`'s docstring promised that a
   step whose teacher falls outside the frozen fiber is still codec-checked,
   replayed and compared. That held for `TEACHER_RULE_OUTSIDE_FROZEN_SUPPORT` but
   **not** for `TEACHER_FAMILY_OUTSIDE_FROZEN_SUPPORT`, whose executor rule the
   codec *can* decode — so a corrupt payload could hide behind an unsupported
   family. The code now matches the contract, following the pattern the
   `atom_delete` branch already used.

8. **A constant-vs-constant check.** `semantic_packed_trace_store.py` compared the
   manifest's process label against a hardcoded V1 module constant while accepting
   a V2 pinned identity, so the label was never cross-checked against the
   identity. Now compared against the resolved identity, matching its sibling
   `trace_shard_v3.py`.

9. **A silently-wrong checkpoint load path.** `evaluate_tracelet_rollouts.py`
   threads no semantic mode keywords, and semantic V1 and Process V2 have
   **identical parameter shapes**, so a V2 checkpoint would load cleanly and
   sample the V1 fiber without a word. A fail-closed guard now refuses. Teaching
   that evaluator the semantic processes is deliberately left as a separate
   change.

## 7. Reported findings I am rejecting

* **`__all__` violates the repository convention.** It does not, in this file's
  neighbourhood: **19 of 33** modules in `src/compose_v4/rewrite/` define
  `__all__`, and the stated rule "match the existing style of the file you are
  editing" governs. Recorded rather than silently ignored.

## 8. Open risks and things deliberately not done

* **Gates 6 and 8 are untested** (§5). Documented, not hidden. Constructing a
  witness for either requires changing a predicate the decision freezes.
* **SCAR-incident connected-nonleaf deletion is excluded**, pending a separate
  SCAR semantic decision. SCAR-adjacent *leaf* deletion stays admitted, because
  the decision withdraws no inherited capability. Verified: 0 inherited-source V2
  admissions fall outside the V1 dense mask over 813 states, and the same holds
  structurally, since a real-degree-≤1 slot is never an articulation point.
* **Aromatic connected-nonleaf deletion is excluded**, pending a separate
  semantic decision and resolver.
* **`test_sharding_the_work_cannot_change_the_published_artifact` is
  concurrency-sensitive**: it reads the live repository through `repo_root=ROOT`,
  so it can fail if the worktree changes underneath it. It passes in isolation and
  with its own file. This is a property of that file's existing fixture design
  rather than something the new test introduces, but a failure from it during a
  concurrent edit is not by itself evidence of a defect.
* **The rebind's published address is schedule-dependent by design** —
  `entries_per_task` is part of the run identity, so a 20-container run and a
  1-container run publish to different run roots. The *data* is schedule-invariant
  (§5), which is the property that matters; the addressing is a deliberate
  provenance choice, now pinned by a test so a *newly* schedule-dependent field
  fails rather than passing quietly.
* **The RingCore catalog drift** (§5) means several Gate-0-shaped tests cannot run
  in this environment at all. They fail identically on base.
* **Executing the rebind proof scan, Active8 materialization, Gate 0, T1, P50 and
  training remain not done, by instruction.**

## 9. Stop conditions honoured

* No Modal job, V2 proof scan, Active8, Gate 0, T1, P50, or training was launched.
* The executor, aromatic semantics, charge policy, canonicalization,
  persistent-slot semantics, multi-neighbour insertion, `ring_system_delete`,
  `ring_system_grow`, and the other seven Active8 operators are unmodified —
  verified by empty diffs and a 401-array export comparison (§5).
* V1 identities and artifacts are preserved;
  `configs/editing_v2_semantic_process_v1.json` is byte-identical to base.
* No new chemistry semantics were improvised.
* History was not rewritten; round two is additive repair on the same branch.
* No commit message mentions any AI assistant.
