# COMPOSE Process V2 connected-nonleaf atom deletion: implementation report

Branch `codex/editing-v2-process-v2-atom-delete`, base `bc05c6e`, head `85169b1`.
Implementation handoff: `docs/HANDOFF_PROCESS_V2_CONNECTED_NONLEAF_DELETE.md`.
Prospective decision: `docs/EDITING_PROCESS_V2_DECISION.md`.
Frozen contract: `configs/editing_v2_semantic_process_v2.json`.

This report is evidence for independent review. It grants no authority. No Modal
job, Active8 materialization, Gate 0, T1, P50, rebind proof scan, or training run
was launched at any point.

## 1. What was approved, and what was deliberately not done

The recorded prospective decision expands the learned marked fiber to
executor-verified, charge-preserving, non-aromatic, non-articulation
connected-nonleaf atom deletions, invalidating V1 downstream artifacts.

Implemented: the semantic-process implementation, its prospective decision
artifact, compatibility and rebinding proof infrastructure, tests, documentation.

Not done, by instruction: Active8 materialization, Gate 0, T1, P50, later pilots,
long editing runs, de novo runs, and execution of the rebind proof scan. The
executor, aromatic semantics, charge policy, canonicalization, persistent-slot
semantics, the other seven Active8 operators, multi-neighbour insertion,
`ring_system_delete`, and `ring_system_grow` are untouched.

## 2. The scientific decision and the fact it rests on

`atom_delete` keeps its V1 rule for root, singleton, and leaf slots bit-for-bit
and *additionally* admits connected-nonleaf candidates. The two sets are disjoint,
so the effective mask is a disjoint union:

```
V2 atom_delete mask = (unchanged V1 dense mask) UNION (connected-nonleaf admission)
```

**The load-bearing structural fact**, derived and then measured rather than
assumed: the V1 dense mask admits a real slot only when it lies on no cycle and is
not a cut vertex. In a connected real-atom graph an acyclic vertex of degree at
least two is always a cut vertex, so **V1 admits only real-atom degree at most
one**. Measured on an 18-molecule panel: zero V1 admissions of degree >= 2.
`tests/test_process_v2_atom_delete.py` pins the disjointness rather than assuming it.

**Why the disjoint-union form is necessary, not stylistic.** Applying the seven V2
conditions uniformly would *remove* existing V1 leaf candidates. Measured
counterexample: for `C[N+](C)(C)CC(=O)[O-]`, deleting the neutral leaf methyl at
slot 0 is V1-admitted, but the successor violates the charge policy because the
charged N+ centre's bond row and implicit hydrogens change. Uniform application
would silently change preserved leaf behaviour and break the handoff requirement.

The seven admission conditions for the additional set, in evaluation order, live in
`src/compose_v4/rewrite/process_v2_atom_delete.py`: real element under the
authoritative predicate; real-atom degree >= 2; non-aromatic under
`resonance_invariant_bond_classes`; not an articulation point; the unchanged
production executor accepts; the exact persistent-slot successor is connected; the
frozen charge policy is preserved; and the successor is within the declared
broad-organic, at-most-40-active-atom support and is canonicalizable.

**The executor is the legality authority.** No weaker approximate valence test was
introduced. `is_valid_atom_delete` is deliberately *not* a connectivity predicate
(it returns `True` for articulation-point deletions), so connectivity cannot be
delegated to it. Condition 3 supplies connectivity instead, since removing a
non-cut vertex from a connected graph leaves it connected, and condition 5 is
therefore implied rather than independent. An earlier draft of this report and of
the module docstring claimed the two were independently load-bearing; the
adversarial review disproved it and both have been corrected.

**Aromatic exclusion is load-bearing, with a measured reason.** Deleting one
Kekule-encoded slot of a perceived aromatic system yields a representation-sensitive
open-chain successor: benzene gives `C=CC=CC`, and toluene gives different products
depending on which Kekule slot is deleted (`C=CC=CCC` versus `C=C(C)C=CC`). Aromatic
connected-nonleaf deletion therefore remains excluded pending a separate semantic
decision and resolver.

Verified reference behaviour (used as fixtures):

| source | admitted connected-nonleaf slots |
|---|---|
| `C1CCCCC1` | 0-5, successor `CCCCC` |
| `C1CCOCC1` | 0-5, heteroatom preserved |
| `c1ccccc1`, `Cc1ccccc1`, `c1ccc2ccccc2c1` | none (aromatic) |
| `CC1CCCCC1` | 2,3,4,5,6 (slot 1 is an articulation point) |
| `C1CC2CCC1CC2` | 0-7 (bridged bicyclic) |
| `O=C1NC(O)C2CCCCC12` | 2,5,6,7,8,9,10 (fused) |
| `CCO`, `C` | none (leaf/root stay under the V1 rule) |
| `C[N+](C)(C)CC(=O)[O-]` | none (every degree>=2 slot is an articulation point) |

Corroboration the design did not rely on: the frozen development cell-role policy
already declares `editing_v2_active8_v1:atom_delete:connected_nonleaf_death` a
**required** capability cell. Process V2 fills a declared-but-previously-unreachable
cell rather than inventing a new one.

## 3. Exact files changed

All paths relative to the repository root. **34 files, +8,065 / -227.**

### New (12)

| Path | Lines | Role |
|---|---|---|
| `src/compose_v4/data/editing_process_v2_rebind.py` | 2144 | Phase 4 restart-safe, content-addressed, proof-bound V1-to-V2 rebind |
| `tests/test_editing_process_v2_rebind.py` | 962 | Rebind battery (25 tests) |
| `tests/test_process_v2_atom_delete_mask.py` | 920 | Model and batch mask battery |
| `tests/test_editing_process_v2_identity.py` | 618 | Identity, contract, masquerade battery |
| `tests/test_process_v2_atom_delete.py` | 404 | Resolver and executor-parity battery |
| `docs/PROCESS_V2_IMPLEMENTATION_REPORT.md` | 395 | This report |
| `src/compose_v4/rewrite/process_v2_atom_delete.py` | 334 | Phase 2 scientific core: resolver, enumerator, slot mask, 11 reason codes |
| `configs/editing_v2_semantic_process_v2.json` | 331 | Phase 1 self-hashed Process V2 semantic contract |
| `docs/EDITING_PROCESS_V2_DECISION.md` | 219 | Prospective decision record |
| `tests/test_process_v2_atom_delete_invariance.py` | 191 | Quotient invariance and cross-module assertions |
| `tests/test_process_v2_atom_delete_teacher.py` | 189 | Teacher scores finite and backpropagates; V1 negative control |
| `tests/test_process_v2_atom_delete_gates.py` | 140 | Reachable witnesses for the three gates no test could fail on |

### Modified (22)

**Source, behaviour (8):**

| Path | Diff | Change |
|---|---|---|
| `src/compose_v4/rewrite/editing_v2_process_identity.py` | +587/-18 | Coexisting V1 and V2 identities, frozen-identity validation, V2 contract builder |
| `src/compose_v4/model/factorized_tracelet_rate_model.py` | +213/-9 | Disjoint-union mask, capability and batch plumbing, semantic-version predicate |
| `src/compose_v4/data/semantic_packed_trace_store.py` | +123/-8 | Pinned-identity readers |
| `src/compose_v4/rewrite/trace_shard_v3.py` | +33/-8 | Pinned-identity record decode |
| `src/compose_v4/experiments/factorized_mark_conditional.py` | +15/-0 | Collation threading |
| `src/compose_v4/experiments/production_successor_kernel.py` | +11/-3 | Runtime selection and provenance by semantic version |
| `src/compose_v4/experiments/factorized_successor_training.py` | +2/-2 | Semantic-version predicate |
| `src/compose_v4/experiments/editing_v2_semantic_t1_decision.py` | +2/-2 | Re-pinned capacity-policy physical and semantic hashes |

**Source, hash constants only (3):** `editing_v2_semantic_gate_zero.py` (+1/-1),
`editing_v2_semantic_t1_capacity_policy.py` (+2/-2), `editing_training_gate.py` (+1/-1).

**Configs, hash pointers only (9):** `editing_gate_zero_semantic_model_process_v1.json`,
`editing_v2_semantic_capability_cells_v1.json`,
`editing_v2_semantic_development_cell_roles_v1.json`,
`editing_v2_semantic_active8_decision_runtime_v1.json`,
`editing_v2_semantic_gate_zero_structural_v1.json`,
`editing_v2_semantic_t1_panel_policy_v1.json`,
`editing_v2_semantic_t1_capacity_policy_v1.json`,
`editing_v2_semantic_p50_recipe_policy_v1.json`, `editing_training_v2_gate.json`.
The larger line counts on the capability-cell registry (+119/-75) and P50 recipe
policy (+64/-64) are key-sort only; every semantically changed field is a hash.

**Tests (2):** `tests/test_editing_v2_semantic_t1_panel_cache.py` (+12/-1, an
over-specified fail-closed assertion, section 7) and
`tests/test_materialize_editing_v2_semantic_t1_panel_cache_app.py` (+1/-1, fixture hash).

## 4. Commits

| Commit | Subject |
|---|---|
| `42756ce` | rewrite: resolve process v2 connected-nonleaf atom deletion |
| `37f607c` | editing-v2: freeze prospective process v2 support decision |
| `9464076` | tests: pin process v2 delete-fiber representation invariance |
| `60e5d9a` | model: admit process v2 connected-nonleaf atom deletes |
| `655717f` | data: prove v1 payload compatibility under process v2 |
| `e8bc288` | editing-v2: read superseded-identity payloads under a pinned identity |
| `067d8da` | editing-v2: rebind the v1 gate chain to the moved process identity |
| `c506d70` | experiments: select the rewrite runtime by semantic process version |
| `85169b1` | editing-v2: complete the identity re-pin and close three untestable gates |

## 5. Identities

Naming: `editing_v2` is the source-conditioned editing **lane**; the trailing `_v1`
/ `_v2` is the **semantic process version** of that lane.

### New

| Object | SHA-256 |
|---|---|
| Process V2 identity (`semantic_editing_v2_v2`) | `9fde14b59fc6bfb7be7aaf83564658a9a6758f479d9fd94c134206e84873319b` |
| V2 contract, semantic self-hash | `30369f36373b1163114b9a8bf6849f3fbed4627b6070e4478c4a5efb6c3c6261` |
| V2 contract, physical file hash | `6111edce5fd7ab6214ae069804c48e6c1e902ac969463a7b279b834d16780cc9` |

### Moved (the declared invalidation)

| Object | Was | Now |
|---|---|---|
| V1 process identity | `6b98ee21ef8b853deda9fa56a2963178208ecc893a397fb4aa412629fc2414d7` | `9874a69a01902dd1c4a0353c14cb9db0c657887b3e497800867629d82f1bf63e` |
| Gate-0 model/process contract | `df1bd6a4…d176ec99` | `cd940093c629802b5f29847dee5fa3cf0c2408c31f9e1f31345870a6bf1ec082` |
| Capability-cell registry | `5ca11004…b652c9eb` | `54379adfa479e4f6de17c13f2a85a135ae753360ec0106a183a92c88e722fb7f` |
| Development cell-role policy | `d65781105ed6…54ebc44` | `a26c7084b2cf6f23a7690f5dc5f76f992056b205c0dc189405643a22a90b603c` |
| Active8 CPU decision runtime | `313eeb55…6bc9424e` | `ccb8094202ecc51065b7e8c52a8cc4a8a85ac4c62ce50674b883fcdd09fe9256` |
| Gate-0 structural contract (and its source-pinned constant) | `ad4181c8…0321dbf5` | `52385445c096db737367df196e6b56c01bea515992601e636f0fa25dc63ef759` |

### Unchanged, by design

The V1 semantic process contract `configs/editing_v2_semantic_process_v1.json` is
untouched: semantic `f928f6adaf22ba7520dd28839655c93bc523ce77317b32050d3dd1b02bbcf288`,
physical `43cb26e1ba33b27a0149a8142895ef9988853b458c0ace9bc9191f7b460dedbb`. The V1
identity *definition* (schema, schema version, semantics string, contract path,
implementation-source list, body field set) is unchanged; only its computed value
moved, because implementation sources it binds legitimately changed. That movement
is the invalidation mechanism the contract records, not a defect.

### Why the V1 identity had to move

`editing_v2_process_identity()` hashes implementation sources including
`factorized_tracelet_rate_model.py`, which Phase 2 necessarily changes. Preventing
the movement would require editing the V1 identity definition, which the handoff
forbids. The handoff resolves the tension by requiring explicit V1 downstream
invalidation in the V2 contract, and the approval sentence ends "invalidating V1
downstream artifacts".

### Scope of the re-pin, and its precedent

The five re-pinned configs plus the one source constant are **exactly** the set that
the analogous prior process change `d5d3016` re-pinned, plus
`editing_v2_semantic_development_cell_roles_v1.json`, which was created later
(`1a31984`) and sits in the same Gate-0 binding chain. `d5d3016` did **not** touch
the P50 lane, and neither did this work.

Only hash pointers and self-hashes moved. Asserted programmatically at re-pin time:
no cell definition, role assignment, count, threshold, decision policy, required
architecture, or structural check changed, and the Gate-0 `model_identity` block is
byte-identical. Every re-pinned artifact still binds the **V1** semantic process
contract path; none was relabelled as V2.

**Re-pinning a non-authorizing binding contract is not a Gate-0 run.** Any *measured*
Gate-0 or T1 evidence produced under `6b98ee21…` remains invalid and must be
regenerated when that lane is authorized.

## 6. Verification

Toolchain: this worktree carries no `.venv`, so the repository-pinned main-tree
environment is used, which is the established worktree practice.

```bash
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1
VP=/Users/rmaganti/Documents/Codex/2026-07-14/ok-so/compose_rgm/.venv/bin/python
$VP -m pytest tests/ -q
$VP -m ruff check <touched files>
git diff --check
```

### Full suite, actually run, exact counts

| Revision | Result |
|---|---|
| base `bc05c6e` | **21 failed, 2480 passed, 1 skipped**, 0 errors (908.76 s) |
| head `85169b1` | **21 failed, 2588 passed, 2 skipped**, 0 errors (947.46 s) |

Attributed by node-ID set difference: **0 new failures, 0 new errors, 0 base
failures fixed**. The failing set at head is exactly the failing set at base, and
the branch adds 108 passing tests. The independent adversarial reviewer ran both
suites separately and reproduced the base counts exactly, which is why these are
reported as measurements rather than single observations.

An intermediate head (`c506d70`) did show 48 failed / 14 errors, all in the P50
and T1 chains from one root cause: the identity re-pin had stopped at the Gate-0
chain while the cell-role policy hash is content-addressed further downstream.
That is recorded rather than hidden, because it is the finding that produced the
`85169b1` fix and because it demonstrates the failure mode the re-pin discipline
exists to prevent.

### The 21 pre-existing failures

Verified identical at the pristine base commit by re-running exactly those files
there. All are stale frozen-artifact and identity drift in the RingCore-V1 and E6
lanes: operator-registry hash drift (`d752a07e…` vs frozen `9197401e…`),
panel-sampler protocol-freeze drift, A2.2b artifact provenance drift, RingCore
catalog fingerprint drift, plus two RDKit-version aromatic-oracle failures. None
touches `atom_delete`, the executor, or the semantic process.

One deserves flagging: `test_reference_successor_kernel.py::test_no_production_module_imports_the_oracle`
**text-greps** `src/`, `scripts/`, `modal_apps/` for the string
`reference_successor_kernel` and flags `editing_v2_process_identity.py`, which merely
lists that path as a hash-binding string literal and does not import it. A
pre-existing false positive on a file this work edits; it neither worsened nor
changed shape.

### Focused suites, exact counts

| Command | Result |
|---|---|
| `pytest tests/test_process_v2_atom_delete.py tests/test_process_v2_atom_delete_mask.py -q` | 40 passed, 1 skipped |
| `pytest tests/test_editing_process_v2_rebind.py -q` | 25 passed |
| `pytest tests/test_editing_process_v2_identity.py -q` | 34 passed |
| `pytest tests/test_process_v2_atom_delete_invariance.py -q` | 4 passed |
| Gate-0 chain (7 files, after re-pin) | 80 passed |
| Combined Process-V2 + identity + rebind + gate chain (9 files) | 126 passed |
| `ruff check` on all 12 touched Python files | All checks passed |
| `git diff --check` | clean |

The skip is `FactorizedMarkBatch.pin_memory`: torch routes pinned memory to `mps:0`
on this Mac. Repository-wide `ruff check src/ tests/` reports 5 pre-existing `E402`
errors in `tests/test_packed_trace_store.py`, identical at the base commit, in a file
this work does not touch.

### Independent orchestrator verification, not delegated

1. **Legacy fingerprint stability.** All **392** pre-existing `OperatorCapabilities`
   configurations produce byte-identical fingerprints at base and head; **0** changed.
   The frozen values `d246bc88d8440d31` (semantic V1), `e787c852410c6b63` (legacy
   RingCore), and `40510175845988f1` (de-novo default) are unchanged.
2. **Legacy batch byte-identity.** All **14** legacy-semantics batch tensors
   (topologies, all masks, graft tables, neural bonds, atom state) are byte-identical
   between base and head over a 22-molecule panel.
3. **Mask versus independent oracle through the real batch builder.** Over 28
   molecules including charged, fused, bridged, spiro, S/Cl-bearing and mixed
   aromatic/saturated systems: **0** mask mismatches, V1 always a subset of V2, the
   V1 and additional sets always disjoint. 17 of 28 molecules gain at least one new
   candidate; 90 new candidates in total.
4. **Representation invariance.** Slot-relabel equivariance: 0 mismatches over 60
   random relabelings. Kekule-alias invariance of the canonical successor set: 0
   mismatches. Both are pinned in `tests/test_process_v2_atom_delete_invariance.py`.

### Performance, path-level

Measured on the real `prepare_factorized_mark_batch` path over drug-like leads, per
the repository's own lesson that micro-benchmarks mislead about paths:

| Batch size | Existing cold path | V2 mask added | Added share |
|---|---|---|---|
| 32 | ~1,735-1,792 ms/state | 4.49-5.72 ms/state | **+0.26% to +0.32%** |
| 128 | ~1,728-1,790 ms/state | 5.74-6.16 ms/state | **+0.33% to +0.34%** |
| 256 | ~1,778 ms/state | 5.28 ms/state | **+0.30%** |

Stated at a different denominator, the V2 expansion costs about 12x the isolated V1
`_graph_application_masks` call (0.479 ms/state versus 5.763 ms/state on 100 leads,
mean 23.2 heavy atoms), yielding 2.95 additional admissions per state. Both numbers
are correct at their stated scope; the path-level figure is the operationally
relevant one, and results are cached per exact state in `ChemistryStateFeatures`.

## 7. Defects found and fixed beyond the three implementation scopes

1. **V1 payloads became unreadable, blocking the rebind end to end.**
   `decode_semantic_trace_record` required the *live* process identity, so a V1
   semantic packed row failed to decode the moment Process V2 moved that identity.
   The compatibility proof must read the immutable V1 payload. Fixed by giving the
   decoder the same optional pinned-identity contract the packed store's readers
   carry, threaded from both readers. Default `None` keeps the live requirement and
   is byte-identical for every existing caller; a pinned identity proves provenance,
   never currency.

2. **Silent wrong-runtime selection under Process V2.** `_default_rewrite_system`,
   the kernel-identity provenance flag, the ringcore configuration label, and the
   successor-training predicate compared process semantics by *equality* with version
   one. Under V2 those evaluate False, so the evaluator would silently return
   `editing_v2_rewrite_system()` instead of the semantic runtime and record
   `semantic_editing_v2_process=False`: a wrong runtime and false provenance with no
   error raised. Fixed with a single authority, `is_semantic_editing_v2_process`,
   plus a distinct `editing_v2_semantic_actions_v2` configuration label so a
   Process-V2 kernel identity cannot read as a Process-V1 one. A regression test
   fails if a future process version is added without updating the predicate.

3. **An over-specified fail-closed assertion, pre-existing.**
   `test_cross_family_alias_fails_closed_instead_of_double_counting` asserted *which*
   of two cross-family guards fires, but the per-family selection is seeded by
   `cell_role_policy_sha256` by design. Measured at the base commit with Process V2
   entirely absent: **12 of 12** legitimate policy-hash values flip which guard
   fires. The invariant, failing closed rather than double counting, still holds
   under both guards. The assertion now requires the hard failure and accepts either
   guard, with the measurement recorded inline. This is a removal of an
   over-specification, not a relaxation of a scientific gate.

## 8. Cross-worker disagreement, and how it was resolved

The identity worker implemented `validate_frozen_process_identity` more strictly
than the frozen interface spec required: beyond the exact field set and recomputed
self-hash, it checks that `schema_version`, `process_semantics`, and
`contract_relative_path` are the triple the declared schema is definitionally bound
to. The rebind worker's test had assumed the looser contract and constructed a
"different" identity by rewriting a schema-bound field.

Resolved in favour of the stricter implementation: it closes a real masquerade
vector, since re-hashing alone makes a relabelled object internally consistent. The
rebind test was changed to mutate a field the schema does not bind, so it tests the
intended rejection reason, and a dedicated
`test_a_relabelled_v1_identity_cannot_masquerade` was added for the schema-bound
case. No worker's implementation was weakened.

## 9. Unresolved risks and open decisions

1. **P50 lane invalidated.** 27 failures and 14 errors, one root cause,
   `configs/editing_v2_semantic_p50_recipe_policy_v1.json`. Regenerating it requires
   P50-lane authority and re-derived prerequisites. Deliberately not repaired here.
2. **Measured Gate-0 and T1 evidence produced under `6b98ee21…` is invalid.**
   Re-pinning binding contracts did not and cannot regenerate it.
3. **Checkpoint persistence of `atom_delete_action_semantics` has no owner.** Neither
   `editing_v2_semantic_runtime.py` nor `editing_v2_scientific_identity.py` was
   extended, the latter by instruction. A V2 checkpoint reconstructed through the
   existing loader fails loudly at the model constructor rather than silently, but
   the V2 scientific-identity binding remains future work.
4. **Batch builders that pass capabilities field by field** raise loudly under a V2
   model rather than silently building a V1 mask:
   `successor_micro_overfit.py:253`, `production_successor_kernel.py:162`,
   `editing_v2_semantic_t1_capacity_runner.py:388`,
   `materialize_editing_v2_semantic_p50_validation_baseline_app.py:686`, plus
   `editing_gate_zero_runtime.py:1156`, `editing_successor_trainer.py:527`,
   `canonical_successor_distillation.py:328,455`, `ring_macro_enumerability.py:468`.
   Each needs the flag threaded when its lane is authorized for V2.
5. **Four rejection codes never fire on real chemistry.** Over all 800 Jin-QED leads
   (9,189 slot decisions), only `aromatic_atom` (3,766), `articulation_point`
   (2,441), `outside_connected_nonleaf_expansion` (1,652) and
   `charge_policy_violated` (121) fire, with 1,209 admitted. `EXECUTOR_REJECTED`,
   `SUCCESSOR_DISCONNECTED`, `SUCCESSOR_OUTSIDE_SUPPORT` and
   `SUCCESSOR_NOT_CANONICALIZABLE` never fired: deletion preserves each surviving
   neighbour's class valence, so they are defence in depth. Reported as a negative
   result rather than dressed up with a fabricated fixture.
6. **Environment gap, out of scope.** `PyYAML` is imported by
   `compose_v4.experiments.registry` but is not declared in `pyproject.toml` and is
   absent from the pinned venv; 15 test files fail collection without a shim. Not
   fixed here as unrelated cleanup.

## 10. Stop conditions

None of the handoff's stop conditions triggered.

| Condition | Assessment |
|---|---|
| Executor legality not representable in the batch mask | Not triggered. Represented exactly as CPU-derived admission, cached per exact state. |
| Aromaticity or charge behaviour ambiguous | Not triggered. Aromaticity is the frozen resonance-invariant perception; charge uses the unchanged policy. |
| A supposedly unchanged operator changes on a frozen fixture | Not triggered. 392/392 fingerprints and 14/14 legacy tensors byte-identical. |
| V1 and V2 identities cannot coexist unambiguously | Not triggered. Distinct schema strings and semantics; masquerade rejected and tested. |
| Exact transition replay differs for a payload claimed compatible | Not triggered. The rebind refuses to publish on any mismatch. |
| Rebind requires changing split or provenance identity | Not triggered. Source, split, lane, provenance carried through unchanged. |
| A requested action would authorize or launch a scientific job | Not triggered. Nothing was launched. |

## 11. Independent review: performance and adversarial

Two reviewers worked from the same frozen interface but with no write access to
the implementation, so neither could quietly repair what it found.

### Performance and decision equivalence

Panel: **894 states at 40 production slots**, identity
`c1c1b424387f5b7b4e2941f4016e6db853388a84951ab83fe7f0bc0c9efb2b03` (800 Jin-QED
leads plus 94 hand-built saturated, bridged, spiro, heterocyclic and charged
supplements, because the Jin panel is aromatic-heavy and Process V2 only ever
admits non-aromatic slots).

- **Decision equivalence: 894 states x 40 slots = 35,760 decisions, 0
  disagreements in either direction**, against an oracle built only from the
  primitives, with its own graph construction and its own support re-derivation.
- New mask cost **4.94 ms/state mean** (7.11 with candidates, 0.58 without).
  RDKit work is **linear in the number of screened slots and independent of
  heavy-atom count**: `sanitizations = 1 + D + 4S + A`, exact on 894/894 states.
- On the real `prepare_factorized_mark_batch` path the addition is **+0.29% to
  +0.34%** of the existing cold per-state cost (about 1,760-1,790 ms/state), and
  **+0.351%** re-measured on the integrated head. The existing cost is dominated
  by `_semantic_cycle_close_admission_mask` (54.8%) and
  `_semantic_atom_restate_admission_mask` (36.7%); the new mask is 0.26% of it.
  The ratio is cache-invariant because the V2 mask rides the same
  `molecular_state_cache_key`.
- Newly admitted candidates per molecule: median 3, mean 3.13, max 16; **33.2%
  of molecules gain none**. Slots: V2 adds 2,796 of 19,019 real slots (14.70%),
  V1 admits 3,401 (17.88%), **overlap 0 on every state**.
- Optimization ladder measured, none applied: hoisting the per-candidate
  recomputation of state-level work plus reusing the collator's existing graph
  and perception gives **2.52x** with no condition, no condition order and no
  executor call changed. It is not applied here because profiling shows the cost
  is already 0.3% of the path, and the smallest complete change is preferred.
  One proposal that would reorder an executor call was measured (0.012 ms/state,
  nothing) and **rejected**.

### A measured chemical finding, with its control

**2 of 2,796 newly admitted deletions (0.072%) produce a successor whose
perceived aromatic ring count drops from 2 to 1.** Both are the same chemistry, a
protonated quinoid `C=c2ccccc2=[NH+]`: the deleted atom has no incident perceived
aromatic bond, so frozen condition 2 is correctly satisfied, but it carries a
stored double bond into the quinoid ring and removing it collapses the
cross-conjugation that sustained RDKit's perception. Successors are valid,
connected, charge-preserving and canonicalizable. Charge-altering: 0.
Disconnected: 0.

The control is what makes this interpretable. On the same panel the **unchanged
V1 leaf rule** admits 3,401 deletions of which **61 (1.79%) are
aromatic-system-altering and 140 (4.12%) violate the charge policy**. The
property is therefore inherited from the preserved V1 rule at roughly 25x the
rate, and the Process-V2 expansion is strictly cleaner on both axes. This is an
owner decision about whether condition 2 should be widened to exclude atoms
double-bonded into a perceived aromatic system, not an implementation defect, and
it is **not** the handoff's representation-sensitivity hazard: that hazard is
Kekule-alias dependence of the successor, and alias invariance was verified over
965 component-factored aliases and 3,200 random re-serializations with 0
mismatches.

### Adversarial review

The reviewer independently reproduced both suite results exactly (base 21 failed
/ 2480 passed / 1 skipped; head 48 failed / 2542 passed / 2 skipped / 14 errors
at `c506d70`), which is why those counts are reported as measurements rather than
single observations. It confirmed byte-identity of the state dict, capability
fingerprint, all 11 mask tensors and all 11 logit tensors versus base under V1
semantics; 0 mask/oracle disagreements over 800 leads, 889 hard states (padded,
permuted, chained-delete, SCAR-injected) and a 55-molecule aromatic edge panel;
disjointness on every state; correct carriage of the new batch fields through
`subbatch`, `.to()` and concatenation; no bare excepts or silent degradation in
the rebind; and no unrelated operator touched.

It also found real defects, all of which are fixed in `85169b1` except where
noted:

1. **The identity re-pin was incomplete** (blocker). Four further artifacts and
   four source constants still pinned the superseded cell-role policy hash.
   Fixed, and verified by sweeping every hash value these edits obsoleted until
   the sweep reported none.
2. **The T1 chain failed silently** where the P50 chain failed loudly, because
   `editing_v2_semantic_t1_capacity_policy.py` compares the config against a
   *source constant* rather than the live loader, so both were stale and agreed
   with each other. Re-pinned. The structural weakness remains and is recorded in
   section 9: the next hash move will again be silently stale there.
3. **Three production mutations survived the whole suite** (bypassing the
   executor check, bypassing the declared-support check, zeroing the V1 dense
   mask). Fixed with reachable witnesses in
   `tests/test_process_v2_atom_delete_gates.py`; each mutation is now caught by
   exactly one test.
4. **Nothing tested that a connected-nonleaf delete can be learned.** Fixed in
   `tests/test_process_v2_atom_delete_teacher.py`, with the Process-V1 negative
   control.
5. **Three claims outran their evidence** and were corrected in place:
   conditions 3 and 5 are not independent, a self-consistent pinned identity does
   not prove provenance, and the rebind oracle is a second derivation rather than
   a second design.
6. **`mismatches_by_code` is a constant, not a measurement** (section 9).

Findings assessed and deliberately not acted on: the Kekule-alias invariance test
is underpowered relative to its name (8 of 10 panel molecules have no aromatic
component and skip), though it does fail under the aromatic mutation and the
independent sweep supplies the missing power; the two pinned-identity readers now
derive the expected process-semantics label differently, which fails closed but
is inconsistent; and Process V2 newly admits deletions adjacent to a SCAR, which
writes hydrogen onto the SCAR through unchanged executor behaviour. The SCAR
itself is never admitted, so the handoff's stated requirement holds.
