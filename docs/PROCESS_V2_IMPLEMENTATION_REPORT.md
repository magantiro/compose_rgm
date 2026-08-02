# COMPOSE Process V2 connected-nonleaf atom deletion: implementation report

Branch `codex/editing-v2-process-v2-atom-delete`, base `bc05c6e`, head `c506d70`.
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
introduced. Note that `is_valid_atom_delete` is deliberately *not* a connectivity
predicate: it returns `True` for articulation-point deletions, so the articulation
and connectivity conditions are independently load-bearing.

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

All paths relative to the repository root. 23 files, +7,212 / -140.

### New (9)

| Path | Lines | Role |
|---|---|---|
| `src/compose_v4/rewrite/process_v2_atom_delete.py` | 323 | Phase 2 scientific core: resolver, enumerator, slot mask, 11 reason codes |
| `src/compose_v4/data/editing_process_v2_rebind.py` | 2133 | Phase 4 restart-safe, content-addressed, proof-bound V1-to-V2 rebind |
| `configs/editing_v2_semantic_process_v2.json` | 331 | Phase 1 self-hashed Process V2 semantic contract |
| `docs/EDITING_PROCESS_V2_DECISION.md` | 201 | Prospective decision record |
| `tests/test_process_v2_atom_delete_mask.py` | 920 | Model and batch mask battery |
| `tests/test_editing_process_v2_rebind.py` | 962 | Rebind battery |
| `tests/test_editing_process_v2_identity.py` | 618 | Identity, contract, masquerade battery |
| `tests/test_process_v2_atom_delete.py` | 404 | Resolver and executor-parity battery |
| `tests/test_process_v2_atom_delete_invariance.py` | 191 | Quotient invariance and cross-module assertions |

### Modified (14)

Source (8):
`src/compose_v4/rewrite/editing_v2_process_identity.py` (+605, coexisting V1/V2
identities), `src/compose_v4/model/factorized_tracelet_rate_model.py` (+222, mask,
capability, batch plumbing), `src/compose_v4/data/semantic_packed_trace_store.py`
(+129, pinned-identity readers), `src/compose_v4/rewrite/trace_shard_v3.py` (+41,
pinned-identity decode), `src/compose_v4/experiments/factorized_mark_conditional.py`
(+15, collation), `src/compose_v4/experiments/production_successor_kernel.py` (+14,
runtime selection and provenance),
`src/compose_v4/experiments/factorized_successor_training.py` (+4, predicate),
`src/compose_v4/experiments/editing_v2_semantic_gate_zero.py` (+2, source-pinned
frozen contract hash).

Configs, hash pointers only (5): `editing_gate_zero_semantic_model_process_v1.json`,
`editing_v2_semantic_capability_cells_v1.json`,
`editing_v2_semantic_development_cell_roles_v1.json`,
`editing_v2_semantic_active8_decision_runtime_v1.json`,
`editing_v2_semantic_gate_zero_structural_v1.json`.

Test (1): `tests/test_editing_v2_semantic_t1_panel_cache.py` (+13, see section 7).

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

## 5. Identities

Naming: `editing_v2` is the source-conditioned editing **lane**; the trailing `_v1`
/ `_v2` is the **semantic process version** of that lane.

### New

| Object | SHA-256 |
|---|---|
| Process V2 identity (`semantic_editing_v2_v2`) | `05a6cb0e285ff2bb5cc30f66786ca6948d6835e8f4cfd4ecfd738cc6f652e477` |
| V2 contract, semantic self-hash | `bfbaf9dfb700f90ae6ed863139a6fe552aa4d36b355cb3334cd7e2df4ae282f1` |
| V2 contract, physical file hash | `042f5d3c759e2adf4ce487afa2e856e2bbbf3a5c941f81ed47e26a23d1a1e584` |

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
| head `c506d70` | **48 failed, 2542 passed, 2 skipped, 14 errors** (898.56 s) |

Delta attribution, by node-ID set difference: 21 base failures still failing, **0
base failures newly fixed**, +27 failures and +14 errors, **all confined to the P50
lane** (`test_editing_v2_semantic_p50_validation_baseline.py` 14 errors,
`p50_successor_cache.py` 11, `p50_runner.py` 11, `p50_recipe_stream.py` 5) with a
single root cause: `semantic P50 recipe policy identity or supported projection
disagrees`, from `configs/editing_v2_semantic_p50_recipe_policy_v1.json` pinning the
moved identity chain. This is the declared downstream invalidation of a stage this
task is explicitly forbidden to authorize; repairing it requires P50-lane authority
and re-derived prerequisites, so it is reported rather than re-pinned.

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
