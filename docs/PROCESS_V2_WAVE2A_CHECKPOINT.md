# Process-V2 Wave 2A: compatible evidence boundary and cache-fed rebind

Checkpoint report for Codex review. **Nothing remote ran.** No Modal job, corpus
materialization, Active8 decision, Gate 0, T1, P50, or training run was started
at any point, and every artifact this work publishes carries its authority
fields explicitly false.

## 1. Branch, base, head, worktrees

| | |
|---|---|
| branch | `codex/editing-v2-process-v2-wave2a-compatible-chain` |
| base | `dbfe1e19c314cb55d9516b0657397e52d9f7c902` |
| last code commit | `422221d` (this report is the commit after it; a report that
names its own sha invalidates itself on amend) |
| integration worktree | `/private/tmp/compose-process-v2-wave2a-compatible-chain` |

Implementation ran as four owned workstreams in isolated worktrees, integrated
by cherry-pick with every diff inspected:

| worktree | branch | owner | concern |
|---|---|---|---|
| `/private/tmp/w2a-agent-a` | `w2a/agent-a` | A | admitted source, binding, Active8 source, contracts |
| `/private/tmp/w2a-agent-b` | `w2a/agent-b` | B | chunk cache, target-only reader, cache-fed rebind |
| `/private/tmp/w2a-agent-c` | `w2a/agent-c` | C | chain verifier |
| `/private/tmp/compose-process-v2-wave2a-basegate` | detached | -- | base gate, never written |
| `/private/tmp/w2a-ac-suite` | detached | -- | hash-sensitive suite snapshot, never written |

The last two exist because §8 requires every repository-wide or hash-sensitive
command to run in a tree nothing else writes to. That is not a formality here:
the `dbfe1e1` diagnostic proved the P50 implementation freshness check re-reads
the live tree, so an editor write landing mid-run creates a real stale-plan
refusal. Snapshot worktrees make the gate independent of what the agents are
doing.

## 1b. Commit inventory (36 code commits, plus this report)

| sha | subject |
|---|---|
| `4239894` | data: refuse a granted authority field under any name at any depth |
| `6652d39` | data: version the resolved evidence binding at 2 |
| `1902f9e` | data: scope the vocabulary-free authority guard to non-granting artifacts |
| `f70578a` | data: make the admitted-source identity one exact validated shape |
| `f4baef1` | experiments: rebuild the process v2 chain at schema 3 with cumulative lineage |
| `ed024b1` | experiments: embed the real admitted source in the evidence binding |
| `27052b0` | data: version and validate the process v2 active8 source identity |
| `32e74d1` | experiments: rename the census bounded-p50 decision to the frozen spelling |
| `58b97c7` | tests: close the gaps a mutation run found in the new validators |
| `b000169` | experiments: retract the identity claim and expose the lineage generations |
| `419d12f` | tests: refuse a second self-hash candidate in the admitted-source identity |
| `a4c095a` | tests: prove the version-one binding fixture fails on its fields, not its envelope |
| `8aec677` | scripts: prove each process v2 root and refuse a vanishing declaration |
| `0f7e351` | tests: pin process v2 root validation and the unchecked pin inventory |
| `1bb3c7f` | tests: attack the process v2 pointer, identity, path and receipt rules |
| `b86e074` | tests: select the chain shapes these attacks need structurally |
| `394fc99` | tests: follow the cumulative lineage shape in the chain attacks |
| `bd89c18` | editing_v2_process_v2_chunk_cache: bind a narrow revision and read one target chunk |
| `3e95571` | editing_process_v2_rebind: prove one verified cache chunk per task |
| `fe865a7` | scripts: benchmark process v2 chunk sizes on a production-shaped source |
| `e46d29a` | tests: close the guards a mutation sweep found unprotected |
| `132ad2c` | tests: assert the whole authority vocabulary on the resolved source |
| `a5ceb35` | tests: drop the trailing blank lines at the active8 source file end |
| `a249e1e` | tests: refuse a granted authority field nested inside a tuple |
| `0bad0f3` | editing_v2_process_v2_evidence_binding: refuse a field the declared shape does not carry |
| `b1b0420` | editing_v2_process_v2_contract_chain: refuse a lineage value that is live on disk |
| `c38f27f` | tests: pin the twelve identity guards a mutation run found untested |
| `1140630` | verify_process_v2_chain: refuse a substituted root validator |
| `e92d5a7` | verify_process_v2_chain: hold a stage receipt to what it can do |
| `98b045b` | verify_process_v2_chain: enforce the normal-form rule it documents |
| `9750775` | verify_process_v2_chain: defer the authority scan to its owning module |

Five further commits followed the inventory above, after the second adversarial
pass: two of Agent B's later commits that the first integration missed, the
cache-fed run test, the decode-path closure fix (§10), and the learnings entry.
The review fixes are described in §8b and §10.

## 2. Environment

```
python 3.14.2   torch 2.11.0   PyYAML 6.0.3   pytest 9.0.3   ruff 0.15.22
interpreter  compose_rgm_claude_generators/.venv/bin/python
guards       KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src
```

The sibling `compose_rgm/.venv` is NOT compliant: it lacks PyYAML and produces
15 phantom collection errors. Every number in this report used the interpreter
above.

## 3. The frozen boundary held

Both scientific identities were recomputed live at every integration step and
are unchanged:

| role | sha256 | |
|---|---|---|
| V1 `semantic_editing_v2_v1` | `6c4721f0dd37132aae657e7aa5f1bfc01cef270662f228171c4587eb7dd48491` | unchanged |
| V2 `semantic_editing_v2_v2` | `0c938177a34819e6e828920c1f66e240c6eb251fe7c9ea6cfe6757829dceb2dd` | unchanged |

`src/compose_v4/rewrite/editing_v2_process_identity.py` and
`configs/editing_v2_semantic_process_v2.json` are byte-identical to base
(`git diff dbfe1e1 --` empty; physical hashes `f45592f3…` and `b5fa06d7…`).

### A stale claim in the inherited documentation, corrected

`docs/HANDOFF_PROCESS_V2_EVIDENCE_CHAIN.md` §7 and the evidence-binding module
docstring both asserted that renaming the adapter's authority field **would move
the scientific Process-V2 identity**. That claim is false, and it mattered:
taken at face value it makes §3.1 impossible, because §3.1 requires the admitted
source to build its authority mapping through `authority_false_block()`, which
means dropping the retired `p50_authorized` spelling.

Measured: the V2 identity hashes exactly 19 files -- the 18 V1 implementation
files plus `process_v2_atom_delete.py` -- together with the contract and the
action codec. `editing_process_v2_admitted_source.py` is not among them.

What the rename **does** move is `adapter_implementation_sha256`, because
`_adapter_implementation_sha256` hashes the adapter's own file bytes. That is
expected and is required by §3.1. The two identities were conflated. The module
docstring now carries the retraction in place; the handoff document still
carries the stale claim and is not ours to edit.

## 4. Baseline and comparison

Base `dbfe1e1`, full suite in an isolated worktree:

```
16 failed, 3156 passed, 1 skipped, 1 xfailed, 0 errors in 1047.17s
```

Cross-check: the `dbfe1e1` P50 diagnostic independently measured `16 failed,
2980 passed` at `b39420c`. dbfe1e1 = b39420c + `dbf1367` (166 cases) +
`e4d78cf` (10 tests) + one doc, and 2980 + 166 + 10 = 3156 exactly.

The 16 pre-existing failures, grouped by cause:

- **9** RingCore catalog fingerprint drift (`test_ringcore_validation_panel_builder`
  x6, `test_ring_core_identity_gate` x2, `test_ring_core_zero_mixture` x1). An
  environment fact on this machine, recorded in `learnings.md` 2026-08-02:
  `build_production_ringcore_catalog` reconstructs `82fd910cafe2eeb7` against
  the frozen `639ff6078c32d43c`.
- **6** E6 frozen-artifact staleness (`test_e6_a2_3_readiness` x5,
  `test_e6_graph_audit` x1).
- **1** import-boundary violation, below.

### The import-boundary failure is structurally unfixable under §1.1

`test_reference_successor_kernel.py::test_no_production_module_imports_the_oracle`
fails because `src/compose_v4/rewrite/editing_v2_process_identity.py` imports the
reference successor kernel, which production code is forbidden to import. That
file is frozen byte-for-byte by §1.1 **and** is one of the 19 hashed into both
scientific identities, so removing the import moves V1 and V2. The violation is
pinned in place until an owner decides to move the identity. **Reported, not
fixed. This is an owner decision.**

Repository-wide Ruff at base: **64** errors (E402 x30, E702 x28, F401 x3,
F841 x2, E741 x1).

## 5. Schema and identity transitions

| artifact | before | after |
|---|---|---|
| admitted-source identity | v2 | **v3** |
| `physical_execution_identity` | echoed parent version | own schema, **v1** |
| resolved evidence binding | v1 | **v2** |
| Active8 source identity | v1 | **v2** |
| contract chain | v2 | **v3** |
| contract revision | `process_v2_explicit_dependency_graph` | `process_v2_binding_v2_and_cumulative_design_lineage` |
| semantic census output | v2 | **v3** |
| adapter implementation | `d5b099eb…` | `bed33ca5…` |
| chain generation id | -- | `bd446d878058a38a8e2bacc87df0a073aef022cef7641496778156518300675c` |

Seven prospective contract `contract_sha256`, before -> after: runtime
`8b829cb3…`->`fc0bc106…`, cells `b9ef14f3…`->`3891411d…`, roles
`4c8f9d5c…`->`a88086e6…`, gate-0 `06ac0425…`->`b7d663eb…`, T1 panel
`8cc3c39d…`->`f6a305f2…`, T1 capacity `17569116…`->`913b9bd8…`, P50
`b27e28a0…`->`885f2b13…`.

All seven rebuild **twice byte-identically** and match the committed bytes.

## 6. Disposition of the eleven confirmed defects

| # | defect | disposition |
|---|---|---|
| 1 | a real `identity()` cannot build the evidence binding | **fixed** -- the fabricated projection is deleted, not translated |
| 2 | the green binding tests use a synthetic restatement | **fixed** -- the real descriptor is driven end to end |
| 3 | `_require_no_granted_authority()` claims recursion, inspects top level only | **fixed** in the shared module, applied by the binding |
| 4 | the v3 owning validator needs exact-shape, nested-identity, census and recursive-authority coverage | **fixed** |
| 5 | the chunk plan binds no behaviour-affecting source code | **fixed** -- an owner-computed narrow revision |
| 6 | a resealed nested source-revision mutation leaves the cache identity unchanged | **fixed** |
| 7 | a resealed completion with a wrong chunk count passes | **fixed** |
| 8 | reading one chunk validates and hashes its siblings | **fixed** -- see below |
| 9 | cache records are returned as generic JSON | **fixed** -- `SemanticPackedRowRead` |
| 10 | the rebind launcher accepts a caller-supplied raw V1 payload root | **fixed** by deletion |
| 11 | malformed pointers, insufficient root validation, incomplete identity matching, path containment, target schema, unresolved deferrals | **fixed** -- four of these six were actively broken, not merely absent |

### Defect 8 was worse than reported, and the fix is structural

The handoff described sibling validation as quadratic across workers. Measured,
it was worse within a single reader: `read_process_v2_chunk_records` called
`validate_process_v2_chunk_cache_source`, which `_sha256_file`d **every** chunk
in the manifest plus an `iterdir`; and `iter_process_v2_chunk_cache_records`
then called that per chunk. That is **(N+1)xN full chunk-file hashes to read one
source once** -- at production scale (~32.3k entries per source,
`records_per_chunk=2048`, N~16-17) about 290 full gzip reads to read 17 chunks.
It reinstated at the read boundary exactly the triangular rescan the module's
own docstring was written to remove.

The fix is verifiable from the call graph rather than from a benchmark: the new
manifest validator opens no file at all, the whole-directory hash loop and
`iterdir` moved to the publication boundary, and the target read path hashes
exactly one file -- the target chunk.

### Defect 11: four of the six were live, not missing

Malformed pointers and identity edges **vanished** rather than failing (a
`frozenset != POINTER_FIELDS` then `continue`, the same defect class Wave 1
fixed in one place and left in another). Identity matching keyed on three fields
with `str()` coercion, so `"1"` matched `1`. And an unresolved `DEFERRED` edge
printed `AGREES` -- an undecided delegation was satisfying the gate. Only path
containment and `target_schema` comparison were simply absent.

## 7. Findings reported rather than fixed

These are owner decisions or out-of-scope observations. None is fixed here.

1. **The frozen module violates the test-oracle import boundary.**
   `editing_v2_process_identity.py` imports the reference successor kernel.
   Fixing it moves both scientific identities. §4 above.
2. **`editing_gate_zero_semantic_model_process_v2.json:.process_identity_sha256`
   is an unchecked pin.** Making it a checked edge would require inferring a pin
   from a field-name convention, which is exactly what typed pointers replace and
   exactly the Wave-1 verifier defect. It is listed by name in the verifier's
   `unchecked pins -- STATED, NOT VERIFIED` inventory instead. The real fix gives
   that artifact a typed pointer, which regenerates a frozen artifact.
3. **`authority_envelope()` publishes `cache_materialized: True`**
   (`build_process_v2_chunk_cache_app.py`). It is not an `_authorized` name, so
   neither authority guard sees it. Expanding the authority vocabulary is an
   owner decision.
4. **`docs/HANDOFF_PROCESS_V2_EVIDENCE_CHAIN.md` §7 still carries the retracted
   identity claim** corrected in §3 above. The module docstring is fixed; the
   document is not ours to edit.
5. **A retraction of my own.** I initially recorded `p500_authorized` as an
   unmanaged fourth authority spelling. That was wrong: it is declared in the
   `p50_recipe_body` projection of the Process-V2 policy registry and enforced by
   the P50 execution contract, which refuses unless a T1 decision grants
   `bounded_p50_authorized` and denies `p500_authorized`. Its absence from
   `AUTHORITY_FIELDS` is correct scoping, because that tuple is the vocabulary of
   artifacts that must never grant while `p500_authorized` belongs to stage
   decisions, where a field is legitimately true. **Nothing to fix.** The residue
   is a usage constraint now recorded at the guard's definition: apply
   `require_no_granted_authority` only to non-granting artifacts, never to a
   stage decision or permit.
6. **The resolved item.** `docs/HANDOFF_PROCESS_V2_EVIDENCE_CHAIN.md` §9 item 5
   -- the two P50 authority spellings a consumer could grep past each other on --
   is closed by this wave. Outside the frozen process contract there is now one
   spelling, and the retired one is refused structurally rather than by
   convention.

## 8. Verification

Every command below ran under the §2 interpreter and guards. Repository-wide and
hash-sensitive commands ran in a snapshot worktree nothing else writes to.

| gate | result |
|---|---|
| repository-wide Ruff | **64 = base 64**, none from a touched file |
| Ruff on every touched file | clean |
| `git diff --check` across the branch | clean (one trailing-blank-line defect found and fixed) |
| seven contracts rebuilt twice, two separate processes | identical generation `bd446d87…` |
| chain verifier | `AGREES`, 0 findings |
| V1 / V2 scientific identities | unchanged |
| frozen file bytes | identical to base |

### Full suite, compared by exact failure set

Pre-fix integrated head (`132ad2c`, A+B+C):

```
17 failed, 3331 passed, 1 skipped, 1 xfailed   vs base  16 failed, 3156 passed
regressions: 1 (adjudicated below)   base failures reintroduced: 0   fixed-then-broken: 0
+175 passing
```

Final head (36 commits, all review fixes integrated), run alone on an unloaded
machine:

```
16 failed, 3407 passed, 1 skipped, 1 xfailed   in 1160.49s
regressions: 0    base failures reintroduced: 0    fixed-then-broken: 0
+251 passing; the failure SET is IDENTICAL to base
```

The phantom regression below does not appear in this run, which is the simplest
possible confirmation of its diagnosis: same code, unloaded machine, gone. That
run also took 19:20 rather than 25:33.

### The one apparent regression was a load artifact, proven three ways

`test_training_support_cache.py::test_training_support_cache_waits_for_certificate_complete_replacement`
appeared on head and not on base. It is not a regression:

1. it passes **3/3 in isolation** at head;
2. it carries a **1.0 s wait timeout**, and that suite ran **25:33 against base's
   17:27** -- 47% slower, because two fix agents were running their own suites on
   the same machine;
3. decisively, its **import closure is 33 `compose_v4` modules and its overlap
   with the eight source files this wave changed is empty**, so it cannot reach
   any changed code and a regression is structurally impossible.

Point 3 is the one that settles it. Passing in isolation would have been weak
evidence alone, because a regression can be load-dependent; an empty import
overlap cannot be.

**This is the second instance of one hazard and it belongs in the record.** The
`dbfe1e1` P50 diagnostic established that running a gate while anything else
writes to or loads the machine produces failures that are real refusals of an
unreal condition. Here the write hazard was avoided (snapshot worktrees) but the
LOAD hazard was not, and it manufactured a phantom regression that cost real
adjudication time. Parallelism is free during implementation and is not free
during measurement: the final suite must run alone.

## 8b. Independent adversarial review, and what it changed

Two reviewers ran read-only against the integrated branch, mutating only inside
throwaway copies. They did not merely confirm the implementers -- they overturned
four claims, three of which had already been reported to the owner as settled.

**A methodological finding that invalidates naive mutation harnesses, recorded
because it will recur.** A plain `cp -a` copy is not a git repository, so
`_git_show`/`_revision_available` fail and **six chain tests silently SKIP** --
including the two that anchor lineage to `git show`. Any mutation run over a bare
copy therefore reports every lineage mutation as killed. Read the SKIP count;
this repository's own learnings already say so, and it caught a reviewer here.

### Severe, found and fixed

**A substituted root validator could manufacture agreement.** `validators={}`
correctly yields `UNVERIFIED`, but a populated mapping of no-op callables did
not: over a governed root tampered at the graph leaf, the verifier returned
`AGREES` while real validators returned `DISAGREES`, and it silently absorbed
that root's unchecked pins because `proven.add(name)` ran whether or not the
validator proved anything. This re-opened exactly the hole root validation
exists to close. Reproduced by the integration lead against the fix: now
`INCONCLUSIVE`, so an injected validator can make a run weaker and never
stronger.

**The stage-receipt path accepted receipts it must refuse.** A receipt is the
only object that converts a blocking `UNVERIFIED` into agreement, so it is
launch-permit shaped. It accepted a receipt granting `p500_authorized: True`
(the vocabulary-aware guard was applied, the vocabulary-free one was not), a
receipt whose own `status` said it verified nothing, one whose `verified_by` was
never compared to anything, and one whose path was absolute and outside the
checkout. Five shape mutations also survived.

**The evidence binding accepted arbitrary extra top-level fields.** Both sibling
validators carried an exact-field check; the binding did not, in the one artifact
that carries measured provenance downstream and whose own docstring names
smuggled measured evidence as failure mode #1. Fixed and mutation-proven: removing
the check fails `test_a_smuggled_measured_evidence_block_is_refused`.

### Claims overturned

- **"`validators=` cannot weaken a run to `AGREES`."** True only of the empty
  mapping. See above.
- **"A receipt is refused if it grants authority."** True only for the
  seven-name vocabulary.
- **"A non-normalized target that resolves inside is refused."** True only for
  the `..` spelling; `.`, `//`, `./` and a trailing `/` were all accepted, and
  each is read and re-checked under a second graph name.
- **"No lineage physical-hash check can be claimed, because it is
  unfalsifiable."** Correct about the self-referential formulation actually
  written, and overstated as a general claim: taking the expectation from
  **on-disk bytes** makes the guard falsifiable, it catches a cross-artifact case
  the existing `target` check misses, and the sibling verifier already implements
  exactly that check. The implementer subsequently wrote the guard rather than
  only narrowing the claim, so both formulations now exist at their proper
  layers.
- **"30/30 mutations killed."** Honest but narrow. An independent set of 60 left
  22 alive, and one of the original 30 was compound. The survivors were working
  guards with no test coverage, not defects -- four were confirmed uniquely
  load-bearing and firing.

## 9. The three riskiest areas for independent Codex review

**1. The completeness of `CACHE_IMPLEMENTATION_FILES`.** The narrow cache
revision is only as good as its file list. The whole §4.1 guarantee -- that a
caller-supplied revision is never authoritative because the owner recomputes it
-- rests on that tuple naming *every* behaviour-affecting module. A behaviour
change in a module outside the list silently keeps the same cache identity, which
is the failure the revision exists to prevent, and it would not show up as a test
failure. This is the single assumption in the wave with the widest blast radius:
it decides whether two artifacts that claim the same identity really were
produced by the same code.

**2. The raw-vs-cache equivalence tests, specifically whether the oracle side is
genuinely raw.** The defect this wave exists to fix (defect 2) was invisible for
exactly one reason: no test drove the production descriptor, so a divergence
between what the binding required and what the adapter emitted survived a green
suite. The same shape is available here. If both sides of an equivalence
assertion end up computed through the new cache path, the equivalence is vacuous
and would still be green. The oracle side must actually read raw packed shards
through `read_v1_records_through_the_bounded_raw_oracle`.

**3. The near-miss threshold in the chain verifier.** A pointer-shaped node is
FAILED as malformed at `_NEAR_MISS_KEY_EDITS = 2` key edits. That is a heuristic
with a measured margin: the closest legitimate non-edge in the committed graph
sits at 3 edits, and widening the bound to 3 fails a frozen artifact. The margin
is asserted, so it fails loudly rather than silently, but a future artifact
landing in the gap changes a correct structure into a reported defect. The
threshold is a judgement call and should be reviewed as one.

### Two residuals worth carrying forward

- **Chain-verifier diagnosis is iterative, not single-pass.** When the
  identity-edge rule refuses a root, its owning validator is skipped, so a root
  carrying both an identity-edge defect and an independent validator defect
  reports only the first. Soundness is preserved -- the run already fails, and a
  guard turns any skip-without-block into `UNVERIFIED`, verified by forcing the
  skip for every root, which kills 19 tests. But one green run after one fix does
  not mean everything was checked.
- **The cache-fed rebind relocates content-addressed rebind artifacts.** This is
  the approved operational identity move of §1.2, not a scientific one. The
  scientific V1 and V2 identities are unchanged and proven so.

## 10. A second adversarial pass on the cache found one more severe defect

Reviewed independently after the first pass: 81 mutations, 43 survivors, plus a
constructive escape. **It confirmed §9 risk 1 as a measured fact rather than a
hypothesis, which is the finding of this wave.**

**The cache identity did not bind the code that READS the cache.**
`CACHE_IMPLEMENTATION_FILES` covered what the cache stores, never what it decodes,
while the module docstring claimed "stores, reads or validates". Measured: two
trees differing only in `rewrite/progress.py`, reading the SAME committed cache,
produced the identical `cache_implementation_sha256` and DIFFERENT decoded rows.

**And its guard could not fail.** It asserted
`set(revision["implementation_files"]) == set(CACHE_IMPLEMENTATION_FILES)`, where
the left side is built from the right. Deleting `editing_v2_process_v2_chunk_cache
.py` from its OWN file list left all 217 tests green.

Fixed. The closure is now derived by EXECUTION, not by reading imports: modules
running under `sys.settrace` during a real decode, plus every type reachable in
the decoded object graph including MROs, plus every payload class the frozen codec
surface can construct. 9 -> 18 modules. The replacement guard loops over the
derived closure, edits each module on a tree copy and requires the identity to
move; it never reads the constant, so shortening the constant fails it.

A hand-written module list was wrong in BOTH directions -- it named
`action_codec_v2.py` (does not exist; `action_codec.py` IS v2) and
`provenance_overlay.py` (not in the closure), and missed `molecular_graph`,
`state`, `operators`, `tracelets` and `editing_v2_process_identity`. Nothing
pinned the old revision value, so no downstream re-pin was needed.

### Reported, not fixed, from this pass

- **43 of 81 mutations survive** on the cache and rebind modules. These are
  working guards with no test coverage, not defects; four were forged and
  confirmed load-bearing. Catalogue in the agent artifacts.
- **§4.5 is INCOMPLETE.** The benchmark script exists and honestly measures the
  real path, but **no report was produced** in `diagnostics/` or git, so the
  chunk-size selection was never recorded. Its memory criterion also cannot
  discriminate: `process_peak_rss_mb` is non-monotone across chunk sizes and is
  dominated by fixed interpreter footprint, so at 31x headroom both criteria pass
  at every size and the rule reduces to "pick the largest requested".
  `WORKER_MEMORY_MB` / `WORKER_TIMEOUT_SECONDS` carry a comment claiming they are
  read from the launcher; they are duplicated literals.
- **A correction to a claim in this report's own §6.**
  `validate_process_v2_chunk_cache_manifest` does not open "no file at all": it
  opens 9 repository sources, because it recomputes the narrow revision. The
  load-bearing claim holds and was independently measured -- under the artifact
  root a single target read touches exactly 2 files, the manifest and the target
  chunk, with ZERO siblings -- but the reader cannot run where the source tree is
  absent or relocated.
- The reducer hashes every published chunk exactly twice. Bounded to the reducer.
- `editing_process_v2_admitted_source.py` never references `source_geometry`, so
  §4.3's "cannot publish an authoritative artifact" currently holds only because
  every launcher opts into the guard. Pre-existing, untouched by this wave.

### Confirmed clean under attack

Raw-vs-cache equivalence is NOT vacuous: the oracle side genuinely reads raw
packed shards and is compared field-by-field including array bytes. No state is
reconstructed from SMILES. Raw reading has exactly one call site. Both resealed
mutations fail as required. The narrow revision is authoritative at every
boundary. Reduction is byte-identical at concurrency 1/20/40 with shuffled waves
and shuffled within-wave order. An incomplete reduction publishes no committed
generation.
