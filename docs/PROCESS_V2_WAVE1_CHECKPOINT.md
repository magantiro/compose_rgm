# Process-V2 Wave 1: Review Checkpoint 1

**Branch** `codex/editing-v2-process-v2-runnable-chain` · **base** `3f3258e` ·
**head** `e0fe7f3` · **worktree** `/private/tmp/compose-process-v2-runnable-chain`

Wave 1 of the runnable-chain specification: Sections 3 and 4, plus the Section 9
protocol fixtures and an adversarial pass. **Stopped here as instructed.** Wave 2
is not begun.

Nothing remote or scientific ran. No Modal job, materialization, Gate 0, T1, P50,
or training. No authority-bearing measured artifact was created: every generated
object carries all seven authority fields false and a `_NO_DOWNSTREAM_AUTHORITY`
status.

## 1. Outcome

Wave 1 is complete and verified, with **one blocking owner decision** (§6) and a
handful of findings that change what a later wave should assume.

All five defects the specification asserted were **confirmed**, none refuted, and
two proved broader than described. The interface module written to *prevent*
rework had four real defects of its own, found by the adversarial pass.

## 2. Verification

Interpreter: `/Users/rmaganti/.../compose_rgm_claude_generators/.venv/bin/python`
— python 3.14.2, **torch 2.11.0, PyYAML 6.0.3**, numpy 2.4.2, rdkit 2025.09.6,
pytest 9.1.1, ruff 0.15.22, macOS 26.5.2 arm64. Guards
`KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src`.

```
base 3f3258e : 16 failed, 2729 passed, 1 skipped,             0 collection errors
head e0fe7f3 : 16 failed, 2980 passed, 1 skipped, 1 xfailed,  0 collection errors

regressions (only on head) : NONE
fixed (only on base)       : none
collection errors          : 0 both sides
```

**+251 tests passing, zero regressions**, compared as failure sets rather than
counts. The 16 shared failures are `test_e6_*` (6), `test_ringcore_*` /
`test_ring_core_*` (9), and `test_reference_successor_kernel` (1). None
references `process_v2`; see §7 for what two of those groups actually are.

| Gate | Result |
|---|---|
| Focused Process-V2 suites | 203 (A+C+schema) + 80 (B) passed |
| New chain verifier | `AGREES`, **0 findings**, 11 artifacts, 43 pointers, 7 identity edges each resolving to exactly one node |
| V2 scientific identity | `0c938177…` **unchanged**, as §3.2 requires |
| V1 process identity | `6c4721f0…` unchanged |
| Seven contracts | rebuild twice, byte-identical to committed |
| Ruff, touched files | clean (12 files) |
| Ruff, repo-wide | 64 base = 64 head, none from a touched file |
| `git diff --check` | clean |
| V1 configs modified | **0** |
| AI mention in commits | 0 |

## 3. Commits

| Commit | Subject |
|---|---|
| `89f2b24` | data: freeze the shared process v2 chain interfaces |
| `90a9670` | data: enforce the frozen process v2 interfaces instead of declaring them |
| `a2f39b4` | tests: pin the process v2 structural protocol and import boundaries |
| `37012ea` | experiments: add the resolved evidence binding and a sound chain verifier |
| `e0fe7f3` | data: bind the exact completion and build a one-pass process v2 chunk cache |

28 files, **+12,120 / −1,022**.

## 4. Schema and identity changes

| Name | Version | Change |
|---|---|---|
| `compose.editing_v2.process_v2.*` (6 shared interfaces) | 1 | NEW: resolved evidence binding, exact completion binding, chunk manifest, chunk-cache completion, admission completion, rejection ledger, generation-committed marker |
| the 7 chain contract schemas | 1 → **2** | dependency edges added; the unfillable `admitted_source` slot removed; `contract_revision`, `resolved_evidence_binding`, `shared_policy_registry`, `superseded_design_lineage` added |
| `compose.editing_v2.shared_policy_registry` | 1 | NEW |

The seven `3f3258e` contract hashes are preserved as `lineage_reference` typed
pointers inside each artifact's `superseded_design_lineage`, and validation
refuses a lineage value equal to a live one.

## 5. What Wave 1 fixed

**§3.1** The `admitted_source` slot was unfillable by construction: the builder
emits null, the validator rebuilds null and requires equality, so a correctly
resealed body with measured hashes always failed. Confirmed by attempting the
fill, then removed; measured provenance moved to a separate resolved evidence
binding that carries no authority.

**§3.2** Dependency edges are explicit rather than implied by tuple order.

**§3.3** The generic verifier is replaced. Identity edges resolve contextually to
exactly one node, so a V2 artifact carrying the V1 value now fails instead of
matching either. Pointers are declared and discovered structurally, so a missing
or misspelled target is a reportable failure rather than an edge that vanishes.
An undecidable required edge yields `INCONCLUSIVE`, which is not success.

**§3.5** ~500 transcribed lines of V1 policy replaced by projection from the
pinned frozen sources, proved byte-equivalent twice: against the frozen config
subtree read from disk, and against the version-1 contract bodies at `3f3258e`
via `git show`.

**§3.6** Publication is transactional: stage under a content-addressed
generation, validate the whole graph by reading it back, then write the
`COMMITTED` marker last. Readers refuse an unmarked generation.

**§4.1** The binder requires the exact completion and all twenty tasks instead of
deriving corpus size from visible directories, which could silently reduce
support.

**§4.2** Planning called Git inside an image with no `.git`, so it could not run
remotely at all. The source revision is now computed locally and passed in.

**§4.3** A one-pass content-addressed chunk cache is **built and tested, but
nothing consumes it yet** -- see the correction in §7. The rebind still reads
through the rescanning range reader, so the cost saving is available, not
realized. Wiring it means editing `editing_process_v2_rebind.py`, which is in
its own `_SOURCE_FILES`, so doing so moves `source_revision_sha256` and
therefore `run_identity_sha256`, relocating every rebind artifact. That is an
owner decision, listed in §10.

**§4.4** The container limit was cosmetic — validated and printed while the
decorator capped at 20. It now bounds actual submission geometry.

**§4.5** The volume is reloaded at every visibility boundary.

**§4.6** Admissions stream per shard instead of loading every proof into one
dictionary, and the rejection ledger publishes full reason-coded records.

## 6. BLOCKING: the §3.4 authority rename cannot be done in place

**§3.4 is not implemented, deliberately.** It collides with §3.2.

`p50_authorized` sits at `editing_v2_process_identity.py:669`, inside the
`authority` block of `build_editing_process_v2_contract()`, which is serialized
to `configs/editing_v2_semantic_process_v2.json`, whose **self-hash and physical
bytes are both hashed into the V2 process identity**. Measured:

```
contract self-hash now   : 5e317e65cfbff0a8c8770765d1248925d0e068f4e2dd7f8a4cf474b85955ee1e
contract self-hash after : 2a2ad8171d2a34a54c85cf6d2d6b4d10c16ef5beb71d271fa7f1d37bece63dc0
V2 identity now          : 0c938177a34819e6e828920c1f66e240c6eb251fe7c9ea6cfe6757829dceb2dd
V2 identity after rename : dc6eeb962dd7ee3a4b3ea1728dba98c154d9c844c539b33dfe5cbd71c5db369f
```

§3.2: *"If implementation changes an identity-source file and moves that
scientific identity, stop for Codex review."* Verified independently twice.

Additionally, **seven test assertions across four files currently require the
retired spelling**, including an exact set equality over the authority key set
(`test_editing_process_v2_identity.py:229`) and a dict equality whose comment
says it asserts both spellings on purpose
(`test_process_v2_rebind_end_to_end.py:836`).

**The decision needed:** may the rename be applied only *outside* the
identity-source set — the two adapter sites and one census site, which move only
`admitted_source_sha256` and a diagnostic report shape — while
`editing_v2_process_identity.py:669` and `configs/editing_v2_semantic_process_v2.json:189`
keep the spelling permanently as a documented frozen legacy field? If so, the
`RETIRED_AUTHORITY_FIELDS` refusal must be scoped not to reject the process
contract's own authority block. One spelling everywhere is a Gate-0-invalidating
identity move and is an owner call.

Done in its place: the evidence binding does not ingest the adapter's authority
block at all, and refuses any granted `*_authorized` field spelling-agnostically.
That is correct under either decision.

## 7. Negative findings and corrections to the record

**The frozen interface module had four real defects**, all found by the
adversarial pass and fixed in `90a9670`. `require_authority_false` inspected only
the top level, and *every* authority block in this repository is nested, so the
one function whose job is refusing a granted field could not see where authority
lives. `require_census_reconciles` coerced with `int()`, accepting `"646779"`,
`3.9`, `True`, and a **negative** rejection count reconciling an inflated
admitted total. Pointers accepted any 64-character string, so non-hex passed and
one digest in two cases produced two pointers. And it had **zero importers and
zero tests**, so every mutation to it survived trivially.

**One of my own coherence rules was wrong.** I forbade a lineage reference from
carrying a semantic hash, which breaks the one thing §3.2 requires lineage for —
preserving the superseded contract self-hashes. It caused 32 of 36 initial
integration failures. Only an external asset, which declares no self-hash, is
genuinely excluded.

**The new verifier reintroduced the vanishing-pointer defect through an exception
handler.** It did `except ProcessV2SchemaError: continue`, so a *malformed*
declared pointer silently left the graph exactly as a missing target used to.
Fixed: a node carrying the declared pointer key set is a pointer, and one that
will not validate fails. Role coherence is also now checked before the hash,
because a mislabelled role otherwise degraded into a generic stale pointer.

**Two asserted defects are broader than the specification stated.** The Active8
source module hits **three** live-V1 gates, not one, and dies at the first. And
§3.3(b)'s missing-target diagnostic was not merely unreachable: a misspelling
silently moved six literals from `agree` to `unclassified` with no finding at any
severity.

**`test_no_production_module_imports_the_oracle` is a false positive.** It is a
substring scan, and the oracle's *path* is listed in the hashed implementation
file set that defines the V1 process identity. An AST scan finds **zero** import
edges anywhere in production. Same defect class the specification names for the
verifier, inverted. Not touched; it belongs to its owner.

**Nine of the sixteen shared failures are frozen-hash mismatches** consistent
with the V1 process identity having moved when Process V2 edited a hashed
implementation file. Pre-existing at base, but with a nameable cause rather than
ambient noise.

**The old verifier can no longer be made green on this branch.** With
`--base-revision 89f2b24` it reports 39 FAILs, all false positives on the
deliberately preserved lineage: it grants a lineage exemption only to two
hard-coded constants, so it structurally cannot express "an artifact deliberately
records its own superseded hash." A direct consequence of §3.2, and an argument
for retiring it in favour of `scripts/verify_process_v2_chain.py`.

**The specification's §4.3 referent is wrong, and the correction cuts both ways.**
There is no 2,048-entry range plan anywhere in the repository; the rebind's
actual default is `DEFAULT_ENTRIES_PER_TASK = 64`, giving roughly 506 ranges per
shard rather than seventeen. Measured on a production-shaped shard, the range
reader's projected overhead is **28.0 s/shard, 15x one fused pass** at the real
default of 64. But at the specification's assumed 2,048 it would be 0.9 s/shard,
i.e. **0.5x -- cheaper than one validating fused pass**. The one-pass cache is
the right call at the granularity that actually ships; it would not have been at
the granularity the specification assumed. Separately,
`semantic_active8_chunk_cache.py` already makes one sequential pass and is *not*
the module carrying the rescan defect; the defect lives in the rebind itself and
in `active8_inventory_mapreduce.py`.

**`ADMITTED_SOURCE_SCHEMA_VERSION` moves 1 to 2**, adding four fields to
`identity()` and changing `admitted_source_sha256`. Documented as incompatible,
not additive. Workstream A's evidence binding imports that constant, so the two
are coupled and should be reviewed together.

**B reported one whole-suite failure I could not reproduce.**
`test_editing_v2_semantic_p50_successor_cache.py::test_open_physically_reopens_prepared_recipe`
failed in B's tree (17 failed there) and does **not** appear in the integrated
head run (16 failed, absent from the failure set); it also passes in isolation at
the integrated head, 12/12. The suspected mechanism is that
`semantic_p50_successor_cache_implementation_sha256` rglobs every `.py` under
`src/compose_v4`, so new modules move it, and collection order then exposes a
pre-existing cross-test isolation sensitivity. B could not identify the
interacting test and neither did I. Recorded as an unexplained order-dependent
risk rather than dismissed: it did not occur in the measured integrated run, but
the mechanism is real and a bisect is warranted.

**Six mutations survive in `resolve_process_v2_admitted_source`'s
completion-binding predicate** (`editing_process_v2_admitted_source.py:536-552`).
Six of seven guards have no test: run identity, plan hash, task inventory,
payload binding, the completion's own self-hash, and its `training_authorized`
flag can each be disabled with all 68 rebind tests still passing. Not fixed in
Wave 1; it is the first item for Wave 2.

**Prior whole-suite figures were invalid.** The earlier branch measured with
`compose_rgm/.venv` (torch 2.13.0, no PyYAML). That environment produced 15
collection errors hiding ~211 tests, invented four failures that pass here, and
reported a RingCore catalog "drift" that does not exist under the compliant
interpreter.

### Provenance note on Workstream B

B's files were integrated from its worktree while it was still verifying, before
its report arrived. The report has since been received and resolves the concern:
B had finished, the apparently clean worktree was a `git stash` taken to run its
own base comparison, **nothing was reverted**, and the SHA-256 of all nine files
matches what was committed. The integration captured B's final state. Its
findings are folded into §5, §7 and §8 below.

## 8. Local microbenchmarks — NOT production evidence

Synthetic fixtures, local CPU, tiny payloads. They inform initial resource
requests only; production throughput and peak memory are deferred to the later
authorized execution pass.

```
contract build            13.4 ms
contract publish          42.4 ms
generation validation     26.9 ms
```

Workstream A. Workstream B, on one production-shaped shard (32,339 rows, 281.8 MB
decompressed, 1.91 MB gzipped): fused pass **1.864 s**, 17,350 rec/s, peak
**39.6 MB**. Peak memory converges rather than scaling with corpus size -- 64x the
data costs 3.3x the peak -- bounded by the read block, the decompress block and
one chunk. The streaming resolver's peak grows at **+146 KB** per source entry
against the eager global-dictionary's **+440 KB**, a 3.0x shallower slope,
reproducible to two decimals.

B also found and fixed a real bug in its own first implementation: it was
unbounded, and one 1 MiB compressed block of repetitive rows expanded to the
whole shard, giving 26.5 MB peak for an 8 MB corpus.

Precise accounting, because it matters: the whole job opens each gzip **twice** --
once hash-only inside the payload binding, which §4.1 requires, and once fused in
the build task. Only the second decompresses. "Exactly one fused pass" is a claim
about the cache build, which is what §4.3 specifies.

## 9. The three riskiest areas for review

1. **The completion-binding predicate** (`editing_process_v2_admitted_source.py:536-552`)
   — six of seven guards unprotected by any test.
2. **`scripts/verify_process_v2_chain.py`** — it is the new authority for the
   whole graph, it is new code, and its predecessor was unsound in three
   independent ways. Its identity-edge resolution and its status semantics
   deserve the closest reading.
3. **The chunk cache's one-pass and restart guarantees**
   (`editing_v2_process_v2_chunk_cache.py`) — correctness here determines whether
   the later 40-worker execution reads consistent input.

## 10. Proposed Wave-2 interface inputs (not implemented)

Offered for review, not built:

- the Process-V2 decision index satisfying `StructuralDecisionIndex`, consuming
  the rebind output as the sole legality and whole-trace admission authority;
- a Process-V2 Active8 plan/map/reduce/completion/source adapter in a distinct
  schema and artifact namespace, not a subclass of the V1 index;
- a Process-V2 Gate-0 loader, structural computation wrapper, result and decision
  schemas, keeping the V1 path unchanged;
- the strict xfail in `tests/test_process_v2_import_boundaries.py` becomes a real
  gate the moment the V1-loader dependency is removed.

A second owner decision, smaller than §6: **may the rebind be wired to consume
the chunk cache?** It requires editing `editing_process_v2_rebind.py`, which is in
its own `_SOURCE_FILES`, so it moves `run_identity_sha256` and relocates every
rebind artifact. The payoff is large -- roughly 10,106 range tasks become ~320
chunk tasks, 8 waves at a bound of 40 instead of 253 -- but it is an identity move
and not mine to make.

Wave 2 also needs the §6 decision before its authority-field vocabulary settles.
