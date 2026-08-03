# Process-V2 Wave-2 — the P50 successor-cache "stale plan" failure

Diagnostic for the one unexplained whole-suite failure carried out of Wave 1
(`docs/PROCESS_V2_WAVE1_CHECKPOINT.md` §7):

```
tests/test_editing_v2_semantic_p50_successor_cache.py::test_open_physically_reopens_prepared_recipe
SemanticP50SuccessorCacheError: semantic P50 successor-cache plan implementation or policy is stale
```

Nothing remote or scientific ran. No Modal job, materialization, Gate 0, T1, P50
or training. No `src/`, config, or existing test file was modified.

## 1. Verdict

**It is not an ordering effect, and it is not a cross-test isolation defect.**
The Wave-1 hypothesis — that `semantic_p50_successor_cache_implementation_sha256`
rglobs every `.py` under `src/compose_v4`, so adding modules moves the hash and
collection order then exposes a sensitivity — is **disproven**: a persistently
added module moves the hash and the module still passes 12/12, because the plan
side and the check side are both computed from the same live tree.

The real sensitivity is a **time-of-check/time-of-use race against the working
tree**. `validate_semantic_p50_successor_cache_plan` re-reads all 216 `.py` files
under `src/compose_v4` on every call and compares the result with the value
frozen into the plan when the plan was built, milliseconds earlier. Any write to
that directory landing between the two reads fails the test. Nothing inside the
suite performs such a write; something outside the suite process does.

There are in fact **two** race windows against that read, not one — a file may
also vanish between the rglob and the read *inside a single call*, which raises a
sibling error. §11.1 measures both. The verdict and the fix are unchanged.

Proven: the mechanism, constructively, with a purely external writer and no
production code touched. Inferred (not proven): the specific external event in
Workstream B's tree — the Wave-1 checkpoint itself records a `git stash` taken in
B's worktree "to run its own base comparison", and that B's files were integrated
from that worktree "while it was still verifying".

## 2. Environment

Base `b39420cef287fc1e2ba66a729f5c80df64d5f23a` (`b39420c`), clean tree.
Interpreter `compose_rgm_claude_generators/.venv/bin/python` — python 3.14.2,
torch 2.11.0, PyYAML 6.0.3, pytest 9.0.3. Guards
`KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src`.

No `conftest.py` exists and no ordering plugin is installed (`pytest-randomly`,
`pytest-xdist` absent), so collection order is deterministic: alphabetical by
file, source order within a file. `-p no:randomly` is a no-op here.

## 3. The failing predicate

`src/compose_v4/experiments/editing_v2_semantic_p50_successor_cache.py:1321`

```python
current_implementation = semantic_p50_successor_cache_implementation_sha256(repo_root=repo_root)
if (
    plan.get("implementation_sha256") != current_implementation
    or plan.get("policy") != _policy()
):
    raise SemanticP50SuccessorCacheError(
        "semantic P50 successor-cache plan implementation or policy is stale"
    )
```

`plan["implementation_sha256"]` is written once by
`build_semantic_p50_successor_cache_plan` (`:1134`). `validate_…_plan` is then
called again by `build_…_leaf` (`:1608`), `build_…_manifest` (`:1822`) and
`open_…` (`:2455`), each recomputing the hash from the live tree with the same
`repo_root`. The test's `REPO_ROOT` is the real repository
(`tests/test_editing_v2_semantic_p50_successor_cache.py:45`).

Which clause moves, measured directly:

```
impl before extra file : 7408bfb75ebabae46627a37ed2bd7bd4ad07f4049a39ec7f1d98478d48261665
impl with extra file   : 2b739feb365057a81799efabc755ebd142b15d66ea459888354a4be2205a8231
impl after removal     : 7408bfb75ebabae46627a37ed2bd7bd4ad07f4049a39ec7f1d98478d48261665
policy identical       : True   (policy_sha256 42a325d9eb6909b5da4ae3023faebbf466db56b802ff412ff7b4e5b15cf8111f)
```

Only the implementation clause is live-tree dependent. `_policy()` is a pure
function of two module constants and never moves.

## 4. What was ruled out

| Candidate | Result |
|---|---|
| `functools.lru_cache` on the hash | **absent** — `hasattr(fn, "cache_info")` is `False`; every call re-reads all 216 files |
| a leaked `monkeypatch` | all uses are the function-scoped fixture or an explicit `monkey.undo()` in `finally` |
| a leaked mutable module constant (`NO_DOWNSTREAM_AUTHORITY`, `SUPPORT_COMPILATION_TIME`) | would move both sides identically; measured invariant across the whole suite |
| `__pycache__` picked up by the rglob | 0 `.py` files under any `__pycache__`; the glob is `*.py`, caches hold `*.pyc` |
| a symlink under `src/compose_v4` with mutable content | none (`find -type l` empty) |
| a test writing into the real `src/compose_v4` | every `src/compose_v4/...` write in the suite targets a `tmp_path` fixture repo (`test_run_editing_v2_semantic_p50_app.py:35`, `test_editing_v2_semantic_p50_execution_contracts.py:93`); all `shutil.copy*` calls copy *out of* the repo into tmp |
| a script or subprocess regenerating a source file | `write_editing_process_v2_contract` writes `configs/…json`, outside the glob; all test `git` calls against the real root are `show`/`rev-parse` |

## 5. Order search — negative

Two full-suite runs at `b39420c`, one in each direction. Between them, **every**
test module in the suite runs before the P50 module in at least one run.

```
natural order   pytest tests/ -q -rf
                16 failed, 2980 passed, 1 skipped, 1 xfailed  in 1029.97s
                -> identical to the integrated-head figures; P50 absent from the failure set

reverse order   pytest $(ls tests/test_*.py | sort -r) -q -rf
                17 failed, 2979 passed, 1 skipped, 1 xfailed  in 1205.38s
                -> P50 absent from the failure set
```

The extra reverse-order failure is
`test_process_v2_modal_runtime_surface.py::test_resolution_memory_grows_far_slower_than_the_global_dictionary_it_replaced`,
a memory-slope benchmark; that run overlapped the natural-order run on the same
machine. It is unrelated to P50.

The reverse run carried an instrumentation plugin that wrapped the production
`semantic_p50_successor_cache_implementation_sha256` and `_policy`, recorded every
call against its owning test id, snapshotted the rglob file set and per-file bytes
at every test teardown, and polled `src/compose_v4` for name/size/mtime changes on
a 10 ms background loop. Over the whole 20-minute run:

```
distinct impl values          : 1   (7408bfb7…)
distinct policy values        : 1
within-test divergences       : 0
cross-test changes            : 0
tree changes (teardown)       : 0
tree changes (10 ms polling)  : 0
tests that call the hash      : 9   (all in the P50 module; no other module uses it)
```

**No test in the suite moves either hash input.** That is a stronger statement
than any bisect: with the inputs provably constant, no permutation of the suite
can make the two sides of the predicate disagree.

## 6. The Wave-1 hypothesis is disproven

Negative control — add a *persistent* extra module and run the target module:

```
printf '# persistent diagnostic module\nVALUE = 1\n' > src/compose_v4/_p50_persistent_probe.py
pytest tests/test_editing_v2_semantic_p50_successor_cache.py -q
-> 12 passed in 51.21s
```

The hash value moves (§3) and the module still passes, because the plan is built
from the same moved tree it is later checked against. "New modules move the hash"
is true and irrelevant; only *movement during a test* matters.

## 7. Constructive proof — the actual mechanism

### 7.1 Deterministic, one write inside the plan→reopen window

A plugin wraps `open_semantic_p50_successor_cache` and, for the target nodeid
only, creates a transient `.py` file under `src/compose_v4` for the duration of
that one call — exactly what an external editor does if its write lands there.
No tracked file is touched.

```
pytest tests/test_editing_v2_semantic_p50_successor_cache.py -q -rf -p p50_inject
  (P50_INJECT_NODEID=test_open_physically_reopens_prepared_recipe)

editing_v2_semantic_p50_successor_cache.py:1326: SemanticP50SuccessorCacheError:
    semantic P50 successor-cache plan implementation or policy is stale
AssertionError: Regex pattern did not match.
  Expected regex: 'canonical JSON'
  Actual message: 'semantic P50 successor-cache plan implementation or policy is stale'

1 failed, 11 passed in 48.93s
```

Byte-for-byte the reported failure, on the reported test, and **exactly one test
fails**.

### 7.2 Fully external, no production code wrapped

A separate process creates and removes one transient `.py` file under
`src/compose_v4` at a fixed offset into an otherwise untouched module run:

| external write at | outcome |
|---|---|
| +8 s | 12 passed |
| **+16 s** | **1 failed — `test_open_physically_reopens_prepared_recipe`** |
| +24 s | 1 failed — `test_manifest_rejects_same_count_leaf_swap` |
| +32 s | 1 failed — `test_open_rejects_physically_rederived_validation_set_mismatch` |
| +40 s | 1 failed — `test_open_rejects_same_address_wrong_states_and_teacher_fibers_after_full_rehash` |

A single 300 ms write always produces **at most one** failure, always with the
same message, and *which* test fails is a pure function of when the write lands.
The +16 s trial reproduces the reported failure exactly, with the production
module completely unmodified. Because the mapping is timing-based, these offsets
are calibrated to this machine and this worktree — §11 gives the timing-free
version to re-run.

## 8. Why the window is wide

`semantic_p50_successor_cache_implementation_sha256` reads all 216 files on every
call: **42.85 ms** measured. Instrumented call counts for one module run:

```
37 calls  test_complete_cache_opens_exact_train_and_validation_unions
 2 calls  test_leaf_rejects_missing_extra_duplicate_and_same_count_substitution
37 calls  test_manifest_rejects_same_count_leaf_swap
37 calls  test_open_physically_reopens_prepared_recipe
37 calls  test_open_rejects_cross_run_identity_and_physical_leaf_tamper
37 calls  test_open_rejects_physically_rederived_validation_set_mismatch
37 calls  test_open_rejects_same_address_wrong_states_and_teacher_fibers_after_full_rehash
37 calls  test_open_rejects_same_state_fabricated_candidate_coordinates_after_full_rehash
 1 call   test_validation_contract_rejects_role_relabel_even_when_rehashed
-----
262 calls x 42.85 ms = 11.2 s of live-tree reading inside a 49 s module run
```

Seven tests re-read the tree 37 times each: once in the plan build, once per task
leaf (34), once for the manifest reduce, once for the physical reopen. The two
low-exposure tests fix that decomposition — `_build_plan` alone is 1 call,
`_build_plan` plus one leaf is 2. Roughly **23% of the module's wall time is
spent inside the vulnerable read loop**, which is why a single stray write has a
high probability of landing in it.

## 9. Causal chain

1. `build_semantic_p50_successor_cache_plan` freezes
   `implementation_sha256 = H(tree@t0)` into the plan.
2. Every later `validate_semantic_p50_successor_cache_plan` recomputes
   `H(tree@t1)` from the live working tree and requires equality.
3. Any write under `src/compose_v4` with `t0 < t_write < t1` makes them differ.
4. The predicate raises "…plan implementation or policy is stale"; in
   `test_open_physically_reopens_prepared_recipe` it fires before the
   "canonical JSON" error the test expects, so the `pytest.raises(match=…)`
   fails.
5. Exactly one test fails, because a plan built after the write is consistent
   again.
6. Nothing in the suite performs step 3 (§5), so the writer is external to the
   pytest process: an editor save, `git stash`, `git checkout`, `git stash pop`,
   a formatter, or an agent applying an edit while a background suite runs.
7. Workstream B's tree is documented to have had exactly such activity around its
   verification window (`PROCESS_V2_WAVE1_CHECKPOINT.md` §"Provenance note on
   Workstream B": a `git stash` "taken to run its own base comparison", and
   integration performed "while it was still verifying"). Step 7 is **inferred**;
   steps 1–6 are proven.

## 10. Recommendation

**No production change.** The hash semantics are deliberate: the freshness check
is the invalidation signal, and memoising it, snapshotting it once per process,
or narrowing its input set would weaken exactly the guarantee it exists to give.
Nothing here justifies touching frozen hash semantics.

**Primary fix is procedural, zero code.** Run whole-suite gates from a clean,
detached worktree that nothing else writes to, and do not edit, `git stash`,
`git checkout` or `git add -p` the tree while a suite is in flight — the same
launch discipline `.claude/context/learnings.md` already records for Modal
launches (2026-07-27), extended to local suite runs. A background suite plus a
foreground edit is the exact shape that produced this.

**Optional hardening, test-isolation only, NOT applied here** (it edits an
existing test file, which this task is not permitted to touch). Make the P50 test
module read a frozen snapshot instead of the live repo:

```python
# tests/test_editing_v2_semantic_p50_successor_cache.py
@pytest.fixture(scope="module")
def repo_snapshot(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Freeze the implementation tree so an external edit cannot race the plan."""
    root = tmp_path_factory.mktemp("p50_repo") / "compose"
    shutil.copytree(
        REPO_ROOT / "src" / "compose_v4",
        root / "src" / "compose_v4",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for relative in cache._IMPLEMENTATION_SOURCES:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(REPO_ROOT / relative, target)
    return root
```

then thread `repo_snapshot` through `_build_plan`, `_materialize` and all twelve
`repo_root=REPO_ROOT` call sites in place of `REPO_ROOT`. One copy per module
(~216 files, ~10 ms) removes all 262 live-tree reads. It changes no production
code, no hash rule, and no assertion — the module tests cache identity logic, not
the repository's contents.

**No regression test is added.** The failure has no in-repo cause to regress
against: a test cannot guard against an editor writing to the working tree, and a
test that itself created and deleted files under `src/compose_v4` to demonstrate
the property would reintroduce the very hazard being reported. §11 is the
reproduction of record.

## 11. Reproduction of record

Self-contained; no plugin, no production edit, no timing calibration. From a
clean worktree at `b39420c`, an external writer toggles one transient `.py` file
under `src/compose_v4` for the duration of the run:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src
PROBE=src/compose_v4/_p50_race_probe.py
( while true; do echo '# probe' > "$PROBE"; sleep 0.05; rm -f "$PROBE"; sleep 0.05; done ) &
TOGGLER=$!
python -m pytest \
  "tests/test_editing_v2_semantic_p50_successor_cache.py::test_open_physically_reopens_prepared_recipe" \
  -q -rf
kill "$TOGGLER"; rm -f "$PROBE"
```

Result, 3/3 trials:

```
editing_v2_semantic_p50_successor_cache.py:1326: SemanticP50SuccessorCacheError:
    semantic P50 successor-cache plan implementation or policy is stale
FAILED tests/test_editing_v2_semantic_p50_successor_cache.py::test_open_physically_reopens_prepared_recipe
1 failed in 2.06s / 1.78s / 1.58s
```

Control, identical command with the writer removed: `1 passed in 7.61s`.

### 11.1 Independent re-run — the writer has two failure modes, not one

Re-run by the integration lead in the Wave-2 hardening worktree, same interpreter
and guards, six trials plus a control:

```
control (no writer) : 1 passed in 8.22s
trial 1 STALE   trial 2 STALE   trial 3 ABSENT
trial 4 STALE   trial 5 STALE   trial 6 STALE
```

`STALE` is the reported message; `ABSENT` is
`semantic P50 cache implementation source is absent: src/compose_v4/_p50_race_probe.py`.
So the reproduction is **stochastic between two sibling messages, 5/6 and 1/6**,
not deterministic on one. The claim of 3/3 above was a small sample.

Both modes are the same defect, and the hash function's own shape explains why
there are exactly two windows (`:295-315`):

```python
transitive_sources = {... for path in source_root.rglob("*.py") if path.is_file()}   # window A opens
relative_sources = tuple(sorted(set(_IMPLEMENTATION_SOURCES) | transitive_sources))
for relative in relative_sources:
    path = root / relative
    if not path.is_file():
        raise SemanticP50SuccessorCacheError(f"...source is absent: {relative}")     # window A closes
    digest.update(path.read_bytes())
```

- **Window A, inside one call** — a file enumerated by the rglob is deleted
  before its `read_bytes`, so the `is_file()` guard raises `ABSENT`. Width: one
  hash call, ~43 ms.
- **Window B, between two calls** — the file set or contents differ between the
  plan freeze and a later `validate_…`, so the digests differ and §3's predicate
  raises `STALE`. Width: the whole plan→reopen span.

A create-then-delete toggler exposes both; a writer that only *creates* (an
editor save, `git stash pop`) can only hit window B, which is why the reported
Wave-1 failure is the `STALE` one. This does not change the verdict, the causal
chain, or the recommendation — it widens the proven mechanism from one race
window to two, and both are closed by the same procedural fix in §10.

Running the whole module under the same writer fails **exactly the seven tests
the call-count analysis predicts** (§8) and passes the other five:

```
7 failed, 5 passed in 7.62s
  test_complete_cache_opens_exact_train_and_validation_unions
  test_open_rejects_cross_run_identity_and_physical_leaf_tamper
  test_open_physically_reopens_prepared_recipe
  test_manifest_rejects_same_count_leaf_swap
  test_open_rejects_physically_rederived_validation_set_mismatch
  test_open_rejects_same_address_wrong_states_and_teacher_fibers_after_full_rehash
  test_open_rejects_same_state_fabricated_candidate_coordinates_after_full_rehash
```

The five survivors are exactly the low-exposure tests: the two prepared-recipe
tests and the receipt test never touch the hash at all, and the remaining two
make only 1 and 2 calls — a single call cannot disagree with itself, and a
two-call window is ~43 ms, narrow enough to slip between toggles.

A single-shot write at a fixed time offset (§7.2) reproduces it too but is
machine-calibrated — the same `+16 s` offset that failed the target test on one
worktree passed on a colder one. Use the toggler above for a repeatable check.

## 12. Not determined

- Which external write hit Workstream B's tree, and when. B's tree and run log
  were not available; §9 step 7 rests on the Wave-1 checkpoint's own provenance
  note, not on a timestamp.
- Whether B's failure arose during `_materialize` (an uncaught error) or during
  `open_…` (the `pytest.raises` mismatch). Both produce the same message and the
  same one-test signature; the reported text does not distinguish them.
- Multi-module interactions were not searched exhaustively (276 modules). The
  argument in §5 replaces that search: the hash inputs were measured constant
  across the full instrumented reverse-order run (10 ms polling, 20 min), and the
  natural-order run — uninstrumented — reproduced the integrated-head failure set
  exactly. So no permutation covered by those two runs can move them.
