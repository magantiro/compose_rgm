# Preservation migration contract

What may be changed in this repository, what may not, and why. Every constraint
below is MEASURED at `061ead93` (baseline tag `pre-cleanup-2026-09-23`), not
inferred from convention. The measurements live in
`diagnostics/repo_hygiene/` and the manifests in this directory.

The goal of a structural pass here is **clean current API + 100 percent
historical recoverability + zero capability regression**, in that order of
difficulty and the reverse order of priority. Where cleanliness and
recoverability conflict, recoverability wins and the conflict gets reported.

## 1. What is read-only, and why it is larger than it looks

`tools/repo_pinned_file_set.py` finds 1,879 of 8,805 tracked files that are the
subject of a sha256 pin, including 542 of the 595 modules in `src/compose_v4`.

That per-file set **understates** the constraint. Three fingerprints hash
DIRECTORIES rather than named files:

| fingerprint | covers | what a change does |
|---|---|---|
| `modal_apps/train_tracelet_gm.py::_source_fingerprint` | every `.py`/`.json` under `src/`, `scripts/`, `recipes/`, plus that launcher | moves `run_identity`; the immutable stage manifest then refuses the run label, so an in-flight run cannot be resumed |
| `modal_apps/t4_objective_reset_app.py::main` | every `.py` under `src/`, plus `configs/t4_objective_reset_runtime_v1.json` | `files_sha256` stops matching the stored receipt, forcing re-verification |
| `editing_v2_semantic_p50_successor_cache.py::semantic_p50_successor_cache_implementation_sha256` | every `.py` under `src/compose_v4` | every existing P50 cache plan **raises** `plan implementation or policy is stale` |

Measured by recomputing each digest against a perturbed tree:

- a whitespace-only edit to ANY file under `scripts/` moves the training run identity;
- a whitespace-only edit to an individually UNPINNED `src/` module moves two identities;
- **ADDING** a new module under `src/compose_v4` moves the cache identity exactly
  as editing one does.

**Therefore `src/`, `scripts/` and `recipes/` are read-only in AGGREGATE**, not
merely per file. There is no per-file exemption, and no new file may be added to
them without paying one of the costs above.

## 2. The facade, costed rather than refused

A clean public surface (`compose_v4/api/`, `proposals/`, `execution/`) that
re-exports the frozen implementation underneath is the right long-term shape. It
is not free, and the requested placement is the most expensive of the three:

| placement | P50 cache plans refuse to load | train / t4-reset identity moves |
|---|---|---|
| `src/compose_v4/api/` (as requested) | **YES** | YES |
| `src/compose_api/` | no | YES |
| `api/` outside `src/` | no | no |

The P50 column is the expensive one: the cache it invalidates is recorded at
roughly 1,025 core-hours to rebuild, and the plans live on a Modal volume rather
than in git.

**Decision taken here: the facade is NOT added in this pass.** The measurement is
reported so the owner can choose the placement knowingly. If the facade is
wanted, `api/` outside `src/` costs nothing in identity terms and needs only a
`pyproject.toml` packaging entry; `src/compose_api/` costs a fresh training run
label and nothing else.

## 3. Target tree, with each move classified

Derived from the measured import graph, not from the reference repository's
directory names.

```
src/compose_v4/
  chem/ rewrite/ model/ gm/          executor + learned process      PINNED  do not move
  control/ policy/                   controllers                     PINNED  do not move
  data/                              corpus                          PINNED  do not move
  eval/ benchmark/ oracles/ lipids/  evaluation                      PINNED  do not move
  experiments/                       21 library modules + 281 drivers PINNED  see below
tests/                               pytest                          SAFE
tools/                               repo tooling                    PARTLY SAFE (102 of 250 pinned)
modal_apps/                          cloud entrypoints               RISKY   (87 pinned; many self-hash)
configs/ docs/ diagnostics/ repro/   records                         APPEND ONLY
archive/                             retired, provenance preserved   APPEND ONLY
third_party/                         vendored upstream               IMMUTABLE
```

### The seam is not where the directory names suggest

Import edges make `experiments/` (302 modules, 1,080 outbound, 50 inbound) look
like a pure consumer and therefore cleanly separable. It is not.
**21 of those 302 modules are imported by 37 library-core modules**, led by
`whole_ring_plan` (11 core dependents) and `production_successor_kernel` (6;
the README already states it must not be edited because it fixes
`process_identity_sha256`). Those 21 are library code filed as experiments.

Moving `experiments/` out wholesale would cut across a real dependency, which is
worse than the mess it replaces. The correct sequence is: promote those 21 into
the library, then treat the remaining 281 as drivers. Both halves are inside
`src/`, so both are blocked by section 1.

## 4. Things that look like defects and are not

Checked before proposing removal, because a gate or a duplicate here usually
encodes a past failure.

- **1,192 Python files under `diagnostics/`** are almost entirely SEALED SOURCE
  CAPSULES: byte-exact frozen copies of `src/compose_v4` for two scored
  campaigns. They are the record of the exact source a scored run executed. A
  cleanliness pass that deleted them as duplication would destroy the ability to
  reproduce those campaigns.
- **Cross-module private imports** are, in the cases found, the shared-enumerator
  pattern this repository depends on for correctness: one enumerator used by
  both the model mask and the corruption path is what guarantees a teacher
  action lands in the mask it is scored against. Duplicating it would be the
  defect.
- **Stale pins** are normally immutable launch records. Re-pointing one to make a
  cleanup pass succeed forges the record rather than fixing anything.

## 5. Rules for any change

1. Never edit, move, rename, split or reformat a path in `repro/pinned_paths_v1.json`.
2. Never add, edit or remove any file under `src/`, `scripts/` or `recipes/`
   without accepting a named identity cost from section 1.
3. Never delete. Retire to `archive/<original-area>/` with an entry in
   `archive/MANIFEST.json` recording original path, sha, commit, reason and
   replacement, plus a compat shim if anything imports the old path. Anything
   pinned by a historical contract does not move to archive; it stays put.
4. Never edit a historical artifact in place. Corrections become `*_v2` or an
   explicit correction record referencing the original hash.
5. One dependency layer per commit. No repo-wide autoformat, no mass import
   rewrite, no commit that moves files and changes behaviour together.
6. No semantic optimization during cleanup. A structural move and a performance
   change must not share a commit.
7. Branches are left alone entirely.
8. Where unsure: report, do not move.

## 6. Merge gates

`python3 tools/repro_gate.py --base 061ead93` prints one count per line and all
must be zero:

```
pinned_historical_files_modified      lost_public_symbols
lost_entrypoints                      lost_capability_groups
new_test_regressions                  newly_broken_pins
external_artifacts_without_backup
```

`lost_public_symbols` is necessary and **not sufficient**: an import can resolve
while the mechanism behind it is reached by no production caller, which this
repository has found six times. The entry-point, test and artifact lines are
separate gates, not corroboration of the symbol line.

`external_artifacts_without_backup` is expected nonzero today and is reported
rather than suppressed. See `external_artifacts_v1.json`.

Rollback point: tag `pre-cleanup-2026-09-23` at `061ead93`.
