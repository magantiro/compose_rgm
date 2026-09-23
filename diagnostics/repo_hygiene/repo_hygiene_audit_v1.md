# Repository hygiene audit v1

Machine-readable companion: `repo_hygiene_audit_v1.json`. Every number below is
MEASURED on this branch at `51b898f5`, against baseline `061ead93` (tag
`pre-cleanup-2026-09-23`), unless marked INFERRED. The two differ only by
additive hygiene artifacts; no `src/` file is touched by either.

## Headline

The library cannot be restructured by moving files. 542 of 595 src/compose_v4 modules are individually pinned, and three whole-tree fingerprints make the remaining 53 read-only too. Adding a module under src/ moves an identity, so even a facade package inside src/ is not free.

The reference for "clean" is one focused library package with separated
submodules, examples and tests. This repository cannot be moved toward that
shape by moving library files. The useful output is an argued target tree, the
subset that is safe to apply, and an explicit cost for the rest.

## 1. The constraint, measured

`src/compose_v4` holds 595 modules, of which **542 are individually
pinned**. Repo-wide the figure is 783 of 3346 tracked Python
files. Every subpackage except `experiments` and `control` is 100 percent
pinned, and those two are still covered by the tree fingerprints below, so the
53 unpinned modules are not a working surface:

| subpackage | modules | pinned | unpinned |
|---|---|---|---|
| `experiments` | 302 | 269 | 33 |
| `control` | 137 | 118 | 19 |
| `data` | 63 | 63 | 0 |
| `rewrite` | 34 | 34 | 0 |
| `benchmark` | 12 | 12 | 0 |
| `model` | 9 | 9 | 0 |
| `policy` | 9 | 9 | 0 |
| `chem` | 7 | 7 | 0 |

The per-file set UNDERSTATES the constraint. Three whole-tree fingerprints hash
DIRECTORIES rather than named files:

| fingerprint | covers | feeds |
|---|---|---|
| `train_tracelet_gm.py::_source_fingerprint` | every `.py`/`.json` under `src/`, `scripts/`, `recipes/` | `run_identity` |
| `t4_objective_reset_app.py::main` | every `.py` under `src/`, plus one config | `files_sha256` receipt |
| `semantic_p50_successor_cache_implementation_sha256` | every `.py` under `src/compose_v4` | cache implementation identity |

Measured consequence, by recomputing each digest against a perturbed tree:

- a whitespace-only edit to ANY file under `scripts/` moves the training run identity;
- a whitespace-only edit to an individually UNPINNED `src/` module moves two identities;
- ADDING a new module under `src/compose_v4` moves the successor-cache identity.

So `src/`, `scripts/` and `recipes/` are read-only in AGGREGATE, and a facade
package placed inside `src/` is not free. That is the single fact that decides
the shape of this work.

## 2. Structure, against the reference

A newcomer sees 26 tracked top-level directories and 11 top-level
files, of which eight compete to be the entry document: `README.md`,
`README_NAVIGATION.md`, `CLAUDE.md`, `CLAUDE_GENERATOR_READ_NOW.md`,
`COMPOSE_FULL_HANDOFF_2026-08-19.md`, `HANDOFF.md`, `REORG_REPORT.md`,
`AGENTS.md`. Seven directories hold manuscripts or submission bundles.

### Target tree, with each move classified

```
compose_v4/
  chem/  rewrite/  model/  gm/        executor and learned process   PINNED, do not move
  control/  policy/                   controllers                    PINNED, do not move
  data/                               corpus                         PINNED, do not move
  eval/ benchmark/ oracles/ lipids/   evaluation                     PINNED, do not move
  experiments/                        SEE BELOW: 21 library + 281 drivers
examples/                             experiment drivers             RISKY (see 3)
tests/                                pytest                         SAFE
tools/                                repo tooling                   PARTLY SAFE (102 of 250 pinned)
modal_apps/                           cloud entrypoints              PARTLY SAFE (87 of 286 pinned)
docs/ configs/ diagnostics/ repro/    records                        APPEND ONLY
archive/                              retired, provenance preserved  APPEND ONLY
```

The reference layout would put `experiments/` outside the library package. That
move is RISKY and must not be made as stated; section 3 says why.

## 3. Modularity

### The seam is not where the directory names suggest

Import edges between subpackages show `rewrite` (34 modules, 550 inbound) and
`chem` (7 modules, 342 inbound) as the core, and `experiments` (302 modules,
1,080 outbound) as overwhelmingly a consumer. That reads like a clean cut.

It is not. **21 modules under `experiments/` are imported by 37
library-core modules**, led by `whole_ring_plan` (11 core dependents) and
`production_successor_kernel` (6, and the README already states it must not be
edited because it fixes `process_identity_sha256`). Those 21 are
library code filed as experiments. Moving `experiments/` out wholesale would cut
across a real dependency, which is worse than the mess it replaces.

The correct target is therefore: promote those 21 into the
library, and only then treat the remaining 281 as drivers. Both halves are
inside `src/`, so both are blocked by section 1.

### Other findings

- `src/compose_v4/control/archive/` places retired code inside the library
  package, and two of its subpackages carry no `__init__.py`, so they are the
  only modules that fail to import for a structural reason rather than a missing
  optional dependency.
- Oversized modules are listed in the JSON. The largest is
  `model/factorized_tracelet_rate_model.py` at 5,902 lines, which is pinned into
  the Process-V2 identity and cannot be split.
- The 1,192 Python files under `diagnostics/` are almost entirely SEALED SOURCE
  CAPSULES: byte-exact frozen copies of `src/compose_v4` for two scored
  campaigns. They look like duplication and are the opposite: they are the
  record of the exact source a scored run executed. They must not be touched.
- `diagnostics/genmol_prescreen/` and `diagnostics/ivg_src/` are vendored
  COMPETITOR source sitting in the results directory. `third_party/` is where
  they belong, but they are referenced from committed artifacts, so this is
  reported rather than moved.
- `tests/test_t4_route_diagnosis.py` imports `from diagnostics.t4_route_diagnosis.analyze`,
  so a test depends on analysis code living in the artifacts directory.

### Cross-module private imports

Judged case by case. The repository's own standing rule is that a SHARED
enumerator used by two callers is the correctness guarantee, and duplicating it
would be the defect: `pendant_graft_candidates` is used by both the model mask
and the corruption path precisely so a teacher graft always lands in the mask it
is scored against. Those are correct. No case was found where a cross-module
private import should instead be duplicated.

## 4. Dead code

Absence is established two ways, because neither signal alone is sufficient: a
deferred import never appears in `sys.modules`, and a local variable sharing a
module's name reads as a grep hit.

**No source, driver or app file is unreferenced**: `src/` 0, `scripts/` 0,
`modal_apps/` 0 of 1,768 unreferenced tracked files. The unreferenced set is
concentrated in `diagnostics/` (897), `tests/` (386), `upload/` (260) and the
manuscript directories. Those are results and submission bundles, not dead code.

## 5. Lint

`ruff check .` reports 4341 findings, not the ~314 that refers to `src/`
alone. Bucketed by pinned state, **758 are in pinned files and
3583 are not** - but the aggregate fingerprint in section 1 means the
unpinned count overstates what is safely fixable: of the 514 findings in `src/`,
513 are in pinned files and the one that is not is still covered by the tree
digests. `archive/` (352) and `third_party/` (212) must stay immutable on
separate grounds.

The genuinely fixable surface is `tests/` and the unpinned part of `tools/`.

## 6. Weight

Pack is 193.30 MiB; the tracked tree is about 166 MB, dominated by `artifacts/`
(101 MB) and `upload/` (61.7 MB). The reported ~382 MB of committed PMO campaign
artifacts is NOT on this branch: `diagnostics/pmo_population_controller_v1*`
here is 0.1 MB across 29 files. Reported, not removed.

## 7. Reproducibility

Traced artifact -> contract -> pinned hashes -> commit for T4 and PMO.

- 127,098 pins across 432 artifacts: 110,208 resolve,
  6,014 are stale, 10,876 address something absent. A stale pin is
  normally an immutable launch record; re-pointing one would forge the record.
- **Already-broken links: 0.** A first pass reported 26 files as
  permanently lost because `git log --all` searches LOCAL refs only. All 26 live
  on three remote branches never fetched here, and 64 of 64 checkable pins match
  their branch blobs EXACTLY. `origin` carries 168 heads against 122 local refs,
  so a local-only search is not authoritative about what exists.
- **8,828 pinned artifacts exist only as UNTRACKED files in a single
  checkout.** They are referenced by hash from committed contracts, absent from
  git, and single-copy on one disk. This is the real reproducibility exposure.
- Three known off-Git objects have fewer than two verified backups.
- The recorded note that two Modal volumes share the name
  `compose-v4-artifacts` undercounts: there are THREE, across the `rahul`,
  `nitya` and `rahul-94866` profiles, and the created-by column does not
  separate them. The `nitya` volume, which the note omits, is the one the
  running T4 campaigns are recorded as using.

Details: `repro/pinned_paths_v1.json`, `repro/external_artifacts_v1.json`,
`repro/environments_v1.json`, `repro/capability_manifest_v1.json`.
