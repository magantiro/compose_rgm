# Cleanup report v1

Preservation migration on branch `repo-hygiene-modularization-20260923`, from
baseline `061ead93` (tag `pre-cleanup-2026-09-23`).

Machine-readable gate: `repro/cleanup_report_v1.json`, regenerate with
`python3 tools/repro_gate.py --base 061ead93`.

**Rollback: `git checkout 061ead93` (tag `pre-cleanup-2026-09-23`).**

## Outcome in one line

No file was moved, renamed, deleted or archived. The pass produced the
measurement that says why moving anything in `src/` is not safe, the manifests
needed to re-run a published number, and the gates that make a future pass
falsifiable.

## What changed

Every change is additive except one README edit. Nothing under `src/`,
`scripts/`, `recipes/`, `archive/`, `third_party/` or any existing artifact was
touched.

| area | files | what |
|---|---|---|
| `tools/` | 8 new | the measurement and gate instruments |
| `diagnostics/repo_hygiene/` | 6 new | pinned set, pin resolution, capability and entry-point baselines, the audit |
| `repro/` | 7 new | pinned paths, external artifacts, environments, capability manifest, reproducibility graph, migration contract, this report |
| `tests/` | 1 new | T4/PMO reachability smoke, mutation-proven, green under two production kernels |
| `README.md` | 1 edit | states the content-addressing constraint a newcomer hits first |

## Merge gates

| gate | count | note |
|---|---|---|
| `pinned_historical_files_modified` | **0** | `git diff --stat 061ead93 HEAD -- <1,879 pinned paths>` is empty |
| `lost_public_symbols` | **0** | the before and after reports are BYTE-IDENTICAL, sha256 `85ab89636a352c1b...` |
| `lost_entrypoints` | **0** | 877 -> 884 entry points, 0 regressed, 0 vanished, +7 new tools |
| `lost_capability_groups` | **0** | 7 groups, all non-empty |
| `new_test_regressions` | PENDING | baseline suite still running; not claimed until run |
| `newly_broken_pins` | **0** | absent-pin count unchanged at 10,876, all traced |
| `external_artifacts_without_backup` | **3** | **NONZERO AND REPORTED, see gap section** |

`external_artifacts_without_backup` is the only measured line that does not read
zero. It is a pre-existing gap this pass measured rather than created, and
suppressing it would be worse than naming it.

One line reads PENDING rather than zero. An unrun check is not a passing check.

The entry-point re-measurement WAS run under load from the baseline suite, which
this repository warns manufactures phantom regressions. That risk is bounded
here: the probe's only load-sensitive failure mode is its 90 second per-file
timeout against imports that take 3 to 5 seconds, and the result was 0 regressed
and 0 vanished, so load could not have hidden a regression behind a timeout. Had
any file regressed, it would have been re-verified on an idle machine before
being reported.

## Findings, ranked by how much future error they prevent

### 1. `src/` is read-only in AGGREGATE, not per file (MEASURED)

1,879 of 8,805 tracked files are the subject of a sha256 pin, including 542 of
the 595 modules in `src/compose_v4`. Every subpackage except `experiments` and
`control` is 100 percent pinned, and those two are covered by the tree
fingerprints below, so the 53 unpinned modules are not a working surface.

The per-file set **understates** the constraint. Three fingerprints hash whole
directories, so an individually unpinned file is still covered and **adding** a
file moves the identity exactly as editing one does. Recomputed against a
perturbed tree:

- a whitespace-only edit to ANY file under `scripts/` moves the training run identity;
- a whitespace-only edit to an unpinned `src/` module moves two identities;
- adding a module under `src/compose_v4` makes every existing P50 successor-cache
  plan **raise** `plan implementation or policy is stale`.

This is why no library file was moved. It also means the requested
`compose_v4/api/` facade is the most expensive of its three placements.

### 2. The facade, costed (MEASURED)

| placement | P50 cache plans refuse to load | train / t4-reset identity moves |
|---|---|---|
| `src/compose_v4/api/` (as requested) | **YES** | YES |
| `src/compose_api/` | no | YES |
| `api/` outside `src/` | no | no |

The P50 cache is recorded at roughly 1,025 core-hours to rebuild and lives on a
Modal volume rather than in git. **Not added in this pass**; reported so the
placement can be chosen knowingly. `api/` outside `src/` costs nothing in
identity terms and needs one `pyproject.toml` entry.

### 3. The seam is not where the directory names suggest (MEASURED)

Import edges make `experiments/` (302 modules, 1,080 outbound, 50 inbound) look
like a pure consumer and therefore cleanly separable. **21 of those 302 modules
are imported by 37 library-core modules**, led by `whole_ring_plan` (11
dependents) and `production_successor_kernel` (6). Those 21 are library code
filed as experiments.

A flow_matching-shaped move of `experiments/` out of the package would cut
across a real dependency. The correct sequence is to promote the 21 first. Both
halves sit inside `src/`, so both are blocked by finding 1.

### 4. The off-Git exposure is ONE directory, 11.4 GB, deliberately ignored (MEASURED)

Of 127,098 pins in 432 artifacts: 110,208 resolve, 6,014 are stale (normal for
an immutable launch record), and 10,876 address something absent. Of the 8,835
repo-shaped absent paths, **8,828 are under a single directory**:

| | |
|---|---|
| path | `diagnostics/pmo_dynamic_v21/runs/` |
| pinned paths it holds | 8,828 |
| on disk | 17,064 files, **11.4 GB** |
| only known location | `/Users/rmaganti/compose_rgm_git`, untracked |
| why untracked | `.gitignore:55`, a deliberate exclusion |

Verified two ways: 300 of 300 sampled unresolved paths are ignored by that one
rule, and 40 of 40 sampled are present-but-untracked in the main checkout and in
no local ref's history.

The exclusion is a policy decision and the exposure is a separate consequence of
pinning those paths by hash; the two were decided independently and only the
second is a problem. **The remedy is not to commit it** - 11.4 GB would bloat the
repository - but to copy it to two durable locations and verify the sha after
copying. Operational blocker measured the same day: this disk had about 8 GB
free, so a local second copy does not currently fit.

This is the real reproducibility exposure, and it is larger and far more
specific than anything in the directory layout.

### 5. A correction I had to make to my own finding (MEASURED)

The remaining 26 paths were first reported as **permanently lost**, on the
strength of `git log --all` finding them in no commit on any branch. That was
wrong: `git log --all` searches LOCAL refs, and origin carries 168 heads against
122 local refs. All 26 live on three unfetched remote branches, and **64 of 64
checkable pins match their branch blobs exactly**. Nothing is lost.

Recovery: `git fetch origin "+refs/heads/<branch>:refs/remotes/origin/<branch>"`
for `codex/compose-baseline-qualification`, `codex/compose-constraints-hard`,
`codex/compose-multiobjective-package`.

### 6. A third Modal volume shares the name (MEASURED)

The recorded note that two volumes share the name `compose-v4-artifacts`
undercounts. There are **three**:

| profile | created |
|---|---|
| `rahul` | 2026-07-17 03:16 EDT |
| `nitya` | 2026-07-29 14:41 EDT |
| `rahul-94866` | 2026-08-09 00:13 EDT |

The `created by` column does not separate them, so identity is profile plus
creation date. The `nitya` volume, which the existing note omits entirely, is
the one the running T4 campaigns are recorded as using.

### 7. Reproducibility graph: zero broken links (MEASURED)

Traced for the frozen T4 panel, the T4 controller campaigns, the PMO scored runs
and the PMO diagnostics. Every payload self-hash that exists verifies, every
branch name referenced resolves on origin, and the only stale pin in the T4
rescue chain is `control/dynamic_program_synthesis.py`, whose three recorded
hashes (`4f254eb6` -> `a1370685` -> `ece5dcd0`) track the region-law repair and
this branch's own work on that file.

## Open gap, not closed by this pass

`external_artifacts_without_backup = 3`. These have fewer than two verified
durable copies:

| object | verified copies | note |
|---|---|---|
| `ringcore_a7546e2_best.pt` | 1 (`~/compose_ckpt_backup/`) | its Modal run directory is already gone from both `rahul` profiles |
| `pmo_1k_ab_molecules` | 0 | Modal volume only; local artifacts carry counters, so a post-hoc question about the molecules cannot be answered |
| `lineage_b_denovo_run` | 0 | on the `rahul` 2026-07-17 volume, NOT the identically named 2026-08-09 one |

Closing it means copying each to a second durable location and verifying the
sha **after** copying. That is a data-movement task with an owner decision about
destination, not a code change, so it was reported rather than improvised.

## What was deliberately NOT done

- **No file moved, renamed or deleted.** The 2026-08-19 reorganization reached
  the same conclusion independently and moved nothing;
  `archive/ARCHIVE_MANIFEST.md` records that evidence. Nothing was archived, so
  no archive manifest entry was required.
- **No lint fixes.** 4,341 findings repo-wide, but 513 of the 514 in `src/` are
  in pinned files and the remaining one is covered by the tree digests;
  `archive/` (352) and `third_party/` (212) must stay immutable on separate
  grounds. The genuinely fixable surface is `tests/` and part of `tools/`, and
  cosmetics come last.
- **No facade.** See finding 2.
- **No branch work** of any kind.
- **No semantic change.** Nothing in this pass can alter an executed result,
  which is why the test comparison is a preservation check rather than evidence
  about the controller.

## Kernel portability of the new guard

The reachability smoke test passes under the laptop `.venv` (rdkit 2026.03.6,
python 3.12) and under the PMO production kernel (`~/compose_pmo_pinned_env`,
rdkit 2023.09.6, python 3.11): 6 passed in both. It asserts import reachability
and call signatures, so it carries no chemistry dependence, which is what a
capability gate should look like. It is deliberately NOT a numerical
equivalence test; that belongs with the kernel and is pinned into the process
identity.

## Before / after, measured

| quantity | baseline `061ead93` | branch HEAD | delta |
|---|---|---|---|
| `compose_v4` modules | 595 | 595 | 0 |
| importable modules | 590 | 590 | 0 |
| public symbol pairs (import view) | 17,081 | 17,081 | **0 lost, 0 gained** |
| public symbol pairs (AST view) | 17,025 | 17,025 | **0 lost, 0 gained** |
| capability report sha256 | `85ab8963...` | `85ab8963...` | **byte-identical** |
| entry points discovered | 877 | 884 | +7 (new tools) |
| entry points importing | 821 | 829 | **0 regressed, 0 vanished** |
| collected test nodes | 6,296 | 6,302 | +6 (new smoke tests) |
| pinned files modified | - | - | **0** |

The capability report is byte-identical before and after, which is stronger than
the superset the gate requires: not one symbol, module or import outcome moved.

Full-suite after-state at HEAD: **297 failed, 5,893 passed, 2 skipped, 4 xfailed,
102 errors** in 44m53s under the laptop `.venv`. The matching true-baseline run
at `061ead93` is in a detached worktree; the node-set comparison lands in
`repro/test_fingerprint_after_v1_vs_baseline.json`.

Attribution already established for the areas this pass touches: the identical
`region or ring or fiber or completion_law or region_law` selection run at
`061ead93` and at HEAD gives **31 bad nodes on each side with identical sets** -
zero regressions, zero repairs
(`diagnostics/repo_hygiene/focused_suite_attribution_v1.json`).

## Golden and smoke results

| check | result |
|---|---|
| T4 path reachable (campaign -> fiber -> synthesis -> region law) | PASS |
| PMO path reachable (controller -> families -> realization) | PASS |
| executor path reachable | PASS |
| `Fiber` still takes `delta` | PASS |
| `synthesize_dynamic_program` still accepts `region_law` | PASS, mutation-killed |
| PMO `restore()` accepts the same arm flags as `__init__` | PASS |

Six tests, green under the laptop `.venv` (rdkit 2026.03.6) and the PMO
production kernel (rdkit 2023.09.6). Zero oracle calls, no docking, no Modal.

## Archived files

**None.** Nothing was moved, so `archive/MANIFEST.json` needed no entry.
`archive/ARCHIVE_MANIFEST.md` already records that the 2026-08-19 reorganization
reached the same conclusion and also moved nothing, with its evidence.

## Verification commands

```bash
git diff --stat 061ead93 HEAD -- $(python3 -c "import json;print(' '.join(sorted(json.load(open('repro/pinned_paths_v1.json'))['files'])))")
python3 tools/repo_capability_baseline.py --out AFTER.json --compare diagnostics/repo_hygiene/capability_baseline_061ead93.json
python3 tools/repo_entry_point_check.py  --out AFTER.json --compare diagnostics/repo_hygiene/entry_point_baseline_061ead93.json
python3 tools/repro_graph.py --out repro/reproducibility_graph_v1.json
python3 tools/repro_gate.py --base 061ead93
```
