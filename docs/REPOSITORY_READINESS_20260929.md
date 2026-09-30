# Repository readiness repair

Date: 2026-09-29. Base: `8aa27f7f3bb8935dd7b1d8d8b48a92dfba2b294f`.

## Objective and scope

The user authorized repairing the test suite, assessing whether its expectations
are valid, and preparing the repository for other researchers to use. COMPOSE's
scientific identity remains the learned executable stochastic rewrite process
over complete molecular graphs. Its primary output is the marked transition law
and the molecular successors induced by exact execution. The paper evaluates
reference learning, constrained construction, program selection, molecular
editing, and feedback-driven optimization against their declared controls.
Representation, chemistry, size, split, baseline, and evaluation support remain
those of each versioned experiment.

This milestone concerns local software correctness and usability. It authorizes
test and implementation repairs, packaging, and documentation. It makes no new
benchmark claim and authorizes no publication, scored experiment, training,
docking, cloud deployment, or recovery of live campaigns. Existing worktrees,
user changes, frozen scientific results, and authorization identities are
preserved.

## Acceptance

1. The suite collects in the documented supported Python environment.
2. Every observed failure is assessed against the intended public behavior and
   scientific contract before changing either implementation or expectation.
   Repairs retain tests for validity, support, leakage, provenance, accounting,
   determinism, and failure handling.
3. Tests use small offline fixtures. A required external asset is documented and
   distinguished from an ordinary software defect. No failing test is hidden by
   a broad exclusion or a relaxed gate.
4. Focused checks pass for each repair; the final candidate receives repository
   verification and one complete suite run with its real exit status captured.
5. Installation, import, CLI help, and available offline benchmark workflows
   work from a source export independent of historical working directories.
   The PMO and T4 integration dependency identified in the handoff remains
   explicit until their producer code is integrated and tested.
6. The record reports the actual final passes, failures, errors, skips, runtime,
   environment, source revision, and unresolved input access requirements.
7. Changes are inspected and committed in scoped local units. No push.

## Verification environment

Core checks use Python 3.11 with RDKit 2024.3.5, Torch 2.4.0, NumPy 1.26.4,
SciPy 1.13.1, and NetworkX 3.3. PMO oracle checks, where applicable, require
their distinct RDKit 2023.9.6 environment and do not score molecules during
this milestone. The existing laptop `.venv` uses Python 3.12 and is not the
paper chemistry environment.

The other agent's 4,688 passes, 231 failures, and 101 errors describe a reported,
time-limited run on a different branch. They are not this checkout's baseline
and do not by themselves prove that every failure was pre-existing.

## Decision log

- Begin by reproducing collection failures and identifying failure families.
  Diagnose on the current checkout, with focused tests during repairs.
- Judge readiness using installation and usable experiment interfaces as well
  as test assertions. A passing assertion that does not protect meaningful
  behavior is not sufficient evidence.
- Added the missing PyYAML runtime dependency and a pinned Python 3.11 core/test
  installation recipe. The isolated local environment passed dependency
  compatibility checks; it is not a substitute for the separate PMO oracle
  kernel.
- Replaced tests that relied on Python's private typing internals or validated
  only their own mock arithmetic. The QED sampler tests now observe the actual
  local transition probabilities, while executor aggregation tests require
  real many-to-one mark aliases and fail on interface drift.
- Preserved the historical E6 receipts and tested their refusal under changed
  source identities. Fresh bounded fixture audits test current behavior without
  re-sealing or promoting the old scientific artifacts.
- Corrected a readiness audit that ignored its requested repository root and a
  training-report consistency check where float32 reductions crossed an input
  extremum by up to two ULPs. Frozen scientific thresholds remain strict.
- Removed a full-checkout recursive scan from a source-identity test; its
  replacement still refuses omitted serialized inputs. Removed a vacuous
  golden-path assertion. An attempted shared cycle-rank refactor touched frozen
  scientific source files; it was reversed after the scope audit. The current
  source bytes match the pre-refactor revision.
- A complete clean-clone diagnostic in the pinned core environment finished
  with **5,301 passed, 70 failed, 6 skipped, 1 xfailed** in 58m59s. The actual
  pytest exit status was 1. Its log and JUnit XML are at
  `diagnostics/repository_readiness_v1/clean_clone_full_final.{log,xml}`.
  This is a baseline, not suite sign-off for the subsequent repairs.
- The first complete post-repair clean-clone run at `6f55922b` finished with
  **5,347 passed, 10 failed, 18 skipped, 1 xfailed** in 59m03s. All ten
  failures required gradients. An in-memory after-test probe found that the
  T4 population fixture executed an inference app body that disabled gradients
  process-wide in pytest. The test harness now restores the caller's gradient
  mode in a `finally` block without editing the frozen app. The ten failing
  test modules passed together (42 tests), and the complete T4 test group
  passed with the after-test gradient probe (272 tests).
- The final full clean-clone run at `4ca47afa` passed with **5,357 passed,
  0 failed, 18 skipped, 1 xfailed** in 58m55s and actual pytest exit code 0.
  The log, JUnit XML, and captured exit code are local under
  `diagnostics/repository_readiness_v1/clean_clone_final_4ca47afa_20260929.*`;
  their SHA-256 values, source revision, environment, and skip categories are
  recorded in `diagnostics/repository_readiness_v1/clean_clone_verification_20260929.json`.
  The read-only Process-V2 hash-chain verifier also exited 0 with status
  `AGREES`; it retains warnings about historical or unverifiable pointers.
- `ruff check src/` currently reports 494 legacy findings. Touched Python
  files pass scoped Ruff lint. Scoped format checking passes 31 of 32 touched
  Python files; `tools/t4_launch.py` retains one unrelated argument-layout
  difference that is also present at the pre-repair base revision. Formatting
  that historical launcher would change its physical identity without fixing
  this milestone's behavior, so it is not silently reformatted. Mass
  formatting hash-bound historical sources is outside this repair.

## What the complete diagnostic established

The 70 failures were not 70 unrelated defects. Forty-four gradient tests were
contaminated by `load_package` disabling PyTorch gradients process-wide. One
test each exposed a leaked temporary import path, a stale literal source-code
assertion, and exact comparison of rounded float32 values. These four causes
have targeted implementation or test repairs. Their focused tests passed.
The first complete rerun exposed the separate T4 test-fixture leak described
above. Its focused repair and the final full clean-clone suite passed.

Nine RingCore failures and one T4 qualification failure were tests that treated
historical source identities as if they were the current checkout. The frozen
RingCore operator and cycle hashes reproduce exactly from Git revision
`a7546e2`; the frozen panel sampler hash reproduces from `65e0feca`.
The T4 qualification's registry hash reproduces from the parent of
`d613c9b8`, before a later registry repair. Tests now exercise the frozen
identity and require the current checkout to refuse those historical inputs.
No frozen hash, result, or source file was re-pinned.

Five PMO recovery tests need the original pre-recovery campaign tree, and four
PMO source-parity tests need the hash-bound InVirtuoGen source checkout. Those
ignored inputs are absent from a source export. Tests now identify the exact
missing prerequisite and skip there, or fail with
`COMPOSE_REQUIRE_EXTERNAL_ASSETS=1`. The local PMO recovery folder has since
advanced: three of those tests still fail against it. A read-only audit found
that all 8,828 census-listed original files (4,539,938,269 bytes) remain
present with their exact recorded SHA-256 hashes; the folder also contains
8,236 additional post-recovery files. This supports materializing an exact
pre-recovery view by selecting the census-listed bytes, but that view has not
yet been built or validated. The old snapshot is not silently substituted from
the later state.

Two legacy PMO tests require PyTDC 0.3.6, not the core kernel. One serialized
pan-lung test requires the exact stack in its bundle manifest. They skip when
the prerequisite kernel is absent and fail with
`COMPOSE_REQUIRE_ALTERNATE_KERNELS=1`. The legacy property-program source hash
does not match the available PyTDC 1.1.15 environment; its 0.3.6 kernel remains
unverified locally.

The clean export's 18 skips are five fragment-generation asset checks, five
PMO pre-recovery artifact checks, four InVirtuoGen source-parity checks, two
legacy PMO-kernel checks, one pan-lung-kernel check, and one CUDA pinned-memory
check. The single strict expected failure is a known Process-V2 import-boundary
defect: its source resolver still imports a V1 loader that revalidates the live
V1 identity. That is an explicitly deferred Wave-2 repair, not a passing
invariant or a hidden skip. The final full run confirmed the same 18 skip
classifications and one strict expected failure.

The slot-safety scan initially flagged Dynamic-v2.1's `atom_types > 0` context
calculation. A code-path audit changed that verdict: the expression selects
**occupied slots** (including a SCAR), which is the appropriate predicate for
the graph-topology part of that context. The reported Active8 route starts from
SMILES graphs without SCARs; admissible insertion/restatement exclude SCAR,
deletion writes NULL, and the other primitive edits do not create it. Replacing
the expression with `is_element` would change the meaning of a scar-bearing
topology calculation and the frozen PMO source identity. The attempted edit was
reversed byte-for-byte (SHA-256
`d5f4d432f3a51d49a1c6f962e079a5ae729c731031998d78caf14f1d3891b743`).
The test now ratchets this one historical site and checks the producer-domain
invariant and context behavior. This is a corrected **test false positive**, not
a repaired PMO defect. A separate future path that supplies scar-bearing
states would need its own audit; none is asserted here.

Post-repair focused checks: 53 RingCore/T4 provenance tests passed; 93 tests
covering the gradient, import, PMO, pan-lung, and assertion repairs passed with
3 separately identified kernel skips; 2 prerequisite-gate tests passed. The
slot-safety module passed all six tests after that audit. These focused checks
are supplemented by the final full-suite result above.

The documented fragment CLI verified seven artifact hashes and the complete
saved metric panels for four independent tasks. Its offline table command
produced `tables.json`, `tables.md`, and `provenance.json` in a fresh temporary
directory; the observed motif, decoration, linker/morphing, and superstructure
quality means were 42.633, 36.700, 31.533, and 39.033 percent. This reproduces
saved-table reductions only, not generation from model weights or raw molecular
evaluation. The isolated Python 3.11 environment passed `uv pip check` for 60
installed packages. The command was run with `PYTHONPATH=src` so imports came
from this checkout rather than the environment's earlier editable clone. The
fragment `verify` command also passed from the final `4ca47afa` clean clone:
seven artifact hashes and four complete saved metric panels.

## Release status

Not yet ready to present as a fully reproducible public repository. The current
branch still has missing historical recovery input, unverified legacy kernels,
and an unresolved integration choice between this
checkout and the separately developed PMO/T4 branch. A fresh source export
passes the complete core suite, but its 18 skips, one known expected failure,
and the provenance and input-access boundaries below preclude a claim of full
paper-result reproduction. No branch was merged, pushed, or published during
this repair.

The PMO/T4 branch at `bc8cc421` contains useful offline reproducers, but its
paper-facing guide still asserts that one checkpoint is required by every
benchmark. That is inconsistent with the experiment-specific reference usage
recorded for this manuscript. Treat its tools as candidates for selective
integration, not its documentation as an authority to overwrite this branch's
paper-to-code map.

## Paper-release path audit

A passing test suite on `compose-iclr` would not yet make a reviewer release.
The compact-evidence paths are currently split as follows:

| Submitted result | Preserved evidence | Clean `compose-iclr` export |
| --- | --- | --- |
| Reference likelihood | `diagnostics/editing_v2_experiment1_reference_law_paper.json` | Summary present; raw transition corpus external |
| Fragment benchmark and ablations | `experiments/fragments/manifest.json` and its pinned reductions | Saved metrics reproducible; full generation assets external |
| QED editing | `fragment-qed-ablations-20260925:diagnostics/qed_griddd_arm_a_rederivation_v1.json` | Final rederivation absent |
| PMO-1K headline | Historical `99d87b65:diagnostics/pmo_ablation_frozen_v1/inputs/pmo_1k_final.json` | Per-seed reduction absent; raw receipts external |
| PMO A/B ablation | `pmo-chain-ablation-20260925:diagnostics/pmo_abc_ablation_v1/reduction_v1.json` | Reduction and arm producer absent |
| T4 table | `bc8cc421:diagnostics/T4_FROZEN_RESULT_v1.json` | Frozen table and producer absent; docking receipts external |

The later `pmo_1k_final.json` per-seed reduction yields means of 0.563212
final Top-10 and 0.481770 AUC over the declared 22 objectives, matching the
manuscript's rounded 0.563 and 0.482. It does **not** reproduce every printed
objective cell. Its JNK3 third seed has Top-10 0.210 and AUC 0.1886,
giving three-seed means 0.246 and 0.209833, while the manuscript prints 0.245
and 0.208. The earlier `pmo_1k_prov.json` instead marks that same seed
`partial: 815` with Top-10 0.206 and AUC 0.183914; combining it with the two
other seeds gives 0.244667 (sample SD 0.0406) and 0.208271 (sample SD 0.0317),
which round to **both** printed JNK3 mean±SD cells. The evidence therefore
points to a provisional 815-call snapshot being copied into a table captioned
as 1,000 calls. The source files are preserved at Git revision `99d87b65`.
The available direct receipt-count verification covers the six ablation
objectives, not JNK3, so the later JNK3 seed's receipt count still needs an
independent check. Do not relabel the printed row as a completed 1,000-call
measurement in the release; record a correction or explicit caveat after
checking the original receipts.

This checkout contains the fragment saved-result reducer and its pinned small
artifacts, but it does not track the submitted PMO/T4 producer entry points
`scripts/pmo_reward_adaptive_canary.py`,
`modal_apps/pmo_fibercontrol_targets_app.py`,
`src/compose_v4/experiments/t4_fiber_campaign.py`, or the frozen T4 table
artifact. Those paths are present on the separate PMO/T4 development line
(except that its later commits removed the PMO A/B reduction from the branch
tip). The A/B result is preserved on the `pmo-chain-ablation-20260925` branch.
The final QED rederivation used by the local compact-evidence draft is on
`fragment-qed-ablations-20260925`. The tracked
`docs/OFFICIAL800_QED_CURVE.json` is explicitly incomplete (`complete: false`,
199 landed sources) and must not be substituted for the submitted 800-source
result.
The two lines have 23 and 615 commits, respectively, since their merge base
`101cc73d`; this is a scientific-lineage integration, not a cosmetic directory
merge. A no-checkout `git merge-tree` dry run found eight conflict paths,
including the source-identity ledger, a T4 app, inference loading, three PMO
experiment modules, and a T4 contract builder. No branch was merged or
rebased during this audit.

A separate no-checkout merge preview between `bc8cc421` and
`pmo-chain-ablation-20260925` found four conflicts in the PMO app, scored
runner, Dynamic-v2.1 synthesis hook, and launcher. These are arm-defining
files. Resolving them into a new hybrid HEAD would not by itself reproduce the
historical A/B campaigns, whose identities depend on their launch revisions
and deployed images. The release map must point each result to its exact
producer revision and verified artifact, even if the default checkout offers
one convenient development interface.

The other branch's `experiments/paper/README.md` claims a single reference
checkpoint for every benchmark. Its PMO and T4 guides still point to
reproduction commands and data files removed by later commit `1f1c2c86`
following the owner's instruction to keep score artifacts out of Git.
Those pages are therefore not an authoritative release guide as currently
committed. A safe integration must carry verified producer code, exact result
reductions, and a correct experiment-specific input map together. It must
distinguish arithmetic reduction of saved metrics from re-running molecular
generation, oracle search, or docking.

There is also an untracked local `reviewer/` compact-evidence draft. Its
standard-library verifier passes the five included file hashes and recomputes
the fragment comparisons and the submitted 14 completed PMO A/B pairs. This
is an arithmetic check on projected metric rows, not raw experiment replay.
The draft currently omits T4, records the submitted PDF hash as pending, and
prints an instruction to read `KNOWN_GAPS.md` although that file is absent.
It has not been added to Git or presented as a completed reviewer package.

One additional table-accounting check is pending resolution with the T4
producer: `diagnostics/T4_FROZEN_RESULT_v1.json` on that branch declares a
250-call ceiling, while its 30 rows record `charged_calls` from 0 to 249 rather
than 250 each. The submitted text's phrase "using 250 evaluations per lead"
should not be interpreted as measured 250 calls for every row without checking
the producer receipts. Specifically, FA7 starting molecule 1 at similarity
floor 0.6 has `compose: null` and `charged_calls: 0` in the frozen artifact,
while the submitted table prints -6.4, equal to the starting molecule's score.
That displayed fallback must be labeled as such in a release reducer; it is not
a measured COMPOSE endpoint from that cell. The frozen artifact remains unchanged.

The intended release gate is end-to-end and has separate outcomes: (1) exact
submitted-manuscript identity; (2) source export plus pinned environment;
(3) hash-verified input availability; (4) task-specific saved-result reduction
and, where supported, generation parity; (5) full-suite and focused scientific
invariants; (6) an explicit list of non-distributed assets and unsupported
reproduction steps. Missing evidence is a failed or unavailable gate, never an
implicit pass.

The local manuscript candidates are distinct: `COMPOSE_ICLR_2027_revision4/main.pdf`
has SHA-256 `e05a8fff0da6201f77c29ca218d5c79f6acf505bc6101afd1797f07bc7ff3db8`,
and `COMPOSE_ICLR_2027_zip5_appendix_results/main.pdf` has SHA-256
`9a1117459ad5bc38e27c9c6bbe67593cd7fa6d6384db1f5e5041907fe92b41dd`.
Their corresponding copies under `output/pdf/` match byte-for-byte. The owner
has not yet identified which exact PDF was submitted. The PMO/T4 handoff calls
the zip5 directory the submitted one based on file timestamps and the deadline,
but supplies no submission receipt hash. Neither candidate hash is therefore
promoted to the release manifest yet.

## Safe integration sequence after this software gate

1. Confirm the exact submitted PDF bytes and choose the repository's release
   base. Keep historical PMO A/B and frozen T4 producer revisions separately
   addressable; do not resolve arm-defining conflicts into an invented producer.
2. Declare which checkpoints, catalogs, evaluator sources, raw receipts, and
   docking assets may be distributed. For non-distributed inputs, publish their
   expected identities and an honest access requirement, not a substitute.
3. Add task-specific offline reductions with explicit input hashes for the
   final QED, PMO-1K, PMO A/B, and T4 rows. Reconcile the JNK3 provisional
   seed and the FA7 zero-call fallback against their original receipts before
   marking the corresponding paper rows reproduced.
4. Test those reducers from a fresh source export and, only where all exact
   external inputs are available, run bounded generation or evaluator parity.
   Keep saved-metric arithmetic, full campaign replay, and docking reruns as
   distinct verification levels. Commit scoped integration units locally; do
   not push or publish until the owner approves the evidence and access map.
