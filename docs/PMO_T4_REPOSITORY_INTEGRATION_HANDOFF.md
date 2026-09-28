# PMO and T4 repository integration handoff

Date: 2026-09-28. Scope: local organization and offline reproduction of existing
results. This is an implementation brief, not a new experimental authorization
or a statement that PMO/T4 reproduction has already passed.

## Ownership and objective

The agent that ran PMO and T4 should reconstruct their exact run lineage and
implement their task interfaces. The fragment integration supplies the repository
pattern; it does not establish the identity of another experiment's controller.

COMPOSE's scientific object remains executable molecular transformations on its
declared graph support. PMO tests feedback-driven optimization; T4 tests
optimization from starting leads under endpoint constraints. Relevant comparisons
include external benchmark methods, structured proposals versus uniform legal
chains, and the secondary created-atom-binding intervention. Preserve the exact
representation, support, baselines, budgets, and claim boundaries of each producer
revision. This organization task makes no new scientific claim.

The practical goal is a normal, modular research repository: a new user can find
the correct experiment, verify its inputs, reproduce its saved tables, and see
exactly what is needed to run its implementation. Do not create another reviewer
folder, duplicate repository, generic workflow framework, or prose-only audit.
Deliver working offline commands and tests, then document the remaining runtime
and asset-access gaps.

## Read the current work

Primary checkout: `/Users/rmaganti/compose_rgm_git`, branch `compose-iclr`.
The inspected fragment integration ends at
`163afab35fc56636f768a6209cd22d7f6223fa84`.

Read the applicable repository instructions first, then:

1. [Fragment task guide](../experiments/fragments/README.md).
2. [Fragment generation guide](../experiments/fragments/GENERATION.md).
3. [Integration scope and decision log](FRAGMENT_REPOSITORY_INTEGRATION.md).
4. `experiments/fragments/manifest.json`, `assets.json`, and
   [the source-preservation explanation](../experiments/fragments/runtime/README.md).
5. `src/compose_v4/experiments/fragments/` and its focused tests.
6. [Measured runtime verification](../diagnostics/fragment_runtime_integration_v1/README.md).

Useful local commits, oldest first:

| Commit | Purpose |
| --- | --- |
| `fa34d906` | Preserve frozen fragment metric artifacts and provenance |
| `b094879d` | Portable saved-result reduction interface |
| `6027450b` | Verified benchmark and ablation reductions |
| `c6125456` | Preserve isolated runtime sources and saved fixtures |
| `e8a4f3e7` | Offline asset and generation interface |
| `163afab3` | Runtime parity receipts and verification boundaries |

These commits have not been pushed. A different chat in the same checkout sees
the files immediately. A different Git worktree shares commits, not working-file
updates. Read this checkout directly or inspect a committed file without changing
the campaign checkout:

```bash
git -C /Users/rmaganti/compose_rgm_git show 163afab3:experiments/fragments/README.md
```

Do not merge or cherry-pick the whole integration into a frozen launch worktree.
Preserve those worktrees as sources of evidence. Coordinate one writer for the
integration checkout; keep your additions task-scoped and avoid changes to the
fragment package or shared core without a separately agreed dependency repair.

The repository start page and paper-to-code map describe older campaigns as well
as durable conventions. Read their dates and resolve producer lineage against
receipts; do not mistake their historical operational status for today's status.
The user's 2026-09-27 campaign handoff is a search map, not a replacement for
artifact verification or current authorization.

## Hard boundaries

- No publication, push, remote-URL change, cloud access, deployment, training,
  campaign launch/resume, new oracle call, or redocking under this brief.
- No deleting, pruning, moving, rewriting history, or overwriting existing files.
  Preserve dirty and untracked work. Do not use blanket staging.
- Never change a frozen configuration, authorization hash, proposal law, executor,
  eligibility predicate, canonicalization rule, seed, or evaluator to make the
  organized path work. Historical paths must remain valid.
- Do not edit the submitted manuscript or replace its reported cohort with later
  results. Record data/manuscript discrepancies rather than modifying evidence.
- Preserve `.claude/context/learnings.md` as an append-only ledger.
- Missing source assets, unresolved producer identities, and incompatible runtime
  revisions are explicit blockers for the affected command, not reasons to guess.
  Continue the independent offline work that does have verified inputs.

## Layout to implement

Use the fragment pattern, subject to checking existing names and dependencies:

```text
experiments/pmo/                 task guide, evidence/asset manifests, environment
experiments/t4/                  task guide, evidence/asset manifests, environment
src/compose_v4/experiments/pmo/   thin CLI, validation, pure reduction, reporting
src/compose_v4/experiments/t4/    thin CLI, validation, pure reduction, reporting
tests/test_pmo_*.py              focused offline behavior and provenance tests
tests/test_t4_*.py               focused offline behavior and provenance tests
diagnostics/<task-specific>/     preserved inputs and versioned integration results
```

These PMO/T4 interfaces are proposed, not already implemented. Prefer small typed
functions and explicit data structures. Keep reusable chemistry, models, proposal
construction, and control in their responsibility-based modules. Separate file
loading, schema validation, numerical transformation, and output publication.
Do not copy domain logic into command-line drivers or import the fragment task
package as a general-purpose PMO/T4 framework.

First add compatible task adapters and navigation without relocating scientific
code. The fragment source archives solve a measured historical-core mismatch;
they are not a mandate to zip every task. Reuse compatible code normally. If the
exact producer needs an incompatible revision, preserve its minimal verified
dependency closure and use isolation, with original-byte hashes and bounded
parity. Do not merge incompatible kernels or silently use a newer development
implementation. Physical consolidation is a later decision after dependency
and pin checks, not the prerequisite for a clean user interface.

## Phase 1: retrace and preserve

Record a task-specific acceptance/decision document before implementation. Start
with `git status`, the producer revisions, applicable instructions, and local
file inventories. The following paths existed at handoff preparation; their
scientific contents have not been re-audited in this handoff:

| Local source | Inspected revision or role |
| --- | --- |
| `/Users/rmaganti/compose_pmo_chain` | `9466406691bff175092f524462f7dfc7316b7636`, PMO campaign worktree |
| `/Users/rmaganti/compose_t4_nitya` | `1d2e002988d73cc3cc7ed964088d7b36859bdd65`, T4 campaign worktree |
| `/Users/rmaganti/compose_pmo_ablation/inputs/pmo_1k_final.json` | Reported sole local copy of arm-A per-seed metrics |
| `compose_pmo_chain/diagnostics/pmo_abc_ablation_v1/reduction_v1.json` | Candidate PMO reduction; verify its actual included cells |
| `compose_t4_nitya/diagnostics/T4_FROZEN_RESULT_v1.{md,json}` | Frozen T4 record; inspect both, do not regenerate in place |

Both campaign worktrees contain untracked material. Inventory it explicitly,
including launch receipts. A clean tracked diff is not proof that all evidence
has been preserved in Git.

For each paper table or supporting result, trace:

```text
reported row -> declared cohort/reduction -> per-run receipts -> launch/image/code
             -> configuration -> model/data/oracle assets -> exact environment
```

Record paths, physical SHA-256 hashes, payload hashes where present, schema,
run/cell/arm identity, source revision, seed derivation, budget, completion state,
exclusions, aggregation rule, asset origin/access/license, and verification level.
Distinguish raw-receipt recomputation, reduction of saved metrics, and transcription
of an external published value. Preserve original bytes alongside derived schemas.

First make verified durable copies of the small load-bearing unversioned PMO
inputs, frozen configuration, and needed receipts. Do not move the originals.
Review copied content for secrets before tracking it. For bulk assets, inventory
sizes and available disk space before copying; the fragment pass had only about
12 GiB free. Do not duplicate entire worktrees or use `/private/tmp` as the only
source of truth. A same-disk copy is not an off-machine backup.

Before proposing any path change, search configurations, source, scripts, image
definitions, and manifests for literal references and hash pins. Include ignored
files: `diagnostics/` contains runtime inputs, not just disposable diagnostics.

## Phase 2: reproduce saved results, with no scoring

### PMO

- Keep the 22-objective PMO-1K benchmark, the submitted 14-pair A/B ablation,
  the subsequently reported 16-pair completed A/B reduction, and the secondary
  A/C comparison as distinct, explicitly named evidence sets. Verify each set
  that is actually available. Do not infer membership from a file's `final` name.
- The user-supplied submitted paper uses 14 pairs. The later campaign handoff
  reports 16 completed B and 17 completed C campaigns. This handoff does not
  certify those counts. Bind each available set to its exact inputs; never
  substitute later campaigns into the submitted-table recipe.
- For the six-objective ablation, verify the 1,008-charged-call allowance and
  metrics over the first 1,000 resolved receipts. Read each campaign's own
  `canary_v1.json`. Audit completion with terminal provenance and receipt counts,
  not `progress.json`, a plateau, or absence of an active process.
- Preserve the exact per-cell seeds, candidate-pool configuration, initialization,
  no-prescreen protocol, and A's reused-campaign identity. Use the A/B matched
  set, not an unnecessary A/B/C intersection. Specify cell versus objective
  weighting, sample versus population SD, and paired-test conventions.
- Record B's successful-structured-construction conditioning, inherited planned
  length, uniformity over legal native marks, state-dependent legal fiber,
  bounded early termination, arbitration RNG, and changed construction/reuse
  mechanisms. Matching seed labels does not give identical random draws.
- Document C's preserved recipe/source bindings, created-atom operand resampling,
  dedicated RNG, replay, and failures from its deployed revision. Keep its
  interpretation secondary; it does not isolate an additive share of B's effect.
- Retain the arm-dependent albuterol contextual-model error separately from
  preemption failures and incomplete budgets. Never impute a full-budget AUC.
  Keep exploitability limitations for predictor oracles such as GSK3B.
- Reproduce Top-10/AUC tables and curves from saved metrics/trajectories where
  those inputs exist. Preserve the exact AUC grid, origin, truncation, and budget.
  A saved summary alone does not verify the entire oracle/query pipeline.

Starting points to trace in the PMO worktree: `scripts/pmo_reward_adaptive_canary.py`,
`modal_apps/pmo_fibercontrol_targets_app.py`,
`tools/launch_pmo_fibercontrol_targets.py`, the B/C modules and hooks in
`src/compose_v4/control/`, `scripts/reductions_20260927/`, launch receipts, and
`diagnostics/pmo_armc_verification_v1/VERIFICATION.md`. Inspect the actual launched
path; an earlier 250-call pilot contract is not automatically the 1,008-call
ablation contract. Do not invent a complete hash chain where arm identity rests
on launch revision and deployed image rather than pinned synthesis files.

### T4

- Separate the exact submitted table, frozen replicate 1, subsequent replicate
  panel, support-expansion phases, and revival runs. Verify table-to-run lineage
  before choosing a default recipe; identify any unresolved submitted-file hash.
- Verify the five targets, three starting molecules, two deltas, stochastic
  replicate identities, eligibility gates, and per-run budget against executable
  contracts. Do not apply a single protocol retroactively to earlier controllers.
- Validate payload-wrapped JSON and its hashes before reading `payload`. Resolve
  delta by contract identity and its executable `delta` field, not prose alone.
- A checkpoint incumbent is not a completed result. Validate `result.json`,
  status, termination reason, and budget under the applicable contract. Preserve
  failures and missing values; do not relabel plateaued or interrupted cells.
- Declare whether an entry is a single-run best, mean of per-run best values,
  or another recorded statistic. Keep denominators and replica counts explicit.
  Preserve missing comparator cells; do not sum unequal-coverage columns.
- Keep starting-molecule identity separate from optimization and docking seeds.
  Do not redock published baselines or existing COMPOSE endpoints to tidy data.
- Preserve the docking adapter, receptor/binary identities, box, unseeded conformer
  stage, docking seed, search flags, scratch-collision guard, failure handling,
  and exact inclusive/exclusive eligibility predicates for each recipe.
  Do not promise bitwise redocking reproducibility where it was not established.

Starting points in the T4 worktree: `modal_apps/t4_unified_controller_app.py`,
its wrappers and contracts, `src/compose_v4/experiments/t4_fiber_campaign.py`,
`t4_docking_adapter.py`, the frozen result files, and original launch receipts.
Historical apps that produced frozen rows remain part of the record. Do not
choose a controller merely because its name resembles the latest app.

## Phase 3: environments and runtime access

Keep the reported PMO environment (Python 3.11, RDKit 2023.9.6, PyTDC 1.1.15
plus its compatibility shim) separate from the T4/fragment chemistry environment
(Python 3.11, RDKit 2024.3.5). Verify the full versions from producer receipts.
The laptop's newer `.venv` is not an equivalent chemistry kernel. Do not
recanonicalize stored identities under it.

Inventory PMO's ignored oracle assets at
`diagnostics/pmo_ivg_oracle_parity/ivg_oracle_assets/`, including licensing and
the image's literal dependency path. Preserve `AssetPinnedOracle`, per-call
working-directory handling, lazy-load protection, and positive-control policy.
Offline tests here use saved receipts or mocks, not fresh oracle calls.

For T4, include receptor/binary hashes and both diagnostic checkpoint directories
pinned by its contracts. A mutable upstream download URL is not an immutable
asset identity. Missing assets need exact identities and an access requirement,
not an automatic latest-version download. Do not expose credentials.

Remote artifact locators must record workspace, profile, volume, and namespace:
the campaign handoff reports distinct same-named PMO volumes under `nitya` and
`rahul`/KOSHA-LABS, and separate T4 main and revival volumes. Do not query them
under this local brief. If required receipts exist only remotely, list exactly
what needs authorized retrieval and finish the unaffected local integration.

Document historical scientific-run commands separately from safe offline
commands. Ordinary `verify`, `tables`, asset inspection, help, and tests must
never create cloud resources or score molecules. Any future launch or resume
requires its own explicit scope and budget; documentation does not authorize it.

## Acceptance and return to the integrating agent

Implement PMO first, then T4, in small local commits. For each task:

1. Provide an ordinary task README with environment, input access, offline
   verification/reduction commands, exact evidence sets, and runtime limitations.
2. Make saved-table commands run from a minimal source export with no historical
   worktree, credentials, cloud SDK login, network, or live oracle. Heavy runtime
   dependencies must not be needed merely to reduce saved scalar metrics.
3. Test wrong/missing hashes, malformed schemas, duplicate identities, mismatched
   arms/contracts, incomplete budgets, explicit exclusions, deterministic
   reductions, and output collisions. Use small behavior-level fixtures.
4. Stage complete outputs and publish atomically without overwriting old ones.
   Store machine-readable results plus readable tables, with input/code hashes,
   environment, configuration, cohort, and verification scope.
5. Check numerical agreement with the original reducer on the identical cohort.
   Keep a discrepancy ledger. Missing raw data must reduce the stated verification
   level, not disappear behind a successful summary-table check.
6. Run focused tests and touched-code lint while iterating. Run applicable local
   repository verification and the full suite once at the integration boundary.
   Report actual failures/skips; do not weaken gates or repeatedly rerun unchanged
   unrelated failures. The fragment boundary already recorded a Python 3.11
   collection error involving `StructuralDecisionIndex.__protocol_attrs__` and
   repository-wide lint debt. That is not a globally passing baseline.
7. Inspect final/staged diffs, original-file hashes, and generated artifacts.
   Commit only intentional task files; leave unrelated user work untouched.
   No push. Report commit IDs and integration dependencies before cross-branch work.

Return the exact commands, files, source identities, measured checks, remaining
asset/provenance gaps, and any proposed next step needing approval. Distinguish
table reproducibility, saved-trace replay, and new end-to-end campaign execution.
Do not describe the whole repository as pristine or fully reproducible until
its corresponding checks and asset-access requirements are actually satisfied.

## Preparation record

This handoff inspected the primary fragment integration, task documentation, Git
revisions/status, and existence of the PMO/T4 starting files above. It did not
reduce PMO/T4 results, certify their latest completion counts, access Modal, or
change any campaign file. All pre-existing untracked material was left in place.
