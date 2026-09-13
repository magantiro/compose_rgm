# Fast coordinated-program search

## Identity and current scope

The problem is useful complete-candidate yield under finite proposal and oracle
budgets. Output remains an exact executable program, attachments, completed
molecule, ancestry and work receipt. COMPOSE's platform identity, executor,
40-heavy-atom T4 support and frozen molecular reference are unchanged.

The user's immediate requested change is an explicit program-only optimization
proposal. It does not sample the unchanged reference process or claim Doob
exactness. Mutation, verified branch recombination and input-atom editing remain;
the expensive reference-continuation channel has zero allocation in this named
mode. The legacy mixed controller remains a runnable control. This is a proposal
restriction, not a reduction of the executor's declared chemistry support.

Initial scope: local implementation, focused checks and zero-oracle profiling.
Do not relaunch the interrupted campaign or spend its apparent remaining budget
on an unrecorded new recipe. Retain the unresolved PMO reservation as charged.

## Smallest decision

First compare program-only against the current mixed recipe on the same exact
BRAF/JAK2 warm archives and fixed randomizations, with identical work/candidate
ceilings, endpoint gates and hardware where the qualified runtime is available.
Record archive admission separately from proposal work, unique eligible endpoints,
attempt reasons, reference calls, CPU/wall seconds and input/source hashes. If the
reference runtime is unavailable locally, record that limitation and use historical
mixed timings only as unmatched context, never as a measured speedup.

Then compare exact-attempt reuse against the fast fixed proposal, not against a
different chemical menu. Cache exact source/program/binding/execution-limit
identities. Reuse deterministic executions or rejections only, never fabricate
task labels or blacklist a whole family. Unqueried eligible molecules remain
available for later query selection. Preserve resume identity and bounded memory.

Expected observable improvement: more new eligible candidates per proposal second
without reference computation blocking a batch. Positive yield supports continuing
with bounded untried neighborhoods. Negative or sparse yield directs work to the
recorded compatibility/duplicate causes. Throughput alone does not establish better
docking, PMO query efficiency or benchmark superiority.

## Acceptance and boundaries

- Explicit serialized program-only configuration, with old mixed defaults intact.
- Real archived constructions generate new eligible endpoints with no reference
  runtime; retain exact replay, size, ancestry and endpoint guards.
- Model fitting/prediction occurs only when the pool exceeds query slots. Direct
  allocation retains a pre-oracle lock and all proposal work.
- Repeated exact proposals do not redo deterministic executor work; bounded caches
  retain explicit cache-hit accounting and cannot silently drop unqueried outputs.
- PMO can make every charged initialization state an editing parent, while retaining
  original exact states and query accounting. Legacy initialization is reproducible.
- Dispatch failures are recorded per unit; ambiguous-query protection remains
  fail-closed without cancelling unrelated units.
- Focused tests, touched-file lint/format, artifact inspection and recorded limitations.
  Full-suite/release verification remains required before a milestone-complete claim.

No new predictor architecture, direct policy fitting, reference training or paid
benchmark result belongs to this first repair. Those follow measured throughput
and separately frozen experiment manifests.

## Local outcome, 2026-09-13 UTC

All four program-only batches filled twelve unique, eligible slots each. These
are candidates new to each declared warm archive, not 48 globally distinct
molecules or new docking observations. Exact source/program/attachment/actions/
prefix states and every attempt outcome agree before and after the execution
optimizations. No learned reference calls, task oracle calls or model fits.

| Exact warm context | Initial program-only seconds | With reuse and exact bond masks |
| --- | ---: | ---: |
| BRAF seed1, search seed 20260921 | 10.904 | 7.533 |
| BRAF seed1, search seed 20260922 | 22.191 | 15.803 |
| JAK2 seed1, search seed 20260921 | 14.011 | 13.941 |
| JAK2 seed1, search seed 20260922 | 21.598 | 17.938 |
| Sum, four batches | 68.705 | 55.215 |

This is a 19.6% reduction in summed proposal wall time on the same local arm64
CPU, not a speedup measured against the remote mixed campaign. Each configuration
has one timing per source/randomization, no timing confidence interval. Peak RSS
was approximately 460 MB. Archive admission is reported separately. The profile
uses RDKit 2024.03.5, numpy 1.26.4, float64 probabilities and integer executor
states, one worker. Material input hashes, source hashes, configurations, exact
candidate traces and dirty-worktree status are retained per profile.

The separate untried-conditional-mutation assay also filled all four pools:
245 versus 258 attempts and 51.073 versus 55.215 proposal seconds. But canonical
duplicate attempts increased from 58 to 62, and primitive executor work increased
in one JAK2 context. This is not evidence of improved docking or query efficiency.
Keep `mutation_sampling="random"` as the fast control; `"untried"` remains an
experimental comparison, not a promoted learned proposal.

### Implementation map

- `ProgramSearchConfig.program_only_recipe()`: explicit 7/9 mutation, 2/9
  recombination, zero broad-reference continuation. Retains verified decomposition,
  fresh-current-state semantics and the 0.25 input-atom-edit allocation. The old
  mixed factory/configuration remains. Enable the measured cache with
  `proposal_cache_entries=128`; it is otherwise off for comparison compatibility.
- `edit_program.py`: exact bond masks accelerate injective attachment matching,
  including nonbond constraints, without changing ordering, visit caps or support.
- `program_work_cache.py`: bounded FIFO reuse of binding censuses, current-state
  action enumerations and exact construction results/rejections. No oracle labels.
  Caches are process-local and cold after resume, so no stale implementation can
  import cached outcomes. Attempts/RNG/history remain durable. Fixed-work resume
  preserves choices; wall-limited stops can differ with a cold cache.
- `adaptive_program_optimizer.py`: opt-in bounded histories of previously drawn
  valid symbolic mutation choices, keyed to exact source/program/binding context.
  This is conditional sampling without replacement, not exhaustive complete-program
  enumeration or a learned policy. Different decisions can still produce the same
  endpoint. Symbolic compilation failures may still recur; no family is blacklisted.
- `parent_edit_search.py` / `program_campaign.py`: model fitting is lazy; a pool
  no larger than the query allowance is directly locked and scored. No fitting or
  prediction is performed solely to return the whole pool. The selection-policy
  identity is explicit, so old manifests cannot silently resume under this rule.
- `program_campaign.py`: optional `initialization_mode="all_scored_pool"` makes
  every charged exact initial state available uniformly during bootstrap and later
  initial-parent exploration (default fraction 0.2). It does not invent empty edit
  programs or pretend these source choices are score-adaptive. The legacy bootstrap
  remains available. A three-round synthetic-label test exercises this full path;
  it is not a completed real PMO comparison.
- `parent_edit_cycles.py` / Modal adapter: frozen-task dispatch replaces the stale
  probe whitelist. Restart guards return a per-unit failure with charged/unresolved
  receipts intact. Exception-valued ordered map results retain unaffected workers;
  uncertain accounting is explicitly marked incomplete. Not redeployed.

### Artifacts and decision

`diagnostics/fast_program_search/throughput_comparison.json` verifies exact output
equivalence and records the timings above. `fixed_1`, `reuse_1`, `bindings_1` and
`untried_1` retain every completed batch and snapshot; `instrumented_1` is a
separate cProfile run, not included in timing comparisons. The neighborhood assay
is `neighborhood_comparison.json`. These are local development artifacts with
source hashes, not clean-commit benchmark releases.

Decision: retain the fast program-only configuration and execution optimizations.
Do not scale the previous mixed campaign, promote untried mutation on speed alone,
or train another future-value head. Next: freeze one short direct-scoring PMO
run with all charged starts available, before any larger grid. A changed paid
recipe still needs its own bounded launch manifest/authority and clean source.
The previous 647 PMO reservations, including the unresolved one, are unchanged.

Runnable local profile (no task oracle):

```sh
PYTHONPATH=src:.:tests OMP_NUM_THREADS=1 python tools/fast_program_search.py \
  --cache-entries 128 --output diagnostics/fast_program_search/new_profile
```

Use the pinned chemistry environment noted above. Add `--mutation-sampling untried`
only for that explicit proposal ablation. The command refuses existing output paths.
Both profiles and comparisons bind inputs and source identities. No new commits,
push, remote deployment or full milestone-completion claim accompanies this local
checkpoint; old user changes and all paid results are preserved.

Verification at this checkpoint: 58 focused tests passed in 11.04 seconds across
the optimizer, cache, attachment matching, task/selector, program transfer,
decomposition/capability, campaign and worker-isolation dependencies. Touched-file
Ruff checks and formatting passed; `git diff --check` passed. The artifact
comparison independently checked all four candidate lists, exact primitive
traces, assignments and attempt outcomes. The full repository suite was not run;
this is a local development checkpoint, not a milestone/release sign-off.

## Approved next execution: one fast PMO run

The user approved the proposed short run with "great lets od it". The new frozen
`configs/fast_pmo_run.json` authorizes only albuterol similarity, seed 20260921,
128 queries including sixteen already locked task-independent starts, seven
rounds of at most sixteen new candidates. One CPU container, 4 GiB, 600-second
hard timeout, 420-second campaign limit and $1 reserved including build/storage
headroom. No automatic retry, other task, docking, training or grid follows.

Use the measured program-only random-mutation recipe with cache128, uniform
all-scored initialization-parent access, no reference loading, and no model fit.
The same 146 shared T4-derived programs remain development information. No old
PMO scores enter this run; prior task inspection is disclosed. This is not a
fresh held-out evaluation. Per-query best and top-ten curves include every
initial query. Budget-128 AUC is only a short-run diagnostic, not a 10k PMO score.

Expected observable: completed score/propose cycles with useful candidate yield
and measurable proposal time. A positive outcome supports a matched fast-control
comparison; empty pools direct attention to proposal support; infrastructure or
oracle failure triggers receipt inspection, not automatic retry. Preserve all
shortfalls and the previously unresolved charged PMO reservation.

`tools/fast_pmo_run.py prepare` freezes existing input hashes without generating
molecules. After focused verification and a clean committed launch worktree,
deploy `modal_apps.fast_pmo_run_app`, then use `tools/fast_pmo_run.py launch
--receipt <durable-path>`. `retrieve` publishes the self-hashed final record;
the volume additionally retains candidate locks, exact traces, query receipts
and complete-round archives. A pre-spawn intent prevents ambiguous local retries.

The first invocation (`fc-01M2C7JNEYDFMCJWJPBV79HGDZ`, source `0763d5d52210`)
failed before worker entry: the auto-mounted Modal package resolved host ROOT
as `/root` and read `/root/configs/fast_pmo_run.json` during import. Inputs are
under `/root/compose`. Cancelled the pending call; its artifact namespace and
query reservations are absent. `diagnostics/fast_pmo_run/startup_review.json`
retains the zero-query audit. Image-input discovery is now host-only, with a
focused container-import regression check. This is a manual startup repair,
not an oracle retry or changed proposal, input lock, budget or scoring recipe.
