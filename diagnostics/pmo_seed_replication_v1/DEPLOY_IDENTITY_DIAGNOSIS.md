# PMO fibercontrol deploy: the identity mismatch, diagnosed and fixed

MEASURED 2026-09-25. Zero charged oracle calls were spent on this diagnosis.

## The error

Every dead container raised, inside `pmo_population_v1.load_contract`:

    ValueError: input identity mismatch:
      src/compose_v4/control/dynamic_program_synthesis_v21.py:
      expected 3e591297..., got fc769f2c...

## Where each hash lives

| sha256 of `dynamic_program_synthesis_v21.py` | pinned by / found in | committed? |
|---|---|---|
| `3e591297` image EXPECTED | `configs/pmo_population_controller_v2_proposal_repair{,_contract}.json` (UNTRACKED) | **never committed on any branch** |
| `fc769f2c` image FOUND    | `configs/pmo_population_controller_v3_seed_weighting.json` | **never committed on any branch** |
| `9dec8f5a` | `configs/pmo_population_controller_v4_transplant_lane.json`; t4 worktree HEAD | committed |
| `d5f4d432` | `configs/pmo_dynamic_v21_development_v1.json`; THIS tree | committed |

Method: enumerate every committed blob of the file over `git log --all` and compare.
Both hashes in the error message correspond to **no commit on any branch**.

## Root cause

The image was built from an uncommitted working tree that no longer exists, and it
was *internally* inconsistent at build time -- not drifted afterwards:

1. `src/` was baked at working-tree state `fc769f2c`.
2. `PMO_FIBERCONTROL_CONTRACT` selected the v2 contract, itself sealed against an
   EARLIER working-tree state `3e591297`, and never committed.
3. The selection mechanism itself is uncommitted: `CONTRACT = os.environ.get(
   "PMO_POPULATION_CONTRACT") or ...` exists only in the **modified working copy** of
   `pmo_population_v1.py` in the `t4-objective-dynamic-reset-20260916` worktree.

So three states, two of which are unrecoverable. `add_local_file(..., copy=True)`
froze the contract and the source together at build time, and the fail-closed check
then refused every container before it charged anything -- which is the guard working.

## The fix: no re-seal was required

This tree (`c7fdddca`, clean, committed) is already self-consistent:

* Its `pmo_population_v1.py` hardcodes `CONTRACT = "configs/pmo_population_controller_v1.json"`,
  so the env override is INERT here and the v2/v3/v4 contracts are not present to be
  mis-selected.
* `load_contract()` -- the exact call that raised -- was executed locally against this
  tree under the PMO kernel (python 3.11.13 / rdkit 2023.9.6) and **passed**, verifying
  all 10 pinned files.
* `scored_launch_authorized: False` and `modal_launch_authorized: False`, which
  `load_contract` itself REQUIRES (`cannot authorize scoring`). Nothing authorizing was
  touched; no contract file was edited at all.
* `initialization.lock_sha256 = c8032311daa067628048679465fb4397d2a26bf0593143e3ed0a4cbec7195478`,
  `controller.seed = 20260920` -- identical to the contract the existing seeds ran under.

Comparability: `git diff 3e6baf94 c7fdddca` touches only the Modal app, the canary
driver and the launcher. **No pinned controller module differs**, so the controller
source is identical to the existing seeds'. The canary diff is scoped by
`if not requires_positive_control(name): return oracle`, so the 20 pure-RDKit tasks are
byte-identical and only drd2/gsk3b/jnk3 gain the asset-pinned oracle + positive control.

## Deployed

App and volume `compose-pmo-fibercontrol-replication` (profile `nitya`), chosen because
it was empty and idle: redeploying `-targets` would have rebaked an image its 2 live
tasks could pick up on a preemption retry.

## Correction to the reported failure count

`returncode: 1` is NOT the startup-death signal: it appears on campaigns that ran to
completion and then hit the fail-closed `pending round needs explicit receipt-based
recovery` guard. Counting oracle ledger entries instead:

* 5 campaigns died on the identity mismatch, each charging **zero** (no `oracle/` dir,
  0-byte stdout): the replicate-70 launches at 2026-09-24T04:27:55Z.
* The rest have ledgers of 856-5451 entries.
