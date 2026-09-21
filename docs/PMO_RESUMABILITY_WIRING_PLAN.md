# PMO resumability wiring plan

Status: **NOT APPLIED.** Hold until the live `B_memory_1k` celecoxib run closes.
The layer it wires is committed and tested on `pmo-resumability-20260921`
(`durable_resume.py`, 39 tests, mutation audit 10/10 caught).

Everything below is MEASURED against `pmo-ab-integration-20260921` — the branch the
live arms run from — not against this branch's base, which differs (its controller
has no `online_memory` and its app has no `resume_in_place`). Diff the seams against
that branch or the anchors will not match.

---

## 0. The design decision that sets the blast radius

`program_campaign.py` is pinned by **9** contracts. Two of them —
`pmo_dynamic_v21_development_v1.json` and `qed_griddd_controller_comparison_v1.json`
— pin it and **nothing else** in the seam set (no app, no `pmo_population_v1.py`, no
controller). So:

| approach | contracts to re-pin | drags in |
|---|---|---|
| edit `program_campaign.py` | **9** | QED comparison + dynamic-v21, both unrelated |
| reconcile in `execute_task` instead | **7** | PMO-population family only |

**Recommendation: keep `program_campaign.py` out of the diff.** It is achievable
because `ProgramQueryLedger` already has the property that makes a replayed round
free: `query()` returns the cached row for an endpoint it has already scored
(`if endpoint in self.cache: return self.cache[endpoint]`), and
`run_program_campaign` restores the controller from the previous round's
`complete.json` snapshot and re-derives proposals from round-addressed RNG. A
replayed round therefore re-proposes the same molecules and re-charges none of them.

Only two things block that replay today, and **both can be resolved before
`run_program_campaign` is called**:

1. `ProgramQueryLedger.__init__` raises `ambiguous charged oracle attempt` when a
   `query_*/started.json` has no `result.json` (death during the oracle call).
2. `run_program_campaign` raises `pending round needs explicit receipt-based
   recovery` for a `round_*/pending.json` with no sibling `complete.json`.

Both are *artifacts on disk*. `execute_task` can repair (1) and archive (2) before
handing over, leaving `program_campaign.py` byte-identical.

---

## Seam 1 — `modal_apps/pmo_population_v1_app.py`: lease replaces started-forever

**Anchors (live branch):** `elif spec.get("resume_in_place"):` L235,
`terminal_call = spec.get("resume_after_terminal_call_id")` L252,
`elif result_path.exists() or started.exists():` L277, `def progress(row)` L327.

Replace the human-proof resume branch and the permanent refusal with one
classification.

```python
from compose_v4.control.durable_resume import (
    COMPLETE, LeaseHeld, classify_run_start,
)

LEASE_SECONDS = 600.0

# ... after the extend_to_budget branch, replacing BOTH the resume_in_place branch
# and `elif result_path.exists() or started.exists(): raise`:
verdict, lease = classify_run_start(
    result_path=result_path,
    lease_path=folder / "lease.json",
    run_id=run_id,
    worker_id=os.environ.get("MODAL_TASK_ID") or f"pid-{os.getpid()}",
    lease_seconds=LEASE_SECONDS,
)
if verdict == COMPLETE:
    raise RuntimeError(
        "task is COMPLETE; raising its budget is `extend_to_budget`, not a resume"
    )
```

`classify_run_start` raises `LeaseHeld` when a live worker holds the run — the
refusal we want — and returns `RESUME_IN_PLACE` when a dead worker's lease has
lapsed, which is the case that previously needed a human to name
`resume_after_terminal_call_id`.

`started.json` keeps being written (artifact continuity) but **stops gating**.

**Heartbeat.** Renew in `progress` (fires per round) *and* in the ledger's `flush`
hook (fires per oracle call), so takeover latency is bounded by `LEASE_SECONDS`
rather than by round duration, which the realizer lane can stretch to minutes:

```python
    def progress(row: dict) -> None:
        lease.renew()
        _write_json(folder / "progress.json", {...})   # unchanged
        volume.commit()
```

`LEASE_SECONDS = 600` is a starting value, not a measurement: it must exceed the
longest gap between two renewals. Confirm against a real round's wall clock before
the first scored launch; if a single proposal phase can exceed it, renew inside the
proposal loop too rather than lengthening the lease.

**Only after this lands may `modal_retries` rise above 0.** Until then a retry
re-enters the old guard.

---

## Seam 2 — `src/compose_v4/experiments/pmo_population_v1.py`: reconcile before campaign

**Anchor:** `ledger = ProgramQueryLedger(folder / "oracle", task, evaluate, budget=...)` L160.

Insert a reconciliation *before* that line. It has two jobs and no others.

```python
from compose_v4.control.durable_resume import reconcile_legacy_oracle_folder

repair = reconcile_legacy_oracle_folder(
    folder / "oracle",
    folder / "campaign",
    evaluate=evaluate,
)
```

`reconcile_legacy_oracle_folder` is **the one piece still to be written** and is
deliberately small (target: ~60 lines plus tests, in `durable_resume.py`, which is
unpinned):

1. For each `query_*/started.json` with no `result.json`: re-evaluate that endpoint
   once and write the missing `result.json` in the schema
   `ProgramQueryLedger.__init__` expects, with `repaired: true`. The call stays
   **charged** (it was reserved before the oracle ran, so it cannot be proven not to
   have reached it) and the re-evaluation is one extra **physical** call, reported
   separately. Bounded by one per preemption.
2. For each `round_*/pending.json` with no `complete.json`: copy it to
   `pending_attempt_<n>.json` and unlink the original, so the campaign replays that
   round instead of refusing. Never delete — preserve then remove.

Return a dict recording repaired call ids and archived rounds; write it to
`folder / "resume_reconciliation.json"` and include it in the task result so a
resumed run is legible from its artifacts alone.

**This is the only new code the wiring needs.** It must ship with its own kill-test
scenario before use.

---

## Seam 3 — `src/compose_v4/control/pmo_population_controller.py`: close the last silent fallback

**Anchors:** L1088–L1089.

```python
result.credit = PopulationCredit.restore(state["credit"]) if state.get("credit") else PopulationCredit()
result._pool_continuity = BootstrapPoolContinuity.restore(state.get("pool_continuity"))
```

A snapshot missing either key restores an **empty** object and the run continues
producing plausible numbers. This is the one remaining silent-cold-start path:
`online_memory` immediately above is already correctly guarded (it raises when a
snapshot carries memory into a memory-less arm, and when a warm memory meets a
snapshot that carries none).

**Do not simply delete the fallback** — it exists so pre-persistence snapshots still
load. Mirror the memory guard's shape instead, which distinguishes cold from warm:

```python
credit_payload = state.get("credit")
if credit_payload is not None:
    result.credit = PopulationCredit.restore(credit_payload)
elif result.credit.is_warm():          # add: any recorded trial
    raise ValueError(
        "controller has warm credit but the snapshot carries none; resuming "
        "would silently restart allocation from nothing"
    )
# same shape for _pool_continuity
```

A freshly constructed controller is cold, so a legacy snapshot still loads; a warm
one refuses. `is_warm()` does not exist yet and is a small addition to
`PopulationCredit` / `BootstrapPoolContinuity` (both already pinned, so no new
re-pin surface). For credit it is exactly

```python
def is_warm(self) -> bool:
    return any(cell.trials for cell in self.cells.values())
```

— `bool(self.cells)` would be wrong, because a cell can exist with zero trials.

---

## 4. Component profile — no mapping needed

`PMO_REQUIRED_COMPONENTS` already names the keys
`PmoPopulationController.snapshot` really emits, with dotted paths for the nested
block (`pmo_population.credit`, `pmo_population.online_memory`, …) and
`PMO_NULLABLE_COMPONENTS = {pending, pmo_population.online_memory}`. Arm B should be
constructed with `non_null_components = PMO_REQUIRED_COMPONENTS - {"pending"}` so a
resumed arm B can never continue as arm A.

---

## 5. Re-pin list — exact, and in this order

Adding `src/compose_v4/control/durable_resume.py` to every `implementation_sha256`
below, plus the three edited files' new hashes.

**Contracts (7, assuming seam 2 stays out of `program_campaign.py`):**

| contract | current files | after |
|---|---|---|
| `pmo_ab_b_memory_1k_contract_v1.json` **(live)** | 11 | 12 |
| `pmo_ab_a_baseline_1k_contract_v1.json` **(live)** | 11 | 12 |
| `pmo_ab_b_memory_contract_v1.json` | 11 | 12 |
| `pmo_ab_a_baseline_contract_v1.json` | 11 | 12 |
| `pmo_population_controller_v1_scored_contract.json` | 8 | 9 |
| `pmo_population_controller_v1_scored_contract_corrected.json` | 8 | 9 |
| `pmo_population_controller_v1.json` | 11 | 12 |

**Not touched** if the recommendation is followed:
`pmo_dynamic_v21_development_v1.json`, `qed_griddd_controller_comparison_v1.json`.

**Capsule:** `diagnostics/pmo_population_controller_v1/source_capsule_manifest_v2.json`
— `source_files` 15 → 16.

**Order.** Edit sources → re-pin each contract's `implementation_sha256` → recompute
each payload hash → update the capsule's `source_files` **and** its
`contract_payload_sha256` → update the launcher's `PAYLOAD` constant
(`tools/launch_pmo_population_v1_corrected.py:49`) → obtain a **fresh owner
authorization** for the new payload → `modal deploy`.

**Three traps, all previously paid for here:**

- `tools/reseal_pmo_population_contracts.py` re-hashes keys **already present**. A
  NEW key (`durable_resume.py`) must be added to the file list by hand first, or it
  is silently skipped.
- `support_observed` must be **re-measured**, never re-pinned — a config that
  validates while binding a stale measurement is the recurring failure here.
- Re-pointing the launcher's `PAYLOAD` before the owner authorizes the new payload
  **manufactures consent**. The refusal is correct until the authorization names the
  new hash. Archive the superseded receipt under the payload hash **the receipt
  carries**, not the contract's current one.

---

## 6. Verification gate — before any scored launch

1. `pytest tests/test_pmo_resumability.py` green (39) under the pinned kernel.
2. `python tests/resumability_mutation_audit.py` → 10 caught / 0 survived /
   0 not-applied / control green.
3. **A new kill-test scenario driving the real `execute_task`** with a synthetic
   scorer, killed mid-round, proving: charged count unchanged, repaired calls
   reported, archived pending round replayed, final archive identical to an
   uninterrupted arm.
4. A two-container race against one run directory: the second must refuse with
   `LeaseHeld`, then succeed after the first's lease lapses.
5. `modal_retries` raised **only after** 1–4 pass.

---

## 7. Flagged, not fixed

- **`pmo_online_memory.py` is in the live arms' `implementation_sha256` (11-file
  contracts) but only in the capsule for the older 8-file `_corrected` contract.**
  It is on the scoring path — it changes which molecules get proposed — so the
  asymmetry is worth an owner decision, not a silent fix.
- **`LEASE_SECONDS = 600` is a proposal, not a measurement.** Time a real round
  first.
- **Lease liveness assumes one worker per run directory.** Two Modal containers
  given the same `run_id` and different folders are outside what the fence covers.
