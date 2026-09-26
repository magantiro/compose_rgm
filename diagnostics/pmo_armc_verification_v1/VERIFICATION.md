# Arm C — consolidated verification

**Commit `e11cd89d64c03a2183b5ac7c6443c2b4f505290c`**, pushed to
`origin/pmo-chain-ablation-20260925`. Every number below is from THAT snapshot, measured in a
detached sparse checkout of it (`git status` clean) under **python 3.11.13 / rdkit 2023.9.6**,
the PMO production kernel. Zero oracle calls. Counts from other snapshots are not mixed in.

## 1. Ordinary tests

`tests/test_pmo_binding_intervention.py` — **15 passed**, reproduced three times on an
unloaded machine. `ruff` clean on `src/`.

## 2. Mutation battery — 12 deliberate defects, all caught; control survives

Run from the frozen checkout, aborting if the unmodified baseline fails. Baseline PASS.

| mutation | verdict |
|---|---|
| v21 archives the PRESCRIBED program | KILLED |
| v21 ignores the arm-C flag | KILLED |
| occurrence dropped from the rng key | KILLED |
| reschedule self-consistency check removed | KILLED |
| replay self-consistency check removed | KILLED |
| delete bookkeeping reverts to the prescribed mark | KILLED |
| birth slot becomes rebindable | KILLED |
| handle counter reuses live length | KILLED |
| identity consumes randomness | KILLED |
| input operands also rebound | KILLED |
| rng seeded by python `hash()` | KILLED |
| arm B / arm C exclusion no longer raises | KILLED |
| **COSMETIC CONTROL (comment only, must SURVIVE)** | **SURVIVES** |

### A contaminated verdict, corrected
In the batch run the cosmetic control read KILLED and several kills reported inflated failure
counts, because the battery ran while the same laptop was deploying two Modal apps, launching
18 campaigns and issuing T4 queries. Re-run individually on a quiet machine:

* cosmetic control (both the battery's exact string and another comment) → **15 passed, SURVIVES**
* `reschedule` mutation → **1 failed, 14 passed**, failing exactly
  `test_a_modified_program_that_reschedules_differently_is_REFUSED`
* `replay` mutation → **1 failed, 14 passed**, failing exactly its own test

So the kill verdicts are right and specific; the batch failure COUNTS and the cosmetic verdict
were load artifacts. **Recorded caveat: those two constructed tests are load-sensitive.** Cause
not established — it will be captured the next time it occurs rather than guessed at.

An earlier battery at `a738a0eb` had those two guards SURVIVING (a genuine coverage gap, since
0 of 640 real attempts reach either condition) and one battery before it ran on a RED baseline;
**neither is counted here.**

## 3. The five required properties, checked against the commit's blobs

| property | |
|---|---|
| proposal-occurrence term in the RNG key | present |
| v21 stores the MODIFIED program, not the prescribed one | present |
| modified program replays to its recorded endpoint | present |
| intended operation ORDER preserved (reschedule check) | present |
| size profile / dependency metadata recomputed from the modified program | present |
| blake2b, never Python `hash()` | present |
| arm B / arm C mutually exclusive, fail-closed | present |
| **call sites of the rebound executor in the entire tree** | **exactly 1**, in v21's proposal path |

The last line is the strongest available form of "only the designated proposal-stage execution
applies the intervention": internal construction inside `_channel_proposal` and the archive's
admission replay cannot reach it.

## 4. Development panel — 80 programs, 640 attempts (8 occurrences each)

| | |
|---|---|
| completion | **639 / 640 = 99.84%** |
| applicable (program refers to an atom it created) | 367 / 639 = **57.4%** |
| of those, ≥1 alternative admissible binding | 335 |
| **canonical endpoint changed vs Arm A** | **301 / 639 = 47.1%** |
| refusals | `rebinding` **1**; `rebinding_reschedule` **0**; `rebinding_replay` **0** |
| cost | 0.126 s per attempt |

**Applicability and endpoint change are different numbers** (57.4% vs 47.1%): a substituted
binding may legally rebuild the same canonical molecule. Roughly 43% of proposals carry no
created-atom operand and are UNCHANGED by construction — correct ablation behaviour, not a
shortfall, and no proposal is altered to manufacture contrast.

## 5. Archive integration — the check that made this an experiment, not a throttle

`add_measured_program` replays the STORED program through `compile_program_graph` +
`scheduled_program` and demands exact endpoint AND state-sequence identity.

    arm C admitted to archive                                   79 / 79
    arm A admitted (control)                                     79 / 79
    arm C endpoints differing from arm A                             38
    of those, storing the PRESCRIBED program would be REJECTED   38 / 38

Before the fix the proposal path archived the prescribed program beside Arm C's endpoint, so
**every endpoint-changing proposal would have been rejected** and only proposals where
rebinding changed nothing could have entered the archive.

Production path (`propose_batch`, arm C on, zero oracle calls): 8/8 candidates carry
`ablation_stage`, and 8/8 stored programs replay to their own candidate endpoint and states.

## 6. Defects found before any scored call

1. **created-handle index reuse** — the counter was derived from the live list, so after an
   `atom_delete` the next insert collided with a surviving handle. Caught by the invariant that
   the prescribed binding must stay admissible before any divergence (1 violation → 0).
2. **`atom_delete` bookkeeping followed the prescribed mark**, not the rebound one, retiring the
   wrong atom. Caught by `EditProgram.validate`. Fixing it took completion 97.5% → ~100% and
   removed every refusal; the "2 rebinding failures" reported earlier were this bug.
3. **the archive stored the prescribed program** — §5, fatal, 38/38.
4. **the occurrence term was silently deleted from the file** by an earlier mutation battery
   restoring a pre-edit snapshot over the live working tree. Caught by the occurrence test.
5. **launch receipts collided on a second-resolution stamp**, so replicate 80 overwrote
   replicate 60's call ids. Recovered from stdout, and the path is now arm/replicate-scoped and
   refuses to overwrite.

## 7. Disclosed limitations

* The proposal-synthesis path is **not** in the contract's `implementation_sha256` (8 pinned
  files, none of them Arm C's). Arm identity rests on the launch receipt's `git_commit` plus the
  deployed image, not a fail-closed hash chain. Pre-existing and identical for Arm B, so it does
  not bias the comparison.
* The two constructed self-consistency tests are load-sensitive (§2).
* `rebinding_reschedule` and `rebinding_replay` never fire on real data (0/640); they are
  defence-in-depth, covered only by constructed tests.

## 8. What a positive A-versus-C result would license

> Preserving the operation sequence, payloads and source-atom bindings, replacing the prescribed
> connections to newly created atoms with alternative legal bindings changes optimization
> performance.

It does **not** isolate the dependency scheduler, and Arm C necessarily changes molecular
connectivity. **A→C and C→B are not an additive decomposition of A→B**, and the score gap will
not be divided by the affected fraction.
