# V2 corpus: physical state, 2026-08-09

Written after a near-loss: the 70,301-entry training corpus existed in exactly
one place, a scratchpad directory the project treats as disposable, and was
briefly believed not to exist at all. Everything below is **measured from the
artifacts**, not inferred from a plan. Where a number comes from a plan, it
says so.

## 1. What physically exists

| corpus | entries | distinct pairs | families | role | verified |
|---|---|---|---|---|---|
| `train_65k` | **70,301** | 67,614 | **8** | train | 294 receipts, 34 tasks |
| `v2_corpus` | **36,864** | 36,864 | 3 | train | 1,514 receipts, 19 tasks |
| `valid_set` | **14,140** | 14,089 | 8 | validation | 118 receipts, 9 tasks |

**Combined train: 107,165 rows / 104,478 distinct canonical pairs.**

Disjointness, measured on blake2b digests of `(source_state_sha256,
canonical_successor_key)`:

- `train_65k` ∩ `v2_corpus` = **0**
- `valid_set` ∩ (train union) = **0**
- 2,687 duplicate pairs exist *inside* `train_65k` and nowhere else

Combined train family distribution:

| family | rows | share |
|---|---|---|
| bond_reroute | 24,295 | 22.7% |
| atom_delete | 21,288 | 19.9% |
| atom_restate | 21,283 | 19.9% |
| atom_insert | 21,114 | 19.7% |
| cycle_insert | 9,827 | 9.2% |
| cycle_attach | 6,832 | 6.4% |
| bond_reorder | 1,903 | 1.8% |
| ring_system_restate | 623 | 0.6% |

## 2. Where it lives, and how many copies

| location | contents | copies |
|---|---|---|
| `~/compose_trainset_backup/train_65k` | 294 receipts, 6.6 GB | 2 (+ scratchpad) |
| `~/compose_trainset_backup/valid_set` | 118 receipts, 1.3 GB | 2 (+ scratchpad) |
| `~/compose_trainset_backup/v2_corpus` | 66 receipts, 1.4 GB — **local half only** | 2 (+ scratchpad) |
| Modal volume `process_v2_v2_corpus` | 1,448 receipts, ~2.1 GB — **Modal half** | **1 — SINGLE COPY** |

**Open risk:** the 21,778 entries compiled on Modal exist only on the Modal
volume. They are reproducible (~$4 and 25 min) but not backed up.

The source chunk cache — 645,019 records, already role-partitioned
(train 545,392 / controller_validation 38,862 / final_test 33,683 /
validation 27,082) across five lanes — also lives under the scratchpad and on
both Modal volumes.

## 3. Two ways "what exists" was got wrong today

Both were the same mistake in different clothes: **a plan was read as a result.**

1. `scripts/editing_v2_corpus_census.py` computes
   `compiled = {c["task_identity_sha256"] for c in plan["chunks"]}` — it labels
   a task "compiled" because a plan *names* it. That produced the
   "145,180 pre-existing rows" figure. The correct reading is that the 34-chunk
   plan was executed **locally** and produced `train_65k` (70,301 of 145,756
   planned); the Modal fan-out of the same plan published 8 entries.
2. Searching only `v2_corpus` for compiled output, concluding no backbone
   existed, and nearly ordering a $24–34 recompile of four families that were
   already on disk.

Fix for (1): census published receipts, never a plan. Fix for (2): inventory
by scanning for `ENTRIES.json` trees, which is how `train_65k` was found.

## 4. Against the frozen recipe

The frozen V2 recipe (`diagnostics/editing_v2_postdedup_census.json`,
107,872 rows, 16.4% synthetic) is **essentially realized at 107,165 rows**, with
one real gap:

- **`ring_system_restate` is at 623, against a recipe floor of ~5,307.** The
  synthetic increment that was to supply the other ~4,684 rows was never
  compiled — today's work compiled the two *real* lanes instead.
- The realized synthetic share is therefore below the intended 16.4% and must
  be measured, not copied from the recipe.

Per the standing decision: the compiled corpus is the **available library**;
the frozen manifest is the **sampling law**. Report both the available
composition and the realized training coefficients; they are different numbers.

## 5. Verified properties of the new expansion

- Modal-compiled and locally-compiled rows are **byte-identical** for the same
  range, including `p50_entry_sha256`
- every entry `partition_role=train`; zero slices missing a receipt or entries
- `model_scores_or_probabilities_stored: false` — model weights do not enter
  compiled rows
- catalog fingerprint `639ff6078c32d43c` and process identity
  `0c938177a34819e6…` held throughout

## 6. Open items

1. Back up the Modal half of `v2_corpus` (only copy)
2. Decide the `ring_system_restate` top-up: ~4,684 synthetic rows
3. Recompute realized coefficients on the combined library
4. Confirm split source-disjointness is structural (roles are assigned per
   task in the source cache, so it likely is)
5. Seal the final test role (33,683 source records exist, uncompiled)
6. `scripts/editing_v2_corpus_census.py` still reports plan-as-compiled
