# Broad-organic B-edit mining — validation report

**Status: `GO_FOR_FULL_500K_MINING` (broad scope).** The broad-organic validation shard passed on Modal
(commit a1618c8): 500k scan → 400k eligible train in 48s, 841 MMP pairs → 1,387 verified records in one 20k
shard, ~30 min projected full-run cost. STOP here for explicit authorization before the full 500k mining run.

This report covers the corpus-scope decision, the shared broad-organic loader, the coverage/supervision/
chemistry evidence built locally, the shardable global-grouping mining infrastructure, and the validation
shard. It is the STOP-for-authorization deliverable before the full 500k mining run.

## 1. Decision: broad-organic charge-preserving scope (LOCKED)

The B-edit training/mining corpus scope is **broad `ORGANIC_VOCABULARY` (C, N, O, F, S, P, Cl, Br, I, B) +
retained charged molecular states + charge-PRESERVING editing** under the current operator contract. It
replaces the CNOF-neutral filter (`load_cnof_corpus_split`) in every production edit path.

**Why (the fork we surfaced):** the retired filter keeps only C/N/O/F **and** neutral molecules. On the
canonical 500k GuacaMol corpus that is ~48% (measured 237,945 eligible on Modal; ~240k projected from the
local census), dropping **all S (≈32%), Cl (≈17%), Br, P, I** and the ≈6% charged. The master-plan §0b
locks "GuacaMol (broad)"; `ORGANIC_VOCABULARY` already carries S/P/Cl/Br/I heads. Training CNOF-only would
leave those heads cold and make most benchmark leads unrepresentable (see §4). Broad scope retains ~98%.

Charge is **preserved, not optimized**: no operator may silently neutralize/introduce/shift a formal
charge; there are no charge-changing operators. This is a *protected-charged-context* scope, not a
charge-design claim.

## 2. Shared, versioned corpus scope (`src/compose_v4/data/organic_corpus.py`)

One immutable `CorpusScope` object (`BROAD_ORGANIC_V1`, `scope_hash = e59fb09801459470`) is the single source
of truth, used by mining, MMP construction, corruption, training, validation, and evaluation. Membership:
every atom's element is in the ACTIVE registry vocabulary (never a restated list) AND its exact
`(element, valence, formal_charge)` is a representable class (`AtomVocabulary.class_index` is not `None`) —
the same derivation the model's teacher path uses, which is what admits representable charged states while
excluding the rest. `scan_corpus` does one parallelizable pass yielding the deduped accepted pool + a rich
census: finely counted rejection reasons, per-element and non-CNOF-combination tallies, and the three
DISTINCT charge categories (neutral / zwitterion-net-zero / nonzero-net-charge). Startup prints + hashes the
scope; the hash is stored in the mining summary, checkpoint metadata, and (pending) sampler metadata so a
cross-scope load can fail loudly. Tests: `tests/test_organic_corpus.py` (7).

## 3. Corpus census (500k, measured on Modal)

The broad scan of the full 500k corpus (12 workers) completed in **48.3s** and retained **490,466 / 500,000
= 98.1%** eligible broad-organic molecules (vs the CNOF-neutral filter's ~240k). Rejections: too-big 8,185 /
unsupported-element 1,112 / unparseable 201 (9,498 total). This is a **2.05×** larger training corpus than
CNOF-neutral.

- **Elements (molecules containing):** C 490,445 · N 457,350 · O 456,999 · **S 160,260 (33%)** · F 91,704 ·
  **Cl 86,033 (18%)** · Br 20,864 · P 6,385 · I 2,659 · B 685 — the S/Cl/Br/P/I mass CNOF-neutral discards.
- **Charge categories (distinct):** neutral 461,296 · **zwitterion-net-zero 23,870** · nonzero-net-charge
  5,300 — kept only by the charge-preserving broad scope.

The full census is persisted to the shard summary on the artifact volume
(`/artifacts/edit_mining_validation_broad/mining_summary_shard0000.json`). Local 5k proxy (identical scope +
code) matched at 98.0% retained.

## 4. Benchmark-lead coverage (Jin ICLR19 QED test, 800 leads)

`scripts/benchmark_lead_scope_coverage.py` → `diagnostics/composition/benchmark_lead_scope_coverage.json`:

| Scope | Leads accepted | Fraction |
|---|---:|---:|
| **broad-organic v1 (production)** | **800 / 800** | **100.0%** |
| broad-organic neutral-only (ablation) | 557 / 800 | 69.6% |
| CNOF-neutral (retired) | 264 / 800 | 33.0% |

The broad scope **recovers 536 leads (67%)** that CNOF-neutral would exclude; **233 leads (29%) carry a net
formal charge** (kept only by the broad scope). Zero leads are excluded under the production scope. A
CNOF-only B-edit could not represent two-thirds of the leads it is meant to optimize.

## 5. Cold-vocabulary audit (`scripts/cold_vocab_audit.py`)

The base is 4-class CNOF; B-edit widens the class heads to the 15 `ORGANIC` classes, so the 11 non-CNOF
slots (S v2/4/6, P v3/5, Cl, Br, I v1/3/5, B) are `NEWLY_INITIALIZED` — cold until the broad corpus
supervises them. A molecule merely *containing* sulfur does not warm the sulfur output slots; the audit
measures actual positive edit targets (atom_insert/atom_restate teacher marks producing each class) from
real broad-corpus corruption. Result: the 4 CNOF classes are `INHERITED_TRAINED` and **all 11 non-CNOF
classes are `BEDIT_SUPERVISED` with positive targets** (chiefly via `atom_restate` / heteroatom scan) — the
cold/absent set is **empty**. → `diagnostics/composition/cold_vocab_audit.json`.

## 6. Chemistry-specific legality + charge preservation (`tests/test_broad_vocab_chemistry.py`, 5)

Thioether / sulfoxide / sulfone / sulfonamide, phosphorus, aryl/alkyl chloride, bromide, iodide, and charged
amine / carboxylate / zwitterion all: encode/decode, are in scope, and yield **valid executor successors**
under the sampler. A **charge-preserving rollout policy** (reject any mark that shifts net charge) keeps
charged leads valid and editable, and the corruption **preserves net charge** on charged leads.

**Fix shipped (`src/compose_v4/rewrite/source_corruption.py`):** the per-atom charged-center guard did not
stop an edit on a *neutral* atom of a delocalized charged motif from shifting net charge (re-derivation can
protonate a carboxylate). Added a universal **net-charge-invariance guard** — reject any candidate whose
successor changes the net formal charge (also blocks a neutral molecule silently gaining a charge). Net
charge is summed over the `is_element` mask (states are slot-stable; a `[:n_real_atoms]` slice drops trailing
real atoms — a measurement bug that first masqueraded as a corruption bug).

## 7. Shardable global-grouping mining (`scripts/mine_edit_traces.py`) + cross-shard design

Mining is a genuine **map → global-group → compile → global-reduce**, not independent per-shard mining
(`tests/test_edit_trace_mining.py`, 4, incl. a cross-shard grouping proof):

- **Shard-local (map, per molecule, embarrassingly parallel):** standardize/validate; emit MMP one-cut
  `(core-key, smiles)`; emit `(Murcko-scaffold, smiles)`; corruption is per-molecule (characterized here,
  regenerated in memory at train time from corpus+seed).
- **Global-reduction (must see all shards):** MMP core-key grouping (a core spans shards); Murcko scaffold
  grouping + sparse k-NN (a scaffold spans shards); reverse-pair + directed `(source, target_key)` dedup;
  per-transformation / per-source / per-family caps (global counts).
- **Compile (per pair):** one-cut compile + executor replay, both directions (re-consumable pool JSONL).

The corpus→TRAIN-partition split is the SAME shared `load_organic_corpus_split` the trainer uses (mine
TRAIN-only → no val/test leakage). Determinism: fixed split seed, stride shards over the canonical-sorted
train list, seeded corruption. Nothing is hardcoded to 500k.

## 8. Validation shard (Modal, commit a1618c8, broad scope)

One deterministic 20k stride shard of the 400k broad-organic TRAIN partition (scaffold-NN skipped: slow,
~mostly-A2.3-deferred; MMP + corruption + census carry the decision). Measured:

| Phase | Result | Time |
|---|---|---:|
| scan + split (500k → 400k eligible train) | shard = 20,000 | **48.1s** (one-time) |
| map + group (MMP one-cut) | **841 pairs** grouped | — |
| compile + reduce (both directions) | **pool = 1,387** verified records (mmp 1,390 → dedup 1,387) | **89.9s** |
| corruption characterization (100 samples) | mix as §6 (regenerated at train time, not mined) | fast |

The **841 MMP pairs → 1,387 compiled, executor-verified, re-consumable records** in a single 20k shard (vs
0–7 in the 1k fixtures) confirm MMP pairs scale **super-linearly** with corpus size — and this is a
single-shard **LOWER BOUND**: the full-run global grouping over the whole 400k train partition recovers
cross-shard core-key pairs a single shard cannot see. Provenance in the summary: commit a1618c8, corpus
SHA-256 (pinned on Modal), scope hash `e59fb09801459470`, standardization / operator-registry / compiler /
scope-module hashes.

**Projected full-run cost** (20 shards, from the shard's measured phase timing): one-time scan+split **48.3s**
+ per-shard map+compile **90.4s** → **0.52 wall-hours serial (~31 min), ~0.52 CPU-hours**; embarrassingly
parallel under Modal fan-out, so far less in practice. Projected compiled pool ≥ **27,740 records** (20 ×
1,387, a strict LOWER BOUND — cross-shard global grouping recovers more). The corruption layer is **not
mined** (regenerated in memory at train time from corpus+seed), so it adds no mining cost. Cheap and
well-bounded. Full machine-readable summary: `diagnostics/composition/broad_mining_validation_shard_summary.json`.

> **Bug caught + fixed during validation:** the Modal container image has no `git` binary, so the miner's
> provenance call crashed at the end of the first two runs — after all mining, before the summary + pool
> were written, which would have silently discarded the full-run output. Now git-safe (commit 183023e); the
> launcher supplies the commit. The confirming run completed and persisted summary + pool.

## 9. Wiring + provenance

- Miner (`scripts/mine_edit_traces.py`) and trainer (`scripts/train_tracelet_cnof_gate.py`, under
  `--organic-vocabulary`) both load `BROAD_ORGANIC_V1`; the CNOF loader survives only for the de-novo base /
  ablation. `corpus_scope` + `corpus_scope_hash` are persisted in checkpoint metadata; the mining summary
  carries the scope descriptor + full census.
- Pre-launch gate green throughout (485 tests + ruff + clean-tree + corpus-on-volume + provenance hashes);
  every Modal launch from a clean committed worktree at the launch tag.

## 10. Remaining (Phase-2 / post-authorization)

- Hierarchical training sampler (layer → curriculum bin → example) with measured cold-element coverage
  floors + optional cold-param warm-up.
- Load-side fail-loud on `corpus_scope_hash` mismatch (evaluate_tracelet_rollouts).
- A2.3 general scaffold-pair compiler (the scaffold-NN layer is candidate-mined but ~98% deferred one-cut).
- Production preflight gate (loads the real base checkpoint; broad-element embeddings/heads load; every
  enabled broad-element op has positive targets; cold params get finite gradients; no unsupported
  charge-changing target enters training; charged-source rollouts valid at every intermediate; report
  neutral/charged/CNOF/S/halogen/P separately; `NO_GO_BROAD_CHARGE_CONTEXT` on charged-context failure).

## 11. Decision: `GO_FOR_FULL_500K_MINING` (broad scope)

Every gate is green: the broad-organic scan runs at 500k scale (48s → 400k eligible train, no leakage), MMP
mining compiles **1,387 executor-verified records** from a single 20k shard with super-linear scaling, the
projected full-run cost is ~30 min of embarrassingly-parallel MMP mining, the shared loader + scope hash are
wired through mining/training/metadata, and the coverage (100% of benchmark leads), cold-vocab (all classes
supervised), and chemistry/charge-preservation evidence all pass. No `NO_GO` condition holds.

**Recommendation:** authorize the full 500k broad-organic MMP mining run (all 20 shards → global grouping →
compiled pool + scaled manifest). It does NOT authorize the B-edit A100 training run — that follows the
production preflight (§10), which loads the real base checkpoint and must independently clear the
broad-element load / cold-gradient / charged-context gates (returning `NO_GO_BROAD_CHARGE_CONTEXT` on any
charged-context failure, no silent neutral fallback).

> STOP here for explicit authorization before launching the full 500k mining run.
