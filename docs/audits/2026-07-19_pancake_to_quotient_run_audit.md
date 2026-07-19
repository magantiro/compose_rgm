# Pancake-to-quotient run audit — 2026-07-19

## Decision boundary

This audit separates changes to the scientific generator from changes that
only move or cache computation.  The current run is not evidence that the RGM
direction failed: it already removes the measured molecular self-transition
failure of the legacy checkpoint.  It is also not yet evidence that the new
ring formulation improves chemical fidelity.  No additional long run is
authorized until the step-1,500 and step-2,500 sample audits resolve that
question.

The shortest next experiment is a small, checkpoint-initialized comparison
under the same quotient-correct executor.  It must reuse fixed paths and
evaluation data and must not rebuild a flat per-row 4,096-template ring mask.

## Frozen comparison evidence

| Quantity | Legacy pre-quotient checkpoint | Current quotient-correct run |
|---|---:|---:|
| Selected or latest durable step | 6,250 | 2,500 |
| Validation family accuracy | 84.44% | 58.08% |
| Validation GM loss | 22.822, old objective | 11.698, current objective |
| Matched neutral-CNOF FCD | 11.288 on 2,000 | pending |
| Canonical molecular self-events | observed; one archived trajectory has 89 | 0/2,750 at step 1,000 |
| Delete-to-one collapse | not the dominant legacy failure | 0/100 trajectories at step 1,000 |
| Small-ring molecule prevalence | 29.9% | 51.0% at step 1,000 |
| Reference small-ring prevalence | 5.26% | 5.26% |
| Fused molecule prevalence | 26.45% | 48.0% at step 1,000 |
| Reference fused prevalence | 52.18% | 52.18% |

The family-accuracy comparison is diagnostic, not objective-equivalent.  The
legacy validation stream was 40.03% Graft among nonterminal examples; the
current fixed validation batch is 28.24% Graft and contains more difficult
insert/delete/restate/reorder labels.  That distribution shift explains part,
but not all, of the accuracy gap.

The current 3,000-step cosine schedule is confounded with the scientific
changes.  At step 2,500 its learning rate had already fallen from 3e-4 to
4.22e-5 while validation loss was still improving.  The legacy 30,000-step
schedule retained approximately 3e-4 at comparable steps.  A new scientific
conclusion must not be drawn from this schedule mismatch.

## Scientific change audit

| Change | Intended benefit | Evidence now | Decision |
|---|---|---|---|
| Target-independent empirical-size carbon-tree source | Directly samplable valid prior; expose editing rather than null construction | Source validity and endpoint reachability pass; unconditional quality advantage over one-atom/null is untested | **Retain for the main ablation; do not claim quality advantage yet** |
| Flexible-size tree transport | Exercise grow and shrink while preserving the source marginal | Exact endpoints and connected valid paths pass; current samples do not collapse to one atom | **Retain** |
| Subtree Graft macro | Compress topology revision and preserve useful substructures | Current trajectories avoid delete-to-one; Graft remains active | **Retain** |
| Canonical-successor Graft quotient | Merge atom-slot/symmetry aliases and remove molecular self-Grafts | Zero canonical self-events at step 1,000 versus severe archived legacy churn | **Proven useful; retain** |
| Whole-ring-system grow/delete | Commit ring topology and electronics atomically; avoid ring-ear/cage construction | Full-system reachability and teacher support pass; fused/spiro coverage improved; small/bridged rings are overproduced | **Retain as an operator, but recalibrate and ablate** |
| Semantic aromatic representation with executable Kekule lowering | Make the learned topology resonance invariant while preserving executable chemistry | Exact teacher audits and aromatic executor tests pass | **Retain** |
| Factorized C/N/O/F ring labels | Generalize atom labels beyond memorized typed fragments | Held-out-label support tests pass; sample heterocycle frequency is too high | **Retain the factorization; audit its learned/base rates** |
| Empirical ring-topology base measure | Reflect common versus rare corpus rings without removing learned residual rates | Rare 3/4-member rings remain grossly overrepresented at step 1,000 | **Unproven; inspect normalization and train/sample frequencies before the next run** |
| Exact shared ring application conditions for teacher and sampler | Prevent probability mass on non-executable or duplicated ring actions | Semantic equality and zero-missing-teacher audits pass | **Retain the semantics** |
| Flat per-row 4,096-template support/certificates | Reference implementation of the exact support above | Correct but caused the dominant CPU latency and a pathological final shard | **Retire as production representation; retain only as an oracle** |
| Hierarchical hazard/family/action rate model | Separate clock, family choice, and operands | Hazard error improves; low-frequency non-ring family classification remains weak | **Retain provisionally; use a properly matched schedule before changing the loss** |
| 3,000-step cosine decay | Obtain a fast quality readout | Learning rate decayed while loss still improved; family accuracy plateaued near 58% | **Reject for the next comparison** |

## Infrastructure change audit

| Commit group | Scientific effect | Operational result | Decision |
|---|---|---|---|
| `6900f95`–`2be9258`: checkpoint selection, CPU rollout separation, trajectory audit, boundary smoke | None | Durable checkpoints and full trajectories are now available; A100 is released before rollout evaluation | **Keep** |
| `7ea89d0`: fixed evaluation cache and streamed evaluation | None; mathematically identical full evaluation | Removes 37–52 minutes of repeated chemistry work and avoids evaluation OOM | **Keep** |
| `f0ecfa9`: state caches, Graft orbit reduction, incremental tree keys, sparse ring transport | Exact implementation acceleration | Exact oracle agreement; large cache-hit microbenchmark improvement | **Keep** |
| `f63d996`: tree-DP ring prefiltering | Exact implementation acceleration | Useful prerequisite but insufficient for the semantic-certificate long tail | **Keep as a prefilter** |
| `b228e76`–`3bd973f`: per-row support cache and multi-container compilation | None | Moved CPU work off the A100 but required excessive orchestration and storage | **Replace with unique-state/path-witness sparse compilation** |
| `1ac6f19`: hazard and family diagnostics | None | Exposed schedule and family-calibration failures | **Keep** |
| `cdac478`–`c39520c`: precomputed/persisted teacher semantic certificates | None; same exact likelihood | GPU batches became fast after caching, but certificate construction remained pathological | **Keep certificate semantics; produce them once with the path witness, not per row** |
| Current uncommitted stale-shard wait fix | None | Prevents an old support-only shard from causing an immediate boundary crash while its atomic replacement is being compiled | **Keep after regression verification** |

## Re-baselined decision after the partial step-2,500 audit

The next experiment is no longer blocked on a new full-stream compiler.  Eighty
completed step-2,500 trajectories are all valid and unique, with zero event-
budget exhaustion.  Small-ring molecule prevalence improves monotonically from
51.0% at step 1,000 to 42.2% at step 1,500 and 36.25% in this partial step-2,500
set.  Fused prevalence recovers to 32.5%; mean heteroatom fraction is 0.2615
versus 0.2647 in the reference.  The remaining mismatch is substantial, but
the direction is improving while the scheduled learning rate has already fallen
to 4.22e-5.

The shortest controlled run is therefore:

1. Initialize from the quotient-correct step-2,500 selected weights.
2. Reset the optimizer and schedule; run 750 updates at a 1e-4 learning-rate
   plateau after a 50-update warmup.
3. Reuse the existing paths, fixed evaluation batches, and exact support rows;
   do not compile another 1.92M-row stream.
4. Evaluate at 250-update intervals and sample 100 molecules from every
   materially improved checkpoint.  Stop on loss reversal, chemistry
   deterioration, self-events, or validity failure.
5. Only if this arm fails, initialize the 109 shape-compatible shared tensors
   from the legacy pancake checkpoint, reinitialize changed ring-specific
   layers, and run the identical corrected objective and schedule.

This isolates the schedule/initialization question before changing capacity,
teacher paths, or the loss.  The sparse unique-state/path-witness compiler
remains required before the next large from-scratch stream and lipid-scale
training, but it does not block this cached continuation.

## Replacement for flat exhaustive ring support

The production kernel should compile the same exact action set into sparse,
canonical objects:

1. Canonicalize topology, resonance aliases, automorphisms, and identical
   successors once when the catalog is built.
2. Attach the canonical topology ID, scaffold placement, and semantic teacher
   witness to each compiled path step.
3. Index canonical ring rules by topology and local valence requirements.
4. Query that index once per unique molecular state and return only supported
   canonical candidates.
5. Compute the exact within-family partition with sparse segmented log-sum-exp.
6. During ancestral sampling, update rule matches only around the atoms changed
   by the previous rewrite.

The current exhaustive implementation remains the oracle.  The new kernel is
accepted only after exact successor-set, enabled-family, teacher-probability,
total-rate, and gradient equality tests on enumerable states and a broad random
set including saturated, aromatic, fused, bridged, and spiro systems.

## Minimal next-run matrix

1. Finish the step-1,500 and step-2,500 100-sample audits.  Hard-fail on any
   validity/connectivity error or canonical self-event.  Compare ring sizes,
   bridged/fused/spiro rates, heteroatoms, bonds, event mixture, atom-count
   trajectories, novelty, and the high-variance 100-sample FCD components.
2. If step 2,500 materially improves chemistry, initialize from it and run a
   500–1,000 update plateau-LR continuation with a fresh optimizer.  Reuse all
   fixed data and start from a disjoint deterministic stream range.
3. If chemistry does not improve, run a matched two-arm warm-start test:
   current step-2,500 versus the 109 shape-compatible shared tensors from the
   legacy step-6,250 model.  Reinitialize the changed ring-template embedding
   and new ring-role head in the legacy arm.  Use identical quotient-correct
   semantics, batches, LR, and evaluation seeds.
4. Do not compile another full training stream.  The next gate needs at most
   1,000 cached/sparsely compiled updates and early 100-sample evaluation.

## Lipid transition gate

COMPOSE-Lipid does not require GuacaMol ring fidelity to be perfected before
data engineering and computational smoke training begin.  It does require the
shared editor and conditional machinery to be credible.

The public AGILE wet-lab library provides a useful concrete boundary, though it
must not be treated as representative of the final COMPOSE-Lipid corpus.  An
RDKit audit of all 1,200 released `combined_mol_SMILES` found 720 ring-bearing
molecules (60%), but only 0.6 rings per molecule: 300 five-membered and 420
six-membered rings, with no 3/4-member rings and aromatic rings in 10% of the
library.  Every ring occurs in one of the 20 head groups; the 12 linker and five
tail building blocks are acyclic.  Thus this lipid slice is not ring-free, but
its ring topology is drastically simpler than the GuacaMol reference.  It also
averages 47.77 heavy atoms, so graph size, long/repeated tails, charge and
protonation, linker chemistry, and symmetry replace fused-ring topology as the
main near-term modeling risks.

Move to lipid smoke generation after:

1. the non-ring/acyclic small-molecule slice has 100% all-step validity and
   connectivity, no collapse or self-thrashing, realistic size/atom/bond/charge
   marginals, high uniqueness/novelty, and adequate coverage;
2. one cheap Paper-1 conditional proof (initially QED targeting/optimization)
   shows that valid-rewrite guidance or reward fine-tuning improves a matched
   filtering baseline under the same oracle budget; and
3. the proposed lipid datasets are audited for ring prevalence, chemistry
   support, size, charge, linker/head/tail composition, duplicates, splits,
   assay provenance, and synthesis eligibility.

Paper-1 conditional benchmarks remain the method-validation matrix.  Paper 2
adds lipid-specific endpoints—delivery/potency oracles, formulation and
synthesis constraints, linker-conditioned transfer, and prospective selection.
For an acyclic-first lipid smoke model, the ring family can be disabled or
strongly down-weighted according to the training distribution; the final
claimed lipid support must match the actual audited datasets rather than an
assumption that all ionizable lipids are acyclic.

The staged authorization is therefore:

1. **Now:** ingest and audit lipid datasets; compile lipid chemistry support;
   benchmark throughput on larger graphs; build the study-aware oracle split.
2. **After a credible non-ring small-molecule checkpoint and one matched QED
   proof:** train an AGILE-like restricted lipid smoke model with only observed
   five/six-member head-ring support and generate exploratory, non-claim-bearing
   samples.
3. **Before locked Paper-2 generation:** freeze the full corpus-derived ring and
   chemistry support, pass exact reachability on the eligible corpus, freeze a
   10,000-sample lipid base generator, validate the oracle, and only then
   activate guided/reward-fine-tuned candidate selection.
