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

## Step-6,250 pancake audit: global distribution and local execution

The exact 2,000-sample artifact was re-audited rather than inferred from two
illustrative trajectories.  The machine-readable event report is
`diagnostics/pancake_step6250_eval2000_event_audit.json`.

### Global distribution error

The retained checkpoint is a strong rescue base, not a completed generator.
Its neutral-CNOF matched FCD is 11.288, with 100% validity, uniqueness, and
novelty.  The largest structural deficits are under-generation rather than
global over-cyclization: mean heavy atoms are 25.07 versus 26.61, mean bonds
are 26.47 versus 29.02, and mean cycle rank is 2.40 versus 3.41.  Atom-count,
bond-order, and cycle-rank total variation are 0.117, 0.075, and 0.3295,
respectively.  Three/four-member rings are locally overrepresented, but the
model simultaneously lacks roughly one unit of total cycle rank per molecule.

Deletes exceed inserts by 3,536 across 2,000 trajectories, exactly a mean net
change of -1.768 atoms per molecule.  Reconstructed source trees average 26.84
atoms while final molecules average 25.07.  This identifies family-rate/size
calibration as a separate error from ring topology.

### Learned phase collapse

The sequential compiler order became an almost deterministic generative
schedule:

- atom insertion mean normalized event position: 0.080;
- Graft: 0.437;
- atom deletion: 0.733;
- bond reorder: 0.853;
- atom restate: 0.919; and
- whole-ring grow: 0.987.

The first ring event occurs at mean position 0.978; 99.31% of ring events lie
in the final event decile, and ring-grow is the last event in 94.3% of
trajectories.  This confirms that the model learned one compiler linearization
instead of a time-coherent superposition of valid edit orders.  Atomic ring
commitment prevents independent decoration of ring atoms in the teacher, but
late commitment still leaves inference with a decorated scaffold and a
degenerate residual ring support.  The support-renormalization failure is
therefore downstream of path ordering, not the whole explanation.

### Local execution waste

Graft accounts for 76.07% of all 143,561 sampled events and a mean 73.08% of
each trajectory.  It is a majority of events in 93.25% of trajectories and at
least three quarters in 57.05%.  Compact 2,000-sample artifacts do not retain
intermediate states, but the full index-32 replay proves that 89 of 118 events
can be canonical molecular self-transitions.  These no-ops consume CTMC time
and crowd out the insert, delete, bond, and ring events needed to repair the
global deficits.

### Re-baselined repair order

1. **Rescue the exact pancake checkpoint at inference.**  Remove canonical
   self-successors and merge equivalent Graft matches; calibrate family
   intercepts on frozen validation data; replace Boolean ring-family
   renormalization with supported-match/topology rate mass.  This requires no
   path or support rebuild.
2. **Replace the fixed compiler linearization.**  Build a dependency DAG over
   existing valid rewrite steps using read/write footprints and exact
   commutation checks.  Sample validated linear extensions so ring systems
   commit as soon as their prerequisites exist and independent branch,
   topology, size, and label operations interleave.  Replay every scheduled
   path to the exact endpoint; fall back to the sequential trace on any failed
   certificate.
3. **Retain the good substrate.**  Keep the carbon-tree prior, flexible size,
   Graft, whole-ring grow/delete, aromatic lowering, all-step
   validity/connectivity, and ancestral CTMC sampler.  The measured failure is
   the scheduler and rate normalization, not evidence that these operators or
   the RGM thesis should be discarded.
4. **Keep exhaustive chemistry as an oracle.**  Production uses sparse local
   rule matching; the expensive catalog traversal is reserved for equality
   tests and rare fallback cases.

### Symmetry correction must thin, not renormalize

The first direct rescue pilot deliberately tested the simplest possible
canonical mask on the frozen pancake checkpoint.  Eight completed shards (80
samples) retained 100% final validity, connectivity, non-null rate, and
uniqueness, with zero recorded canonical self-events.  However, mean size rose
to 30.75 atoms, atom deletion nearly disappeared (2 total events), and the
small-ring prevalence was 33.75%.  Masking self-Grafts and renormalizing their
conditional family distribution therefore changes the learned process; it is
not a lossless checkpoint repair.

The corrected legacy inference uses CTMC thinning.  A pre-quotient Graft is
sampled from the original learned operand distribution.  If its canonical
successor equals the source, operational time advances but the molecular state
does not change and the event is recorded separately as a virtual jump.  This
is equivalent to multiplying the chemical Graft intensity by the learned
non-self mass.  It preserves the checkpoint's state law (up to the model's
permutation equivariance), avoids presenting gauge events as chemical edits,
and does not alter clean quotient-model training.  The 100-sample thinned
evaluation is the active direct-rescue gate.

The dependency scheduler is a meaningful formulation change, but not an
open-ended computational project: it transforms existing traces, performs one
validated replay per sampled linearization, and reuses all molecular endpoints
and operator implementations.

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

### Continuation result and authorized fallback

The first arm failed cleanly.  The fresh-optimizer continuation selected none
of its 250/500/750-update checkpoints: validation loss rose from 11.6914 to
12.0112, 12.0760, and 12.0837, and early stopping fired.  The original
step-2,500 checkpoint remains intact.  This rejects another `1e-4` continuation
from the same weights; it does not reject the quotient semantics.

The authorized fallback is a 500-update maximum legacy-transfer gate at
`5e-5`, evaluated every 100 updates with patience two.  Exactly 109 tensors
match by full name and shape and are transferred.  The resized
`ring_system_template_key.weight` and four new `ring_system_role_head` tensors
retain fresh initialization.  The gate reuses the current path, validation,
test, and training-support caches and therefore requires no chemistry
recompilation.  If it does not improve the corrected validation objective, it
is stopped rather than extended.

### Completed step-2,500 ring attribution

The completed 100-sample rollout is 100% valid, connected, unique, and novel,
with zero canonical self-events, zero delete-to-one collapse, and zero event
budget exhaustion.  Its full-reference FCD is 24.708 and its neutral-CNOF
matched FCD is 19.180; both are high-variance 100-sample diagnostics, not final
benchmark estimates.  The material chemistry failure is now localized: 40% of
molecules contain a three- or four-member ring, versus 5.26% in the reference.

This is not a Graft artifact.  Replay of every committed state shows that all
53 newly created small rings arise at `ring_system_grow` events; all 40 final
small-ring molecules first acquire the small ring from that family, and no
small ring is subsequently removed.  The 265 ring-grow events therefore create
a small ring 20.0% of the time.  By contrast, only 3.01% of the empirical
semantic ring-template mass in the fitted catalog contains a three- or
four-member cycle.  The error is thus state-conditional template/support
calibration or a learned residual overwhelming the base measure—not an
empirical catalog dominated by small rings and not an uncoordinated Graft.

The simplest next ring intervention, if the legacy-transfer gate does not
resolve this, is a matched one-dimensional/topology-group calibration:
factor ring selection into corpus-frequency topology signature (cycle-size
multiset, fused/bridged/spiro class, aromaticity) plus a learned residual, while
retaining positive support for rare rings.  Do not add another ring operator or
hard-ban rare rings.  Confirm the fix by replay attribution and matched
100-sample ring histograms before any longer run.

### Legacy-transfer gate result

The bounded transfer gate completed for 500 cached updates and did not resolve
the failure.  Its best validation point was update 400 at loss 12.7654 and
53.81% family accuracy; the selected quotient incumbent remains better at loss
11.6977 and 58.08%.  The transfer checkpoint was therefore rejected before
rollout/FCD computation.  This negative result rules out a cheap initialization
rescue and activates the support-versus-residual ring diagnostic above.

### Exact support-versus-logit diagnostic

A read-only audit reconstructed 12 states immediately before retained
`ring_system_grow` events and evaluated the exact legal template set.  The
machine-readable row-level result is retained at
`diagnostics/ring_calibration/step2500_exact_support_audit_12.json`.  The
catalog contains 3,092 semantic templates, of which 285 contain a three- or
four-member minimum-basis cycle; their unconditional empirical prior mass is
only 3.017%.  In broadly supported states, the production small-ring mass was
typically 1.7--4.6%, so the learned model does not globally amplify rare rings.

The failure is instead concentrated in late, highly decorated states.  Three
audited support sets contained 2, 2, and 5 legal templates, respectively, and
every legal template contained a three- or four-member cycle.  Their
support-conditioned small-ring mass was therefore exactly 100%, and each
committed event created a small ring.  Across the 12 audited states, exact
support conditioning raised mean production small-ring mass to 28.0% despite
the 3.017% global base measure.  This localizes the MVP failure to hierarchical
family enablement: the family head sees a Boolean “some ring action exists” but
does not know that the remaining support is a tiny, rare-topology residue.

The immediate inference-only experiment excludes templates containing cycles
of size at most four from the exact sampling support.  If no ordinary template
remains, ring-grow returns no action and ancestral sampling falls back to
another valid rewrite family.  This changes neither the checkpoint nor the
training objective, retains ordinary/fused/bridged/spiro and aromatic/saturated
ring systems, and is an ablation rather than the final claim.  A 100-sample
matched rollout determines whether this simple support policy is sufficient.
The final trainable solution, if needed, is to expose support-quality summaries
to the family decision or train a topology-group gate; it is not another ring
operator.

### Simplified production ring-rate formulation

The final formulation must not reproduce the failure by merely replacing one
large mask with another.  A Boolean ``ring-grow enabled`` family followed by a
softmax over the remaining legal templates renormalizes a rare residual support
set to probability one.  That is exactly what the late-state audit observed.

Replace that hierarchy with **masked, unnormalized topology-group
intensities**.  Five/six-member ordinary, fused, spiro, bridged, and rare-small
topology groups receive separate nonnegative CTMC rates.  Inapplicable groups
have rate zero; removing common groups therefore lowers the total ring hazard
rather than transferring all of it to a rare small-ring rule.  Conditional on
a topology group firing, the model decodes an attachment interface and local
electronic labels.  The empirical corpus frequency is an offset/base measure,
while the network learns a context-dependent residual.

The corresponding executor is sparse and local:

1. pre-index ring rules by topology group and attachment/valence signature;
2. query only signatures present at the current state;
3. apply structural conditions before neural scoring;
4. canonicalize and merge only the retrieved executable successors; and
5. run one exact sanitize/executor check on the selected successor, with the
   exhaustive catalog retained only as a test oracle and rare fallback.

This preserves rare rings, whole-ring coordination, aromatic lowering,
fused/bridged/spiro support, canonical successor semantics, and all-state
validity.  It removes the scientific defect (support-collapse renormalization)
and the computational defect (flat per-row traversal of thousands of
templates) together.  Acceptance requires equality with the exhaustive oracle
on enabled topology groups and canonical successor sets, plus matched rollout
improvement in small-ring prevalence without losing fused/spiro/bridged
coverage.

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
