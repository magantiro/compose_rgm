# Joint whole-completion structural prior: development contract

## Identity and authorization

2026-09-24 user authorization: implement the proposed whole-completion correction
after the isolated manual decoration audit. The scientific output remains a
complete COMPOSE program and its exact, connected molecular endpoint. Test whether
conditioning the *joint* decoration plan on training-observed whole-molecule size
and ring count repairs over-decoration without sacrificing validity/diversity.
This is a structural-prior ablation, not a full learned conditional chemistry model
or a guarantee of better quality. Superstructure, de novo, T4, PMO, and the ongoing
400-attempt complete-program pilot are immutable inputs, not edit targets.

## One change, fixed before its metric pilot

Reuse precisely the accepted source rows of the split-first 10,000-molecule
training-region catalog. Verify its source, partition assignments and hashes;
no additional corpus examples or benchmark molecules enter fit. Count one vote
per admitted molecule in the joint (heavy atoms, RDKit ring count) histogram.
No QED/SA, activity, target identities, or comparator outputs enter this prior.
Report the histogram and dominant cells rather than silently balancing the corpus.

Given all declared interfaces and retained-core descriptors, use the unchanged
context-compatible one-boundary region library and square-root occurrence weights.
Dynamic programming sums compatible allocations by total added atoms/rings.
Choose a reachable final (atoms, rings) cell according to the training histogram,
then sample a complete allocation from the product region law *conditioned on that
cell*. Do not multiply histogram mass by the number of available combinations:
that would reintroduce combinatorial bias toward large decorations. The DP sums
all supported allocations, with no beam, pruning, drug branches, or sampled-panel
approximation. Add one total pseudocount uniformly across reachable final cells
to preserve positive density outside observed histogram cells. This is density
smoothing, not expansion/reduction of executor or catalog support.

The unchanged library admits regions of at least two atoms; do not add methyl
templates based on the manual audit. Preserve aromatic/aliphatic/fused rings,
branches, heteroatoms and existing source provenance. A prompt with more interfaces
uses the same joint allocation; do not silently restrict it to one/two sites.

Compile using shared `compile_region`, program-stage admission, exact replay,
hard locks and the same native T4 refinements. Keep 40 active atoms, 48 slots,
32 primitives, eight blocks, the checkpoint, eight-candidate panel, mean-local-log
model scoring, and unit-temperature selection unchanged. Whole-completion cells
condition the private initial proposal; a T4 refinement can change the final cell.
Record that change and the selected endpoint distribution explicitly. No property
filter or post-hoc size filter is introduced. No claim of exact CTMC path sampling.

## Qualification and falsifiers

1. Unit tests against exhaustive tiny enumeration: exact cell masses, conditional
   allocation probabilities, multi-interface capacity, positive tail support,
   determinism, impossible contexts, rings and malformed inputs. Verify training
   row admission and reject non-training rows. Use existing chemistry/compiler
   tests; do not duplicate chemistry implementations.
2. Same-checkpoint support preview: two attempts for each of all ten motif and
   ten decoration prompts, eight candidates per attempt. No QED/SA during this
   preview. Require the original support conditions: >=90% outputs in each task,
   every prompt emits, 100% committed connected validity/preservation/fidelity,
   >50% multi-primitive selected, >=10% ring-adding selected, every native T4 lane
   model-supported, and a T4 descendant selected in each task. Do not relax a
   failed gate. Report planned/final atom/ring counts and candidate refusal work.
3. Only after qualification, one matched development panel, all ten motif and
   ten decoration prompts, twenty attempts each, seed zero. Compare with saved
   local and full-program arms only after verifying their input identities.
   Score every locked attempt with the unchanged official evaluator. Preserve
   the known Distance-label caveat; it is not GenMol's original-drug distance.
4. Promotion requires decoration macro quality at least ten percentage points
   above the unchanged full-program arm, output >=95%, uniqueness >=90%, diversity
   >=0.56 and no committed validity/constraint failure. Motif may not lose more
   than five quality points or 0.03 diversity against its unchanged full-program
   arm. These are development gates, not a paper win. If the final size/ring
   distribution shifts but quality fails this gate, reject the prior; do not tune
   per-drug budgets, smoothing, or temperature on this panel.

Persist every planned cell, sampled region/source, program/action, model score,
RNG state, candidate refusal, endpoint, evaluator/config/code/input hash. Use
small resumable units, at most one additional CPU worker while the frozen pilot
is active. No training job, Modal deployment or full official run is authorized
by this development contract. Successful promotion still requires user review.
