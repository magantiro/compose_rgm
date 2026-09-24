# Complete-program fragment adapter: implementation and development gate

## Scope and scientific identity

User authorization: 2026-09-24, integrate existing COMPOSE complete-program
machinery under fragment constraints, without retraining, reward guidance or a
large benchmark launch. Frozen superstructure, T4 and PMO remain unchanged.

The output is one complete connected molecular graph, produced by exact replay
of a bounded coordinated program. The hypothesis is that the existing structural
region compiler plus the frozen learned mark model can improve conditional
completion over the local attachment-controlled adapter. This is an inference to
test, not an established explanation of the earlier quality loss.

Use the actual shared structural-subgoal representation, deterministic region
compiler, program extraction and replay. Do not copy their chemistry into a new
fragment implementation. Required core atoms/bonds and perceived chemistry stay
locked at every internal state. Only declared external interfaces may change.
Attachment coverage is required at the **completed endpoint**, not after every
primitive. Program proposals are deduplicated by canonical endpoint within each
bounded candidate panel; repeated outputs across attempts remain in metrics.

## Prior and support

Keep checkpoint SHA-256
`24117dfeaee91729bb4ebccb5eb5b993b4a6218605045e51917823d086b4c1e4`.
Do not switch on its untrained whole-ring head. Score the compiled primitive
traces using its actual native mark log probabilities, with its original
vocabulary and semantics. Use the mean log mark probability as a declared
length-normalized **panel ranking score**, not a claim of exact path probability,
Doob conditioning, or the T4 reward controller. Select stochastically by softmax
at unit temperature. Non-finite model scores are explicit support abstentions;
no silent fallback to uniform or unlearned selection.

Scoring-codec clarification from the first incomplete support preview: semantic
cycle records are translated to the checkpoint's historical record type only
after its native executor agrees on canonical molecular identity, per-slot atom
identity/charge/hydrogens and resonance-invariant mapped bond classes. Raw Kekule
phase is not molecular support. This correction does not change the committed
trace or the exact fragment lock. The incomplete raw-array-equality preview is
retained separately; no quality results were calculated or used to select it.

Content may come from whole observed training regions with one or two boundary
edges, including ordinary chains, branches, aromatic/aliphatic rings and fused
heterocyclic systems. No acyclic-only or C/N/O/F-only rule is permitted. Executor
and model support are measured separately. Preserve source rows and boundary
signatures. Graph representation does not claim stereo preservation beyond its
existing support; record stereochemical source annotations rather than pretending
to learn a stereo head. Bound execution to the existing 48 slots, 40 active atoms
and 32 primitives; report over-budget regions instead of truncating them.

Before region extraction apply the existing `ringcore-v1` scaffold partition,
algorithm `murcko+carbonized-wl3` v2, ratios 0.90/0.05/0.05, to the hash-bound
GuacaMol 500k source. Exclude the ten benchmark drug identities. The first local
implementation gate may use the first 10,000 eligible training molecules in the
already frozen source order; this is a documented density/compute subset, not a
restriction on representable chemical structures. No QED/SA values, benchmark
identity or answer structures enter content ranking or generation.

The previous acyclic catalog did not apply RingCore's internal scaffold split.
It is quarantined as a negative engineering diagnostic, not a split-clean prior
or a paper result. Its original artifacts remain immutable.

## Gates and order

1. Unit tests: exact multi-step branch, aromatic ring, saturated heterocycle,
   fused-ring region and two-boundary construction; refuse core mutation,
   undeclared attachment, invalid chemistry and unsupported model scores.
   Verify scorer influence, determinism and explicit no-output accounting.
2. Instrument offered, compiled, lock-compatible, model-supported and selected
   complete programs. Report primitive count, ring addition, structured/anchored
   and two-boundary provenance. Compare against saved local traces and the
   unmodified T4 constructors. Do not call this a full-stack test if support
   collapses to one-step or acyclic completions.
   Freeze eight complete candidate attempts per output attempt, canonical panel
   deduplication, square-root observed-region occurrence weights, uniformly
   permuted declared boundary order, and source-capacity reservation of the actual
   smallest compatible observed region per remaining required attachment. Draw every region before compiling the
   completed program. Do not count failed candidates as fresh output attempts or
   hide their work. The support preview uses two output attempts per prompt;
   it is not the quality pilot and will not select per-drug settings.
3. Only after the support gate passes, one matched development pilot: all ten
   motif and all ten decoration prompts, 20 attempts each, seed 0, same frozen
   checkpoint and official evaluator. Reuse the immutable matching baseline
   only after hashes/settings are verified. Persist every candidate, model score,
   selected program, endpoint, seed and refusal. Report candidate/compiler/model
   work and wall time; candidate compute is not assumed equal across adapters.
4. Report all official chemical metrics on all attempted outputs. Prompt fidelity
   is separate. Preserve negative results. No per-drug tuning or larger launch.

Linker/morphing are not admitted to this pilot. The current joint-plan binder
uses bounded beam-style prefix pruning; do not import it under a no-beam claim.
Two-boundary compiler unit cases are capability checks, not official linker
results. Superstructure is an unchanged regression control, not an optimization
target here.

## Support preview v3: capability passes; completion efficiency fails

The complete 40-attempt preview emitted 20/20 motif and 17/20 decoration
endpoints. Every selected program was multi-primitive; 14/20 motif and 17/17
decoration endpoints added a ring. Candidate counts were 125/160 and 77/160
model-supported, respectively. This is capability evidence, not quality evidence.
No QED/SA values were calculated for selection or this decision.

Two Maribavir decoration attempts and one Liothyronine decoration attempt
exhausted their panels. All required contexts had observed compatible regions,
but their minimum size was two atoms, not the one atom reserved by the initial
implementation. Reserving one let earlier regions consume necessary capacity.
The next revision reserves the measured minimum per remaining context, with
explicit abstention when the sum cannot fit. This is a generic feasibility
correction, not a change to executor support or a drug-specific exception.
Keep the v3 manifest, shards and pre-correction adapter snapshot immutable.
Repeat the same no-score support preview before the one matched quality pilot.

This adapter reuses the shared structural-goal compiler and frozen local model;
it is not yet the entire T4 proposal mixture or a route-distilled actor. The
T4 substituted-ring constructor passes an integration fixture, but no T4 lane
distribution or two-boundary benchmark result is claimed from this preview.

## Native T4 integration, before any quality pilot

The user clarified that the actual working T4 constructors must be reused, not
only a compatible structural compiler. `fragment_t4_programs.py` now imports
the unchanged `progressive_structured_sampler.synthesize_lane` dispatcher for
shallow, structured and anchored replacement. Shared `attachment_bindings`
restricts original-atom operands to declared interfaces/unlocked atoms; the
existing dependency/conflict/resource scheduler retains complete block
boundaries. Exact replay and the hard lock remain final admission requirements.

On an entirely locked prompt there is nothing an excision constructor may
delete. Accordingly the generative mapping is: privately propose training-region
completion, optionally refine its editable regions through an actual complete
T4 program, then replay/admit the combined program. Both stages share the same
32-primitive/eight-block/40-atom total support. No seed prefix is a separately
accepted output. Failed refinement is a recorded candidate refusal, not a
fallback to the seed. The frozen molecular model scores the entire combined
trace. No T4 docking, QED/SA margin, reward allocation or final property filter
is imported. This is a generation-specific proposal policy, not a claim of
the identical T4 optimizer or proposal distribution.

Freeze eight candidate draws in repeated four-lane order: training-region only,
then region plus native shallow, structured, anchored replacement. Do not tune
the lane weights by drug. The next all-prompt no-score preview must show every
native lane has model-supported proposals, each task selects at least one native
T4 descendant, >50% selected programs are multi-primitive, >=10% add rings,
>=90% of output attempts complete per task, and every prompt produces an output.
These checks are fixed before running the preview or inspecting quality. They
establish integration/coverage, not a quality improvement or promotion.

The one quality pilot remains 20 attempts per prompt and uses the same complete
proposal/scoring policy as its qualified preview. The saved local baseline is
matched on prompts/checkpoint/seeds/attempts/evaluator, not equal proposal work.
Report the training-content/coordination/refinement bundle as such; it cannot
individually identify the causal contribution of the three changes. A later
isolating ablation requires its own freeze and is not silently added here.

## Verification status before the matched pilot

The focused program/compiler/region/resume tests passed (41 tests before adding
the two-worker partition test). The broader fragment/T4 dependency test command
returned 123 passes and one historical contract failure:
`test_t4_dynamic_v1.py::test_v1_development_contract_is_three_cell_route_empty_and_hash_bound`.
That frozen contract expects `adaptive_program_optimizer.py` SHA-256
`3d185c06d4b12055119ce30e0cd373c84aa8c3dcbecf1184e361b46c22658e21`, whereas both
the unmodified worktree-base file and current file hash to
`dabaff25d07654abebafa3ed42ecc3c454f2d16287bd2a0f2c52f440ed7c04c7`.
No contract or unrelated controller file was changed to pass this test. The
repository-wide suite has not passed or been claimed to pass for this milestone.

The pilot runner supports at most four independent one-thread CPU workers,
partitioned by whole drug pairs. Seeds remain per prompt; each candidate draw
and selected attempt is checkpointed. Final reduction requires all 20 named
prompt rows, not a cross-run glob. This is an execution-speed change only;
proposal work and candidate law are unchanged. Worker count is part of the
immutable pilot manifest.
The local resource check reported 12 logical CPUs and 32 GiB RAM; the preview
worker used about 1 GiB resident memory. Four independent workers preserve each
prompt's RNG stream and leave other local workloads CPU/memory headroom. This
parallelism changes wall time, not sample/candidate budgets or proposal support.
