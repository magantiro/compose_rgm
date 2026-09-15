# T4 Route-Distilled Complete-Program Decoder

## Problem, output and claim

The current route-distilled actor recognizes teacher decisions but generates
programs at the wrong scale. The 77 teacher programs contain 5 to 32 primitive
decisions (median 17), while the runtime proposal mechanism permits at most
three coarse modules and the recognizer compressed only 2 of 1,372 decisions
into compound modules.

The primary model output for this milestone is a target-free distribution over
a complete legal rewrite program:

\[
q_\theta(a_{1:T},T\mid x_0),\qquad 1\le T\le 32.
\]

Every next action is selected from the exact canonical legal fiber at the
current state. The decoder may use relative references to atoms created by an
earlier action. The claim under test is that complete autoregressive decoding,
not another endpoint ranker or a larger fixed macro vocabulary, repairs the
observed candidate-support failure.

This is trained-on-T4 route distillation. It is not winner-independent or
held-out benchmark evidence.

## Frozen evidence motivating the revision

- 77 complete teacher routes, 1,378 primitives and 15 source groups.
- Exact teacher replay coverage and precision are both 1.0.
- Only 2 compound decisions were recognized; 1,370 primitive fallbacks were
  preserved.
- The fitted actor achieved 0.9157 source-balanced teacher-decision top-1 and
  0.9932 top-10 accuracy.
- Across the first nine completed grouped-source shards, both the context actor
  and source-balanced marginal generator had zero autonomous exact or radius-2
  transformation recovery in 128 attempts per source.
- When teacher endpoints were injected only for a rank diagnostic, the existing
  graph ranker achieved 0.3878 transformation recall at 10, 0.7347 at 32 and
  1.0 at 128.

These observations distinguish recognition from generation. The prior
zero-recovery result remains immutable.

## Data and split

Use the same 77 signed teacher traces and the frozen three grouped folds. Each
fold holds out one source molecule from each of the five proteins, and all
routes for a source remain together. Fit vocabularies, feature preprocessing,
rule probabilities, WHERE/HOW scores, stop probabilities and program-level
rankers on training sources only for grouped evaluation.

After architecture selection, an explicitly separate all-route checkpoint may
fit all 77 traces for the later trained-on-T4 performance experiment. Grouped
metrics and all-route training metrics must never be merged.

## Decoder

At prefix state \(x_t\):

1. enumerate the exact canonical legal successor fiber for every supported
   rewrite rule;
2. predict the primitive rule and STOP/CONTINUE probabilities;
3. rank legal WHERE/HOW instantiations using graph, changed-region, dependency
   and prefix context;
4. advance a bounded diverse beam;
5. exact-execute each retained successor;
6. emit complete candidate programs at STOP and at the 32-primitive ceiling.

The initial support remains the declared Active8 rewrite support. No edit
family or executor behavior may be added after inspecting held-source results.
The decoder must reject self events, duplicate canonical prefixes, more than 40
active atoms, more than 32 primitives and more than eight dependency regions.

Dependency-region structure is conditioning information, not a manually named
macro library. Primitive emission order remains exact. Created-handle
backreferences are relative and must resolve against the actual prefix state.

## Model comparison

Use one architecture with causal ablations, not a controller zoo:

1. `marginal_legal_decoder`: source-balanced marginal rule and stop law,
   uniform legal WHERE/HOW ranking;
2. `learned_legal_decoder`: graph/prefix-conditioned rule and stop law plus
   learned legal WHERE/HOW ranking;
3. `learned_legal_decoder_plus_sequence_ranker`: the same generated lock,
   reranked by the already registered complete-transformation ranker.

All arms share legal support, source split, maximum depth, beam settings,
candidate count and candidate-work accounting. The sequence ranker may reorder
only autonomously generated candidates.

## Losses

The primary training objective is teacher-forced autoregressive likelihood over
the observed legal action and STOP/CONTINUE decisions:

\[
\mathcal L_{\rm AR}=-\sum_t \log q_\theta(a_t^\star\mid x_t,a_{<t}^\star).
\]

A complete-program contrastive objective may be added only over programs
generated from the same training source. It must not inject held teachers into
the candidate generator. Report the autoregressive-only arm before claiming
benefit from program-level ranking.

## Decoding and candidate locks

Evaluate beam widths 1 and 8 first, with snapshots at primitive depths 8, 16,
24 and 32. Generate up to 128 complete programs per held source for the first
bounded gate. If this establishes healthy support, a separately sealed
proposal-count extension may measure recall at 1,000 without changing the
model.

Candidate generation has no access to target identity, docking scores or
teacher endpoints. Seal source/fold identity, exact states, actions, model and
input hashes before the evaluator loads teacher identities.

## Metrics and gates

Report coverage and precision separately:

- teacher-forced primitive-rule and legal-successor top-k accuracy/NLL;
- emitted program count and source coverage;
- exact-execution precision;
- unique canonical endpoint yield;
- exact route and endpoint recall/precision/MRR;
- radius-2 transformation recall/precision/MRR;
- candidate shortfall, support abstention and dependency-budget abstention;
- legal fibers, expanded prefixes, retained successors, wall time and peak
  memory.

The engineering gate requires exact-execution precision 1.0, deterministic
sealed locks, support enforcement, complete provenance and at least one novel
complete endpoint per held source. Learned generation must be compared with the
matched marginal decoder. A null autonomous-recall result is reported as a
negative result and does not become positive evidence merely because a ranker
can recover an injected teacher.

Teacher recovery is not the task objective. A small molecular-utility pilot may
therefore be authorized separately from a score-blind lock even when exact
teacher recall is imperfect, provided the engineering gate passes and candidate
selection is frozen before docking. That pilot cannot be used to rewrite this
decoder gate or replace candidates after observing scores.

## Dynamic integration after the gate

The later optimization controller will union two independently generated pools:

\[
C=C_{\rm distilled}\cup C_{\rm v0}.
\]

Both pools pass the same exact executor, endpoint filters and deduplication.
They enter one run-local archive. Distilled long-horizon programs provide
teacher-informed structural jumps; the unchanged v0 lane preserves broad novel
exploration and may refine distilled descendants. No Full-146 route is loaded
at runtime.

This integration and any scored five-cell optimization run require a later
launch contract after the zero-oracle decoder evidence is sealed.

## Prohibitions

- no target/protein, seed, route ID or target-to-program runtime feature;
- no endpoint, SMILES, absolute source address or executable teacher trace in a
  runtime checkpoint;
- no teacher injection into autonomous candidate generation;
- no docking or task scores during model fitting or candidate locking;
- no alteration of Active8 support, executor semantics, live runs or frozen
  comparator artifacts;
- no relaxation or relabeling of a previously failed gate.
