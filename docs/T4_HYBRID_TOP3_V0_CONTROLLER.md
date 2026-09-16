# T4 hybrid top-three plus Dynamic-v0 controller

## Question

The target-conditioned utility model failed its frozen V4 comparison. This
revision therefore rejects that arm and tests the smallest evidence-driven
alternative without scoring anything: one structural-control candidate, two
distinct candidates ranked by the cross-fitted target-blind utility model, and
then the unchanged Dynamic-v0 refinement law.

The generated object is unchanged. Candidates are complete exact-realized
structural programs from the immutable baseline learned compositional pool.
This revision does not generate another program, change the compiler, change
eligibility, inspect a target score, or fit a model.

## Frozen schedule

For each of `5ht1b_0`, `braf_1`, `jak2_1`, `parp1_0` and `fa7_0`, use the
whole-source fold checkpoint matching the source index. Retain only exact
eligible baseline-learned candidates. Canonicalize endpoints, then select:

1. the highest fixed target-blind structural-control score;
2. the highest cross-fitted target-blind utility score at a different endpoint;
3. the next highest target-blind utility score at a third endpoint.

Ties use original generator rank and candidate identity. The target-conditioned
arm is not evaluated or used. A cell with fewer than three distinct eligible
endpoints records a macro-support abstention and starts unchanged Dynamic-v0 at
call 1. There is no fallback candidate, score-aware fill, retry, replacement or
backfill.

For a supported cell, calls 1 to 3 are the frozen macro candidates. Every
finite scored candidate is admitted in charged-call order to one run-local
archive, while a charged failure is retained but not imputed or replaced.
Unchanged Dynamic-v0 begins at call 4. All cells report best-so-far at calls 1,
5, 10, 20, 50 and 100 with no plateau stopping.

## Evidence boundary

This milestone is zero-oracle. It produces candidate and conditional-admission
locks only. It is not prospective utility evidence, a five-cell qualification,
or an InVirtuoGen comparison. The lock must report support abstentions rather
than silently shrinking the panel or fabricating candidates.

Any scored run requires a later exact authorization binding the physical and
payload hashes of both locks, a clean launch revision, frozen evaluator and
docking seeds, the unchanged Dynamic-v0 implementation, five single-CPU
workers at most, and a 500-call ceiling. No call is authorized by this file.
