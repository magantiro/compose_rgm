# Matched linker cell-allocation pilot

## Scientific identity

The problem is repeated linker endpoints at the published 100-attempt scale.
The generated object remains a complete executable COMPOSE program and one
connected molecular endpoint between two supplied fragments. The hypothesis is
that tempering the training-derived final-size/ring cell probabilities increases
distinct valid linker yield without reducing the quality of the distinct
outputs or changing hard structural fidelity. The frozen reference checkpoint,
eight-offer panel, learned complete-program selector, connector catalog,
executor, evaluator and prompt set remain unchanged.

The causal comparison has two arms. `frozen` uses the existing joint cell prior.
`sqrt_train_mass` assigns each reachable cell probability proportional to the
square root of its existing positive prior mass, then samples catalog content
within that cell with the unchanged square-root occurrence law. Both arms
sample independently from the same prompt-derived seed. The intervention
preserves every reachable cell and introduces no QED, SA, target-drug-specific
rule, cross-attempt rejection, replacement draw, or post-hoc deduplication.

The molecular support remains the declared 40-active-atom, 48-slot,
32-primitive, eight-block, non-stereochemical COMPOSE graph representation.
Only the frozen split-first training connector catalog supplies structural
content. The benchmark prompts condition the two boundary interfaces but do
not supply fitted content or a quality label.

## Development comparison and acceptance

Use all ten pinned linker prompts, development seed 4, 100 complete attempts
per prompt and arm, eight offered candidates per attempt, one CPU worker and
the same frozen model. This totals 2,000 attempts and 16,000 offers. Seed 0
was the earlier 20-attempt development panel; seeds 1-3 are the ongoing frozen
official evaluation and are excluded from this pilot. Prior observations from
seeds 1-3 motivated the general uniqueness diagnosis but are not used to tune
the proposed law or choose a prompt. A future claim about the revised law
requires new independent seeds after this development decision.

Record every attempted offer, refusal, exact execution, model score, selected
endpoint, RNG state and duration. Lock all 100 samples for a prompt and arm
before applying the pinned official fragment evaluator. Missing outputs remain
in the denominator. Report validity, exact mapped-core/path fidelity,
uniqueness, quality, diversity, output coverage, and per-prompt results. The
quality among distinct outputs is a diagnostic, not the benchmark headline.
Scaffold morphing is an alias of the same linker input pairs and generated
samples, not an independent replicate.

The predeclared development promotion gate requires at least 1,000 outputs,
100% chemical validity and exact core/path fidelity in each arm. Relative to
the matched `frozen` arm, `sqrt_train_mass` must gain at least three quality
points and eight uniqueness points, with diversity falling by no more than
0.02. Failure is reported without changing this gate. Passing it authorizes
preparation of a fresh-seed evaluation contract; it is not itself a final
benchmark win. The published GenMol V2 one-step linker and InVirtuoGen values
are offline comparators only and do not enter sampling or promotion.

## Execution boundary

The independent worktree for this revision is
`/Users/rmaganti/compose_rgm_git/.worktrees/fragment-linker-broaden-v1`.
The original fragment worktree and its running official linker, motif and
decoration jobs are read-only. `prepare` must verify the self-hashed contract,
source revision, clean tracked code, pinned physical inputs, evaluator and
software versions, then publish one immutable manifest. `run` may start only
after explicit authorization bound to that contract payload hash. A completed
attempt replays its saved RNG chain on resume. An interrupted started attempt
fails closed for explicit recovery, and there is no automatic retry.

The matched pilot is a new development result. Neither its outputs nor its
metrics may be inserted into the still-running official seed-1-to-3 result or
reported as independent confirmation.
