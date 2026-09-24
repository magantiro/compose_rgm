# Motif and decoration focused-program development pilots

## Identity and scope

COMPOSE's output here is a complete, connected molecular graph reached by an exact
multi-primitive program. The test is whether the supplied one-boundary fragment
constraint should govern *program termination* rather than requiring a universal
post-completion T4 edit. This is a single fresh-seed development comparison, not a
published-scale benchmark or a new chemical-validity claim. The sealed seed-zero
joint-completion pilot remains the baseline and is never modified.

The declared executor support remains the existing atom/element vocabulary,
40-heavy-atom endpoint limit, 32 primitives, and eight program blocks. The
training-only region catalogs and checkpoint remain frozen. There is no target
drug rule, endpoint lookup, QED/SA-guided selection, beam, or post-hoc redrawing.

## Two distinct boundary topologies

Motif extension offers eight complete motif-rooted region programs from the same
training prior/model used by the seed-zero pilot. The baseline mixed two such
programs with six subsequent T4 refinements; in its frozen output, the selected
unrefined coherent regions accounted for 43/101 joint QED/SA passes, versus
17/94 among the selected T4-refined endpoints. That is an association and a
reason to test the policy, not a causal result. The fresh seed-one pilot asks
whether stopping after coherent motif completion improves the full ten-prompt
mean. It does not change content or model rates.

Scaffold decoration uses the separately frozen training-only smaller-cut-side
pendant library and its capacity-aware complete-program compiler, stopping as
soon as all declared scaffold interfaces are decorated. A zero-quality support
gate passed on ten prompts: 20/20 attempts produced valid, connected and
constraint-faithful outputs; 108 model-supported offers contained one-atom
content and 99 contained ring-bearing content. The support result does not
establish quality. The fresh seed-one quality pilot now tests that question.

Both policies use eight candidate offers per attempt, exact graph/fragment
admission, duplicate suppression, and the frozen COMPOSE learned mean local log
mark with unit-temperature finite-panel softmax.

## Frozen evaluation and falsifiers

Each policy runs 20 ordered attempts on each of the same ten benchmark prompts.
An attempt without a committed endpoint remains a failed placeholder in the
official InVirtuoGen evaluator population. Store every candidate, refusal,
program, selected endpoint, RNG state, score, prompt check, and timing. Lock each
20-sample prompt before evaluating it. Report mean official validity,
uniqueness, quality and diversity across ten prompts, plus exact constraint
fidelity on committed endpoints. Log progress every five attempts and after
every sealed prompt, with remaining candidate-work estimate and available disk.

The motif development gate is quality at least 39.27%, uniqueness at least
96.83%, diversity at least 0.62, output at least 95% of attempts, and 100%
validity/fidelity among commits. The first three thresholds are the reported
IVG comparator row. A single seed meeting them is still only a development
signal, not a formal published-result win.

The decoration development gate, declared before looking at seed-one quality,
is at least +10 quality points over the sealed seed-zero joint-completion
18.5% (thus 28.5%), uniqueness at least 90%, diversity at least 0.56, output
at least 95%, and 100% validity/fidelity among commits. Reaching 28.5% does
not beat IVG's reported 36.37%; it is a promotion test for an independent
multi-seed evaluation, not a headline success.

All thresholds are held fixed on failure. No per-prompt tuning or sampler
modification is permitted within these runs. The comparison is not fully
paired because the policies consume different RNG sequences; it uses the same
prompts, checkpoint, official evaluator, attempt/candidate budgets and pinned
fresh seed. The negative outcome must be preserved if either gate fails.

## Provenance and interruptions

The self-hashed configs `configs/fragment_motif_focused_dev_v1.json` and
`configs/fragment_pendant_decoration_dev_v1.json` pin the source hashes,
software versions, evaluator bytes, gates, and output roots. Prepared manifests
pin the code revision. Receipts are atomic per attempt; an interrupted attempt
with a start receipt but no completed receipt fails closed, never silently
redraws. Each prompt gets an immutable sample lock and metrics row. Results
remain development evidence until the official three-seed protocol is separately
authorized and completed.
