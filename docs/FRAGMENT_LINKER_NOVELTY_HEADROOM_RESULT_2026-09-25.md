# Linker novelty headroom on the locked development panels

We examined the 200 saved, eight-offer attempts from the repaired seed-zero
linker development run without generating or scoring a new molecule. The
original selector emitted 183 distinct endpoints. All 17 repeated selections
occurred in panels that also contained at least one as-yet-unemitted,
model-supported endpoint. The missed opportunities were concentrated in
Lesinurad (five), Maribavir (six), Lovastatin (three), Eliglustat (two) and
Cyclothiazide (one); the other five prompts had no repeated selection at this
20-attempt scale.

Conditional on the *observed original archive* at each attempt, the existing
score softmax assigned a summed probability of 180.79 to unseen endpoints.
Multiplying unseen weights by four changes that conditional sum to 193.09.
The difference measures one-step selection headroom only. An actual novelty4
run would alter its archive and RNG chain, and could change quality or
diversity. These numbers must not be reported as counterfactual benchmark
metrics or as evidence that the predeclared pilot gate passes.

The pinned evaluator counts quality-passing *distinct* valid molecules and
divides by the fixed attempt count. Returning a molecule already emitted in
the same prompt cannot add to either uniqueness or quality. Selecting an
unseen, valid endpoint could add to quality if it passes the existing
drug-likeness and synthesizability criterion; its effect on diversity remains
empirical. This explains why a selection-level intervention can plausibly
improve both headline metrics without using either property as guidance.

The machine-readable result is
`diagnostics/fragment_linker_novelty_headroom_v1/result.json` (SHA-256
`b319163bbc9ea2debfba89b2d086a15005aabf8c25af721d0909b4eb61e62735`).
It records all 200 attempt hashes, the analysis revision, software version,
configuration, every conditional probability and all exclusions (none).
No QED, SA, oracle or docking evaluation was performed. The separate
cell-allocation pilot tests proposal breadth; this result isolates a
selection-level source of repeat outputs.
