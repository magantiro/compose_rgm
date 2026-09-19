# Fragment-constrained COMPOSE proposal smoke v1

This is a computed, CPU-only, zero-oracle support check. It loaded the frozen
50-prompt SAFE/GenMol manifest and, before executing any proposal, selected the
first manifest prompt for each of the five task labels. All five selected
prompts are the Baricitinib rows because the manifest is drug-major.

All five proposals completed under the unchanged 40-active-atom,
32-primitive, eight-block limits. Every primitive ran through the production
Editing-V2 executor; every committed state was valid and connected; every final
candidate passed the exact fragment constraint; and each retained fragment
remained present after its recorded construction/attachment lock. Oracle and
scoring calls were both zero. The machine-readable result is `result.json`
(SHA-256 `eeff3e28fd878fe5bcd4c5aa2cfbf142e847d66d38cd502fa8f1eab5647732a2`).

This establishes only that the bounded generic proposal mechanism can exercise
all five labels on the frozen smoke selection. It is not a 100-sample benchmark
run, a learned-controller result, an optimization result, or evidence of
quality, uniqueness, diversity, or comparator performance. Linker design and
scaffold morphing intentionally share the released prompt, so their deterministic
variant-0 endpoints are identical. The current proposal family uses only
generic carbon extensions and may abstain on the fixed support/budget gates.
