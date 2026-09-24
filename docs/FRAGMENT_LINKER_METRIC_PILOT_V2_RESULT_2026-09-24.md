# Linker/morphing development result after the perceived-ring repair

The bounded linker pilot completed all ten prompts, 20 attempts per prompt,
seed zero. It emitted 200/200 connected, chemically valid molecules. All 200
retained the supplied cores exactly and constructed a nonempty path through the
declared linker interfaces. The published-style macro metrics were validity
100.0%, uniqueness 91.5%, quality 35.0%, and diversity 0.54755. The complete
machine-readable result is
`diagnostics/fragment_training_linker_metric_pilot_v2/summary.json`; its manifest
SHA-256 is `4d68777c1a29f7ff7e562244e41a8401abb32b15aab95fee1efd1aaedf1ca65b`.

The source v1 run stopped after 127 complete attempts because RDKit perceived
ring counts are not additive across a bridged connector assembly. The v2
revision imported those 127 attempts and six sealed prompt rows byte-for-byte,
then replayed the interrupted Liothyronine attempt from its saved RNG state.
The disagreeing draw was recorded as a consumed, typed refusal with planned
cell (21 atoms, 4 perceived rings) and exact cell (21, 3). No v1 artifact was
overwritten, no candidate was silently relabelled, and the remaining 72
attempts were newly sampled under the narrow repaired source. Verification
rehashed all 220 v2 attempt/row/lock artifacts, all material manifest inputs,
and all 139 imported v1 files; every check passed.

The reported IVG linker/morphing row is quality 22.33%, uniqueness 84.76%,
diversity 0.52, validity 60.37%. The pinned GenMol V2 author README row is
quality 28.6%, uniqueness 87.1%, diversity 0.566, validity 81.8%. Thus this
development panel is above IVG on all four reported measures and above GenMol
V2 on quality, uniqueness, and validity, but below GenMol V2 on diversity.
These are cross-study reported comparators with different sample budgets. A
single 20-attempt-per-prompt seed is not a formal benchmark win or independent
replication. The scaffold-morphing prompts are identical to the linker prompts
in this benchmark formulation; the same outputs may be reported as an alias,
not as a second independent result.

Negative cells remain prominent: Cyclothiazide and Lovastatin each scored 0%
quality, and Maribavir 5%. The Cyclothiazide fixed-core autopsy shows simple
exact linkers also exceed the SA threshold, but that does not prove the prompt
is unsatisfiable. These cells should be compared with per-prompt comparator
outputs if obtainable. No prompt-specific chemistry or test-time QED/SA
guidance was used.

Next decision: freeze this linker law and obtain authorization for a full
official sample/seed evaluation with the identical source and evaluator. Do
not tune the completed pilot to its weak individual prompts. The separate
motif/decoration joint-completion pilot remains a distinct measurement.
