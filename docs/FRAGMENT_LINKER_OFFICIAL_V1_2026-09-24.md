# Linker and scaffold-morphing published-scale evaluation

## Scientific identity and scope

The problem is fragment-constrained molecular generation from two supplied
cores. The generated object is one exact COMPOSE dependency-aware program and
its committed connected molecular-graph endpoint. The claim tested is whether
the frozen training-derived two-boundary proposal law can sample chemically
valid, core-preserving, nonempty-linker completions with published-scale
validity, uniqueness, quality and diversity competitive with the pinned
InVirtuoGen and GenMol comparisons. This is not a docking, PMO, T4 or de novo
claim. The comparator is the published fragment-generation row; the causal
development comparator is the 200-attempt seed-zero linker pilot, which is not
included in this evaluation.

The declared support remains the existing COMPOSE 40-active-atom, 48-slot,
32-primitive and eight-block executor, broad-organic element vocabulary and
non-stereochemical graph representation. Inference draws only from the frozen
split-first training connector catalog and training-only joint size/ring prior.
The frozen model scores eight complete candidates at each attempt; it never
uses QED, synthetic accessibility, a drug identity, a reference endpoint or a
test-time quality filter to select one. Absent model-supported completions
remain no-output attempts in the official denominator.

## Frozen evaluation decision

User authorization on 2026-09-24: run the full linker benchmark after the
200-attempt seed-zero development panel completed. The self-hashed contract is
`configs/fragment_linker_official_v1.json`. Evaluate ten pinned prompts, 100
attempts per prompt, on fresh seeds 1, 2 and 3, for 3,000 attempted outputs
and 24,000 offered candidate draws. No development prefix is reused. Compute
official InVirtuoGen validity, uniqueness, quality and diversity from each
sealed 100-attempt prompt population, including no-output placeholders. Average
prompt metrics within seed, then average the three seed means. Also report
output rate, chemically valid connected outputs, and exact mapped-core/path
fidelity separately. The exact linker and morphing input pairs are verified
identical; morphing is an alias of the same generated samples, not a second
independent replicate.

`tools/run_fragment_linker_official_v1.py prepare` must verify every pinned
physical dependency, evaluator and software version and create an immutable
manifest before `run`. The run writes each complete eight-draw attempt
atomically, including all candidate receipts, seed and RNG state. Completed
attempts replay their RNG chain on resume. An interrupted started attempt fails
closed and requires an explicitly audited exact-state recovery; no automatic
redraw, replacement or hidden retry is allowed. It locks all 100 samples before
scoring a prompt. A durable progress receipt is updated every ten attempts,
with output census, model-work ETA and free disk. The process stops before
another attempt when free disk falls below the frozen five-GiB floor. The
three-seed result is published only after every named attempt and lock is
present. No unrun cell will be imputed from the seed-zero pilot.

The 200-attempt developmental metrics are in
`docs/FRAGMENT_LINKER_METRIC_PILOT_V2_RESULT_2026-09-24.md`. They motivated
the user decision to evaluate the already-frozen sampler, not a post-hoc
change to its law. The full run is a separate measurement. Any comparison
remains limited by differences in model training corpora, sampling compute and
the representation's absent stereochemical guarantee. The currently pinned
official evaluator reports no GenMol-compatible distance; distance is not
silently substituted from a different fingerprint/reference definition.
