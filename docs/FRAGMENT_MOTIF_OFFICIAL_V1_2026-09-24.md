# Fresh-seed published-scale motif extension evaluation

The scientific question is whether the frozen focused, dependency-aware COMPOSE
motif-extension sampler produces chemically valid, prompt-faithful, diverse and
high-quality completions at the published fragment benchmark scale. The output
is a complete exact-executor program and a connected molecular endpoint, or an
explicit no-output attempt. This evaluates the declared 40-active-atom,
48-slot, 32-primitive, eight-block, non-stereochemical COMPOSE support; it is
not a claim of universal molecule generation.

The ten released GenMol/IVG motif prompts are fixed. The seed-one, 20-attempt
development result is sealed and excluded. The proposed formal run uses seeds
2, 3 and 4, 100 attempts per prompt per seed, eight complete program offers
per attempt and 24,000 offered candidates. The same training-only region
catalog, joint completion prior, COMPOSE checkpoint, exact executor and
native learned program score used by the frozen development pilot remain
unchanged. No QED, SA, drug identity, original full-drug structure or
benchmark reference enters candidate selection.

The official evaluator scores each immutable 100-attempt prompt population,
including no-output placeholders. Quality, validity, uniqueness and diversity
are reduced as a ten-prompt mean within each seed, then a three-seed mean and
population standard deviation. Program/action provenance, selected endpoint,
all offered candidates, RNG states, failure reasons and exact prompt-fidelity
checks are retained for every attempt. Chemical validity of committed outputs,
constraint fidelity and attempt-level output rate are reported separately.
An interrupted started attempt fails closed. Saved complete attempts replay
their exact RNG chain on resume; no candidate is redrawn or replaced. The
runner stops before a new attempt below five GiB free disk.

The self-hashed contract is `configs/fragment_motif_official_v1.json` and the
runner is `tools/run_fragment_motif_official_v1.py`. Preparation creates only
the manifest and performs no sampling or scoring. The 3,000-attempt `run`
phase requires the exact payload-specific scored-evaluation authorization
before launch. No threshold is retrofitted to declare a win; published IVG
and GenMol values are read-only comparators. This run is separate from the
live linker and decoration runs and does not mutate their artifacts.

Selection of this development-passed sampler used public prompt performance,
so the three-seed result is a matched published-scale estimate of the frozen
method, not a novel held-drug or held-scaffold generalization test. The
checkpoint's lack of stereochemistry remains a support limitation.
