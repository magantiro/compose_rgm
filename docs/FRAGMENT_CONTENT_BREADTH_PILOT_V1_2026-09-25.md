# Matched scaffold-decoration content-breadth pilot

The current ten-prompt, 200-attempt joint-mass development run returned quality
37.0%, uniqueness 97.0%, diversity 0.5565473529, and validity 100.0%. Its
prospectively frozen 0.56 diversity gate failed, so it was not promoted to a
full multi-seed benchmark. The optional training-content sampler increases
distinct sampled content plans without using benchmark quality labels. A
30-attempt-per-arm execution smoke produced 26 valid exact completions with the
frozen content law and 25 with uniform within-cell content, with no invalid or
nonfaithful committed molecules. Neither observation establishes a benchmark
gain.

The next bounded comparison is `configs/fragment_content_breadth_pilot_v1.json`,
payload SHA-256
`7e0b76650289280ed763ca81e148bd9676aab465794b72d298f2c82465070526`.
It fixes all ten decoration prompts, fresh seed six, 20 attempted outputs per
prompt and arm, eight complete offers per attempt, one CPU worker, and the same
RingCore checkpoint, exact executor, joint total-mass law, learned selector,
prompt checks, and official evaluator. The only intervention makes the
training-supported pendant content uniform within the already chosen
attachment-context/size/ring cell. Both arms retain the same structural
support. Every attempt and model score is recorded, including no-output
attempts. Samples are locked before quality, uniqueness, diversity, or
validity are computed.

The promotion decision is frozen before scoring. Uniform content must reach
diversity at least 0.56 and exceed its matched frozen arm by at least 0.01,
lose no more than two quality points, retain at least 90% uniqueness, and
produce at least 190 of 200 outputs per arm. Every emitted molecule must be
chemically valid and satisfy the exact prompt constraints. Missing outputs
remain in the official attempt denominator. A failed criterion remains a
negative result. A passing single-seed development comparison would authorize
consideration of a separate fresh-seed full benchmark; it is not itself that
benchmark.

The runner is `tools/run_fragment_content_breadth_pilot_v1.py`. `prepare`
verifies physical hashes, software versions, the source lineage, and evaluator
identity before publishing a manifest. `run` requires a separate
`authorization.json` containing exactly `{"approved": true,
"payload_sha256": "7e0b76650289280ed763ca81e148bd9676aab465794b72d298f2c82465070526"}`.
An interrupted started attempt cannot be redrawn. Completed attempts replay
their recorded RNG state on resume. The current source worktree reads the
already frozen split-first training catalog and mass prior by absolute,
hash-verified path, and its evaluator cache is a local copy of the
hash-verified upstream bytes.

This is a RingCore fragment-development comparison. The distinct Editing-V2
checkpoint used by the deployed QED controller remains a separate unresolved
paper-lineage decision. This pilot does not resolve that mismatch or touch the
paused learned-reference and fixed-size-QED ablations.
