# Fragment interface-release pilot (zero oracle)

## Scope and scientific identity

This is a bounded development ablation of COMPOSE's learned, fragment-conditioned
graph-edit sampler. The generated object is a complete connected molecular graph;
the benchmark supplies retained core atoms/bonds and, for motif extension and
scaffold decoration, open attachment interfaces. The central test is whether
permanently forbidding growth from all undeclared core atoms *after* the required
sites have been covered unnecessarily reduces molecular quality and diversity.

This branch starts at `fragment-attachment-panel-20260921` commit
`21e113f65e233db9c339f6902b8dccfa3f3fb278`. The checkpoint is
`/Users/rmaganti/compose_fragment_ckpt/ringcore_a7546e2_best.pt`, SHA-256
`24117dfeaee91729bb4ebccb5eb5b993b4a6218605045e51917823d086b4c1e4`.
The ten-drug prompt file is
`data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv`, SHA-256
`a4fb8357d0f1102cbdc8d79d802e15f66a59a9722c0b7125ce693fe7a29872a9`.
The official InVirtuoGen evaluator revision is
`b50bb3ae2bdcb9df581f0b219d79cf14b05d0fbb`; its metrics module SHA-256 is
`3c4bb7c6727cbeaf02d3d5eebf1deac27f77bab68d929b61dbe4154911e2b099`.
The full upstream blob list is pinned in `tools/fetch_official_fragment_evaluator.py`.
Run this pilot with `/Users/rmaganti/compose_fragment_pinned_env/bin/python`,
the existing local analogue of the historical Modal chemistry kernel:
Python 3.11, RDKit 2024.03.5, NumPy 1.26.4, Torch 2.4.0, and pandas 2.2.3.
The repository root `.venv` has a different RDKit/NumPy/Torch stack and lacks
pandas, so it cannot reproduce the fragment sweep and must not substitute for
the pinned fragment environment.

The earlier matched three-seed/100-attempt comparison of attachment control ON
versus the frozen sampler OFF found much higher declared-site task success, but
lower committed-endpoint diversity and, for decoration, lower quality. Those
results are development evidence, not a selected final benchmark. This pilot
changes exactly one already existing controller field: `restrict_interfaces`.
Both arms keep `enabled=True`, `attachment_first=True`,
`redirect_attachment=True`, the same checkpoint, prompt, seed, and sampler
settings. In the release arm only, `restrict_interfaces=False` permits a legal
new bond from an undeclared retained-core atom *once required-site coverage has
been achieved*. Before coverage, attachment-first still requires every accepted
event to strictly reduce the unsatisfied interface set. No chemistry rule,
executor, region lock, oracle, model weight, or benchmark target changes.

## Frozen small-pilot protocol

This is **not** an official benchmark row. Run 20 attempts for seed 0 at each of
four alphabetically dispersed prompts (BARICITINIB, ERLOTINIB, LIOTHYRONINE,
MARIBAVIR) in each of motif extension and scaffold decoration, plus two
superstructure controls (BARICITINIB, LOVASTATIN). There are 10 prompt/task
pairs, 200 attempts per arm, 400 attempted trajectories total. The two arms
start from independently constructed but identically seeded RNG streams for
each prompt. Keep the historical sampler defaults: 32 events, operational
horizon 16, 24 mark attempts per event. The public `--allow-post-coverage-core-growth`
flag is the sole A/B switch and is invalid unless `--attachment-control` is on.

Persist every attempt's canonical committed endpoint (or null), emitted
constraint-satisfying endpoint (or null), seed, prompt, event count, and refusal
counter deltas. Bind both arms to input, code, checkpoint, and evaluator hashes;
record software versions and all exclusions. The full committed-endpoint lists
are the denominator for chemical validity, uniqueness, quality, and diversity;
attempts are the denominator for output, interface coverage, containment, and
full task success. Separately report official-formula metrics on emitted samples
with the explicitly nonofficial 20-attempt denominator. A metric computed on
emitted molecules must never be described as committed-endpoint quality.

## Hard gates and falsifiers

1. Every committed endpoint must parse, sanitize, and be connected; exact
   retained-graph lock preservation is asserted in focused tests and by the
   unchanged production `RegionLock` on every accepted event.
2. Before coverage, accepted edits still improve interface coverage. After
   coverage, only the release arm permits otherwise legal non-interface growth.
3. Superstructure declares no attachment interface, so the two arms must be
   byte-identical in attempt endpoints and counters under matched seeds.
4. Default CLI behavior and its canonical configuration hash remain unchanged.
5. The release arm is rejected as a full-benchmark candidate if this pilot
   loses more than 10 percentage points in motif or decoration attempt-level
   full task success, or if neither committed quality nor committed diversity
   improves. A positive small pilot is only qualification for a separately
   frozen full matched comparison; it is not evidence of IVG/GenMol superiority.
6. If checkpoint/evaluator identities cannot be verified, abstain rather than
   substitute a model or metric. No PMO, T4, live job, training, or oracle call
   is in scope.

The eventual comparison must report attempt output, committed chemical
validity, exact locked-graph preservation, RDKit fragment containment, correct
attachment, full task success, uniqueness, quality, diversity, and distance
with explicit denominators. This pilot may identify a tradeoff; it cannot
resolve a full-suite statistical or paper claim.
