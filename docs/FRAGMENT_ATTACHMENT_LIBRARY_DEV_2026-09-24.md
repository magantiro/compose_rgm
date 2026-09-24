# Training-derived attachment-fragment development gate

Status: **NO PROMOTION**, completed negative development pilot. This document
authorizes no scored full benchmark or change to the frozen superstructure run.
At 200 attempts per task/arm, motif quality rose 19.5% to 31.5%, but diversity
fell 0.598 to 0.494. Decoration quality fell 31.0% to 16.0%; output fell 100% to
98%. See `diagnostics/fragment_attachment_library_pilot_v1/analysis.json` and the
immutable shards. These results fail the predeclared gate.

Provenance correction, 2026-09-24: the source is a historical GuacaMol training
file, but RingCore further partitions it by scaffold. This catalog excluded the
benchmark drug identities but did not apply that internal partition before
extracting fragments. It must not be called a RingCore split-clean prior or used
for promotion. Original artifacts are preserved; the source paragraph below
records the original plan, not a verified internal-split guarantee.

## Identity and support

- Problem: motif-extension and scaffold-decoration prompts specify where a valid completion must attach, but the current atom-level attachment-controlled sampler has low motif quality/uniqueness and lower decoration quality than the final InVirtuoGen reference.
- Primary output: complete, connected, chemically valid COMPOSE molecular graphs retaining the supplied locked fragment and decorating only its declared attachment interfaces.
- Claim under test: a single task-blind proposal lane that grafts observed multi-atom training substituents through the exact rewrite executor improves quality and output without collapsing uniqueness/diversity or prompt fidelity.
- Setting: the released ten-drug motif and decoration prompts, a fixed model/checkpoint and evaluator, and one matched small development panel before any full benchmark. Superstructure is frozen. Two-ended linker/morphing prompts are outside this first gate until their benchmark semantics are resolved.
- Baselines: the saved attachment-controlled 3,000-attempt rows and a same-prompt, same-seed matched small control. Causal ablation changes only the constructive proposal lane; the official QED/SA evaluator never enters proposal selection.
- Declared proposal support: neutral, non-stereochemical, acyclic C/N/O/F substituents of at most six heavy atoms, attached by one single bond at a benchmark-declared interface. This is a *proposal lane*, not a reduction of the base COMPOSE graph/executor support. Coverage and excluded source bonds must be reported. Aromatic/ring-containing and two-ended fragments are explicitly not claimed by this first pilot.

## Frozen source and leakage boundary

Use only the historical GuacaMol **training** subset at `/Users/rmaganti/compose_denovo_artifacts/guacamol/guacamol_subset_500000_seed0.smiles`, physical SHA-256 `70526d92f1f08d8e292cb31218f81b6924a2182f772c43348015110669d47791`. The held-out validation file and the ten benchmark reference drugs must not supply fragments, frequencies or attachment statistics. Record source row counts, parse failures, duplicate canonical identities, exclusion reasons, local software versions, code revision and catalog hash. Fail if the source hash differs; do not silently substitute another corpus.

Extract both orientations of eligible BRICS bridge cuts. Canonicalize a rooted fragment with one dummy attachment and retain its proximal atom-context class (element, aromaticity, ring membership). Count occurrences in the training source; do not use QED, synthetic accessibility, a benchmark target SMILES or prompt identity to select entries. Preserve the full counted library and publish any bounded runtime shortlist and its coverage separately. Flatten training frequency by a predeclared square-root weighting so dominant fragments do not collapse the proposal law.

## Mechanism and falsifier

For each declared site, select one context-compatible observed rooted fragment, compile its atoms as a contiguous exact `atom_insert` program, and privately execute every primitive. Admit only if every intermediate state passes the core lock and the final molecule satisfies the benchmark constraint. If a fragment is unavailable or cannot execute, try one explicit minimal compatible carbon completion through the same executor. Every failure and fallback is an attempted output, not an uncharged retry; record offered fragment, actions, reason and endpoint. Do not post-hoc filter molecules by QED/SA or resample until quality passes.

Pre-qualification: all emitted endpoints must be connected and chemically valid, all core atoms/bonds preserved, and all attachments must be at declared sites; record output rate, fragment-library coverage, fallback rate, fragment diversity and per-attempt provenance. A 100% attempt-level validity claim requires a full all-prompt measurement, not the executor invariant alone.

Run one matched development panel on all ten motif and all ten decoration prompts, initially 20 attempts per prompt at one frozen seed. Compare official validity, uniqueness, quality and diversity plus prompt fidelity and exact proposal work. Do not promote unless motif quality rises by at least ten percentage points and decoration by at least five, neither loses more than 0.03 diversity or 0.05 uniqueness, and validity/fidelity do not regress. These are directional development gates, not a claim of statistical significance or a paper benchmark. A negative result stops this library mechanism; do not sweep fragment sizes, frequency powers or mixing weights after seeing it.

Linker/morphing is a separate correctness gate: first reproduce the published prompt semantics and remove the harness-created bridge if it is not part of the released input. Only then consider the same training-derived representation with two declared ends. Do not copy a linker result into morphing until the input and procedure are verified identical under the pinned benchmark version.
