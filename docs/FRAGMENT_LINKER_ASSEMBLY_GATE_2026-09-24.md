# Bridge-free linker assembly: correctness gate

## Scope and claim

This is the separately authorized linker/morphing correctness work, conducted
while the motif/decoration matched pilot remains frozen. It does not change
that pilot, superstructure, a learned checkpoint, T4, PMO, or any live campaign.
No generation benchmark or new model training is authorized by this gate.

The output is a complete connected molecule containing the two supplied cores,
connected through their declared external attachment sites. The hypothesis to
test is that the existing dependency-aware fragment program compiler can assemble
a proposed two-ended region and the second core without starting from an
artificial one-carbon bridge. Use existing compiler and executor machinery,
not a separate chemistry implementation or beam search.

## Primary-source semantics

GenMol revision `add09fc83b7255bd09c797e527c0f4b51f5fb7c1`,
`scripts/exps/frag/run.py`, uses the released `linker_design` column for both
linker variants and associates it with scaffold morphing. Its sampler encodes
the two supplied fragments together for the one-step variant. No carbon bridge
is present in the input. `src/genmol/utils/utils_chem.py` removes attachment
dummies for its substructure filter; that filter is not a path-length check.

The locally hash-pinned IVG `in_virtuo_gen/evaluation/downstream.py` at revision
`b50bb3ae2bdcb9df581f0b219d79cf14b05d0fbb` renames the supplied scaffold-morphing
column to linker, concatenates the fragments as conditioning, and copies linker
results to morphing. It does not supply a bridge or independently measure linker
length. Match all ten static input pairs after erasing representation-only
dummy numbering and stereochemistry, and record mismatches rather than assuming
equivalence. Do not run upstream code or change the upstream cache.

The published inputs do not establish an explicit minimum linker length. A
nontrivial connector and absence of shortcuts are COMPOSE's separately reported
stronger correctness choice, not a newly attributed official metric. The official
validity/uniqueness/quality/diversity denominators remain unchanged.

Another reporting distinction: the current local `distance` helper compares
ECFP4-2048 to the capped prompt. GenMol's published distance instead uses
ECFP4-1024 to the original drug. Do not label those interchangeable. Keep the
ongoing pilot unchanged; compute a separately named GenMol-compatible distance
from frozen outputs for a future official comparison.

## Representation and support

Disconnected cores are conditioning data, not an executable MolecularGraph.
The runtime requires connected-or-null molecular states. Initialize execution
with the larger H-capped core, construct the proposed connector and exact second
core in dependent blocks, and expose only the completed program as an output.
The initial core is locked pathwise. The second core is immutable conditioning
and its exact mapped graph must be realized at completion; it cannot be claimed
present throughout intermediate assembly states. Preserve this distinction.

Reuse the existing source/target slot mapper, group compiler, program graph
scheduler, executor and strict RegionLock. Proposed connector content may be
acyclic, branched, aromatic, heterocyclic or fused, within existing executor
support. No hand-picked benchmark motif, property guidance, or hidden reference
structure is permitted. Maximum support remains 40 active atoms, 48 slots,
32 primitives and eight blocks, including construction of the supplied second
core. Unsupported connectors are explicit refusals, never truncated content.

## Frozen correctness tests, before any quality work

- Exact two-, three- and four-atom linear connector fixtures.
- Ring-containing connector fixtures using the same program compiler.
- Every executed state is chemically valid and connected.
- Exact mapped core identities, internal bonds and aromaticity are preserved;
  external contacts occur only at the two declared sites.
- Both sites connect through a nonempty new region, with no direct core-core
  shortcut or additional external contacts.
- Negative controls: direct bond, wrong attachment, missing/overlapping core,
  disconnected endpoint, core mutation, malformed connector and budget overflow.
- Changing drug name/reference cannot change the constructed program.
- Linker and morphing produce identical programs for identical input pairs.
- A bounded all-ten-prompt no-score audit records every result/refusal and exact
  primitive trace. No QED/SA, comparison metric or learned-model selection.

This gate establishes semantics and constructive support only. It is not a
claim of learned linker generation, competitive quality, stereo preservation
beyond the existing representation, or an official linker/morphing row.

## Stochastic training-connector integration

The subsequent user-authorized parallel development step prepares a stochastic
connector lane through this same bridge-free compiler. Consume only the 55,502
two-boundary entries in the immutable split-first region catalog with SHA-256
`a8775b466f479c2594ff5d8afd81049a57969211d3152035c93e9db51e8b1347`.
Reuse the whole-training-molecule joint prior from
`docs/FRAGMENT_JOINT_COMPLETION_PRIOR_DEV_2026-09-24.md`; do not fit a second
linker-specific statistic or extract additional regions.

Enumerate both boundary orientations against the supplied cores' generic atom
contexts. Group context/atom-capacity-compatible entries by the complete endpoint
heavy-atom and RDKit ring counts. Sample a cell using the shared training histogram
plus its frozen total pseudocount of one, then sample content using square-root
occurrence weights within that cell. Choose uniformly between compatible boundary
orientations. An entry receives its content weight once even when both orientations
match. The two cores each have exactly one contact, so their rings and the
connector's rings add without a new cross-core ring. Verify the final counts after
exact execution. These are structural reachability counts; they do not assert
primitive-budget, compiler, or learned-model support.

`fragment_linker_sampler.sample_linker_panel` offers exactly eight complete
candidate draws and retains every refusal. It uses the shared
`learned_program_scores` mean native log-mark score and unit-temperature
`select_learned_program` selection after canonical endpoint deduplication.
Non-finite scores produce explicit model abstentions. A panel with no supported
candidate produces no output. There is no beam, QED/SA selection, deterministic
fallback connector, drug-specific rule, checkpoint change, or scored campaign.

The lane records the catalog identity, sampled structural cell, observed source
rows, bound orientation, exact complete trace, compilation refusals and model
support separately. The initially present core is locked throughout execution;
the second supplied core remains conditioning until exact construction completes.
An additional explicit mapped-core check prevents a different incidental
substructure match from substituting for the intended core identity.

Unit tests establish the sampling law, exact chemistry, preservation, shared
scorer wiring and no-output behavior. They do not establish finite support under
the real checkpoint. A separately recorded bounded diagnostic may check that
boundary on at most one additional CPU worker. No official linker/morphing metric
row or large pilot is authorized by this integration preparation.

## Bounded learned-panel support gate

After the one-draw diagnostic produced nine exact programs and two finite native
model examples, prepare one support-only census using the unchanged stochastic
sampler: all ten linker input pairs, two output attempts per pair, exactly eight
candidate draws per attempt, seed zero, one CPU worker/thread, zero oracle calls
and no QED/SA or official metrics. Morphing uses the identical input-pair program
law and is not counted as another independent support cohort.

Freeze the gate before running: at least 18 outputs among all 20 attempted
outputs, at least one output for every prompt, and 100% chemical connectedness
and exact core/interface/path fidelity among selected outputs. Preserve a failed
gate without changing its threshold, support or selected connector content.
Candidate compilation/model refusals consume their offered candidate slots;
they are not removed from work counts or treated as new output attempts.

The runner is `tools/run_fragment_training_linker_support.py`. Its `prepare`
phase writes the input/configuration/gate manifest without loading the
checkpoint. Its `run` phase verifies that manifest before scoring, stores all
eight per-candidate receipts inside each atomic attempt result, and reduces only
the declared 20 named attempts. The durable recovery unit is one completed
eight-draw attempt. An incomplete started attempt fails closed on resume; the
runner does not silently regenerate an interrupted panel. Preserve the
one-draw diagnostic and ongoing motif/decoration pilot. Wait for the shared CPU
resource slot before starting this gate. Passing establishes bounded panel
support, not benchmark quality or authority for a larger scored launch.
