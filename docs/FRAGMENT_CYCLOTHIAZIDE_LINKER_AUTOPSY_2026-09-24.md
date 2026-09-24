# Cyclothiazide linker development autopsy

This is a post-generation diagnosis of the sealed, single-seed 20-attempt
Cyclothiazide linker row. It is not a changed sampler, a new benchmark run, or
an optimized connector selection. The frozen pilot and all other prompt rows
remain untouched.

## Measured result and denominators

The locked row produced 20/20 connected, chemically valid outputs, all with
exactly preserved supplied cores and a nonempty path between the declared
interfaces. Official quality was 0/20, uniqueness 19/20 and diversity 0.4463.
On the 20 locked outputs, 2/20 met QED >= 0.6, 0/20 met SA <= 4, and 0/20 met
both. Mean QED was 0.4247; mean SA was 5.2055 (range 4.8922 to 5.6177).

The molecule grids color the thiazide-containing supplied core blue, the
bridged-ring supplied core green, and only the new path orange:

- [Locked attempts 0–9](../diagnostics/fragment_cyclothiazide_linker_counterfactual_v2/locked_outputs_00.png)
- [Locked attempts 10–19](../diagnostics/fragment_cyclothiazide_linker_counterfactual_v2/locked_outputs_10.png)
- [Fixed manual connector panel](../diagnostics/fragment_cyclothiazide_linker_counterfactual_v2/manual_connectors.png)

The sampled connectors are structurally varied, but they commonly add amides,
heteroatoms, rings or long chains between two already complex cores. The
outputs average 30.75 heavy atoms, molecular weight 486.4, topological polar
surface area 157.1 and 4.5 rotatable bonds. The original drug has 24 heavy
atoms, molecular weight 389.9, polar surface area 118.4 and 2 rotatable bonds.
These size/polarity changes are consistent with the QED drop, but this
retrospective comparison does not isolate their individual causal effects.

## Fixed connector edit, run through COMPOSE

Before scoring any hand edits, the diagnostic fixed six ordinary two-ended
connectors: one/two carbons, O, N, CO and CNC. Each was compiled by the existing
exact COMPOSE linker program, sanitized and checked for exact core/path fidelity.
All six compiled successfully. None passed the joint quality criterion. The
one-carbon connector gave QED 0.6624 and SA 5.1971; the two-carbon connector
gave QED 0.6473 and SA 5.1495. Four of six met the QED cutoff, but all six had
SA between 5.1495 and 5.3089. This demonstrates that merely shortening or
simplifying the added path does not rescue this prompt.

The original supplied drug is an informative **non-admissible control**: its
two supplied cores are directly bonded, with zero internal linker atoms. Its
QED is 0.6572 and SA is 5.0462. We did not count it as a generated linker or
as a quality pass. The original drug's high SA, and the preserved bridgehead
and stereochemical SA penalties in every completion, implicate the fixed
conditioning chemistry as a substantial contributor to this cell's difficulty.
SA is a whole-molecule, non-monotonic score; this evidence does **not** prove
that every possible core-preserving linker must fail SA.

## Decision

Do not add a Cyclothiazide-specific sampler rule or weaken the required cores.
Do not interpret this 0/20 cell as a chemical-validity or linker-path failure.
Await the complete ten-prompt aggregate and, if available, like-for-like
per-prompt GenMol/IVG outputs before deciding whether there is a generic
linker-quality defect. No raw comparator Cyclothiazide completion distribution
has been verified in this autopsy.

The reproducible [result JSON](../diagnostics/fragment_cyclothiazide_linker_counterfactual_v2/result.json)
contains every locked SMILES and score, exact prompt/core/path mapping, all
six manual candidates and refusals, RDKit SA component decomposition, input
SHA-256 hashes, code revision and library versions. The script is
`tools/diagnose_fragment_linker_counterfactuals.py`. Its four focused tests
passed in the pinned chemistry environment. The earlier v1 diagnostic artifact
is preserved; v2 only adds molecular descriptors to the same locked analysis.
