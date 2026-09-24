# Decoration quality: fixed scaffold, added structure, and counterfactuals

## Scope

User-requested inspection of actual molecules and manual QED/SA counterfactuals.
No sampler, checkpoint, training data, active campaign, or benchmark output changed.
These are post-generation development diagnostics on the already saved twenty
Eliglustat decoration attempts per arm, not new generator results or promotion.
The platform claim remains exact executable molecular editing; this audit tests
whether a particular completion proposal distribution over-decorates its input.

The supplied fragment has 18 heavy atoms and two attachment interfaces. Its fused
two-oxygen ring and pyrrolidine are required input, not repeatedly invented output.
The highlighted structures and source hashes are in
`diagnostics/fragment_prompt_context_eliglustat_v1/receipt.json`.

## Computed score decomposition

The full 20/20 outputs per arm are decomposed with the installed official RDKit
QED implementation, reconstructing QED to 1e-12. New-minus-baseline mean weighted
log-QED terms: molecular weight -0.322710, rotatable bonds -0.159047, aromatic ring
count -0.096729, lipophilicity -0.060945. The structural-alert contribution improves
by +0.129331. The remaining negative contributions are smaller.

This attributes a score difference mathematically, not a causal effect of an
individual molecular feature. The new proposals combine larger additions with
different chemistry. Do not infer that deleting rings universally improves QED.
All descriptors, input/code hashes, and first-pass/first-fail illustration choices
are recorded in `diagnostics/fragment_eliglustat_qed_decomposition_v1/result.json`.

## Manual counterfactual, not inference policy

Before scoring edited molecules, fix one diagnostic operation: replace the largest
single-boundary non-core pendant with methyl; preserve all supplied core atoms,
internal bonds, and declared attachment occupancy. Ties use the prompt atom order.
Also inspect the degenerate control replacing all such pendants by methyl. No
choice uses QED/SA, and every saved attempt is retained in accounting.

| Saved/edited set | Completed | Joint QED/SA pass | Distinct | Mean QED | Mean SA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original new adapter | 20/20 | 1/20 | 20 | 0.382771 | 4.184903 |
| Largest pendant to methyl | 20/20 | 18/20 | 20 | 0.785518 | 3.378822 |
| All pendants to methyl | 20/20 | 20/20 | 1 | 0.841357 | 2.991047 |

For attempt zero, the first edit changes 36 to 26 heavy atoms, QED 0.501735 to
0.803586, and SA 4.323925 to 3.728727. All edited endpoints pass connectedness,
mapped core atom/bond checks, and the existing attachment-condition checker.
Stereochemical preservation is not claimed beyond the current model support.

The all-methyl control collapses to one molecule. Its 20 raw passes would supply
only one unique passing output over twenty attempts under the official quality
denominator. It is not a solution to the quality/diversity objective. Neither edit
is a COMPOSE-executed or prospective generative result. The diagnostic changes
size and content together and does not establish a size-only causal effect.
Artifacts: `diagnostics/fragment_manual_decoration_counterfactual_v1/result.json`.

## Public comparator availability

Read-only public-tree audit at IVG b50bb3ae2bdcb9df581f0b219d79cf14b05d0fbb
and GenMol add09fc83b7255bd09c797e527c0f4b51f5fb7c1:

- IVG commits four fragment sample grids and aggregate result files. Its published
  decoration PNG illustrates Cyclothiazide, not Eliglustat. The plotting function
  samples valid outputs rather than ranking by quality. The image is not evidence
  that each displayed molecule passes QED/SA.
- The checked IVG big-model fragment JSON has empty `smiles` arrays for every task.
  No raw per-prompt completion set was found in either checked repository tree.
- GenMol commits fragment generation code, prompts, and settings; the checked
  tree does not contain the corresponding generated fragment sample files.
- This is not an exhaustive search of releases or external archives. We cannot
  report comparator per-prompt SA distributions from images.

Downloads retain upstream licenses and Git-blob/SHA-256 verification in
`diagnostics/fragment_comparator_outputs_audit_v1/receipt.json`. Comparator outputs
are diagnostic references only, not training fragments or proposal content.

## Interpretation and next boundary

The adapter executes coordinated programs, but its content law samples regions
sequentially using coarse attachment context, square-root occurrence weights,
and remaining-capacity feasibility. Its learned panel score is the mean local
mark log probability, not a validated whole-completion likelihood or stopping law.
These implementation facts and the counterfactual support whole-molecule
completion conditioning as the next hypothesis. They do not justify score-guided
postprocessing, methyl templates, removal of ring support, or per-drug settings.
The current frozen matched pilot remains the required full-panel comparison.

Verification: six focused diagnostic tests passed, including QED reconstruction,
one/two-interface preservation, and refusal to cut a two-boundary cyclic region
as a pendant. Ruff and `git diff --check` passed for this diagnostic work. The
repository-wide suite is not claimed to pass; no sampler milestone is complete.
