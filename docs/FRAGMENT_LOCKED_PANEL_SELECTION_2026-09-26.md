# Locked fragment-panel selection comparisons

## Result binding

The comparisons below use the finished fragment configurations and their original attempted outputs. No molecular proposal was generated for this analysis. Every new replay verified the self-hashed run contract, the run manifest, all 30 prompt-seed row hashes, all 3,000 individual attempt hashes, every deployed selected endpoint, and the deployed selected probability recomputed from the recorded native scores. The official evaluator reproduced each locked headline summary before any counterfactual was reported.

| Task | Locked run | Seeds | Attempts | Full panel | Selection comparison |
| --- | --- | --- | ---: | ---: | --- |
| Motif extension | `fragment_motif_official_v1` | 2, 3, 4 | 3,000 | 8 attempted offers | Existing completed reference-score versus uniform replay, reused |
| Scaffold decoration | `fragment_decoration_official_v2`, frozen arm | 8, 9, 10 | 3,000 | 8 attempted offers | New reference-score versus uniform replay |
| Linker design | `fragment_linker_novelty_official_v2` | 6, 7, 8 | 3,000 | 8 attempted offers | New deployed reference-score plus novelty-4 versus uniform replay |
| Scaffold morphing | Identical linker outputs | 6, 7, 8 | 0 additional | Identical linker panels | No independent experiment or additional replication unit |
| Superstructure | Existing completed primitive-sampling comparison | 0, 1, 2 | 3,000 per arm | Not a finite endpoint panel | Separate construction-stage result, not rerun |

The old linker common-panel analysis used `fragment_linker_official_v1` and is retained as version-specific supporting evidence. Its frozen contract hash is `558aef979d5093b5ff5678087fcde292ab4fd3b795e1e218b2474f62c3c2d2fb`, which differs from the current novelty-v2 contract hash `08471bb5b91707bed5cd90922a110c1478cab2fba538b8e9cb814f6301992e97`. We did not attach the old replay to the current linker row.

## Full-panel results

All means below average the 30 prompt-seed metric rows. Quality, uniqueness, validity, and prompt fidelity are percentages. Quality is the published uniqueness-adjusted benchmark metric. Faithful quality-pass yield is the percentage of all attempted output slots that contain a prompt-faithful molecule passing the evaluator's existing per-molecule quality criterion. These two quality measures are not interchangeable.

| Task and selector | Quality | Uniqueness | Diversity | Validity | Prompt fidelity | Faithful quality-pass yield |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Motif, deployed reference score | 42.63 | 93.63 | 0.6640 | 99.97 | 99.97 | 45.47 |
| Motif, uniform | 37.20 | 93.53 | 0.6756 | 99.97 | 99.97 | 39.43 |
| Decoration, deployed reference score | 36.70 | 93.67 | 0.5626 | 100.00 | 100.00 | 42.37 |
| Decoration, uniform | 34.47 | 95.00 | 0.5585 | 100.00 | 100.00 | 38.40 |
| Linker, deployed reference score plus novelty-4 | 31.53 | 81.03 | 0.5636 | 100.00 | 100.00 | 36.90 |
| Linker, uniform | 24.00 | 73.47 | 0.5744 | 100.00 | 100.00 | 30.67 |

The prompt is the structural replication unit. After averaging the three generation seeds within each prompt, the learned or deployed selector's quality advantage was +5.43 points for motif (95% prompt-bootstrap interval +3.17 to +7.87), +2.23 for decoration (+0.60 to +4.07), and +7.53 for linker (+3.57 to +11.50). Motif quality improved on nine prompts and tied on one. Decoration improved on five, tied on three, and favored uniform on two. Linker improved on seven and tied on three. The full per-prompt and prompt-seed rows, including unfavorable results, are preserved in the machine-readable artifacts.

There are tradeoffs. Uniform selection increased motif diversity by 0.01154 and linker diversity by 0.01082. Decoration uniform selection increased uniqueness by 1.33 points on average; its diversity difference was mixed across prompts. The current linker contrast changes both native reference-score weighting and the deployed fourfold preference for previously unseen endpoints. It cannot isolate the learned reference score alone. Proposal construction and native model-support admission are held fixed in all three finite-panel comparisons.

Conditional expected prompt-faithful quality-pass probability on the saved full panels is 44.79% versus 39.90% for motif, 42.42% versus 38.42% for decoration, and 35.36% versus 30.42% for linker, deployed versus uniform. These expectations are additive per-slot diagnostics. They are not substitutes for the published uniqueness-adjusted cohort quality metric.

## Attempted-offer prefix analysis

The source receipts preserve all eight offers in their original attempted order, including compiler failures, duplicates, and model-support abstentions. The analysis retained the first 1, 2, or 4 attempted offers and applied the original eligibility and endpoint deduplication rules. At a one-offer budget, learned and uniform selection coincided for every attempted output slot. Budget 8 reused and reproduced the locked full-panel result. The smaller budgets are retrospective truncations of the eight-offer generator's recorded proposals, not independent runs of a newly configured smaller generator. Selector replays are not additional generation seeds, and the offer-count axis is not a measured wall-clock speedup.

| Task | Attempted offers | Deployed quality | Uniform quality | Deployed validity | Uniform validity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Motif | 1 | 26.70 | 26.70 | 75.13 | 75.13 |
| Motif | 2 | 34.37 | 33.30 | 92.80 | 92.80 |
| Motif | 4 | 40.53 | 37.30 | 99.20 | 99.20 |
| Motif | 8 | 42.63 | 37.20 | 99.97 | 99.97 |
| Decoration | 1 | 31.93 | 31.93 | 91.43 | 91.43 |
| Decoration | 2 | 35.13 | 34.03 | 99.13 | 99.13 |
| Decoration | 4 | 36.70 | 34.10 | 99.93 | 99.93 |
| Decoration | 8 | 36.70 | 34.47 | 100.00 | 100.00 |
| Linker | 1 | 21.83 | 21.83 | 92.20 | 92.20 |
| Linker | 2 | 26.27 | 23.43 | 99.20 | 99.20 |
| Linker | 4 | 29.73 | 24.07 | 99.97 | 99.97 |
| Linker | 8 | 31.53 | 24.00 | 100.00 | 100.00 |

Prompt fidelity matched validity at every prefix in these three runs. Full prefix rows also record uniqueness, diversity, and faithful quality-pass yield. Empty eligible panels remained failures and were never refilled.

## Provenance and verification

| Artifact | SHA-256 |
| --- | --- |
| Existing motif full-panel selection result | `3d05f15e29dcf5f72f89480841ee9a67e380a8d1d2f176580f7a4e3c23a9e559` |
| New motif prefix result | `6977b8a69257f230a3e8dc2221ea2c3497d41b5975ad5a53d8b59f28f01ef61a` |
| New decoration result | `c6452cc46128f7b9cb13f23d0089ef5f582c2168458bad644e82bd1f026655ab` |
| New linker result | `a7b4f41b172ba70e58340f1232e3eb79be78b5cdc35d64cc94e0131128549ee1` |

The new results are in `diagnostics/fragment_locked_panel_selection_v1/`. The reproducible commands, from the repository root, are:

```sh
.venv_pinned_chem/bin/python tools/analyze_fragment_locked_panel_selection_v1.py scaffold_decoration --output diagnostics/fragment_locked_panel_selection_v1/decoration_result.json
.venv_pinned_chem/bin/python tools/analyze_fragment_locked_panel_selection_v1.py linker_design --output diagnostics/fragment_locked_panel_selection_v1/linker_result.json
.venv_pinned_chem/bin/python tools/analyze_fragment_motif_locked_prefix_v1.py
```

Each command refuses to overwrite an existing result. The recorded outputs include exact contract, manifest, summary, prompt, row, and attempt hashes, software versions, analysis-source hash, and seed derivation. Focused selector tests passed (4 tests), and lint passed for both analysis scripts and the test. A repository-wide verification suite was not run for this isolated retrospective analysis.

## Proposed manuscript wording, not applied

We compared the deployed selection rule with uniform selection over identical saved panels of executable molecular completions. The reference-score selector increased mean quality by 5.43 and 2.23 percentage points in motif extension and scaffold decoration, respectively. In linker design, the deployed rule combines reference scores with a preference for previously unseen outputs and increased quality by 7.53 points. These comparisons isolate final panel selection conditional on the existing proposal and support checks. The separate superstructure ablation tests learned edit probabilities during primitive construction.
