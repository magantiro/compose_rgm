# Fragment results reproduced from saved metric rows

No new molecules are generated and no molecular evaluator is run. External baseline columns are not re-derived.

## COMPOSE benchmark

Mean ± sample standard deviation across three generation seeds, each averaging ten prompts with 100 attempted slots per prompt. Quality, uniqueness and validity are percentages; diversity is a fraction. Scaffold morphing shares linker outputs, not an independent experiment.

| Task | Seeds | Quality | Uniqueness | Diversity | Validity |
| --- | --- | --- | --- | --- | --- |
| Motif extension | 2, 3, 4 | 42.633 ± 0.379 | 93.632 ± 0.600 | 0.6640 ± 0.0014 | 99.967 ± 0.058 |
| Scaffold decoration | 8, 9, 10 | 36.700 ± 1.800 | 93.667 ± 0.833 | 0.5626 ± 0.0019 | 100.000 ± 0.000 |
| Linker design / scaffold morphing | 6, 7, 8 | 31.533 ± 1.401 | 81.033 ± 0.351 | 0.5636 ± 0.0020 | 100.000 ± 0.000 |
| Superstructure generation | 0, 1, 2 | 39.033 ± 3.089 | 97.333 ± 0.611 | 0.7251 ± 0.0007 | 100.000 ± 0.000 |

## Ablations

Superstructure compares learned with uniform family/native-mark sampling, retaining the learned hazard. The other tasks compare selectors on the same saved panels. The linker contrast changes reference weighting and the fourfold preference for unseen endpoints together.

| Task | Deployed quality | Uniform quality | Difference (pp) | 95% prompt-bootstrap interval | Interval |
| --- | --- | --- | --- | --- | --- |
| Motif extension | 42.633 | 37.200 | +5.433 | [3.167, 7.867] | recomputed |
| Scaffold decoration | 36.700 | 34.467 | +2.233 | [0.600, 4.067] | recomputed |
| Linker design / scaffold morphing | 31.533 | 24.000 | +7.533 | [3.567, 11.500] | recomputed |
| Superstructure generation | 39.033 | 25.400 | +13.633 | [4.267, 23.067] | recomputed |

## Attempted-offer prefixes

Retrospective truncation of saved offers, including failed offers. Not a runtime speedup or a new generation campaign.

| Task | Attempted offers | Deployed quality | Uniform quality | Deployed validity | Uniform validity |
| --- | --- | --- | --- | --- | --- |
| Motif extension | 1 | 26.700 | 26.700 | 75.133 | 75.133 |
| Motif extension | 2 | 34.367 | 33.300 | 92.800 | 92.800 |
| Motif extension | 4 | 40.533 | 37.300 | 99.200 | 99.200 |
| Motif extension | 8 | 42.633 | 37.200 | 99.967 | 99.967 |
| Scaffold decoration | 1 | 31.933 | 31.933 | 91.433 | 91.433 |
| Scaffold decoration | 2 | 35.133 | 34.033 | 99.133 | 99.133 |
| Scaffold decoration | 4 | 36.700 | 34.100 | 99.933 | 99.933 |
| Scaffold decoration | 8 | 36.700 | 34.467 | 100.000 | 100.000 |
| Linker design / scaffold morphing | 1 | 21.833 | 21.833 | 92.200 | 92.200 |
| Linker design / scaffold morphing | 2 | 26.267 | 23.433 | 99.200 | 99.200 |
| Linker design / scaffold morphing | 4 | 29.733 | 24.067 | 99.967 | 99.967 |
| Linker design / scaffold morphing | 8 | 31.533 | 24.000 | 100.000 | 100.000 |

## Per-prompt quality effects

Deployed minus uniform in percentage points, averaging three seeds first.

| Prompt | Motif extension | Scaffold decoration | Linker design / scaffold morphing | Superstructure generation |
| --- | --- | --- | --- | --- |
| BARICITINIB | +7.333 | -0.333 | +16.000 | +21.000 |
| CYCLOTHIAZIDE | +3.333 | +0.000 | +0.000 | +0.000 |
| ELIGLUSTAT | +12.667 | -0.333 | +15.000 | +25.000 |
| ERLOTINIB | +3.000 | +3.667 | +1.000 | -6.667 |
| FUTIBATINIB | +4.667 | +7.667 | +13.333 | +32.333 |
| LESINURAD | +1.000 | +6.667 | +6.333 | +13.333 |
| LIOTHYRONINE | +4.667 | +2.667 | +11.333 | -6.333 |
| LOVASTATIN | +0.000 | +0.000 | +0.000 | +0.000 |
| MARIBAVIR | +10.333 | +2.333 | +0.000 | +37.000 |
| SPIRAPRIL | +7.333 | +0.000 | +12.333 | +20.667 |

## Limits

- This manifest indexes saved metrics, not a complete molecule-generation release.
- No new generation, oracle evaluation, or model fitting is performed.
- Scaffold morphing reuses linker outputs and is not independently aggregated.
- Linker selection changes reference scoring and novelty weighting together.
- Prefix results are retrospective truncations of saved attempted offers, not runtime measurements.
- The motif_intervals artifact also contains an older linker comparison; only its motif_extension subtree is applicable here.
- Historical absolute paths inside imported evidence are provenance, never runtime inputs.
- Submitted Table 1 prints motif validity as 100%; saved rows give 99.9667%. No manuscript is edited.
- Saved intervals are carried with provenance unless --recompute-intervals is requested.
- Published external comparator columns are not independently verified or regenerated here.
