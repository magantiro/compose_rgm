# Completed superstructure transition-reference comparison

The completed comparison contains ten benchmark prompts, three seeds per
prompt, and 100 attempted outputs per prompt-seed for each arm. The learned arm
uses the frozen RingCore family and within-family native-mark probabilities.
The comparison arm chooses uniformly among available families and then
uniformly among legal native mark coordinates within the selected family.
Both arms retain the same native masks, structural controls, learned hazard,
operational horizon, stopping rules, executor, and evaluator. The comparison
is neither a fully unlearned generator nor uniform sampling over distinct
canonical successors.

| Arm | Quality % | Uniqueness % | Diversity | Chemical validity % | Prompt-compliant outputs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Learned edit selection | 39.033 | 97.333 | 0.725074 | 100.0 | 3,000 / 3,000 |
| Uniform native-mark selection | 25.400 | 90.167 | 0.708503 | 100.0 | 3,000 / 3,000 |
| Learned minus uniform | +13.633 | +7.167 | +0.016571 | 0 | 0 |

We averaged the three seeds within each prompt and resampled the ten prompt
differences 20,000 times. The 95% percentile interval for the mean quality
difference is +4.27 to +23.07 percentage points. The corresponding intervals
are +1.07 to +11.70 points for uniqueness and -0.00052 to +0.03111 for
diversity. These intervals describe variation across the ten supplied
structural problems; the 3,000 attempted molecules are not independent
structural units. Matching seed labels do not imply matched molecular
trajectories because the arms consume random draws differently.

| Prompt | Learned quality % | Uniform quality % | Difference, points |
| --- | ---: | ---: | ---: |
| BARICITINIB | 46.67 | 25.67 | +21.00 |
| CYCLOTHIAZIDE | 0.00 | 0.00 | 0.00 |
| ELIGLUSTAT | 66.67 | 41.67 | +25.00 |
| ERLOTINIB | 37.00 | 43.67 | -6.67 |
| FUTIBATINIB | 43.00 | 10.67 | +32.33 |
| LESINURAD | 49.33 | 36.00 | +13.33 |
| LIOTHYRONINE | 44.00 | 50.33 | -6.33 |
| LOVASTATIN | 0.00 | 0.00 | 0.00 |
| MARIBAVIR | 73.33 | 36.33 | +37.00 |
| SPIRAPRIL | 30.33 | 9.67 | +20.67 |

| Seed | Learned quality % | Uniform quality % | Learned uniqueness % | Uniform uniqueness % | Learned diversity | Uniform diversity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 41.8 | 26.3 | 98.0 | 91.3 | 0.724991 | 0.711823 |
| 1 | 35.7 | 23.5 | 97.2 | 89.1 | 0.724446 | 0.706222 |
| 2 | 39.6 | 26.4 | 96.8 | 90.1 | 0.725786 | 0.707463 |

The learned probabilities improved mean benchmark quality in this constrained
primitive sampler. Six prompts improved, two favored uniform selection, and
two tied at zero. The result does not establish improvement for every prompt
or for the separate finite-panel fragment controllers. Full prompt-seed rows,
the prompt-level uncertainty calculation, input hashes, and the deterministic
analysis seed are in
`diagnostics/fragment_superstructure_reference_final_v1/result.json`.
The original learned and comparison runs remain unchanged.
