# Saved completion and catalog-allocation census

This is a post-generation, development-only diagnostic. It changes no sampler,
checkpoint, training catalog, frozen pilot, T4 artifact or evaluator. All quality
scores below apply only to saved complete molecules. No molecular generation,
checkpoint loading, oracle call or docking was performed.

## Final full-pilot census, 15:50 UTC

The separately frozen full census contains all twenty prompt rows and 400
attempts per arm. The earlier partial snapshot below remains unchanged.
`diagnostics/fragment_completion_census_full_v2/census.json` has SHA-256
`e6242615b86a2895f7ff1120203f5b08afd27cb653bc6f846e10ca1732fd1926`.
The compact, versioned result is
`diagnostics/fragment_completion_census_full_receipt_v2/receipt.json`, SHA-256
`6e8aa073601edfed4ad69448cca9477075ad00533cc313fcab2d90c218f9cc8a`.
It retains every input hash, aggregate, per-prompt denominator, failure count,
distribution and source-census hash, while the full local census retains all
candidate and molecule records. Both metric rows agree with the finished pilot's
`summary.json`.

| Full task, 200 attempts/arm | Outputs | Raw joint passes | Unique joint passes | Quality (%) | Uniqueness (%) | Diversity | Validity (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Motif | 200 to 200 | 56 to 74 | 39 to 74 | 19.5 to 37.0 | 80.5 to 100.0 | 0.598177 to 0.682225 | 100 to 100 |
| Decoration | 200 to 198 | 66 to 4 | 62 to 4 | 31.0 to 2.0 | 98.0 to 100.0 | 0.581507 to 0.680463 | 100 to 99 |

All ten decoration prompts have larger mean outputs, more rings, lower QED and
higher SA than their own baseline. Eight lose quality and two remain at zero.
The full decoration mean size is 25.885 to 38.575758 atoms, ring count 3.31 to
5.181818, QED 0.591245 to 0.281685, and SA 3.735183 to 4.188035. Descriptor means
use emitted endpoints, so their denominators are 200 and 198. The two no-output
attempts still remain in all official quality and validity denominators.

Failure census for the complete-program adapter, exactly twenty attempts per row:

| Task | Prompt | Both pass | QED only fails | SA only fails | Both fail | No output |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Motif | Baricitinib | 14 | 2 | 2 | 2 | 0 |
| Motif | Cyclothiazide | 4 | 8 | 0 | 8 | 0 |
| Motif | Eliglustat | 12 | 5 | 1 | 2 | 0 |
| Motif | Erlotinib | 4 | 14 | 0 | 2 | 0 |
| Motif | Futibatinib | 9 | 6 | 1 | 4 | 0 |
| Motif | Lesinurad | 11 | 5 | 0 | 4 | 0 |
| Motif | Liothyronine | 11 | 6 | 0 | 3 | 0 |
| Motif | Lovastatin | 0 | 0 | 4 | 16 | 0 |
| Motif | Maribavir | 5 | 8 | 0 | 7 | 0 |
| Motif | Spirapril | 4 | 0 | 13 | 3 | 0 |
| Decoration | Baricitinib | 1 | 16 | 0 | 2 | 1 |
| Decoration | Cyclothiazide | 0 | 0 | 0 | 19 | 1 |
| Decoration | Eliglustat | 1 | 6 | 0 | 13 | 0 |
| Decoration | Erlotinib | 0 | 18 | 0 | 2 | 0 |
| Decoration | Futibatinib | 0 | 13 | 0 | 7 | 0 |
| Decoration | Lesinurad | 2 | 16 | 0 | 2 | 0 |
| Decoration | Liothyronine | 0 | 18 | 0 | 2 | 0 |
| Decoration | Lovastatin | 0 | 0 | 0 | 20 | 0 |
| Decoration | Maribavir | 0 | 0 | 0 | 20 | 0 |
| Decoration | Spirapril | 0 | 0 | 0 | 20 | 0 |

The full decoration panels retain the same ranking finding: softmax expected
size is 0.158172 atoms below uniform panel selection, while supported uniform
panels already average 38.594192 atoms. There are 734 compiled endpoints from
1,600 offered draws, 554 model-supported candidates and 198 selected outputs.
Selected refinement pairs (77) change mean size by -0.142857 atoms, QED by
-0.025153 and SA by +0.425643. These observations support a content-allocation
problem and do not establish a size-only causal effect.

### Maribavir and fixed visual inspection

Maribavir decoration contributes the final twenty attempts. Its required core
has fourteen atoms, three rings and six interfaces. Mean output atoms change
23.45 to 39.40; rings 3.05 to 4.95; QED 0.632384 to 0.240262; SA 4.188757 to
4.727922. Distinct joint passes fall from three to zero. Every new output fails
both thresholds, including the best new QED (0.460978) and lowest new SA
(4.037875). All twenty new outputs satisfy the existing fragment-condition
checker; seventeen baseline outputs do so, separately from chemical quality.

The full receipt directory includes inspected grids of all twenty Maribavir
outputs per arm and the first saved attempt for each of the ten decoration
prompts. This fixed illustration rule uses no score-based selection. The red
highlight identifies the supplied core, so its repeated ring systems are not
misidentified as newly generated motifs. Maribavir's new outputs visibly add
multiple substituents, often with additional aryl or heterocyclic groups, around
that six-interface core; the baseline more often uses short substituents. The
first-attempt panel across all ten prompts likewise shows substantial new
content attached around different required cores, not one repeated generated
scaffold. Structural variety and high uniqueness coexist with low quality.
These visual observations do not establish experimental synthetic accessibility.
No output or score from the current no-QED support run was inspected here.

## Frozen completed-row snapshot

The census at 2026-09-24 15:38:03 UTC includes all 19 completed prompt rows then
available: ten motif prompts and nine decoration prompts, 20 attempts per prompt
and arm. Maribavir decoration had no completed row and is excluded entirely.
Later progress does not alter this snapshot. The result is
`diagnostics/fragment_completion_census_partial_v1/census.json`, SHA-256
`c60201d9a6d08a3edf0b5826da1aff9b355b1e4dc51b0d637c0eee4c13f5364e`.
It binds every included row, attempt, baseline, manifest, analysis/evaluation
source, RDKit QED implementation and SA implementation/data by physical hash.

Computed official metrics, baseline to complete-program adapter:

| Completed task cohort | Outputs | Raw joint passes | Unique joint passes | Quality (%) | Uniqueness (%) | Diversity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Motif, 200 attempts/arm | 200 to 200 | 56 to 74 | 39 to 74 | 19.50 to 37.00 | 80.50 to 100.00 | 0.598177 to 0.682225 |
| Decoration, 180 attempts/arm | 180 to 178 | 63 to 4 | 59 to 4 | 32.777778 to 2.222222 | 97.777778 to 100.00 | 0.568980 to 0.671842 |

Quality uses distinct passing molecules within each prompt divided by all
attempts. The joint criterion is QED >= 0.6 and SA <= 4. Means of the official
prompt metrics retain their original denominators. New decoration validity is
98.888889%; all other validity values above are 100%. New emitted molecules all
pass the existing fragment-condition checker. Baseline fidelity misses remain
in the official chemical metric population, as required by that evaluator.

## Size, rings and quality across prompts

All nine completed decoration prompts have larger mean outputs, more perceived
rings, lower mean QED and higher mean SA. Seven lose quality; two were and remain
zero. These are associations across a bundled proposal change, not a causal
size-only effect. Ring counts are RDKit perceived rings, not graph cycle rank.

| Decoration prompt | Core atoms/rings | Mean atoms, old to new | Mean rings, old to new | Mean QED, old to new | Unique joint passes, old to new |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baricitinib | 18/4 | 24.90 to 35.74 | 4.00 to 5.95 | 0.577 to 0.364 | 8 to 1 |
| Cyclothiazide | 19/4 | 23.40 to 37.84 | 4.00 to 5.79 | 0.588 to 0.317 | 0 to 0 |
| Eliglustat | 18/3 | 23.75 to 38.30 | 3.00 to 5.40 | 0.658 to 0.383 | 9 to 1 |
| Erlotinib | 17/3 | 27.25 to 38.50 | 3.05 to 4.90 | 0.506 to 0.204 | 6 to 0 |
| Futibatinib | 22/4 | 29.35 to 39.60 | 4.00 to 5.30 | 0.556 to 0.274 | 9 to 0 |
| Lesinurad | 18/4 | 27.75 to 37.40 | 4.00 to 6.20 | 0.577 to 0.300 | 11 to 2 |
| Liothyronine | 13/2 | 26.80 to 39.95 | 2.00 to 4.40 | 0.568 to 0.207 | 9 to 0 |
| Lovastatin | 19/3 | 26.45 to 39.10 | 3.00 to 4.30 | 0.568 to 0.289 | 0 to 0 |
| Spirapril | 22/3 | 25.75 to 39.75 | 3.00 to 4.70 | 0.683 to 0.244 | 7 to 0 |

The 178 new decoration outputs average 38.483 heavy atoms, 5.208 rings, QED
0.286339 and SA 4.127374, versus 26.156 atoms, 3.339 rings, QED 0.586674 and SA
3.684786 in the 180 baseline outputs. Among the new attempts, 87 fail QED only,
87 fail both, four pass both and two produce no output. None fail SA alone.
The corresponding baseline counts are 61 QED-only, 30 both, 26 SA-only and 63
raw passes. QED loss therefore cannot be explained solely by an SA gate.

Motif provides a contrasting result: mean atoms increase 17.11 to 25.06 and
rings 1.215 to 2.790, but QED remains approximately flat (0.558480 to 0.564615)
and unique passing yield rises. Seven motif prompts improve quality, two tie and
Eliglustat loses one passing molecule. Baricitinib has unchanged raw passes
(14/20 each), while distinct passes increase from three to fourteen. Futibatinib,
Liothyronine and Maribavir increase raw passes from 1 to 9, 3 to 11 and 1 to 5.
Thus a blanket claim that larger or more cyclic output is harmful is contradicted
by these data. Lovastatin motif remains at zero passes in both arms.

## Proposal support, ranking and refinement

There are exactly eight offered draws per completed output attempt. Motif has
1,600 draws: 921 compiled/lock-compatible endpoints, 706 model-supported and
200 selected. Decoration has 1,440 draws: 664 compiled/lock-compatible endpoints,
501 model-supported and 178 selected. Mean unique supported panel size is 3.53
for motif and 2.783333 for decoration. Two decoration panels are empty.

| Lane | Motif offered/compiled/supported/selected | Decoration offered/compiled/supported/selected |
| --- | ---: | ---: |
| Training region | 400/342/317/107 | 360/299/277/112 |
| Shallow T4 refinement | 400/206/156/39 | 360/191/130/33 |
| Structured T4 refinement | 400/206/110/20 | 360/120/55/18 |
| Anchored T4 refinement | 400/167/123/34 | 360/54/39/15 |

Every lane contributes selected endpoints. However, 215/921 compiled motif
endpoints and 163/664 compiled decoration endpoints lack frozen-model support.
Recorded reasons include nonfinite native probabilities (151 and 111), ring
restate outside the frozen model fiber (36 and 21), root-involving reroute
incompatibility (16 and 18), and scoring-successor chemistry mismatch (12 and
13). These are explicit rejections, not silently scored candidates. No evaluator
or selected-support inconsistency was found by this saved-record audit; it does
not replace independent exact executor replay or diagnose those native-model
support gaps away.

The saved mean-local-mark softmax does **not** favor larger output within these
panels. Averaging conditional expectations over nonempty panels, its change
relative to uniform panel selection is -1.350767 atoms and -0.195209 rings for
motif, and -0.175085 atoms and +0.025240 rings for decoration. Decoration uniform
panels already average 38.492416 atoms; the actual selection averages 38.483146.
The highest-score candidate averages 37.977528 atoms. These comparisons reuse
the existing panel only and make no claim about alternative proposal laws.

Decoration region-only proposals already average 38.588629 atoms before model
support filtering. Only 4/299 compiled region-only proposals jointly pass;
4/277 remain supported. This precedes the final selection. Selected native T4
decoration descendants differ from their own saved region seeds by -0.030303
atoms, +0.181818 rings, -0.028472 QED and +0.454173 SA on average (66 pairs).
There are zero pass-to-fail and one fail-to-pass transitions in these pairs,
because most seeds already fail. Refinement is therefore not the source of the
large decoration size increase in this cohort.

Selected motif refinements are less benign: 93 saved seed/descendant pairs
change by +3.462366 atoms, +0.559140 rings, -0.131704 QED and +0.825711 SA,
with 30 pass-to-fail and one fail-to-pass transition. These changes describe
the saved transformations and selected subset. They are not an isolated causal
comparison of enabling refinement.

## Exact old-law allocation calculation

`diagnostics/fragment_old_region_allocation_census_v1/allocation.json`, SHA-256
`a6fa72d4a7c4b7dd741b67cdf2fa5a3866bac2e7654067d438293dde0c21fac8`,
integrates the frozen catalog's one-boundary law exactly by dynamic programming.
It aggregates each region's square-root occurrence weight in its context and
(added atoms, added rings) cell, integrates uniformly permuted boundary order,
and applies the existing minimum-size reservation at each remaining boundary.
There is no sampling, molecule construction, compilation, model scoring or
quality calculation. This is pre-compilation content-allocation evidence, not
an accepted endpoint or runtime-support distribution.

All eight one-boundary context classes have minimum region size two. Their
untruncated weighted mean sizes range from 13.02 atoms (aromatic N) to 21.28
(acyclic O); acyclic C has mean 18.16 and aromatic C has mean 16.86. Thus even
frequency weighting leaves a substantial large-region allocation mass.

| Decoration prompt | Interfaces | Expected total atoms before compilation | Probability of exactly 40 atoms |
| --- | ---: | ---: | ---: |
| Baricitinib | 2 | 36.946 | 0.2595 |
| Cyclothiazide | 2 | 37.878 | 0.3366 |
| Eliglustat | 2 | 38.058 | 0.3658 |
| Erlotinib | 3 | 39.260 | 0.6150 |
| Futibatinib | 4 | 39.863 | 0.8890 |
| Lesinurad | 2 | 37.696 | 0.3182 |
| Liothyronine | 5 | 39.884 | 0.9041 |
| Lovastatin | 4 | 39.678 | 0.7891 |
| Maribavir | 6 | 39.953 | 0.9583 |
| Spirapril | 3 | 39.594 | 0.7302 |

The Maribavir allocation row is a deterministic catalog-law calculation and
does not supply the missing pilot quality result. The single-interface motif
expectations range from 23.646 to 30.971 total atoms. This supplies direct
distributional evidence that sequential region allocation nearly fills the
capacity for multi-interface decoration. It supports testing joint completion
conditioning while retaining ring and size support. It does not authorize a
quality-guided proposal, fixed small-substituent repair or post-result pilot
change.

## Proposed joint law, before any generated outcome

The independently prepared training-only joint prior was inspected read-only in
`diagnostics/fragment_allocation_law_comparison_v1/comparison.json`, SHA-256
`871eb4b75b95cffb6a95bbc52a020fbf1d5329250a4ebf41392d27a12cb0ac6c`.
This comparison calls its production `plan_table` on each prompt and verifies
the old census contexts and descriptors against `build_prompt_context`.
Every old/new reachable (total atoms, total rings) cell set is identical across
all twenty prompts; all new cell probabilities are positive. No candidate was
drawn, compiled, model-scored or quality-scored for this comparison.

| All ten prompts per task, equal prompt weight | Old expected atoms | Joint expected atoms | Old expected rings | Joint expected rings | Old probability at 40 atoms | Joint probability at 40 atoms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Motif | 26.1665 | 27.3302 | 2.9799 | 3.3820 | 0.01892 | 0.01319 |
| Decoration | 38.8811 | 31.3408 | 5.1899 | 4.2130 | 0.61658 | 0.02476 |

The proposed joint law decreases expected decoration size for every prompt,
to a range of 29.520 to 34.056 atoms, without removing reachable structural
cells. It does not simply shrink every task: expected motif size increases on
eight prompts and decreases on Cyclothiazide and Lovastatin. These are exact
properties of the fitted proposal allocation law, not evidence of realized
quality, successful compilation, model support or promotion. Native refinement
can subsequently change the chosen structural cell.

## Reproduction and verification

Run both census tools with the pilot chemistry environment, one CPU thread:

```sh
PYTHONPATH=.fragment_eval_deps:src:scripts:tools OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE /Users/rmaganti/compose_rgm_git/.venv_pinned_chem/bin/python tools/census_fragment_completion_pilot.py --pilot-dir diagnostics/fragment_complete_program_t4_matched_pilot_v1 --baseline-dir diagnostics/fragment_attachment_library_pilot_v1 --output-dir diagnostics/fragment_completion_census_NEW
PYTHONPATH=.fragment_eval_deps:src:scripts:tools OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE /Users/rmaganti/compose_rgm_git/.venv_pinned_chem/bin/python tools/census_fragment_region_allocation.py --catalog diagnostics/fragment_training_region_catalog_v1/catalog.json --output-dir diagnostics/fragment_old_region_allocation_census_NEW
PYTHONPATH=.fragment_eval_deps:src:scripts:tools OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE /Users/rmaganti/compose_rgm_git/.venv_pinned_chem/bin/python tools/compare_fragment_allocation_laws.py --catalog diagnostics/fragment_training_region_catalog_v1/catalog.json --prior diagnostics/fragment_joint_completion_prior_v1/prior.json --old-census diagnostics/fragment_old_region_allocation_census_v1/allocation.json --output-dir diagnostics/fragment_allocation_law_comparison_NEW
PYTHONPATH=.fragment_eval_deps:src:scripts:tools OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE /Users/rmaganti/compose_rgm_git/.venv_pinned_chem/bin/python tools/summarize_fragment_completion_census.py --census diagnostics/fragment_completion_census_full_v2/census.json --output-dir diagnostics/fragment_completion_census_full_receipt_NEW --render-maribavir
```

The first command takes a new complete-row snapshot if the live pilot advances;
the original snapshot remains authoritative only for its listed inputs. All
official metrics were independently recomputed from the included saved outputs
using Python 3.11.13 and RDKit 2024.03.5 and matched their frozen values within
1e-10. Every saved panel selection probability reconciled to its declared
canonical deduplication and softmax rule. All bound input hashes were unchanged
when each analysis finished.

Seven focused tests passed using the repository `.venv` pytest: two panel
accounting tests, three existing inspection/quality-denominator tests and one
exact allocation-law test, followed by one compact-receipt denominator test.
The pinned chemistry environment has no pytest;
that initial test invocation failed before collecting tests. The actual numeric
censuses used the pinned chemistry environment. Ruff and `git diff --check`
passed. No repository-wide suite or completed sampler milestone is claimed.
