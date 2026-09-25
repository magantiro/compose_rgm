# Fragment common-panel selection, prompt-level uncertainty

We paired learned and uniform endpoint selection within each of the ten benchmark
prompts, averaging the three saved seeds before computing a difference. The
comparison uses the same recorded eight-offer panels for each selection law.
It isolates selection on admitted panel support, not the full effect of the
learned reference on constructing or admitting proposals.

| Task | Quality difference, percentage points | 95% prompt-bootstrap interval | Diversity difference | 95% prompt-bootstrap interval |
| --- | ---: | ---: | ---: | ---: |
| Motif extension | +5.433 | +3.167 to +7.867 | -0.01154 | -0.01785 to -0.00545 |
| Linker design | +5.267 | +1.967 to +9.133 | -0.01175 | -0.01917 to -0.00451 |

Positive differences favor learned selection. Learned selection improved
quality on nine motif prompts and seven linker prompts, with one and three
ties, respectively. Uniform selection improved diversity on nine motif
prompts and eight linker prompts. The interval is a descriptive percentile
bootstrap over ten prompts with 20,000 resamples per task and a fixed seed.
Attempts within one prompt are not counted as independent replicates. The
linker comparison is also the scaffold-morphing comparison because morphing
reuses linker outputs rather than constituting a second experiment.

The analysis is reproducible with
`python tools/analyze_fragment_common_panel_prompt_ci_v1.py` from the repository
root, using the two hash-bound saved-panel result files. Machine-readable
per-prompt differences, sample counts, seed, source hashes and software
versions are in `diagnostics/fragment_common_panel_prompt_ci_v1/result.json`.
