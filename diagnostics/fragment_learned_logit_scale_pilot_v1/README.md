# Learned legal-mark logit tilt: negative development screen

Decision: **do not promote** the `sampling_logit_scale=2.0` chemistry-prior candidate. This was the single prespecified learned-logit tilt, not a parameter sweep. The model already supplied learned logits over legal rewrite marks, so this candidate sharpened those logits during proposal sampling. It did not use QED, synthetic accessibility (SA), a target molecule, or any score at inference. The frozen official superstructure sampler and run were not changed.

The self-hashed protocol is `configs/fragment_learned_logit_scale_pilot_v1.json` (payload SHA-256 `fdb138168d49b6e3fea94173a0f222654df47991604b46b35830bdb87370efb5`). The machine-readable result is `result.json` (SHA-256 `d7175ec4e97c55d27e1dad9bc8ceb29e78c44026cf223b6ee269280b529167b1`). The result records checkpoint, frozen baseline shard, evaluator, prompt-manifest and implementation hashes, software versions, configurations, and all attempt-level molecules and provenance. The control is the first 10 attempts of seed 0 from each of the 10 frozen official superstructure shards; the candidate used the same prompt, seed, checkpoint, executor, and hard lock with the sole sampling-logit change. This is a 100-attempt development ablation, not an official benchmark estimate.

| Metric | Frozen control | Tilted candidate |
| --- | ---: | ---: |
| Quality (QED >= 0.6 and SA <= 4), percent of attempts | 40 | 24 |
| Diversity | 0.709697 | 0.572078 |
| Committed output rate | 1.000 | 1.000 |
| Prompt fidelity | 1.000 | 1.000 |
| Chemical validity and locked-fragment preservation of committed outputs | 1.000 | 1.000 |

The predeclared gate required at least **+10 percentage points** in quality and no more than **0.03** diversity loss, with output/fidelity losses no greater than **0.05** and exact committed validity/preservation. Quality instead changed by **-16 points** and diversity by **-0.137619**. The candidate fails both central criteria decisively. This does not establish that a stronger, split-clean chemical prior cannot work; it only rejects this simple sharpening of the existing learned legal-mark distribution. No larger candidate run or promotion is justified.

The planned 24-seed de novo cross-check was started but stopped before a complete artifact because this same shared candidate had already failed the fragment gate and the first uncheckpointed trajectory was computationally expensive. It produced **no interpretable de novo result**. Do not count that attempted cross-check as an ablation outcome or vary its sample size post hoc to rescue the decision.

The independent official superstructure baseline completed unchanged at 3,000/3,000 committed, chemically valid, fragment-preserving and prompt-compliant outputs. Its aggregate result is in `diagnostics/fragment_superstructure_official_v2/result.json` in the frozen `fragment-interface-ablation-20260923` worktree: quality 39.0333%, uniqueness 97.3333%, diversity 0.725074, distance 0.643256. Two prompts, Cyclothiazide and Lovastatin, had 0% quality, but the full aggregate is the relevant benchmark outcome. No prompt-specific remedy was introduced.

Checks: the focused fragment tests passed (74 tests), the logit-scale test passed, the result-schema/summary invariant check passed, and lint on new/touched small files passed. The large model file has 16 pre-existing Ruff findings in both the source and pilot branches; unrelated formatting was not changed. The repository-wide suite was not run, because this negative candidate is not being promoted or declared a milestone completion.
