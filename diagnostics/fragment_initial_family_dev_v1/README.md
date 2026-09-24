# Locked-core first-event proposal diagnosis, 2026-09-24

Evidence role: zero-oracle development and matched qualification, **not** the
three-seed/100-attempt official benchmark or a promoted controller. Inputs,
implementation hashes, software, all molecule SMILES and per-attempt action and
refusal records are in the JSON files in this directory. The pre-outcome
all-ten protocol is `docs/FRAGMENT_INITIAL_LOCKED_FAMILY_DEV_2026-09-24.md`.

## Mechanism

The base molecular executor keeps its committed endpoints chemically legal,
but it does not know that a benchmark-supplied core is locked. The fragment
adapter's pathwise lock refuses incompatible edits after the learned mark has
been sampled. In the frozen first-event audit, the three development prompts
had legal insertion counts of 9, 25 and 65 out of 256 sampled marks; every
observed other family was refused at the fully locked initial state. These
counts motivated an opt-in learned family draw conditioned on `atom_insert`
*only* while every real initial atom is locked. The exact executor and lock
still check the resulting bound action. This is not an exact legal-mark-fiber
sampler and does not guarantee every future sampled mark is admissible.

## Measured outcomes

| Development assay | Arm | Official-valid / attempts | Chemically valid / committed | Fragment preserving / committed | Quality | Uniqueness | Diversity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Three selected prompts, seed 9 | historical | 46 / 60 | 46 / 46 | 46 / 46 | 25.0% | 93.3% | 0.708 |
| Three selected prompts, seed 9 | conditioned | 60 / 60 | 60 / 60 | 60 / 60 | 25.0% | 98.3% | 0.706 |
| Ten prompts, seed 10 | historical | 184 / 200 | 184 / 184 | 184 / 184 | 36.0% | 96.7% | 0.722 |
| Ten prompts, seed 10 | conditioned | 200 / 200 | 200 / 200 | 200 / 200 | 34.5% | 97.0% | 0.731 |
| Ten prompts, seed 10 | conditioned + effective-chemistry lock | 200 / 200 | 200 / 200 | 200 / 200 | 34.5% | 97.0% | 0.728 |

The ten-prompt historical arm's 16 misses were all zero-event attempts that
exhausted 24 first-event draws. The conditioned arm accepted the first
conditioned draw in all 200 attempts, and lock refusals decreased from 1,835
to 1,051 over the complete trajectories. The residual refusals occur after
the first event or from other downstream constraints; the repair does not mask
every bound operand. The strict opt-in lock produced the same output count in
this assay, but an earlier three-prompt pilot showed a one-output cost on
Futibatinib. The strict lock's sample preservation count is not proof of
universal chemical-environment invariance.

Quality is a negative/mixed finding. On the ten-prompt assay the conditioned
arm lost 1.5 percentage points of mean official quality despite better output.
The QED/SA decomposition in `quality_seed10_n20_all10.json` shows that
Erlotinib and Futibatinib losses were primarily QED-threshold crossings, not
SA failures or a large shift in heavy-atom count. Spirapril improved from 10%
to 25% quality, mostly because it produced more endpoints. The deterministic
`quality_examples_seed10.png` grid and its selection manifest show first
three distinct quality-pass and first three quality-fail molecules for each
baseline/conditioned Erlotinib and Futibatinib group. The examples do not
show a new grossly invalid ring or valence pattern; many failures are ordinary
decorations falling below QED 0.6. This is a qualitative inspection, not a
blinded medicinal-chemistry assessment.

The historical full official superstructure row remains 93.53% validity,
97.58% uniqueness, 37.67% quality and 0.726 diversity on 3,000 attempts.
The seed-10 200-attempt development values must not replace it or be compared
as a full official result. The apparent ~24-minute per-arm sum was dominated
by one slow prompt in each arm and does not yet establish a speed advantage.

## Interpretation and next gate

This is strong evidence that moving a task-supplied **structural constraint**
into the proposal law recovers output that post-draw refusal was wasting, while
keeping exact chemical execution. It does not solve the separate quality-prior
problem. No data support changing the global molecular model, QED/SA rules,
or full downstream control based on this assay.

Before promotion, freeze a clean revision and run independent per-prompt,
restart-safe 100-attempt shards for all ten prompts and three seeds, with the
official evaluator and the same all-attempt denominator. Compare the historical
law, conditioned law and opt-in strict lock on validity, quality, uniqueness,
diversity, molecule-level preservation, refusals and wall time. If strict
locking or family conditioning materially harms quality/diversity or any
prompt's output, report the negative result and retain the baseline.
