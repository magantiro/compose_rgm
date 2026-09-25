# Motif-extension common-panel selection ablation

We replayed learned and uniform endpoint selection on the same 3,000 sealed
motif-extension eight-offer panels from the completed three-seed benchmark.
The exact executor, training-derived proposal catalog, structural restrictions,
model-supported admission, prompt population and official evaluator were
unchanged. The learned arm reproduces the named motif result exactly. The
uniform arm samples among distinct model-supported endpoints in each recorded
panel. This is a conditional selection ablation, not removal of learned
transition probabilities from proposal construction or a new generation run.

| Selection law | Quality % | Uniqueness % | Diversity | Validity % | Prompt-faithful quality yield % |
| --- | ---: | ---: | ---: | ---: | ---: |
| Learned reference score | 42.6333 | 93.6316 | 0.664021 | 99.9667 | 45.4667 |
| Uniform on common support | 37.2000 | 93.5323 | 0.675562 | 99.9667 | 39.4333 |

The learned panel selector gained 5.4333 quality points and 6.0334
prompt-faithful quality-yield points at essentially unchanged validity and
uniqueness. Uniform selection gained 0.011541 diversity. Learned selection
improved quality on nine of ten prompts and tied at zero on Lovastatin. The
comparison establishes a useful contribution of learned scoring to this
finite-panel motif controller, with a modest diversity tradeoff. It does not
isolate the entire learned reference process or test a new proposal law.

Machine-readable result:
`diagnostics/fragment_common_panel_motif_v1/result.json` (SHA-256
`3d05f15e29dcf5f72f89480841ee9a67e380a8d1d2f176580f7a4e3c23a9e559`).
It binds the official motif contract and summary hashes, all 3,000 individual
attempt hashes, selected endpoints, prompt-level rows, deterministic uniform
selection seeds, checkpoint identity and execution environment. The reported
means average the 30 prompt-seed rows. The existing 2,999/3,000 output and
validity accounting remains in the denominator for both arms.
