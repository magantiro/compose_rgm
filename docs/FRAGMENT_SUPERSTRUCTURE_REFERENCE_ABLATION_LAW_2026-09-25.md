# Superstructure transition-reference ablation law

Status: frozen intervention definition before comparative evaluation. This is
not a result or a launch receipt.

The primary task is the ten-prompt superstructure-generation panel in
`configs/fragment_superstructure_official_v2.json`. Its completed learned arm
is `diagnostics/fragment_superstructure_official_v2/result.json` at source
revision `d3ce6dfbf740588f23790dec71bf72e6f149a8b5` and contract payload
hash `77c994c8364d088c6e5bf1fe688794cac5869c7e4abf98c92a87d08e1b1c751e`.
The arm emitted 3,000 of 3,000 attempts and recorded every output. The frozen
RingCore checkpoint SHA-256 is
`24117dfeaee91729bb4ebccb5eb5b993b4a6218605045e51917823d086b4c1e4`.

## Exact nonlearned mark law

For each current state and operational time, construct the *same native
factorized action masks* as the frozen RingCore sampler, with the same operator
flags, vocabulary, structural representation, disabled-family settings, and
optional allowed-family restriction. A family is available if its native mask
contains at least one action coordinate. For atom insertion, the root and
connected-insertion masks form one family. Choose uniformly among available
families. Within the chosen family, choose uniformly among its native masked
action coordinates, counting root and connected insertion coordinates together.
If the chosen family cannot materialize an action, remove it and redraw from
the remaining available families, matching the learned sampler's no-action
fallback. This law uses no learned family or operand/payload logits. It is a
uniform-family, uniform-native-mark law, **not** uniform over canonical
molecular successors. Canonically equivalent marks retain their native
multiplicity in both arms.

The learned arm uses the original family and operand/payload logits. The
nonlearned arm replaces both probability-dependent choices, including the
initial `atom_insert`-conditioned draw. All action masks are computed from
the state and frozen operator configuration, without thresholding a learned
score. The model's frozen total hazard remains in both arms so that the
operational-time stopping rule and holding-time distribution are matched.
Thus the causal intervention is on the conditional transition-mark law, not
the CTMC rate or the molecular support.

## Matched protocol and interpretation

Keep `sample_completion`, attachment redirection, effective-chemistry lock,
executor validation, 24 mark attempts per event, 32 committed-event cap,
operational horizon 16, ten prompts, 100 attempts per prompt and seeds 0/1/2.
No failed attempt is replaced. Score all 3,000 attempt slots with the same
SHA-pinned official evaluator, and report prompt fidelity separately from
chemical validity. The learned arm's stored result can be reused only after
its code, checkpoint, prompt, seed, sampler-config and evaluator identities
match the new comparison contract; otherwise rerun both arms under that
contract. Do not substitute a later decoration, motif, or linker pilot.

This comparison tests the learned mark probabilities on the active
superstructure sampler. It does not isolate reference effects on finite-panel
program ranking in motif or linker, and does not compare with a separately
trained uniform model. An optional saved-panel motif replay is a distinct
conditional selection diagnostic.
