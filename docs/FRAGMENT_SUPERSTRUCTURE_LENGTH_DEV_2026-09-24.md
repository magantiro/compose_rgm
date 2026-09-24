# Superstructure trajectory-length development gate

Status: **deferred, no fresh-seed run launched**. This zero-oracle development
comparison was predeclared before the user directed us to prioritize the full
official fragment rows and to separate chemical validity from prompt fidelity.
Its exploratory prefix result remains evidence, but this document authorizes
no immediate sampler selection or replacement for the official benchmark. It
is not a claim of superiority to InVirtuoGen or GenMol.

The post-ring fragment sampler preserves complete connected graph states. Its
strict locked-core arm emitted 200/200 valid, fragment-containing endpoints on
seed 10. Replaying those exact trajectories at event caps 4, 8, 12, 16 and 32
showed an exploratory quality-diversity tradeoff: official quality was 48.5%,
61.0%, 50.0%, 40.0% and 34.5%, respectively; within-prompt diversity was
.599, .670, .714, .724 and .728. Full action replay recovered every original
endpoint and reproduced the cap-32 official metrics exactly. This prefix audit
is `diagnostics/fragment_initial_family_dev_v1/prefix_caps_seed10_n20_all10.json`
(SHA-256 `4f5d39c6fdeb4c5d4d2231486f4b3323c014a53cb10459fadbdcd4d017f55c93`).
It is not a fresh reduced-cap sample: the existing runner shares one RNG stream
across attempts within a prompt, so stopping an attempt sooner changes later
draws. The exploratory diagnosis also found that only four of 194 distinct
conditioned outputs improved synthetic accessibility over their supplied core;
Cyclothiazide and Lovastatin cores start above SA 4.0.

## Frozen independent development comparison

Run seed **11**, all ten released superstructure prompts, 20 attempts per
prompt, with one global controller for both arms. The only arm difference is
`max_events`: 8 versus 32. Both use the same learned checkpoint, 48 slots,
operational horizon 16, 24 mark draws per event, attachment control, first
all-locked-family conditioning and effective-chemistry hard lock. No task/drug
identity, reference drug, score or quality threshold enters the proposal law.
The benchmark metric is the pinned InVirtuoGen evaluator, unweighted across
prompts. Attempt records and canonical SMILES must be retained.

Frozen material identities:

- Prompt CSV SHA-256: `a4fb8357d0f1102cbdc8d79d802e15f66a59a9722c0b7125ce693fe7a29872a9`.
- Checkpoint SHA-256: `24117dfeaee91729bb4ebccb5eb5b993b4a6218605045e51917823d086b4c1e4`.
- Model source SHA-256: `9e802713df89be8f7c12dc8caa20157732a28b7d014bfce2498ba2980161a6fc`.
- Sampler source SHA-256: `1813d5c606f75e957d4e327fb2e201f97ecfeafda398d8ab53ff1905e473c2aa`.
- Attachment source SHA-256: `cc4a9600e3c7873af2c1c28193cba836522dde5e79dc9ab06fbf0fc350462318`.
- Runner source SHA-256: `c64298556f72638656bb79fddb19d50ac451d6ce7019038ed113dea55dee894a`.
- Pinned evaluator: InVirtuoGen commit `b50bb3ae2bdcb9df581f0b219d79cf14b05d0fbb`, `train_utils/metrics.py` SHA-256 `3c4bb7c6727cbeaf02d3d5eebf1deac27f77bab68d929b61dbe4154911e2b099`.

Run each arm to an independent atomic JSON artifact under
`diagnostics/fragment_superstructure_length_dev_v1/`. Before reduction, verify
every identity above, all 200 attempt records per arm, exact chemical validity
of every committed endpoint, independent fragment containment, no output
classification, and reconstruction of official metrics from emitted SMILES.
Report per-prompt as well as ten-prompt validity, uniqueness, quality and
diversity; record wall time and accepted event counts. Compare official quality
on the full 200-attempt denominator, not conditional on successful output.

The short-cap hypothesis qualifies for a later frozen official run only if:

1. both arms have 100% chemically valid committed endpoints and 100% required
   fragment containment among commits;
2. cap 8 loses no more than two percentage points of attempt-level output;
3. cap 8 gains at least ten percentage points in unweighted official quality;
4. cap 8 retains at least 90% official uniqueness and .65 within-prompt
   diversity; and
5. cap 8 has nondecreasing quality on at least seven of ten prompts.

If any gate fails, preserve and report the negative result. Even a pass selects
only a candidate for an independently frozen full official suite. It does not
justify retroactively comparing this 20-attempt development row to published
100-attempt rows. Do not alter the benchmark quality definition or the hard
chemical/fragment lock to rescue the short-cap arm.
