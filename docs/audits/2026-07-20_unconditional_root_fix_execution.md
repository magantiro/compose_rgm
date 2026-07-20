# Unconditional root-fix execution — 2026-07-20

This note separates measured failures, running repairs, and future decisions.
The retained unconditional incumbent is still the calibrated step-6,250
pancake sampler. No pilot below is promoted from training loss alone.

## Frozen failure panel

The fixed 600-sample incumbent panel identifies three separable mechanisms:

1. **Non-ring chemistry marks:** triple-containing molecules are 54.0% versus
   6.91% in the matched reference. This is a bond-order mark-calibration error.
2. **Ring electronic composition:** rings with at least three heteroatoms,
   O--O, N--N, and adjacent aromatic `[nH]` patterns are overrepresented. The
   independent atom-by-role electronic decoder cannot model those joint
   correlations.
3. **Ring topology:** cycle rank and fused/spiro/bridged prevalence are low,
   despite adequate structured-template coverage. This is a ring-event count,
   legal-support, or within-ring topology-selection problem; it is not fixed by
   atom/bond mark calibration.

Small three/four-member rings, aromatic fraction, QED, and SA are retained as
outcomes. They are not treated as independent causes until the three mechanisms
above are resolved.

## Running matched chemistry pilots

Both pilots warm-start the exact retained step-6,250 checkpoint and reuse the
same compiled paths, sparse semantic support, evaluation tensors, seed, and
500-update schedule. They freeze the encoder, total hazard, rewrite-family
law, Graft, ring template/topology scores, and placement scores.

### P1-only

- Recipe: `recipes/tree_fcd_transfer_unconditional_chemistry_marks_only.json`
- Modal app: `ap-ASZcYuUQzMBpYpe5LLqQXA`
- Function call: `fc-01KY07SYAGRCNTTK1A8KWRXWAW`
- Artifact: `/artifacts/compose-v4-unconditional-chemistry-marks-only-20260720-v1`
- Trainable surface: grow/retype/bond-order and independent ring atom/role
  heads only.

### P1-context

- Recipe: `recipes/tree_fcd_transfer_unconditional_chemistry_contextual.json`
- Modal app: `ap-GgShIHST1BEixLDb3vtdHe`
- Function call: `fc-01KY08MKTSC7QT3VCPB5KH0K70`
- Artifact: `/artifacts/compose-v4-unconditional-chemistry-contextual-20260720-v1`
- Added trainable surface: zero-started symmetric category-pair potentials for
  ring-wide composition and for bonded ring positions.

The contextual model keeps a whole ring system as one atomic rewrite. The
finite internal electronic mark is merely factorized autoregressively. Teacher
likelihood and ancestral mark sampling use the same prefix-conditioned
potential. The exact legal support and every chemistry cache remain unchanged.

## Promotion gate

Run the same 200-source/seed rollout panel for the selected checkpoint from
each arm. A candidate must retain 100% endpoint and pathwise validity and
connectivity, zero molecular self-events, no delete-to-one collapse, no
event-budget exhaustion, and no material regression in size, cycle rank,
small-ring prevalence, or Graft backtracking. The decision metrics are:

- triple-containing fraction and triple bonds per molecule;
- ring >=3-hetero, O--O, N--N, adjacent `[nH]`, and multiple-aromatic-`[nH]`;
- aromatic-molecule and aromatic-atom fractions;
- cycle rank, fused/spiro/bridged prevalence, and three/four-member rings.

P1-context is promoted over P1-only only if the joint heterocycle panel
improves without degrading the triple correction or trajectory invariants.
Overall family accuracy is not a decision metric for these pilots because the
family logits are intentionally frozen.

## Separate fused-ring gate

Exact endpoint-preserving adjacent commutation is undergoing a preregistered
32-path falsification test. Scheduling is retained only if all endpoints are
identical and it either doubles median admitted fused-template prior mass or
halves <=5-template degeneracy without increasing small-only support. A failed
gate kills scheduling; it does not trigger another long training run.

If scheduling fails, the next bounded mechanism is a within-ring topology
factorization (topology group, then template/placement), not the rejected P2
family-mass scalar. P2 changed the probability of entering ring-grow but did
not directly supervise the fused-versus-single choice after entry.
