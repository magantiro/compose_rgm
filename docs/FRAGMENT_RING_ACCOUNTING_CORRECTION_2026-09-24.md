# Ring-capability accounting correction

The completed joint-completion support v1 run contained a reporting defect.
In RDKit 2024.03.5, obtaining RingInfo from a temporary MolFromSmiles object
allowed the parent molecule to be destroyed before NumRings was read. This
silently reported zero initial rings and overstated the `new_ring` capability.
The content sampler and compiler already retained their molecule objects; the
sampled structural cells, executed programs, learned scores, selections and RNG
states were unaffected. No quality metric was evaluated during this gate.

The original run remains immutable. `repair_fragment_ring_accounting.py`
copied all forty sealed attempts into a new v2 directory and changed only
`new_ring` labels in candidate/selected capability metadata. Forty-two labels
changed. `accounting_correction.json` binds original and corrected attempt hashes.
The corrected runner then resumed the completed attempts and recomputed rows
and the unchanged support gates, without drawing another proposal.

The corrected gate passed: motif 20/20 and decoration 20/20 outputs, exact
fragment fidelity for every output, all selected programs multi-primitive.
The decoration ring-adding fraction is **0.55**, not 1.00. Motif remains 1.00.
Selected T4 descendants are 0.40 and 0.60 respectively, and all three native
T4 lanes have model-supported candidates. Mean endpoint heavy atoms/rings are
27.60/3.30 for motif and 32.45/4.15 for decoration. These are twenty-output
support observations per task, not official quality or diversity results.

Four focused tests passed in the pinned chemistry environment: zero/one/two
ring fixtures and an exact check that metadata repair changes no other attempt
field. Ruff and diff checks passed. No repository-wide pass is claimed.

The subsequent seed-zero matched metric pilot uses the identical sealed first
two attempts for each prompt, including any failures, and generates attempts
2 through 19. This avoids duplicate compute, not duplicate accounting. The
preview and pilot are explicitly **not independent replications**. The prefix
receipt binds every reused byte; three focused tests cover exact copying,
failure retention, failed-gate refusal and changed-attempt refusal.

Authoritative directories:

- Original: `diagnostics/fragment_joint_completion_support_v1`
- Corrected: `diagnostics/fragment_joint_completion_support_v2`
- Unranked drawings: `diagnostics/fragment_joint_completion_support_visuals_v1`
- Matched pilot: `diagnostics/fragment_joint_completion_matched_pilot_v1`

The molecular sampler, model checkpoint, training prior, support limits,
qualification thresholds and QED/SA-free generation policy did not change.
