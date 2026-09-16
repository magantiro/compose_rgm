# T4 target-conditioned utility plus Dynamic-v0 qualification v2

## Decision frozen by this revision

This revision preserves the v1 contract at commit
`b372f6c67873a4b656066df2361b0d99751be392` and changes only its prospective
promotion logic. The clarification was made before any utility-model or
qualification score governed by this comparison was observed. Call 1 remains a
mandatory diagnostic and the utility jump must still beat its target-blind
structural control across the complete five-cell paired panel. A per-cell
Dynamic-v0 loss at call 1, however, is no longer sufficient by itself to reject
a continuing search process.

The five cells remain `5ht1b_0`, `braf_1`, `jak2_1`, `parp1_0` and `fa7_0`.
The utility jump is call 1 of the unchanged Dynamic-v0 continuation. The
target-blind structural arm makes one direct query per cell. The continuing arm
runs to call 100 without score-based plateau stopping. Its matched best-so-far
summaries are reported at exactly calls 1, 5, 10, 20, 50 and 100. The complete
per-call query ledger is still required.

## Early trajectory gate

For each cell with a historical Dynamic-v0 curve, define the signed best-so-far
difference

`d_c(q) = utility_plus_v0(c, q) - dynamic_v0(c, q)`.

Lower values are better. Over the predeclared early checkpoints, the normalized
trapezoidal cumulative regret is

`R_c = [4(d1+d5)/2 + 5(d5+d10)/2 + 10(d10+d20)/2] / 19`.

This is the difference between the two piecewise-linear best-so-far areas from
call 1 through call 20. Compute it separately for `5ht1b_0`, `braf_1` and
`jak2_1`, then take their unweighted arithmetic mean. The early trajectory gate
passes only when that cell-balanced mean is at most zero. There is no tolerance
or fitted margin. Direct unrounded evaluator values govern the comparison.

The cell-balanced area cannot hide a controller that never catches up. At call
20, the utility-plus-v0 score must also be no worse than Dynamic-v0 on every one
of the three referenced cells. Call-1 differences are reported for all three,
but no longer form a separate per-cell gate.

## Paired and later gates

The utility jump and target-blind structural jump retain a complete five-cell
paired gate. The cell-balanced mean of utility-minus-structural score must be
strictly negative, the utility jump must win strictly on at least three cells,
and all five pairs must be finite and comparable.

At calls 50 and 100 on `jak2_1`, the nonnegative gap from utility-plus-v0 to
Full-146 must be no larger than the corresponding Dynamic-v0 gap at both calls
and strictly smaller at least once. The frozen Dynamic-v0 gaps are 1.4 and 1.2,
respectively. `fa7_0` remains a required missing-reference abstention because
the immutable historical artifact contains neither a Dynamic-v0 nor Full-146
curve for that cell. `parp1_0` likewise abstains from Dynamic-v0 comparisons.
Neither curve may be borrowed or reconstructed.

These rules use only exact non-inferiority, strict improvement and call-axis
weights determined by the predeclared checkpoints. No empirical tolerance is
estimated from the new controller.

## Immutable lineage and authority

The v2 contract binds the v1 commit, tree and all three v1 artifact blobs. It
also binds the same macro-to-archive integration revision, target-conditioned
selector design and historical Dynamic-v0/Full-146 reference used by v1,
including their Git object identities, physical SHA-256 values and payload
hashes where applicable.

This contract authorizes zero oracle calls, zero docking calls, zero Modal
launches and zero live-run reads. It changes no frozen artifact. An exact
utility-checkpoint and candidate/admission supplement, a clean byte-identical
launch revision, focused launch checks, and a separate explicit authorization
for at most 505 charged calls remain necessary before scoring.

The authoritative machine-readable decision is
`configs/t4_target_conditioned_utility_v0_qualification_v2.json`.
Its deterministic payload SHA-256 is
`8132f1351c9fa687f0786739536af245da32dd3542f2f77b15f14883878d60ac`.
