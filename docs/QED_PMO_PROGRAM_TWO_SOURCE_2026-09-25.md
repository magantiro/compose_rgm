# Two-source QED program-controller diagnostic

The question is whether the PMO population controller's coordinated
joint-dependency proposals and feedback allocation improve source-conditioned
QED editing relative to the older Dynamic-v2.1 program controller. The
generated object is an executable COMPOSE program ending in a complete valid
molecule. This is a small mechanism diagnostic, not an 800-source GrIDDD result.

Use the first two rows, indices 0 and 1, of the pinned exact public Jin QED
test list. Both sources lie in the declared QED 0.7-0.8 cohort and fit the
48-slot executable graph representation. The two controller arms share each
source, deterministic source seed, 25 charged local QED evaluations, three
rounds of at most eight candidate queries, 64 proposal attempts per batch,
20 seconds per proposal batch, the exact executor, and source-relative Morgan
radius-2 2048-bit Tanimoto similarity. The first charge is the source itself.
Unused calls, duplicates and failed proposals are reported, not replaced.

The old arm uses Dynamic-v2.1 complete-program proposals and its feedback
controller. The new arm reuses the production PMO population controller with
its shared, task-blind joint-dependency plan checkpoint and recursive archive.
The checkpoint is frozen and contains no QED label or source-to-program map.
The sole controller intervention is this added PMO population/jump machinery;
the initial program constructor and the score function are identical.

For both arms, a valid endpoint receives QED(y) if its source-relative
similarity is at least 0.4 and zero otherwise. A molecule succeeds if QED is
at least 0.9 and similarity is at least 0.4. This local QED score is used
*during* feedback search. It is not the future-value SMC law, and the 25
scored endpoints are not 25 independent generated samples. The comparison
therefore estimates whether the PMO controller helps this two-source search,
not whether it beats GrIDDD or the paper-era SMC sampler. No SA or T4 docking
gate is applied. The similarity floor affects reward rather than proposal
support, so off-floor scored candidates consume calls and remain visible.

The run records every charged endpoint, program channel, QED, similarity,
success, query shortfall and progress receipt. The local scorer is deterministic
RDKit and no paid or remote oracle is called. The frozen source list and PMO
checkpoint are material-hash bound in a self-hashed contract. Do not use either
source result to retune this pilot. A general benchmark claim would require a
separately authorized, source-level evaluation with its own candidate-count
and work-accounting protocol.
