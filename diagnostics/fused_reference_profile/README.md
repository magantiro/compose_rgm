# Frozen reference audit: failed before sampling

One approved attempt ran on committed revision
`8e844b422b6c4af6cca547e8f4b3d9274d99ae1e`, after strict clean-worktree preflight
and deployment. Call `fc-01M1ZCGXTQHMCCE7EAEVS1AHF9` has terminated with the
recorded ValueError. No second attempt, docking, training, guidance comparison,
or T4 enablement was launched.

## Outcome and interpretation

**Measured:** the unchanged production runtime gates passed, including catalog
fingerprint `639ff6078c32d43c` and the bound checkpoint and run-path hashes.
Initialization took 208.801 seconds. The function failed after 218.693 seconds
(before the final volume commit), during the first marked-law boundary check.
There were 48 public executor calls inside enumeration: 36 recorded execution
returns and 12 invalid-rewrite outcomes. Receipts cover every call index 0–47,
including nested calls; receipt completion order is not call-entry order.

Zero option rows and zero sampled edits were produced. Thus ring-construction
success, realized structural scale, option-candidate diversity, and proposal
time are **unmeasured**, not failed chemistry metrics. No docking scores exist
for this attempt. The raw law was not saved because validation failed before
publication. Raw counterfactual executor products are not sampled offspring.

The fixed applicability-only diagnostic selected atoms 14 and 15, a segment
releasing 2/19 = 0.105263 of the source atoms, with two eligible orientations
of the same edge. The unchanged region enumerator returned 91 regions, of which
46 passed the necessary edge-applicability check. These are not Q(M) or Q(o)
samples, nor a demonstration of large-scope rewriting. No oriented edge was
sampled. In the full controller, region selection remains the outer local/global
allocation, and this option remains an opt-in inner proposal channel.

**Inferred diagnosis, supported by the recorded failure location and source:**
the production evaluator checks normalized marked mass with absolute tolerance
2e-5. The new audit wrapper instead used NumPy `isclose` with both absolute and
relative tolerances 1e-8, effectively about 2e-8 at total mass one. The evaluator
returned, then that wrapper rejected its output. This is a numerical interface
mismatch introduced by the audit, not a measured failure of the ring option.
The exact rejected mass was not recorded, so its numerical error cannot be
reported. A synthetic predicate example in `attempt_1/review.json` documents
the mismatch; it is explicitly not a model result.

The pinned production runtime passing a gate that fails locally supports an
environment-specific discrepancy. It does not identify RDKit alone as the
cause: Python, NumPy, PyTorch, and architecture differ as well. Actual software
and hardware observations were only written on the success path and therefore
are absent from this failed attempt. The launch image recipe is preserved;
it is not a substitute for those missing runtime observations.

## Evidence and verification

`attempt_1/review.json` binds all seven downloaded remote receipts by SHA-256
and records their byte-for-byte equality with the volume, exact call status,
input identities, executor census, timings, and limitations. The volume root is:

```text
/fused_reference_profile/2d38474b17c67fe54f83edcb3b2bfe3627048396610cdce92c803cb089255728
```

The full raw receipts are preserved under the matching run ID inside
`attempt_1/`. `readiness.json` and `focused.xml` record 61 passing focused tests,
scoped lint/format checks, and unchanged existing application function ASTs.
The prior repository-wide run remains non-green: 4,423 passed, 47 failed,
58 errors, two skips, one xfail. No gate was waived and no full-suite pass or
completed repository milestone is claimed.

## Decision and next boundary

Retain this failed audit exactly. No deployed code or contract was changed after
inspecting it, and it was not retried. The proposed next action requires a new
explicit decision: repair only the audit interface against the unchanged
production numerical contract, preserve the rejected-law and initialization
diagnostics even on failure, add float32 regression coverage, and then authorize
one separately identified capped reference audit. Do not change R_theta, Q(M),
kappa, executor semantics, macro support, or docking-based selection criteria.
