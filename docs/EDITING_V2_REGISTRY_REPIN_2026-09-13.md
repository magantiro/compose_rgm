# Editing-V2 registry lineage repair, 2026-09-13

## Problem and scope

The repository-wide test suite fails during collection because
`editing_v2_semantic_capability_cells_v1.json` pins semantic process identity
`6c4721f0dd37132aae657e7aa5f1bfc01cef270662f228171c4587eb7dd48491`, while
the current implementation computes
`0a10a2fae24d51853dc30674e31842ea11313571d991e822f23dab7e1124445b`.
The Perindopril curriculum did not modify either dependency.

The repository already contains a read-only fixed-point verifier for this hash
chain. That verifier currently aborts with `OSError: File name too long` because
its JSON pointer discovery calls `Path.is_file()` on arbitrary prose strings.

This milestone repairs the verifier and re-pins only stale content-addressed
pointers discovered by it. It does not alter scientific payloads.

## Frozen invariants

- No process semantics, operator families, executor rules or support change.
- No registry cell, context, bin, scope, split, threshold or policy changes.
- No training, Gate-0, T1, P50 or downstream authority is granted.
- Historical and superseded identities remain explicit lineage, never rewritten
  as if they were current.
- V1-named configuration changes, if any, must be hash-pointer-only and must not
  become Process-V2 authority.
- The T4 benchmark and Perindopril curriculum payload remain unchanged.

## Procedure and acceptance

1. Add a regression showing arbitrary long JSON prose is ignored by filesystem
   pointer discovery instead of raising an operating-system error.
2. Run the read-only verifier and preserve its complete disagreement report.
3. Re-pin discovered current-target pointers in dependency order. Recompute
   self-hashes deterministically and repeat until the verifier reaches a fixed
   point.
4. Prove each edited JSON file differs only in declared hash-pointer fields.
   Any required Python edit must be a named hash constant or the verifier defect.
5. Run the focused verifier, registry and process-identity tests.
6. Run the repository-wide suite once on the frozen repair candidate.

Acceptance requires the verifier to report `AGREES`, the semantic-capability
registry to load, focused tests to pass, Ruff and `git diff --check` to pass, and
the full suite to pass. Negative or unresolved findings remain blockers and are
not converted into warnings for convenience.

## Verification record

The pointer repair reached a fixed point on 2026-09-13. The verifier reported
`AGREES` with 297 classified literals agreeing, eight explicit lineage
references, one stale non-chain PMO serializer reference, and no Process-V2
chain disagreement. The live semantic identities are:

- V1: `0a10a2fae24d51853dc30674e31842ea11313571d991e822f23dab7e1124445b`
- Process V2: `f2739338c561cdc38d550aba08e5ec756005beb9b6a97aa9b4177bd57b270682`

The focused Process-V2 verifier, chain, schema, Active8, Gate-0, P50 refusal,
and materializer tests passed: 242 tests. The two P50 tests now preserve the
historical receipts as superseded evidence and verify that the current chain
refuses them; the artifacts themselves were not rewritten.

The repository-wide suite was run once under the pinned PMO environment and
interrupted after 2,475 passing tests because one test did not terminate. Before
the interruption it reported 20 failures. The eight Process-V2 lineage failures
were repaired and the corresponding 89-test slice then passed. The remaining
failures are outside this registry repair:

- six E6 tests reject frozen artifacts whose implementation and registry hashes
  already differ from the current repository;
- three legacy T1 numerical tests require their frozen numerical environment;
- one existing `cycle_rank` uniqueness test finds definitions in both
  `enumerable_ringcore.py` and `region_replacement.py`;
- two multiprocessing tests are denied shared memory by the local macOS/OpenMP
  environment.

A temporary environment matching the E6 artifact's recorded NumPy 2.4.2,
PyTorch 2.11.0, and RDKit 2025.09.6 versions makes the three legacy T1 numerical
tests pass. E6 still refuses because the current implementation and registry
physical hashes differ from its frozen provenance, and the duplicate
`cycle_rank` failure remains. Those failures are preserved as blockers to a
repository-wide green claim. They are not repaired or waived by this milestone.
