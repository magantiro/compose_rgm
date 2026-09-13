# Frozen T4 Program-Vocabulary Audit

## Scientific question

Characterize the exact 146-program bank frozen for the T4 delta-0.4 benchmark.
The audit asks whether the bank is only a general, context-bound transformation
vocabulary, or whether task-, seed- or winner-specific information remains in
the deployed representation.

The primary output is a sealed, per-program machine-readable audit. The central
claim under test is narrow: the frozen bank erases physical source addresses and
rebinds typed inputs against molecular context, but its actual specificity and
cross-cell applicability must be measured rather than inferred from that design.

## Frozen setting and inputs

The audit reads, but does not modify:

- `configs/t4_frozen_program_benchmark_v2.json`;
- the exact shared library referenced by that contract;
- `docs/GENMOL_T4_SEEDS.json`;
- `diagnostics/ivg_t4_census/census.json`;
- `diagnostics/t4_program_retrieval/attempt_2/result.json` and its sealed units;
- four measured development archives in
  `diagnostics/t4_program_curriculum/attempt_1`.

All material inputs and implementation files receive SHA-256 identities in the
result. The audit must run from a clean committed source tree. It makes no task
oracle or Modal call and cannot update the frozen controller, library, running
benchmark, or any prior artifact.

## Bounded procedure

1. Require the exact library path and count declared by the frozen benchmark
   contract. Parse every entry through the production `EditProgram` schema.
2. Scan serialized and parsed entries for literal target names, seed identities,
   docking-score fields, endpoint fields, endpoint SMILES and raw integer atom
   operands. Map every opaque source-group identity by enumerating the 15 public
   target and seed pairs.
3. Record program size, block labels, primitive families, typed input and created
   handles, and whether the program is a compiled complete public-winner route.
4. For every program on every one of the 15 frozen exact source states, enumerate
   at most 64 context-ranked bindings and at most 4,096 binding-search visits.
   Report structural binding availability separately from exact execution. Try
   each returned binding only until one exactly executes, under the frozen limit
   of 32 primitives and eight blocks. Record truncation and attempted executions.
5. Verify the existing direct-retrieval result and its unit hashes. Report known
   public-winner endpoint overlap as answer-known reconstruction evidence, not
   discovery evidence.
6. In each of the four existing measured development archives, identify the best
   observation, locate its exact archived program, and map every recorded library
   draw to same-cell, same-target-other-seed, or cross-target provenance.

Binding coverage is only an applicability measure. Exact execution precision is
the fraction of attempted bindings that completed before the first success; it
is not task quality. Binding enumeration is capped, so zero measured bindings is
not a universal non-applicability proof when the census reports truncation.

## Acceptance criteria

- The frozen contract is self-hash-valid, the library hash matches its recorded
  input identity, and exactly 146 unique programs parse.
- All library source groups are either mapped uniquely to the 15 public T4 cells
  or the audit fails.
- Literal-field, atom-reference and route-specificity findings are reported for
  all programs without suppressing unfavorable results.
- Structural-binding and exact-execution applicability are reported per program
  and summarized across cells and targets, with caps and truncation explicit.
- Existing direct-retrieval winner overlap and four measured best-candidate
  provenance records are hash-verified and reconciled.
- The result records code revision, complete configuration, input and
  implementation hashes, software, hardware, elapsed time, executor attempts,
  exclusions and zero oracle calls.
- Focused tests and lint pass, the generated result is inspected, and source
  inputs remain byte-identical.

## Claim boundary

This is retrospective, answer-known development evidence. Address erasure,
cross-cell binding, or cross-target use does not prove held-out generalization.
Conversely, a complete winner-derived route is not converted into a generic
grammar merely because its atom addresses are rebound. The final interpretation
must state both facts and use the description supported by the measured result.
