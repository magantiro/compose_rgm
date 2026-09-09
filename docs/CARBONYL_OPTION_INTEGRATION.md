# Carbonyl editing as separate controller options

User-authorized bounded implementation, 2026-09-09. Molecular output remains a
complete 1..40-atom supported molecule in exact persistent slots. The scientific
question is whether a coherent core-remodeling program can be selected once by
WHAT and completed by HOW without reselecting the option after every primitive.
This is controller engineering, not a new trained process or performance claim.

## Granularity and scope

- Existing pendant/fused `RingSpec` programs remain one WHAT decision each.
- Add `add_carbonyl`: one oxygen insertion, double-bonded to an eligible carbon.
  Keep it separate from ring construction so oxidation and topology can vary
  independently. This is a descriptor-restricted ordinary grow channel.
- Add `insert_ring_carbonyl`: one four-primitive program, opening a nonaromatic
  cycle bond, inserting a carbon, adding its double-bonded oxygen, and closing
  to the other original endpoint. The program remembers the oriented original
  edge and both created atoms. Original atom identities and unrelated induced
  bonds must remain unchanged. Cycle rank is preserved; atom count increases
  by two. It is a general edge-subdivision program, not a winner fragment.
- Keep ordinary linker edits separate. Do not invent a dimethylamino-specific
  removal/replacement macro from the inspected target.

Both options are explicit opt-ins. The default registry, historical configs,
existing `expand_ring` restrictions, WHERE geometry, R_theta, kappa and generic
support remain unchanged. The new insertion program may consider nonaromatic
junction edges, but it does not relax any existing option or executor guard.
It must still satisfy the production executor, region context, charge policy,
product-gate mode and endpoint filters. Aromatic edges and bridges are outside
this program's descriptor support; generic support is not narrowed.

The base option prior balances existing purpose groups, putting carbonyl
addition under material and core insertion under restructuring. No new group
mass is created by naming a variant. Retain the existing exploration floor.
Primitive support is intersected with the descriptor before the inherited
macro normalization/cap, as in the existing ring/expansion channels. No
target/winner, docking score or learned Q(o) enters applicability or the prior.

## Acceptance and evidence boundary

1. Real-executor examples complete the one- and four-step options.
2. A single WHAT selection stays active through all four HOW steps, then
   returns to WHERE with the exact primitive budget decremented.
3. Frozen context, aromatic/acyclic exclusions, finite atom capacity, malformed
   progress and codec round-trip fail or pass as specified.
4. Lazy/reference enumeration agrees with explicit rows on small fixtures;
   generic and existing option behavior remain unchanged when the flag is off.
5. The known first-winner suffix can be replayed through the new option kernel,
   labeled answer-known descriptor support, never learned-law discovery.

Use focused dependency tests, lint, exact diff review and a compact provenance
receipt. No remote run, docking, fitting, broader benchmark, or implicit change
to the already recorded paired-run recipe follows from this implementation.
Support under engineering reference rows does not establish positive R_theta
mass or autonomous selection. Those remain the next bounded scientific check.

## Interface and minimal check

Construct `MolecularHierarchy(kernel, include_carbonyl_options=True)` to expose
the two new options to WHAT. `False` is the backward-compatible default. This
does not silently enable them in a saved T4 recipe. A future run must serialize
this choice in a new recipe identity and account for all primitive work.

Run `PYTHONPATH=src:. python -m pytest -q tests/test_carbonyl_option.py` in the
declared chemistry environment. The small reference rows in this fixture are
state-derived C/O/F engineering rows, not the learned reference checkpoint.
The answer-known suffix test consumes
`diagnostics/t4_whole_ring_plan/result.json`, SHA-256
`252dccee4785d3a8e972df30c4a075e7e98bdf02e120300990bcb1de9d21cee9`.
Only that test is skipped if the documented vendored asset is absent.

For the previously demonstrated 21-primitive route, the decomposition is four
ordinary linker edits, two existing whole-ring options, one carbonyl addition,
and one four-step core insertion: eight WHAT choices. This is an interface
decomposition, not a shortest-path proof or an autonomous eight-choice result.
The broader core-remodeling problem is not solved by this one insertion channel.
Exact target recovery remains an answer-known development diagnostic; no
reusable fragments or target-specific atom indices enter the production option.
