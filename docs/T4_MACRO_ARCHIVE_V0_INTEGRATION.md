# T4 macro-to-archive integration, zero-oracle milestone

## Scientific identity

- **Problem:** the learned compositional generator and its uniform control emit
  exact structural transformations, but there is no sealed boundary that can
  admit those transformations to one archive and then expose them to the
  unchanged Dynamic-v0 refinement law.
- **Primary output:** a deterministic, typed, zero-oracle integration receipt
  over immutable candidate references. It records lane provenance, exact
  realization, filtering, canonical endpoint collisions, selector disposition,
  archive admission and v0 handoff.
- **Claim under test:** implementation only. This milestone tests that learned
  broad macros and uniform exploratory macros can enter one provenance-preserving
  archive without changing either generator, the exact realizer, or Dynamic-v0.
- **Validation setting:** offline fixtures and immutable candidate locks. There
  are zero docking calls, zero task-oracle calls and zero Modal launches.

## Frozen boundary

The candidate flow is:

```text
learned candidate references + uniform candidate references
  -> immutable candidate resolver
  -> exact-realization receipt validation
  -> unscored benchmark-filter interface
  -> canonical endpoint deduplication
  -> selector/admission interface
  -> one run-local archive
  -> unchanged Dynamic-v0 refinement adapter
```

The learned lane accepts selector-ranked references from schema
`t4_complete_macro_selector_candidate_lock_v1`. A reference identifies a raw
candidate in an immutable generator lock; the selector artifact does not copy or
alter chemistry. The uniform lane uses the same raw-candidate schema and bypasses
the learned selector score. Both lanes pass through the same realization,
filtering, deduplication and archive code.

Endpoint collisions are resolved deterministically. One canonical endpoint is
one archive entry, while every contributing lane and candidate identity remains
in its collision ledger. Lane order therefore cannot create an additional
candidate or erase provenance.

Four independent, serializable random streams are reserved for learned proposal
generation, uniform proposal generation, archive admission and Dynamic-v0
refinement. The plumbing itself is deterministic and consumes none of them.

## Scope boundary

This revision does not generate a candidate, select a model, choose a selector
cutoff, compute QED/SA/similarity, score a molecule, mutate an archive member or
launch an experiment. It must not inspect a comparator or use a protein identity
as a policy feature. The selector cutoff and the choice between the baseline and
expanded learned generator remain explicit freeze decisions for a later
score-blind candidate lock.

The prospective five-cell qualification contract is separate at
`configs/t4_macro_archive_v0_five_cell_qualification_v1.json`. It fixes the
scientific envelope but does not authorize a launch.

The selector became immutable after the parent integration contract was
sealed. Its exact artifact and the baseline generator lock that owns its
candidate identities are therefore bound without rewriting the parent contract
in `configs/t4_macro_archive_v0_selector_binding_v1.json`. This supplement does
not choose K or a scored controller.
