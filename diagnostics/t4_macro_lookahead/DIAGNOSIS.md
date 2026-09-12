# Completed-run diagnosis: missing coordinated proposals and weak delayed value

This is retrospective PARP1 seed0, delta=0.4 development analysis. The IVG
structure is an answer-known diagnostic, not a controller input. No new
molecule generation, replay, docking, model fitting or remote job was performed.

The authoritative reduction is `8776743bbb813429663f339f60cbce587e59d427248dfaee80020014b11a6b28/diagnosis.json`,
SHA-256 `6fa43172da40b8287ee3872a01f50d0a00d4e1a5fbe19d8314fc97d16e4ad07f`.
It records row-level proposals, branch decisions, parent identities, prior-only
predictions, source and input hashes, and software versions.

## Measured and computed findings

- 381 option attempts, 369 complete outputs, 354 new canonical molecules beyond
  the initial pool. Only 41 new identities passed the unchanged endpoint screen;
  40 were docked, and one remained pending. There is no large unseen eligible
  backlog that better final ranking could recover from this run.
- Overlapping failures among the 354 new identities: 277 similarity, 170 QED,
  129 SA and six inherited medchem failures. These are endpoint exclusions, not
  hard masks on primitive intermediates.
- Pendant constructions completed 54/54, but only two outputs were eligible.
  Fused constructions completed 17/21, with one eligible output, the new -11.0
  best. The older `build_ring_system` program completed only 3/10, none eligible.
  These counts measure this adaptively allocated sample, not inherent rates of
  the options or a justification to remove their support.
- All 15 sampled descendants of the -11.0 candidate were ineligible. These
  include three same-round continuations before its docking label and twelve
  next-round outputs from two workers. No claim of an unrecoverable dead end
  follows from this small sample.
- In 45/64 local branch decisions neither branch contained an eligible completed
  state, so selection used the declared constraint-deficit repair fallback,
  not a docking-based terminal value. Only 19 decisions used terminal value.
  The selected continuation state was a first-option result in 53 cases and a
  lookahead result in 11; four selections used the exploratory-state draw.
- Twelve newly docked candidates came from guided follow-ups; their best was
  -9.5. The -11.0 candidate was a first-option proposal. This is not a matched
  estimate of the effect of lookahead.
- Frozen prior-only predictions reproduced on all 40 new dockings. Their MAE
  was 0.530 kcal/mol versus 1.092 for the prior-label mean. The guide has local
  ranking information, but this adaptive sample does not calibrate extrapolation.
- The final surrogate predicts the diagnostic IVG endpoint at -8.748 and our
  best at -9.939. Planning uses the observed -11.0 for that already docked best.
  The IVG endpoint was not docked in this episode; its external score has not
  been reverified by this analysis.

## Molecular graph comparison, not SMILES string distance

| Property | Current best | Diagnostic IVG target |
| --- | ---: | ---: |
| Heavy atoms | 36 | 30 |
| Graph cycle rank | 6 | 5 |
| Composition | C29 N4 O3 | C25 N2 O3 |
| QED | 0.602 | 0.731 |
| SA | 3.947 | 2.757 |
| Similarity to original seed | 0.420 | 0.475 |

Cycle rank is E-V+1 for these connected molecular graphs, not ring-system
count or a choice of perceived ring basis. The winner does not require simply
more rings or a larger molecule than the incumbent.

The current best retains the dimethylaminomethyl branch and appends a different
peripheral arrangement. The target instead has an ethylene-linked carbon-only
fused peripheral skeleton, a specifically located peripheral carbonyl and a
different carbonyl-bearing core ring. None of the final 511 pooled molecules
matches any of the five completed saved-route stages. None matches the precise
peripheral ketone or modified-core queries recorded in the JSON. Five match the
broader tetralin query, all ineligible. These are substructure coverage findings,
not a computed shortest distance or proof that this is the only useful chemistry.

The existing answer-known 21-primitive witness proves executable construction
for this target in its pinned environment. All five completed stages satisfy
endpoint constraints. The current production option interface has pendant/fused
ring and carbonyl programs, but no compound linker-remodel option. A four-edit
linker change still relies on coordinating ordinary shrink/grow actions. Source
inspection also shows the new lookahead scores completed branches, while the
next option and its internal site/chemical choices remain reference proposals.

## Interpretation and next proposed action

The evidence locates a proposal/continuation bottleneck rather than an inability
to execute rings or a large backlog blocked by oracle ranking. The improvement
does not establish a task-aware planner that identifies which coordinated
replacement will have a high future docking score. The present algorithm
compares small random branch samples; when they are infeasible, most decisions
revert to immediate constraint repair.

Broaden coordinated branch/linker replacement and coverage of attachment and
carbonyl positions, not the ring vocabulary or executor support. Preserve the
local/global region interface and generic edits. Any change in proposal or
allocation law must be declared as a new development recipe, not represented
as a probability-preserving acceleration of the completed run.

The smallest useful additional oracle diagnostic would compare the five saved
completed route stages, including the endpoint, and the current best under one
matched docking setup. Check for compatible cached scores first. Keep this
answer-known panel and its labels separate from winner-blind training and search.
It would distinguish a misleading task surrogate from a path that genuinely
requires delayed objective improvement, without another broad optimizer run.
Neither the panel nor a controller change is launched or authorized by this note.

## Verification and limitations

The analysis ran in 9.74 seconds excluding imports. It verified physical
collection hashes, sealed payloads, 76 imported production source hashes against
the deployed revision, worker-parent identities, all saved branch values, all
40 prior docking predictions and endpoint properties, and canonical pool
uniqueness. Analysis Ruff formatting and lint checks passed using explicit
`compose_v4` first-party import classification. No broad suite was run.
These local analysis files are not committed; no optimizer code changed.

Reproduce using the episode source and matched chemistry:

```sh
PYTHONPATH=/private/tmp/compose-t4-chemistry.hizM8Y:/private/tmp/compose-macro-lookahead/src:. \
  OMP_NUM_THREADS=1 .venv/bin/python diagnostics/t4_macro_lookahead/diagnose.py
```
