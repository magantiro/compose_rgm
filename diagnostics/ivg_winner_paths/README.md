# IVG winner-path audit: executable chemistry and a region-filter defect

Completed 2026-09-09. Diagnostic only: no controller modification, model fitting,
learned-law enumeration or new docking. All inspected winners are development
material. The paths use their targets as search hints and are not blind recovery.

## Findings

The bounded search found and exactly replayed 82/91 distinct seed–winner pairs
(90.1% coverage), preserving all 101 cell/run/co-best identities from the pinned
census. Conditional witness correctness is 82/82 exact 2D endpoint replays.
Every saved state was produced by the existing Active8 semantic executor;
intermediates were not reconstructed from SMILES. The limit remained 40 active
atoms in 48 persistent slots, connected non-null states and frozen charge policy.

| Target | Replayed paths / pairs | Other outcomes |
| --- | ---: | --- |
| PARP1 | 17 / 18 | 1 unsupported valence representation |
| JAK2 | 19 / 20 | 1 unsupported valence representation |
| BRAF | 16 / 16 | None |
| FA7 | 18 / 18 | None |
| 5HT1B | 12 / 19 | 5 charge changes prohibited; 2 mappings unresolved |

The two unsupported endpoints contain neutral `[N]` configurations rejected by
the frozen valence vocabulary. They were not silently neutralized or restated.
For two other targets, both sampled mappings violate charged-center preservation;
this does not prove that every possible mapping fails. Full reasons and original
source identities remain in the pair receipts.

37 witnesses use at most 16 Active8 edits; 45 are longer. Found lengths range
from 5 to 47 and are upper bounds, not proven shortest paths. Four endpoints
have a certified lower bound above 16 from atom count and positive cycle-rank
change. This concerns one 16-edit horizon from the original seed, not cumulative
reachability over multiple optimizer rounds or from a different warm parent.

## Concrete controller defect

All 17 `RingSystemRestate` actions in the recovered witnesses have nonempty
`changes`, but production `control/region_rewrite.py:touched_slots` reports an
empty footprint. It reads top-level `a,b,u,v,slot` and insertion neighbors, not
the endpoints nested in `RingSystemRestate.changes`. `admissible_indices` then
rejects the empty reach as `not_in_locus`, for every region.

This is a source-inspected and reproduced implementation defect, not an inferred
failure of controller mathematics. `review.json:composite_action_footprints`
records all 17 expected endpoint sets, observed empty sets and zero admitting
regions, bound to exact source and receipt hashes. The remaining 65 witness
paths pass this particular touch filter at every step, but this is not proof
of full macro contracts, learned-fiber support or positive controlled probability.

The filter has NOT been repaired in this audit. The scoped next repair is to
recognize the typed composite action's actual bond endpoints while preserving
frozen-context checks. Do not bypass the filter, broaden regions, change kappa
or assume that a verified executor mark already exists in the learned law.

## PARP1 seed0, d=0.4: actual paths to all three released winners

All three original-seed paths reconstruct the supported 2D endpoint exactly.
The source has 19 heavy atoms, cycle rank 3 and one ring system.

| Pair ID prefix | Witness edits | Certified lower bound | Endpoint cycle rank / ring systems | Infeasible prefix lengths |
| --- | ---: | ---: | ---: | --- |
| `5705946f25f3` | 23 | 13 | 5 / 2 | 11–22 |
| `b776cfbb4080` | 25 | 15 | 6 / 3 | 11–24 |
| `76f129c70c87` | 27 | 17 | 6 / 3 | 12–26 |

Each endpoint passes the locally recomputed T4 constraints. Each saved path
passes through a long feasible → infeasible → feasible excursion, and its
state after exactly 16 edits is infeasible. The full census has 53 such
excursions among 82 witnesses. All 82 have an infeasible intermediate and a
feasible endpoint, but some start infeasible; those are not automatically
counted as a valley.

For example, the 23-edit witness uses 3 deletions, 14 one-neighbor insertions,
one bond reorder, one ring-system restatement, one ring opening and 3 closures.
The final molecule is:

```text
O=C1CCCc2cc(CCc3ccc4c(c3)CNC(=O)c3cccn3C4=O)ccc21
```

Its ring-restatement action, after prefix length 15, changes bonds among saved
slots `{4,5,6,7,8,18}`. The production touch extractor reports `{}`. All 175
enumerated regions at that state are consequently rejected for this action.
The pair receipt contains the exact source tensor, V4 action sequence, every
state tensor, canonical path and per-step properties. `review.json` contains
readable canonical paths and rule sequences for every recovered pair.

Interpretation: the executor can realize substantial ring chemistry, but these
particular witnesses expose both a disabled electronic-rewrite channel and a
delayed-feasibility problem. They do not establish that the optimizer can discover
the same routes. Longer witnesses do not prove that other shorter routes fail;
only the stated lower bounds certify a horizon exclusion. A sensible next
diagnostic is pinned-runtime replay plus option/region and canonical-successor
support on one witness, after the footprint repair, not another blind long run.

## Value and runtime abstentions

Local RDKit is 2026.03.6; the frozen Modal runtime is 2024.03.5. These are local
production-executor witnesses, not yet pinned-Modal-runtime replay evidence.
Endpoint constraints are local recomputations, not new upstream labels.

The existing 51-label PARP1 value snapshot was hash-verified, but its stored
training fingerprints did not all reproduce under the local RDKit runtime.
Accordingly no task-value predictions were used or reported. Other cells also
abstain because no applicable frozen snapshot was supplied. The previous blind
uncapped lookahead result remains separate in
../t4_uncapped_lookahead_probe/README.md: 32 completed rollouts, no task-dependent
decisions, no feasible candidates, and zero new docking calls.

## Compute, interruptions and verification

- Saved pair searches total 63.596 seconds on one local CPU worker. This is
  molecular search time, not total engineering time or end-to-end elapsed time.
- An initial reporting bottleneck repeatedly mapped each molecule for every
  region. Two retained real-path annotations fell from 17.032 to 0.533 seconds
  and from 77.102 to 1.708 seconds, with all output rows equal. Production
  controller code was not changed to achieve this diagnostic speedup.
- The first slow pass was stopped after two complete receipts. Those were
  converted with unchanged search/executor dependencies and full annotation
  equality. A later NumPy integer serialization failure stopped publication
  after 83 complete pairs. Those 83 were hash-checked, replay-checked and reused,
  not molecularly searched again. Only the remaining eight pairs were resumed.
  Work interrupted before publication is not included in saved-unit timings.
- The final resume took 21.745 seconds including reuse verification and the
  remaining searches. Peak RSS was 481,951,744 bytes. No accelerator was used.
- Clean worktree `/private/tmp/compose-winner-paths.W1n7sh`, scientific producer
  `54bf6f2bfd7f` (full identity in audit.json); unchanged path-search code from
  `88007a2`. The annotation-only intermediate revision was `17b90b2`.
- Strict preflight passed with zero mounted-source drift. Seven focused tests
  passed in 4.19 seconds at `54bf6f2`, including exact path replay, pendant/fused/
  expansion examples, charge/size rejection, incomplete search, serialization
  and coordinate-transport parity. Receipt: focused_tests_54bf6f2.xml.
- Ruff lint and format checks passed for all four new Python files. The final
  review validated 170 input file hashes and was byte-identical on two reductions.
  No unrelated repository-wide suite was run; this is not a release milestone.
- Unrelated in-progress scaffold/model edits were excluded and left untouched.

## Authoritative artifacts and reproduction

`audit.json` binds the census, original seed registry, frozen snapshot, software,
configuration, scientific source closure and all 91 primary pair receipts.
`pairs/*.json.gz` preserves exact paths and prior conversion inputs. Canonical
deduplication did not discard run/cell/tie provenance. `review.json` is a
deterministic, input-hashed reduction, not another molecular search.

- Audit SHA-256: `3b7fb79a541b94dd5421f30ffc0516c5cdc4d9316845cd40561d799ec517f922`
- Review SHA-256: `2f9fdd4c6baddb08e727e14064c67ffc1c704ff94540e686cf9fa28fd35e7d26`
- Focused-test SHA-256: `eaa5694b77e5fd028c746994c8fa2b472ef3be9fe48bc23a7c9fd6644d294044`

From the clean scientific worktree, use the workspace's Python environment and
the absolute input/output paths recorded in audit.json:

```sh
PYTHONPATH=src python tools/ivg_winner_paths.py --census /path/to/census.json --snapshot /path/to/candidate_lock.json --output /path/to/ivg_winner_paths
```

For a reporting-only reduction, use the committed projector:

```sh
PYTHONPATH=src python tools/ivg_winner_path_review.py /path/to/ivg_winner_paths
```

No templates, motifs, proposal weights or predictor parameters were extracted
from these winners. No inference of held-out performance or superiority to IVG
is authorized by this audit.
