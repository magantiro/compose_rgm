# REFINE_RING — frozen controller decision (2026-08-27)

## The question

Can production `restate2` — which acts anywhere in the molecule — realistically
refine the ring it just built (dearomatise it, then retype a ring carbon to N)?
Or does that require a scope operation?

## The causal gate

`diagnostics/restate_gate.json`. n = 2000 per arm, from the narrow-B linked-ring
product `FC(F)(F)c1cc(N2CC[NH2+]CC2)ccc1-c1ccccc1`, new-ring atoms 16–21.

| endpoint metric | P: production `restate2` | U: unscoped control | S: ring-scoped |
|---|---|---|---|
| **gained a NEW saturated-N ring** | **1.60 %** | 1.70 % | **39.40 %** |
| dearomatised an aromatic ring | 26.65 % | 25.55 % | 53.15 % |
| piperidine present | 4.35 % | 4.70 % | 29.80 % |
| distinct endpoints | 1605 | 1634 | 688 |

**24.6×** on the decisive metric. P ≈ U rules out the SMILES round trip and any
production-plumbing artefact as the cause: it is the scope.

Two readings that must NOT be repeated:

- `gained_pip` is NOT the headline. The substrate already contains a piperazine,
  so generic restating can mutate THAT ring into something counted as a
  piperidine without ever refining the newly built ring. `gained_sat_n` — a new
  saturated-N ring — is the honest column.
- 688 distinct endpoints from 2000 scoped draws is the evidence that this
  narrows the LOCUS and does not prescribe the chemistry. Had it collapsed to a
  handful of products it would have been a disguised piperidine macro.

## What was implemented

`refine_ring()` and `action_locus()` in `src/compose_v4/control/macro_engine.py`.

    admit a  <=>  declared_locus(a) & R != {}

R is the EXACT constructed cycle: `fr.path` for linked, `fr.path + (u, v)` for
fused — 4 new atoms + 2 fusion atoms. R_theta renormalises over that local
fiber through the production normaliser (`macro_action_distribution`). The LAW
is untouched; only its DOMAIN is.

`prefer_fn` cannot express this. It is a predicate on the PRODUCT that relaxes
to the bare gate after twelve failed draws, so an action R_theta ranks at 30 is
never reached. A scope that relaxes under pressure is a preference, and a
preference is what already fails.

Refinement runs INSIDE the builder, never in the caller: the caller receives
SMILES, and one canonicalisation renumbers atoms so `fr.path` would name a
different ring.

## Bug found and fixed by the 3-ring-type smoke

The first fused implementation scoped to `ring_block_atoms`, whose flood fill
walks ANY bond joining two on-ring atoms. On
`FC(F)(F)c1cccc(N2CC[NH2+]C3CCCCC32)c1` the piperazine N and a benzene carbon
are both on rings but joined by an ACYCLIC bond, so it crossed and returned all
16 ring atoms — two independent ring systems merged. REFINE_RING then retyped a
scaffold benzene carbon (`c1cccc` -> `c1cncc`): exactly the all-molecule
behaviour it exists to remove.

Ring-BOND traversal (`ring_system_atoms`) fixes the biaryl case but still
returns the whole bicyclic system for a ring fused onto a benzene. The fix is to
use build provenance directly and no traversal at all. `ring_block_atoms` was
left untouched — `aromatise_ring` depends on its deliberate breadth.

    "ring atom" adjacency  !=  "ring bond" connectivity

## Contract smoke, 3 ring types x 3 seeds — PASS

`diagnostics/ring_type_smoke.json`.

| check | result |
|---|---|
| locus == constructed cycle == 6, off-cycle 0 | OK on all 8 OK builds |
| every admitted action declares locus in R | 25/25, 13/13, 56/56, 44/44 |
| no admitted action with locus disjoint from R | 0 |
| no atom retype outside R | 0 |
| no connectivity change outside R | 0 |
| no change outside the constructed ring's ring system | 0 |
| `refine=0` byte-identical to the archived builder | 0 failures / 9 |
| `coupled_omissions` | 18, fused-aromatic only |

`coupled_omissions` are NOT failures. In a fused aromatic system the constructed
cycle and its parent share one pi block, so restating a PARENT-ring atom changes
bond orders throughout, R included. Measured: the sources are parent-RING atoms
only ([4,5,6] and [7,8,9]), and fused/SATURATED — not electronically correlated
— omits zero. REFINE_RING correctly EXCLUDES those actions. The fiber is
conservative there by design.

    scope of the chosen edit != all atoms whose representation changes

`fused/6/C/aromatic` is UNSAT on the 5HT1B seed under BOTH old and new code
(byte-identical), so that is a pre-existing builder limitation, not a REFINE_RING
failure. It builds fine on phenol and acetanilide.

## Frozen configuration

`refine = 2` is the PROSPECTIVELY FROZEN production controller, on the evidence
above. `refine = 0` is a PREDECLARED ABLATION. The official T4 and PMO runs must
not be used to choose between them — that would turn the benchmark into
development data.

Memoisation keys on `refine` AND `refine_seed`; refinement is stochastic and a
seedless key would freeze one draw and hand every particle the identical ring.

## Do not redo

- The 2000-arm causal gate. It was run on the linked ring, whose scoping passed
  every check. Its 24.6× result stands.
- `ring_block_atoms`. Load-bearing for `aromatise_ring`; changing it reintroduces
  `kekule_missing`.
- The `refine=0` parity check needs the archived builder staged back:
  `cp archive/macro_engine.pre_refine_ring.py src/compose_v4/control/_macro_engine_prerefine.py`
