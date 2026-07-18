# Phase 0 validity-runtime results

Run on 2026-07-14 with seed `20260714`:

```json
{
  "programs": 580,
  "committed_rewrites": 10020,
  "exact_forward": 580,
  "exact_reverse": 580,
  "canonical_permutation_matches": 580
}
```

The fixture mixture includes acyclic, charged, triple-bonded, monocyclic,
fused aromatic, fused heteroaromatic, bridged, and spiro molecules. Every
forward and inverse intermediate passed the same production validity boundary.

This result establishes the Phase 0 execution invariant. It does not establish
distribution learning, novelty, or sample quality.
