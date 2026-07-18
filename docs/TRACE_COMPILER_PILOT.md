# Broad-corpus trace compiler pilot

## Result

The deterministic micro-trace compiler was audited on the first 1,000 entries
of a held-out GuacaMol subset on 2026-07-14.

```text
read                                  1,000
parse/vocabulary failures                 5
eligible molecular graphs               995
exact forward-and-reverse programs       985
eligible exact round-trip rate        98.995%
committed forward/reverse rewrites     61,600
invalid visible commits                    0
```

The five representation failures were out-of-vocabulary Se/Si molecules and
one radical encoding that violates the inherited atom-state vocabulary.

## The useful failure

All ten compiler failures in the pilot expose the same base-construction
limitation: valence-six sulfur centers such as sulfones can require a transient
hydrogen cap above the current `H = 0..4` atom-state vocabulary when a strictly
connected, one-parent spanning-tree construction is used.

This is not an executor validity failure and should not be hidden with repair.
It is evidence about path expressivity. Three principled options remain:

1. Begin the first learning experiment on standard-valence C/N/O/F chemistry,
   where the micro compiler is closed.
2. Add a verified multi-site transaction for hypervalent bridge centers.
3. Enlarge the transient state vocabulary only if the added capped species are
   chemically defensible, rather than merely RDKit-sanitizable.

The project chooses option 1 for the first base-model test. Options 2 and 3 are
deferred until learning works. This preserves the minimal experiment while
recording exactly where richer stochastic rewriting semantics may be needed.

## Interpretation

This pilot is stronger than the small fixture suite and weaker than the Phase 1
exit gate. It shows broad coverage and locates a coherent 1% path-language
failure. It does not justify claiming universal molecular trace compilation.
