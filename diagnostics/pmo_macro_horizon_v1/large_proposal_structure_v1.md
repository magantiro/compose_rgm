# What the large generic proposals actually do

Zero oracle calls. Every persisted large proposal (>=17 primitives) compared to its own
source by MCS: atoms outside the common substructure are changed, connected components of
that set are the changed regions. Same measure for the required macro (source -> witness).

| measure | required | mods=3 | mods=8 | |
|---|---:|---:|---:|---|
| d_heavy | +6 | +3 | +5 | matched |
| d_rings | +1 | 0 | +1 | matched |
| n_changed_regions | 2 | 1 | 2 | matched |
| **largest_changed_region** | **19** | 6 | **7** | **2.7x gap** |
| **retained_fraction** | **0.30** | 0.67 | **0.64** | **inverted** |

medians; n = 5 required, 77 mods=3, 200 mods=8.

**These are not scattered local edits.** The region COUNT is already right. They are one or
two coherent regions that are too small -- nibbling the periphery instead of replacing a
large connected chunk. Raising the module budget adds a SECOND region rather than enlarging
the first, which is exactly why it buys scale without buying coherence.

**Design spec for the jump lane:** change ONE region of ~19 atoms, releasing ~70% of the
source. Not "make bigger programs" -- make the changed region bigger.

**Correction.** An earlier claim that the required macro needs +13 heavy atoms and +2 rings
while proposals deliver +5 and +0 was one segment's numbers quoted as the general case.
Across all five required macros the medians are +6 and +1, which the generator nearly
matches. Heavy-atom and ring deltas are NOT the discriminator.
