# Executable Region Resampling — pivot plan

Branch `region-resampling`, opened 2026-08-31 off
`codex/editing-v2-successor-fiber-fastpath` @ 2c80015.

## Why pivot

The August diagnostics (`docs/DIAGNOSTIC_CAMPAIGN_2026_08.md`) closed the
bottom-up route. Three facts, all measured:

1. `R_theta` is not the bottleneck for T4 heteroatom chemistry.
2. A fixed global operation is simply wrong for most states (20/22 dev cells
   have zero feasible endpoints anywhere in the growth basin).
3. Some supported transformations have catastrophic primitive path mass
   (7->8 expansion: 1e-8 to 1e-14 over 2-3 valid edits).

Together these say the search layer needs **adaptive scale** and **shortcut
proposals** — not another macro, and not a stronger narrow controller.

## The object

Choose a connected mutable region `M` of the current molecule; let `C = x \ M`
be the preserved context and `dM` its boundary attachment. Search scale is set
by how much of the molecule is released:

    one atom        -> tiny local edit
    a substituent   -> functional-group replacement
    a linker        -> linker redesign
    a ring system   -> ring replacement / expansion / composition change
    half a scaffold -> global scaffold change

Crucially, the replacement is realized as a path of COMPLETE VALID MOLECULES on
the COMPOSE graph:

    x = x_0 -> x_1 -> ... -> x_T = y,    x_t in X for all t

not through a masked intermediate representation. That is the difference from
fragment-remasking approaches, and it is the property worth a theorem.

## Where the existing work lands

Semantic macros are **not** the global search space. They become fast compilers
for particular region replacements — low-variance shortcuts for transformations
the generic mechanism could realize but whose primitive kinetics are hopeless.
Finding 6 of the campaign is the motivating example.

## Two axes, both measured

    D_struct(x,y)   structural displacement (sigma_0/sigma_1 change, scaffold
                    edit distance, fraction of local environments changed)
    D_R(x,y)        kinetic difficulty, min over paths of -sum log R_theta

"Global" is NOT "many edits": 7->8 expansion is few edits and effectively
unsamplable. The four quadrants are different reasons to need temporally
extended proposals.

## Build order

1. Region selection and boundary bookkeeping over the executable graph.
2. Valid replacement compilers for the three generic cases:
   one-boundary (substituent), two-boundary (linker/segment),
   cycle-containing (ring system).
3. Population/SMC on top, with a deliberately tiny scale selector
   (local | regional | global) and a nonzero exploration floor —
   sparse credit already burned us once.
4. Donor regions from the archive as a second proposal source, giving
   crossover-style recombination without a separate GA mechanism.

## Coverage stress test, not imitation

Characterize the MOVES of strong optimizers (Graph GA, GenMol fragment
remasking, MolLEO, f-RAG) — minimal connected changed region, scaffold fraction
replaced, ring-system change, dHeavy, dRings, 1-Tanimoto — and compare the
distribution to COMPOSE's. The question is whether our proposal family spans the
same local->global range, not whether we can copy their winners.

## Theory targets

- valid-state support preservation under region replacement
- canonical successor / trajectory fibers (already established)
- proposal invariance under overlapping local and global mechanisms
  (mixture density, per docs and memory)
- constructive reachability for one-cut, two-cut, and ring-system replacement
