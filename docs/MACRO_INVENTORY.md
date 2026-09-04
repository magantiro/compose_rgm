# Macro / program option inventory

Read out of `src/compose_v4/control/macro_engine.py` on 2026-09-04, not from
memory. This is the vocabulary the three-level controller draws `o` from.

## Option families and their primitive support

| option | primitive support | contract |
|---|---|---|
| `local` | atom_restate_semantic, bond_reorder | — |
| `grow` | atom_insert | — |
| `append` | atom_insert | closure must reach among NEWLY ADDED atoms |
| `scaffold_extend` | atom_insert | backbone-capable atoms only; no terminal decoration |
| `decorate` | atom_insert | terminal substituents (halogens) AFTER construction |
| `cyclize` | cycle_close, **bond_insert** | — |
| `append_system` | cycle_close, bond_insert | closure forms a NEW ring system from grown material |
| `annulate` | cycle_close, bond_insert | closure fuses INTO an existing ring system |
| `small_ring` | cycle_close, bond_insert | form a 3-4 membered ring |
| `aromatize` | atom_restate_semantic, ring_system_restate, bond_reorder | reorganise bond orders; no atom-count change |
| `restate` | atom_restate_semantic, bond_reorder, ring_system_restate | — |
| `open` | cycle_open, bond_reroute | — |
| `rebuild` | cycle_open, atom_insert, bond_reroute | — |
| `shrink` | atom_delete, cycle_open | remove material, preserve the core |

`MACROS` (the exported tuple) lists ten: local, grow, cyclize, aromatize,
rebuild, restate, open, append, scaffold_extend, shrink. Four more exist in
`MACRO_FAMILIES`/`MACRO_CONTRACTS` but not in that tuple: `append_system`,
`annulate`, `small_ring`, `decorate`.

## Multi-step programs

    BUILD_RING_SYSTEM   = scaffold_extend x8 -> append_system x1 -> restate x2
    REFINEMENT_BRANCHES = none 0 | light 1 | conjugate 3 | deep 5

## Support-restriction machinery already implemented

    proposal_support(families, probs, cap=300, floor=20)
    macro_action_distribution(families, probs, support, macro, clean,
                              temperature=2.0, epsilon=0.15)
    backbone_only(before_smiles, composition="mixed")     BACKBONE_ELEMENTS {C,N,O}
    append_system_closure(before_smiles, min_new=6)       APPEND_MIN_NEW_ATOMS
    disjoint_closure_only(before_smiles)
    ring_systems(mol), synthesis_preference(smiles)
    TERMINAL_ELEMENTS {F, Cl, Br, I}

## Why this layer exists — measurements recorded in the source

* **Ring formation is not `cycle_close` alone.** On a real six-carbon precursor
  the state offered ZERO cycle_close actions and 127 `bond_insert`, two of which
  close a six-membered carbocycle. A cyclize macro scoped to cycle_close could
  never find the hexagon in front of it.
* **`atom_insert` is halogen-heavy.** Of ~9 atoms added across 80 grows, only
  ~2.6 were carbon and ~2 were halogens, which are terminal and cannot extend a
  chain. An epsilon A/B showed this is R_theta's own insert law, so it is fixed
  by restricting the macro's SUPPORT, not by retuning exploration.
* **Pendant rings need enough new chain.** A pendant ring must close among newly
  added atoms; the simplest IVG pendant needs 8, and grow horizons of 3/4/6
  meant every legal closure reached back to the scaffold and could only annulate.

## Relevance to the current T4 result

The variable-scope region controller proposes over RAW primitives: it filters by
region membership in `admissible_indices` and applies no macro support
restriction. Measured on the parp1 seed-0 law: `atom_insert` carries 0.457 of
R_theta's mass while `cycle_close` carries 0.00007, and 40 docked molecules
showed halogen/sulfur decoration with 1 ring-system change. Those are precisely
the biases the macros above were built to counter, rediscovered by bypassing
them.
