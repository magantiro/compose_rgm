# Part 5 — §5 an authentic executed trace, and §6 what each ablation removes

## §5 — one real executed program, dependency layer

OBSERVED EXECUTION (not reconstructed). Source: scored PMO campaign
`scored_ranolazine_mpo_seed20274925_20260924T202214Z` (kosha-labs
`compose-pmo-fibercontrol-replication`), `campaign/round_0031/pending.json`,
`batch.candidates`. Channel `structured_program_channel`.

    SOURCE   (40 heavy)  CC(C(=O)C=C1Nc2ccc(NCC(OCC3CCCC3C)C3CC(F)C(C(=O)O)C3)cc2S1)C1=C(O)C=CC1
    ENDPOINT (35 heavy)  CC(CNC1=CCC(SC(N)=CC(=O)C(C)C2=C(O)C=C3NCCNC32)=C1)OCC1CCCC1C

15 primitives, 16 states, 3 blocks, 13 bound input handles
(`assignment` = source slots `[31,33,34,38,39,37,35,36,30,29,8,9,24]`).

| # | rule | operands | what it does |
|---|---|---|---|
| 1-8 | `atom_delete` | `input 0..7` | **block `substituent_delete`** — excises the fluoro-cyclopentane-carboxylic-acid substituent (source slots 31,33,34,38,39,37,35,36) |
| 9 | `atom_insert` | N onto `input 8`, slot `created 0` | **block `construct_substituted_ring`** begins: anchor a nitrogen on source slot 30 |
| 10 | `atom_insert` | C onto **`created 0`**, slot `created 1` | extends the chain from the atom born one edit earlier |
| 11 | `atom_insert` | C onto **`created 1`**, slot `created 2` | extends again |
| 12 | `atom_insert` | N onto **`created 2`**, slot `created 3` | completes an N-C-C-N chain |
| 13 | `cycle_close` | `a=input 9`, `b=`**`created 3`** | **closes the ring** between source slot 29 and the LAST atom of the chain built in 9-12 |
| 14 | `atom_delete` | `input 10` | **block `ring_path_remodel`** |
| 15 | `cycle_close` | `a=input 11`, `b=input 12` | second closure on source slots 9 and 24 |

**The coordination, in molecular terms.** Marks 10-12 each name the atom *born in the
immediately preceding edit*, so four inserts build one connected N-C-C-N chain rather than
four unrelated atoms. Mark 13 then consumes `created 3` — the chain's terminal nitrogen — and
bonds it back to a retained source atom, **completing a piperazine-like ring that marks 9-12
only prepared**. The endpoint carries that ring: `…C=C3NCCNC32`. No single primitive edit
expresses this; the closure is only legal because the four preceding inserts put an atom in
reach, and the handle system is what lets mark 13 name it.

Atom bookkeeping: 40 - 8 (block 1) + 4 (block 2 inserts) - 1 (block 3) = **35**, matching the
decoded endpoint. Peak never exceeds 40, which is what `_topological`'s
`atoms + peak_delta <= 40` guard enforces.

Block footprints are disjoint on inputs (0-7 / 8-9 / 10-12), so `compile_program_graph` finds
no write-write or write-read overlap and emits no serialization edges; the blocks therefore
run in observed order under the default priority.

MISSING TRACES, marked rather than reconstructed: no saved `MacroRollout` with per-step
`r_theta_prob` / `within_family_rank` was located for the option/QED family, and no fragment
`EditProgram` trace was pulled. Those §5 rows are outstanding.

## §6 — what each planned ablation actually removes

### PMO: structured programs vs length-matched legal chains

Changes **together**:
1. **joint multi-module planning** — the K-module composition under a capacity-aware count law;
2. **dependency-aware construction** — typed handles, producer/consumer edges, hazard
   serialization, peak-capacity reservation (the mechanism §5 demonstrates);
3. **catalogue / donor information** — the jump lane's teacher-derived plan library
   (`fit_scope: shared_all_routes`) and transplant donors drawn from the run's own archive;
4. **program reuse** — `ProgramOptimizer._recombine` / `_mutate` over archived programs.

Does **not** change:
- **learned scoring** — absent from both arms on this path;
- **anchor sharing** — the shipped arm never had it (`focus_policy` reaches no caller).

So it is a *joint planning + dataflow scheduling + catalogue* ablation, not an isolated
coordination ablation, and it must be reported that way.

### Fragments: the reference ablation

Unlike PMO, the fragment path **does** consume R_theta — `run_fragment_constrained_suite.py`
calls `load_factorized_rollout_checkpoint(ringcore_a7546e2_best.pt)` and samples marks per
event (`mark_attempts_per_event`). The probability-dependent stages that must change are
therefore the per-event mark law itself; everything else — the region lock, the declared
attachment interfaces, the arm setting (`frozen_sampler_baseline` / `attachment_control` /
`path_program`), the official 100-generations-per-drug protocol and the unweighted 10-drug
task mean — must be held fixed.

Two flags belong with any fragment ablation:
- the pinned checkpoint is `PROVISIONAL_EDITING_CHECKPOINT`, which
  `configs/ringcore_v1_checkpoint_selection.json` **forbids for frozen results**;
- `scaffold_morphing` reuses the linker prompts AND the linker results, so it is not an
  independent row.

## Supported scope of the shared-process claim

| level | what is actually shared |
|---|---|
| **primitive execution** | ALL task families — one Active8 / editing-V2 executor and codec; every committed state is a complete connected molecule by construction |
| **program representation** | PMO, T4, fragments — the same `EditProgram` + `ProgramGraph` dependency layer |
| **constructors** | PMO and T4 share `synthesize_dynamic_program` verbatim; the option/QED family shares `macro_engine` instead |
| **learned parameters** | NOT shared. De novo uses `lineageB/checkpoint.best_so_far.pt`; fragments use `ringcore_a7546e2_best.pt`; the option/QED family scores R_theta per step; **PMO and T4 use no learned parameters in proposal construction at all** |

So "one shared validity-closed generative process" is supported at the level of **primitive
execution and program representation**. It is NOT supported as "one learned law drives every
task": three families consume a learned reference, two do not, and the two that do use
*different checkpoints*. UNRESOLVED: which checkpoint the option/QED family loads at runtime —
not traced.
