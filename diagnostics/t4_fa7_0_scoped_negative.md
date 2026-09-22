## 2026-09-21 (T4 fa7_0 at delta=0.6: REACHABLE but not reached; the barrier is the goal-abstraction layer)

Written for merge into `.claude/context/learnings.md`. Kept in the `t4_fa7_0_*`
namespace rather than appended directly, because other agents were editing the
shared file concurrently.

All numbers MEASURED under the pinned kernel (python 3.11.13 / rdkit 2024.03.5 /
numpy 1.26.4 / torch 2.4.0), `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1`.
ZERO oracle calls, zero docking, zero Modal across every result below.

### The headline: fa7_0 is reachable, and the yield does not clear a rescue bar

- **The cell is NOT infeasible.** All five stored witnesses pass the unmodified
  production `Fiber.check`, and a production configuration independently
  GENERATED an eligible endpoint from the real seed. So "delta=0.6 admits nothing
  for fa7_0" is false and should not be written.
- **It is also not solved.** Across 3 seeds x 480 draws with the region law on,
  the completion law produced **one** eligible endpoint, at one seed. That is
  not reproducible seed-to-seed and does not justify docking calls.
- Report it as a scoped negative: **reachable, not reached at this budget**, with
  the barrier located rather than guessed.

### The one generated molecule, and why the qualitative result beats the count

  OFF arm, 1,440 draws, ONLY endpoint:
    C=c1ccc(CN(CCC(C)C)C(=O)c2cccc3ccccc23)cc1=C
    sim 0.6897 / QED 0.6422 / SA 2.5254, heavy delta -5
    an exocyclic quinoid that has DELETED the amidine -- the FA7 S1-pocket
    pharmacophore. ALERTS 2 -> 0, which is exactly where its QED came from.
  ON arm (completion law), ONLY endpoint:
    COC(=O)N(CCC(C)C)Cc1ccc2ccc(C(=N)N)cc2c1
    sim 0.6610 / QED 0.6264 / SA 2.331, heavy delta -8
    retains BOTH amidine and carboxamide; passes under benchmark_only,
    compose_valid AND legacy_screened; round-trips through the executor.

**Benchmark eligibility and chemical plausibility are two numbers and they
disagree here.** The unconditioned path's only success is a molecule nobody would
defend; the conditioned path's only success is a credible one. State the kind of
molecule beside the count, or the count misleads in both directions.

### Where the barrier IS: the goal-abstraction layer

`expand` does not gate the molecule a proposal law conditions. It abstracts the
synthesized program with `extract_structural_goal`, expands `_variants`, re-binds
each subgoal with `attachment_bindings(...).assignments[0]`, and gates whatever
`instantiate_goal` builds.

  **`recovered_fraction = 0.0000`** over 150 draws -- the program's own endpoint
  appeared among the gated endpoints ZERO times.

So a law that ranks the module endpoint improves a molecule the campaign never
scores. That is the barrier, and it is architectural rather than chemical.

### TWO INERT-REPAIR TRAPS, and the rule that covers both

1. Conditioning the MODULE endpoint is inert because the campaign scores
   something else (`recovered_fraction = 0`).
2. Conditioning the GATED endpoints is inert for the opposite reason: `expand`
   keeps EVERY eligible endpoint (`found[gate["smiles"]] = {...}` on each pass,
   no budget, no top-k). There is no selection, so a ranking cannot change what
   it returns.

**RULE: a proposal law bites only if BOTH hold -- the ranked object IS the scored
object, AND something downstream DISCARDS candidates.** Verifying the identity is
necessary and NOT sufficient. Check for a discard before building a ranker; it is
one read of the consuming loop and it is cheaper than the law.

### Placement is a REAL axis pointing the WRONG way (do not spend on it)

`attachment_bindings` enumerates up to 128 bindings per subgoal and the loop used
`assignments[0]`. Measured over 57 real fa7_0 goals / 101 subgoals:
**24.8% carry more than one binding** ({1: 76, 2: 21, 4: 3, 6: 1}).

Widening to four bindings, 240 draws, wrapped gate, 0 gate disagreements:

    narrow (k=1)   3,568 distinct endpoints offered    0 eligible
    broad  (k=4)   3,961 distinct endpoints offered    0 eligible
    ADDED 393 (+11.0%)   LOST 0   ADDED-AND-ELIGIBLE 0
    first refusal of the added: similarity 264 (67%), unparseable/over-ceiling 94
    (24%), qed 35 (9%)

So the discarded placements are **DISTINCT_BUT_INELIGIBLE**, not duplicates: the
axis is real, 11% more molecules are reachable through it, and the gate refuses
all of them -- two thirds on SIMILARITY. That is mechanistically unsurprising for
a similarity-constrained task: moving the attachment site moves away from the
reference, so `assignments[0]` is already near-optimal for the binding constraint
and every alternative is worse. **Placement is not the lever for T4 at delta=0.6.**
It was also byte-identical on eligible counts across 3 of 3 cells at 480 draws.
Left default-off; do not turn it on (8x the instantiate-and-gate work for nothing).

Side observation worth keeping: 94 of the 393 are unparseable or over-ceiling,
including disconnected fragments (`C=O.CC(C)...`) and radicals (`[CH2]`, `[C]=O`),
which independently corroborates that the intervention layer still emits radicals.

### What DID work, and stands on its own

The replace-completion law (`src/compose_v4/control/replace_completion_law.py`,
contract `proposal.shallow.completion_law`) conditions the completion half of
`segment_replace`, which drew blind: `rng.integers(1, capacity+1)` over 1..8 plus
a uniform C/N/O chain. Measured, every eligible completion inserts ONE or TWO
atoms, so ~75% of the draw mass landed where nothing is eligible.

  480 draws per arm, arms differ ONLY in the contract field:
    fa7_1     7 -> 13 eligible   (+86%)
    braf_2    4 ->  6            (+50%)
    5ht1b_0  20 -> 35            (+75%)
  No control regressed. 6/6 mutations killed, both shared-sink hops separately.
  Absent field = byte-identical OFF; a uniform law is NOT a drop-in (it reproduces
  the OFF draw only ~1/K of the time), which is why none is registered.

**This is a general improvement to the T4 proposal law and is worth having
independently of whether fa7_0 ever closes.**

### Method notes that cost something to learn

- **A mutation that silently fails to apply must ABORT, not score as killed.**
  Two mutations missed on an indentation mismatch; the runner refusing to emit a
  verdict is what caught it.
- **A floor test whose margin RETURNS a bad value never reaches the branch it
  claims to guard** -- it must RAISE. One mutation survived on exactly this and
  the repaired test kills it. A guard is only tested where it binds.
- **A broad per-draw `except` will swallow your own probe's bug.** Slicing a
  2-tuple as `[:3]` reported `executed=0 refused=600`, which reads as "the module
  is inexpressible" and is instead a typo. Assert the return SHAPE.
- Wrap the live gate to see a funnel; `expand` returns only survivors, so its
  interior is invisible from the return value. Require the decomposition to
  predict the real verdict on every call (`gate_disagreements == 0`) or the
  attribution is void.
