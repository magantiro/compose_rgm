# The validity check rebuilds the whole molecule per candidate: 52.9x available

**Measured, with identical verdicts. Blocked by the same provenance hash as the
admission-mask memoization.**

## What it does now

`is_rdkit_valid(mg)` is the authoritative validity gate and is called once per
candidate edit -- roughly 1,300 times per law enumeration. It does:

    smi = molecular_graph_to_smiles(mg)      # build RWMol atom by atom,
                                             # SanitizeMol, write canonical SMILES
    return Chem.MolFromSmiles(smi) is not None   # tokenize, rebuild, sanitize again

Two suspicions, one wrong and one right.

**Wrong: the round trip is the waste.** Skipping the canonical-SMILES write and
the re-parse, keeping the same builder, is worth **1.08x**. The predicates agree
on all 1392 candidates. The write and parse are only ~22 of ~308 microseconds.

**Right: the reconstruction is the waste.** A candidate differs from its source
by ONE bond and two hydrogen counts, but the molecule is rebuilt from nothing --
about 30 atoms and 30 bonds, each a separate Python-to-C++ call -- and then
sanitized. Building the source Mol ONCE, copying it in C++, and adding the single
bond:

    rebuild per candidate (current)   327.9 us
    copy source + add one bond          6.2 us
    speedup                            52.90x

    valid: current 178 / 1392   incremental 178 / 1392
    DISAGREEMENTS: 0

## What it is worth, and what it is not

`is_rdkit_valid` is 40.2% of an unoptimized `enumerate_cycle_close_edges`
(631 ms of 1569 ms), so 52.9x on that step alone is about 1.65x on the
enumeration. Its value is in COMBINATION with the admission-mask memoization,
which removes the other 53% -- and that combination is not projected here,
because component speedups folded into component breakdowns have been wrong
before in this work.

Not a drop-in. The incremental path hard-codes the bond-insert edit: add one
bond, decrement two hydrogen counts. Atom-restate and every other operator need
their own incremental construction, each separately qualified.

## Status

`src/compose_v4/chem/molecular_graph.py` is the FIRST entry in the Process-V2
identity hash, so this is blocked exactly as the memoization is. It does change
the calculus recorded in `docs/ADMISSION_MASK_OPTIMIZATION.md`: the earlier
recommendation to drop the corpus-regeneration question was priced against 3.38x
alone. The bundle of exact, measured optimizations is now larger, and if a
regeneration is ever undertaken all of them should land together rather than one
at a time.
