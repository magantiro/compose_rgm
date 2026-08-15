# Graph GA (GB-GA) — adapter qualification

**Status: `DESIGN_ONLY`. Nothing installed, nothing run.**

## Identity

| field | value |
|---|---|
| paper | Jensen — *A graph-based genetic algorithm and generative model/Monte Carlo tree search for the exploration of chemical space*, Chem. Sci. 10(12) 3567–3572 (2019) |
| doi | [10.1039/c8sc05372c](https://doi.org/10.1039/c8sc05372c); PMID 30996948, PMC6438151, CC BY 3.0 |
| code | <https://github.com/jensengroup/GB_GA> (paper ESI cites `GB-GA/tree/v0.0`) |
| benchmark wrapper | `main/graph_ga/` in <https://github.com/wenhao-gao/mol_opt> (PMO, MIT) |
| license | MIT, both repos |
| checkpoint | None — there is no trained model |
| citation key | `jensen2019graphga`; PMO is `gao2022pmo` |

## Why this is the first adapter to build

It is the only method in the registry with **no model, no training, no
checkpoint, no GPU story and no dependency beyond RDKit.** The paper's own
timing is "30 seconds each on a laptop". PMO ranks it second of 25 methods by
summed AUC top-10 (13.751, behind REINVENT's 14.196), so it is a strong baseline
rather than a token old one — which is exactly what `docs/EXPERIMENT_PLAN.md`
says about it.

Building this adapter first is the cheapest possible end-to-end proof that the
COMPOSE oracle shim and the chosen oracle-counting convention work.

## What was verified

- **No objective-specific training.** `scoring_function` is an argument to
  `GB_GA.GA(args)`. Import audit confirms no torch/tensorflow/CUDA anywhere.
- **Seeding is a SMILES file, sampled with replacement.**
  `make_initial_population` does `population.append(random.choice(mol_list))`,
  so a one-line file yields `population_size` copies of that molecule. There is
  no source-molecule concept in the method. PMO's variant differs: it takes the
  first `population_size` lines rather than sampling.
- **The source is not preserved**, and the paper says so: "The molecules found
  by the GB-GA bear little resemblance to the molecules used to construct the
  initial mating pool"; nearest-ZINC Tanimoto 0.27 and 0.12.
- **No pathwise mechanism.** Crossover cuts ring and non-ring bonds with equal
  probability at random positions, and `mutate.py` includes `delete_atom()` with
  SMARTS such as `[*:1]~[D1]>>[*:1]`. The only per-step filters are structural
  sanity checks (macrocycles, allene centres in rings, <5 heavy atoms, bad
  valences, oversize).
- **No mid-run objective switch.** A single `scoring_function` is unpacked once
  and the generation loop never re-reads it.
- **Lineage is a two-parent DAG and is not persisted.** `reproduce()` returns
  only the child; the parent debug print is commented out; the only logged
  trajectory is one `(best_score, best_smiles)` tuple per generation.

## The oracle-counting fork — decide before writing code

The two conventions differ and the difference is not cosmetic:

| | native GB-GA | PMO wrapper |
|---|---|---|
| duplicates | **charged** — `calculate_scores` scores the whole offspring batch with no dedup | **not charged** — `Oracle.score_smi` keys `mol_buffer` on canonical SMILES and a repeat hits `pass` |
| invalid children | not charged — the `while` loop only appends non-`None` children | not charged — returns 0 without entering the buffer |
| survivors | never re-scored | never re-scored (cached) |
| budget | `population_size × (generations + 1)` = 1020 in the paper's config | 10,000 unique valid canonical SMILES |

COMPOSE's own rule in `configs/comparator_registry_v3.json` says to count every
evaluation including rejected candidates, which is the native convention and is
**incompatible with published PMO numbers**. Pick one, state it, and apply it to
every method.

## Adapter work required

1. Wrap the COMPOSE frozen goal language and `src/compose_v4/drd2_oracle.py` as
   a `scoring_function`.
2. Declare and record the counting convention (above).
3. Seed driver: one-line SMILES file per source, or PMO's first-N-lines
   behaviour — chosen explicitly, not by default.
4. Element-set and size-bound alignment to the COMPOSE kernel, or the deviation
   recorded.

## CPU feasibility

Yes, trivially. RDKit and the standard library only; parallelism is
`multiprocessing.Pool`.
