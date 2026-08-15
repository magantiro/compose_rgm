# Vendored upstream GB-GA — verbatim, unmodified

| field | value |
|---|---|
| source | <https://github.com/jensengroup/GB_GA> |
| commit | `4b49f1822c5190e8b2bbb8b7403eed30af9e50fd` (2021-06-15) |
| license | MIT, `LICENSE` in this directory, Copyright (c) 2018 Jensen Group |
| paper | Jensen, *Chem. Sci.* 10(12) 3567–3572 (2019), [10.1039/c8sc05372c](https://doi.org/10.1039/c8sc05372c) |
| vendored | 2026-08-13 by workstream D |

## Files and their sha256 as vendored

```
fd4adcd40523263abb3be38a7ee70e0766be5326048c812f157cabd949f7a71e  GB_GA.py
6f1e84da39ce270d2a8981041eb6227ac1557d029f6b0290b1c95d8cda7d91ea  crossover.py
0839dd392ae0c38022b3df0b742037ae76a704d9c72a058a33bd1e3f9fe17e54  mutate.py
```

**These three files are byte-identical to upstream.** Verify with:

```bash
git clone https://github.com/jensengroup/GB_GA /tmp/gbga && \
  cd /tmp/gbga && git checkout 4b49f182 && \
  shasum -a 256 GB_GA.py crossover.py mutate.py
```

## What is deliberately NOT vendored, and why

- **`scoring_functions.py`** — it performs three module-level `np.loadtxt` calls
  against CWD-relative paths and pulls in 750 KB of ZINC-derived normalization
  data plus a 3.8 MB `fpscores.pkl.gz`. None of that is the algorithm; it is the
  paper's own penalized-logP objective, which a COMPOSE comparison replaces with
  the frozen COMPOSE objective. `GB_GA.py` imports it at module level, so the
  adapter installs a **faithful shim** exposing only `calculate_scores` with the
  upstream semantics reproduced verbatim. See
  `src/compose_v4/experiments/graph_ga_adapter.py`.
- **`ZINC_250k.smi`** (11 MB), `SA_scores.txt`, `logP_values.txt`,
  `cycle_scores.txt`, `sascorer.py`, the `GA_*.py` drivers — benchmark data and
  drivers, not the algorithm.

## Two properties of this code that the adapter must handle

Both were found by running it, not by reading it, and both are invisible from
the README.

### 1. Two undocumented module-level globals are REQUIRED

`crossover.mol_OK` reads `size_stdev` and `average_size`, which upstream sets
only in its driver (`GA_logP.py` lines 19–20: `co.average_size = 39.15`,
`co.size_stdev = 3.50`, ZINC-derived). The comment in `mol_OK` says merely
"parameters set in GA_mol".

If they are unset, `mol_OK` raises `NameError`, its bare `except: return False`
swallows it, **every** candidate is rejected, `crossover` always returns `None`,
and the run hangs — see 2. There is no error message.

They are not a nuisance: together they impose a **soft Gaussian size prior on
every offspring** (`target_size = size_stdev * randn() + average_size`;
offspring must have `5 < atoms < target_size`). Leaving them at the ZINC values
while running on a COMPOSE source panel would ask the baseline to build
ZINC-sized molecules on a differently-sized panel. **The adapter derives them
from the actual source panel and records the values**, because this is a
fairness parameter, not a constant.

### 2. `reproduce()` has an unbounded loop with no oracle cost

```python
def reproduce(mating_pool, population_size, mutation_rate):
  new_population = []
  while len(new_population) < population_size:
    ...
```

Failed crossovers and mutations consume **no oracle calls**, so an oracle-call
budget does not bound GB-GA's wall time. The adapter therefore imposes a
wall-clock guard and counts failed reproduction attempts. This matters for the
fairness contract: a budget-matched comparison bounds oracle calls, not runtime,
and GB-GA is the one method here that can spend unbounded time at zero budget.
