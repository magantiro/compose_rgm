# What the paper contains, and where it lives in this repository

This page maps the ICLR 2027 submission to the code that implements it, and
states what you need beyond a checkout to run each benchmark.

**Measured scores are not stored here.** The repository ships the machinery that
produces the numbers; the numbers themselves are in the paper. This matches the
paper's own reproducibility statement: *large corpora, checkpoints and per-run
traces are stored outside the source repository, and a fresh checkout alone is
insufficient to recover the reported measurements.*

---

## The method

| Paper | What it defines | Code |
|---|---|---|
| §2.1, §C.1 | Executable molecular state space `X`; the executor's legal-edit set `A(x)` | [`src/compose_v4/chem/`](../../src/compose_v4/chem/), [`src/compose_v4/rewrite/`](../../src/compose_v4/rewrite/) |
| Table 4 | The eight primitive rewrite families | [`src/compose_v4/rewrite/operators.py`](../../src/compose_v4/rewrite/operators.py) |
| §2.2, §C.2–C.3 | Reference law `R_θ` over canonical successors | [`src/compose_v4/model/`](../../src/compose_v4/model/), [`src/compose_v4/gm/`](../../src/compose_v4/gm/) |
| §2.3, §C.4 | Structured programs `π` composed from executable edits | [`src/compose_v4/control/`](../../src/compose_v4/control/) |
| §2.4–2.6 | Controllers: structural conditioning, feedback-driven optimization | [`src/compose_v4/control/`](../../src/compose_v4/control/) |
| §2.7, §D | Finite-horizon future values, Doob transform, twisted SMC | [`editing_v2_bridge_control.py`](../../src/compose_v4/experiments/editing_v2_bridge_control.py) |
| §F | Corpus scope and transition construction | [`src/compose_v4/data/organic_corpus.py`](../../src/compose_v4/data/organic_corpus.py) |

**A naming note the paper already makes (§E.1) and the code confirms.** The
paper-facing names `cycle_close`, `cycle_open` and `ring_aromaticity_restate`
are recorded in the implementation as `cycle_insert`, `cycle_attach` and
`ring_system_restate`. The other five agree across both conventions. All eight
are present in `rewrite/`.

## The experiments

| Paper | Benchmark | Entry point |
|---|---|---|
| §3.1, Table 6 | Reference-law evaluation on held-out transitions | [`scripts/emit_reference_law_data_backed.py`](../../scripts/emit_reference_law_data_backed.py) |
| §3.2, Table 1 | Fragment-constrained generation, five SAFE tasks | [`benchmark/fragment_constrained.py`](../../src/compose_v4/benchmark/fragment_constrained.py), [`fragment_constrained_runner.py`](../../src/compose_v4/benchmark/fragment_constrained_runner.py) |
| §3.3, Table 10 | Similarity-constrained QED editing (ZINC-250k) | [`scripts/evaluate_qed_controlled_rollouts.py`](../../scripts/evaluate_qed_controlled_rollouts.py) |
| §3.4, Tables 2, 11 | PMO-1K black-box optimization | [`experiments/pmo_population_v1.py`](../../src/compose_v4/experiments/pmo_population_v1.py), [`modal_apps/pmo_population_v1_app.py`](../../modal_apps/pmo_population_v1_app.py) |
| §3.5, Table 3 | T4 similarity-constrained lead optimization | [`experiments/t4_fiber_campaign.py`](../../src/compose_v4/experiments/t4_fiber_campaign.py), `modal_apps/t4_*` |
| §G.2 | Quality / uniqueness / diversity / validity metrics | [`eval/molecular_quality.py`](../../src/compose_v4/eval/molecular_quality.py) |

`experiments/INDEX.md` lists every runnable entrypoint in the tree with its
command, generated from source.

---

## What you must supply to run any of this

A checkout is not sufficient, and the paper says so. Each item below is external
to the repository.

| Asset | Needed for | Where it comes from |
|---|---|---|
| Reference checkpoint `R_θ` | **everything** — all four benchmarks share it | Modal artifact volume; not in git |
| QuickVina2 binary, receptors, docking boxes | T4 (§3.5) | built into the execution image |
| TDC oracle assets for `drd2`, `gsk3b`, `jnk3` | PMO (§3.4) | PyTDC downloads on first use |
| GuacaMol corpus + compiled transition store | retraining `R_θ` | external; only needed to retrain |
| Modal account | running any benchmark at scale | — |

Without `R_θ` nothing runs. It is the single artifact every experiment depends
on, and it is not in this repository.

## Environments — two, and they are not interchangeable

```bash
# Core: executor, reference process, fragment, QED editing, T4
python 3.11 · rdkit 2024.3.5 · torch 2.4.0 · numpy 1.26.4 · scipy 1.13.1

# PMO only
python 3.11 · rdkit 2023.9.6 · PyTDC 1.1.15 (--no-deps) · numpy 1.26.4 · setuptools 69.5.1
```

PMO needs its own environment because PyTDC 1.1.15 pins `rdkit<2024.3.1`, so
installing it against 2024.3.5 is unsatisfiable. Its image therefore carries
rdkit 2023.9.6, which means the PMO runtime — oracle *and* executor — runs a
different chemistry kernel from the rest of the paper. RDKit version is held
fixed within each reported comparison because canonical identity, ring
perception and sanitization affect both executable support and scoring.

## Benchmark protocols, as the paper defines them

- **Fragment (§3.2).** Ten drug-derived prompts per task, five tasks. Each
  attempted generation occupies one output slot; a no-output attempt stays in
  the denominator. Scaffold morphing reuses the linker outputs and is not an
  independent sampling experiment.
- **QED editing (§3.3, §I.1).** 800 sources with initial QED in [0.7, 0.8];
  success is QED ≥ 0.9 at Tanimoto ≥ 0.4 to the source; K = 8 returned
  trajectories per source.
- **PMO (§3.4, §G.5).** 1,000 oracle calls per objective and seed, three seeds,
  no prescreen, shared task-independent initialization bank. The reported mean
  spans 22 objectives and excludes valsartan SMARTS.
- **T4 (§3.5, §G.6).** Five targets × three seeds × δ ∈ {0.4, 0.6} = 30 cells,
  250 docking evaluations per cell. Feasible means QED ≥ 0.6, SA ≤ 4,
  Tanimoto ≥ δ. Endpoint constraints only — intermediate states need pass just
  the executor. Docking replicate noise is 0.70 kcal/mol, so single-cell
  differences below that are not resolved improvements.

---

## Not part of this paper

The tree also carries superseded manuscript drafts (`paper/`, `paper_arxiv/`,
`paper_iclr_control_substrate/`, `paper_iclr_stochastic_rewriting/`) and a
separate NeurIPS **workshop** package (`paper_gem_neurips2026/`) reporting a
*different* T4 experiment against GenMol, RetMol and GraphGA at 500 calls per
cell. Those numbers are not this paper's and should not be mixed with Table 3.

Development diagnostics under `diagnostics/` are working artifacts from the
research process, not published results.
