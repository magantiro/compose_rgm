# REINVENT — adapter qualification

**Status: `DESIGN_ONLY`. Nothing installed, nothing run.**

> **This directory covers two different systems and they must not be conflated.**
> The PMO benchmark's `main/reinvent` is a **REINVENT 2.0-era Olivecrona-style
> RNN reimplementation**. It does **not** wrap REINVENT 4. Every capability
> below marked "REINVENT 4" is absent from the PMO-benchmarked configuration,
> and PMO's published numbers may not be cited in a row that claims a REINVENT 4
> capability.

## Identity

| field | value |
|---|---|
| REINVENT 4 | Loeffler, He, Tibo, Janet, Voronov, Mervin, Engkvist — *Reinvent 4: Modern AI-driven generative molecule design*, J. Cheminform. 2024, [10.1186/s13321-024-00812-5](https://doi.org/10.1186/s13321-024-00812-5), PMC10882833 |
| REINVENT 2.0 | Blaschke et al., J. Chem. Inf. Model. 2020, [10.1021/acs.jcim.0c00915](https://doi.org/10.1021/acs.jcim.0c00915); repo <https://github.com/MolecularAI/Reinvent> (license UNVERIFIED) |
| code (v4) | <https://github.com/MolecularAI/REINVENT4>, Apache-2.0 |
| priors | Zenodo [10.5281/zenodo.15641296](https://doi.org/10.5281/zenodo.15641296), apache2.0. `libinvent.prior`, `linkinvent.prior`, `pepinvent.prior`, `reinvent_pubchem.prior`, `libinvent_transformer_pubchem.prior`, `linkinvent_transformer_pubchem.prior` |
| PMO wrapper | `main/reinvent/` in <https://github.com/wenhao-gao/mol_opt>, MIT, ships its own `data/Prior.ckpt` |
| citation key | **REINVENT 4 has no entry in `paper_iclr_stochastic_rewriting/references.bib` and must be added before it is cited.** `gao2022pmo` is PMO and must not be used as the REINVENT 4 citation. |

## The finding that matters most in this whole lane

**REINVENT 4 natively changes the objective mid-run.** Staged (curriculum)
learning is a first-class run mode. From the paper, verbatim:

> This is basically curriculum learning (CL) which in REINVENT 4 is implemented
> as a multi-stage RL. **The main purpose is to allow the user to optimize a
> prior model conditioned on a calculated target profile by varying the scoring
> function in stages.** ... Multiple stages can be provided at once (automatic
> CL). After each stage a checkpoint file is written to disk which can be used
> for the next stage (manual CL).

`configs/staged_learning.toml` ships two `[[stage]]` blocks, each with its own
`[stage.scoring]` and `chkpt_file`.

The COMPOSE paper must not claim that no existing method changes objective
mid-run. What remains distinct is precise and defensible:

- REINVENT re-targets a **policy**; there is no in-flight molecule to continue
  from, because generation is a full SMILES decode per episode.
- The switch fires at a **stage boundary** set by `max_score` or `max_steps`,
  not at an arbitrary chosen step of a realized history.
- After the switch the agent **re-adapts by further RL** — it retrains. COMPOSE
  recomputes only the control law on a frozen `R_theta`.

## Other verified capabilities

- **Source conditioning exists**, in three flavours with different semantics.
  LibInvent decorates a scaffold **with explicit attachment points** and
  preserves it by construction; LinkInvent preserves two warheads by
  construction; Mol2Mol takes a complete molecule but explicitly does **not**
  preserve it — "the scaffold can change within the limits of the given
  similarity".
- **Objective-specific RL fine-tuning is the method.** Cost is
  `batch_size × steps`; the shipped example is batch 64 with 25–100 steps per
  stage, so ≤6,400 scored molecules per stage. The prior itself costs zero
  oracle calls.
- **No intermediate molecular states.** Generation is autoregressive over SMILES
  tokens, so intermediates are token prefixes that are generally not parseable.
  Every pathwise metric is `N/A`.
- **No pathwise substructure constraint.** `MatchingSubstructure` is tagged
  `@add_tag("__component", "penalty")` and returns `0.5 * (1.0 + match)` — a soft
  2× score penalty, not a constraint. Custom alerts are a SMARTS blocklist
  multiplying the total score by 0 or 1.
- **Oracle calls are not charged for duplicates or invalids.**
  `scoring/scorer.py` computes `valid_mask = np.logical_and(invalid_mask,
  duplicate_mask)` and the scoring components cache per SMILES. The diversity
  filter additionally zero-scores every repeat of a canonical SMILES globally,
  which is a behaviour change and not merely an accounting one.

## Two adapters, not one

**Adapter A — oracle efficiency.** Use PMO's `main/reinvent` through
`BaseOptimizer` so the shared `Oracle` counter applies, with the COMPOSE goal
language assigned via `self.oracle.assign_evaluator`. Gives comparability to
published PMO numbers. REINVENT 2.0 semantics.

**Adapter B — staged learning.** REINVENT 4 with a two-stage
`staged_learning.toml`, Mol2Mol or LibInvent input, and the COMPOSE objective as
a custom scoring component. **This is the only external arm that natively
changes objective mid-run, and it must be run for the C4c claim to be stated
honestly.**

## Deviations to record before any run

- PMO's own caution: "REINVENT's performance is highly dependent on σ; we found
  the best-performing value to be much larger than the values suggested in the
  original paper ... thus suggest reoptimizing the hyper-parameters whenever the
  testing environment is changed." Using PMO's tuned `sigma = 500` on a COMPOSE
  objective is itself a deviation.
- Which prior is used. The six Mol2Mol similarity variants (medium, high,
  scaffold, generic scaffold, MMP, PubChem) are materially different models.
- Download and hash the Zenodo priors before any claim-bearing run.

## CPU feasibility

Yes. Paper: "All run modes can either run on a GPU or a CPU." README: "A GPU is
not strictly necessary but strongly recommended ... For Reinforcement learning
(RL) a GPU is less important because most scoring components run on the CPU.
Note that if no GPU is installed in your computer the code will run on the CPU
automatically." CLI flag `-d/--device`.
