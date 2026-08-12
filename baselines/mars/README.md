# MARS — adapter qualification

**Status: `DESIGN_ONLY`. Nothing installed, nothing run.**

## Identity

| field | value |
|---|---|
| paper | Xie, Shi, Zhou, Yang, Zhang, Yu, Li — *MARS: Markov Molecular Sampling for Multi-objective Drug Discovery*, ICLR 2021 (Spotlight) |
| preprint | [arXiv:2103.10432](https://arxiv.org/abs/2103.10432); OpenReview `kHSu4ebxFXY`; arXiv DOI `10.48550/arXiv.2103.10432` (ICLR mints no proceedings DOI) |
| code | <https://github.com/bytedance/markov-molecular-sampling> — **archived, read-only**, last push 2022-02-03, 2 commits. The URL printed in the paper (`github.com/yutxie/mars`) is a README-only pointer to this repo. |
| license | **CC BY-NC 4.0 — non-commercial.** The `LICENSE` file's first line is the Creative Commons Attribution-NonCommercial 4.0 text; GitHub's detector shows `NOASSERTION`. `estimator/scorer/sa_scorer.py` carries a separate BSD-style Novartis notice. |
| checkpoint | **None for the proposal.** No editor/proposal checkpoint is shipped, and no shipped code path writes one (`sampler.py` calls `train(...)` without `save_dir`, so the only `torch.save` in `common/train.py` is unreachable). Shipped binaries are oracles only: `kinase_rf/gsk3b.pkl`, `kinase_rf/jnk3.pkl`, `fpscores.pkl.gz`. |
| citation key | `xie2021mars` |

## What was verified

- **Native object: annealed MCMC sampling from a fixed unnormalized multi-objective
  target** `π(x) = Σ_k s_k(x)`, with temperature `T = 0.95^⌊t/5⌋` (paper §3.1, §4.1).
- **Every chain state is a complete valid molecule.** Invalid proposals are
  rejected and the chain stays put (`proposal.py`, `common/chem.py::check_validity`).
- **Two edit operators only**: fragment add and fragment delete, chosen with
  probability ½ each — the learned action head is commented out and the logits
  are hard-zeroed (`proposal/proposal.py`). Vocabulary is the top-1000 ChEMBL
  fragments with ≤10 heavy atoms and a single attachment point; **the vocabulary
  is not shipped** (`data/vocab_*` is gitignored) and must be regenerated from
  the 1,488,640-SMILES `data/chembl.txt`.
- **Source conditioning works but is undocumented.** `--mols_init FILE` loads
  SMILES and calls `random.choices(mols, k=num_mols)`; a one-line file therefore
  replicates one exact molecule into every chain. A programmatic caller can pass
  a per-chain list straight to `Sampler.sample(run_dir, mols_init)`. Default with
  no flag is ethane `CC` for every chain.
- **The proposal is trained online during sampling** on edits that improved the
  score (`sampler.py`, Algorithm 1 lines 11–13), ≤25 gradient steps per MCMC step.
  There is no pretraining stage.
- **Mid-run objective change: no native support at all.** Objectives are parsed
  once (`main.py`), consumed at `Estimator`/`Sampler` construction, and the
  sampling loop has no hook to swap them. A full-text search of the paper for
  switch / warm-start / resume / retrain returns nothing on this topic.
- **No pathwise constraint mechanism.** No SMARTS, no substructure matching, no
  atom freezing. The only greps that hit `mask` are training-loss masks in
  `editor_basic.py`. `break_bond` may delete any single-bond-attached fragment,
  including one we wanted preserved.
- **Rejected proposals are scored.** All candidates go through
  `estimator.get_scores` before the accept/reject loop, and there is no cache.

## The three pieces of hidden state that make a switch non-trivial

Recorded because they decide what "restart at the switch molecule" can honestly mean:

1. **Imitation dataset `D` accumulates and is never cleared.** After a switch it
   still holds up to 50,000 edits labelled "improving" under the *old* reward,
   diluted only randomly as new records arrive.
2. **The adapted proposal cannot be carried across a process boundary** — no
   shipped code path saves it, so a relaunch starts from a randomly initialised
   editor.
3. **The temperature counter is monotonic and floored.** `sampler.py` applies an
   undocumented `T = max(T, 1e-2)` floor not present in the paper's formula;
   `0.95^k ≤ 0.01` at `k = 90`, i.e. ~step 450. A switch after that lands in a
   frozen, effectively greedy regime with no reheat mechanism.

Consequence: the only *stock-code* switch arm is **restart-at-the-switch-molecule
with a fresh sampler** — new `Estimator`, `mols_init` = the realized molecule,
proposal re-initialised, temperature reset. Anything warmer is a patch we write,
and must be reported as a deviation.

## Adapter work required

1. Objective shim: replace `Estimator` with one calling the COMPOSE frozen goal
   language and `src/compose_v4/drd2_oracle.py`, with a counter wrapping every
   call. Note `estimator/scorer/scorer.py` has `drd2_scorer` commented out of its
   imports while still dispatching to it, and the DRD2 model file is not shipped
   (repo issue #2) — so a DRD2 objective *must* come from our side regardless.
2. Chain-count reduction. MARS's native regime is 1000–5000 chains × ~550–1000
   steps. Any budget-matched COMPOSE comparison runs it far outside that regime;
   see the fairness contract.
3. Driver for `mols_init` from a source panel, one molecule per chain.
4. Trajectory export: MARS keeps no history; the accepted-state sequence must be
   logged by us if pathwise metrics are ever wanted (they should not be — MARS
   has no pathwise mechanism and gets `N/A`).
5. Vocabulary regeneration step (`python -m MARS.datasets.prepro_vocab`) — a
   prerequisite, must run from *outside* a directory literally named `MARS`
   because of hardcoded `MARS/...` paths.

## Known friction

- Repo archived: no new issues can be filed, two existing issues are unanswered.
- Issue #2 documents that adding an objective is not a flag change.
- `datasets/utils.py` truncates `smiles2idx` to 10 entries regardless of
  `vocab_size`; harmless under the default `sa` sampler, breaks `Sampler_MH`'s
  reverse-proposal correction.
- Code defaults differ from the paper: `num_mols` 1000 vs 5000,
  `dataset_size` 50,000 vs the paper's 75,000.
- README reports **best-step**, not final-step, results.

## CPU feasibility

README documents a CPU install (`conda install pytorch cpuonly`;
`conda install -c dglteam dgl`) and `--device cpu` is accepted. The pinned stack
is python 3.9.6 / torch 1.9.0 / **DGL 0.7.0** (not torch-geometric) /
scikit-learn 0.23.2 / rdkit 2021.03.4. Whether that stack still resolves on a
current index is **unverified** — nothing was installed. The paper's 12-hour
figure is one V100 plus 64 virtual CPU cores.
