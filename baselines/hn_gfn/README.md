# HN-GFN — adapter qualification

**Status: `DESIGN_ONLY`. Nothing installed, nothing run.**

## Identity

| field | value |
|---|---|
| paper | Zhu, Wu, Hu, Yan, Hsieh, Hou, Wu — *Sample-efficient Multi-objective Molecular Optimization with GFlowNets*, NeurIPS 2023 |
| preprint | [arXiv:2302.04040](https://arxiv.org/abs/2302.04040) (v2, 2023-11-02); OpenReview `uoG1fLIK2s` |
| code | <https://github.com/violet-sto/HN-GFN> (last push 2023-12-24) |
| license | MIT |
| checkpoint | **No HN-GFN/GFlowNet checkpoint released.** Bundled artifacts are oracles/proxies only (`oracle/gsk3b.pkl` 27 MB, `oracle/jnk3.pkl` 11 MB, `data/docked_mols.h5` 98 MB, `data/pretrained_proxy/*`). |
| citation key | `zhu2023hngfn` |

## What was verified

- **Native object: target-free multi-objective optimization with Pareto
  coverage**, and HN-GFN proper is the *acquisition-function optimizer inside a
  MOBO loop*, not a standalone optimizer. The standalone use in §5.1 is
  explicitly labelled a "synthetic scenario" because it queries the true oracle
  at every training step.
- **One model spans a continuum of preference vectors without retraining.** A
  hypernetwork `h(λ; φ)` emits the weights of the prediction heads while the MPNN
  trunk is shared; `λ ~ Dir(α)` is resampled per training iteration.
  This is a real capability and the paper demonstrates it (§5.1).
- **The objective SET is fixed at training time.** `n_objectives` is the first
  linear layer's input width in `model_pred_hyper.py` and the surrogate's output
  width in `proxy/proxy.py`. A genuinely new objective changes `M` and requires a
  full retrain. The paper makes no transfer-to-new-objective claim.
- **Generation is from scratch, append-only.** Trajectories begin at the empty
  molecule (`rollout()` → `BlockMoleculeDataExtended()`), attach fragments from a
  105-block vocabulary, max 8 blocks, and there are **no fragment-removal
  actions**. Non-terminal states carry open attachment stems and get reward 0.
- **No source-molecule conditioning.** No seed/scaffold parameter exists; the
  only policy conditioning input is the preference vector.
- **No mid-trajectory retargeting.** The preference vector is sampled once per
  rollout and held fixed for every step. The "hindsight" strategy re-scores
  **complete terminal molecules** under other preferences and files them in the
  replay buffer — relabelling, not intervention.

## Why this method matters more than its classification suggests

HN-GFN is the strongest published instance of "one trained model, many goals,
no retraining". A reviewer will raise it against the COMPOSE frozen-process
claim, and the answer has to be precise rather than dismissive:

| | HN-GFN | COMPOSE C4c |
|---|---|---|
| what varies without retraining | scalarization weights `λ` over a **fixed** objective set | an **unanticipated** goal, including one not in any training distribution |
| where the trajectory starts | the empty molecule, always | the realized molecule `x_τ` from a history produced under a different goal |
| when the goal may change | before the rollout | **mid-trajectory**, with the prefix preserved |
| what is reused across goals | the trained policy | the frozen reference process `R_theta`, with only the control law recomputed |

The honest statement is that HN-GFN already occupies the "flexible goals from one
model" column; COMPOSE's distinct claim is the *conjunction* of unanticipated
goal, realized-history continuity, and finite remaining-budget replanning. That
is exactly what `docs/RETARGETING_SAME_PREFIX_DESIGN.md` already says, and this
verification supports it rather than weakening it.

## The oracle-accounting landmine

HN-GFN's headline budget is **1000 true-oracle calls** (`|D_0| = 200` plus 8
rounds × batch 100). GFlowNet training inside each round is rewarded by the
**surrogate**, not the true oracle. So HN-GFN's oracle efficiency is bought with
a learned proxy in the loop.

Placing that 1000 next to a COMPOSE budget that queries the true oracle for every
candidate is **not a fair comparison in either direction**. Either both methods
get a surrogate, or the surrogate's training and query cost is reported as a
separate column. This must be settled before any number is put in a table.

Note also that the "oracles" here are themselves cheap ML predictors (random
forests for GSK3β/JNK3, RDKit QED/SA) — the budget simulates expense.

## Adapter work required

Substantial, and probably not worth it for anything except a Pareto-coverage
comparison:

1. Objective shim replacing the closed enum in `oracle/scorer/scorer.py` with the
   COMPOSE frozen goals. DRD2 is already in that enum but as a python-3.6 sklearn
   pickle; use the COMPOSE numpy oracle instead.
2. Full retrain per objective set (~10 h, 13 h with hindsight, on one V100 for
   the 8-round loop). **There is no CPU path to this.**
3. Fragment-vocabulary decomposition for any source molecule, if source
   conditioning were ever attempted — the 105-block vocabulary is not general and
   no decomposition path for arbitrary SMILES is provided.

## Known friction

- **Open, unanswered issue #1**: `ImportError: cannot import name
  'AnalyticMultiOutputObjective' from botorch` — the import at
  `proxy/regression.py` targets symbols removed from modern BoTorch.
- **No pinned dependency file at all** (no requirements.txt, environment.yml,
  setup.py or pyproject.toml).
- `np.bool` / `np.int` usages fail on NumPy ≥ 1.24
  (`dataset.py`, `mol_mdp_ext.py`, `utils/chem.py`).
- Deprecated `Dirichlet(...).sample_n(1)`.

## CPU feasibility

**No usable CPU path.** Code defaults to `--device cuda`; the published cost is
10–13 h on one V100 for the full loop, and no checkpoint is released, so
inference-only CPU use is impossible.
