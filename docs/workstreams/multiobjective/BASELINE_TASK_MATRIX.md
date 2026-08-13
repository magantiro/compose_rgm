# Method / task matrix — multiobjective external baselines

**Artifact status: `DESIGN_ONLY`.** Nothing installed, nothing run, no compute
spent. Every cost figure below is a **projection or a published figure**, never a
measurement made by this lane. Where a published figure exists it is quoted with
its source; where it does not, the cell says `UNVERIFIED` rather than carrying an
estimate.

---

## The tasks

**Q1 capability**, **Q2 frontier quality**, **Q3 structural value of closed-loop
control**, **Q4 why learn `R_θ`** are the four burdens the Pareto block must
carry. External methods can only address a subset, and saying which is the point
of this matrix.

| task | what it is | who can enter |
|---|---|---|
| **A1** per-source preference redirection from a supplied molecule at `BUDGET = 6` | Q1 | Panel A only |
| **A2** per-source front quality: HV, HV-AUC, coverage, spread, envelope fidelity | Q2 | Panel A only |
| **A3** closed-loop versus generate-and-rank at matched resources | Q3 | Panel A only, internal |
| **A4** `R_θ` versus empirical-family under identical control | Q4 | Panel A only, internal |
| **B1** global multiobjective competence on a separately frozen common benchmark | ordinary competence | Panel B |

Panel A and Panel B never appear in the same comparison. See
`SOURCE_CONDITIONING_AUDIT.md`; the rule is enforced by
`multiobjective_qualification.assert_not_cross_panel`.

---

## The matrix

`native` = the authors' own algorithm runs the task unmodified.
`adapter` = runs natively, with only our evaluation/accounting adapter.
`inappropriate` = the method's native scientific object is not this task.

| method | A1 | A2 | A3 | A4 | B1 | admissible as |
|---|---|---|---|---|---|---|
| COMPOSE fixed-preference control | native | native | native | native | see note | Panel A primary |
| generate-and-rank P3/P4 | native | native | native | — | inappropriate | Panel A, internal control |
| empirical-family reference ablation | native | native | — | native | inappropriate | Panel A, internal control |
| preference-blind floor (`unguided`) | native | native | — | — | inappropriate | Panel A, floor |
| **HN-GFN** | inappropriate | inappropriate | inappropriate | inappropriate | **adapter** | **Panel B** |
| **InversionGNN** | inappropriate | inappropriate | inappropriate | inappropriate | **adapter, conditional** | **Panel B, conditional** |
| OP-GFN | inappropriate | inappropriate | inappropriate | inappropriate | inappropriate | excluded |
| MOG-DFM / AReUReDi / pCoMole / PepTune | inappropriate | inappropriate | inappropriate | inappropriate | inappropriate | related work |

**Note on COMPOSE in B1.** COMPOSE's process is defined as editing from a
supplied source. Whether a global de novo COMPOSE arm exists at all is an open
question recorded in `DECISION_LOG.md`. If it does not, B1 is a competence
comparison **among external methods** with COMPOSE absent, and the caption says
so. Inventing a de novo COMPOSE to fill the cell would be building a new method.

**Every `inappropriate` in the A columns has the same single cause**, and it is
not a judgement about method quality: A1–A4 require starting from a molecule we
supply. HN-GFN begins every rollout at `main.py:145`
`m = BlockMoleculeDataExtended()`, an empty block molecule. OP-GFN begins at
`graph_sampling.py:79` `self.env.new()`, an empty `Graph()`. InversionGNN can
begin from a supplied molecule but cannot be held to an edit budget from it.

---

## Per-method qualification record

### HN-GFN — the required external row

| field | value | source |
|---|---|---|
| paper | Zhu, Wu, Hu, Yan, Hsieh, Hou, Wu, *Sample-efficient Multi-objective Molecular Optimization with GFlowNets*, NeurIPS 2023 | arXiv:2302.04040 |
| code | `github.com/violet-sto/HN-GFN` @ `90078b8ceeee3e907deeced9b096a6e395b71177` (2023-12-25) | clone, verified |
| licence | MIT | `LICENSE` |
| checkpoints | none for the GFlowNet; a pretrained *proxy* is shipped at `data/pretrained_proxy/` | tree listing |
| assets shipped | `data/blocks_105.json` fragment vocabulary; `data/docked_mols.h5` (93 MB); `oracle/scorer/drd2/clf_py36.pkl`; `oracle/scorer/kinase_rf/{gsk3b,jnk3}.pkl` | tree listing |
| native scalarization | linear weighted sum, `--scalar` default `WeightedSum` | `main.py:54`, `main_mobo.py:62`, `main.py:226-227` |
| preference conditioning | hypernetwork on the output heads; weights drawn per batch from a Dirichlet | `model_pred_hyper.py`, `main.py:239` |
| native objectives | `drd2`, `jnk3`, `gsk3b`, `seh`, `qed`, `sa`, `mw`, `logp`, `penalized_logp` | `oracle/scorer/scorer.py:26-52` |
| true-oracle budget | 200 initial + 8 rounds × 100 = **1,000** | `main_mobo.py:50-52` |
| surrogate in the loop | yes — the GFlowNet's reward is the proxy; the true oracle is called once per outer round | `main_mobo.py:211`, `main_mobo.py:393` |
| device default | `cuda` | `main_mobo.py:39` |
| **published runtime** | **"our proposed HN-GFN costs 10 hours"** on **"1 Tesla V100 GPU"**; "with the hindsight-like training strategy, the running time will increase roughly by 33%" | paper Appendix B.4, quoted verbatim |
| environment | upstream ships **no** requirements.txt, environment.yml, setup.py or pyproject.toml | tree listing |
| known environment blocker | `proxy/regression.py` imports `AnalyticMultiOutputObjective` / `IdentityAnalyticMultiOutputObjective`, removed from modern BoTorch; the last working version is not pinned anywhere and must be bisected | `baselines/hn_gfn/environment.lock` on branch `codex/compose-baseline-qualification`, and upstream issue #1 |

**Cost to run one held-in smoke: `10–13 GPU-hours on a V100-class device`,
from the authors' own figure.** There is no released GFlowNet checkpoint, so
there is no inference-only path and no CPU path. A CPU run is not a slower
version of this; it is not a run.

> **This collides with a standing lane instruction.** Lane 5 is CPU-only and
> no-compute. An HN-GFN smoke therefore requires an explicit, separate GPU
> authorization from the lead, on top of Stage 2 launch approval. It is the only
> item in this lane that does.

### InversionGNN — conditional

| field | value | source |
|---|---|---|
| paper | Niu, Gao, Xu, Liu, Bian, Rong, Huang, Li, *InversionGNN: A Dual Path Network for Multi-Property Molecular Optimization*, ICLR 2025 | arXiv:2503.01488 |
| code | `github.com/ivanniu/InversionGNN` @ `cfdf1d9a981ca4ce5dd7293dc38373ddf9377718` (2025-08-29) | clone, verified |
| licence | **none — no `LICENSE` file in the tree**; default all-rights-reserved | `find -iname '*licen*'` returns 0 |
| checkpoints | **none**; `model_ckpt = ""` then `torch.load(model_ckpt)` | `molecular/denovo.py:75-76` |
| device | `cpu` | `molecular/denovo.py:74` |
| Pareto mechanism | non-dominating gradient direction plus an Exact-Pareto-Optimal LP | `molecular/epo_lp.py` |
| objectives | GSK3β, JNK3, QED, SA (`logp` in code). **No DRD2.** | paper §5.2; repo grep |
| oracle budget | paper allocates 10K calls for surrogate pretraining (2-objective) plus `N_weight × 1K` for optimization | paper §5.3 |
| **blocking defect 1** | no checkpoint and no shipped training labels (`data/zinc_label.txt` absent) | tree listing |
| **blocking defect 2** | call/definition arity mismatch: call at `molecular/denovo.py:107` passes six arguments to a four-parameter function defined at `molecular/inference_utils.py:153` | both lines read directly |
| reported hardware | **UNVERIFIED** — the paper reports no GPU, no hours, no wall clock | paper grep |

**Cost to run one held-in smoke: `UNVERIFIED`.** The paper's oracle unit maps
cleanly onto ours — 1,000 oracle calls is roughly one weight vector, one run —
but no timing is reported anywhere, and the repo cannot complete a run as
shipped. A cost figure would be invention.

### OP-GFN — excluded

| field | value | source |
|---|---|---|
| paper | Chen, Mauch, *Order-Preserving GFlowNets*, ICLR 2024 | arXiv:2310.00386 |
| code | `github.com/yhangchen/OP-GFN` @ `369e910891d3c9e20b8272d8291310fb9cc11c4d` (2024-03-11) | clone, verified |
| licence | **CC BY-NC-ND 4.0** — NonCommercial **and NoDerivatives** | `LICENSE.md:1` |
| preference conditioning | **absent in OP-GFN's own mode**: `preference_type` is `None` unless `--type pref`, and `--type pref` is the PC-GFN baseline | `multi/gflownet/tasks/seh_frag_moo.py:388,398,414` |
| objectives | asserted closed to `{seh, qed, sa, mw}` | `seh_frag_moo.py:62` |
| training | 20,000 steps at batch 64, `"device": "cuda"` | `seh_frag_moo.py:371-380` |

**Excluded, on three independent grounds, any one sufficient.** It is not
preference-conditioned, which is the property under study. Its objective set is
closed and excludes any COMPOSE-compatible potency axis. And **NoDerivatives**
means the thin adapter that would make it runnable here is the thing the licence
does not permit us to distribute — an engineering-proof stop.

---

## What Panel B needs before it can be frozen

Panel B requires an objective pair both methods compute **natively**, under a
budget both can spend, with a reference point frozen before any outcome.

The one verified asset that makes this cheap: **HN-GFN's DRD2 model file is the
same file COMPOSE's frozen oracle was extracted from** — sha256
`dbc473fca922c834dbaee6eaba832caaff26d4f891734078fb1af359a111100f`,
35,417,609 bytes, matching `pickle_sha256` and `pickle_bytes` in
`artifacts/oracles/drd2_svm_v1/drd2_oracle_manifest.json`, with the same
fingerprint definition. So a DRD2 axis is shared without adaptation.

The one gap: InversionGNN has no DRD2. A Panel B pair of `{DRD2, QED}` admits
HN-GFN natively and excludes InversionGNN; a pair of `{GSK3β, JNK3}` admits both
externals and excludes COMPOSE's frozen potency axis. **This is a real design
tension and it must be resolved by a freeze made before outcomes exist**, not by
whichever choice reads better afterwards. Options are laid out in `PROTOCOL.md`;
the decision is main's.

---

## Where the methods sit — three rings

Recorded from the lead's redirect of 2026-08-13. The purpose of the rings is to
**answer scientific objections, not to beat task specialists**. Placement only;
this is not an audit of Ring 1 or Ring 2, and nothing below is asserted beyond
what has been verified elsewhere in this directory.

### Ring 1 — same methodological realm

| method | placement | status here |
|---|---|---|
| DDSBM | graph CTMC / source-conditioned transformation | not audited by this lane; `baselines/ddsbm/` exists on `codex/compose-baseline-qualification` |
| Edit Flows | edit-based CTMC lineage | **related work only** — its published state space is variable-length *sequences*, and a homemade molecular-graph port would mean solving the support/executor problem that is part of COMPOSE's own contribution |
| GraphXForm | source-conditioned graph editing | not audited by this lane; qualified by the predecessor lane |
| InVirtuoGen | named in the redirect | **not audited, no primary source read by this lane — `UNVERIFIED`** |

### Ring 2 — hard-constraint control

CDD, PRODIGY, ConStruct. **Lane 6 owns this audit. Not duplicated here**, and no
claim about these methods appears anywhere in this workstream.

### Ring 3 — COMPOSE causal controls

Possibly more important than either external ring: hard mask versus post-hoc;
frozen `R_θ` versus empirical family; closed-loop versus generate-and-rank;
continuation versus restart; endpoint versus trajectory constraint.

These are Category 2 comparators under `docs/BASELINE_IMPLEMENTATION_POLICY.md` —
scientific controls we implement ourselves because they exist only as
counterfactuals to COMPOSE. They are Panel A members. Lane 4 owns the runs.

**The audit's practical consequence for emphasis.** With Panel B unable to carry
a head-to-head row (see `BENCHMARK_ALIGNMENT_AUDIT.md`), the multiobjective
argument rests on Ring 3 and Panel A. That is where the scientific content was
anyway; the audit removes the option of leaning on an external row instead.

---

## Cost summary

| item | cost | basis |
|---|---|---|
| Stage 0 and Stage 1 (this deliverable) | **0 compute** | local reads, four clones, one test file |
| HN-GFN, one held-in smoke | **10–13 GPU-hours, V100-class** | authors' Appendix B.4 |
| HN-GFN environment resolution | ~1 day of engineering, BoTorch version bisection, no compute | upstream issue #1 |
| InversionGNN, one held-in smoke | **UNVERIFIED**; CPU-capable by `device='cpu'`, blocked by two code defects | repo |
| InversionGNN surrogate pretraining | 10K oracle calls (2-objective), hardware `UNVERIFIED` | paper §5.3 |
| OP-GFN | not applicable — excluded | licence |

The real cost of this lane is not compute. It is environment archaeology, and it
is concentrated entirely in HN-GFN.
