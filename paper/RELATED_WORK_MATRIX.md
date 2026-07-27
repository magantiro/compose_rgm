# RELATED_WORK_MATRIX.md

Differentiator matrix for the related-work section. Columns: **Method/family**, **Category**, **State
representation**, **Every intermediate a complete molecule?**, **Steering mechanism**, **Exact /
distributionally-principled control?**, **Dynamic retarget / pathwise?**, **Relation to us**,
**Citation (verification)**. Every citation was checked against a primary source (arXiv / OpenReview /
publisher DOI) in a dedicated pass; venue caveats are noted. This matrix is an authoring aid; the paper's
prose is `sections/related.tex`.

| Method/family | Category | State representation | Every intermediate a molecule? | Steering mechanism | Exact/principled control? | Dynamic retarget / pathwise? | Relation to us | Citation (verification) |
|---|---|---|---|---|---|---|---|---|
| Junction-tree / hierarchical VAE | latent graph generator | latent code + tree | no (molecule at decode) | latent optimization | no | no | ambient/latent contrast | Jin et al. 2018/2020 (VERIFIED) |
| Graph-to-graph / Modof | paired translation / fragment edit | graph + junction tree | endpoints only | supervised translation | no | no | supervised translator vs our source-agnostic prior | Jin et al. 2019; Chen et al. 2021 (VERIFIED; Modof = Nat. Mach. Intell.) |
| MolDQN | RL atom/bond editor | molecular graph | yes | Q-learning reward | no (reward, not distributional) | limited | valid editing, no learned path law / exact control | Zhou et al. 2019 (VERIFIED) |
| RetMol | retrieval-controlled generation | frozen generator + exemplars | no (decode) | retrieval fusion | no | no | control via retrieval vs Doob | Wang et al. 2023 (VERIFIED) |
| GenMol | masked discrete-diffusion generalist | SAFE fragment sequence (masked) | no (molecule at decode) | fragment remasking + context guidance | no distributional guarantee on discrete space | partial (masking) | masked generalist; our states are always molecules | Lee et al. 2025 (VERIFIED; ICML 2025) |
| InVirtuoGen | discrete-flow fragment model | fragmented SMILES | no | GA + proximal property opt | no | no | discrete flow refinement; different state | Kaech et al. 2025 (VERIFIED; preprint, cite paper title) |
| MARS | annealed fragment-edit MCMC | molecular graph | yes | MCMC + adaptive proposal | no (sampler, not exact tilt) | no | heuristic valid editing vs learned GM process | Xie et al. 2021 (VERIFIED) |
| GraphGA | graph genetic algorithm | molecular graph | yes | GA mutation/crossover | no | no | heuristic proposals; fair baseline on our operators | Jensen 2019 (VERIFIED) |
| Reaction GFlowNet (RGFN) | reaction-based flow sampler | synthesis DAG | yes (synthesizable) | GFlowNet flow | reward-proportional, not Doob | no | executable transitions; different control object | Koziarski et al. 2024 (VERIFIED) |
| MolEditRL | structure-preserving diffusion editor | discrete diffusion | endpoints | RL reward | no | partial | learned multi-step editing (replaces the nonexistent "SMER-Opt") | arXiv:2505.20131 (VERIFIED; substitute) |
| MolWorld | reachability/world-model editor | molecule-transfer graph (MMP edges) | yes | world-model reachability | no distributional control | reachability, not exact | reachability-aware valid editing vs exact path control | arXiv:2605.08954 (VERIFIED; 2026 preprint) |
| MMP / mmpdb | transformation supervision | fragmented pairs | n/a (data) | n/a | n/a | n/a | source of our analogue training layer | Hussain & Rea 2010; Dalke et al. 2018 (VERIFIED) |
| NSGA-II/III, MOEA/D, SMS-EMOA | evolutionary multi-objective | population | n/a | selection/decomposition/HV | no learned process | no | metric machinery; fair over our operators | Deb et al. 2002/2014; Zhang & Li 2007; Beume et al. 2007 (VERIFIED) |
| MO-GFlowNets, goal-conditioned GFN | Pareto-diverse sampler | constructive DAG | yes | reward/preference conditioning | reward-proportional | preference-conditioned | Pareto sampling; our control is a Doob tilt | Jain et al. 2023; Bengio et al. 2021/2023 (VERIFIED) |
| ParetoFlow | guided flow, offline MOO | continuous design | n/a (not molecular sequences) | predictor guidance | no exact discrete tilt | no | continuous offline MOO, orthogonal setting | Yuan et al. 2025 (VERIFIED; ICLR 2025) |
| MOG-DFM | discrete-FM multi-objective guidance | pretrained discrete FM | depends on base | rank-directional + hypercone | heuristic guidance | rank-directional | ORTHOGONAL; ported over our kernel in the same-base table | Chen et al. 2025 (VERIFIED; ICML 2025 GenBio Workshop, NOT main proceedings) |
| AReUReDi | discrete MO refinement | SMILES | yes | Tchebycheff + locally-balanced MH | invariance-preserving MH | annealed | MH over SMILES; different control | arXiv:2510.00352 (VERIFIED; preprint under review) |
| Generator Matching | generative principle | arbitrary Markov generator | n/a | n/a | n/a | n/a | the regression principle we instantiate | Holderrieth et al. 2025 (VERIFIED; ICLR 2025 oral) |
| Flow / discrete flow / CTMC diffusion | generative flows | tensors / sequences / CTMC | typically no | learned velocity/rates | n/a | n/a | ambient generative dynamics contrast | Lipman 2023; Gat 2024; Campbell 2022/2024; Lou 2024 (VERIFIED) |
| Doob h-transform; Schrödinger bridge | stochastic control | Markov process / SDE | n/a | conditioning / bridge | exact conditioning (classical) | n/a | the classical device our controller is | Doob 1984; Rogers-Williams 2000; De Bortoli 2021; Zhou 2024 (VERIFIED) |
| Classifier/reward guidance; reward/adjoint fine-tuning | guidance / fine-tuning | diffusion/flow | n/a | gradient/reward; SOC | no discrete distributional guarantee; changes weights | no | steer a fixed prior distributionally, not fine-tune | Dhariwal 2021; Ho 2022; Black 2024; Domingo-Enrich 2025; Wang 2025 DRAKES; Zekri 2025 SEPO (VERIFIED; discrete-adjoint proper analog is a 2026 preprint, omitted) |
| Twisted SMC / Feynman-Kac correctors | inference-time steering | particles over base | n/a | reweight + resample | asymptotically exact | n/a | our optional accuracy knob, not the default | Wu 2023; Skreta 2025; Singhal 2025 (VERIFIED; Singhal = arXiv preprint) |
| PMO / sample-efficiency (GP-BO, Genetic GFN, Saturn/Aug. Memory, REINVENT) | sample-efficiency benchmark/methods | various | various | RL/BO/GFN | no | no | breadth/efficiency context (SEGO does not exist -> Saturn) | Gao 2022; Tripp 2021; Kim 2024; Guo 2024; Loeffler 2024 (VERIFIED) |
| Stochastic graph rewriting | process semantics | rewrite state graph | yes | n/a | n/a | n/a | the rewriting semantics we build on | Danos et al. 2015; Behr et al. 2021 (VERIFIED) |

## Verification notes and corrections folded into references.bib
- **SMER / SMER-Opt does NOT exist** as a primary source; replaced with **MolEditRL** (arXiv:2505.20131).
- **MolWorld** is a **2026** preprint (arXiv:2605.08954), not 2024-2025.
- **InVirtuoGen**: cite the paper title "Refine Drugs, Don't Complete Them..." (arXiv:2509.26405); the model name is InVirtuoGen.
- **MOG-DFM**: ICML 2025 **GenBio Workshop spotlight** (arXiv:2505.07086), NOT main-conference proceedings; framed ORTHOGONAL, ported over our kernel, never an external baseline.
- **AReUReDi**: arXiv:2510.00352, preprint under review.
- **ParetoFlow**: ICLR 2025; targets **offline continuous** MOO, not molecular sequences.
- **SEGO does not exist** as a molecular method; nearest real 2024 sample-efficient work is **Saturn** (Guo & Schwaller 2024). SEISMO exists but is a single-objective 2026 LLM agent -> not grouped with Pareto methods.
- **Discrete Adjoint Matching** (the exact analog) is a Feb-2026 preprint past the knowledge cutoff; we instead cite the solidly-verified DRAKES (ICLR 2025) and SEPO (2025) for discrete fine-tuning, correctly characterized as reward/policy-gradient, not adjoint.
- **Modof** = Nature Machine Intelligence 2021 (journal). **Normalized hypervolume** has no single canonical source; we cite Zitzler & Thiele 1999 for the indicator.
