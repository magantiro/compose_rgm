# Unconditional sufficiency and transition to conditional generation

**Date:** 2026-07-20  
**Decision scope:** what COMPOSE must establish unconditionally before its
property-targeting and molecular-optimization program can proceed.

## Executive decision

COMPOSE does **not** need to win unconditional FCD, nor does it need a main-text
taxonomy of fused, bridged, and spiro rings, before conditional training begins.
It does need a credible and transparent unconditional base: final and pathwise
validity/connectivity, diversity, novelty, size flexibility, coarse distribution
fidelity, absence of executor pathologies, and enough support for the chemistry
required by the downstream tasks.

The peer-review standard depends on the claim. Distribution-learning papers such
as DiGress and EDM-SyCo make unconditional fidelity a headline and therefore
report FCD. Conditional-design papers such as GrIDDD and training-free
multi-objective diffusion do not necessarily report FCD at all. GrIDDD gives a
small unconditional control and moves directly to property targeting,
optimization, and out-of-distribution size control. Morph reports a compact set
of validity/stability/diversity and 3D-geometry checks, then trains task-conditioned
variants. None of the audited papers reports a systematic fused/bridged/spiro
ring-distribution table.

Accordingly, COMPOSE should freeze a sufficient unconditional baseline and begin
conditional engineering now. Ring-topology diagnostics remain valuable because
COMPOSE explicitly introduces whole-ring rewrite operators, but they belong in a
mechanism appendix and regression gate rather than serving as a prerequisite for
the conditional program.

## Primary-source audit

| Work | Venue/status | Unconditional evidence actually reported | FCD? | Granular ring topology? | How conditional capability is obtained |
|---|---|---|---|---|---|
| [GrIDDD](https://papers.nips.cc/paper_files/paper/2025/file/70f286e0fc977c0a3a64ef96849c8d7d-Paper-Conference.pdf) | NeurIPS 2025 | Unconditional validity appears as a control in property-targeting tables. A 100-sample padding ablation reports validity, connected components, single-component count, and node/edge cross-entropy. | No occurrence in the paper. | No fused/bridged/spiro audit. | The conditioning vector is injected into the denoiser during training using classifier-free guidance. Optimization starts from a corrupted lead and supplies the target-property vector. |
| [DiGress](https://openreview.net/forum?id=UaAD-Nu86WX) | ICLR 2023 | Standard MOSES/GuacaMol distribution learning: validity, uniqueness, novelty, filters, FCD, nearest-neighbor/scaffold and KL measures. | Yes. | No fused/bridged/spiro table. | Classifier-based guidance with an auxiliary property regressor over noisy graphs. |
| [MAGNet](https://proceedings.iclr.cc/paper_files/paper/2025/file/43d1d3bdd92204c96fa4ac3c578f6a33-Paper-Conference.pdf) | ICLR 2025 | MOSES/GuacaMol metrics plus scaffold-distribution and motif-MMD analyses; the authors explicitly argue FCD alone is insufficient for structural diversity. | Yes. | Ring-like versus chain-like scaffolds and uncommon rings are discussed, but no fused/bridged/spiro distribution table. | Latent optimization for properties and constrained decoding for scaffold/fragment conditioning. |
| [EDM-SyCo / Lift Your Molecules](https://proceedings.iclr.cc/paper_files/paper/2025/file/2af641762dc02035c31a9314b2d090b6-Paper-Conference.pdf) | ICLR 2025 | Unconditional molecular distribution learning is a headline; reports validity, uniqueness, novelty, FCD and KL-style distribution metrics, plus coarse invalidity causes. | Yes. | No fused/bridged/spiro table. | Adapts sampling with regressor guidance and inpainting; also studies a property-conditioned variant. |
| [Training-free Multi-objective Diffusion](https://proceedings.iclr.cc/paper_files/paper/2024/file/c8ff6807d1f362bb22b4f0be7b66e9ca-Paper-Conference.pdf) | ICLR 2024 | Uses pretrained unconditional EDM/GeoLDM rather than re-establishing an unconditional benchmark suite. | No occurrence in the paper. | No granular ring audit. | Plug-and-play property guidance; the paper contrasts this with retraining a conditional model for every target or training a time-dependent property predictor. |
| [Context-Guided Diffusion](https://proceedings.mlr.press/v235/klarner24a.html) | ICML 2024 | Treats the pretrained diffusion model as a base and evaluates task-specific conditional and OOD performance. | Not a reported headline metric. | No granular ring audit. | Plug-and-play context guidance on the pretrained generator. |
| [Morph](https://arxiv.org/pdf/2606.07239) | 2026 preprint; not yet a top-tier accepted precedent | QM9/GEOM-Drugs stability, validity, uniqueness, novelty, and PoseBusters-style geometry/chemistry checks. | No occurrence in the paper. | No fused/bridged/spiro distribution table. | Trains models conditioned on atom count or molecular property; scaffold decoration uses a task-specific matching construction. |

## What this means for COMPOSE

### Main-text unconditional sufficiency package

Report the following from a fixed, predeclared sample set:

1. Final validity, connectivity, uniqueness, and novelty.
2. Pathwise validity/connectivity and the rate of virtual self-events,
   immediate backtracks, collapse, and event-budget exhaustion.
3. Flexible-size behavior: atom-count distribution, growth/shrinkage events,
   and successful generation from multiple prior sizes.
4. Coarse molecular fidelity: element, bond, size, cycle-rank, QED, and SA
   distributions, together with representative samples.
5. Runtime and event count under the actual ancestral CTMC sampler.
6. FCD as a secondary diagnostic if computed with a sufficiently large,
   clearly stated sample size. It is not the paper's headline gate.

This supports the bounded claim that the same rewrite process is a credible
unconditional de-novo generator. It does not support a claim of state-of-the-art
unconditional distribution matching unless the corresponding benchmarks win.

### Appendix and regression diagnostics

Keep ring size, fused, bridged, spiro, aromatic, and small-ring prevalence in
the appendix and in development regression tests. The literature audit shows
that this granularity is not a standard admission criterion. COMPOSE has an
extra reason to retain it: whole-ring rewrites and valid-state closure make ring
topology part of the claimed mechanism, and the diagnostics catch cages,
macrocycles, over-fusion, and electronically implausible rings that aggregate
validity can hide.

The current calibrated pancake result is therefore sufficient to start the
conditional lane, but not to declare unconditional chemistry solved. Its
remaining cycle-rank, fusion, QED, and SA gaps should be recorded as known
baseline limitations and improved in parallel only through bounded, evidenced
changes.

## What “engineering the model for conditional generation” legitimately means

Top-tier papers do not all freeze an unconditional network and merely append an
oracle. They commonly choose one of four task-specific routes:

1. **Direct conditional training.** Inject the requested property or context
   into the model and train conditional transition probabilities/rates.
2. **Guidance on a pretrained base.** Freeze or retain the base generator and
   add a classifier, regressor, energy, or context-guidance term at sampling.
3. **Conditional fine-tuning or adapters.** Preserve the base support and learn
   small target-specific residuals, schedules, or control heads.
4. **Task-specific transition construction.** Use inpainting, scaffold masks,
   lead corruption, similarity constraints, or target-dependent size models.

For COMPOSE v1, the most defensible route is the already specified frozen-base
adapter:

\[
\lambda_f^z(x,t)=\lambda_f^0(x,t)\exp\Delta_f(x,t,z),
\qquad
p_f^z(y\mid x,t)=\operatorname{softmax}\!\left(\log p_f^0(y\mid x,t)+\delta_f(x,y,t,z)\right).
\]

The family residual changes **when and what class of valid rewrite fires**; the
normalized successor residual changes **which legal successor fires within that
family**. This is more natural for a valid-state editor than treating the model
as a black-box endpoint sampler, and it provides a clean no-double-counting
decomposition.

The experimental sequence remains:

1. Freeze the qualified unconditional base and its support.
2. Train the direct conditional adapter on QED/property targets.
3. Run native direct conditioning as the primary comparison with exactly 20
   candidates per starting molecule on the matched GrIDDD protocol.
4. Separately evaluate valid-state oracle/controller guidance under a matched
   oracle-call and compute ledger.
5. Compare direct, guidance-only, and combined control. Attribute any advantage
   to usable-oracle efficiency, constraint adherence, anytime valid candidates,
   or final targeting—not merely to 100% endpoint validity.

## Locked positioning

One validity-closed, non-monotone, flexible-size Rewrite Generator Matching
framework spans de-novo generation, targeting, optimization, and editing. The
unconditional model establishes that the learned CTMC is a credible molecular
generator. The distinctive evidence comes from what validity closure enables
under conditional control: every committed intermediate is executable, can be
scored without a repair step, and remains available as an anytime candidate.

That is the correct boundary between a sufficient unconditional result and the
paper's stronger conditional contribution.
