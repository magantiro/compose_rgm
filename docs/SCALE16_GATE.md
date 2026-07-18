# 1,333-molecule C/N/O/F scale gate

## Question

Can the validity-closed rewrite generator scale beyond the 128-molecule pilot,
and can learned state dependence improve on a matched stochastic-rewrite prior
without sacrificing the pathwise guarantee?

This is an intermediate model-development gate, not a benchmark comparison.

## Protocol

- Corpus: full local 50,000-molecule GuacaMol scan.
- Eligibility: unique, neutral, connected C/N/O/F molecules with at most 16
  heavy atoms and supported standard valence.
- Eligible molecules: 1,333.
- Random split: 1,067 train / 133 validation / 133 test.
- Teacher data: two randomized valid null-to-target rewrite traces per training
  molecule.
- Model selection: conditional Generator-Matching loss on fixed validation
  examples; the test set is used only after selection.
- Sampling: target-free ancestral CTMC sampling from the null state, horizon
  7.0, at most 32 events.
- Evaluation: 200 generated samples against the 133-molecule test set. FCD at
  this sample size is indicative rather than a benchmark-quality estimate.

A Bemis-Murcko scaffold partition is also implemented. At the same requested
sizes, indivisible scaffold groups produce 1,015 / 174 / 144 molecules with
zero cross-split scaffold overlap. The random split is retained for this gate
to isolate rate-learning behavior before the harder generalization test.

## Matched models

All three generators use the identical exact legal action fiber, rewrite
executor, compiler family, operational CTMC clock, and ancestral sampler.

1. **Corpus-marginal rewrite prior.** A non-neural time-dependent policy fitted
   to training rule-family and action-signature frequencies. It has no graph-
   state dependence.
2. **Uniform-base neural generator.** The earlier neural model learns the full
   rate distribution from an uninformative initialization (1,200 updates).
3. **Prior-tilted RGM.** A neural residual exponentially tilts the matched
   corpus prior's legal action probabilities and total hazard (600 updates).
   It equals the prior exactly at initialization, preserves zero support, and
   retains exact endpoint hazard shutoff.

The prior-tilted checkpoint selected by validation loss occurred at update 100;
later updates overfit. This is an optimization signal, not a reason to use the
last checkpoint.

## Results

Reference means are 13.97 atoms and cycle rank 1.707. Lower is better for all
TV, FCD, and SA columns.

| Metric | Corpus prior | Uniform neural | Prior-tilted RGM |
|---|---:|---:|---:|
| Valid / connected / non-null | 100% | 100% | 100% |
| Unique | 100% | 100% | 99.5% |
| Mean atoms | 13.37 | 14.22 | 13.13 |
| Mean cycle rank | 2.17 | 2.23 | **1.67** |
| Atom-count TV | 0.279 | 0.277 | **0.239** |
| Bond-order TV | 0.088 | 0.065 | **0.021** |
| Cycle-rank TV | **0.226** | 0.345 | 0.295 |
| Element TV | 0.031 | **0.017** | 0.054 |
| Ring-size TV | 0.274 | 0.596 | **0.252** |
| Fused fraction (reference 0.338) | 0.535 | 0.520 | **0.405** |
| Bridged fraction (reference 0.060) | 0.455 | 0.335 | **0.300** |
| Spiro fraction (reference 0.030) | 0.150 | 0.180 | **0.145** |
| FCD | **20.38** | 22.08 | 21.07 |
| Mean QED (reference 0.638) | **0.528** | 0.433 | 0.445 |
| Mean SA (reference 2.644) | 5.401 | 5.795 | **5.173** |

The prior-tilted model therefore beats the matched prior on atom count, bond
orders, ring sizes, mean cycle rank, fused/bridged/spiro calibration, and SA.
It does not beat the prior on FCD, QED, cycle-rank TV, or element frequencies.
This is evidence that learned graph-state dependence is useful, but it is not
yet a decisive overall distributional win.

## Why fused and bridged ring statistics are still high

This failure is topology-frequency miscalibration, not a failure of chemical
validity. Every closure is legal and every intermediate molecule validates.
However:

1. A state with many atoms exposes combinatorially many legal atom pairs.
2. A small fraction of closure probability spread across that large fiber can
   create too many rings.
3. Local valence legality does not determine whether a closure produces the
   corpus-typical ring topology.
4. The corpus prior conditions closures on time and cycle-size signature but
   not the full molecular state, so it strongly overproduces fused and bridged
   systems.
5. The neural tilt learns a substantial correction, but conditional GM loss is
   only an indirect checkpoint criterion for terminal ring-system calibration.

Calling this “low fused recall” is slightly misleading: the model produces too
many fused systems. The harder remaining question is whether it covers the
right fused motifs at the right frequency, which requires larger scaffold-
aware evaluation and motif-level statistics.

## Decision and next experiment

The architectural direction survives the gate. The next run should keep the
prior-tilted parameterization and change optimization/evaluation, not abandon
the rewrite generator:

1. Add residual regularization toward the corpus prior and reduce the learning
   rate; the early selected checkpoint shows rapid overfitting.
2. Report validation metrics stratified by acyclic, monocyclic, fused,
   bridged, and spiro targets, and use a validation-only composite selection
   rule alongside GM loss.
3. Make ring-closure topology context explicit in the action residual while
   preserving the same legal action fiber and GM objective.
4. Run the zero-overlap scaffold split after the random-split optimization is
   stable.
5. Only then expand beyond C/N/O/F and 16 heavy atoms.

The gate supports the paper thesis—learned executable rewrite generators are
viable and their state dependence improves important structural statistics—
while locating the next bottleneck in distribution calibration.

## Follow-up: topology-aware, order-deferred RGM

The first gate exposed two concrete problems: the neural tilt overfit quickly,
and null-to-target traces assigned final bond orders during construction. The
latter provided almost no de novo supervision for `bond_reorder` and required
aromatic Kekule patterns to emerge from several independently chosen build
decisions.

The follow-up makes three changes while preserving the state space, executor,
GM objective, clock, and ancestral sampler:

1. add cycle-size-if-bonded, current atom ring membership, and resulting ring-
   system size to the action encoding;
2. regularize action and total-hazard tilts toward the corpus rewrite prior;
3. compile topology first with valid single-bond skeletons, then recover double,
   triple, and Kekule patterns through valid `bond_reorder` events.

The model uses learning rate `5e-4`, weight decay `1e-5`, action KL weight
`0.05`, and log-hazard-tilt penalty `0.01`. The selected checkpoint is update
600, indicating that the trust region removed the previous rapid-overfitting
failure. Evaluation again uses 200 learned and 200 matched-prior samples.

| Metric | Deferred-order corpus prior | Topology-aware regularized RGM | Test reference |
|---|---:|---:|---:|
| Valid / connected / non-null | 100% | 100% | — |
| Unique | 100% | 100% | — |
| Mean atoms | 13.33 | **13.96** | 13.97 |
| Mean cycle rank | 2.17 | **1.61** | 1.71 |
| Atom-count TV | 0.271 | **0.259** | 0 |
| Bond-order TV | 0.0766 | **0.0024** | 0 |
| Cycle-rank TV | 0.254 | **0.094** | 0 |
| Ring-size TV | 0.284 | **0.054** | 0 |
| Fused fraction | 0.560 | **0.395** | 0.338 |
| Bridged fraction | 0.450 | **0.245** | 0.060 |
| Spiro fraction | 0.235 | **0.075** | 0.030 |
| Mean aromatic rings | 0.025 | **0.130** | 1.105 |
| FCD | 22.24 | **19.42** | — |
| Mean QED | 0.511 | **0.539** | 0.638 |
| Mean SA | 5.727 | **4.771** | 2.644 |

The learned generator uses `bond_reorder` for 20.8% of its events and improves
aromatic-ring recovery fivefold over its matched prior. It also beats the old
non-deferred prior's FCD of 20.38. This is the first scaled overall-quality win
from learned state dependence, rather than a trade among structural metrics.

The remaining aromaticity gap is severe: aromatic atom fraction is 0.049 versus
0.426 in the test set. Sequential bond reorders are therefore useful but still
an inefficient representation of a correlated ring-system electronic change.
The next justified macro is a reversible, validated ring-system aromatization /
dearomatization transaction that supports heteroaromatics and fused systems and
quotients alternative Kekule lowerings at the same chemical successor. The
recommended hierarchy is specified in `RING_REWRITE_ARCHITECTURE.md`.
