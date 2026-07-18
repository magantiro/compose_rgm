# First learned whole-graph rate gate

## Question

Can a neural generator learn complete-successor jump rates from
endpoint-conditioned rewrite processes and generate molecules from null without
a target, beam search, fragment vocabulary, or chemical repair?

## Deliberately small setup

- Uniform reference over 12 neutral C/N/O/F molecules with at most three heavy
  atoms, including single, double, triple, and cyclic structures.
- Formal null source with three atom slots.
- Exact connected action fiber over all six micro families: atom
  insert/delete/restate and bond insert/delete/reorder.
- Random valid compiler traces and binomial-progress conditional CTMCs.
- Cumulative-hazard operational clock, which removes the physical-time rate
  singularity.
- Six interior times and 84 canonical chemical state-time groups.
- Exact conditional-to-marginal averaging is used only because this tiny path
  mixture is enumerable. It is a diagnostic oracle for the learning theorem;
  scalable training will estimate the same target from conditional samples.
- Permutation-equivariant whole-graph network, hidden width 48, two message-
  passing layers, 400 minibatch steps.

The first version incorrectly marginalized raw padded arrays. Correctly
quotienting atom-slot layouts into canonical chemical states reduced the rate-
vector error by more than an order of magnitude and fixed endpoint-frequency
distortion. This is direct evidence that complete-state aggregation is
practical, not merely formal.

## Rate-fit result

```text
metric                                  initial       final
mean rate-vector L1 error                 2.252       0.0146
mean total-hazard absolute error          1.759       0.0125
predicted mass on teacher support        24.72%       99.95%
top complete-successor accuracy           0.00%      100.00%
mean hazard at zero-target states         1.983       0.000005
```

## Frozen-checkpoint target-free evaluation

At operational horizon 5.0 and time step 0.1, 1,000 independent ancestral
rollouts produced:

```text
valid                                      100.0%
connected or null                          100.0%
non-null                                    99.8%
inside the 12-molecule reference support   99.5%
reference modes recovered                  12 / 12
reference total-variation distance          0.0523
event-budget exhaustion                      0.0%
mean committed rewrites                      1.972
```

The five off-support outcomes were still legal states: two null endpoints, two
`CCC`, and one `CC#N`. No target, target trace, or endpoint oracle was available
to the sampler.

Time-step checks at 0.20, 0.10, and 0.05 gave total-variation distances 0.0617,
0.0523, and 0.0590 respectively, with 100% validity and all reference modes in
every run. There is no observed coarse-step failure at this scale.

## What this establishes

- The conditional progress generator, marginal Generator Matching target,
  neural complete-successor rates, hard chemical action fiber, and ancestral
  sampler work together end to end.
- Null is a viable source for this gate.
- Stochastic rewriting contributes operational value: invalid and fragmenting
  transitions have zero support, while exact rule execution makes every sampled
  trajectory auditable.
- Canonical chemical quotienting is necessary for accurate learning.

## What this does not establish

- Competitive molecular-generation quality or novelty.
- Scaling exact fiber enumeration beyond very small molecules.
- Success on held-out molecules, larger rings, stereochemistry, properties, or
  drug-like chemistry.
- Resolution of the hypervalent-sulfur compiler limitation.
- Superiority to Edit Flows, diffusion, Morph, DDSBM, or fragment baselines.

## Next gate

Replace exhaustive candidate enumeration with factorized operator and operand
heads whose masks encode the same exact action fiber. Then train by conditional
Generator Matching on a non-enumerated C/N/O/F corpus with a held-out split.
The tiny exact marginal oracle remains only a regression test.
