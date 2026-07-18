# Generator Matching versus flow matching and diffusion

## One-line answer

Generator Matching does not train a diffusion model and a flow model and then
choose between them. It selects a Markov process class and learns that process's
infinitesimal generator. Flow, diffusion, and jump processes correspond to
different generator forms.

## The common recipe

All three paradigms connect a tractable source distribution to a data
distribution through time-indexed probability paths. Generator Matching makes
the shared recipe explicit:

1. Construct an endpoint-conditioned process that reaches one datum.
2. Write its conditional infinitesimal generator.
3. Regress a model toward the marginal generator obtained by averaging the
   conditional generators given the current state.
4. Sample the learned process from the source without exposing a target.

The model class is determined by the generator chosen in step 2.

## Three generator forms

### Deterministic flow

For a continuous state `x` and velocity field `v_t(x)`,

```text
(L_t f)(x) = v_t(x) dot grad f(x).
```

Flow Matching learns the vector field and samples an ODE. There is no random
state transition once the initial sample and ODE solver are fixed.

### Diffusion

For drift `b_t(x)` and diffusion covariance `a_t(x)`,

```text
(L_t f)(x)
  = b_t(x) dot grad f(x)
    + 1/2 trace(a_t(x) Hessian f(x)).
```

Score/diffusion models learn information needed for a reverse SDE or its
probability-flow ODE. The second-order term represents continuous noise.

### Molecular rewrite jump process

For a molecule `G`, legal rewrites `a`, executor `T_a`, and rates `q_t`,

```text
(L_t f)(G)
  = sum_[a in A(G)] q_t(a | G) [f(T_a(G)) - f(G)].
```

COMPOSE learns the jump rates and samples a CTMC. The state can change topology
and dimension; no Euclidean velocity or Gaussian corruption is required.

## Why the distinction matters here

The natural infinitesimal object for an executable graph rewrite is a firing
rate, not a coordinate velocity. Generator Matching lets the chemistry
executor define possible state changes while the network learns how frequently
each enabled change should fire. A future 3D model could deliberately
superpose this topology jump generator with a coordinate flow, but the present
2D base model does not mix the two.

## Primary references

- Holderrieth et al., [Generator Matching: Generative modeling with arbitrary
  Markov processes](https://arxiv.org/abs/2410.20587).
- Lipman et al., [Flow Matching for Generative
  Modeling](https://arxiv.org/abs/2210.02747).
- Song et al., [Score-Based Generative Modeling through Stochastic Differential
  Equations](https://arxiv.org/abs/2011.13456).
