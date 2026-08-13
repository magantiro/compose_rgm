# Positioning against the discrete-edit-process lineage

Written before the Pareto smoke reported, because the honest version of this is
easier to write now than after we have a hypervolume number we like.

## Pareto control is NOT the core novelty

Preference-conditioned and Pareto-directed molecular generation is an
established line. **HN-GFN** (NeurIPS 2023) is a preference-conditioned
hypernetwork GFlowNet built specifically to explore multiobjective molecular
Pareto fronts with sample efficiency. **pCoMole** (2026 workshop) does
Pareto-constrained biomolecular *sequence* editing on a pretrained Edit Flow,
using a preference tilt / Doob-`h` construction and short Monte Carlo rollouts
to approximate future desirability.

That last one is conceptually very close to part of what this project does, and
it comes from the same broader research lineage. **Be explicit about it rather
than distancing artificially.** A reviewer who knows it and finds us quiet about
it will assume the worst.

So Pareto control in this paper is **evidence that the control abstraction
generalizes to target-free goals** — not a claim to have invented preference
control.

## What is actually differentiating

The **executable molecular graph process**:

- a complete, valid molecule at *every* state, not only at the endpoint;
- an exact chemical rewrite kernel with graph-level insert / delete / reroute /
  ring operators;
- a canonical successor pushforward;
- exact bounded-space verification of the control law;
- **continuation from a realized molecular graph after the goal changes**;
- trajectory-level constraints on intermediate molecular states.

Every experiment in the paper is a consequence of that one object. The claim is
about *what kind of thing a molecule is* during generation — a persistent state
in an executable stochastic process — not about owning any single control task.

## Edit Flows: related work, not a baseline

**Edit Flows** (NeurIPS 2025) defines a CTMC over variable-length *sequences*
with insertion, deletion and substitution. Its published experiments are text,
code and image captioning — not molecular graphs.

Making it a molecular baseline would require inventing "Edit Flows, but over
molecular graphs with chemical legality." **That is building a new method for
them and then comparing against our own construction of it**, which is worth
less than no comparison at all. No public molecular Edit-Flows implementation
exists to use instead.

Sanctioned framing:

> Edit Flows establishes CTMC-based insertion/deletion/substitution generation
> over variable-length sequences. pCoMole shows preference-tilted Doob control
> of such sequence edit processes for Pareto-constrained biomolecular editing.
> COMPOSE instead develops an executable stochastic reference process over
> chemically valid molecular *graphs*, and studies inference-time control of
> graph-rewrite trajectories — including exact-state retargeting and
> trajectory-level intervention.

## The asymmetric evaluation burden

| setting | question | burden |
|---|---|---|
| conventional optimization | is COMPOSE competitive? | establish practical credibility; **we do not need to win** |
| hard finite-horizon editing | does future reachability matter? | sealed, 40% → 62% |
| same-realized-state retargeting | can the goal change mid-trajectory? | **the distinctive claim** |
| pathwise constraints | is endpoint acceptability sufficient? | conditional on Stage B |
| Pareto efficiency | can one frozen process be recontrolled across preferences? | efficiency, not final-HV supremacy |

The conventional table exists to show this is not an elegant control framework
that is useless at ordinary molecular design. It does not exist to show COMPOSE
is the highest-scoring generator on every benchmark. **The positive burden
belongs on the structure-enabled experiments.**
