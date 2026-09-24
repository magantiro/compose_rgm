# PMO split-clean route-prior production-yield gate

## Scientific question

This zero-oracle gate asks whether exact PMO route evidence can improve the
production proposal process before online FiberControl sees any reward. It does
not retrieve or execute a teacher route.

The exact training-only lineage contains 184 source/action/intermediate traces and
a dependency-region representation that replays every trace. The frozen split
holds out whole task families and every shared lineage before fitting. Celecoxib
rediscovery uses the fold that excludes all rediscovery routes, GSK3B uses the fold
that excludes all bioactivity routes, and Perindopril MPO uses the fold that
excludes all MPO routes.

## Runtime route proposer

For each fold, training routes are reduced to a source-balanced transition table
over generic executor rules and STOP. The checkpoint contains only aggregate
probabilities, numeric identities and support constants. It contains no task,
family, source molecule, route, endpoint, action, binding or teacher trace.

At runtime the proposer samples one rule from that transition prior, enumerates the
complete legal Active8 fiber for that rule on the current molecule, and samples a
bound action using the already-qualified held-family legal-action ranker with a
10% exploration floor. Every chosen action is exact-executed. The completed
program is independently replayed through the dependency-region representation.
The proposal horizon is eight primitives for this prelaunch gate; it is not a
claim that eight primitives cover the full PMO route distribution.

## Equal-attempt comparison

The frozen initialization contains 16 task-independent source molecules. Each task
and source receives two proposal attempts per arm:

* old Dynamic-v0: two ordinary v0 draws;
* additive route: the exact same first v0 draw plus one route-prior draw.

This preserves a 50% unchanged exploration floor and matches program-attempt
counts. Compute is intentionally not assumed equal, so wall time, legal fibers and
exact-execution work are reported. The output reports validity, unique endpoint
yield, exact-execution precision, primitive and component counts, failures and
wall time. It uses no task answer, winner or PMO score.

Run with:

```bash
/Users/rmaganti/compose_rgm_git/.venv/bin/python \
  tools/pmo_route_fiber_production_yield.py
```

Outputs are `configs/pmo_route_fiber_production_yield_v3.json`,
`diagnostics/pmo_route_fiber_pilot/route_transition_checkpoint_v3.json` and
`diagnostics/pmo_route_fiber_pilot/production_yield_v3.json`.

Passing this gate shows only that route distillation changes executable production
yield. A PMO utility or IVG claim requires a separately locked scored pilot.
