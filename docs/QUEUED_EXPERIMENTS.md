# Queued experiments — recorded before they are earned

Ideas written down **at the moment they were conceived**, with their trigger
conditions, so that if one is later run it cannot be mistaken for something
devised after a convenient pattern appeared in the data. Nothing here is
designed and nothing here is running.

---

## Pareto continuation / front traversal

**Status: `QUEUED_CONDITIONAL — NO DESIGN / NO RUN`**

**Trigger.** The larger fixed-preference Pareto development establishes frontier
quality and preference control strong enough to justify studying *reuse along
the front*. Not before. If that development returns weak preference response or
weak HV, this is not run at all — it is not a rescue.

**The contrast.**

```
continuation:   x_0 --w_i--> x_i --w_j--> neighbouring Pareto region
restart:        x_0 --w_j-------------->  the same region
```

**Primary scientific question.**

> Does retaining an already-realized useful molecular state reduce the work
> required to move to an adjacent design preference?

**Why it is COMPOSE-specific.** It is the Pareto analogue of what the
retargeting lane established for goal changes: the process is executable and
stateful, so an existing history is an asset rather than something to discard.
A method that only samples endpoints has no realized state to continue *from*.

**Known hazards, recorded now.**

- The comparison must hold controller, budget and objective parity, varying only
  the starting state — the same discipline the retargeting controller-parity
  audit had to retrofit after the fact.
- "Continuation is cheaper" is close to definitional if the continuation is
  given a shorter horizon. Budget must be matched on the axis the claim is made
  on, and the resource frontier reported rather than a single scalar.
- The neighbouring-preference pairs must be declared before any outcome is seen,
  not chosen as the pairs where continuation happens to help.

---

## Reference-prioritized shortlisting (`R_θ` plausibility shortlist)

**Status: `QUEUED_CONDITIONAL — NO DESIGN / NO RUN`**

Already recorded in `docs/PARETO_DEVELOPMENT_PREREGISTRATION.md`. Repeated here
so the queue is in one place.

**Trigger.** Full-fiber control must first establish the capability. Only then
is it worth asking whether

```
legal support → R_θ plausibility shortlist → expensive goal evaluation → control
```

retains most frontier quality at a fraction of the objective evaluations. A
scaling result, never a substitute for proving the capability.

---

## Gate-0 source-index portability fix

**Status: `QUEUED — DEFECT RECORDED, FIX DEFERRED`**

`source_index_sha256` embeds the absolute mount path, so identical content
mounted elsewhere fails authentication. Diagnosed in
`docs/PARETO_PARITY_ENVIRONMENT_STATUS.md`. Fixing it changes the semantics of a
frozen verification artifact and invalidates the recorded decision hash, so it
belongs to a future artifact version with its own review — not to a speedup.

---

## COMPOSE-arm `Trajectory`-construction request

**Status: `QUEUED — DISCLOSED, CORRECTION DEFERRED`**

The COMPOSE arms carry the same one-request-per-trajectory harness overhead as
`gen_rank`, at 0.02% and 0.00% of their totals. It is corrected **semantically**
by `pareto_oracle_semantics` with no re-run. Removing it from the execution path
would change committed counters and destroy the serial baseline the fan-out
parity replay must match, so it belongs with a future re-run.
