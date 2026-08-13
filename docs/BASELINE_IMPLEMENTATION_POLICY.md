# Baseline implementation policy

**Project-wide. Binds Lane 3 and every future comparator decision.**

The paper carries two categories of comparator, and conflating them is what
produces the question no answer survives:

> *Did COMPOSE beat the published method, or your approximation of it?*

---

## Category 1 — External method baselines

> **Native algorithm + our thin evaluation/accounting adapter. Nothing else.**

For any named published method, use the **authors' own implementation and their
own optimization procedure** wherever one exists. We implement only the minimum
adapter needed to put its outputs onto our common evaluation and accounting
layer.

**What the adapter may do:** hand the method molecules; route its final
molecules through our pinned evaluator; canonicalise; record the three resource
counters.

**What the adapter may NOT do:** choose actions, rewrite masks, disable
`TERMINATE`, alter the proposal distribution, change the objective, or
substitute our scalarization for theirs.

| method | rule |
|---|---|
| **GraphXForm** | its actual checkpoint and its published objective-specific fine-tuning / self-improvement / search procedure. **Do not** invent a greedy GraphXForm action loop, and **do not** mask `TERMINATE` because the native model terminates early — that ceases to be GraphXForm |
| **HN-GFN** | native multiobjective/preference machinery and **native scalarization** for the primary result. Do not rewrite it to use COMPOSE's utility for aesthetic symmetry. Its augmented-Tchebycheff mode may appear only as a clearly labelled secondary |
| **DDSBM** | the official graph-transformation implementation. Molecular input/objective adaptation may be unavoidable; the learned bridge/process stays **their** algorithm |
| **REINVENT 4, GraphGA, MARS** | native. Environment and compatibility work is fine — the MARS sklearn fix is the model case, because it **restores** originally intended oracle behaviour rather than changing the algorithm. A vocabulary rebuild can be legitimate where it is part of running the published method on the task. **Do not redesign the optimizer** |

### The stop rule — baseline engineering creep

> **If making an external method competitive starts to require inventing new
> search rules, objectives, action masks, or architectures for it, STOP. At that
> point we are no longer comparing against a published baseline.**

---

## Category 2 — Controlled internal comparators

> **We implement these ourselves, because they are defined relative to COMPOSE.**

These are **scientific controls, not literature benchmarks.** Their purpose is
mechanism isolation, and no published package implements them because they only
exist as a counterfactual to our own method.

- **The Pareto generate-and-rank comparator (P3/P4).** There is no published
  `COMPOSE-open-loop`. We are deliberately constructing *same generative
  machinery, same resources, purpose applied only after generation* against
  *purpose fed back during generation*. Implementing it ourselves is correct.
- Uniform-canonical and empirical-family Claim-1 baselines.
- The empirical-family reference-law ablation.
- Greedy versus verified COMPOSE.
- The exact finite-state solver.
- Pathwise endpoint-only versus trajectory-constrained control.
- Preference-blind floors.
- The eventual `R_θ`-shortlist ablation, if it is ever run.

**These two categories are reported as two explicitly separate groups.** That
separation strengthens the experimental section rather than weakening it: it
says precisely which numbers are "the published method" and which are "our
controlled counterfactual."

---

## The middle category — published idea, no usable implementation

Judgment is required, in this order:

1. **Use the official code with the smallest principled adapter.**
2. If the authors specify the algorithm but ship no usable implementation, make
   a faithful adaptation and label it explicitly as an **adaptation**, never as
   "the published method".
3. If reproducing it would require substantial new research or engineering
   choices, **do not pretend we have a faithful baseline.** Put it in related
   work and choose a stronger executable comparator instead.

**Option 3 is a real option.** We do not owe reviewers a homemade imitation of
every relevant paper.

- **Edit Flows** — conceptually relevant, but its published object is
  variable-length *sequences*, not molecular graphs. We do **not** build "Edit
  Flows for molecular graphs" to put a name in a table; that would be inventing
  another method. Related work.
- **pCoMole** — same principle, if adapting it from biomolecular sequence
  editing would require designing substantial graph-specific machinery. An
  important conceptual comparison, not an engineering branch.

---

## Consequence for the current GraphXForm job

The bounded job stands, with its boundary sharpened:

> Build the adapter ourselves. **Stop at the boundary where GraphXForm's own
> native training / fine-tuning procedure begins.**

Once we spend compute, it must be running **their** method, not ours wearing a
GraphXForm label. The adapter canonicalises, evaluates, counts and connects the
frozen oracle — and does not choose actions.

See `docs/RELATED_WORK_POSITIONING.md` for what each comparator is allowed to
establish, and `docs/workstreams/baseline-qualification/` for per-method status.
