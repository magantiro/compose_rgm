# Why COMPOSE v4 is not a fragment-assembly model

## The potential confusion

The first teacher compiler begins at the formal null state, inserts atoms along
a spanning tree, and closes non-tree bonds. Viewed in isolation, that trace can
look like atom-by-atom or fragment growing. It is crucial not to identify the
teacher trace with the generative model.

## The actual model object

The learned object is a time-dependent generator on complete molecular graphs:

```text
q_theta,t(a | G, c),  a in A_c(G).
```

`A_c(G)` is the full legal action fiber at the current molecule. The minimal
basis includes atom insert/delete/restate and bond insert/delete/reorder at any
legal operands. A fragment vocabulary is neither the state space nor the model
output space.

At training time, a compiler samples privileged endpoint-conditioned paths and
a progress CTMC supplies conditional rate targets. The network sees only the
current whole graph, time, and public condition. At inference it receives no
target, alignment, construction trace, or next-fragment label; it ancestrally
samples from its marginal learned rates.

## What would collapse the method into assembly

COMPOSE would effectively become an assembly model if it:

- replayed a target-specific construction order at inference;
- restricted actions to attaching members of a fixed motif vocabulary;
- made an unfinished fragment graph the visible state;
- predicted only the next atom or fragment in one canonical order.

The implementation must therefore randomize valid traces, train complete-
successor rather than construction-index rates, and exercise bidirectional edit
families where the task requires them. The present deterministic compiler is a
correctness scaffold, not the final path distribution.

## Role of future macro operators

A verified ring, path, or motif rewrite may later shorten event horizons. Such
a macro is an optional transition that lowers to the same validity-preserving
micro semantics. It does not turn fragments into the ontology of the model.
