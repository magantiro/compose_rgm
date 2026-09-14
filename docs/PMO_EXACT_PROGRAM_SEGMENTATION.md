# PMO split-first exact-program segmentation corpus

## Problem and authorized output

The preceding sanitized export supports local action ranking but discarded the
exact graph/action pairs required to bind structural roles and execute a complete
program. This revision creates one training-only exact corpus and tests whether a
generic structural segmentation supplies a legitimate complete-candidate
representation. It does not train or select a controller.

Only the 181 complete programs and five Perindopril witnesses in the self-hashed
PMO route-distillation inclusion manifest are admitted. Source hashes, envelope
hashes, exact replay and duplicate reconciliation remain unchanged. Task scores
are neither loaded nor treated as pathwise labels.

## Split before derivation

The task-family folds are frozen before any segment or panel is constructed:

1. bioactivity and formula;
2. rediscovery and multi-property; and
3. similarity, druglikeness and multiobjective optimization.

Every shared-base lineage and exact trace belongs to exactly one test fold. The
training-only corpus records this assignment alongside exact persistent-slot
source/intermediate states, executable primitive records, endpoints and route
membership. Those fields are legitimate supervision after splitting, but no
runtime checkpoint may contain them.

## Generic segmentation algorithm

For each exact trace, the segmenter constructs primitive footprints and a directed
created-handle dependency graph. A created handle records its producer, relative
creation ordinal and every later consumer. Producer-to-last-consumer intervals are
protected. A cycle-open followed by a structurally connected cycle-close or
ring-restatement also creates a protected interval.

Adjacent operations are placed in the same segment when their changed footprints
overlap or touch through the exact boundary graph. Protected cuts take precedence.
Segments contain at most eight primitives. A protected interval longer than eight
uses explicit one-primitive fallback rather than receiving a semantic macro name.
The only multi-action label is `connected_change_dependency_segment`; cycle and
dependency properties are recorded as descriptors, not interpreted mechanisms.

Every segment must reproduce every saved intermediate state through the production
semantic executor. Complete representation support additionally requires at most
32 primitives and eight resulting segments. Coverage, primitive fallback,
cross-segment dependencies, exact replay precision and module-count distributions
are reported per held-out fold and overall.

## Same-source generic negative panels

For each frozen base lineage, generate exactly 32 task-blind proposals from its
exact source using the unchanged generic Dynamic-v1 compiler, at most three generic
modules, 32 primitives and eight blocks. Seed derivation uses only the sealed
campaign seed and lineage hash. Every attempt, rejection, exact trace, endpoint,
duplicate and compiler metadata remains training-only.

The panel labels exact teacher collision, radius-two transformation equivalence or
generic negative. No task score enters the label. Fold-wise evaluation reports
execution precision, unique endpoint yield, exact and transformation-equivalent
recall and precision, shortfall, proposal time and failures. A teacher endpoint is
used only for offline labeling and is never inserted into the generated panel.

## Gate for the next policy revision

A complete-policy revision is legitimate only if held-out segmentation has exact
replay precision 1.0, nonzero multi-action coverage, zero split leakage and
nonzero exact-executed generic negative yield, with complete representation support
in every held-out fold. The current artifact may establish representation and
panel availability, but it cannot establish that a learned policy improves
complete-route recall. The next gate should compare a marginal
segment decoder, a graph/region/dependency-conditioned decoder and a contrastive
complete-candidate ranker on the frozen same-source panels. Checkpoints must remain
free of task names, lineages, source graphs, endpoints, absolute addresses,
assignments and executable teacher routes.

## Run

```bash
PYTHONPATH=src:. .venv/bin/python tools/pmo_exact_program_segmentation.py \
  --output diagnostics/pmo_exact_program_segmentation/attempt_2
```

This is a deterministic single-CPU, zero-oracle command. It refuses overwrite and
publishes a self-hashed training corpus plus a compact audit.
