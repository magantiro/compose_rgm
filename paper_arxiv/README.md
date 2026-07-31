# COMPOSE arXiv manuscript

This is the preferred live long-form manuscript package. It uses the
Homological-Flows-derived arXiv visual system documented in `STYLE_SPEC.md`;
the conference-formatted manuscript is a companion representation, not a
separate scientific authority. The package is a complete
paper-shaped draft: front matter, framework, Rewrite Generator Matching,
canonical molecular pushforward, finite-horizon control, the COMPOSE molecular
implementation, experimental design, results layout, limitations, and
conclusion are all present. Empirical result tables and figure slots are
included, but unresolved measurements and conclusions remain explicit
placeholders rather than unverified claims.

`COMPLETION_PLAN.md` defines the full-paper section map and the result-slot
policy. During drafting, each unresolved empirical field retains a stable
`\resultpending{KEY}` in source and renders visibly as `XXX` in the PDF.
`make placeholders` prints the keyed inventory across the complete manuscript;
`make release-check` fails until every placeholder has been replaced from a
frozen result artifact.

`SCIENTIFIC_TRACEABILITY.md` records the authoritative source and claim boundary
for each load-bearing sentence introduced so far.

## Authority and synchronization

Scientific content must remain synchronized against:

- `docs/PAPER1_FRAMING_AUTHORITATIVE.md` for thesis, title, contribution order,
  and novelty boundaries;
- `docs/HANDOFF_COMPOSE_TRACEABILITY_2026-07-29.md` and the authoritative pasted
  handoff for current scientific precedence;
- `AGENTS.md` and current self-hashed contracts for implementation and training
  authorization;
- `configs/editing_v2_candidate_routing_policy_v1.json` and
  `configs/editing_v2_split_assignment_policy_v1.json` for the current
  deterministic five-lane routing and prospective four-role assignment;
- `docs/CLAIM_LEDGER.md` plus provenance-complete artifacts for evidence status;
- `configs/comparator_registry_v1.json` for required comparison arms; and
- `configs/experiment_registry.yaml` for tasks, endpoints, budgets, and outputs.

When either manuscript disagrees with a current contract or the evidence
ledger, the prose is stale. Reconcile the discrepancy explicitly rather than
choosing the convenient version. `COMPLETION_PLAN.md` lists the comparator rows
and result-key families that must remain synchronized across formats.

## Build

```bash
cd paper_arxiv
make pdf
```

The build uses `pdflatex` and `bibtex` through `latexmk`.

## Render and inspect

```bash
make render
make check
```

Rendered pages are written to `../tmp/pdfs/compose_arxiv_current/`. The `check`
target prints PDF metadata, embedded fonts, unresolved-reference warnings, and
any overfull or underfull boxes. Visual inspection of every rendered page is
still required.

## Prepare arXiv source

```bash
make arxiv-source
```

This creates `COMPOSE_arxiv_source.tar.gz` from the self-contained manuscript
sources. The author line is intentionally anonymous while the author list is
not frozen.

## Scientific boundary

The title, abstract, and Introduction follow the current authoritative thesis:

```text
executable state-dependent rewrites
  -> Generator Matching
  -> trans-dimensional molecular generation
  -> canonical molecular-successor pushforward
  -> exact and dynamic stochastic control
```

The completed 16,000-step editing run is described only as a mark-level
diagnostic: it used 661,105 admitted training traces and produced all 32
scheduled snapshots, but ring opening and Graft remain unresolved and no
checkpoint has been selected. The already inspected final-test aggregates are
barred from model, threshold, or recipe selection.

Editing-V2 declares five evidence lanes:
`observed_local_analogue`, `operator_aware_real_endpoint`,
`linker_positional_topology_analogue`, `real_endpoint_multistep_path`, and
`reversible_synthetic_walk`. Genuine observed-series action provenance is not
available, so compiler-generated multistep paths are not described as observed
series. A deterministic policy now routes candidate traces into those lanes,
and a separate policy targets an 85/5/5/5 assignment of indivisible components
by declared census mass to training, validation, controller-validation, and
sealed final-test roles.
Those policies are prospective and non-authorizing: no admitted V2 corpus,
physical lane shards, or training permission is implied. The corpus remains
`DESIGN_NOT_TRAINING_AUTHORIZED`; whole-trace admission, Gate 0, T1, P50, P500,
and P2000 must pass in order before a long run. A successor-aware editing
trainer is a new protocol, not a retroactive reinterpretation of the completed
run. Fixed-budget editing and timed de novo generation require separate
checkpoints and inference contracts, and no production de-novo checkpoint
currently exists.
