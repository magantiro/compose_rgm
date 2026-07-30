# COMPOSE arXiv manuscript

This is the clean arXiv-format manuscript package. It is now a complete
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
diagnostic. A successor-aware repaired trainer is a new protocol, not a
retroactive reinterpretation of that run. Fixed-budget editing and timed de
novo generation require separate checkpoints and inference contracts.
