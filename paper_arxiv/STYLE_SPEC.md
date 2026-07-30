# COMPOSE arXiv style specification

## Reference and scope

This package reproduces the visual system of the locally supplied reference
`homological_flows.pdf`; it does not reuse that paper's prose or scientific
content.

- Reference SHA-256:
  `86d181a8abe43c119e1d84b8ff251df8badce74b9a0df94c2e6a5556c1ceebae`
- Reference geometry: U.S. letter, single column, 612 x 792 pt.
- Reference typography: embedded Latin Modern Roman, Sans, and Math fonts.
- Current manuscript scope: a complete paper-shaped draft with explicit
  result and figure placeholders where frozen empirical evidence is pending.

## Page system

- Paper: U.S. letter.
- Main type: Latin Modern Roman at 10 pt.
- Text margins: 0.85 in left/right, 0.82 in top, 0.68 in bottom.
- Footer: centered page number, no header or rules.
- Paragraphs: no first-line indent; compact vertical separation.
- Running text is fully justified with `microtype` enabled.

## Hierarchy

- Title panel: full text width, pale gray (`#F1F3F5`), 12 pt corner radius,
  no border.
- Title: left aligned, Latin Modern Sans Bold, 19 pt.
- Author line: left aligned, Latin Modern Sans Bold, 10.5 pt.
- Abstract: left aligned, Latin Modern Sans, approximately 9.2 pt.
- Section headings: Latin Modern Sans Bold, 17 pt.
- Run-in paragraph headings: Latin Modern Sans Bold, 10 pt.

## Links and citations

- Citations use author-year form through `natbib`.
- Citation, cross-reference, and URL links use restrained blue
  (`#2468C4`).
- PDF metadata is set explicitly in `main.tex`.

## arXiv compatibility

- Compiler: `pdflatex` plus `bibtex`.
- No shell escape, system fonts, remote assets, or non-TeX runtime dependency.
- Every required source file is local to `paper_arxiv/`.
- `make arxiv-source` produces a source archive containing only the TeX source,
  style file, bibliography, and build notes.
- The build must report only embedded Latin Modern/AMS fonts and no overfull
  boxes before a release artifact is accepted.

## Deliberate deviations from the reference

- The title-panel abstract is Sans because the design brief explicitly
  requests a Sans title, author line, and abstract.
- COMPOSE figures and tables use restrained gray panels, booktabs rules, and
  compact red `XXX` tokens for pending evidence. The source retains stable
  placeholder keys even though the PDF renders only the compact token.
