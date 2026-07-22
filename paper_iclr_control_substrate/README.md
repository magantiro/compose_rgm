# COMPOSE: Generator-Matched Stochastic Rewriting - ICLR 2026 draft

This directory is a new manuscript built from the user's prior official ICLR
Overleaf package. The template and submission style are preserved, while the
scientific story is rewritten around Rewrite Generator Matching: stochastic
molecular graph rewriting supplies the legal CTMC support, and Generator
Matching learns its context- and time-dependent rates.

## Files

- `main.tex`: complete main paper and appendix.
- `numbers.tex`: centralized `XXX` result placeholders.
- `references.bib`: bibliography, including verified stochastic rewriting and
  molecular graph grammar references.
- `main.pdf`: locally compiled anonymous draft.
- `iclr2026_conference.sty` and `.bst`: official supplied submission style.

## Result boundary

Every unmeasured final result is deliberately `XXX`. The one developmental
training paragraph is labeled as such and must not be promoted into the main
benchmark table. Replace placeholders only from frozen experiment artifacts.

## Build

```bash
bash build.sh
```

Leave `\iclrfinalcopy` commented for double-blind review.
