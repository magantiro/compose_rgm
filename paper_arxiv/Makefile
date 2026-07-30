SHELL := /bin/sh

MAIN := main
RENDER_DIR := ../tmp/pdfs/compose_arxiv_current
ARXIV_ARCHIVE := COMPOSE_arxiv_source.tar.gz
ARXIV_SOURCES := main.tex compose_arxiv.sty references.bib README.md STYLE_SPEC.md SCIENTIFIC_TRACEABILITY.md Makefile

.PHONY: all pdf render check qa clean arxiv-source

all: pdf

pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error $(MAIN).tex

render: pdf
	mkdir -p $(RENDER_DIR)
	rm -f $(RENDER_DIR)/page-*.png
	pdftoppm -png -r 144 $(MAIN).pdf $(RENDER_DIR)/page

check: pdf
	pdfinfo $(MAIN).pdf
	pdffonts $(MAIN).pdf
	@printf '%s\n' '--- LaTeX warnings and box diagnostics ---'
	@if grep -E 'LaTeX (Font )?Warning|Package .* Warning|Overfull|Underfull|undefined' $(MAIN).log; then \
	  printf '%s\n' 'QA failed: resolve the diagnostics above.'; \
	  exit 1; \
	else \
	  printf '%s\n' 'none'; \
	fi

qa: render check

arxiv-source:
	tar -czf $(ARXIV_ARCHIVE) $(ARXIV_SOURCES)

clean:
	latexmk -C $(MAIN).tex
	rm -f $(ARXIV_ARCHIVE)
