SHELL := /bin/sh

MAIN := main
RENDER_DIR := ../tmp/pdfs/compose_arxiv_current
ARXIV_ARCHIVE := COMPOSE_arxiv_source.tar.gz
ARXIV_SOURCES := main.tex sections/compose.tex sections/experiments.tex sections/results.tex sections/discussion.tex sections/appendix_algorithms_proofs.tex sections/appendix_experiments.tex figures/framework.tikz compose_arxiv.sty references.bib README.md STYLE_SPEC.md SCIENTIFIC_TRACEABILITY.md COMPLETION_PLAN.md Makefile
MANUSCRIPT_TEX := main.tex sections/compose.tex sections/experiments.tex sections/results.tex sections/discussion.tex sections/appendix_algorithms_proofs.tex sections/appendix_experiments.tex

.PHONY: all pdf render check placeholders release-check qa clean arxiv-source

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

placeholders:
	@printf '%s\n' '--- Pending manuscript result fields ---'
	@rg -n '\\resultpending|\bXXX\b' $(MANUSCRIPT_TEX) || true

release-check: qa
	@if rg -n '\\resultpending|\bXXX\b' $(MANUSCRIPT_TEX); then \
	  printf '%s\n' 'Release blocked: unresolved result placeholders remain.'; \
	  exit 1; \
	else \
	  printf '%s\n' 'Release placeholder check passed.'; \
	fi

qa: render check

arxiv-source:
	tar -czf $(ARXIV_ARCHIVE) $(ARXIV_SOURCES)

clean:
	latexmk -C $(MAIN).tex
	rm -f $(ARXIV_ARCHIVE)
