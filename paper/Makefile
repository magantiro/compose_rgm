# Manuscript build. TeX toolchain: latexmk + pdflatex on /Library/TeX/texbin.
#
#   make paper-draft   placeholders visible; always produces manuscript.pdf
#   make paper-final   runs the gates; FAILS while any placeholder / METHODCHECK /
#                      unverified citation / stale-result ref / page overflow remains
#   make checks        run the registry + placeholder gates without compiling to final
#   make clean         remove latexmk build artifacts
.PHONY: paper-draft paper-final checks clean

paper-draft:
	bash scripts/compile_paper.sh draft

paper-final:
	bash scripts/compile_paper.sh final

checks:
	python3 scripts/check_scope_consistency.py
	python3 scripts/check_claim_registry.py
	python3 scripts/check_placeholders.py

clean:
	latexmk -C main.tex || true
	rm -f manuscript.pdf results.tex
