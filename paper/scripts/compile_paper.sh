#!/usr/bin/env bash
# Compile the manuscript in draft or final mode.
#
#   compile_paper.sh draft   -> placeholders visible; always produces manuscript.pdf
#   compile_paper.sh final   -> runs the gates; FAILS (nonzero) while any placeholder,
#                               METHODCHECK, unverified citation, stale-result ref, or
#                               page-limit overflow remains. Only writes the final PDF
#                               when every gate is green.
#
# TeX toolchain: latexmk + pdflatex (put /Library/TeX/texbin on PATH).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER="$(dirname "$HERE")"
cd "$PAPER" || exit 2
export PATH="/Library/TeX/texbin:$PATH"
MODE="${1:-draft}"
PY="${PYTHON:-python3}"

build() {  # $1 = mode -- regenerate results.tex, guarded by the max_atoms=40 scope contract
  "$PY" scripts/check_scope_consistency.py || {
    echo "compile_paper: scope-consistency gate FAILED -- registry drifts from the max_atoms=40 contract;"
    echo "               refusing to generate results.tex. Fix config_registry.yaml or the manifest."
    return 9
  }
  "$PY" scripts/build_results_tex.py --mode "$1"
}
compile() {
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
}

if [ "$MODE" = "draft" ]; then
  build draft || exit 3
  compile || { echo "compile_paper: latexmk failed"; exit 4; }
  cp -f main.pdf manuscript.pdf
  echo "compile_paper: DRAFT ready -> manuscript.pdf (placeholders visible)"
  exit 0
fi

if [ "$MODE" = "final" ]; then
  # 1) refresh compilable artifacts (aux/log) in draft mode so the gates have data.
  build draft || exit 3
  compile || { echo "compile_paper: draft compile failed; fix TeX before final"; exit 4; }
  # 2) run the gates.
  RC=0
  "$PY" scripts/check_claim_registry.py || RC=1
  "$PY" scripts/check_placeholders.py   || RC=1
  if [ "$RC" -ne 0 ]; then
    echo ""
    echo "compile_paper: FINAL BLOCKED -- gates failed (see above). No final PDF written."
    exit 1
  fi
  # 3) all green -> build + compile in final mode, publish manuscript.pdf.
  build final || { echo "compile_paper: unexpected pending results in final build"; exit 5; }
  compile || { echo "compile_paper: final compile failed"; exit 4; }
  cp -f main.pdf manuscript.pdf
  echo "compile_paper: FINAL ready -> manuscript.pdf"
  exit 0
fi

echo "usage: compile_paper.sh {draft|final}"
exit 2
