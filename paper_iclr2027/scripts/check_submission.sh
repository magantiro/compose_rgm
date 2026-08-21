#!/bin/sh
# Fails if the manuscript still carries draft-only material.
# Run before producing a submission PDF.
cd "$(dirname "$0")/.." || exit 2
fail=0
report() { printf '  FAIL  %s\n' "$1"; fail=1; }

grep -q '\\newcommand{\\DEVELOPMENTALPATHWISE}' main.tex && \
  report "developmental n=24 pathwise values still in use (\\Pw* macros in main.tex); the frozen n=48 confirmation has not replaced them"

n=$(grep -ho '\\XXX{}' sections/*.tex | wc -l | tr -d ' ')
[ "$n" != "0" ] && report "$n unmeasured \\XXX{} placeholder(s) remain in sections/"

n=$(grep -ho '\\gap{' sections/*.tex | wc -l | tr -d ' ')
[ "$n" != "0" ] && report "$n \\gap{} draft block(s) remain in sections/"

n=$(grep -h 'color{red}' sections/*.tex 2>/dev/null | wc -l | tr -d ' ')
[ "$n" != "0" ] && report "$n red draft annotation(s) remain in sections/"

[ "$fail" = "0" ] && { echo "  OK  no draft-only material found"; exit 0; }
echo; echo "Submission build blocked."; exit 1
