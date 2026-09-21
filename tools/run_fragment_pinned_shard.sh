#!/bin/bash
# One (arm, task, drug, seed) shard of the attachment-control sweep, run under
# the PINNED production chemistry kernel.
#
# This differs from run_fragment_attachment_shard.sh in exactly one respect:
# the interpreter. That script used the laptop .venv (python 3.12 / rdkit
# 2026.03.6); this one uses the pinned stack (python 3.11 / rdkit 2024.3.5 /
# numpy 1.26.4 / scipy 1.13.1 / networkx 3.3 / torch 2.4.0), which is the
# kernel the production Modal image pins. A fragment search constructs exotic
# intermediates, and rdkit versions are known to disagree on those, so a
# publication number has to come from the pinned kernel rather than from a
# rescoring of emissions a different kernel produced.
#
# pandas/tqdm are installed in the pinned env because the official upstream
# evaluator imports them; the vendored .pydeps tree is deliberately NOT on the
# path, since its numpy is built for cpython-312 and would shadow the pin.
set -u
ARM="$1"; TASK="$2"; DRUG="$3"; SEED="$4"
ROOT=/Users/rmaganti/compose_frag_attach
PY=/Users/rmaganti/compose_fragment_pinned_env/bin/python
OUT="$ROOT/diagnostics/fragment_attachment_pinned_v1/shards/${ARM}/${TASK}__${DRUG}__seed${SEED}.json"
[ -s "$OUT" ] && exit 0
mkdir -p "$(dirname "$OUT")"
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT/src:$ROOT/scripts"
cd "$ROOT" || exit 1
case "$ARM" in
  baseline)   CONTROL="" ;;
  attachment) CONTROL="--attachment-control" ;;
  *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac
"$PY" tools/run_fragment_constrained_suite.py \
  --checkpoint /Users/rmaganti/compose_fragment_ckpt/ringcore_a7546e2_best.pt \
  --output "$OUT" --task "$TASK" --drug "$DRUG" --seed-list "$SEED" --samples 100 \
  $CONTROL > "${OUT%.json}.log" 2>&1
