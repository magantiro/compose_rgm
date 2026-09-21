#!/bin/bash
# One (task, drug, seed) shard of the frozen fragment-constrained sweep.
set -u
TASK="$1"; DRUG="$2"; SEED="$3"
ROOT=/Users/rmaganti/compose_fragment_v2
OUT="$ROOT/diagnostics/fragment_official_suite_v2/shards/${TASK}__${DRUG}__seed${SEED}.json"
[ -s "$OUT" ] && exit 0
export PATH="/Users/rmaganti/compose_rgm_git/.venv/bin:$PATH"
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT/src:$ROOT/scripts:$ROOT/.pydeps"
cd "$ROOT" || exit 1
python tools/run_fragment_constrained_suite.py \
  --checkpoint /Users/rmaganti/compose_fragment_ckpt/ringcore_a7546e2_best.pt \
  --output "$OUT" --task "$TASK" --drug "$DRUG" --seed-list "$SEED" --samples 100 \
  > "${OUT%.json}.log" 2>&1
