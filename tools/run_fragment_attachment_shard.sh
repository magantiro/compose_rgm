#!/bin/bash
# One (arm, task, drug, seed) shard of the frozen attachment-control sweep.
#
# ARM is the only thing that differs between the two sets: "baseline" runs the
# frozen sampler exactly as the v2 sweep did, "attachment" adds the declared-
# interface controller. Sampler configuration, checkpoint, seeds and prompt set
# are identical, so the comparison is controlled.
set -u
ARM="$1"; TASK="$2"; DRUG="$3"; SEED="$4"
ROOT=/Users/rmaganti/compose_frag_attach
OUT="$ROOT/diagnostics/fragment_attachment_control_v1/shards/${ARM}/${TASK}__${DRUG}__seed${SEED}.json"
[ -s "$OUT" ] && exit 0
mkdir -p "$(dirname "$OUT")"
export PATH="/Users/rmaganti/compose_rgm_git/.venv/bin:$PATH"
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT/src:$ROOT/scripts:$ROOT/.pydeps"
cd "$ROOT" || exit 1
case "$ARM" in
  baseline)   CONTROL="" ;;
  attachment) CONTROL="--attachment-control" ;;
  *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac
python tools/run_fragment_constrained_suite.py \
  --checkpoint /Users/rmaganti/compose_fragment_ckpt/ringcore_a7546e2_best.pt \
  --output "$OUT" --task "$TASK" --drug "$DRUG" --seed-list "$SEED" --samples 100 \
  $CONTROL > "${OUT%.json}.log" 2>&1
