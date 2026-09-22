#!/bin/sh
# Run one arm of the entry diagnostic over the five gate tasks, at most three
# local processes at once.  Oversubscription on this machine has taken a build
# from 5,507 entries/h to 250, and serial is frequently faster than parallel.
#
#   scripts/pmo_entry_diagnostic_run.sh <arm> <draws> <outdir> <logdir>
set -eu
ARM="$1"; DRAWS="$2"; OUTDIR="$3"; LOGDIR="$4"
mkdir -p "$OUTDIR" "$LOGDIR"
PY="${PMO_ENTRY_PYTHON:-$HOME/compose_pmo_atlas_env/bin/python}"
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src:scripts
i=0
for TASK in celecoxib_rediscovery albuterol_similarity jnk3 perindopril_mpo qed; do
  OUT="$OUTDIR/${ARM}__${TASK}.json"
  if [ -f "$OUT" ]; then echo "skip $TASK (shard present)"; continue; fi
  "$PY" scripts/pmo_entry_diagnostic.py measure \
      --arm "$ARM" --task "$TASK" --draws "$DRAWS" \
      --output "$OUT" > "$LOGDIR/${ARM}__${TASK}.log" 2>&1 &
  i=$((i+1))
  if [ "$i" -ge 3 ]; then wait; i=0; fi
done
wait
echo "done $ARM"
