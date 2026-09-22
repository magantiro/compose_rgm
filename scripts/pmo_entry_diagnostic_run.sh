#!/bin/sh
# Run one arm of the entry diagnostic over one or more tasks, at most three
# local processes at once.  Oversubscription on this machine has taken a build
# from 5,507 entries/h to 250, and serial is frequently faster than parallel.
#
#   scripts/pmo_entry_diagnostic_run.sh <arm> <draws> <outdir> <logdir> \
#       [parent_source] [task ...]
#
# ADDING AN ARM IS A REGISTRATION, not a change to this script or the harness:
#
#   * a proposal MECHANISM  -> register_arm(name, fn, describe=...)
#   * an INITIALIZATION     -> register_parent_source(name, fn, describe=...)
#
# both in src/compose_v4/experiments/pmo_entry_diagnostic.py.  The shard name
# carries the population, so one mechanism measured over two populations lands
# as two ladder rows and is never pooled.
#
# The default population is the deployed run's own visited molecules, which is
# what every shard written before the population became an explicit axis used;
# a run over it reproduces those seeds exactly.
set -eu
ARM="$1"; DRAWS="$2"; OUTDIR="$3"; LOGDIR="$4"
PARENT_SOURCE="${5:-blind_visited_stratified}"
shift 5 2>/dev/null || shift 4
if [ "$#" -gt 0 ]; then TASKS="$*"; else
  TASKS="celecoxib_rediscovery albuterol_similarity jnk3 perindopril_mpo qed"
fi
mkdir -p "$OUTDIR" "$LOGDIR"
PY="${PMO_ENTRY_PYTHON:-$HOME/compose_pmo_atlas_env/bin/python}"
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src:scripts
if [ "$PARENT_SOURCE" = "blind_visited_stratified" ]; then SUFFIX=""; else
  SUFFIX="__${PARENT_SOURCE}"; fi
i=0
for TASK in $TASKS; do
  OUT="$OUTDIR/${ARM}${SUFFIX}__${TASK}.json"
  if [ -f "$OUT" ]; then echo "skip $TASK (shard present)"; continue; fi
  "$PY" scripts/pmo_entry_diagnostic.py measure \
      --arm "$ARM" --parent-source "$PARENT_SOURCE" --task "$TASK" --draws "$DRAWS" \
      --output "$OUT" > "$LOGDIR/${ARM}${SUFFIX}__${TASK}.log" 2>&1 &
  i=$((i+1))
  if [ "$i" -ge 3 ]; then wait; i=0; fi
done
wait
echo "done $ARM over $PARENT_SOURCE"
