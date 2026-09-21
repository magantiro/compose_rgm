#!/bin/bash
# Fan the pinned attachment-control sweep out over WORKERS local processes.
# Shards are idempotent (the runner exits early on a non-empty output), so a
# relaunch redoes nothing and a killed sweep resumes.
set -u
ROOT=/Users/rmaganti/compose_frag_attach
WORKERS="${WORKERS:-6}"
cd "$ROOT" || exit 1
DRUGS="BARICITINIB CYCLOTHIAZIDE ELIGLUSTAT ERLOTINIB FUTIBATINIB LESINURAD LIOTHYRONINE LOVASTATIN MARIBAVIR SPIRAPRIL"
TASKS="motif_extension superstructure_generation scaffold_decoration"
ARMS="baseline attachment"
for ARM in $ARMS; do for TASK in $TASKS; do for DRUG in $DRUGS; do for SEED in 0 1 2; do
  echo "$ARM $TASK $DRUG $SEED"
done; done; done; done | xargs -P "$WORKERS" -n 4 ./tools/run_fragment_pinned_shard.sh
echo "SWEEP_COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
