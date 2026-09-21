#!/bin/bash
# Fan the pinned attachment-control sweep out over WORKERS local processes.
#
# Ordering is by DELIVERABLE, not by arm. A before/after row needs both arms of
# the SAME task, so finishing one arm first leaves every row incomplete until
# the very end; finishing one task's missing arm first lands that row early.
# Shards are idempotent -- the runner exits on a non-empty output -- so a
# relaunch redoes nothing and a killed sweep resumes where it stopped.
set -u
ROOT=/Users/rmaganti/compose_frag_attach
WORKERS="${WORKERS:-6}"
cd "$ROOT" || exit 1
DRUGS="BARICITINIB CYCLOTHIAZIDE ELIGLUSTAT ERLOTINIB FUTIBATINIB LESINURAD LIOTHYRONINE LOVASTATIN MARIBAVIR SPIRAPRIL"
emit() { for DRUG in $DRUGS; do for SEED in 0 1 2; do echo "$1 $2 $DRUG $SEED"; done; done; }
{
  emit attachment motif_extension
  emit attachment superstructure_generation
  emit baseline   scaffold_decoration
  emit attachment scaffold_decoration
  emit baseline   motif_extension
  emit baseline   superstructure_generation
} | xargs -P "$WORKERS" -n 4 ./tools/run_fragment_pinned_shard.sh
echo "SWEEP_COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
