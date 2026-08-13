#!/bin/bash
# Poll the VOLUME for a shard. Never poll the log.
#
#   A client log stops updating the moment the client dies, while the detached
#   run continues. So a log-based waiter reports a stall that is not happening,
#   or misses the finish entirely. The volume is the only honest progress signal.
#
# Usage: poll_volume.sh <volume> <remote_dir> <shard_name> [minutes_between=3]
set -euo pipefail
VOL="$1"; DIR="$2"; SHARD="$3"; EVERY="${4:-3}"
for i in $(seq 1 200); do
  OUT=$(timeout 120 modal volume ls "$VOL" "$DIR" 2>&1 || true)
  if echo "$OUT" | grep -q "$SHARD"; then echo "DONE after $((i*EVERY))min"; echo "$OUT"; exit 0; fi
  if echo "$OUT" | grep -q "partial\|resume"; then echo "[$(date +%H:%M)] alive, checkpoint present (poll $i)";
  else echo "[$(date +%H:%M)] nothing yet (poll $i)"; fi
  sleep $((EVERY*60))
done
echo "TIMED OUT"
