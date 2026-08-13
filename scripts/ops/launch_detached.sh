#!/bin/bash
# Launch a long Modal run so it survives the client dying. Paid for four times.
#
#   --detach alone is NOT enough. It keeps only the LAST TRIGGERED function alive
#   past a client kill, and it did not survive a client-side DNS failure here
#   (three times), a harness reaping the client, or a `timeout` wrapper -- which
#   cancels the job outright. Never wrap a detached launch in a client timeout.
#
#   The app's entrypoint should also use drive.spawn() rather than .remote(), so
#   the client need not stay alive at all.
#
# Usage: launch_detached.sh <worktree> <app.py> <logfile> [args...]
set -euo pipefail
WT="$1"; APP="$2"; LOG="$3"; shift 3
cd "$WT"
nohup modal run --detach "$APP" "$@" > "$LOG" 2>&1 &
disown
echo "launched detached; watch the VOLUME, not $LOG"
