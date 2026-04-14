#!/bin/sh
# Launches a detached sync for a single date on the Fly machine.
# Writes stdout/stderr to /tmp/sync_<date>.log and the exit code to
# /tmp/sync_<date>.status when done, so the GHA workflow can poll
# without holding an SSH session open.
set -eu
date="$1"
log="/tmp/sync_${date}.log"
status="/tmp/sync_${date}.status"
rm -f "$log" "$status"
nohup sh -c "python3 /tmp/sync.py $date > $log 2>&1; echo \$? > $status" >/dev/null 2>&1 &
echo "launched pid=$! date=$date log=$log status=$status"
