#!/usr/bin/env bash
# Run one command under the Thor's shared GPU exclusive lock, as agreed with the co-location measurement:
# the lock file is opened read-only and held with flock(LOCK_EX) for the command's lifetime, released when it exits.
# The lock file belongs to the co-location measurement's guard; this script never creates it. When it is absent the
# command runs without the lock (that guard also refuses to start while another user's GPU process exists).
# Prints wait / acquire / release times (ISO 8601) so a report can state when the GPU was held.
#
# usage: bash with_gpu_lock.sh CMD [ARGS...]
set -u
LOCK=/tmp/lingbot_va_gpu_exclusive.lock
if [ ! -e "$LOCK" ]; then
  echo "gpu_lock: $LOCK absent, running without it $(date -Is)"
  "$@"
  exit $?
fi
exec 9<"$LOCK"
echo "gpu_lock: waiting $(date -Is)"
flock -x 9
echo "gpu_lock: acquired $(date -Is)"
"$@"
rc=$?
echo "gpu_lock: released $(date -Is) rc=$rc"
exit $rc
