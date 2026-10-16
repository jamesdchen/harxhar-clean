#!/bin/bash
# Re-ship a few files to a pooled-momvix deployment root in ONE ssh session (tar in, md5sum out) and
# FAIL unless every remote md5 equals the staged one (2026-10-06; the Hoffman2 login node throttles
# repeated connections).  Tracked files with no change against HEAD are taken from HEAD (LF); the
# rest as they are on disk.  Run from Git Bash locally:
#   STAGE=<empty scratch dir> bash cluster/pool_momvix_update.sh <hoffman2|usc-discovery> <remote root> <file>...
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
HOST=$1 REMOTE=$2
shift 2
STAGE=${STAGE:?set STAGE to an empty local scratch directory}
mkdir -p "$STAGE"
[ -z "$(ls -A "$STAGE")" ] || { echo "STAGE $STAGE is not empty"; exit 1; }
for f in "$@"; do
  mkdir -p "$STAGE/$(dirname "$f")"
  if git ls-files --error-unmatch "$f" >/dev/null 2>&1 && git diff --quiet HEAD -- "$f"; then
    git show "HEAD:$f" > "$STAGE/$f"
  else
    cp "$f" "$STAGE/$f"
  fi
done
CR=$(cd "$STAGE" && cat "$@" | tr -cd '\r' | wc -c)
[ "$CR" -eq 0 ] || { echo "staged files carry $CR CR bytes"; exit 1; }
norm() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }  # md5sum text vs binary marker
LOCAL=$(cd "$STAGE" && md5sum "$@" | norm)
REMOTE_SUMS=$( (cd "$STAGE" && tar czf - "$@") | "$SSH" -o BatchMode=yes "$HOST" "cd $REMOTE && tar xzf - && md5sum $*" | norm)
if [ "$LOCAL" != "$REMOTE_SUMS" ]; then
  echo "UPDATE FAILED: remote md5 differs"; diff <(echo "$LOCAL") <(echo "$REMOTE_SUMS") || true; exit 1
fi
echo "md5 OK: $# files -> $HOST:$REMOTE"
