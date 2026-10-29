#!/usr/bin/env bash
# autopush.sh [HOURS] (2026-10-28, author: update Overleaf faster): watch this checkout's branch and, after every commit
# that changes writeup/ (the sync's own map files aside), run `overleaf_git.sh push-all --check` (the paper and the two
# review layers, flattened from a clean checkout of HEAD) and push the map commits to GitHub.  Polls every 30 s, lets a
# burst of commits settle for 20 s, runs one push at a time (lock), and stops after HOURS (default 12).  A refused or
# failed push is logged and not retried until the next commit.  Run it in the background:
#   bash writeup/overleaf_sync/autopush.sh 12 >> writeup/overleaf_sync/autopush.log 2>&1 &
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BR="$(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
HOURS="${1:-12}"
END=$(($(date +%s) + HOURS * 3600))
LOCK="${TMPDIR:-/tmp}/harxhar_overleaf_autopush.lock"
export GIT_TERMINAL_PROMPT=0 GCM_INTERACTIVE=never
last="$(git -C "$REPO" rev-parse "$BR")"  # the current HEAD counts as pushed
echo "$(date +%H:%M:%S) autopush: watching $BR from $(git -C "$REPO" rev-parse --short "$last") until $(date -d "@$END" +%H:%M)"
while [ "$(date +%s)" -lt "$END" ]; do
    sleep 30
    cur="$(git -C "$REPO" rev-parse "$BR" 2>/dev/null)" || continue
    [ "$cur" = "$last" ] && continue
    n="$(git -C "$REPO" diff --name-only "$last" "$cur" -- writeup \
        ':(exclude)writeup/overleaf_sync/map.json' ':(exclude)writeup/overleaf_sync/map_beats.json' \
        ':(exclude)writeup/overleaf_sync/map_beats_top.json' | wc -l)"
    if [ "$n" -eq 0 ]; then
        last="$cur"
        continue
    fi
    sleep 20  # let a burst of commits settle
    cur="$(git -C "$REPO" rev-parse "$BR")"
    mkdir "$LOCK" 2>/dev/null || continue  # another push is running; try again on the next poll
    echo "$(date +%H:%M:%S) autopush: $n file(s) under writeup/ changed since $(git -C "$REPO" rev-parse --short "$last"); push-all at $(git -C "$REPO" rev-parse --short "$cur")"
    t0=$(date +%s)
    bash "$REPO/writeup/overleaf_sync/overleaf_git.sh" push-all --check 2>&1 | sed -E 's/olp_[A-Za-z0-9_]+/olp_***/g' \
        | grep -E "PASSED|FAILED|FAIL:|pushed|already|refuse|differs|own files|try [0-9]|uncommitted|errors|undefined" || true
    git -C "$REPO" push -q origin "$BR" 2>&1 | tail -1
    echo "$(date +%H:%M:%S) autopush: done in $(($(date +%s) - t0)) s"
    rmdir "$LOCK"
    last="$cur"
done
echo "$(date +%H:%M:%S) autopush: stopped ($HOURS h)"
