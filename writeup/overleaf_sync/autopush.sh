#!/usr/bin/env bash
# autopush.sh [HOURS] (2026-10-28, author: update Overleaf faster; one edit updates the paper and both review layers):
# watch this checkout's branch and, after every commit that changes writeup/ (the sync's map files and the review layers
# themselves aside),
#   1. regenerate the review layers from a clean checkout of HEAD (draft-mode build of paper_rehaul.tex for its .aux, then
#      writeup/make_paper_beats.py) and commit them when they changed (writeup/beats, paper_rehaul_beats*.tex only);
#   2. run `overleaf_git.sh push-all --check` (the paper and the two layers, flattened from a clean checkout of HEAD);
#   3. push the commits to GitHub.
# Polls every 30 s, lets a burst of commits settle for 20 s, runs one push at a time (lock), stops after HOURS (default
# 12).  A refused or failed step is logged and not retried until the next commit.  Run it in the background:
#   bash writeup/overleaf_sync/autopush.sh 12 >> writeup/overleaf_sync/autopush.log 2>&1 &
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BR="$(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
HOURS="${1:-12}"
END=$(($(date +%s) + HOURS * 3600))
LOCK="${TMPDIR:-/tmp}/harxhar_overleaf_autopush.lock"
LAYERS=(writeup/beats writeup/paper_rehaul_beats.tex writeup/paper_rehaul_beats_top.tex)
export GIT_TERMINAL_PROMPT=0 GCM_INTERACTIVE=never

regen_beats() {  # the review layers regenerated from HEAD, committed when they changed
    local tmp wt py f sha
    tmp="$(mktemp -d)"
    wt="$tmp/wt"
    sha="$(git -C "$REPO" rev-parse --short HEAD)"
    git -C "$REPO" worktree add -q --no-checkout --detach "$wt" HEAD || return 1
    git -C "$wt" sparse-checkout set --cone writeup && git -C "$wt" checkout -q --detach HEAD
    (
        cd "$wt/writeup" || exit 1
        pdflatex -interaction=nonstopmode -draftmode paper_rehaul.tex >/dev/null 2>&1
        bibtex paper_rehaul >/dev/null 2>&1
        pdflatex -interaction=nonstopmode -draftmode paper_rehaul.tex >/dev/null 2>&1
        pdflatex -interaction=nonstopmode -draftmode paper_rehaul.tex >/dev/null 2>&1
    )
    py=python3
    command -v python3 >/dev/null 2>&1 || py=python
    if ! "$py" "$wt/writeup/make_paper_beats.py" --src "$wt/writeup" --aux "$wt/writeup/paper_rehaul.aux" >"$tmp/beats.log" 2>&1; then
        echo "autopush: make_paper_beats.py failed at $sha:"
        tail -5 "$tmp/beats.log"
        git -C "$REPO" worktree remove --force "$wt" >/dev/null 2>&1
        return 1
    fi
    # mirror the regenerated layers into this checkout (stale stubs removed with git rm)
    while IFS= read -r f; do
        [ -e "$wt/$f" ] || git -C "$REPO" rm -q -- "$f"
    done < <(git -C "$REPO" ls-files -- writeup/beats)
    cp -r "$wt/writeup/beats/." "$REPO/writeup/beats/"
    cp "$wt/writeup/paper_rehaul_beats.tex" "$wt/writeup/paper_rehaul_beats_top.tex" "$REPO/writeup/"
    git -C "$REPO" worktree remove --force "$wt" >/dev/null 2>&1
    git -C "$REPO" add -A -- "${LAYERS[@]}"
    if git -C "$REPO" diff --cached --quiet -- "${LAYERS[@]}"; then
        echo "autopush: review layers already match $sha"
    else
        git -C "$REPO" commit -q -m "Review layers regenerated from $sha (autopush)" -- "${LAYERS[@]}"
        echo "autopush: review layers regenerated from $sha and committed"
    fi
}

last="$(git -C "$REPO" rev-parse "$BR")"  # the current HEAD counts as pushed
echo "$(date +%H:%M:%S) autopush: watching $BR from $(git -C "$REPO" rev-parse --short "$last") until $(date -d "@$END" +%H:%M)"
while [ "$(date +%s)" -lt "$END" ]; do
    sleep 30
    cur="$(git -C "$REPO" rev-parse "$BR" 2>/dev/null)" || continue
    [ "$cur" = "$last" ] && continue
    n="$(git -C "$REPO" diff --name-only "$last" "$cur" -- writeup \
        ':(exclude)writeup/overleaf_sync/map.json' ':(exclude)writeup/overleaf_sync/map_beats.json' \
        ':(exclude)writeup/overleaf_sync/map_beats_top.json' ':(exclude)writeup/beats' \
        ':(exclude)writeup/paper_rehaul_beats.tex' ':(exclude)writeup/paper_rehaul_beats_top.tex' | wc -l)"
    if [ "$n" -eq 0 ]; then
        last="$cur"
        continue
    fi
    sleep 20  # let a burst of commits settle
    mkdir "$LOCK" 2>/dev/null || continue  # another push is running; try again on the next poll
    cur="$(git -C "$REPO" rev-parse "$BR")"
    echo "$(date +%H:%M:%S) autopush: $n file(s) under writeup/ changed since $(git -C "$REPO" rev-parse --short "$last"); at $(git -C "$REPO" rev-parse --short "$cur")"
    t0=$(date +%s)
    regen_beats
    bash "$REPO/writeup/overleaf_sync/overleaf_git.sh" push-all --check 2>&1 | sed -E 's/olp_[A-Za-z0-9_]+/olp_***/g' \
        | grep -E "PASSED|FAILED|FAIL:|pushed|already|refuse|differs|own files|try [0-9]|uncommitted|errors|undefined" || true
    git -C "$REPO" push -q origin "$BR" 2>&1 | tail -1
    echo "$(date +%H:%M:%S) autopush: done in $(($(date +%s) - t0)) s"
    rmdir "$LOCK"
    last="$(git -C "$REPO" rev-parse "$BR")"  # includes the layer and map commits just made
done
echo "$(date +%H:%M:%S) autopush: stopped ($HOURS h)"
