#!/usr/bin/env bash
# autopush.sh -- bring Overleaf up to date after a commit that changes the paper (2026-10-28, author: update Overleaf
# faster; one edit updates the paper and both review layers). 2026-10-29 (author: "Shouldn't the Overleaf update be
# automatic/a part of the pipeline"): started by the git post-commit hook (hooks/post-commit, installed by
# install_hook.sh), no longer a polling loop in a Claude Code background task. How to install, disable and read the
# log: README.md, "Automatic push after every commit".
#
#   bash writeup/overleaf_sync/autopush.sh trigger SHA  # what the hook starts, detached: if commit SHA changed writeup/,
#                                                       # let the commits settle (20 s), run one pass, and one more if
#                                                       # commits arrived during it; one runner at a time (lock); its
#                                                       # output, token masked, is appended to autopush.log
#   bash writeup/overleaf_sync/autopush.sh once         # one pass now, in the foreground
#
# A pass:
#   1. regenerates the review layers from a clean checkout of HEAD (draft-mode build of paper_rehaul.tex for its .aux,
#      then writeup/make_paper_beats.py) and commits them when they changed (writeup/beats, paper_rehaul_beats*.tex);
#   2. runs `overleaf_git.sh push-all --check` (the paper and the two layers, flattened from a clean checkout of HEAD);
#      Overleaf edits not pulled back: "REFUSED" and the `pull --dry-run` summary are logged and nothing is forced;
#   3. pushes this branch's commits (the layer and map commits included) to GitHub.
# A refused or failed step is logged and not retried until the next commit. The pass's own commits touch only the
# layers and the maps, which the hook ignores, and run with OVERLEAF_AUTOPUSH_RUNNING=1, which the hook also skips.
set -u
# started from a hook, the runner inherits git's GIT_INDEX_FILE (relative), GIT_DIR, ...: with them, git in the
# temporary worktrees below would use this checkout's index
unset $(git rev-parse --local-env-vars)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SYNC="$REPO/writeup/overleaf_sync"
OG="$SYNC/overleaf_git.sh"
LOG="$SYNC/autopush.log"
STATE="$(git -C "$REPO" rev-parse --path-format=absolute --git-path overleaf-autopush)"  # per worktree
LOCK="$STATE/lock"
PENDING="$STATE/pending"   # the time of the last commit that asked for a push
SETTLE="${OVERLEAF_AUTOPUSH_SETTLE:-20}"
MASK='s/olp_[A-Za-z0-9_]+/olp_***/g'
LAYERS=(writeup/beats writeup/paper_rehaul_beats.tex writeup/paper_rehaul_beats_top.tex)
ODIR="${OVERLEAF_DIR:-$REPO/../harxhar-overleaf}"  # the Overleaf clone (overleaf_git.sh's DIR)
export GIT_TERMINAL_PROMPT=0 GCM_INTERACTIVE=never OVERLEAF_AUTOPUSH_RUNNING=1

ts() { date '+%Y-%m-%d %H:%M:%S'; }

git_retry() {  # git in this checkout, retried while a commit made at the same moment holds the index lock
    local i err
    for i in 1 2 3 4 5 6; do
        if err="$(git -C "$REPO" "$@" 2>&1)"; then
            [[ -z "$err" ]] || echo "$err"
            return 0
        fi
        grep -q "index.lock" <<< "$err" || break
        sleep 2
    done
    echo "$err"
    return 1
}

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
        rm -rf "$tmp"
        return 1
    fi
    # mirror the regenerated layers into this checkout (stale stubs removed with git rm)
    while IFS= read -r f; do
        [ -e "$wt/$f" ] || git_retry rm -q -- "$f"
    done < <(git -C "$REPO" ls-files -- writeup/beats)
    mkdir -p "$REPO/writeup/beats"
    cp -r "$wt/writeup/beats/." "$REPO/writeup/beats/"
    cp "$wt/writeup/paper_rehaul_beats.tex" "$wt/writeup/paper_rehaul_beats_top.tex" "$REPO/writeup/"
    git -C "$REPO" worktree remove --force "$wt" >/dev/null 2>&1
    rm -rf "$tmp"
    git_retry add -A -- "${LAYERS[@]}" || return 1
    if git -C "$REPO" diff --cached --quiet -- "${LAYERS[@]}"; then
        echo "autopush: review layers already match $sha"
    elif git_retry commit -q -m "Review layers regenerated from $sha (autopush)" -- "${LAYERS[@]}"; then
        echo "autopush: review layers regenerated from $sha and committed"
    fi
}

pass() {  # one update of Overleaf and GitHub from HEAD
    local want br out rc t0
    t0=$(date +%s)
    want="$(git -C "$REPO" config --get overleaf.autopush)"
    br="$(git -C "$REPO" symbolic-ref --short -q HEAD)"
    if [[ -n "$want" && "$br" != "$want" ]]; then
        echo "$(ts) autopush: this checkout is on ${br:-a detached HEAD}, not $want; nothing pushed"
        return 0
    fi
    echo "$(ts) autopush: pass at $(git -C "$REPO" rev-parse --short HEAD) on ${br:-a detached HEAD}"
    regen_beats
    out="$(bash "$OG" push-all --check 2>&1)"
    rc=$?
    echo "$out" | grep -E "PASSED|FAILED|FAIL:|pushed|already|refuse|differs|own files|try [0-9]|uncommitted|errors|undefined|did not land|rejected|not committed|nothing reached" \
        || echo "$out" | tail -5
    # 2026-10-29: Overleaf has paper edits that are not pulled back (or it changed while the push was built): report, never force
    if echo "$out" | grep -qE "main2\.tex (differs from the last flatten|holds edits pulled back)|\[(remote )?rejected\]|non-fast-forward|fetch first"; then
        echo "$(ts) autopush: REFUSED: Overleaf has edits that are not pulled back; nothing was forced. pull --dry-run says:"
        bash "$OG" pull --dry-run 2>&1 | grep -E "^(unflatten|overleaf_git):" || true
    elif [[ $rc -ne 0 ]] && ! echo "$out" | grep -qE "beats(_top)?\.tex differs from its last flatten"; then
        echo "$(ts) autopush: FAILED: push-all exited $rc (see above)"
    fi
    # 2026-10-28 (author): an Overleaf-editor edit to a review layer that lies wholly inside its generated %% header (stray
    # keystrokes) is reverted and the layers pushed again; any other edit is left for the author (logged)
    if echo "$out" | grep -qE "beats(_top)?\.tex differs from its last flatten"; then
        if python3 "$SYNC/layer_header_fix.py" "$ODIR" "$SYNC"; then
            git -C "$ODIR" commit -q -m "Revert a stray editor edit in a review layer's generated header (autopush)" \
                -- beats.tex beats_top.tex && echo "autopush: stray header edit reverted; pushing the review layers again"
            bash "$OG" push-beats --check 2>&1 \
                | grep -E "PASSED|FAILED|FAIL:|pushed|already|refuse|differs|own files|try [0-9]|did not land|rejected" || true
        else
            echo "$(ts) autopush: REFUSED: a review layer was edited on Overleaf outside its header; left for the author"
        fi
    fi
    if [[ -n "$br" ]]; then
        git -C "$REPO" push -q origin "$br" 2>&1 | tail -1
    fi
    echo "$(ts) autopush: done in $(($(date +%s) - t0)) s"
}

relevant() {  # 0 if commit $1 changed writeup/, the sync's map files, the generated review layers and the log aside
    local parent
    parent="$(git -C "$REPO" rev-parse -q --verify "$1^" 2>/dev/null)" || parent=4b825dc642cb6eb9a060e54bf8d69288fbee4904
    ! git -C "$REPO" diff --quiet "$parent" "$1" -- writeup \
        ':(exclude)writeup/overleaf_sync/map.json' ':(exclude)writeup/overleaf_sync/map_beats.json' \
        ':(exclude)writeup/overleaf_sync/map_beats_top.json' ':(exclude)writeup/overleaf_sync/autopush.log' \
        ':(exclude)writeup/beats' ':(exclude)writeup/paper_rehaul_beats.tex' ':(exclude)writeup/paper_rehaul_beats_top.tex'
}

take_lock() {  # one runner at a time; a lock whose runner is gone (killed, rebooted) is taken over
    local pid
    mkdir -p "$STATE"
    if mkdir "$LOCK" 2>/dev/null; then echo $$ > "$LOCK/pid"; return 0; fi
    pid="$(cat "$LOCK/pid" 2>/dev/null)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then return 1; fi
    # no pid yet: another runner may have just made the lock; give it a minute
    [[ -n "$pid" || -n "$(find "$LOCK" -maxdepth 0 -mmin +1 2>/dev/null)" ]] || return 1
    echo "$(ts) autopush: taking over the lock of a runner that is gone (pid ${pid:-unknown})"
    rm -rf "$LOCK"
    if mkdir "$LOCK" 2>/dev/null; then echo $$ > "$LOCK/pid"; return 0; fi
    return 1
}

settle() {  # wait until the last request is $SETTLE s old
    local t left
    while t="$(cat "$PENDING" 2>/dev/null)" && [[ -n "$t" ]]; do
        [[ "$t" =~ ^[0-9]+$ ]] || return 0
        left=$((t + SETTLE - $(date +%s)))
        ((left > 0)) || return 0
        sleep $((left < 5 ? left : 5))   # in steps, so a removed request is noticed
    done
}

run() {  # the debounced runner
    take_lock || return 0   # a runner is active: it sees this request (the pending file) and runs once more
    while :; do
        settle
        if [[ -e "$PENDING" ]]; then
            rm -f "$PENDING"   # HEAD already holds every commit that asked so far
            pass
        fi
        [[ -e "$PENDING" ]] && continue
        rm -rf "$LOCK"
        # a commit that came in while the lock was being released: its runner found the lock and left
        [[ -e "$PENDING" ]] && take_lock && continue
        break
    done
}

main() {
    case "${1:-}" in
    trigger)  # from the hook, detached
        relevant "${2:-HEAD}" || return 0
        mkdir -p "$STATE"
        date +%s > "$PENDING"
        run 2>&1 | sed -u -E "$MASK" >> "$LOG"
        ;;
    once)
        take_lock || { echo "autopush: a run is in progress (lock $LOCK)"; return 1; }
        pass 2>&1 | sed -u -E "$MASK"
        rm -rf "$LOCK"
        ;;
    *)
        sed -n '8,12p' "${BASH_SOURCE[0]}"
        return 1
        ;;
    esac
}

# one line, read whole before it runs: a checkout that rewrites this file mid-run cannot change a running pass
main "$@"; exit $?
