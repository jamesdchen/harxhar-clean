#!/usr/bin/env bash
# overleaf_git.example.sh -- copy to overleaf_git.sh (gitignored) and fill in the token line. Moves the flat paper between harxhar-clean and the Overleaf project's own git repository
# (2026-10-20). Overleaf gives every project a git URL; this replaces the separate GitHub repository.
#
# KEEP THIS FILE OUT OF GIT: it holds the Overleaf token below (.gitignore lists it; harxhar-clean is public).
# overleaf_git.example.sh is the tracked copy without the token: copy it to overleaf_git.sh and fill in the token line.
# OVERLEAF_GIT_TOKEN=
#
#   bash writeup/overleaf_sync/overleaf_git.sh push [--first] [--check]   # flatten the source -> Overleaf
#   bash writeup/overleaf_sync/overleaf_git.sh pull [--dry-run]           # Overleaf edits -> section files
#   bash writeup/overleaf_sync/overleaf_git.sh push-beats [--check]       # review layers (HEAD only) -> Overleaf
#   bash writeup/overleaf_sync/overleaf_git.sh push-all [--check]         # push, then push-beats (2026-10-28)
#
# 2026-10-20 (user): the project already holds an earlier draft (main.tex, references.bib, sections/, ...), which stays
# as it is. The flat paper is written as main2.tex, with its bibliography as references2.bib, and nothing else is
# added or changed in the project (set main2.tex as the main document in Overleaf to compile it).
# 2026-10-24: the flat paper's live figures are committed too, at their paths under writeup/ (flatten.py copied them
# but they were never committed, so Overleaf compiled without them); a figure never replaces a project file.
#
# push: (2026-10-28) flattens a clean checkout of HEAD, never the working tree: commit first.
# push: clones (or updates) the Overleaf project in $OVERLEAF_DIR. It refuses if Overleaf's main2.tex differs from the
#       last flatten in writeup/overleaf_sync/map.json, because that means human edits not yet pulled back; --first
#       allows the very first push (no main2.tex yet). It flattens (flatten.py --name main2.tex --bib-suffix 2, with
#       --check if given), commits main2.tex, references2.bib and the figures it copied, pushes to Overleaf, then commits map.json in
#       harxhar-clean (push that commit yourself).
# pull: pulls the Overleaf project and runs unflatten.py on its main2.tex (--dry-run: report only). Then rebuild the
#       paper, and run `push` so Overleaf gets the regenerated main2.tex.
# push-beats (2026-10-27, author: review the story in layers): flattens the two review layers that
#       writeup/make_paper_beats.py generates and that are committed, writeup/paper_rehaul_beats.tex (main text, all
#       bullets) and writeup/paper_rehaul_beats_top.tex (top-level bullets only), into beats.tex and beats_top.tex with
#       referencesb.bib. It reads a temporary clean checkout of HEAD (writeup/ only), never the working tree, so
#       uncommitted numbers cannot reach Overleaf. Each layer has its own map (map_beats.json, map_beats_top.json); it
#       refuses if Overleaf's copy differs from its last flatten (the layers are for reading and comments; edits go
#       to the sources) or if a target file is one of the project's own. main2.tex, references2.bib, map.json and the
#       figures are not touched, and a layer that would copy a figure stops the push.
#
# Authentication: Overleaf's git user is "git" and the password is an Overleaf Git token (Account settings).
#   * $OVERLEAF_GIT_TOKEN, else the OVERLEAF_GIT_TOKEN comment line above, is handed to git by a credential helper
#     (never written into a URL, a git config or a commit);
#   * with neither, git uses your own credential store or prompts (on your machine), or a proxy-injected secret.
# Settings (environment): OVERLEAF_GIT_URL (default the paper's project), OVERLEAF_DIR (default ../harxhar-overleaf,
# next to the harxhar-clean checkout).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SYNC="$REPO/writeup/overleaf_sync"
URL="${OVERLEAF_GIT_URL:-https://git.overleaf.com/68d4843e51cfc223ab18adc2}"
DIR="${OVERLEAF_DIR:-$REPO/../harxhar-overleaf}"
MAP="$SYNC/map.json"
FLAT="main2.tex"      # the project's own main.tex stays untouched
BIB_SUFFIX="2"        # references.bib is written as references2.bib (the project has its own references.bib)
TOKEN="${OVERLEAF_GIT_TOKEN:-$(sed -n 's/^# OVERLEAF_GIT_TOKEN=//p' "${BASH_SOURCE[0]}" | head -n 1)}"

die() { echo "overleaf_git: $*" >&2; exit 1; }

og_once() {  # git with the token helper when a token is known
    if [[ -n "$TOKEN" ]]; then
        OVERLEAF_GIT_TOKEN="$TOKEN" git -c credential.helper= \
            -c credential.helper='!f() { echo username=git; echo "password=${OVERLEAF_GIT_TOKEN}"; }; f' "$@"
    else
        git "$@"
    fi
}

og() {  # og_once, retried while Overleaf's git bridge fails transiently (2026-10-28: 503s are frequent)
    local i err rc
    err="$(mktemp)"
    for i in 1 2 3 4 5 6 7 8; do
        if GIT_TERMINAL_PROMPT=0 GCM_INTERACTIVE=never og_once "$@" 2>"$err"; then rc=0; else rc=$?; fi
        cat "$err" >&2
        if [[ $rc -eq 0 ]] || ! grep -qE "error: (502|503|504)|upstream connect|no healthy upstream|Connection (reset|timed out|refused)|Could not resolve host|unable to access" "$err"; then
            rm -f "$err"
            return $rc
        fi
        echo "overleaf_git: Overleaf's git bridge failed (try $i of 8); retrying in $((15 * i)) s" >&2
        sleep $((15 * i))
    done
    rm -f "$err"
    return $rc
}

checkout() {  # clone the project, or bring an existing clone up to date
    if [[ -d "$DIR/.git" ]]; then
        [[ -z "$(git -C "$DIR" status --porcelain)" ]] || die "$DIR has uncommitted changes; commit or discard them first"
        og -C "$DIR" pull --ff-only
    else
        og clone "$URL" "$DIR"
    fi
}

same_as_last_flatten() {  # 0 if Overleaf's main2.tex equals the flat text recorded in map.json
    same_flat "$MAP" "$DIR/$FLAT"
}

same_flat() {  # 0 if the file $2 equals the flat text recorded in the map $1
    [[ -f "$1" && -f "$2" ]] || return 1
    python3 - "$1" "$2" <<'EOF'
import json, sys
flat = json.load(open(sys.argv[1], encoding="utf-8"))["flat_text"].replace("\r\n", "\n")  # 2026-10-23: normalize both sides (Windows checkouts flatten with CRLF)
cur = open(sys.argv[2], encoding="utf-8", newline="").read().lstrip("﻿").replace("\r\n", "\n")
sys.exit(0 if cur.rstrip("\n") == flat.rstrip("\n") else 1)
EOF
}

cmd="${1:-}"; shift || true
case "$cmd" in
push)
    first=0; check=()
    for a in "$@"; do
        case "$a" in
            --first) first=1 ;;
            --check) check=(--check) ;;
            *) die "unknown option $a" ;;
        esac
    done
    checkout
    if ! same_as_last_flatten && [[ $first -eq 0 ]]; then
        die "Overleaf's $FLAT differs from the last flatten (map.json): run 'pull' first to bring those edits back (or --first for the very first push)"
    fi
    if [[ $first -eq 1 && -e "$DIR/$FLAT" ]] && ! same_as_last_flatten; then
        die "--first, but $DIR/$FLAT already exists with other content; pull it back first"
    fi
    figlist="$(mktemp)"
    # 2026-10-28 (author: update Overleaf faster): flatten a clean checkout of HEAD (writeup/ only), as push-beats
    # does, so uncommitted edits never reach Overleaf and committed work can be pushed while others are editing.
    dirty="$(git -C "$REPO" status --porcelain --untracked-files=no -- writeup | wc -l)"
    [[ "$dirty" -eq 0 ]] || echo "overleaf_git: $dirty uncommitted file(s) under writeup/ are NOT pushed (HEAD is)"
    ptmp="$(mktemp -d)"
    pwt="$ptmp/wt"
    git -C "$REPO" worktree add -q --no-checkout --detach "$pwt" HEAD
    trap 'git -C "$REPO" worktree remove --force "$pwt" >/dev/null 2>&1 || true' EXIT
    git -C "$pwt" sparse-checkout set --cone writeup
    git -C "$pwt" checkout -q --detach HEAD
    python3 "$pwt/writeup/overleaf_sync/flatten.py" --entry "$pwt/writeup/paper_rehaul.tex" --out "$DIR" --map "$MAP" \
        --name "$FLAT" --bib-suffix "$BIB_SUFFIX" --figures-out "$figlist" ${check[@]+"${check[@]}"}
    figs=()
    while IFS= read -r f; do [[ -n "$f" ]] && figs+=("$f"); done < "$figlist"
    rm -f "$figlist"
    for f in ${figs[@]+"${figs[@]}"}; do  # a figure may replace only a file an earlier flatten committed
        if git -C "$DIR" ls-files --error-unmatch -- "$f" >/dev/null 2>&1 \
            && [[ "$(git -C "$DIR" log -1 --format=%s -- "$f")" != "Flatten from harxhar-clean "* ]]; then
            die "$f is one of the project's own files; not overwriting it (restore: git -C \"$DIR\" checkout -- \"$f\")"
        fi
    done
    sha="$(git -C "$REPO" rev-parse --short HEAD)"
    git -C "$DIR" add -- "$FLAT" "references$BIB_SUFFIX.bib" ${figs[@]+"${figs[@]}"}   # only the flat paper's files, never the project's own
    if git -C "$DIR" diff --cached --quiet; then
        echo "overleaf_git: Overleaf already has this flatten; nothing to push"
    else
        git -C "$DIR" commit -q -m "Flatten from harxhar-clean $sha (main2.tex, references2.bib)"
        og -C "$DIR" push
        echo "overleaf_git: pushed the flatten of $sha to Overleaf"
    fi
    git -C "$REPO" add "$MAP"
    if ! git -C "$REPO" diff --cached --quiet -- "$MAP"; then
        git -C "$REPO" commit -q -m "Overleaf sync: map.json of the flatten of $sha pushed to Overleaf" -- "$MAP"
        echo "overleaf_git: committed map.json in harxhar-clean (push it with git push)"
    fi
    ;;
push-beats)
    check=()
    for a in "$@"; do
        case "$a" in
            --check) check=(--check) ;;
            *) die "unknown option $a" ;;
        esac
    done
    layers=("paper_rehaul_beats:beats.tex:map_beats.json" "paper_rehaul_beats_top:beats_top.tex:map_beats_top.json")
    checkout
    for l in "${layers[@]}"; do
        IFS=: read -r entry flat map <<< "$l"
        git -C "$REPO" cat-file -e "HEAD:writeup/$entry.tex" 2>/dev/null \
            || die "writeup/$entry.tex is not committed: run writeup/make_paper_beats.py on a clean checkout and commit its output"
        if [[ -e "$DIR/$flat" ]] && ! same_flat "$SYNC/$map" "$DIR/$flat"; then
            die "Overleaf's $flat differs from its last flatten ($map): it was edited there; move those edits to the sources first"
        fi
    done
    for f in beats.tex beats_top.tex referencesb.bib; do  # never replace a file the project made itself
        # 2026-10-28: a layer whose content is exactly its last flatten is ours, whatever its last commit says (e.g.
        # the revert of a stray edit made in the Overleaf editor)
        fm=""; [[ "$f" == beats.tex ]] && fm=map_beats.json; [[ "$f" == beats_top.tex ]] && fm=map_beats_top.json
        if git -C "$DIR" ls-files --error-unmatch -- "$f" >/dev/null 2>&1 \
            && [[ "$(git -C "$DIR" log -1 --format=%s -- "$f")" != "Review layers from harxhar-clean "* ]] \
            && ! { [[ -n "$fm" ]] && same_flat "$SYNC/$fm" "$DIR/$f"; }; then
            die "$f is one of the project's own files; not overwriting it"
        fi
    done
    tmp="$(mktemp -d)"
    wt="$tmp/wt"
    git -C "$REPO" worktree add -q --no-checkout --detach "$wt" HEAD
    trap 'git -C "$REPO" worktree remove --force "$wt" >/dev/null 2>&1 || true' EXIT
    git -C "$wt" sparse-checkout set --cone writeup
    git -C "$wt" checkout -q --detach HEAD
    files=(referencesb.bib)
    for l in "${layers[@]}"; do
        IFS=: read -r entry flat map <<< "$l"
        python3 "$wt/writeup/overleaf_sync/flatten.py" --entry "$wt/writeup/$entry.tex" --out "$DIR" \
            --map "$SYNC/$map" --name "$flat" --bib-suffix b --figures-out "$tmp/figs_$flat" ${check[@]+"${check[@]}"}
        [[ ! -s "$tmp/figs_$flat" ]] || die "$flat would copy figures ($(tr '\n' ' ' < "$tmp/figs_$flat")); the review layers must not"
        files+=("$flat")
    done
    sha="$(git -C "$REPO" rev-parse --short HEAD)"
    git -C "$DIR" add -- "${files[@]}"   # only the review layers' files
    if git -C "$DIR" diff --cached --quiet; then
        echo "overleaf_git: Overleaf already has these review layers; nothing to push"
    else
        git -C "$DIR" commit -q -m "Review layers from harxhar-clean $sha (beats.tex, beats_top.tex, referencesb.bib)"
        og -C "$DIR" push
        echo "overleaf_git: pushed the review layers of $sha to Overleaf"
    fi
    maps=("$SYNC/map_beats.json" "$SYNC/map_beats_top.json")
    git -C "$REPO" add -- "${maps[@]}"
    if ! git -C "$REPO" diff --cached --quiet -- "${maps[@]}"; then
        git -C "$REPO" commit -q -m "Overleaf sync: review-layer maps of the flatten of $sha pushed to Overleaf" -- "${maps[@]}"
        echo "overleaf_git: committed the review-layer maps in harxhar-clean (push them with git push)"
    fi
    ;;
pull)
    dry=()
    for a in "$@"; do
        case "$a" in
            --dry-run) dry=(--dry-run) ;;
            *) die "unknown option $a" ;;
        esac
    done
    checkout
    if same_as_last_flatten; then
        echo "overleaf_git: no edits in Overleaf since the last flatten"
        exit 0
    fi
    set +e
    python3 "$SYNC/unflatten.py" --main "$DIR/$FLAT" --map "$MAP" ${dry[@]+"${dry[@]}"}
    rc=$?
    set -e
    echo "overleaf_git: report in $SYNC/REPORT.md (unflatten exit $rc: 0 = clean, 2 = conflicts listed in the report)"
    exit $rc
    ;;
push-all)  # 2026-10-28 (author: update Overleaf faster): the paper and the review layers in one command
    bash "${BASH_SOURCE[0]}" push "$@"
    bash "${BASH_SOURCE[0]}" push-beats "$@"
    ;;
*)
    sed -n '9,12p' "${BASH_SOURCE[0]}"
    exit 1
    ;;
esac
