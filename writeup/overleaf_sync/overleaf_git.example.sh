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
#
# 2026-10-20 (user): the project already holds an earlier draft (main.tex, references.bib, sections/, ...), which stays
# as it is. The flat paper is written as main2.tex, with its bibliography as references2.bib, and nothing else is
# added or changed in the project (set main2.tex as the main document in Overleaf to compile it).
#
# push: clones (or updates) the Overleaf project in $OVERLEAF_DIR. It refuses if Overleaf's main2.tex differs from the
#       last flatten in writeup/overleaf_sync/map.json, because that means human edits not yet pulled back; --first
#       allows the very first push (no main2.tex yet). It flattens (flatten.py --name main2.tex --bib-suffix 2, with
#       --check if given), commits only main2.tex and references2.bib, pushes to Overleaf, then commits map.json in
#       harxhar-clean (push that commit yourself).
# pull: pulls the Overleaf project and runs unflatten.py on its main2.tex (--dry-run: report only). Then rebuild the
#       paper, and run `push` so Overleaf gets the regenerated main2.tex.
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

og() {  # git with the token helper when a token is known
    if [[ -n "$TOKEN" ]]; then
        OVERLEAF_GIT_TOKEN="$TOKEN" git -c credential.helper= \
            -c credential.helper='!f() { echo username=git; echo "password=${OVERLEAF_GIT_TOKEN}"; }; f' "$@"
    else
        git "$@"
    fi
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
    [[ -f "$MAP" && -f "$DIR/$FLAT" ]] || return 1
    python3 - "$MAP" "$DIR/$FLAT" <<'EOF'
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
    python3 "$SYNC/flatten.py" --out "$DIR" --map "$MAP" --name "$FLAT" --bib-suffix "$BIB_SUFFIX" ${check[@]+"${check[@]}"}
    sha="$(git -C "$REPO" rev-parse --short HEAD)"
    git -C "$DIR" add -- "$FLAT" "references$BIB_SUFFIX.bib"   # only the flat paper's files, never the project's own
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
*)
    sed -n '9,10p' "${BASH_SOURCE[0]}"
    exit 1
    ;;
esac
