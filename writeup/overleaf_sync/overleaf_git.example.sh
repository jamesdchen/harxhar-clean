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
#       2026-10-29 (X5b, deliberation 2026-10-29): the map is written to a temporary file and moved into place only
#       once the push has landed; a failed --check, a rejected push or any other stop resets the Overleaf clone to
#       what was pulled, so the next run starts clean and map.json never records a text Overleaf did not get.
# pull: pulls the Overleaf project and runs unflatten.py on its main2.tex (--dry-run: report only). Then rebuild the
#       paper, and run `push` so Overleaf gets the regenerated main2.tex.
#       2026-10-29 (X5a): a real pull records what it consumed in map_pulled.json (Overleaf commit, hashes of the
#       pulled main2.tex and of the flatten it was diffed against). Commit that file with the pulled edits: `push`
#       then accepts Overleaf's main2.tex when it is that pulled text (it used to refuse forever after a real pull).
#       Edits unflatten could not apply (conflicts, exit 2) are overwritten by that push, as SPEC section 5 says;
#       they stay in REPORT.md and in the Overleaf commit named in map_pulled.json.
# push-beats (2026-10-27, author: review the story in layers): flattens the two review layers that
#       writeup/make_paper_beats.py generates and that are committed, writeup/paper_rehaul_beats.tex (main text, all
#       bullets) and writeup/paper_rehaul_beats_top.tex (top-level bullets only), into beats.tex and beats_top.tex with
#       referencesb.bib. It reads a temporary clean checkout of HEAD (writeup/ only), never the working tree, so
#       uncommitted numbers cannot reach Overleaf. Each layer has its own map (map_beats.json, map_beats_top.json); it
#       refuses if Overleaf's copy differs from its last flatten (the layers are for reading and comments; edits go
#       to the sources) or if a target file is one of the project's own. main2.tex, references2.bib, map.json and the
#       figures are not touched, and a layer that would copy a figure stops the push. (2026-10-29: its maps, too, are
#       moved into place only once the push has landed.)
# push-all: (2026-10-29, X5c) runs push-beats even when push refuses or fails; exits non-zero if either did.
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
PULLED="$SYNC/map_pulled.json"   # 2026-10-29 (X5a): what the last real pull consumed
FLAT="main2.tex"      # the project's own main.tex stays untouched
BIB_SUFFIX="2"        # references.bib is written as references2.bib (the project has its own references.bib)
# 2026-10-29: flatten an LF checkout. Under core.autocrlf the sources came out CRLF while flatten's own header and
# marker lines are LF, so the map's text had mixed line endings; a main2.tex pulled back from Overleaf is all CRLF, and
# unflatten then saw every header and marker line as an edit (structural conflicts, nothing applied).
LF=(-c core.autocrlf=false -c core.eol=lf)
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
    local ahead dirty ours
    if [[ -d "$DIR/.git" ]]; then
        # 2026-10-29 (X5b): a run killed mid-flatten (e.g. during the minutes-long --check) leaves the files it writes;
        # only these scripts write them in the clone (Overleaf edits arrive as commits), so restore them
        dirty="$(git -C "$DIR" status --porcelain)"
        ours=("$FLAT" "references$BIB_SUFFIX.bib" beats.tex beats_top.tex referencesb.bib)
        if [[ -n "$dirty" ]] && ! cut -c4- <<< "$dirty" | grep -qvxF "${ours[@]/#/-e}"; then
            echo "overleaf_git: restoring the files a stopped run left in $DIR" >&2
            git -C "$DIR" reset -q --hard HEAD
            git -C "$DIR" clean -fq -- "${ours[@]}"
        fi
        [[ -z "$(git -C "$DIR" status --porcelain)" ]] || die "$DIR has uncommitted changes; commit or discard them first"
        # 2026-10-29 (X5b): a run killed between its commit and its push leaves flatten commits Overleaf never got;
        # they would read as Overleaf edits. Drop them when every unpushed commit is one of those.
        ahead="$(git -C "$DIR" log --format=%s '@{u}..HEAD' 2>/dev/null)" || ahead=""
        if [[ -n "$ahead" ]] && ! grep -qvE '^(Flatten|Review layers) from harxhar-clean ' <<< "$ahead"; then
            echo "overleaf_git: dropping $(wc -l <<< "$ahead") flatten commit(s) in $DIR that never reached Overleaf" >&2
            git -C "$DIR" reset -q --hard '@{u}'
        fi
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
    flat_py same "$1" "$2"
}

flat_py() {  # same MAP FILE | pulled MARK MAP FILE | record MARK MAP FILE OVERLEAF_SHA UNFLATTEN_EXIT
    python3 - "$@" <<'EOF'
import hashlib, json, sys
def norm(t): return t.lstrip("﻿").replace("\r\n", "\n").rstrip("\n")  # 2026-10-23: Windows checkouts flatten with CRLF
def flat(p): return norm(json.load(open(p, encoding="utf-8"))["flat_text"])
def text(p): return norm(open(p, encoding="utf-8", newline="").read())
def sha(t): return hashlib.sha256(t.encode("utf-8")).hexdigest()
mode, a = sys.argv[1], sys.argv[2:]
if mode == "same":
    sys.exit(0 if text(a[1]) == flat(a[0]) else 1)
if mode == "pulled":  # the file is the text a pull consumed, and the map still holds the flatten it was diffed against
    m = json.load(open(a[0], encoding="utf-8"))
    sys.exit(0 if m.get("base_sha256") == sha(flat(a[1])) and m.get("pulled_sha256") == sha(text(a[2])) else 1)
assert mode == "record", mode
m = {"overleaf_commit": a[3], "base_sha256": sha(flat(a[1])), "pulled_sha256": sha(text(a[2])), "unflatten_exit": int(a[4])}
with open(a[0], "w", encoding="utf-8", newline="\n") as fh:
    fh.write(json.dumps(m, indent=1) + "\n")
EOF
}

pulled_in() {  # 0 if Overleaf's main2.tex is the text that the pull recorded in the mark file $1 consumed
    [[ -s "$1" && -f "$MAP" && -f "$DIR/$FLAT" ]] || return 1
    flat_py pulled "$1" "$MAP" "$DIR/$FLAT"
}

pulled_at_head() {  # pulled_in for the mark committed at HEAD: the pulled edits are in what push flattens
    local m rc=1
    m="$(mktemp)"
    if git -C "$REPO" show "HEAD:writeup/overleaf_sync/map_pulled.json" > "$m" 2>/dev/null && pulled_in "$m"; then rc=0; fi
    rm -f "$m"
    return $rc
}

restore_clone() {  # 2026-10-29 (X5b): undo a push that did not land: the clone back at $1, the files written removed
    local base="$1"; shift
    git -C "$DIR" reset -q --hard "$base" || true
    git -C "$DIR" clean -fq -- "$@" >/dev/null 2>&1 || true
    echo "overleaf_git: nothing reached Overleaf; the Overleaf clone is back at ${base:0:7} and the map file(s) are unchanged" >&2
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
    # 2026-10-29 (X5a): Overleaf may also hold the text the last pull consumed, once that pull is committed
    if ! same_as_last_flatten && ! pulled_at_head && ! [[ $first -eq 1 && ! -e "$DIR/$FLAT" ]]; then
        pulled_in "$PULLED" && die "Overleaf's $FLAT holds edits pulled back but not committed: commit them with writeup/overleaf_sync/map_pulled.json, then push"
        [[ $first -eq 0 ]] || die "--first, but $DIR/$FLAT already exists with other content; pull it back first"
        die "Overleaf's $FLAT differs from the last flatten (map.json): run 'pull' first to bring those edits back (or --first for the very first push)"
    fi
    # 2026-10-28 (author: update Overleaf faster): flatten a clean checkout of HEAD (writeup/ only), as push-beats
    # does, so uncommitted edits never reach Overleaf and committed work can be pushed while others are editing.
    dirty="$(git -C "$REPO" status --porcelain --untracked-files=no -- writeup | wc -l)"
    [[ "$dirty" -eq 0 ]] || echo "overleaf_git: $dirty uncommitted file(s) under writeup/ are NOT pushed (HEAD is)"
    ptmp="$(mktemp -d)"
    pwt="$ptmp/wt"
    figlist="$ptmp/figures"
    base="$(git -C "$DIR" rev-parse HEAD)"
    landed=0
    trap 'restore=(); [[ $landed -eq 1 ]] || { mapfile -t restore < <(cat "$figlist" 2>/dev/null); restore_clone "$base" "$FLAT" "references$BIB_SUFFIX.bib" ${restore[@]+"${restore[@]}"}; }; git -C "$REPO" worktree remove --force "$pwt" >/dev/null 2>&1 || true; rm -rf "$ptmp"' EXIT
    git -C "$REPO" worktree add -q --no-checkout --detach "$pwt" HEAD
    git -C "$pwt" "${LF[@]}" sparse-checkout set --cone writeup
    git -C "$pwt" "${LF[@]}" checkout -q --detach HEAD
    python3 "$pwt/writeup/overleaf_sync/flatten.py" --entry "$pwt/writeup/paper_rehaul.tex" --out "$DIR" --map "$ptmp/map.json" \
        --name "$FLAT" --bib-suffix "$BIB_SUFFIX" --figures-out "$figlist" ${check[@]+"${check[@]}"}
    figs=()
    while IFS= read -r f; do [[ -n "$f" ]] && figs+=("$f"); done < "$figlist"
    for f in ${figs[@]+"${figs[@]}"}; do  # a figure may replace only a file an earlier flatten committed
        if git -C "$DIR" ls-files --error-unmatch -- "$f" >/dev/null 2>&1 \
            && [[ "$(git -C "$DIR" log -1 --format=%s -- "$f")" != "Flatten from harxhar-clean "* ]]; then
            die "$f is one of the project's own files; not overwriting it"
        fi
    done
    sha="$(git -C "$REPO" rev-parse --short HEAD)"
    git -C "$DIR" add -- "$FLAT" "references$BIB_SUFFIX.bib" ${figs[@]+"${figs[@]}"}   # only the flat paper's files, never the project's own
    if git -C "$DIR" diff --cached --quiet; then
        echo "overleaf_git: Overleaf already has this flatten; nothing to push"
    else
        git -C "$DIR" commit -q -m "Flatten from harxhar-clean $sha (main2.tex, references2.bib)"
        og -C "$DIR" push || die "the push to Overleaf did not land (rejected or failed; if Overleaf changed meanwhile, run 'pull --dry-run')"
        echo "overleaf_git: pushed the flatten of $sha to Overleaf"
    fi
    landed=1
    cp "$ptmp/map.json" "$MAP"   # 2026-10-29 (X5b): only now, with the flatten on Overleaf
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
    base="$(git -C "$DIR" rev-parse HEAD)"
    landed=0
    trap 'restore=(); [[ $landed -eq 1 ]] || { mapfile -t restore < <(cat "$tmp"/figs_* 2>/dev/null); restore_clone "$base" beats.tex beats_top.tex referencesb.bib ${restore[@]+"${restore[@]}"}; }; git -C "$REPO" worktree remove --force "$wt" >/dev/null 2>&1 || true; rm -rf "$tmp"' EXIT
    git -C "$REPO" worktree add -q --no-checkout --detach "$wt" HEAD
    git -C "$wt" "${LF[@]}" sparse-checkout set --cone writeup
    git -C "$wt" "${LF[@]}" checkout -q --detach HEAD
    files=(referencesb.bib)
    for l in "${layers[@]}"; do
        IFS=: read -r entry flat map <<< "$l"
        python3 "$wt/writeup/overleaf_sync/flatten.py" --entry "$wt/writeup/$entry.tex" --out "$DIR" \
            --map "$tmp/$map" --name "$flat" --bib-suffix b --figures-out "$tmp/figs_$flat" ${check[@]+"${check[@]}"}
        [[ ! -s "$tmp/figs_$flat" ]] || die "$flat would copy figures ($(tr '\n' ' ' < "$tmp/figs_$flat")); the review layers must not"
        files+=("$flat")
    done
    sha="$(git -C "$REPO" rev-parse --short HEAD)"
    git -C "$DIR" add -- "${files[@]}"   # only the review layers' files
    if git -C "$DIR" diff --cached --quiet; then
        echo "overleaf_git: Overleaf already has these review layers; nothing to push"
    else
        git -C "$DIR" commit -q -m "Review layers from harxhar-clean $sha (beats.tex, beats_top.tex, referencesb.bib)"
        og -C "$DIR" push || die "the push of the review layers to Overleaf did not land (rejected or failed)"
        echo "overleaf_git: pushed the review layers of $sha to Overleaf"
    fi
    landed=1
    for l in "${layers[@]}"; do  # 2026-10-29 (X5b): the maps only now, with the layers on Overleaf
        IFS=: read -r entry flat map <<< "$l"
        cp "$tmp/$map" "$SYNC/$map"
    done
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
    if pulled_in "$PULLED"; then  # 2026-10-29 (X5a): applying the same edits twice would park them twice
        echo "overleaf_git: Overleaf's $FLAT was already pulled back (map_pulled.json); commit the pulled edits with it, then push"
        exit 0
    fi
    set +e
    python3 "$SYNC/unflatten.py" --main "$DIR/$FLAT" --map "$MAP" ${dry[@]+"${dry[@]}"}
    rc=$?
    set -e
    if [[ ${#dry[@]} -eq 0 && ( $rc -eq 0 || $rc -eq 2 ) ]]; then
        ocommit="$(git -C "$DIR" rev-parse HEAD)"
        flat_py record "$PULLED" "$MAP" "$DIR/$FLAT" "$ocommit" "$rc"
        echo "overleaf_git: recorded this pull in map_pulled.json (Overleaf commit ${ocommit:0:7}): commit it with the pulled edits, then push"
        [[ $rc -eq 0 ]] || echo "overleaf_git: the conflicting edits were not applied; the next push overwrites them on Overleaf (they stay in REPORT.md and in Overleaf commit ${ocommit:0:7})"
    fi
    echo "overleaf_git: report in $SYNC/REPORT.md (unflatten exit $rc: 0 = clean, 2 = conflicts listed in the report)"
    exit $rc
    ;;
push-all)  # 2026-10-28 (author: update Overleaf faster): the paper and the review layers in one command
    # 2026-10-29 (X5c): the review layers are pushed even when the paper push refuses or fails (they used to be skipped)
    rc=0
    bash "${BASH_SOURCE[0]}" push "$@" || rc=$?
    beats_args=()
    for a in "$@"; do [[ "$a" == --first ]] || beats_args+=("$a"); done
    bash "${BASH_SOURCE[0]}" push-beats ${beats_args[@]+"${beats_args[@]}"} || { r=$?; [[ $rc -ne 0 ]] || rc=$r; }
    exit $rc
    ;;
*)
    sed -n '9,12p' "${BASH_SOURCE[0]}"
    exit 1
    ;;
esac
