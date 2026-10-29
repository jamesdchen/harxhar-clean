#!/usr/bin/env bash
# install_hook.sh [--disable] (2026-10-29, author: "Shouldn't the Overleaf update be automatic/a part of the pipeline").
#   bash writeup/overleaf_sync/install_hook.sh            # install the post-commit hook; push after commits on this
#                                                         # worktree's current branch
#   bash writeup/overleaf_sync/install_hook.sh --disable  # stop (the hook file stays and does nothing)
# The hook (hooks/post-commit) is copied into the repository's hooks directory (git rev-parse --git-path hooks), which
# all worktrees share; the per-worktree setting overleaf.autopush = <branch> limits it to this worktree and branch.
# Re-run after editing hooks/post-commit. See README.md, "Automatic push after every commit".
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$REPO/writeup/overleaf_sync/hooks/post-commit"
MARK="harxhar overleaf autopush hook"
die() { echo "install_hook: $*" >&2; exit 1; }

if [[ "${1:-}" == --disable ]]; then
    git -C "$REPO" config --worktree --unset overleaf.autopush || true
    echo "install_hook: automatic Overleaf push disabled in $REPO"
    exit 0
fi
[[ $# -eq 0 ]] || die "usage: install_hook.sh [--disable]"

br="$(git -C "$REPO" symbolic-ref --short -q HEAD)" || die "$REPO is on a detached HEAD; check out the paper's branch first"
hooks="$(git -C "$REPO" rev-parse --path-format=absolute --git-path hooks)"
dest="$hooks/post-commit"
if [[ -e "$dest" ]] && ! grep -q "$MARK" "$dest"; then
    die "$dest exists and is another hook; not replacing it (merge the two by hand)"
fi
me="$(git -C "$REPO" rev-parse --path-format=absolute --show-toplevel)"
while IFS= read -r line; do  # one worktree only: two runners would share the Overleaf clone
    [[ "$line" == "worktree "* ]] || continue
    wt="${line#worktree }"
    [[ "$(git -C "$wt" rev-parse --path-format=absolute --show-toplevel 2>/dev/null)" != "$me" ]] || continue
    other="$(git -C "$wt" config --worktree --get overleaf.autopush 2>/dev/null)" || continue
    die "the automatic push is already enabled in $wt (branch $other); disable it there first"
done < <(git -C "$REPO" worktree list --porcelain)

mkdir -p "$hooks"
tr -d '\r' < "$SRC" > "$dest"   # LF only, whatever the checkout's line endings
chmod +x "$dest" 2>/dev/null || true
git -C "$REPO" config --worktree overleaf.autopush "$br" \
    || die "could not set overleaf.autopush for this worktree (git config extensions.worktreeConfig true, then re-run)"
echo "install_hook: installed $dest"
echo "install_hook: Overleaf is pushed after every commit on $br in $me that changes writeup/"
echo "install_hook: log: writeup/overleaf_sync/autopush.log; disable: bash writeup/overleaf_sync/install_hook.sh --disable"
