# Overleaf sync for the paper (2026-10-20)

The paper's source of record stays in this repository:
* `writeup/paper_rehaul.tex`;
* the section files in `writeup/rehaul/`;
* the numbers and tables in `writeup/generated/`, each written by its `writeup/make_*.py` generator.

Overleaf works on a flat copy in a separate GitHub repository, `jamesdchen/harxhar-paper-overleaf`. It is a single
`main.tex` with literal numbers, no `\input`, plus `references.bib` and any figures. People edit it in Overleaf
like any LaTeX file. The two scripts here move edits between the two:

| Direction | Script | What it does |
|---|---|---|
| source -> Overleaf | `flatten.py` | Inlines every section and generated table and replaces every number macro by its value. Writes `main.tex` and `map.json`, a record of where each character of `main.tex` came from. |
| Overleaf -> source | `unflatten.py` | Diffs the edited `main.tex` against the last flatten. It applies prose edits to the section files, parks the replaced lines in dated comments, and reports edits to numbers and tables as conflicts. |

The full specification is `SPEC.md`.

## Push the current paper to Overleaf

The new repository's README and `.gitignore` are in `overleaf_repo/`. That folder holds `gitignore`, which is the
`.gitignore` to copy in. Copy both into an empty clone before the first flatten.


```sh
git clone https://github.com/jamesdchen/harxhar-paper-overleaf ../harxhar-paper-overleaf   # once
python writeup/overleaf_sync/flatten.py --out ../harxhar-paper-overleaf --check
git -C ../harxhar-paper-overleaf add -A && git -C ../harxhar-paper-overleaf commit -m "Flatten from harxhar-clean <sha>"
git -C ../harxhar-paper-overleaf push
git add writeup/overleaf_sync/map.json && git commit -m "Overleaf sync: map of the flatten pushed to Overleaf"
```

* `--check` builds both `main.tex` and `paper_rehaul.tex` and confirms that their PDF texts are identical.
* In Overleaf, pull the GitHub changes into the project from its GitHub menu. The first time, create the project with
  New Project, then Import from GitHub.
* Commit `map.json` with every flatten that is pushed: `unflatten.py` needs the exact text Overleaf started from.

## Through the Overleaf project's git URL (current route, 2026-10-20)

The paper's Overleaf project, `https://git.overleaf.com/68d4843e51cfc223ab18adc2`, already holds an earlier draft
(`main.tex`, `references.bib`, `sections/` and more). That draft stays as it is. The flat paper is added next to it as
`main2.tex`, with its bibliography as `references2.bib` (`flatten.py --name main2.tex --bib-suffix 2`). In Overleaf,
set `main2.tex` as the main document to compile it.

`overleaf_git.sh` does the round trip in one command each way:

```sh
bash writeup/overleaf_sync/overleaf_git.sh push --first --check   # the first time; later: push [--check]
bash writeup/overleaf_sync/overleaf_git.sh pull --dry-run          # read REPORT.md, then:
bash writeup/overleaf_sync/overleaf_git.sh pull
```

* `push` refuses while Overleaf's `main2.tex` has edits that have not been pulled back. It commits only `main2.tex`
  and `references2.bib` to the project, then commits `map.json` here.
* `overleaf_git.sh` holds the Overleaf git token on a comment line, so it is in `.gitignore` and never committed:
  this repository is public. The tracked copy without the token is `overleaf_git.example.sh`. Copy it to
  `overleaf_git.sh` and fill in the `# OVERLEAF_GIT_TOKEN=` line, or set `OVERLEAF_GIT_TOKEN` in the environment
  instead.
* After a real `pull`, commit the pulled edits together with `map_pulled.json` (2026-10-29). `push` then accepts
  Overleaf's `main2.tex` as the pulled text and pushes over it. Before that commit it refuses, so the pulled edits are
  not reverted on Overleaf.
* A push that does not land (failed `--check`, rejected, stopped) leaves `map.json` and the Overleaf clone as they
  were. `push-all` pushes the review layers even when the paper push refuses.

## Automatic push after every commit (2026-10-29)

A git post-commit hook brings Overleaf up to date after every commit that changes `writeup/`. Commits that change only
the sync's map files or the generated review layers do not trigger it. The push does not depend on Claude Code.

```sh
bash writeup/overleaf_sync/install_hook.sh             # install; enables this worktree's current branch
bash writeup/overleaf_sync/install_hook.sh --disable   # stop (or: git config --worktree --unset overleaf.autopush)
OVERLEAF_AUTOPUSH=0 git commit ...                     # skip a single commit
bash writeup/overleaf_sync/autopush.sh once            # one pass now, in the foreground
```

* **What runs.** The hook returns at once, in well under a second: it starts one `git rev-parse` and nothing waits on
  the push. It starts `autopush.sh trigger <commit>` detached (immune to hangups, no inherited output). If that
  commit changed `writeup/`, the runner:
  1. waits until no commit has come in for 20 s;
  2. regenerates the review layers from a clean checkout of HEAD;
  3. runs `overleaf_git.sh push-all --check`;
  4. runs `git push` (the map and layer commits).

  A commit that arrives during a pass gets one more pass. Only one pass runs at a time. The lock is
  `$(git rev-parse --git-path overleaf-autopush)/lock`. A lock left by a killed run is taken over.
* **Log.** `writeup/overleaf_sync/autopush.log` (gitignored, token masked as `olp_***`). Each pass starts with
  `autopush: pass at <sha>` and ends with `autopush: done in N s`.
* **Failure.**
  * `autopush: REFUSED: Overleaf has edits that are not pulled back; nothing was forced.` Then the `pull --dry-run`
    summary (`unflatten: N changed block(s) ...`). Nothing is pulled automatically. When the pull rules allow it, run
    `overleaf_git.sh pull` and commit the pulled edits with `map_pulled.json`. That commit pushes.
  * `autopush: REFUSED: a review layer was edited on Overleaf outside its header`. Move that edit to the sources.
  * `autopush: FAILED: push-all exited N`. Lines above it such as `--check FAILED`, `did not land` or
    `make_paper_beats.py failed` give the cause.

  Nothing is retried until the next commit, and nothing is ever force-pushed.
* **Limits.** The hook acts in one worktree only, the one where `install_hook.sh` ran (`overleaf.autopush` in its
  `config.worktree`), and only on that branch. `git pull`, merge and rebase do not trigger it. A pass outlives the
  command that made the commit. A pass that is killed anyway (reboot, a closed console) is redone by the next commit:
  the lock is taken over, and what the killed run left in the Overleaf clone is cleaned up. While a pass commits, a
  commit made at the same moment may find `index.lock` taken. Retry it.

## Bring Overleaf edits back

```sh
# in Overleaf: push the project's changes to GitHub (GitHub menu)
git -C ../harxhar-paper-overleaf pull
python writeup/overleaf_sync/unflatten.py --main ../harxhar-paper-overleaf/main.tex --dry-run   # read REPORT.md first
python writeup/overleaf_sync/unflatten.py --main ../harxhar-paper-overleaf/main.tex
```

Then rebuild the paper, flatten again and push, so that Overleaf and the source agree.

What happens to each kind of edit:
* **Prose and comments in a section file.** Applied to that file. A replaced or deleted line is kept as
  `% <date> (overleaf): parked -- edited in Overleaf. Was:` with the old line commented below. Comment-only edits
  and new lines are written directly.
* **A number that came from a generated macro**, such as the 1,082 trading days. Not applied, and reported with the
  macro and the generator to change. If a sentence with numbers in it is rewritten and the numbers survive as typed,
  they are linked back to their macros automatically.
* **A generated table.** Not applied, and reported with its generator: change the generator, never the generated
  file.
* **A new literal number typed into prose.** Applied, with a warning, because every number in the prose should come
  from a generated macro.
* **A section file that changed in this repository since the flatten.** The edit goes through a three-way merge
  (`git merge-file`). A file whose merge conflicts is left untouched and reported.

`unflatten.py` exits 2 when there are conflicts. Non-conflicting edits are still applied unless `--strict` is
given.

## Limits

* Overleaf's GitHub sync is pulled and pushed from the Overleaf menu: it does not run by itself.
* Macro names that appear only inside comments are left as they are. If someone un-comments parked text in
  Overleaf, the macros in it are undefined there. `unflatten.py` still brings the text back, where the macros are
  defined.
* `\ser{raw}{symbol}` (the raw-name switch) stays as a macro in `main.tex`. The preamble's `\rawseriestrue` selects
  raw parquet names.
