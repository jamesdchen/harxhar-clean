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
