# Single-source documents with toggleable layers (draft/final, outline/prose) in LaTeX and alternatives

Scope note: the research ran on 2026-10-28. Package dates and versions are given where the source states them. Statements marked "(repo)" come from this repository's own files, not from the web. Anything I could not verify against a source is under Gaps, even when I believe it from prior knowledge.

---

## Q1. LaTeX mechanisms: which exist, how they work, which hide nested content cleanly, and which break with Overleaf or with \label/\ref

### Takeaway
Two families exist, and they differ in one way that matters here.
- **Token-skipping conditionals** (`\newif`/`\iffalse`, `\ifdraft`, `optional`'s `\opt`, `tagging`'s `\tagged`, macro-argument switches) work anywhere, including inside a single `\item` or a macro argument. They are fragile only when the skipped text has unbalanced `\if...`/`\fi` tokens.
- **Verbatim line-scooping environments** (`comment`, `version(s)`) are robust to ill-formed text. They require `\begin`/`\end` on lines of their own and break inside macro arguments.

Every mechanism that skips text also skips the `\label`s inside it, so references into hidden layers print "??". Only `\includeonly` (file granularity) and external-label injection (`xr`/`xr-hyper`, or the repo's generated `\newlabel` lines) keep numbers stable across layers.

### Cited Findings

**Overview and why not raw `\iffalse`**
- The TeX FAQ calls `\iffalse ... \fi` and `\gobble` "not really satisfactory" as a general commenting tool. TeX still scans the skipped text, so a `\verb+\iftrue+` inside an `\iffalse` block breaks the skip, and `\gobble` is "pretty inefficient" for large amounts of text. — [TeX FAQ: conditional compilation](https://texfaq.org/FAQ-conditional)

**`\include` / `\includeonly` (kernel)**
- Omitted `\include` files are skipped entirely, and numbering is unaffected. `\includeonly` lists the files needed. Related packages: `askinclude` (interactive selection), `stampinclude` (recompile only changed files; delete the aux files for a final build), and `excludeonly` (an `\excludeonly` list). — [TeX FAQ](https://texfaq.org/FAQ-conditional)

**`comment` package (v3.8, July/August 2016)**
- Purpose: "Selectively in/exclude pieces of text: the user can define new comment versions, and each is controlled separately." Usage is `\includecomment{name}` / `\excludecomment{name}`. — [comment manual (writeup.tex)](https://matojo.unizar.es/ctan/macros/latex/contrib/comment/writeup.tex)
- "The opening and closing commands should appear on a line of their own. No starting spaces, nothing after it." Since v3.1, trailing text on a `\begin{env}` line is discarded even when the environment is included. — [comment manual](https://matojo.unizar.es/ctan/macros/latex/contrib/comment/writeup.tex)
- Exclusion works by scooping every line in verbatim mode and throwing it away. An included comment is written to a file (`\CommentCutFile`, default `comment.cut`) and then `\input`. For an included comment, "the \begin and \end lines act as if they don't exist… they don't imply grouping, so assignments &c are not local." `\specialcomment{name}{before}{after}` runs code around each block, which can be used for colored editorial boxes. `\processcomment` is marked "BUG this does not work quite yet BUG". — [comment manual](https://matojo.unizar.es/ctan/macros/latex/contrib/comment/writeup.tex)
- The TeX FAQ calls comment's exclusion method robust: it "can cope with ill-formed bunches of text." — [TeX FAQ](https://texfaq.org/FAQ-conditional)
- Excluding list environments with comment's commands fails with "Lonely \item" errors unless the list's `\end...` macro is reset by hand. — [TopAnswers TeX Q715](https://topanswers.xyz/tex?q=715) (via search excerpt)

**`version` / `versions` packages**
- `version` is lighter than `comment` (`\includeversion`/`\excludeversion`) but "cannot deal with very large areas of text being included/excluded". `versions` adds `\markversion{name}`, which prints the included text with a visible mark around it. — [TeX FAQ](https://texfaq.org/FAQ-conditional)
- The `versions` documentation says its environments break inside macro arguments and provides `\processifversion` for conditional text inside arguments. — [versions-doc.pdf (CTAN)](https://ctan.math.washington.edu/tex-archive/macros/latex/contrib/versions/versions-doc.pdf) (via search excerpt)

**`optional` package**
- `\opt{flag}{text}`. The text "must be well-formed (nothing mismatched), and should not be too big"; for large bodies, use `\input`. Flags can be set in the preamble, on the command line, or interactively. — [TeX FAQ](https://texfaq.org/FAQ-conditional)

**`tagging` package (v1.1, 2019-09-09; first release 2011)**
- `\usetag{...}` and `\droptag{...}` set the active tags. `\tagged{list}{text}` and `\untagged{list}{text}` select text, `\iftagged{list}{y}{n}` branches, and there are `taggedblock`/`untaggedblock` environments. — [TeX FAQ](https://texfaq.org/FAQ-conditional); [tagging README (CTAN)](https://ctan.math.utah.edu/tex-archive/macros/latex/contrib/tagging/README)

**`xcomment` package**
- Excludes an environment's body except for the environments named in its argument, e.g. `\begin{xcomment}{figure,table}`. — [TeX FAQ](https://texfaq.org/FAQ-conditional)

**`extract` package**
- Produces a partial copy of a document (example: a "book of examples") during an ordinary run, with options `generate=`, `extract-env=` and `extract-cmd=`. — [TeX FAQ](https://texfaq.org/FAQ-conditional)

**`ifdraft` package (Heiko Oberdiek, v1.4, 2016-05-16)**
- `\ifdraft{T}{F}` switches on the class options `draft`/`final`, and `\ifoptiondraft`/`\ifoptionfinal` test whether an option was passed explicitly. If neither option is given, `\ifdraft` falls back to final; if both are given, the last one wins. Load it after `\documentclass`. — [CTAN catalogue: ifdraft](https://kebo.pens.ac.id/CTAN/help/Catalogue/entries/ifdraft.html); [ifdraft.pdf](https://ctan.csail.mit.edu/macros/latex2e/contrib/oberdiek/ifdraft.pdf)

**Editorial-note packages**
- `todonotes` (v1.1.7, 2024). The package option `disable` turns `\todo`, `\listoftodos` and `\missingfigure` into "macros with no effect". `disable` can also be given to a single `\todo`. `\todo[inline]{...}` puts the note in the text body, and `noinline` overrides a global `inline`. — [todonotes.pdf (CTAN mirror)](https://ctan.csail.mit.edu/macros/latex2e/contrib/todonotes/todonotes.pdf) (via search excerpt)
- `fixme`. Notes left over after switching from draft to final mode are logged and removed from the body; "Critical notes will abort compilation with an informative message." Another catalogue listing says that in final mode "any remaining fixme notes will produce an error". The two descriptions differ, and the behavior probably depends on the note level, so check the manual. — [TeX Catalogue: fixme](https://ftp.rediris.es/mirror/ctan/help/Catalogue/entries/fixme.html); [fixme NEWS](https://tug.ctan.org/macros/latex/contrib/fixme/NEWS)
- `changes` (v4.x). The `final` option "disables markup of changes, only the correct text will be shown", and the list of changes is disabled too. The package reuses `final` from `\documentclass`, and a local `draft` overrides it. Commands: `\added`, `\deleted`, and `\replaced{new}{old}`, each with optional author and comment arguments. The `final` option only hides markup; to remove the commands from the source you must delete them by hand or use a script (pyMergeChanges.py). Release 4.0.3 fixed a compile error between `\cite` and `final`. — [changes manual (English)](https://ctan.ceremade.dauphine.fr/macros/latex/contrib/changes/changes.english.pdf); [CTAN announcement](https://www.ctan.org/ctan-ann/id/mailman.352.1539330956.4481.ctan-ann@ctan.org)
- `outlining` (Dimitri Block, v0.1, 2017-10-23, LPPL 1.3) is "a tool for creating outlines for scientific documents". The search excerpt of its manual describes `\topic` and `\major` commands, a `disable` option, a list-outline command, and outline entries "built on the todo mechanism". — [CTAN: outlining](https://ctan.org/pkg/outlining); [outlining.pdf](https://ctan.ceremade.dauphine.fr/macros/latex/contrib/outlining/outlining.pdf) (direct fetch failed with a TLS error, so the command details come from the search excerpt only)

**docstrip guards**
- docstrip can produce many files from a single source. Guards (`%<*name>` … `%</name>`) accept full boolean expressions, not just lists, and are used to produce, for example, a "debug" version of a package. — [CTAN catalogue: docstrip](https://ftp.rediris.es/mirror/ctan/help/Catalogue/entries/docstrip.html); [docstrip.dtx](https://tug.ctan.org/macros/latex/base/docstrip.dtx); [dtxgallery](https://ctan.math.washington.edu/tex-archive/documentation/dtxgallery/dtxgallery.pdf)

**Flatten and clean tools (what survives)**
- `latexpand` (v1.7.2 on CTAN) flattens `\input`/`\include` and **removes comments by default**. `--keep-comments` preserves them, `--empty-comments` keeps the `%` markers, `--expand-bbl FILE` inlines the bibliography, `--expand-usepackage` inlines `.sty` files, and `--keep-includes` leaves `\input`/`\include` unexpanded. Known bug: it also strips `%` lines inside verbatim environments. Option names differ between versions. — [latexpand README (CTAN)](https://tug.ctan.org/support/latexpand/README); [CTAN latexpand](https://ctan.org/tex-archive/support/latexpand)
- `arxiv_latex_cleaner` (Google Research; Python ≥3.9; "not an officially supported Google product").
  - Strips all comments, `\begin{comment}...\end{comment}`, `\iffalse...\fi` and `\if0...\fi`, and simplifies conditionals by keeping true branches and deleting false ones. It assumes every command starting with `\if` is a primitive conditional unless listed in `--if_exceptions`.
  - `--commands_to_delete todo`, `--commands_only_to_delete` (removes the command but keeps its argument text), `--environments_to_delete note`, and regex replacements in `cleaner_config.yaml`.
  - Removes `.tex` files that are not reachable from the root. — [arxiv-latex-cleaner README](https://github.com/google-research/arxiv-latex-cleaner)

**Beamer article mode (one source, two granularities)**
- `beamerarticle` lets one source file produce slides (`beamer` class) and an article (`article` class plus `\usepackage{beamerarticle}`). Overlay and mode specifications such as `\mode<article>` and `\mode<presentation>` work on commands like `\item`. With the class option `ignorenonframetext`, text outside frames is dropped from the slides but kept in the article. `beamerswitch` wraps the beamer, handout, trans and article modes into one command invocation. Caveats: the `columns` environment has no article equivalent, and it fails with `tufte-handout`. — [Beamer (Wikipedia)](https://en.wikipedia.org/wiki/Beamer_(LaTeX)); [beamerswitch example (CTAN)](https://tug.ctan.org/macros/latex/contrib/beamer-contrib/beamerswitch/beamerswitch-example.pdf) (via search excerpts)

**References into hidden content**
- "??" appears when `\ref` cannot find the corresponding label. — [Overleaf: There were undefined references](https://www.overleaf.com/learn/latex/Errors%3AThere_were_undefined_references.)
- A resolved label is just the command `\r@<label>`, so a document can test whether a label exists. — [TeX FAQ: labundef](https://texfaq.org/FAQ-labundef)

**The repo's current state**
- The paper already uses `\newif\ifauthornotes` and `\newif\ifrawseries` in `writeup/paper_rehaul.tex`, `paper_rehaul_beats.tex` and `paper_rehaul_beats_top.tex`.
- `fi_bullets_standalone.tex` loads `xr-hyper`.
- The generated top-beats layer (`make_paper_beats.py`) injects `\newlabel{...}{{E.5}{59}{}{}{}}` lines copied from the full paper's `.aux`. These are "labels this layer cites but does not print". — (repo) `C:\Users\james\CC Allowed\harxhar-0dte-professor\writeup\paper_rehaul_beats_top.tex` lines 1–60

### Inferences

**Capability matrix.** This is my synthesis from the findings above plus standard TeX semantics.

| Mechanism | Granularity | Inside `\item` / macro arg? | Hides a bullet's children cleanly? | Labels in hidden part | Overleaf-safe? | Main failure mode |
|---|---|---|---|---|---|---|
| `\newif` (`\ifauthornotes…\fi`) | any token range | yes | yes: wrap the nested `itemize` | lost ("??") | yes, flag set in source | unbalanced `\if`/`\fi` in skipped text; `\fi` forgotten across files |
| `ifdraft` `\ifdraft{T}{F}` | macro arg | yes | yes, but children become a macro argument (no verbatim) | lost | yes: class option in source | draft class option also changes graphics/overfull boxes |
| `optional` `\opt{flag}{…}` | macro arg | yes | small blocks only | lost | yes | "well-formed, not too big" |
| `tagging` `\tagged{t}{…}` / `taggedblock` | arg or block | yes (cmd form) | yes | lost | yes | the env form has line-scoop limits; 2019, little used |
| `comment` / `versions` env | whole lines | **no** (breaks in args) | yes, if `\begin`/`\end` are on their own lines around the child list | lost | yes (writes `comment.cut`) | own-line rule; "Lonely \item" if a list env itself is excluded; no grouping |
| `xcomment{figure,table}` | env | no | inverse: drops prose, keeps floats | lost | yes | coarse |
| `extract` | env/cmd | n/a | can **emit** a beats-only file from tagged commands | n/a | yes, runs at compile time | under-documented |
| `\includeonly` | file | n/a | no | **kept** (numbers from last full run) | yes | file granularity; `\include` forces a page break (gap) |
| docstrip guards | line | n/a | yes | n/a (generated files) | output must be pre-generated and committed | an extra build step, like the repo's Python layers |
| todonotes `disable` / fixme `final` / changes `final` | note | yes | n/a | n/a | yes | `changes final` hides markup but leaves commands in the source |
| beamerarticle `\mode<…>` | block / overlay | yes | yes, by design | per mode | yes | ties you to beamer semantics |

**Nested bullets.** Beat → children → prose maps onto `\newif` (or `\iftoggle`) more naturally than onto `comment`. The children are a nested `itemize`, and a `\newif` wrapper inside an `\item` needs no own-line discipline. It can also sit inside a `\beat{label}{claim}{children}` macro. The `comment` package's line-scooping is what makes it robust to ill-formed text, but for AI-edited files the own-line rule is a common silent-failure mode: an agent appends text after `\end{prose}` and the text is discarded.

**Labels.** Any skip-based layer loses the `\label`s it hides. The repo's generated `\newlabel` injection is a correct and general fix. `xr-hyper`'s `\externaldocument{paper_rehaul}` would do the same automatically from the full build's `.aux`, at the cost of an Overleaf dependency rule (see the Q5 gaps).

**Pending numbers.** For "pending" markers, a fixme-style hard error in final mode, or a custom `\pending` macro that errors when `\ifauthornotes` is false, makes "no missing numbers ship" a build-time invariant rather than a review task.

**Parked text.** Dated `%` comments are removed automatically by both `latexpand` (by default) and `arxiv_latex_cleaner`. So a final flatten removes parked text with no extra work, but `\ifauthornotes` blocks survive `latexpand`; only the cleaner's conditional simplification touches them (see Gaps).

### Gaps
- **etoolbox toggles** (`\newtoggle`, `\toggletrue`/`\togglefalse`, `\iftoggle{name}{T}{F}`): syntax not verified against the CTAN manual in this session. Search attempts returned irrelevant results.
- **`arara`**: not researched. Whether Overleaf honours `% arara:` directives is unverified. Overleaf compiles via latexmk (Q5), so the assumption is that it does not.
- **`\include` forcing `\clearpage`**: unverified in this session (standard LaTeX behavior). This makes `\includeonly` unsuitable for paragraph-level layers.
- **`arxiv_latex_cleaner` and user-defined `\newif` flags**: it is unclear which branch the cleaner keeps for a user-defined `\ifauthornotes`. The README says that all `\if*` commands are "assumed primitive", which implies it may not know a flag's runtime value. Test it before relying on it.
- **`outlining`**: the full command set (`\topic`, `\major`, list-of-outline) comes from a search excerpt only; the PDF fetch failed.
- **Confirmed failure catalog**: no 2018–2026 TeX.SE answer was retrieved for `comment` used inside `\item` or for `\newif` across `\input` boundaries. These failure modes are inferred from the package docs.

---

## Q2. Non-LaTeX single-source systems: how they express conditional layers and "outline level" versus "prose level"

### Takeaway
- **Quarto is the most complete flag system.** Named profiles (`_quarto-{name}.yml`, `--profile`) combine with `.content-visible`/`.content-hidden` blocks and inline spans keyed on `when-profile`, `when-format` or `when-meta`.
- **Pandoc** gives the same power through a few lines of Lua filter.
- **Sphinx/MyST** has `only` with boolean tag expressions, but it explicitly cannot control sections or labels.
- **Typst** has `sys.inputs` (CLI strings) plus `#if`, but a missing label is a hard compile error.
- **Org-mode and Scrivener** are the only systems where "outline level" is native:
  - Org: headings are beats, the body is prose, and export tags or `H:` select the depth.
  - Scrivener: each document has a synopsis (beat) and text (prose), and the compile layout picks which one is output.

### Cited Findings

**Quarto conditional content**
- Mechanism: `.content-visible` / `.content-hidden` on divs (`::: {.content-visible when-format="html"}`), on non-executable code blocks, and on inline spans (`[text]{.content-visible when-format="html"}`). Conditions: `when-format`/`unless-format` (combinable), and `when-meta`/`unless-meta` on boolean metadata with dotted keys. Format aliases: `latex`/`pdf`/`beamer` match each other; `html` matches `html*`, `epub*`, `revealjs`, etc. Hidden content **still executes** its code cells. — [Quarto: Conditional Content](https://quarto.org/docs/authoring/conditional.html)
- Profiles: each `_quarto-{name}.yml` is merged over `_quarto.yml`. They are activated with `QUARTO_PROFILE` or `--profile a,b` (the flag replaces the environment variable). When several profiles set the same scalar, the first-listed one wins; arrays concatenate and objects deep-merge.
  - `profile: group:` declares mutually exclusive profiles, and the first in a group is the default.
  - `profile: default:` sets a profile for editors where environment variables are awkward.
  - Profile files cannot use `metadata-files`.
  - **An unknown profile name is silently ignored**, so a typo goes unnoticed.
  - Content is switched with `when-profile` / `unless-profile` on `.content-visible`/`.content-hidden`. — [Quarto: Project Profiles](https://quarto.org/docs/projects/profiles.html)
- A maintainer recommends profiles over format conditionals when hiding content per audience. — [quarto-dev discussion #10487](https://github.kxpsado.workers.dev/orgs/quarto-dev/discussions/10487) (via search excerpt)

**Pandoc and Lua filters**
- The standard pattern is a `Div(el)` handler that checks a class and returns an empty list to delete the block. `exclude-format` removes Divs from specified output formats based on a div attribute. I found no published filter that keys deletion on a `draft` metadata flag; that combination has to be written by hand: read `meta.draft` in `Pandoc(doc)`, then filter the Divs. — [exclude-format filter](https://www.github.com/robertbachmann/exclude-format); [Oniro CONTRIBUTING.lua](https://gitlab.eclipse.org/eclipse/oniro-core/docs/-/blame/f5416d5d5e0667c8713d189f582e55865eb0146d/contributing/CONTRIBUTING.lua) (via search excerpts)

**Sphinx / MyST**
- `.. only:: html and draft` includes content only when the boolean tag expression is true. Parentheses are allowed, and undefined tags are false. Tags are set with `--tag` or in `conf.py`, and builders set format tags automatically. Warning from the docs: the directive "could not control sections, labels and so on". — [Sphinx directives: only](https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html)
- MyST-Parser can use "mostly all" docutils and Sphinx directives. — [MyST-Parser FAQ](https://myst-parser.readthedocs.io/en/v0.13.3/using/faq.html)

**Jupyter Book 2 / mystmd**
- I found no documented conditional-content or draft flag. The closest items:
  - cell-tag show/hide, returning in the JB2 beta ([2i2c: JB2 alpha](https://2i2c.org/blog/jupyter-book-2/));
  - format-specific raw Typst or LaTeX blocks ([TU Delft starter kit](https://jboss.tudelft.nl/starterkit/introduction/)).
- A Jupyter Book discussion asks how to keep numbers in the prose synced with a JSON file that the analysis pipeline exports. This is the same problem as this paper's macro file. — [executablebooks discussion #732](https://github.com/orgs/executablebooks/discussions/732)

**Manubot (rootstock)**
- The manuscript is a set of Markdown files in `content/`, "ordered according to their two-digit prefix". `content/metadata.yaml` passes metadata to Pandoc.
- Outputs: HTML (GitHub Pages), PDF via WeasyPrint, DOCX.
- Filters: `pandoc-manubot-cite`, `pandoc-fignos`, `pandoc-tablenos`, a spellcheck Lua filter (enabled by `SPELLCHECK=true`).
- An `ai-revision` workflow runs only when triggered manually. — [Manubot rootstock USAGE.md](https://github.com/manubot/rootstock/blob/main/USAGE.md)

**Typst**
- `sys.inputs` is "a dictionary, which makes external inputs available". `--input key=value` becomes `sys.inputs.key`, and "The value is always of type string"; parse richer data with functions like `json`. — [Typst docs: sys](https://typst.app/docs/reference/foundations/sys/)
- Community pattern: `--input solution=true` versus `false` builds task-only or solution PDFs from one file. `sys.inputs.at("key", default: ...)` gives a fallback. The web app has no clear way to set `sys.inputs`. — [Typst forum: conditional compile](https://forum.typst.app/t/does-typst-support-conditional-compile/3491); [Typst forum: web-app parameters](https://forum.typst.app/t/what-is-the-best-way-to-pass-parameters-to-documents-in-the-web-app-like-sys-inputs-in-the-cli/7581) (via search excerpts)
- Referencing a missing label is a hard error ("label <…> does not exist in the document"). A `show ref` workaround breaks bibliography citations. `hide()` leaves the space where the content was. — [Typst forum: suppress missing-reference errors](https://forum.typst.app/t/is-there-a-way-to-suppress-errors-in-case-of-a-missing-reference/3255); [Typst forum: hide spacing](https://forum.typst.app/t/is-eliminating-extra-space-after-using-hide-possible/5859) (via search excerpts)

**Org-mode export**
- Tag selection: `#+SELECT_TAGS:` (default "export") exports only tagged subtrees; `#+EXCLUDE_TAGS:` (default "noexport") removes tagged subtrees even if they are also tagged export. Code blocks in excluded subtrees still run.
- Depth: `H:` sets the deepest level exported as a headline, and deeper levels become list items. `arch:headline` exports only the headlines of archived trees.
- Other filters: `d:` (drawers), `tasks:` (TODO items), `todo:`, `prop:`, `toc:`, `num:`.
- Per-subtree `EXPORT_*` properties override these options. — [Org manual: Export Settings](https://orgmode.org/manual/Export-Settings.html)

**Scrivener**
- Compile "Section Layouts" have checkboxes for Title, Synopsis, Notes and Text. Duplicate a layout, untick Text, tick Synopsis, and assign it via "Assign Section Layouts" to compile synopses only. A built-in "Title and Synopsis" layout exists. Pitfall: if Text stays ticked in the layout in use, the body still exports. — [L&L forum: compile only synopsis](https://forum.literatureandlatte.com/t/can-i-compile-only-synopsis-to-a-pdf/131749); [L&L forum: synopses and notes](https://forum.literatureandlatte.com/t/how-to-only-compile-synopses-and-notes/151905); [L&L blog: section layouts](https://www.literatureandlatte.com/blog/using-section-layouts-to-compile-your-scrivener-project) (forum sources)

### Inferences
- **How each system expresses "outline level"**:
  - *Org*: heading depth (`H:`) plus tags.
  - *Scrivener*: a per-document synopsis field.
  - *beamerarticle / Org Beamer*: a separate mode.
  - *Quarto, Pandoc, Sphinx, Typst*: no native notion. Wrap outline and prose in differently-classed divs (`::: {.beat}` / `::: {.prose}`) and switch with `when-profile`, a Lua filter, or an `only` tag.
- **Quarto's profile model is the cleanest design to copy into LaTeX.** Its parts map as follows:
  - named profiles stored as files → small `layer-*.tex` flag files;
  - "first-listed wins" → set the flags before `\input{body}`;
  - groups with a default → one mutually exclusive layer selector, with final as the default.
  - Its silent-ignore of unknown profiles is a warning: a LaTeX layer selector should `\PackageError` on an unknown layer name.
- **Typst's and Sphinx's failure modes** (missing label = error; `only` cannot remove sections or labels) are the same `\label`/`\ref` problem as in LaTeX, with different severity. A layered paper needs a label strategy in any system.
- **Switching away from LaTeX** would lose Overleaf. Typst's web app has no input flags, and Quarto, Pandoc and Org have no Overleaf round-trip. For this user the non-LaTeX systems are design references, not migration targets.

### Gaps
- **Curvenote**: not researched in this session.
- **Manubot template variables** (`{{ }}` and a template-variables path): not shown on the USAGE page I read, so I can't confirm that Manubot substitutes numbers from JSON.
- **Typst `state`/`context`** for layer switching: not researched beyond `sys.inputs`.
- **mystmd**: no authoritative source found for conditional or draft content. Check mystmd.org directly.
- **Quarto nesting**: whether `.content-visible` divs nest cleanly inside list items is not stated in the Quarto docs.

---

## Q3. Does anyone keep an outline (claims/beats) and the final prose of the same paragraph side by side in one source, so a build can show either?

### Takeaway
Yes, but mostly outside academic-paper LaTeX:
- the Beamer slide/article dual mode (bullets and the prose that explains them, interleaved in one file, selected by `\mode<…>` or `ignorenonframetext`);
- Org-mode's Beamer dual-format recipe;
- Scrivener's synopsis-versus-text per document;
- the small CTAN `outlining` package (2017).

I found no public academic-paper template that pairs a beat and its prose per paragraph behind a flag. The repo's "beat label + nested bullets + generated layers" design is ahead of what is published.

### Cited Findings
- **Org Beamer dual format** (H. James Harkins, worg).
  - Structure: two header files (`slidehead.org` with Beamer; `printhead2.org` with `article` plus `beamerarticle`) and one shared content source.
  - Each third-level section holds a frame with bullets, followed by a `:B_ignoreheading:` subsection of narrative text. The slides show only the frame; the article shows both.
  - Format-specific hacks: `\maketitle` is wrapped in a frame only for Beamer, and a flag (`hjh-exporting-slides`) switches glossary execution. — [Worg: Beamer dual format](https://orgmode.org/worg/exporters/beamer/beamer-dual-format.html)
- **beamerarticle**: text outside frames is "omitted from slideshows" with `ignorenonframetext` but appears in the article. — [Beamer (Wikipedia)](https://en.wikipedia.org/wiki/Beamer_(LaTeX)) (via search excerpt)
- **Scrivener**: each binder document carries a synopsis (outline/beat) and text (prose). Compile layouts choose Title+Synopsis (outline build) or Text (prose build), and users routinely compile "synopsis-only" outlines. — [L&L forum: tutorial on Title-and-Synopsis compile format](https://forum.literatureandlatte.com/t/tutorial-on-scrivener-compile-formats-and-creating-one-with-only-title-and-synopsis/153517); [L&L forum: synopses to generate an outline](https://forum.literatureandlatte.com/t/can-synopses-be-used-to-generate-an-outline/30546)
- **CTAN `outlining`** (2017): "Every scientific document requires outlining before it is written". It has `\topic` and `\major` macros embedded in the document, a `disable` option to hide them, and a command that lists the outline. — [outlining.pdf (search excerpt)](https://ctan.ceremade.dauphine.fr/macros/latex/contrib/outlining/outlining.pdf); [CTAN: outlining](https://ctan.org/pkg/outlining)
- **CTAN `outline` / `outlines`**: these typeset nested outline lists (with per-level styles `\outlinei`…`\outlineiv` in `outlines`). They are not a prose/outline switch. — [outlines.pdf](https://ctan.csail.mit.edu/macros/latex2e/contrib/outlines/outlines.pdf) (via search excerpt)
- **TeXmacs users (2020)** discussed a "shown"/"hidden" macro pair selected by a draft flag. That is the generic pattern, not a template. — [TeXmacs list 2020-07](https://lists.texmacs.org/wws/arc/texmacs-users/2020-07/msg00045.html)
- **Editor-side outline views**: AUCTeX folding and outline-minor-mode collapse a file to its headings, and a TeXShop "outliner mode" was requested in 2008. These are views, not builds. — [macosx-tex 2008 thread](https://email.esm.psu.edu/pipermail/macosx-tex/2008-March/034576.html)
- **Agent-era LaTeX workflows (2025–2026)** are community projects, none with layered views:
  - Overleaf MCP servers that work over the git bridge (e.g. overleaf-forge, OverleafMCP);
  - a `claude-code-my-workflow` template with review agents and a git-hook quality gate (last reviewed 2026-06-20).

  — [overleaf-mcp](https://github.com/ghostiee-11/overleaf-mcp); [x-cmd: claude-code-my-workflow](https://x-cmd.com/install/claude-code-my-workflow) (search excerpts; not verified in depth)

### Inferences
- **The Beamer/Org pattern is the closest precedent.** Beats are bullets inside a "frame" (the outline view), and a following prose block is dropped in outline view. In LaTeX this generalizes to a paragraph-level environment pair, sketched below with no package needed. `\ifbeats` and `\ifprose` are `\newif`s set by the layer selector. Labels belong in the prose, or in a label-only macro outside both, so that every layer defines them.

  ```latex
  \beat{label}{claim}   % always shown, or shown in outline layers
  \ifbeats \begin{itemize}...support bullets...\end{itemize}\fi
  \ifprose ...final paragraph...\fi
  ```

- **The Scrivener model** (synopsis and text as two fields of one unit) maps exactly onto a `\para{claim}{prose}` macro: an outline build typesets `#1`, a final build typesets `#2`. The cost is that the prose becomes a macro argument (no verbatim, and catcodes frozen). That is harmless in a math-and-macro paper.
- **Fewer review passes.** Keeping beat and prose adjacent in the source means an AI edit to the prose sits in the same diff hunk as the claim it serves, so a reviewer checks claim-prose consistency locally. This is the main argument for co-location over the current approach of generating separate layer files.

### Gaps
- No blog post, repo or template (2018–2026) was found that implements per-paragraph beat-and-prose toggling for a journal paper. A targeted GitHub code search for `\ifprose` or `\ifoutline` was not done in this session.
- The `outlining` package's real-world use and maintenance status since 2017 is unknown.

---

## Q4. Semantic line breaks / one sentence per line, and other conventions that make AI edits and git diffs reviewable

### Takeaway
SemBr is a written spec: a break MUST follow each sentence, SHOULD follow independent clauses, and MUST NOT change the rendered output. LaTeX is a verified host language. Combined with `git diff --word-diff` and a PDF-level diff (latexdiff via `git-latexdiff`), it turns an AI edit into a one-line diff per changed sentence. Tooling exists to enforce it (snapper as a git clean/smudge filter; SemBr; textlint). The main friction point is Overleaf's editor, which soft-wraps and lets collaborators reflow text.

### Cited Findings

**The SemBr spec** ([sembr.org](https://sembr.org/); by Mattt, CC BY 4.0, undated)
- "A semantic line break MUST NOT alter the final rendered output of the document."
- A break "MUST occur after a sentence", SHOULD follow independent clauses (comma, semicolon, colon, em dash), MAY follow dependent clauses, is RECOMMENDED before lists, and MUST NOT split hyphenated words.
- An 80-character maximum is RECOMMENDED.
- Rationale for collaborators: "an edit to one sentence changes one line instead of reflowing a paragraph". LaTeX and Markdown are listed as verified; it suggests `git diff --word-diff`.

**LaTeX specifics**
- A plain newline adds a space in the output; a newline after `%` adds none. — [SemBr tool (mgaffney/sembr)](https://github.com/mgaffney/sembr)

**Enforcement tools**
- `snapper` (Rust; LaTeX, Org, Markdown, plaintext) has an `sdiff` sentence-level diff command and a git clean/smudge filter that auto-formats on commit. Versions 0.2–0.9 are on docs.rs/PyPI. — [snapper-fmt (PyPI)](https://pypi.org/project/snapper-fmt/); [TurtleTech blog on snapper](https://turtletech-ehf.pages.dev/posts/semantic-line-breaks-snapper/)
- `SemBr` (mgaffney) uses Transformer models (Python). — [mgaffney/sembr](https://github.com/mgaffney/sembr)
- `textlint-rule-one-sentence-per-line` is a fixable Markdown rule. — [textlint rule](https://github-redirect.dependabot.com/IQTLabs/textlint-rule-one-sentence-per-line)

**Caveat**
- An LWN commenter notes that editors reflow line breaks (Emacs autofill, Overleaf display wrapping), which can create spurious merge conflicts, so collaborators must agree on the convention. — [LWN article 1044470 (comments)](https://lwn.net/Articles/1044470)

**PDF-level review diffs**
- `git latexdiff OLD NEW` (or `HEAD --` for the working tree) produces a marked-up PDF. It flattens with latexpand by default (`--latexdiff-flatten` uses latexdiff's own flatten); `--type=CHANGEBAR` gives change bars instead of colored markup.
- Caveat: `\input`/`\include` are taken from each revision, but figures come from the new revision.
- `latexdiff-vc --flatten -r rev1 -r rev2 master.tex` is the direct route. `latexdiff-git` (logangrado) runs per file and keeps `\input`/`\subfiles` intact. — [git-latexdiff man page](https://man.archlinux.org/man/git-latexdiff.1.en); [latexdiff-git](https://github.com/logangrado/latexdiff-git) (via search excerpts)

**In-source change markup**
- The `changes` package (`\added` / `\deleted` / `\replaced` with author IDs, plus `final`) is an alternative to diffing that keeps the markup in the source. — [changes manual](https://ctan.ceremade.dauphine.fr/macros/latex/contrib/changes/changes.english.pdf)

### Inferences

**Conventions for agent-edited LaTeX**, from most to least load-bearing:
1. One sentence per line (SemBr). A `%`-terminated break where a space must not be added, e.g. after a macro that should abut punctuation.
2. One `\item` per line-block, with the beat label on the `\item` line, so `git diff` hunks align with beats.
3. Numbers only through macros. Diffs then never show numeric churn, only `\macroname` changes, so a prose diff is reviewable without checking numbers.
4. Draft scaffolding behind flags rather than deleted, so the diff of a prose rewrite is not polluted by note removal.
5. A per-commit `git latexdiff HEAD^` PDF as the human review artifact, instead of reading source.

**Parked text.** Dated `%` comments are a diff anti-pattern under SemBr. They grow the file and show up as large added hunks. A `\begin{parked}{2026-10-28}…\end{parked}` environment, excluded by `comment` with its `\begin`/`\end` on their own lines, is greppable and dropped in every build. Alternatively, rely on git history and delete parked text outright; latexpand and arxiv_latex_cleaner strip `%` comments on flatten anyway.

**Overleaf.** Because Overleaf soft-wraps in display only, SemBr source is not reflowed unless a human edits in Overleaf and joins lines. A pre-commit check (snapper or a lint) catches that on the next pull.

### Gaps
- No controlled study or measurement was found of review time with SemBr versus wrapped LaTeX, or for AI-generated edits specifically.
- Whether snapper's LaTeX mode handles macro-heavy sentences (e.g. `\num{...}` with periods inside arguments) was not verified.

---

## Q5. Overleaf constraints: limits, git bridge, track changes and comments, what survives a flatten

### Takeaway
- **Git bridge**: premium-only; a single branch (no branches or tags); symlinks become files; no LFS or submodules.
- **Track changes and comments**: they live outside git and can be lost or displaced by git pushes.
- **Size**: 2 MB per editable text file and 7 MB of editable material per project. Free compiles time out at 10 s; premium at 240 s.
- **Builds**: Overleaf takes no command-line flags. Layer selection must be expressed in the source: a flag file, per-layer wrapper `.tex` files compiled by selecting another file, or a `latexmkrc`.
- **Flatten**: `latexpand` and `arxiv_latex_cleaner` drop `%` comments. Only the cleaner removes `\iffalse` blocks and named note commands or environments.

### Cited Findings

**Plan limits** (current docs page; no dates given) — [Overleaf: plan limits](https://docs.overleaf.com/getting-started/free-and-premium-plans/plan-limits)
- Compile timeout: free 10 s, premium 240 s.
- 2000 files per project.
- 7 MB of editable material per project; 2 MB per individual editable text file ("to ensure a file remains editable, keep it at 2 MB or less").
- 50 MB per upload.
- No enforced total size; under 100 MB recommended with GitHub sync or Git, and 500 MB recommended overall.
- Git integration and track changes are listed as premium features.

**Git integration** — [Overleaf: Git integration](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/git-integration)
- Clones use `main`.
- "does not support branching" and "does not support tagging".
- Symlinks are converted to regular files.
- Execute permissions are not preserved.
- No Git LFS; projects cannot contain submodules.
- A local rename becomes a delete+create, losing metadata, and renamed folders leave empty folders behind.
- "pushes from Git to Overleaf can result in the loss or displacement of track changes and comments"; the docs recommend not mixing Git with track changes or comments.
- Large commits can time out; `http.postBuffer` may need to be raised.

**GitHub sync**
- Keep commits under about 100 files and the project under 100 MB. Users report failures with any single file over 50 MB. — [Overleaf: GitHub sync](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/github); [SO mirror: GitHub sync file-size limit](https://kiwix.lema.org/stackoverflow.com_en_all_2023-11/questions/69262545/overleaf-github-synchronization-limit-on-file-size)

**Main document and multiple outputs**
- A project has one main file, but "you can compile multiple documents inside one Overleaf project without changing the project's main file setting". Any file with `\documentclass` can be selected for compiling.
- Pitfall: if a subfolder file shares a name with a root file, the root file wins, so give layer wrappers unique names.
- Overleaf recommends one project per output document in most cases. — [Overleaf: the main document](https://docs.overleaf.com/getting-started/recompiling-your-project/the-main-document); [Overleaf: Multi-file LaTeX projects](https://www.overleaf.com/learn/latex/Multi-file_LaTeX_projects) (via search excerpts)

**latexmk and latexmkrc**
- Overleaf compiles with latexmk (`-cd`, from the project root), and a project-level `latexmkrc` can customize the build.
- Setting `$output_dir` or `$preview_continuous_mode` may break the preview.
- One source says a project `latexmkrc` stops Overleaf from adding its default one. The current docs say the system-wide file is read *before* user files. The two statements conflict; treat the second (current docs) as authoritative.
- The documented examples are environment variables (e.g. `$ENV{'TZ'}`) and `TEXINPUTS`. — [Overleaf: the latexmkrc file](https://docs.overleaf.com/managing-projects-and-files/the-latexmkrc-file); [Overleaf: How does Overleaf compile my project?](https://www.overleaf.com/help/216-how-does-overleaf-compile-my-project)
- latexmk's `$pdflatex` command string supports `%O` (options) and `%S` (source file), e.g. `$pdflatex = 'pdflatex %O -interaction=nonstopmode --shell-escape %S';`. — [latexmk manual](https://www.sys.kth.se/docs/texlive/texmf-dist/doc/support/latexmk/latexmk.pdf); [gitver latexmkrc example](https://tug.ctan.org/macros/latex/contrib/gitver/latexmkrc)

**Flatten and clean**
- See Q1: latexpand strips `%` comments unless `--keep-comments`. arxiv_latex_cleaner strips comments and `comment` environments, simplifies `\iffalse`/`\if0`, deletes named commands and environments, and drops unreferenced `.tex` files. — [latexpand README](https://tug.ctan.org/support/latexpand/README); [arxiv-latex-cleaner](https://github.com/google-research/arxiv-latex-cleaner)

**The repo's current sync**
- The repo already syncs through the git bridge with a script (`writeup/overleaf_sync/overleaf_git.sh push --check`) and a `map.json` commit. — (repo) the user's global CLAUDE.md, project section

### Inferences

**Layer selection on Overleaf.** The robust pattern is N thin wrapper files, all compilable by selecting them in Overleaf:
- `paper_final.tex`, `paper_beats_top.tex`, `paper_beats_full.tex`, `paper_with_appendix.tex`;
- each sets flags (`\authornotesfalse`, `\beatstrue`, …) and then does `\input{paper_body}`.

This needs no latexmkrc tricks and no command-line arguments, and it survives the git bridge (plain files, no symlinks). It is the source-level equivalent of Quarto profiles. The repo already has separate wrapper files, but they are generated copies; making them flag-only wrappers removes the generation step for the layers that a toggle can express.

**Size and time limits.**
- Python-generated macro files and stubbed-table layers must each stay under 2 MB to remain editable online, and the total editable text must stay under 7 MB. This matters if generated layers are committed as full copies rather than flag wrappers.
- The 10 s free / 240 s premium compile timeout argues against doing heavy processing in `latexmkrc` on Overleaf (e.g. docstrip or Python-like generation at compile time).

**Comments and track changes.** Because Overleaf comments and track changes do not round-trip through git and can be destroyed by pushes, all reviewable annotations in an agent-driven git workflow belong in the source: `\ifauthornotes` boxes, todonotes, or `changes` markup. Overleaf's own review tools should not be used for them.

**Final build.** "Flags off → compile → `latexpand` → `arxiv_latex_cleaner --commands_to_delete pending --environments_to_delete authornote`" gives a submission source with no scaffolding. The cleaner's handling of user-defined `\newif` flags needs a test (Q1 Gaps).

### Gaps
- **`xr`/`xr-hyper` on Overleaf**: the dependency rule (`add_cus_dep` in `latexmkrc` so the external `.aux` is built first) was not verified. The Overleaf blog post with latexmkrc examples redirected to a login page.
- **Macro injection via `$pdflatex`**: whether Overleaf honours a `latexmkrc` that sets `$pdflatex = 'pdflatex %O "\def\layer{final}\input{%S}"'` is not confirmed by any source found.
- **Track-changes visibility**: whether Overleaf exposes comment or track-change metadata to the git bridge in any form is not stated in the docs. Only the loss warning is documented.
- **Current Overleaf free/premium split**: which premium tiers include git, and any changes to compile timeouts after the page snapshot, are not dated on the page.
