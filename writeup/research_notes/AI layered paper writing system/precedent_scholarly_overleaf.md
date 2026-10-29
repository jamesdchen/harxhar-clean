# Precedents: scholarly living documents and AI + Overleaf/LaTeX collaboration for a layered, AI-maintained paper

Scope note: researched 2026-10-28. "Snippet-only" marks a claim seen only in a search-result summary, not on a fetched page. Several items overlap other notes in this folder. showyourwork's `\variable`, Manubot templating, MyST glue and nanopublications are in `results_traceability.md`. LaTeX layer toggles are in `toggleable_layers.md`. Manubot's PR/CI review is in `review_interfaces.md`. They are cross-referenced here, not repeated. "(repo)" marks facts taken from this repository's files, not the web.

---

## Q1. Living/executable papers (Living Papers, Idyll, Distill, Curvenote/MyST, Quarto Manuscripts, Jupyter Book, Stencila, eLife ERA, Authorea, Manubot): do any support layered views (outline vs full) or value propagation with checks?

### Takeaway
Value propagation is common: compute-at-render (Quarto, MyST, Living Papers) or define-once substitution (MyST, Manubot). Checks on those values are rare, and the only staleness check found is showyourwork's dependency graph (see `results_traceability.md`). None of the living-paper systems reviewed offers an outline-vs-full or beats-vs-prose layer. Their only "layers" are output-target layers: the same source renders to interactive HTML or static PDF, with blocks restricted to one target. Living Papers (UIST 2023) is the closest research precedent for one AST with several renderings. It explicitly defers collaborative authoring and review interfaces to future work.

### Cited Findings

**Living Papers (Heer, Conlen, Devireddy, Nguyen, Horowitz; UIST '23, DOI 10.1145/3586183.3606791)**
- The paper presents a toolkit for writing augmented articles that output as interactive web pages or as static PDFs. Its motivation is that print-oriented paper formats limit reading aids, accessibility and information extraction. — [IDL project page](https://idl.uw.edu/papers/living-papers); [paper](https://idl.uw.edu/living-papers-paper/living-papers)
- The AST has three top-level parts: "metadata, data, and the article tree". Node properties can be static, "Expression-typed properties contain a reactive JavaScript expression", or event handlers. Values are defined once and reused through a reactive runtime (Observable's): `bind=confidenceLevel` ties an input to a variable, and an inline `` `js confidenceLevel` `` displays it. Equations can be templated, e.g. `$$x^2 = ${v}^2 = ${v*v}$$`. — [paper](https://idl.uw.edu/living-papers-paper/living-papers)
- Outputs: `html`, `static` HTML, `latex`/PDF, `ast` JSON, `api` and `runtime` modules. "Output-specific blocks" (`html:only`, `latex:only`) give per-target content. The paper does not describe outline-vs-full views or toggled content layers. — [paper](https://idl.uw.edu/living-papers-paper/living-papers)
- For LaTeX, a `convert` transform "evaluates the reactive runtime in a headless web browser". Expression values are copied as-is, and interactive nodes become screenshots or PDFs. The authors warn that naive compilation "can produce an illegible article due to the lack of interaction". — [paper](https://idl.uw.edu/living-papers-paper/living-papers)
- Version control: it uses "plain text formats that operate well with revision control and diff'ing tools such as Git". Anchored annotations attach to AST nodes. Collaborative authoring, review interfaces and WYSIWYG editors that "might operate directly on Living Papers AST structures" are future work. — [paper](https://idl.uw.edu/living-papers-paper/living-papers)
- Stated limitations: no formal summative evaluation, a "large syntactic surface", no accessibility guarantee, and "adoption is difficult and hard to predict". The parser does not track input token positions, so source mapping is limited. — [paper](https://idl.uw.edu/living-papers-paper/living-papers)

**MyST / mystmd (Jupyter Book 2 ecosystem; Curvenote contributes)**
- mystmd can embed labeled content (paragraphs, figures, notebook outputs) across pages. The embed is resolved at render time, and content can also be pulled from *other* MyST projects via `xref` entries in `myst.yml`. — [mystmd embed guide](https://mystmd.org/guide/embed) (snippet-only)
- Substitutions in mystmd come from a contrib plugin, `myst-substitutions`, which replaces `{{ var }}` tokens. Values are defined in `myst.yml` (project level) or page frontmatter, and page values override project values. A `{substitution}` directive handles multi-line content. — [contrib.mystmd.org](https://contrib.mystmd.org/myst-substitutions/) (snippet-only)
- The Sphinx/myst-parser `substitutions:` frontmatter is a different implementation, and myst-parser substitutions cannot be used in URLs. — [myst-parser optional syntax](https://myst-parser.readthedocs.io/en/v0.13.5/_sources/using/syntax-optional.md.txt) (snippet-only)
- MyST-NB `glue` value propagation: see `results_traceability.md`.

**Quarto Manuscripts**
- An article can include computed output "using inline code or embedded from external notebooks". Code is hidden by default while "any output including tables and figures will appear". The web version links an "Article Notebook". — [Quarto manuscripts: JupyterLab](https://quarto.org/docs/manuscripts/authoring/jupyterlab.html); [VS Code](https://quarto.org:443/docs/manuscripts/authoring/vscode.html) (snippet-only)
- One source renders to several formats, including the LaTeX or Word formats journals require. — [Mine Çetinkaya-Rundel's conf talk deck](https://mine.quarto.pub/manuscripts-conf23/) (snippet-only)
- Quarto's freeze/cache staleness hazard: see `results_traceability.md`.

**eLife Executable Research Articles (ERA) and Stencila**
- eLife launched ERAs in 2020. The format was started with Substance and then developed by Stencila, building on Jupyter and R Markdown. — [eLife press release](https://elifesciences.org/for-the-press/eb096af1/elife-launches-executable-research-articles-for-publishing-computationally-reproducible-results); [eLife Labs](https://elifesciences.org/labs/dc5acbde/welcome-to-a-new-era-of-reproducible-publishing)
- A 2021 "next phase" announcement said six ERAs had been published, and the stack was open source so other publishers could adopt it. — [eLife Labs, next phase](https://elifesciences.org/labs/a04d2b80/announcing-the-next-phase-of-executable-research-articles) (snippet-only)
- Stencila v2 is a Rust rewrite. Documents can be stored as Automerge CRDT binaries that can be branched and merged; the Automerge round-trip is alpha and executable code types are beta. — [Stencila GitHub README](https://github.com/stencila/stencila/) (snippet-only)

**Manubot**
- `manubot process --template-variables-path` takes JSON/YAML/TOML values for Jinja2 templating (define-once values). PR-plus-CI review is covered in `results_traceability.md` and `review_interfaces.md`.

### Inferences
- The "one source, several views" idea is mature, but every system's views are *rendering targets* (web vs print). None has *abstraction levels* (beats, bullets, prose). The author's beats → bullets → prose layering is therefore not anticipated by the living-papers literature. Living Papers' `html:only`/`latex:only` blocks are the closest mechanism, and they match the toggle approach in `toggleable_layers.md`.
- Living Papers and Stencila v2 both treat the document as an AST/CRDT rather than lines of text. This is the main design lesson for a layered paper: layers stay in sync more easily when they are views of one structured source. Merging edits made to *rendered* text (what Overleaf collaborators do) is a separate hard problem that neither solves, and Living Papers lists it as future work.
- None of these systems checks a value against its claim in prose (e.g., "larger" vs the macro's sign). Value propagation is solved; value-to-claim consistency is not.
- eLife ERA appears to have stalled after 2021. No post-2021 status was found, which is consistent with Living Papers' adoption warning. Executable-paper formats have not reached econometrics journals, so LaTeX plus generated macros remains the practical substrate for a JFEC submission.

### Gaps
- Idyll, Distill, Jupyter Book 1 and Authorea were not researched this session (budget). From prior knowledge, which is unverified here, none offers outline/prose layers or value checks.
- Curvenote's own reuse features (reusable blocks across articles, versioned blocks) were not confirmed. Only its authorship of the official Quarto manuscript template example appeared — [Quarto manuscripts](https://quarto.org/docs/manuscripts/authoring/jupyterlab.html) (snippet-only).
- Whether mystmd now supports substitutions natively, rather than only through the contrib plugin, is unverified.
- Current (2024–2026) status of eLife ERAs: no source found either way.

---

## Q2. Overleaf and AI: AI Assist/Writefull, git bridge and GitHub sync with agents, published Claude Code/Cursor/MCP workflows, and how they handle collaborator edits, comments and tracked changes

### Takeaway
Overleaf's own AI (AI Assist, built on Writefull, launched June 2025) offers sentence-level rewrite and generate tools, not an agent. Agent integration comes from third parties, by two routes:
- **Git route:** many MCP servers and Claude Code skills (2025–2026) that pull → edit → push through Overleaf's per-project git remote.
- **Live-editor route:** tools that type into the Overleaf editor over its real-time channel (claudeleaf, overleaf-mcp-rt).

Overleaf documents that git/GitHub sync can lose or displace tracked changes and comments, and that renames destroy them. The robust published pattern is **one owner per file** (showyourwork, ChernXiangyu's skill): generated files go repo → Overleaf, and prose goes Overleaf → repo. The author's system is bidirectional on prose (flatten/unflatten plus a refuse-on-unpulled-edits guard). No published workflow found does this.

### Cited Findings

**Overleaf's own AI**
- AI Assist merges the former Error Assist and Writefull Premium: LaTeX error fixes and explanations, Writefull language suggestions, Rewrite (paraphrase, make concise, make scientific, split, join), Generate (title, abstract), table/equation generation and summarising. Free use has a daily cap. Users in managed or opted-out institutional accounts cannot access it. — [Overleaf docs: AI features](https://docs.overleaf.com/integrations-and-add-ons/ai-features); [Overleaf AI Assist](https://cn.overleaf.com/learn/how-to/AI_Assist) (snippet-only)
- Digital Science announced the tools for "20+ million Overleaf users" on 2025-06-24. — [Digital Science press release](https://www.digital-science.com/press-releases/digital-science-launches-new-cutting-edge-ai-writing-tools-for-20-million-overleaf-users/)
- Writefull's TeXGPT answers academic questions and supports LaTeX in Overleaf. — [Overleaf docs: Writefull](https://docs.overleaf.com/integrations-and-add-ons/ai-features/writefull) (snippet-only)
- No Overleaf-native agent, project-wide rule enforcement or git-aware AI feature was found as of 2026-10. — same search; flagged as an absence, not a confirmed negative.

**Overleaf git/GitHub mechanics relevant to agents (official docs)**
- GitHub sync: "pulls from GitHub to Overleaf can result in the loss or displacement of track changes and comments". Overleaf "do[es] not recommend mixing active work in GitHub and the use of track changes and/or comments". — [Overleaf docs: GitHub synchronization](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/github)
- On a sync conflict, "instead of merging into your main branch, we push a new branch to your repository" (named with date and time), which the user merges. Sync "does not happen automatically" and is a premium feature. — [same](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/github)
- Authorship: Overleaf history shows GitHub-side changes as "Anonymous (via GitHub)". Overleaf-side edits are bundled into one GitHub commit under the connecting account, so they do not map one-to-one to collaborators. — [same](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/github)
- Direct git integration: a file changed on the git side can lose metadata, which deletes its tracked changes and comments. Moving or renaming such files from git is discouraged. — [Overleaf: Git integration](https://overleaf.com/learn/how-to/Git_integration); [Advanced Git operations](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/git-integration/advanced-git-operations) (snippet-only)

**Published agent ↔ Overleaf workflows (GitHub, 2025–2026; star counts are at time of fetch)**
- **ChernXiangyu/overleaf-git-skill** (Claude Code skill): code-generated tables, figures and macros go pipeline → Overleaf and are "never edited in the Overleaf editor". Coauthor prose comes Overleaf → local backup and is "never pushed back". It lists Overleaf git limits: no branches (pushes go to `main`), "renames destroy comments and tracked changes", and no symlinks, LFS or execute bit. A push rejected while a collaborator is typing is handled with `pull --rebase` and retry. The repo has 2 commits and 0 stars. — [GitHub](https://github.com/ChernXiangyu/overleaf-git-skill)
- **gsoosk/claude-overleaf** (Claude skill, "turns Claude into a co-author"): pull at session start, then add/commit/push after *every* edit. On unseen Overleaf changes it shows the conflicting sections and asks the user which to keep. Optional GitHub Actions two-way sync: GitHub → Overleaf uses `-X ours` (GitHub wins), and an hourly Overleaf → GitHub pull uses `-X theirs` (Overleaf wins). It does not mention comments or tracked changes and sets no rules on authorial decisions. It is based on feamster/overleaf-sync; 2 stars, ©2026. — [GitHub](https://github.com/gsoosk/claude-overleaf)
- **ricardbcn/overleaf-git-claude-setup**: Claude Code sets up Overleaf git so projects become local folders while co-authors stay in the browser. It fetches before editing and pushes after. Open beta. — [GitHub](https://github.com/ricardbcn/overleaf-git-claude-setup) (snippet-only)
- **MCP servers over git**: Junfei-Z/overleaf-claude-mcp (diff-only reads, section extraction, pull-before-push), amcheste/overleaf-mcp (5 tools, "local, auditable, single-user by design", no branch/merge/diff tooling), mjyoo2/OverleafMCP (section-level edits pushed via git), tamirsida/overleaf_mcp (14 tools), and vibeTeX (git bridge plus compile; needs a paid Overleaf git token). — [Junfei-Z](https://github.com/Junfei-Z/overleaf-claude-mcp); [amcheste](https://github.com/amcheste/overleaf-mcp); [mjyoo2](https://github.com/mjyoo2/overleafmcp); [tamirsida](https://mcpservers.org/servers/tamirsida/overleaf_mcp); [vibeTeX](https://www.vibetex.dev/) (all snippet-only)
- **lonetis/claudeleaf** (SDK/CLI/MCP, MIT, 36 stars, ©2026): drives the Overleaf web editor through "Overleaf's live collaboration channel (ShareJS over Socket.IO)", so edits appear live "exactly as if a person were typing".
  - It recommends a **dedicated bot account** invited as an editor, so agent edits are attributed in Overleaf history.
  - Write tools accept `trackChanges` and submit suggestions for a human to accept or reject (requires premium track changes).
  - It can anchor and list comments, but cannot resolve or delete them because Overleaf does not expose those routes.
  - It warns that edits hit the live document, so revert goes through Overleaf history.
  - [GitHub](https://github.com/lonetis/claudeleaf)
- **DanielHou315/overleaf-mcp-rt**: for self-hosted Overleaf Community Edition. It makes real-time edits "via Overleaf's native OT pipeline" and is pitched as an "Alternative to git-bridge for AI coding agents". — [GitHub](https://github.com/DanielHou315/overleaf-mcp-rt) (snippet-only)
- **showyourwork! Overleaf integration** (experimental). Each file syncs in one direction only, to avoid merge conflicts:
  - `push:` lists generated outputs (repo is the source of truth); `pull:` lists `ms.tex` and the `.bib` (Overleaf is the source of truth for text). A file cannot be in both.
  - Pushes have "no conflict checks" and overwrite figures uploaded by hand.
  - If a pulled file has local changes, the build stops with "Refusing to overwrite with Overleaf version".
  - The token is kept in `$OVERLEAF_TOKEN` or as a GitHub secret.
  - [showyourwork docs: Overleaf](https://show-your.work/en/latest/overleaf/)
- **In-editor multi-agent research systems**:
  - PaperDebugger (Hou, Lin Huikai, Chen, Gong, He; NUS; arXiv:2512.02589, v1 2025-12-02, v2 2026-02-01): a Chrome extension with a Kubernetes orchestration layer and an MCP toolchain. It has Reviewer, Enhancer and Researcher agents, "localized edits" and "diff-based updates". The abstract gives no evaluation numbers ("Early aggregated analytics demonstrate active user engagement") and does not address collaborators, comments or tracked changes. — [arXiv abs](https://arxiv.org/abs/2512.02589)
  - Bibby AI (arXiv:2607.05435) calls PaperDebugger the closest system in spirit. — [arXiv PDF](https://arxiv.org/pdf/2607.05435) (snippet-only)
  - ToFu (arXiv:2607.11423) connects an agent harness to Overleaf through an open-source Overleaf MCP server. — [arXiv PDF](https://arxiv.org/pdf/2607.11423) (snippet-only)
- **Economist precedent**: Goldsmith-Pinkham linked a Claude-Code-driven repo to Overleaf so Claude updates figures and tables in the paper, with the rule "every number in a draft should come from the code". — see `economist_workflows_policies.md`.
- **This repo (repo)**, `writeup/overleaf_sync/overleaf_git.sh` header:
  - `push` flattens a clean checkout of HEAD to `main2.tex`, and "refuses if Overleaf's main2.tex differs from the last flatten in … map.json, because that means human edits not yet pulled back".
  - `pull` runs `unflatten.py` to map Overleaf edits back to section files.
  - `push-beats` pushes the review layers (`beats.tex`, `beats_top.tex`) "for reading and comments; edits go to the sources", and also refuses on divergence.

### Inferences
- **The author's design goes further than any published precedent.** Every published git workflow either (a) makes prose one-directional (showyourwork and ChernXiangyu: Overleaf owns `.tex`, the repo owns generated files) or (b) merges blindly with a fixed winner (gsoosk's `-X ours`/`-X theirs`, which can silently drop a collaborator's or the agent's edit). The author's flatten → Overleaf → unflatten round trip, guarded by a hash map that refuses to push over unpulled human edits, is the showyourwork "Refusing to overwrite" guard extended to bidirectional prose. No published tool was found that unflattens a single Overleaf file back into a multi-file, multi-layer source.
- **Comments and tracked changes are the weak point.** All git-route tools risk losing Overleaf comments when a file changes on the git side, per Overleaf's own docs. Because the author pushes a regenerated `main2.tex`, collaborator comments anchored in it will be at risk on each push after a regeneration. Mitigations used in the wild:
  - Keep comment-bearing files out of the regenerated set (showyourwork, ChernXiangyu).
  - Ask collaborators to resolve comments before a push.
  - Use the live-editor route, where an agent-side bot account can post suggestions as tracked changes (claudeleaf).
- **Attribution.** Git-route edits show as "Anonymous (via GitHub)" or under the token owner. claudeleaf's dedicated bot account is the only published way found to make agent edits distinguishable in Overleaf history. This matters for mixed human/AI provenance.
- **Locked author decisions.** None of the published Claude-Overleaf skills encodes author decisions or style rules. gsoosk explicitly sets none. Decision propagation is therefore not addressed by any public agent-Overleaf workflow.
- **Maintenance.** Most of these repos are small (0–36 stars) and dated ©2026, so they are early and unproven. Their conflict handling should be treated as design sketches.

### Gaps
- Whether Overleaf now (2026) exposes an official API, webhook or agent feature, or a way to export or import comments via git: not found.
- Comment-preservation behaviour when only *part* of a file changes via git (vs whole-file replacement): Overleaf docs say "can" lose or displace, without specifying when. Untested here.
- Maintenance status and security posture of the listed MCP servers: not assessed. Several entries are snippet-only.
- No published account was found of Cursor or Copilot agents with Overleaf beyond the generic MCP servers. No blog post with a long-run experience report (months, several coauthors) was found.

---

## Q3. Argumentative structure / claim graphs maintained during writing; research compendia and pipeline-consistency tools

### Takeaway
Claim-evidence graphs exist almost only on the **reading** side (PaperTrail, Claritas) or as **provenance ontologies** (Evidence Graphs, nanopublications). On the **writing** side the closest precedents are HCI prototypes:
- **VISAR** (UIST '23): an argument graph kept synchronized with a text outline, with drafts generated from outline variants.
- **Dang et al.** (UIST '22): a continuously regenerated summary layer beside each paragraph.

These partly contradict the earlier claim that "no published academic-writing system keeps persistent layers". Persistent, linked outline/summary layers do exist in research prototypes. But they target short argumentative essays, not full LaTeX papers, and none propagates locked author decisions.

### Cited Findings
- **VISAR** (Zhang, Gao, Dhaliwal, Li; UIST '23, DOI 10.1145/3586183.3606800; arXiv:2304.07810 v2 July 2023):
  - Writers brainstorm and revise "hierarchical goals" and organize "argument structures through synchronized text editing and visual programming".
  - It "synchronizes these two workspaces" (text outline and visual canvas) and flags counterarguments, supporting evidence and logical flaws.
  - Writers generate drafts from outline variants ("rapid draft prototyping").
  - User study: N=12, rated useful and usable.
  - The sync mechanism was not seen in detail.
  - [arXiv](https://arxiv.org/abs/2304.07810) (snippet-only)
- **Beyond Text Generation** (Dang, Benharrak, Lehmann, Buschek; UIST '22; arXiv:2208.09323): an editor with paragraph-level summaries in the margin, "updated while typing". Summary levels run from full paragraph to central sentences to keywords. Two studies (N=4, N=8, analytic essays) found the summaries gave an outside view that helped writers rework paragraph content and scope. — [project page](https://hciaiprojects.uni-bayreuth.de/projects/beyond-text-generation); [arXiv](https://arxiv.org/abs/2208.09323v1)
- **PaperTrail** (arXiv:2602.21045): a claim-evidence interface, grounded in argumentation theory, for LLM-based scholarly Q&A. Claim-evidence annotations come from an extraction engine; it is a reading tool. — [arXiv HTML](https://arxiv.org/html/2602.21045) (snippet-only)
- **Evidence Graphs** (bioRxiv 2021.03.29.437561) treat "computations and their provenance as defeasible arguments for provisional results". It is an ontology, not a writing tool. — [bioRxiv](https://www.biorxiv.org/content/10.1101/2021.03.29.437561.full.pdf) (snippet-only)
- **Claritas** (public beta) offers structured annotation and argument mapping across papers, aimed at reading and synthesis. — [evidencemap.app](https://www.evidencemap.app/) (snippet-only)
- **Pipeline-consistency tools**: showyourwork!, Quarto/knitr, Gentzkow-Shapiro, MyST glue, Manubot and nanopublications are in `results_traceability.md` Q2–Q3. targets/drake and DVC + LaTeX were not re-researched.
- **Citation correction for the folder**: `review_interfaces.md` cites arXiv 2506.20595 as "Script&Shift". The arXiv abstract page instead gives the title "AI in the Writing Process: How Purposeful AI Support Fosters Student Writing" (Siddiqui, Pea, Subramonyam; 2025-06-25). It is an RCT with N=90 undergraduates comparing a chat assistant, an integrated multi-subprocess tool and a control. The integrated tool gave greater writer agency. — [arXiv abs](https://arxiv.org/abs/2506.20595)

### Inferences
- **The earlier claim needs qualifying.** It holds for *deployed* systems and for full LaTeX research papers: no such system keeps a persistent beats/outline layer that regenerates prose and survives into the final paper. It does not hold for HCI research in general. VISAR keeps a persistent outline/argument graph linked to the draft (top-down, outline → draft). Dang et al. keep a persistent summary layer derived from the draft (bottom-up, draft → summary). The author's system does both directions: beats are authored top-down, and review layers are regenerated for reading on Overleaf.
- Dang et al.'s finding (summary layers help structural revision) is direct evidence for pushing `beats_top.tex`/`beats.tex` to Overleaf as reading layers. The caveat is small N and essay tasks.
- VISAR shows that typed relations (supports, counters) can be kept beside an outline. A beats file could carry claim IDs and support links. This would connect to the claim ledger proposed in `results_traceability.md`.
- No tool was found that *enforces* an argumentative structure or author decision across a paper (e.g., "we never call X a forecast"). This remains a gap the author's locked-decisions mechanism would fill without precedent.

### Gaps
- Script&Shift (a layered interface separating content development from rhetorical strategy, likely by the same Stanford group, CHI 2025) was not located under its own arXiv ID. Its existence and design are unverified this session.
- Argdown, Kialo-style authoring, and "research compendium" standards (rrtools, Marwick et al.) were not re-researched.
- No tool was found that keeps nanopublications or claim graphs updated *during* drafting, rather than extracting them after publication.

---

## Q4. Studies of collaborative writing with mixed human/AI editors on the same LaTeX project

### Takeaway
No empirical study was found of human collaborators and an AI agent co-editing the same LaTeX/Overleaf project. Nearby evidence:
- **ScholaWrite**: Overleaf keystroke logs of human-only writing.
- **Single-author trust/authorship studies**: authorship drift and the effect of disclosure.
- **PaperMentor**: argues that feedback beats rewriting for agency.
- **A large RCT of LLM feedback on manuscripts.**

### Cited Findings
- **ScholaWrite** (Univ. of Minnesota, Kang group; arXiv:2502.02904, several versions): a Chrome extension logged keystrokes inside Overleaf for 10 CS graduate students writing real conference papers, Nov 2023 – Feb 2024. About 62K text changes from 5 preprints, each labelled with one of 15 cognitive writing intentions (before/after text plus metadata). Human-only writing; author order differs across mirrors. — [arXiv](https://arxiv.org/abs/2502.02904); [HTML v4](https://arxiv.org/html/2502.02904v4) (snippet-only)
- **Authorship Drift** (ACM, DOI 10.1145/3772318.3790276; arXiv:2602.05819): in LLM-assisted writing, trust rose across turns while self-efficacy fell. Writers whose self-efficacy dropped requested more direct edits, which reduced perceived authorship. — [ACM DL](https://dl.acm.org/doi/10.1145/3772318.3790276); [arXiv](https://arxiv.org/pdf/2602.05819) (snippet-only)
- **"Accepted with Minor Revisions: Value of AI-Assisted Scientific Writing"** (ACM, DOI 10.1145/3742413.3789140): disclosing the text's source changed how authors edited it. AI-labelled text got more scrutiny although authors felt equally responsible for correctness. — [ACM DL](https://dl.acm.org/doi/10.1145/3742413.3789140) (snippet-only)
- **PaperMentor** (arXiv:2606.08857, June 2026): a multi-agent writing tutor on Overleaf. It argues that feedback-based assistance (commenting rather than rewriting) better preserves authorial agency, and notes Overleaf's Writefull-based assistant. Only the reference list and snippets were seen. — [arXiv PDF](https://arxiv.org/pdf/2606.08857) (snippet-only)
- **Human-AI Collaboration in Science at Scale** (arXiv:2605.24180): a randomized field experiment over 31,000+ manuscripts assigned standardized LLM-generated feedback. — [arXiv PDF](https://arxiv.org/pdf/2605.24180) (snippet-only)
- **Theory**: "Reassessing Collaborative Writing Theories and Frameworks in the Age of LLMs" (arXiv:2505.16254) argues that human-human concepts such as authorship and consensus building should not transfer directly to human-AI collaboration. — [arXiv HTML](https://arxiv.org/html/2505.16254v2) (snippet-only)

### Inferences
- The author's setup has no empirical precedent: a solo lead author, agents as the main editor, collaborators editing the rendered paper on Overleaf, and edits flowing back via git. Design choices should lean on general findings:
  - Disclosure changes scrutiny. Mark agent-originated text, e.g. via a bot account or tracked-change suggestions.
  - Direct edits erode authorship. Locked author decisions and chat-routed edits help.
  - Feedback over rewriting preserves agency. Treat Overleaf review layers as read-and-comment only, as the repo already does.
- ScholaWrite's Overleaf keystroke-capture method, a browser extension, is the only published way found to observe collaborator behaviour in Overleaf at fine grain. Git-side history bundles edits and loses per-collaborator attribution (see Q2).

### Gaps
- No study was found of merge conflicts, comment loss or attribution confusion in mixed human/AI Overleaf projects. Evidence on these failure modes is only the tools' own READMEs and Overleaf docs.
- Full texts of Authorship Drift, Accepted with Minor Revisions, PaperMentor and the 31K-manuscript RCT were not read. Effect sizes and settings are unverified.
- No finance or econometrics-specific study of AI-assisted co-authoring was found.
