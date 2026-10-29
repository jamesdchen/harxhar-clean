# Precedent: AI tools that work top-down through persistent layers with human approval at the top

Scope: spec-driven AI coding tools (GitHub Spec Kit, AWS Kiro, Tessl, OpenSpec, BMAD, AGENTS.md-style rule files), AI long-form fiction tools (Sudowrite, Novelcrafter, NovelAI), and the few non-fiction or report tools with a plan-approval gate. The goal is lessons for a layered, AI-written JFEC paper: top beats lead down to final LaTeX text; author decisions are locked and propagated; the author reads the PDF and sends edits in chat; collaborators edit on Overleaf.

Research date: 2026-10-28. Tool docs change quickly; page-update dates are noted where visible. Items marked **[UNVERIFIED]** come from search-result summaries, third-party writeups or my prior knowledge, not from a primary page fetched in this session.

---

## Q1. Spec-driven development: how do the tools structure layers, keep upper layers binding, propagate changes, and decide what humans review?

### Takeaway
All the major spec-driven tools use the same top-down chain: constitution or steering, then requirements or spec, then design or plan, then tasks, then code. Each layer is a persistent Markdown file, and a human checkpoint sits between phases. Upper layers bind lower ones only by being re-read in the agent's context: they are interpreted, not enforced, and agents both ignore and over-apply them. Change handling is the weakest part. Most tools are "spec-first": the spec is written for one change and then left behind. Only Tessl aims at "spec-as-source" (humans edit only the spec, and code is regenerated from it). Only OpenSpec has an explicit change-delta model for evolving the persistent spec.

### Cited Findings

**Taxonomy (the most useful framing for the paper system)**
- Birgitta Böckeler (Thoughtworks, 15 Oct 2025) defines three levels:
  - **Spec-first**: a spec is written before AI coding and used for the current task.
  - **Spec-anchored**: the spec is kept after the task and used to evolve and maintain the feature.
  - **Spec-as-source**: the spec is the primary artifact over time, and humans edit only the spec, not the code.
  - She notes that "all SDD approaches and definitions I've found are spec-first," and few are spec-anchored or spec-as-source. — [Böckeler, martinfowler.com](https://martinfowler.com/articles/exploring-gen-ai/sdd-3-tools.html)
- GitHub's framing: "maintaining software means evolving specifications." Tessl's framing: "specs — not code — are the primary artifact" (both quoted by Böckeler). — [Böckeler](https://martinfowler.com/articles/exploring-gen-ai/sdd-3-tools.html)

**GitHub Spec Kit (launched 2 Sep 2025)**
- Launch post by Den Delimarsky (2 Sep 2025). The phases are:
  - `/specify`: a high-level "what/why", from which the agent produces a detailed spec with user journeys and success criteria.
  - `/plan`: the human supplies the stack, architecture and constraints, and the agent produces a technical plan. The human can ask for several variants.
  - `/tasks`: small, reviewable tasks, each implementable and testable in isolation.
  - Implement. — [GitHub blog](https://github.blog/ai-and-ml/generative-ai/spec-driven-development-with-ai-get-started-with-a-new-open-source-toolkit/)
- Human gates: each phase has explicit checkpoints, and "you don't move to the next phase until the current one is fully validated." During implementation the human reviews focused changes rather than large code dumps. — [GitHub blog](https://github.blog/ai-and-ml/generative-ai/spec-driven-development-with-ai-get-started-with-a-new-open-source-toolkit/)
- Change model in the launch post: to change direction, "update the spec, regenerate the plan, and let the agent handle the rest." Specs are described as "living, executable artifacts." The core claim is separating the stable "what" from the flexible "how." The post gives no quantitative evidence. — [GitHub blog](https://github.blog/ai-and-ml/generative-ai/spec-driven-development-with-ai-get-started-with-a-new-open-source-toolkit/)
- The human's role, in GitHub's words: "Crucially, your role isn't just to steer. It's to verify." (quoted by Böckeler) — [Böckeler](https://martinfowler.com/articles/exploring-gen-ai/sdd-3-tools.html)
- Constitution: holds "immutable" principles applied to every change. Each step uses templates, scripts and checklists. One spec spans many files (8 in Böckeler's example). A branch is created per spec, which suggests the spec lives for one change request rather than for the feature. — [Böckeler](https://martinfowler.com/articles/exploring-gen-ai/sdd-3-tools.html)
- Repo as of Oct 2026 **[partly UNVERIFIED: from a fetch summary]**:
  - Commands are now namespaced `/speckit-*`. Besides constitution, specify, plan, tasks and implement, there is a `/speckit-converge` loop: repeat implement and converge until it reports "Converged".
  - Separate bug-fix flows (assess, fix, test) and idea-assessment flows (intake, research, define, shape, decide).
  - Guidance: resolve unknowns "by editing the existing Markdown artifacts rather than rerunning whole stages."
  - The page links an "evolving existing specs" guide and shows ~144k stars. — [github/spec-kit](https://github.com/github/spec-kit)
- **[UNVERIFIED, prior knowledge]** 2025 versions of Spec Kit included `/clarify` (structured questions before planning), `/analyze` (cross-artifact consistency and coverage check across spec, plan and tasks) and `/checklist` (described as "unit tests for English"). The Oct 2026 README fetch did not confirm these names.

**AWS Kiro (IDE; docs pages updated Aug–Oct 2026)**
- Three spec files:
  - `requirements.md` (or `bugfix.md` for bugs): user stories plus acceptance criteria.
  - `design.md`: architecture, sequence diagrams, data flow, error handling and testing strategy.
  - `tasks.md`: discrete, trackable tasks with live status. — [Kiro Specs docs](https://kiro.dev/docs/specs/)
- Requirements use EARS notation: "WHEN [condition/event] THE SYSTEM SHALL [expected behavior]". Kiro says this makes requirements "clear, testable, traceable, and complete." — [Kiro Feature Specs](https://kiro.dev/docs/specs/feature-specs/)
- There are two orderings:
  - **Requirements-First**: requirements, then design, then tasks.
  - **Design-First**: design, then requirements derived from it, then tasks. — [Kiro Feature Specs](https://kiro.dev/docs/specs/feature-specs/)
- Approval gates are implicit. A "Quick Spec" mode runs "all three phases automatically without approval gates between them," which implies standard specs do have gates. Each document has a **Review** button for Markdown Review comments. — [Kiro Specs docs](https://kiro.dev/docs/specs/); [Kiro Feature Specs](https://kiro.dev/docs/specs/feature-specs/)
- An optional "Analyze Requirements" step before design checks requirements for logical inconsistencies, ambiguities, conflicting constraints and gaps. — [Kiro Feature Specs](https://kiro.dev/docs/specs/feature-specs/)
- Tasks trace back to requirement numbers. — [Böckeler](https://martinfowler.com/articles/exploring-gen-ai/sdd-3-tools.html)
- "Run all tasks" builds a dependency graph and executes independent tasks in parallel "waves." — [Kiro Specs docs](https://kiro.dev/docs/specs/)
- Steering (a persistent project memory below the spec level; page updated 21 Oct 2026):
  - Default files are `product.md`, `tech.md` and `structure.md`, in workspace or global scope. Workspace overrides global on conflict.
  - Inclusion modes set in front matter: `always` (default), `fileMatch` (glob-triggered), `manual` (`#name`), and `auto` (matched by description).
  - Live file references `#[[file:path:12-28]]`. "Unresolved references leave a visible marker rather than being silently dropped."
  - `AGENTS.md` is always included.
  - Maintenance advice: review steering at sprint planning and architecture changes, and "treat steering changes like code changes that require review." — [Kiro Steering](https://kiro.dev/docs/steering/)
- Hooks: event-triggered agent prompts or shell commands (file save, create or delete, manual, and others). Kiro's own blog shows a hook that refreshes the README on save and updates API docs when endpoints change. There are format differences across Kiro versions (`.kiro.hook` UI files vs JSON `PostFileSave`). — [Kiro blog: README hooks](https://kiro.dev/blog/how-i-stopped-worrying-about-readme-files/); [Kiro hook types](https://kiro.dev/docs/hooks/types)
- Change propagation: the fetched Kiro pages do not document what happens to design or tasks when requirements are edited. A vendor competitor claims Kiro "allows developers to manually request spec updates after code changes, but does not automatically reconcile specs during execution" **[vendor claim, Augment Code]**. — [Augment Code: living vs static specs](https://www.augmentcode.com/guides/living-specs-vs-static-specs)
- **[UNVERIFIED, prior knowledge]** Kiro launched as a public preview in July 2025.

**Tessl (the only spec-as-source attempt)**
- How it works:
  - A CLI that doubles as an MCP server.
  - Generated code carries `// GENERATED FROM SPEC - DO NOT EDIT`.
  - One spec currently maps to one code file. `@generate` and `@test` tags drive generation.
  - An API section pins the exposed interfaces.
  - Spec and code sync in both directions, and `tessl document --code` reverse-engineers a spec from existing code. — [Böckeler](https://martinfowler.com/articles/exploring-gen-ai/sdd-3-tools.html)
- Non-determinism: regenerating from the same spec gave different code. Making the spec more specific improved repeatability. — [Böckeler](https://martinfowler.com/articles/exploring-gen-ai/sdd-3-tools.html)
- Product history **[single critical source plus search summary]**:
  - The Spec Registry beta launched Sept 2025.
  - In Jan 2026 Tessl reframed itself around an agent-skills registry.
  - A critical review calls spec-as-source "aspirational": the regeneration engine was closed-beta, JavaScript-only and non-deterministic. — [CodeMySpec Tessl review](https://codemyspec.com/blog/tessl-review)

**OpenSpec (explicit change-delta model)**
- Each change gets a folder with `proposal`, `design`, `tasks` and **delta specs**: ADDED / MODIFIED / REMOVED sections, one per affected capability.
- `archive` merges the deltas into the main specs and moves the change to `openspec/changes/archive/YYYY-MM-DD-<name>/`.
- `sync` merges without archiving.
- **[from search summaries of official docs; page not fetched directly]** — [OpenSpec: the workflow](https://openspec.dev/docs/the-workflow); [OpenSpec team workflow](https://openspec.dev/docs/team-workflow)

**BMAD method (multi-agent "agile" chain)**
- Planning agents produce a PRD and an architecture document.
- These are "sharded" (`*shard-doc`) into per-epic files.
- A Scrum-Master agent writes self-contained **story files**. A Dev agent implements one story at a time, and QA reviews it.
- Config tracks `prdSharded` and `architectureShardedLocation`. — [BMAD KB v4.43.1 (CDN mirror)](https://cdn.jsdelivr.net/npm/@ptnglobal/bmad-method@4.43.1/bmad-core/data/bmad-kb.md); [CodeMySpec BMAD explainer](https://codemyspec.com/blog/bmad-method-explained)
- Large PRD and architecture files can run to tens of thousands of tokens, which strains small models. That is the motivation for sharding. — [CodeMySpec BMAD explainer](https://codemyspec.com/blog/bmad-method-explained)
- **[mirrors and third-party sources only; official BMAD-METHOD repo not fetched]**

**Rule files: AGENTS.md, CLAUDE.md, Cursor rules**
- AGENTS.md is a plain-Markdown "README for agents," now stewarded by the Agentic AI Foundation under the Linux Foundation. It is used by "over 60k open-source projects."
- Precedence: the nearest AGENTS.md to the edited file wins, and "explicit user chat prompts override everything." — [agents.md](https://agents.md/)
- Kiro treats AGENTS.md as always-included steering. — [Kiro Steering](https://kiro.dev/docs/steering/)

### Inferences
- The layer chain maps cleanly onto the paper system:

  | Coding layer | Paper layer |
  |---|---|
  | Constitution / steering | Paper-wide conventions: notation, JFEC style, claims we never make |
  | Requirements | Top beats or locked author decisions |
  | Design | Section / paragraph outline |
  | Tasks | Paragraph-level beats |
  | Code | LaTeX prose |

- On Böckeler's axis, the author's system with locked, propagated decisions is at least **spec-anchored**. Collaborators editing prose directly on Overleaf rules out pure spec-as-source, so the system needs Tessl-style two-way sync (prose-to-beats reverse extraction) rather than one-way regeneration.
- "Binding" in every tool is soft: it means re-reading the upper layer in context. The paper system should not assume locked decisions are obeyed just because they are in context. It needs a separate check.
- Kiro's EARS form ("WHEN ... THE SYSTEM SHALL ...") is the best example of writing an upper-layer decision so that a lower layer can be checked against it. A paper analog: "WHEN Section 4 reports the OOS Sharpe, THE TEXT SHALL state it net of costs and cite Table 3."
- OpenSpec's dated delta-and-archive model is the closest analog to "author edits in chat become locked decisions with an audit trail."

### Gaps
- No primary doc found (in this session) describing what Kiro or Spec Kit do automatically to `design.md`/`tasks.md` or code when `requirements.md`/`spec.md` is edited after implementation. The "evolving existing specs" Spec Kit guide was not fetched.
- Cursor rules (`.cursor/rules/*.mdc` with glob or always/agent-requested modes) and CLAUDE.md hierarchy were not re-verified in this session.
- Current (Oct 2026) status of Tessl's spec-as-source Framework is unconfirmed.

---

## Q2. AI fiction / long-form tools: how do they keep facts and decisions consistent across layers, and do outline edits propagate to drafted prose?

### Takeaway
Fiction tools use a persistent "bible" or "codex" of facts plus a layer chain: braindump, then synopsis, then outline, then scenes (beats), then prose. Each layer is generated from the layers above, and the human edits or approves at each step. Consistency comes from **context injection**: the relevant entries are pasted into the prompt, usually selected by keyword or name mention. None of the tools documents automatic propagation of an outline edit into already-drafted prose. Users regenerate whole chapters, and asking to regenerate only what changed is an open user request.

### Cited Findings

**Sudowrite Story Bible / Story Engine (docs updated 13 Jan 2026)**
- Field dependency chain:
  - Braindump feeds Synopsis.
  - Genre affects Synopsis, Outline, Scenes and Draft.
  - Style affects Beat and Draft generation.
  - Synopsis feeds Characters, Worldbuilding, Outline and Scenes.
  - Outline feeds Scenes.
  - Scenes feed Draft (chapter prose), which also draws on Style, Genre, Characters and Worldbuilding. — [Sudowrite: What is Story Bible](https://docs.sudowrite.com/using-sudowrite/1ow1qkGqof9rtcyGnrWUBS/what-is-story-bible/jmWepHcQdJetNrE991fjJC)
- Human control: any field can be filled manually. **Generate** proposes content, and **Rewrite** applies the user's instructions. The docs describe the process as "improv": you review what the AI produces and respond to it. — [Sudowrite Story Bible](https://docs.sudowrite.com/using-sudowrite/1ow1qkGqof9rtcyGnrWUBS/what-is-story-bible/jmWepHcQdJetNrE991fjJC)
- Consistency limits:
  - Scene and Draft generation "only consider Characters and Worldbuilding elements that are explicitly mentioned."
  - If the Synopsis is empty, dependent fields fall back to the Braindump. — [Sudowrite Story Bible](https://docs.sudowrite.com/using-sudowrite/1ow1qkGqof9rtcyGnrWUBS/what-is-story-bible/jmWepHcQdJetNrE991fjJC)
- Propagation: the docs say nothing about upper-layer edits updating existing prose.
- Feb 2025 changelog: Sudowrite is moving away from Beats in the Chapter Generator in favour of Scenes. Scenes are generated from the linked Outline's chapter summaries and support Extra Instructions. **[search summary of changelog and glossary]** — [Sudowrite changelog, 23 Feb 2025](https://feedback.sudowrite.com/changelog/notebook-setting-the-scene-for-better-prose); [Sudowrite glossary](https://docs.sudowrite.com/getting-started/dQph1snuwbfMWG9wRjsNug/glossary/1Symu5y4wtu65nQVYHjhxa)
- User complaints **[search summaries of user feedback posts; no confirmation the features were built]**:
  - Prose generation regenerates the **whole chapter**, forcing full regeneration when only some beats change.
  - Users ask to rewrite one chapter's beats and regenerate just that chapter. — [Sudowrite feedback 84994](https://feedback.sudowrite.com/feedback/84994); [Sudowrite feedback 89421](https://feedback.sudowrite.com/feedback/89421); [Sudowrite feedback 115025](https://feedback.sudowrite.com/feedback/115025)
- "Chapter Continuity" tells the AI which document precedes which, to improve consistency. A Sudowrite blog post's claim that a change in chapter three "ripples forward" through chapter thirty is **marketing, not verified**. — [Sudowrite glossary](https://docs.sudowrite.com/getting-started/dQph1snuwbfMWG9wRjsNug/glossary/1Symu5y4wtu65nQVYHjhxa); [Sudowrite blog on beats](https://sudowrite.com/blog/your-outline-sucks-heres-how-to-use-sudowrite-beats-to-fix-it/)

**Novelcrafter Codex (help pages updated 17 Mar 2025 and 2 Dec 2025)**
- Codex selection for a prompt, in order:
  1. Entries mentioned in the beat or chat message and in the text before the beat.
  2. Entries manually referenced for the scene.
  3. The POV entry.
  4. All global ("Always Include") entries.
  5. Child entries related to the above. The relationship itself is not sent.
  6. Entries with no description are removed. — [Novelcrafter: Codex context in prompting](https://www.novelcrafter.com/help/faq/ai-and-prompting/codex-context-in-prompting)
- Per-entry tracking modes: "Always Include", "Include when detected", "Don't include when detected", "Never include". Ordering in the prompt: globals first, then by type, then by name. — [Novelcrafter: Codex context in prompting](https://www.novelcrafter.com/help/faq/ai-and-prompting/codex-context-in-prompting)
- **Progressions/Additions** are addenda to a codex entry that are **attached to a scene**. The AI "sees" them only when working on that scene or later; earlier scenes get the pre-change fact.
  - Each can be an addition or a **replacement**.
  - They are meant for major lasting changes (a marriage, a new job, a ransacked location), not plot points, which belong in scene instructions. — [Novelcrafter: Progressions/Additions](https://www.novelcrafter.com/docs/codex/additions/)
- Neither page addresses retroactive consistency: what happens to already-drafted earlier prose when a codex fact is edited.

**NovelAI Lorebook**
- Mechanics:
  - Entries activate when their **keys** appear within a search range of the story (max 10,000 characters).
  - Insertion Order decides which entries reserve tokens and enter context first.
  - Budgets can be a fraction of max context.
  - Cascading activation lets keys match inside other lorebook entries.
  - "&" keys require all terms.
  - A Context Viewer shows which entries fired and with how many tokens. — [NovelAI docs: Lorebook](https://docs.novelai.net/en/text/lorebook); [NovelAI Advanced Settings](https://docs.novelai.net/en/text/editor/advancedsettings)

### Inferences
- Fiction tools solve "keep facts consistent," not "keep decisions binding." Their unit is the entity: a character or place. A paper's unit is the **claim**: a number, a hypothesis test result or a framing choice. The codex pattern carries over directly as a "claims and numbers codex": each locked number or wording with its source (CSV, table) is injected whenever the section touches it.
- Retrieval by mention is a known failure source. Novelcrafter and Sudowrite both drop facts that are relevant but not named, and NovelAI only scans a window. For a paper, locked decisions that apply everywhere (notation, sample period, the headline result) should be "always include," not keyword-triggered.
- Novelcrafter's scene-scoped progressions map onto position-dependent facts in a paper: a term or estimator is defined in §3 and must not be used undefined in §2, or a robustness result is introduced in §6 and must not be previewed with different numbers in the introduction.
- The universal gap is propagating an outline edit into existing prose. Fiction tools leave it to manual whole-chapter regeneration. That is acceptable in fiction but destructive in a co-authored paper where collaborators have hand-edited prose. Copy the layering, but not the regeneration granularity.
- NovelAI's Context Viewer suggests a cheap feature: show the author exactly which locked decisions were in context when a paragraph was generated.

### Gaps
- Squibler, Lex and Type.ai were not researched in this session: no verified claims about their layer structure or propagation.
- No primary Sudowrite doc confirms whether a Story Engine "regenerate chapter" preserves user edits.
- Novelcrafter's alias matching and the look-back window for "prior text" are undocumented in the fetched page.

---

## Q3. Non-fiction or report tools with a layered approval flow

### Takeaway
Layered approval with persistent intermediate layers is rare outside coding and fiction. The common non-fiction pattern is a **single plan-approval gate**: the AI drafts a research plan or outline, the human edits or approves it, and the AI writes the full report. Examples are Gemini Deep Research and Stanford STORM's outline-first pipeline. After the gate there is no persistent binding layer and no change propagation.

### Cited Findings
- Gemini Deep Research shows a plan card with "Edit plan" and "Start research." Browsing begins only after approval. Users can change scope, sub-questions, date range, geography, inclusion and exclusion criteria, and source priorities. Google's early description: a "multi-step research plan" that users "can either edit or approve" before the agent starts. **[search summaries of third-party guides and Google coverage; UI labels change over time]** — [Google Workspace blog: Deep Research](https://workspace.google.com/blog/ai-and-machine-learning/meet-deep-research-your-new-ai-research-assistant); [NewsBytes on Deep Research launch](https://www.newsbytesapp.com/news/science/google-unveils-ai-tool-deep-research-for-gemini-advanced-users/story); [New School KB: using Deep Research](https://services.newschool.edu/TDClient/32/Portal/KB/ArticleDet?ID=1365)
- Stanford STORM (arXiv 2402.14207, Feb 2024; code open-source, refactored April 2024):
  - Pipeline: a pre-writing stage (LLM-driven question asking plus retrieval, ending in an **outline**), then a writing stage that expands the outline section by section into a cited article.
  - The authors define outline-quality metrics. **[search summary]** says they report that pre-writing quality correlates with final article quality.
  - Output is not publication-ready and needs significant edits. Experienced Wikipedia editors found it helpful in the pre-writing stage. — [STORM paper, arXiv](https://arxiv.org/abs/2402.14207v2); [STORM project page](https://storm-project.stanford.edu/research/storm/)

### Inferences
- The single gate (approve plan, then generate) is the weakest form of what the author wants, because nothing upstream stays binding after generation. The paper system's distinctive features are persistent locked layers, propagation of later edits, and round-tripping collaborator edits. I found no non-fiction tool that does all three.
- STORM supports investing human attention at the outline layer, since outline quality predicts output quality. It also shows that even a good outline yields text needing heavy editing.

### Gaps
- Notion AI, Google NotebookLM, Microsoft Copilot Pages and enterprise report generators were **not researched** in this session. No verified claims about any layered approval flow in them.
- AI academic-writing assistants (Overleaf AI features, Paperpal, Jenni, "AI Scientist"-style paper pipelines) were not covered here and may be covered by other researchers.

---

## Q4. Evidence and user reports: does top-level approval reduce downstream review, and where do these tools fail?

### Takeaway
There is **no controlled evidence** that approving upper layers reduces downstream review. The only quantified practitioner comparison found shows the opposite: spec review cost far more time than the code review it was supposed to save, and bugs still appeared. The recurring failure modes are:
- verbose upper layers that are expensive to review;
- agents ignoring or over-applying upper-layer rules;
- non-deterministic regeneration;
- bugs that cannot be expressed as a spec change;
- silent drift between upper layers and the artifact.

### Cited Findings

**Review load (quantified)**
- Colin Eberhardt (Scott Logic CTO, 26 Nov 2025) rebuilt a hobby-app feature with Spec Kit (Copilot plus Claude Sonnet 4.5) and compared it with his usual iterative prompting.

  | | Spec Kit | Iterative prompting |
  |---|---|---|
  | Agent time | ~57 min | — |
  | Review time | ~5.5 h (3.5 h feature 1; ~2 h feature 2) | — |
  | Total time | ~6.5 h | ~32 min |
  | Markdown generated | 2,577 lines for feature 1 (the plan step alone ~2,067 lines, incl. a 444-line contract and a 395-line data model); 2,262 lines for feature 2 | — |
  | Code produced | ~689 + ~300 lines | ~1,000 lines |
  | Implementation bugs | Yes | None |

  — [Scott Logic](https://blog.scottlogic.com/2025/11/26/putting-spec-kit-through-its-paces-radical-idea-or-reinvented-waterfall.html)
- He estimates iterative prompting was about **10x faster**. — [Scott Logic](https://blog.scottlogic.com/2025/11/26/putting-spec-kit-through-its-paces-radical-idea-or-reinvented-waterfall.html)
- Bugs still occurred under Spec Kit, for example an unpopulated variable. He "found it hard to express this as a spec change," and Copilot agreed it was not a spec issue. — [Scott Logic](https://blog.scottlogic.com/2025/11/26/putting-spec-kit-through-its-paces-radical-idea-or-reinvented-waterfall.html)
- Verdict: Spec Kit "drags you right back into the past" (waterfall), and "the fastest path is still iterative prompting and review, not industrialised specification pipelines." He still calls SDD "a radical idea" worth discussing. — [Scott Logic](https://blog.scottlogic.com/2025/11/26/putting-spec-kit-through-its-paces-radical-idea-or-reinvented-waterfall.html)
- Böckeler: "I'd rather review code than all these markdown files." Kiro turned a small bug into 4 user stories with 16 acceptance criteria ("a sledgehammer to crack a nut"). Spec Kit felt like overkill for a 3–5 point feature. — [Böckeler](https://martinfowler.com/articles/exploring-gen-ai/sdd-3-tools.html)

**Binding failures and non-determinism**
- Agents ignored or over-applied specs. Spec Kit research notes describing *existing* classes were treated as new specs, which created duplicates. Elsewhere the agent over-followed a constitution article. — [Böckeler](https://martinfowler.com/articles/exploring-gen-ai/sdd-3-tools.html)
- Spec-as-source risks "the downsides of both MDD and LLMs: Inflexibility *and* non-determinism." Tessl regeneration from an unchanged spec produced different code. — [Böckeler](https://martinfowler.com/articles/exploring-gen-ai/sdd-3-tools.html)
- Böckeler's closing question: "Are we making something worse in the attempt of making it better?" — [Böckeler](https://martinfowler.com/articles/exploring-gen-ai/sdd-3-tools.html)

**GitHub's own claim**
- GitHub's launch post presents the benefits (less guessing, a stable "what" vs a flexible "how") as reasoning, with no quantitative evidence. — [GitHub blog](https://github.blog/ai-and-ml/generative-ai/spec-driven-development-with-ai-get-started-with-a-new-open-source-toolkit/)

**Spec drift (2026 literature, mostly practitioner or vendor)**
- A June 2026 arXiv paper (Hartwig Grabowski, "The Spec Growth Engine") names "context explosion" and "silent spec-code drift" as the two main failure modes. It proposes a **drift gate** under which any spec-code divergence blocks merges. **[search summary; no independent evaluation found]** — [Emergent Mind: arXiv 2606.27045](https://www.emergentmind.com/papers/2606.27045)
- Vendor framing contrasts "living specs" that update during execution with static specs. **[vendor content, Augment Code]** — [Augment Code](https://www.augmentcode.com/guides/living-specs-vs-static-specs)
- Practitioner treatments of spec decay and continuous maintenance (not individually verified):
  - [Specs Rot Too (harness-engineering playbook)](https://lipingtababa.github.io/harness-engineering-playbook/en/chapters/02f-spec-decay.html)
  - [Glukhov: keeping specs, tests and code in sync](https://glukhov.org/app-architecture/testing-architecture/specs-tests-code-traceability-ai-development/)

**Fiction-side failure modes**
- Facts are dropped when entries are not explicitly mentioned. — [Sudowrite Story Bible](https://docs.sudowrite.com/using-sudowrite/1ow1qkGqof9rtcyGnrWUBS/what-is-story-bible/jmWepHcQdJetNrE991fjJC); [Novelcrafter](https://www.novelcrafter.com/help/faq/ai-and-prompting/codex-context-in-prompting)
- Whole-chapter regeneration is needed to apply partial beat changes. **[user feedback posts]** — [Sudowrite feedback 84994](https://feedback.sudowrite.com/feedback/84994)

### Inferences
- **Do not assume the top-layer gate saves review.** In coding, the upper layers were longer than the output (about 2.6k Markdown lines for about 0.7k code lines), so review cost rose. For the paper, the beat layers must stay much shorter than the prose they govern, or the author is reviewing the paper twice.
- Some fixes do not belong upstream. Eberhardt's bug had no spec-level expression, and the same holds for typos, LaTeX errors and citation keys. The paper system should allow lowest-layer edits that never touch locked decisions, and push edits upward only when they change meaning.
- The agent that writes prose and the agent that checks it both read the same upper layer. If that layer is stale or misread, their errors are correlated, so the check should be partly mechanical: numbers against CSVs, required phrases, cross-references.
- Non-determinism argues for **freezing approved prose** and regenerating only the paragraphs whose governing beat changed, never whole sections.

### Gaps
- No controlled study (academic or vendor A/B) measuring review time or error rates with and without upper-layer approval was found for either coding or fiction.
- No user-survey data on how often fiction users regenerate prose after outline edits.
- The Grabowski paper's content was seen only via a summary page; its evaluation, if any, is unknown.

---

## Q5. Has anyone built the author's system already, and what should be copied or avoided?

### Takeaway
I found no tool that combines all five of the author's features:
1. persistent multi-level layers from beats to final text;
2. locked author decisions that bind lower layers;
3. propagation of later edits downward;
4. a human who reviews the rendered output (PDF) and edits via chat;
5. round-tripping of collaborators' direct edits to the lowest layer (Overleaf).

The pieces exist separately. The layer chain and phase gates come from Spec Kit and Kiro. The change deltas and dated archive come from OpenSpec. Two-way sync and the "do not edit generated output" marker come from Tessl. The fact codex with always-include and position-scoped facts comes from Novelcrafter. Event hooks that refresh derived docs come from Kiro. Nearest-file-wins plus user-chat-overrides-all comes from AGENTS.md.

### Cited Findings
- Partial precedents are listed in Q1–Q3. The closest structural match is spec-anchored SDD, which Böckeler says almost no tool achieves in practice ("all SDD approaches ... are spec-first"). — [Böckeler](https://martinfowler.com/articles/exploring-gen-ai/sdd-3-tools.html)
- The only tool designed for direct edits to both layers is Tessl (two-way spec/code sync, reverse spec extraction), and its status is in flux. — [Böckeler](https://martinfowler.com/articles/exploring-gen-ai/sdd-3-tools.html); [CodeMySpec Tessl review](https://codemyspec.com/blog/tessl-review)

### Inferences (concrete copy / avoid list for the JFEC paper system)

**Copy:**
1. **Kiro EARS-style locked decisions.** Write each locked decision as a checkable rule, for example "WHEN [section/context], THE TEXT SHALL [claim/number/wording]". Add an "Analyze" pass that looks for conflicting or ambiguous decisions before drafting. ([Kiro](https://kiro.dev/docs/specs/feature-specs/))
2. **Traceability IDs.** Kiro's tasks cite requirement numbers. Every paragraph or beat should carry the IDs of the decisions it implements, so a changed decision maps to exactly the paragraphs to regenerate. ([Böckeler](https://martinfowler.com/articles/exploring-gen-ai/sdd-3-tools.html))
3. **OpenSpec delta proposals.** Turn each chat edit into an ADDED/MODIFIED/REMOVED delta against the locked layer, apply it, and archive it with a date. This gives an audit trail of why the text says what it says. ([OpenSpec](https://openspec.dev/docs/the-workflow))
4. **Steering inclusion modes.**
   - `always` for paper-wide conventions: notation, sample period, headline numbers, forbidden claims.
   - `fileMatch` for section-specific rules.
   - Workspace overrides global; nearest file wins; explicit author chat overrides all. ([Kiro Steering](https://kiro.dev/docs/steering/); [agents.md](https://agents.md/))
5. **Novelcrafter-style codex with progressions.** A "claims/numbers codex" whose entries are always included. Position-scoped entries handle terms and results introduced at a given section. ([Novelcrafter](https://www.novelcrafter.com/docs/codex/additions/))
6. **Visible markers for unresolved references, never silent drops** (Kiro steering behaviour), plus a context viewer (NovelAI) showing which decisions governed each generated paragraph. ([Kiro Steering](https://kiro.dev/docs/steering/); [NovelAI](https://docs.novelai.net/en/text/lorebook))
7. **Hooks or drift gate.** When Overleaf edits are pulled, automatically check the changed prose against the locked decisions and numbers, and block the push or flag a decision if they diverge. This combines Kiro save-hooks with Grabowski's merge-blocking gate. ([Kiro blog](https://kiro.dev/blog/how-i-stopped-worrying-about-readme-files/); [arXiv 2606.27045 summary](https://www.emergentmind.com/papers/2606.27045))
8. **Reverse extraction.** On collaborator prose edits, re-derive the beat from the prose (Tessl's `document --code`) and diff it against the locked beat. Surface meaning changes to the author as proposed decision deltas. ([Böckeler](https://martinfowler.com/articles/exploring-gen-ai/sdd-3-tools.html))

**Avoid:**
1. **Verbose upper layers.** Spec Kit's roughly 2.6k Markdown lines per feature tripled review time. Keep beats terse, and give the author the PDF plus a short decision ledger to review, not intermediate layers. ([Scott Logic](https://blog.scottlogic.com/2025/11/26/putting-spec-kit-through-its-paces-radical-idea-or-reinvented-waterfall.html); [Böckeler](https://martinfowler.com/articles/exploring-gen-ai/sdd-3-tools.html))
2. **Whole-section regeneration** on any upper-layer change, as with Sudowrite chapters and Tessl's non-deterministic regeneration. Approved prose and collaborator edits must stay frozen unless their governing decision changed. ([Sudowrite feedback](https://feedback.sudowrite.com/feedback/84994); [Böckeler](https://martinfowler.com/articles/exploring-gen-ai/sdd-3-tools.html))
3. **Forcing every fix through the top layer.** Copy-edits and LaTeX fixes have no beat-level expression. ([Scott Logic](https://blog.scottlogic.com/2025/11/26/putting-spec-kit-through-its-paces-radical-idea-or-reinvented-waterfall.html))
4. **Treating in-context decisions as enforced.** Agents both ignore and over-apply constitution rules, so verify mechanically where possible. ([Böckeler](https://martinfowler.com/articles/exploring-gen-ai/sdd-3-tools.html))
5. **Retrieval by mention only for binding facts.** Unmentioned but relevant entries get dropped. ([Novelcrafter](https://www.novelcrafter.com/help/faq/ai-and-prompting/codex-context-in-prompting); [Sudowrite](https://docs.sudowrite.com/using-sudowrite/1ow1qkGqof9rtcyGnrWUBS/what-is-story-bible/jmWepHcQdJetNrE991fjJC))
6. **Waterfall rigidity.** Critics' main objection is up-front sequential stages. The paper system should allow iterating at any layer, with propagation, rather than requiring top-down completion before drafting. ([Scott Logic](https://blog.scottlogic.com/2025/11/26/putting-spec-kit-through-its-paces-radical-idea-or-reinvented-waterfall.html))

### Gaps
- No search was done specifically for academic-paper-writing agent systems with Overleaf round-tripping (for example, research prototypes from 2025–2026). The "nobody has built this" conclusion covers only coding, fiction and general-report tools.
- No evidence was found on whether a decision ledger reviewed by one author reduces co-author review burden.
