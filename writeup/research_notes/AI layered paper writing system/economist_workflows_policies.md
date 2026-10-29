# Economist/Finance Practitioner AI Writing Workflows and Journal AI Policies (constraints on an AI-assisted JFEc submission)

Research date: 2026-10-28. Coverage 2023 to 2026. "Primary" means I read the author's or publisher's own page. "Secondary" means a summary, aggregator, or news item, and is flagged. Several practitioner pages are paywalled or only seen via the aggregator wiki at velikov-mihail.github.io/ai-econ-wiki (maintainer not named on the pages I read; it summarizes economist AI posts). Treat wiki-sourced details as secondary.

## Q1. Anton Korinek, "Generative AI for Economic Research" (JEL 2023 and updates): recommended writing workflows

### Takeaway
Korinek's original guide (NBER WP Feb 2023, JEL Dec 2023) rated every writing task "highly useful" (3 of 3). The top one was "synthesizing text": turning rough bullet points into prose. Literature search was rated "experimental" (1 of 3) because of hallucinated references. The later updates (Dec 2024: reasoning models and collaborative workspaces; Aug/Sep 2025: AI agents) moved toward agents as supervised research assistants. I did not find writing-specific guidance in the updates' abstracts.

### Cited Findings
- **Original paper.** NBER WP 30957, "Language Models and Cognitive Automation for Economic Research," February 2023. It describes "25 use cases along six domains ... ideation, writing, background research, data analysis, coding, and mathematical derivations" and rates each from experimental to highly useful. — [NBER w30957 PDF](https://www.nber.org/system/files/working_papers/w30957/w30957.pdf) (primary; text extracted locally)
- **Published version.** JEL vol. 61, no. 4 (2023), pp. 1281-1317, DOI 10.1257/jel.20231736. — [AEA](https://www.aeaweb.org/doi/10.1257/jel.20231736); [EconPapers](https://econpapers.repec.org/RePEc:aea:jeclit:v:61:y:2023:i:4:p:1281-1317)
- **Table 1 ratings, "as of Feb 1, 2023"** (1 = experimental, 2 = useful, 3 = highly useful) — [NBER w30957 PDF](https://www.nber.org/system/files/working_papers/w30957/w30957.pdf):
  - **Writing, all rated 3:** Synthesizing text, Editing text, Evaluating text, Generating catchy titles & headlines, Generating tweets to promote a paper.
  - **Ideation:** Brainstorming 3, Evaluating ideas 2, Providing counterarguments 3.
  - **Background research:** Summarizing 3, Literature Research 1, Formatting References 3, Translating 3, Explaining Concepts 2.
  - **Coding:** Writing code 2, Debugging 2.
  - **Math:** Deriving equations 1, Explaining models 1.
- **Bullets to prose (his headline writing workflow).** "One of the most useful capabilities of LLMs may be to translate rough bullet points into well-structured sentences." His example prompt: "Please write a paragraph that integrates the following arguments and goes beyond. Include a topic sentence." The bullets that follow carry bracketed inline instructions such as "[add examples]" and "[offer solutions]". He says this "allows researchers to concentrate their energy on the ideas in their text as opposed to the writing process itself." He warns the output is "a bit sterile and lacks the idiosyncracies and elements of surprise that characterize human writing — a feature that detectors of LLM-written text zero in on." — [NBER w30957 PDF](https://www.nber.org/system/files/working_papers/w30957/w30957.pdf)
- **LaTeX and slides.** He notes LLMs "can also write text in LaTeX format" and "draft beamer slides in LaTeX code based on bullet points." — [NBER w30957 PDF](https://www.nber.org/system/files/working_papers/w30957/w30957.pdf)
- **Evaluating text.** Asking "What are the main stylistic shortcomings of the following text?" about his own draft abstract was "a bit of a deflating experience." He uses this as a self-review step. — [NBER w30957 PDF](https://www.nber.org/system/files/working_papers/w30957/w30957.pdf)
- **Citations.** LLMs "may hallucinate and make up authoritative-sounding papers that do not actually exist ... all references they provide should always be double-checked." Once references are verified, LLMs are "very capable of formatting them" (e.g., to BibTeX). — [NBER w30957 PDF](https://www.nber.org/system/files/working_papers/w30957/w30957.pdf)
- **His own disclosure, an early example of an acknowledgements-line statement.** The WP thanks "GPT-3 and Claude for inspiration and editorial assistance." — [NBER w30957 PDF](https://www.nber.org/system/files/working_papers/w30957/w30957.pdf)
- **Prompt archive.** The prompts and chats behind the paper's examples are archived on openICPSR (project 194623). — [openICPSR](https://www.openicpsr.org/openicpsr/project/194623/view) (search snippet only)
- **December 2024 update, "LLMs Learn to Collaborate and Reason"** (NBER WP 33198; dated Nov or Dec 2024 depending on copy). It covers "(i) new reasoning capabilities, (ii) novel workspaces for interactive LLM collaboration such as Claude's Artifacts, ChatGPT's Canvas or Microsoft's Copilot, and (iii) recent improvements in LLM-powered internet search." It adds a section 3.7 on promoting research (blog posts, slides, podcasts). — [NBER w33198](https://www.nber.org/papers/w33198); [Brookings PDF](https://www.brookings.edu/wp-content/uploads/2025/01/JEL-2024-Dec-LLMsCollaborateReason.pdf) (abstract and search snippets only; full text not read)
- **August/September 2025 update, "AI Agents for Economic Research"** (NBER WP 34202, Sept 2025; listed as an update to the JEL article). It covers agents that "autonomously conduct literature reviews, write and debug econometric code, fetch and analyze economic data, and coordinate research workflows." It uses "vibe coding" and LangGraph, with a worked FRED-data agent. Korinek discloses unpaid service on Anthropic's Economic Advisory Board/Council. JEL editor David Romer asked him for semi-annual updates. Companion site: GenAIforEcon.org. — [NBER w34202](https://www.nber.org/papers/w34202); [Marginal Revolution, Sept 2025](https://marginalrevolution.com/marginalrevolution/2025/09/ai-agents-for-economic-research.html)
- **Agents as supervised RAs.** A secondary summary says Korinek sees agents as most useful today as supervised research assistants, with fully autonomous assistants possible later. — [ai-econ-wiki summary](https://velikov-mihail.github.io/ai-econ-wiki/summaries/ai-agents-generative-ai) (secondary; not verified in the paper)

### Inferences
- The user's design (nested bullets plus a generated prose layer) is the scaled-up version of Korinek's top-rated "synthesizing text" use case. His bracketed inline instructions ("[add examples]") are a direct precedent for beat-level annotations inside bullets.
- Korinek's own sequence is: draft bullets, have the LLM synthesize, ask it to "evaluate stylistic shortcomings," and verify every citation by hand. That maps onto a generate, critique, verify loop.
- Korinek himself disclosed AI help in the acknowledgements in 2023, before OUP's 2026 policy made that location mandatory.

### Gaps
- I did not read the full text of the Dec 2024 or 2025 updates, so I can't say whether they revise the writing ratings or give agent-based paper-drafting advice.
- I could not confirm whether a 2026 update exists.

## Q2. Practitioner workflows: economists and finance academics using Claude Code, Cursor, or ChatGPT for papers

### Takeaway
The people who post publicly (Cunningham, Goldsmith-Pinkham, Blattman, Gans) converge on the same pattern:
- A persistent markdown context file (CLAUDE.md) plus state and progress files.
- Plan before executing.
- Every number comes from code, never from the model, with the repo linked to Overleaf.
- Git commits as the review unit.
- Independent adversarial reviewers in fresh sessions, or other vendors' models, because self-review is unreliable.
- A personal voice or style file.

Their warnings: confident wrong results ("seduction"), bloat, weak idea-screening, prompt-induced confirmation bias at the interpretation stage, and too many skills.

### Cited Findings

**Scott Cunningham (Baylor; "Scott's Mixtape" Substack Claude Code series, Dec 2025 onward)**
- **Start and cost.** The series began with "Claude Code Changed How I Work (Part 1)," dated December 13, 2025, subtitled "Or How I Paid $200/Month for an AI Coding Agent. I'm Never Going Back." Most posts are paywalled (in 2026 he coin-flips which posts are). — [Substack Part 1](https://causalinf.substack.com/p/claude-code-changed-how-i-work-part) (primary, visible intro only)
- **MixtapeTools repo** ("Tools for empirical economics research with Claude Code"; README says "Last updated: May 2026"; ~482 stars). Layout:
  - `workflow.md` ("How I use Claude Code for research (START HERE)")
  - `claude/CLAUDE.md` template ("Copy it to your project root ... Claude Code will automatically read it every session")
  - `personas/referee2.md`
  - `skills/` (Blindspot, Split-PDF, Newproject, TikZ, referee2)
  - `.claude/` slash commands (`compiledeck`, `compiletex`, `newproject`)
  - a work-in-progress `gtd/` research harness
  - — [GitHub MixtapeTools](https://github.com/scunning1975/MixtapeTools) (primary)
- **Recommended project layout:** `CLAUDE.md` at root; `correspondence/referee2/` for dated reports, decks, and responses; `code/` split by language with `replication/` for Referee 2 scripts; `data/raw`, `data/clean`; `output/tables`, `output/figures`. Rule: "If it's not documented, it didn't happen"; every audit produces a dated referee report in `correspondence/`. — [GitHub MixtapeTools](https://github.com/scunning1975/MixtapeTools)
- **Referee 2 protocol.**
  - Five audits: Code, Cross-Language Replication, Directory, Output Automation, Econometrics.
  - "Referee 2 NEVER modifies author code. It only creates its own replication scripts."
  - "Fresh terminal — independence is structural."
  - "If results aren't identical to 6+ decimal places across implementations, something is wrong."
  - — [GitHub MixtapeTools](https://github.com/scunning1975/MixtapeTools)
- **Bibcheck is limited.** It is bounded to the `.bib` file and "does NOT check whether each citation supports the claim it's attached to in the manuscript." Blindspot audits output perception in the same session and complements Referee 2. — [GitHub MixtapeTools](https://github.com/scunning1975/MixtapeTools)
- **Part 12 habits** (secondary summary):
  - Ask "Guess what I'm about to ask you" to surface misunderstandings.
  - Request figures constantly, since errors show up in plots.
  - CLAUDE.md, READMEs, and session logs act as external memory, because Claude "has amnesia" between sessions.
  - Warnings: "almost right" results are unacceptable; self-review is unreliable; "research is not product development."
  - — [ai-econ-wiki summary of Part 12](https://velikov-mihail.github.io/ai-econ-wiki/summaries/cc-series-12-empirical-research); original at causalinf.substack.com/p/claude-code-part-12-how-i-use-claude (not read directly)
- **Paper-specific installments:** Part 8 resurrects and extends an old abortion paper toward continuous diff-in-diff. Part 9 builds .bib files and automates article retrieval. Part 34 returns to a cannabis paper. — [Part 8](https://causalinf.substack.com/p/claude-code-series-part-8-resurrecting-ee3); [Part 9](https://causalinf.substack.com/p/claude-code-series-part-9-creating); [Part 34](https://causalinf.substack.com/p/claude-code-34-using-dispatch-on) (titles from search results only)

**Paul Goldsmith-Pinkham (Yale SOM finance; NBER)**
- **Series.** "Claude Code for Applied Economists, a Mini-Series," Markus' Academy Ep. 162, March 29, 2026. — [Markus Academy Substack](https://markusacademy.substack.com/p/claude-code-for-applied-economists) (primary)
- **Verification and Overleaf.** He replicated Jay Ritter's post-IPO results and committed each step to GitHub so he could review Claude's work "one change at a time." He connected the repo to Overleaf so Claude updates figures and tables directly in the paper. Core rule: "every number in a draft should come from the code, not from an AI's memory." Episode 8 framing: "Verification is the new bottleneck." — [Markus Academy Substack](https://markusacademy.substack.com/p/claude-code-for-applied-economists)
- **Writing.**
  - Episode 5: build a personal style guide from your own papers.
  - Use skills to turn a referee report into a revision plan (he recommends Jukka Sihvonen's strategic-revision skill).
  - Use LLMs as "editors and thinking partners rather than judgment outsourcers."
  - Episode 6 has "cautions about overusing Skills."
  - Episode 7 covers permissions and autonomy, with containers or Docker sandboxes for safer autonomy.
  - Episode 4 covers planning mode and sub-agents.
  - — [Markus Academy Substack](https://markusacademy.substack.com/p/claude-code-for-applied-economists)
- **Getting-started habits** (secondary summary of paulgp.substack.com/p/getting-started-with-claude-code):
  - `/compact` and writing state to `progress.md`.
  - Focused sessions of about 5-10 turns rather than marathons.
  - Correct the model early (e.g., "use R instead of Python").
  - "IRB data and PII should be walled off."
  - Ghostty, Zellij, and Oh My Zsh as the terminal setup.
  - $20 Pro tier is enough for most.
  - — [ai-econ-wiki summary](https://velikov-mihail.github.io/ai-econ-wiki/summaries/getting-started-researchers)
- **Visualization instruction.** He uses a standing instruction to follow Kieran Healy's visualization best practices. — [ai-econ-wiki summary](https://velikov-mihail.github.io/ai-econ-wiki/summaries/data-analysis-economists) (secondary)

**Chris Blattman (UChicago Harris; claudeblattman.com, started Jan 2026)**
- **Background.** He began building Claude Code workflows in January 2026 with no coding background. The site offers a CLAUDE.md template, "AI Project Folders," and about 20 downloadable skills. — [claudeblattman.com](https://claudeblattman.com/) (primary)
- **"Prompt, Plan, Review, Revise" loop:** brain-dump, then structure with `/prompt`, then stress-test with fresh agents via `/review-plan`, then capture lessons with `/done`. `/recall` searches past sessions. — [claudeblattman.com](https://claudeblattman.com/)
- **`/council`** dispatches up to five critics in parallel (skeptic, pre-mortem, methodologist, budget hawk, ...), then a separate synthesis. It runs a single round with a hard cap of five and no majority vote. He warns: "A council of five Claudes still shares one model's blind spots." His remedy is cross-vendor critics (Codex, Gemini). — [claudeblattman.com](https://claudeblattman.com/)
- **"Teaching AI Your Voice":** a voice file with a ban list, annotated examples, and sentence-level rules, loaded into Claude Code. He also has a "Dial-Back Discipline" page (Apr 2026). — [claudeblattman.com](https://claudeblattman.com/)

**Joshua Gans (Toronto Rotman), "Reflections on Vibe Researching," Jan 3, 2026** (primary)
- **Workflow.** An o1-pro paper took under an hour and was published in *Economics Letters*. Through 2025 he cleared his idea backlog AI-first, skipping pre-submission peer feedback and using journals as validation. — [Gans Substack](https://joshuagans.substack.com/p/reflections-on-vibe-researching)
- **Outcome.** Many working papers. A few acceptances at lower-tier journals, a few R&Rs at better ones, none at top tier. "the experiment — can we do research at high speed without much human input — was a failure." — [Gans Substack](https://joshuagans.substack.com/p/reflections-on-vibe-researching)
- **Failures.**
  - Mathematical overreach: equilibrium claims were incomplete, and only one referee caught it each time.
  - Low-quality ideas, because lower effort removed the natural points to abandon a project.
  - Bloat from easy extensions.
  - "Seduction": "formal results are claimed with confidence"; models "are trained to be pleasing, and by golly, they know what pleases a researcher."
  - — [Gans Substack](https://joshuagans.substack.com/p/reflections-on-vibe-researching)
- **Guardrails.**
  - A pause of "at least a month" before anything goes up.
  - More seminars and peer feedback.
  - Explicit go/no-go points.
  - Cross-checking across models.
  - He recommends Refine.ink.
  - — [Gans Substack](https://joshuagans.substack.com/p/reflections-on-vibe-researching)

**Ben Golub (Northwestern) and Refine (refine.ink)**
- **Product.** Refine is an AI verification and referee service that checks "reasoning, numbers, evidence, and consistency between text, tables, and figures." Issues are ranked and tied to passages; turnaround is 20 to 40 minutes; the first document is free. — [refine.ink](https://www.refine.ink/) (primary, vendor claims)
- **Privacy:** "Nothing you upload ever becomes training data." Claims SOC 2 and ISO 27001. — [refine.ink](https://www.refine.ink/)
- **Adoption claims** (vendor): "Used by leading academic journals, including those of" the AEA and the Econometric Society. Self-reported benchmark: 197 of 215 head-to-head wins. — [refine.ink](https://www.refine.ink/) (unverified)
- **Founding and press** (secondary): co-founded by Golub and Yann Calvó López in 2025. Strongest on theory papers; tables and figures often not parsed. An FT column (March 2026) reported top-5 journals experimenting with it. ACM EC 2026 offered authors optional Refine pre-submission feedback. — [ai-econ-wiki summary](https://velikov-mihail.github.io/ai-econ-wiki/summaries/refine-ink-golub); [Princeton BCF slides, Dec 2025](https://bcf.princeton.edu/wp-content/uploads/2025/12/2025.12.18-MA-Ben-Golub.pdf) (secondary; FT and ACM claims not verified)

**Interpretation-stage bias (empirical warning, June 2026)**
- Alizadeh, Gilardi, Mosleh, Kasneci (arXiv 2606.11456, June 9, 2026) ran 20 runs each of Claude Code and Codex on an immigration and social-policy study. Findings:
  - Claude Code produced nearly three times as many specifications as humans.
  - Estimates were broadly aligned with the human consensus.
  - An explicit confirmatory prompt flipped Claude Code's support verdicts from 10% to 90% "while its coefficient distribution stayed essentially unchanged," via "rule omission rather than rule softening."
  - "the locus of AI bias is not estimation but interpretation."
  - — [arXiv 2606.11456](https://arxiv.org/abs/2606.11456) (abstract)

**Others (thin or unverified)**
- **Tyler Cowen.** Uses o3/o3 Pro daily, paired with Perplexity (July 2025 interview, secondary). A March 2026 MR post, "When will the research paper disappear?", reportedly imagines papers replaced by data and code that readers query. — [search summary via ai-econ-wiki](https://velikov-mihail.github.io/ai-econ-wiki/summaries/research-paper-disappear); [New Savanna critique, May 2026](https://new-savanna.blogspot.com/2026/05/tyler-cowen-is-imagining-that-ai-will.html) (secondary; MR post not read)
- **Ethan Mollick.** "Claude Code and What Comes Next," One Useful Thing, 2026-01-07: a 74-minute single-prompt build; explains subagents, compaction, skills, and MCP. — [ai-econ-wiki summary](https://velikov-mihail.github.io/ai-econ-wiki/summaries/claude-code-what-comes-next) (secondary)
- **Andrew Heiss.** His course AI policy (relayed from Bluesky): "everything human-facing must be human-generated." He keeps an "/ai" statement on andrewheiss.com. — [AI Weekly profile](https://aiweekly.co/whos-who/person/andrew-heiss-phd) (secondary; his page not retrieved)
- **Generic multi-agent manuscript review** (semantic, technical, and exposition reviewers; quality gates). Advice: "Always verify outputs, especially for econometric results"; keep the project in Git. It points to Pedro Sant'Anna's templates. — [zhiyuanryanchen.github.io, Feb 2025](https://zhiyuanryanchen.github.io/claude-code-workflow.html)
- **Workshops** (not read in detail):
  - Alex Rieber and Dennis Steinle (Ulm), "AI Coding Agents for Economics Research" (includes a session on making a replication package) — [Rieber workshop](https://alexrieber.github.io/Workshops/AI_Agents/Tutorial/agent-workshop-howto.html)
  - "Agentic AI for Researchers," with a CLAUDE.md/AGENTS.md session — [codedthinking](https://codedthinking.github.io/ai-workshop/)

### Inferences
- **Practices that match the user's setup:**
  - Every number from code via macros (Goldsmith-Pinkham's rule).
  - A repo synced to Overleaf (Goldsmith-Pinkham).
  - CLAUDE.md plus external state files (Cunningham, Goldsmith-Pinkham, Blattman).
  - Independent reviewers in fresh contexts (Cunningham's Referee 2, Blattman's council).
- **Gaps in the user's practice relative to theirs:**
  - **Cross-vendor review.** Blattman's point that same-model critics share blind spots; Gans also cross-checks across models.
  - **Claim-support check.** Cunningham says explicitly that his bibcheck does not test whether a citation supports the attached claim.
  - **Interpretation rules written before drafting.** The arXiv finding implies agent-drafted interpretive prose is where bias enters. Pre-registered rules mapping estimates to verdict sentences, kept separate from what the drafting prompt asks for, would guard this.
  - **A cooling-off period** before submission (Gans).
- Nobody I found publishes a "nested bullets plus auto-generated beat layer" architecture. Korinek's bullets-to-prose and Blattman's voice file are the nearest precedents.

### Gaps
- **Cunningham:** CLAUDE.md template contents, `workflow.md`, and paper-drafting posts are paywalled or unread. Specifics come from the public README and a secondary wiki.
- **Goldsmith-Pinkham:** his own CLAUDE.md and folder layout were not seen. The Overleaf-linking mechanism (git bridge vs Dropbox) is not specified.
- **John Horton** (MIT; Expected Parrot): no source found on his Claude Code paper-writing workflow.
- **Grant McDermott:** nothing found.
- **Cowen, Mollick, Heiss:** only secondary sources.
- **Cursor:** no economist's Cursor-specific paper-writing workflow found.
- **Golub/Refine:** FT, ACM EC, and journal-adoption claims are unverified.

## Q3. Journal and platform AI policies constraining an AI-assisted JFEc submission

### Takeaway
JFEc's own instructions have no AI clause, so OUP's journal-wide "Policy on AI Use & Disclosure for OUP Journals Authors" (last updated 13 July 2026, announced 10 Sept 2026) governs. Under it:
- Substantial AI rewriting or reorganizing of text, AI-generated or refined code, and AI data analysis or visualization must be disclosed in **both the cover letter and the Acknowledgements**, and in **Methodology** if AI was integral to the research.
- The disclosure must give **tool name and version, how it was used, and the author's role in directing and reviewing**.
- It must be updated at each revision, and the author must assert compliance at submission.
- AI cannot be an author.
- Reviewers at OUP and other publishers are broadly barred from uploading manuscripts to AI tools.

Other finance and economics outlets (Elsevier's JFE and JoE, AEA, the Journal of Finance via AFA/Wiley, Econometrica, SSRN) all allow AI drafting help with disclosure. They differ on where the disclosure goes.

### Cited Findings

**JFEc (Oxford University Press) author instructions**
- **No AI policy on the page.** The JFEc General Instructions page contains no policy on generative AI, LLMs, or AI disclosure. It does offer an optional "AI-driven" Paperpal Preflight technical-check tool via Cactus. — [JFEc General Instructions](https://academic.oup.com/jfec/pages/General_Instructions) (primary, read 2026-10-28)
- **Other JFEc requirements relevant to an agent-built paper:**
  - Empirical papers must "upload source files and relevant data files for editorial replicability review" (for submissions from 13 March 2023). Data are a condition of acceptance unless an exemption is requested in the cover letter.
  - Software and datasets must be cited.
  - Length "should typically not exceed 40 double-spaced typewritten pages"; abstract of 100 words or fewer, with 2-6 keywords and JEL codes.
  - Chicago Manual of Style (15th ed.).
  - LaTeX submitted as a PDF, with source files after acceptance.
  - Preprints (Author's Original Version) are allowed.
  - Manuscripts may be screened by third-party services.
  - — [JFEc General Instructions](https://academic.oup.com/jfec/pages/General_Instructions)

**OUP Journals: "Policy on AI Use & Disclosure for OUP Journals Authors" ("Policy last updated on: 13 July 2026")** — [OUP policy page](https://academic.oup.com/pages/for-authors/journals/preparing-and-submitting-your-manuscript/ai-principles-for-oxford-journals/policy-on-ai-use-and-disclosure-for-oxford-journals-authors) (primary; read via fetch tool, which paraphrased with short quotes)
- **Scope.** Applies to all content types. Journal-specific rules can add requirements and prevail on conflict. Authors must assert at submission that they have read and adhered to the policy.
- **No disclosure needed:** language-clarity help (spelling and grammar); formatting references to journal style, provided they are accurate, with "no hallucinations."
- **Typically requires disclosure:**
  - Substantially editing, rewriting, or reorganizing text
  - Translating
  - Collecting, organizing, formatting, analyzing, or visualizing data
  - Generating or refining code
  - Generating illustrative or explanatory images
  - Manipulating or deriving images or figures from published works (subject to conditions)
- **Prohibited:**
  - Presenting AI-created content as original human research data or results.
  - Creating or altering primary factual or technical images.
- **Where:** the cover letter to editors **and** the manuscript's Acknowledgements; also Methodology "if AI use was integral to the research methodology." Required at initial submission and updated for each revision.
- **Required details:**
  - Tool name and version
  - How it was used (e.g., substantial editing, text generation, document organization, translation)
  - The author's role in directing and reviewing output to ensure accuracy
- **Strongly recommended for complex cases:** dates of use and the specific sections affected.
- **Accountability:** "Authors are accountable for complete and accurate disclosure." Undisclosed or non-compliant use may lead to revision requests or rejection before publication, or to correction or retraction under COPE after.
- **What the page does not say.** It does not ask for prompt disclosure or retention and gives no template statement. Authorship and confidentiality are covered by the principles page and the ethics page.
- **AI Principles for OUP Journals:** "Authorship and contribution," "Accountability," "Confidentiality and data protection," "Avoidance of bias," "Transparency and disclosure." It warns "Gen AI Tools will sometimes grant the tool's owner rights to the content uploaded into it." — [OUP AI Principles](https://academic.oup.com/pages/for-authors/journals/preparing-and-submitting-your-manuscript/ai-principles-for-oxford-journals) (primary)
- **OUP publication-ethics page.** AI tools such as ChatGPT do not qualify as authors (the sentence opens "Neither ... nor"), and OUP checks author lists. AI use ("to help generate content or images, write code, process data, or for translation") "must be disclosed both in cover letters to editors and in the Methods or Acknowledgements section of manuscripts." — [OUP ethics page](https://academic.oup.com/pages/authoring/journals/preparing_your_manuscript/ethics) (primary; older wording, consistent with the 2026 policy)
- **Announcement (10 Sept 2026).** OUP expects transparency about "any significant use of AI tools" and human verification of AI output, and says to check tool terms before uploading unpublished work. Its survey found 46% of journal authors had not fully disclosed AI use "particularly in narrative drafting," and only 42% were asked to attest. — [OUP corporate news](https://corp.oup.com/news/supporting-researchers-to-disclose-ai-use-with-confidence/) (primary); [LetsDataScience](https://letsdatascience.com/news/oxford-university-press-updates-ai-disclosure-guidance-3bfeb2f9) (secondary)
- **Conflicting third-party summary.** Enago (updated April 2026) says OUP holds that gen AI "should not carry out primary author tasks such as generating arguments, writing analysis, or drawing conclusions," and that gen-AI charts or data graphics require OUP's written permission. I did **not** find this wording in OUP's July 2026 journals policy, which lists "text generation" as a disclosable use. It may reflect OUP's **books** guidance or an older version. — [Enago OUP summary](https://www.enago.com/responsible-ai-movement/publisher-ai-guidelines/oup-ai-guidelines) (secondary; conflicts with primary)

**Reviewer confidentiality**
- **OUP:** "OUP selects peer reviewers for their expertise in the field and requires them to evaluate content based on their expert judgement alone." "It is prohibited to upload project proposals and manuscripts, in part or in whole, into a Gen AI tool for any purpose." Daily Nous attributes this to OUP's **books** AI page. — [Daily Nous, 2026-09-28](https://dailynous.com/2026/09/28/ai-in-peer-review-practices-policies/); [OUP books AI page](https://academic.oup.com/pages/for-authors/books/author-use-of-artificial-intelligence)
- **One OUP society journal (ICB)** "does not currently allow the use of AI-assisted technologies ... by reviewers or editors," and notes OUP "is actively evaluating compliant AI Tools." — [ICB/SICB AI policy](https://sites.google.com/sicb.org/authorguideforicb/author-resources/integrity-ethics/ai-policy-icb)
- **Wiley:** "Manuscripts under review are confidential and must not be uploaded to AI Technologies, whether in full or in part"; reviewers must disclose drafting assistance (tool, version, date, role). — [Wiley AI guidelines](https://www.wiley.com/publishing/article/ai-guidelines) (primary)
- **Elsevier:** "Reviewers should not upload a submitted manuscript or any part of it into an AI tool"; editors likewise. — [Elsevier policy](https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals) (primary)
- **AEA:** referee AI use "must be disclosed in the reviewer's cover letter and must be limited to tools that guarantee that the manuscript will not be used for training AI models." — [AER editorial policy](https://www.aeaweb.org/journals/aer/editorial-policy) (primary)
- **Journal of Finance:** AI "may be employed in the preparation of referee reports ... to assist with drafting or editing text or looking up prior references." No AI-specific confidentiality clause. — [AFA editorial policies](https://afajof.org/editorial-policies/) (primary)
- **University of Chicago Press:** "expressly prohibit the use of AI tools in writing readers' reports." — [Daily Nous](https://dailynous.com/2026/09/28/ai-in-peer-review-practices-policies/)

**Elsevier (Journal of Financial Economics, Journal of Econometrics), "Policy updated June 2026"** — [Elsevier generative AI policies](https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals) (primary)
- **Allowed.** AI may support summarizing literature, organizing content, and improving clarity, but "must never be used as a substitute for human critical thinking, expertise and evaluation."
- **Statement.** A "separate AI declaration statement" goes "at the end of your manuscript, immediately before the references." It names the tool and purpose and confirms the author reviewed the output and takes responsibility. Keeping logs, prompts, outputs, or tracked-change drafts is recommended. Grammar, spelling, and punctuation checks are exempt. AI used in the research process goes in Methods.
- **Figures.** Data visualizations are permitted "only when the visual output is directly derived from underlying data" with reproducible methods, and tool, version, and developer named in Methods. Explanatory images need a per-caption disclosure. General gen-AI tools may not make graphical abstracts.
- **Authorship.** "Authors should not list AI tools as an author or co-author, nor cite AI tools as an author."

**AEA journals (AER, AER: Insights, AEJs)** — [AER editorial policy](https://www.aeaweb.org/journals/aer/editorial-policy) (primary; no date on AI section)
- AI "may not be listed as an author."
- "If AI software was used in the preparation of the manuscript, including drafting or editing text, this must be briefly described during the submission process."
- "Authors are solely accountable for, and must thoroughly fact-check, outputs created with the help of AI software."

**Journal of Finance (AFA; published by Wiley)** — [AFA editorial policies](https://afajof.org/editorial-policies/) (primary)
- AI "may be employed in the preparation of a manuscript ... to assist authors in drafting or editing text or improving the quality of their code."
- Authors "may briefly describe" use (permissive, not mandatory) and must not list AI as co-authors.
- For AI-created datasets or classification: "either use open-source AI models or clearly time-stamped versions of proprietary models" and "provide detailed documentation of the prompts used to create the data."
- **Wiley's overlay:** disclose "In your Acknowledgments when AI assists with manuscript drafting, editing, translation, or formatting." "Authors may only use AI Technology as a companion to their writing process, not a replacement." AI cannot "be listed as, an author." — [Wiley AI guidelines](https://www.wiley.com/publishing/article/ai-guidelines)

**Econometric Society (Econometrica, QE)**
- "Authors preparing a manuscript for Econometrica may use AI tools, but remain responsible and accountable for all content." — [Econometrica editorial procedures](https://econometricsociety.org/publications/econometrica/editorial-procedures-and-policies) (search snippet)
- QE: AI "may not be listed as an author." — [QE policies](https://www.econometricsociety.org/publications/quantitative-economics/submissions/editorial-procedures-and-policies) (search snippet)

**SSRN ("Last updated on August 03, 2026")** — [SSRN AI policy](https://www.elsevier.support/ssrn/answer/AI) (primary)
- **Permitted:** grammar and readability, "Refining language and structure," translation, literature discovery.
- **Disclosure:** "any material use of generative AI"; "The disclosure should be included in the PDF"; "briefly describe the tool used, how it was used, and the extent of its contribution." Required when "an AI tool makes substantive changes to sentence structure."
- **Prohibited:**
  - "Fabricating or falsifying data, results, citations, or references"
  - "Generating fictional research findings"
  - "Misrepresenting AI-generated content as original human scholarship"
- **Enforcement.** SSRN "reserves the right to investigate."
- **Enforcement case.** SSRN removed 257 papers by Nicholas Polson (Chicago Booth) and froze the upload accounts in Sept 2026. His papers lacked AI disclosure, but SSRN said the bigger issue was "the unusual pattern of paper submissions" (about 200 papers in nine months of 2026). — [AI Weekly, 2026-09-24](https://aiweekly.co/alerts/ssrn-pulls-257-papers-by-u-chicagos-nicholas-polson) (secondary); original [Washington Post, 2026-09-23](https://washingtonpost.com/education/2026/09/23/nine-months-200-academic-papers-one-professors-ai-assisted-output) (not read)
- **2023 policy for comparison:** AI was allowed only to improve readability and language, with an in-paper disclosure statement. — [SSRN blog, 2023-03-07](https://blog.ssrn.com/2023/03/07/ai-preprints-and-ssrn-a-new-policy) (snippet)

**NBER**
- I found no NBER working-paper AI disclosure policy. — searches returned nothing from nber.org (gap)

### Inferences
- **For the user's JFEc submission, the binding rule is OUP's July 2026 journals policy.** Agent-drafted prose ("substantially ... rewriting or reorganizing text," "text generation"), agent-written estimation and plotting code, and agent-generated macros and figures all fall in the disclosable categories.
  - **Disclosure needed in:** the cover letter, the Acknowledgements, and probably Methodology (if the agent pipeline counts as integral to the research).
  - **Details to give:** Claude model name and version (e.g., model IDs), Claude Code as the harness, and what it did (drafting from author-written bullets/beats, code, macros).
  - **The author's role:** writes the bullets/beats and decision rules; reviews every generated paragraph; numbers come only from code.
  - **Recommended for complex cases:** dates and sections.
  - Refresh the statement at each revision.
- **Draft template (inference, not an OUP template):**
  > "The authors used Anthropic's Claude [model/version] via Claude Code ([dates]) to generate and refine analysis code, to typeset numerical results as LaTeX macros produced by that code, and to draft prose from author-written outlines in Sections [x-y]. The authors specified all arguments, reviewed and edited all generated text and code, and take full responsibility for the content."
- **Replication package.** The JFEc replicability requirement means agent-written code and the macro-generation pipeline will be inspected by the editorial office. Under JF-style norms, prompts that generate data would need documentation. JFEc does not require this, but keeping prompt and decision logs (Elsevier recommends it) is cheap insurance.
- **Tool terms.** OUP's confidentiality principle ("Gen AI Tools will sometimes grant the tool's owner rights to the content uploaded") means the user should confirm that their Claude plan's data-use and training settings are compatible before uploading unpublished work. Same for third-party referee tools like Refine, which contractually excludes training.
- **Refereeing.** If the user referees for JFEc/OUP, Elsevier, or Wiley journals, uploading the submission to Claude or Refine is prohibited or effectively prohibited. AEA allows only no-training tools, with disclosure. This also implies the user's own paper may be refereed without AI help at OUP.
- **SSRN.** Posting the working paper to SSRN requires an in-PDF disclosure. Mass or anomalous posting patterns draw scrutiny (the Polson case).

### Gaps
- **OUP journals reviewer policy.** I could not retrieve the reviewer-specific page. The "prohibited to upload" wording is attested for OUP books and one OUP society journal. For journals generally it is likely but unverified.
- **Exact OUP wording.** The fetch tool paraphrased the July 2026 policy, so verbatim wording beyond short quotes should be checked on the page before quoting in a cover letter.
- **JFEc editor guidance.** I don't know whether JFEc editors have issued any AI guidance beyond OUP's (none on the instructions page).
- **Econometrica.** I did not read its policy directly.
- **NBER.** No WP policy found.
- **Enago's "no primary author tasks" and "written permission for AI charts" claims** remain unreconciled with OUP's journals policy.

## Q4. Surveys of economists' AI use for writing (2024-2026)

### Takeaway
The best 2026 data (an Anthropic survey of 1,260 quantitative social scientists, Feb-Mar 2026) shows economists lead coding-agent adoption at 39%, against about 20% overall. Only about a third of AI users have drafted prose with AI, and "only economists and management researchers commonly draft prose with AI." OUP's own 2026 survey found widespread non-disclosure, particularly of narrative drafting.

### Cited Findings
- **Anthropic, "Coding Agents in the Social Sciences," May 27, 2026** (Lyttelton, Massenkoff, Wilmers). — [Anthropic](https://www.anthropic.com/research/coding-agents-social-sciences) (primary)
  - **Sample:** 1,260 quantitative social scientists, Feb-Mar 2026; about 40% full or associate professors, 25% assistant professors, about 30% PhD students.
  - **Adoption:** 81% have tried chatbots for research; 20% use coding agents at least weekly; economists 39% (highest), political science 25%, education 4%.
  - **Tools:** 86% of agent users use Claude Code, 31% Codex.
  - **Uses:** code generation is top (97% of agent users). Editing prose is the next most common use. "Only about a third of all AI users have used AI to draft prose," and "only economists and management researchers commonly draft prose with AI."
  - **Output:** agent users post about 75% more working papers and start about 10% more empirical projects. There is "No evidence of more journal submissions or faster resubmissions."
  - **Gaps by group:** male-named researchers adopt at more than twice the rate of female-named; top-university researchers about 40% more.
  - **Caveats:** a non-representative sample recruited via Claude Max access; descriptive, not causal; a randomized Claude Code experiment is ongoing.
- **OUP survey (Sept 2026):** 46% of journal authors had not fully disclosed AI use in manuscript preparation, "particularly in narrative drafting"; only 42% were asked to attest. 25% of peer reviewers had not fully disclosed AI use. — [OUP corporate news](https://corp.oup.com/news/supporting-researchers-to-disclose-ai-use-with-confidence/) (primary)
- **IZA DP 18062 (2025):** a text analysis finds a stylistic shift in IZA working-paper abstracts after ChatGPT-3.5, consistent with LLM-assisted writing spreading. — [IDEAS](https://ideas.repec.org/p/iza/izadps/dp18062.html) (search snippet only)
- **Dutch ESB survey:** ESB surveyed Dutch economic researchers' generative AI use; results pending publication. — [ESB](https://esb.nu/enquete-naar-ai-gebruik-voor-onderzoek/) (snippet)
- **Sociology comparison (2025):** a survey of authors in 50 sociology journals found GenAI used mainly for writing tasks (revising, summarizing, editing, translating). — [arXiv 2511.16884](https://arxiv.org/html/2511.16884v2) (snippet; adjacent field)
- **Central banks:** the same sociology paper cites a 2024 finding that fewer than half of PhD economists at European central banks use ChatGPT or Claude. — [arXiv 2511.16884](https://arxiv.org/html/2511.16884v2) (secondhand; original not verified)
- **Professional economists:** Dal Bianco's pulse survey and focus group of professional economists on GenAI (focus on employability, not research writing). — [IRIS Unilink](https://iris.unilink.it/handle/20.500.14085/48421) (snippet)

### Inferences
- **Drafting prose is a minority practice even among AI users,** though concentrated in economics. A heavily agent-drafted JFEc paper is at the frontier, which raises the value of a clear, specific disclosure. OUP's own data show narrative drafting is what authors under-disclose and what the July 2026 policy targets.
- **The Anthropic finding of more working papers but not more journal submissions** echoes Gans: agents raise output volume more than publishable output. Quality gates (cooling-off, external review) are where effort pays.

### Gaps
- I found no finance-specific survey (e.g., AFA or SFS members) on AI writing.
- No economics survey directly measures share-of-text drafted by AI or disclosure compliance among economists specifically.
- ESB results are not yet published.
- The IZA paper and the central-bank figure are from snippets or secondhand sources only.
