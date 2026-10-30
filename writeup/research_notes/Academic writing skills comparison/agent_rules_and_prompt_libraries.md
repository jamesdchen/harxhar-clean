# Writing rules files, prompt libraries and anti-slop lists in other agent ecosystems, compared with the paper-prose skill

Researched 29 Oct 2026. Compared against `.claude/skills/paper-prose/SKILL.md`, `writeup/style/voice.md`, and the pattern files AP-V, AP-S, V5, V6 and V12. GitHub files were read through the authenticated `gh api` on 29 Oct 2026. Web pages were read with a fetch tool that paraphrases, so the quotes from web pages are short. Detector-evasion and marker-frequency evidence (Kobak, Liang, Reinhart, Russell et al., Antislop) is already covered in `writeup/research_notes/AI layered paper writing system/llm_prose_markers.md` (Q1, Q4), and OUP's AI policy in `.../economist_workflows_policies.md`. Those notes are cross-referenced here, not repeated.

## Q1. How do Cursor rules, Copilot instructions, Aider conventions, AGENTS.md and similar files encode prose and LaTeX guidance? Public examples for papers

### Takeaway
All the ecosystems use plain Markdown instructions. The only structure is scoping: Cursor `globs`, Copilot `applyTo`, nested AGENTS.md files and Aider read-only files. None of them has severity levels, rule ids, tests or suppressions. Most rules files in paper repositories cover the build, the data and LaTeX mechanics, not prose. The serious prose rule sets for papers come from a few academics (Sant'Anna, Zhao, Luo, Boeing). They add the mechanisms the formats lack: hooks, read-only auditors, severity grades and measured "house style". Studies of coding agents find that rules files help less than assumed, that LLM-written rules files can hurt, and that negative constraints work better than positive style directives.

### Cited Findings

**Formats (official docs)**
- **Cursor.** Project rules live in `.cursor/rules` and need the `.mdc` extension; a plain `.md` file there is ignored. The frontmatter has three fields: `description`, `globs`, `alwaysApply`. The four rule types are Always Apply, Apply Intelligently (the agent decides from the description), Apply to Specific Files (globs) and Apply Manually (@-mention). `AGENTS.md` is a frontmatter-free alternative and can be nested; the more specific file wins. Best practices, verbatim: "Keep rules under 500 lines"; "Split large rules into multiple, composable rules"; "Provide concrete examples or referenced files"; "Avoid vague guidance. Write rules like clear internal docs"; "Reference files instead of copying their contents." The docs also advise adding a rule only when the Agent keeps making the same mistake. — [Cursor docs: Rules](https://cursor.com/docs/context/rules)
- **GitHub Copilot.** Repository-wide rules go in `.github/copilot-instructions.md`. Path-specific rules go in `.github/instructions/NAME.instructions.md` with frontmatter `applyTo: "app/models/**/*.rb"`, and both sets apply when both match. `excludeAgent: "code-review"` or `"cloud-agent"` hides a file from that feature. Copilot also reads `AGENTS.md` (nearest file wins) and a root `CLAUDE.md` or `GEMINI.md`. Guidance: "Whenever possible, try to avoid providing conflicting sets of instructions." The docs' own prompt for generating the file says instructions "must be no longer than 2 pages" and "must not be task specific". — [GitHub Docs: repository custom instructions](https://docs.github.com/en/copilot/how-tos/configure-custom-instructions/add-repository-instructions)
- **Aider.** A `CONVENTIONS.md` loaded with `/read CONVENTIONS.md` or `--read` is read-only and prompt-cached; a `read:` key in `.aider.conf.yml` loads it every time. The docs' example: "Prefer httpx over requests for making http requests." / "Use types everywhere possible." A community conventions repository exists. — [Aider docs: conventions](https://aider.chat/docs/usage/conventions.html)

**Public examples for papers and LaTeX**
- **Notation rules with BAD/GOOD examples (Cursor, control-engineering paper).** `.cursor/rules/latex-paper-writing.mdc` has `globs: **/*.tex`. Rules: "Every symbol that appears in an equation must be described **exactly once**, at its first appearance, using a `where` clause". It has a "Notation Clash Detection" step: before editing equations, check for "Same symbol, different meaning", "Same concept, different symbol" and "Overloaded subscripts", and "**flag them in chat** before making any changes … Ask the user which symbol/convention to keep before proceeding". The file is "the single source of truth for notation", and the agent must update it after every edit that introduces or renames a symbol. Each entry gives a symbol, its meaning and "first defined in `\label{...}`". Examples are marked "✅ GOOD" and "❌ BAD". — [PatrikValabek/offset-free-koopman-mpc, latex-paper-writing.mdc](https://github.com/PatrikValabek/offset-free-koopman-mpc/blob/main/.cursor/rules/latex-paper-writing.mdc)
- **Academic project template (AGENTS.md; Geoff Boeing).** Its "Paper Guidelines" are LaTeX mechanics: "Prefer `\autoref` to `\ref`"; "Use `\citet` and `\citep` … avoid `\cite`"; "Use en dash (`--`) for ranges of values"; "Use em dash (`---`) to offset adjunctive phrases stronger than those in parentheses"; "Run the pre-commit hooks to lint via `chktex`". Its "Consistency" section has the agent "Verify analysis code output matches the paper's reported numbers, tables, figures, equations, and units" and "Verify paper text's claims against the figures/tables' numbers, signs, rankings". Note that this template allows em dashes, which most anti-slop lists ban. — [gboeing/project-template AGENTS.md](https://github.com/gboeing/project-template/blob/main/AGENTS.md)
- **Typical paper-repository Copilot files cover the build, not prose.** Three repositories found by `gh search code` (filename `copilot-instructions.md`, "latex paper"):
  - MDOcean describes its Calkit/DVC pipeline. It has per-paper `paperlint-*` stages that run `dev/latex/Paper-Linter/paperlint.py` with `always_run: true`.
  - brave-new-green-algorithm maps its `.Rnw` papers and says "Do not modify or delete data files."
  - dellaert/fgz gives code style for a TikZ generator.

  None has a prose rule. — [MDOcean copilot-instructions.md](https://github.com/symbiotic-engineering/MDOcean/blob/main/.github/copilot-instructions.md); [MDOcean calkit.yaml](https://github.com/symbiotic-engineering/MDOcean/blob/main/calkit.yaml); [JJ/brave-new-green-algorithm](https://github.com/JJ/brave-new-green-algorithm/blob/main/.github/copilot-instructions.md); [dellaert/fgz](https://github.com/dellaert/fgz/blob/main/.github/copilot-instructions.md)
- **Economist's Claude Code template (Pedro Sant'Anna; 1,674 stars on 29 Oct 2026).** It ships `.claude/rules/`, `.claude/agents/` and `.claude/output-styles/`. The relevant files:
  - **`academic-writing` output style.** "Hedge to the evidence, never more, never less"; "**Every number carries its units, its uncertainty, and its source** (table/equation). No bare point estimates."; "if a number is needed and unknown, write `[CITE]` / `[NUM]` rather than inventing it"; "If a sentence exceeds ~30 words or two clauses of math, split it."
  - **`proofreading-protocol` rule.** "The agent must NEVER apply changes directly. It proposes all changes for review first", in three phases: review and propose, the user approves, apply.
  - **`humanize-auditor` agent.** Read-only, with ten tell categories graded HIGH, MED or LOW (details in Q3 and Q4).
  — [pedrohcgs/claude-code-my-workflow](https://github.com/pedrohcgs/claude-code-my-workflow); [academic-writing.md](https://github.com/pedrohcgs/claude-code-my-workflow/blob/main/.claude/output-styles/academic-writing.md); [proofreading-protocol.md](https://github.com/pedrohcgs/claude-code-my-workflow/blob/main/.claude/rules/proofreading-protocol.md)
- **Social scientist's CLAUDE.md (Eric Luo).** A "Voice" section and a "Research prose" section:
  - "Effects get a magnitude, a direction, and a benchmark. Never 'significantly improves' on its own."
  - "'Significantly' stays when it means statistically significant."
  - "No trailing editorial clauses ('..., ensuring consistency', '..., highlighting the tradeoff'). If the point is worth making it gets its own sentence. If not, delete it."
  - "Commit to claims. Flat statements and superlatives read as human."
  - "Know when to stop cutting. Applied hard, the rules above flatten prose that was already fine. Do not flag or rewrite: an em dash in something I wrote or inside a quotation, formal and academic vocabulary, a genuine three-item enumeration, a long sentence carrying one long argument … When a sentence breaks no rule above, leave it alone."
  - Papers "keep long single-argument sentences … and the field's evaluative vocabulary".

  — [ericluo04/claude-academic-workflow CLAUDE.md](https://github.com/ericluo04/claude-academic-workflow/blob/main/CLAUDE.md)
- **Cross-agent rules with a write-time gate (Yue Zhao, anywhere-agents).** One `AGENTS.md` is turned into `CLAUDE.md` and `agents/codex.md`. Overrides are layered: `CLAUDE.local.md` > `AGENTS.local.md` > generated > upstream. A `PreToolUse` hook (`guard.py`) **denies** a Write or Edit to `.md`/`.tex`/`.rst`/`.txt` whose new text contains a banned AI-tell word, "with the hits and a `Suggested rewrite:` line". The writing defaults include "Revise in place … a paper, plan, brief, or rule file should not grow each round without new content" and "Vary sentence length and structure". — [yzhao062/anywhere-agents AGENTS.md](https://github.com/yzhao062/anywhere-agents/blob/main/AGENTS.md)
- **"The Elements of Agent Style" (Yue Zhao, agent-style; 730 stars, pushed 27 Oct 2026).** It ships "21 writing rules for AI coding and writing agents. Drop-in for Claude Code, Codex, Copilot, Cursor, and Aider": 12 canonical rules (Strunk & White, Orwell, Pinker, Gopen & Swan) and 9 "field-observed" ones. Each rule carries a source citation, "agent-instruction evidence", a severity, a scope, an enforcement tier and "5 or more" BAD → GOOD examples. A "House Style" section is drawn "from Yue Zhao's pre-AI first-author ML papers" and is given as rates, e.g. "About one in four runs past 30 words". — [yzhao062/agent-style README](https://github.com/yzhao062/agent-style); [RULES.md](https://github.com/yzhao062/agent-style/blob/main/RULES.md)

**Evidence on rules files (coding agents; transfer to prose is untested)**
- **Zhang et al. (arXiv 2604.11088, v2 28 May 2026).** They studied 679 rule files holding 25,532 rules and ran over 5,000 Claude Code runs on SWE-bench Verified. Random rules helped as much as expert-curated ones, +13.8 pp on a discriminative subset, which the authors attribute to "context priming". "Every individually beneficial rule was a negative constraint", and every individually harmful one was a positive directive; the example of a harmful rule is "follow code style". Their advice: "constrain what agents must not do, rather than prescribing what they should." Pass rates held steady from 0 to 50 rules. — [arXiv 2604.11088](https://arxiv.org/abs/2604.11088)
- **"Evaluating AGENTS.md" (ETH Zurich / LogicStar.ai, ICLR 2026; arXiv 2602.11988).**
  - Context files gave no gain in task success and raised inference cost by over 20%. One version of the abstract says they lowered success.
  - Developer-written files beat LLM-generated ones but added only about 4% on AGENTbench.
  - Agents followed the instructions well, but repository overviews did not help.
  - The authors recommend minimal human-written files limited to non-standard requirements.

  I read the abstracts and summaries, not the PDF. — [ICLR 2026 page](https://iclr.cc/virtual/2026/10018749); [opentrain summary of arXiv 2602.11988](https://www.opentrain.ai/papers/evaluating-agents-md-are-repository-level-context-files-helpful-for-coding-agent--arxiv-2602.11988/); [arxiviq summary](https://arxiviq.substack.com/p/evaluating-agentsmd-are-repository)

### Inferences
- The rules-file formats are routing layers, much like our skill's description and branch table, and give no further leverage. Where other projects enforce prose rules, they built the enforcement themselves: hooks (anywhere-agents), pipeline lint stages (MDOcean), pre-commit `chktex` (Boeing) and read-only auditor agents (Sant'Anna).
- Two of our rules have direct analogues elsewhere. Boeing's "verify analysis code output matches the paper's reported numbers" is a manual version of our V2 and C2. PatrikValabek's notation clash detection is our V3, plus a "stop and ask before editing" step that our project routes through the deliberate skill.
- Zhang et al.'s polarity result is consistent with our V12, which bans global style targets ("vary sentence length", "read like a journal"). It also suggests that our drafting patterns should state what not to write, which most of them already do (V6, V10, AP-V, AP-S).

### Gaps
- I did not fetch the Windsurf (`.windsurf/rules`) or Cline (`.clinerules`) docs, so their format details (size limits, activation modes) are not verified here.
- I did not open MDOcean's `paperlint.py`, so I do not know what it checks.
- A GitHub code search turned up no `.cursorrules` or `copilot-instructions.md` written for an economics or finance paper.

## Q2. Prompt libraries and publisher or university guidance for academic writing: tasks covered and how

### Takeaway
The popular prompt libraries are task templates (titles, abstracts, introductions, "rewrite in academic language"). They have no rules for numbers, claims or verification, and their one safeguard is a warning about fake references. Publisher guidance covers disclosure and accountability, not style: Wiley exempts language polishing from disclosure, and OUP requires disclosure in the cover letter and in the Methods or Acknowledgements. University writing centres aim at students and stress brainstorming, revision and disclosure. None of these sources gives rules our skill lacks. Their "make it more academic" prompts are exactly the global style targets that V12 forbids.

### Cited Findings
- **ahmetbersoz/chatgpt-prompts-for-academic-writing (5.0k stars).**
  - Categories: Brainstorming; Article Sections (titles, keywords, abstract, outline, introduction, literature review, methodology, experiments, results, discussion, conclusion, future works); Improving Language (rewriting, grammar, style, transitions, proofreading, translation); Summarization; Plan/Presentation; Working With Documents.
  - Prompts, verbatim: "Rewrite this paragraph in an academic language: [PARAGRAPH]"; "Paraphrase the text using more academic and scientific language."; "Generate an abstract for a scientific paper based on this information for: [PARAGRAPHS]"; "Conduct a literature review on [TOPIC SENTENCE] and provide review paper references".
  - Its only caution: "Be careful and double-check article existence. ChatGPT may generate fake references".
  - It has no reviewer-response section and no disclosure guidance.

  — [GitHub: ahmetbersoz/chatgpt-prompts-for-academic-writing](https://github.com/ahmetbersoz/chatgpt-prompts-for-academic-writing)
- **Forks of that library.** hao203 adds a "Literature Review Generator" plus abstract and paraphrase prompts. I found no open GitHub library of reviewer-response prompts; the response-letter templates I found were on commercial SEO pages. — [hao203 fork](https://github.com/hao203/chatgpt-prompts-for-academic-writing); [Techpresso prompt library](https://academy.techpresso.co/prompts/chatgpt-prompts-academic-writing)
- **Wiley (researcher AI guidelines, late Oct 2025).**
  - Spelling, grammar and general editing need no disclosure. "Substantial" AI editing, development or translation must be disclosed in the Methods, a disclosure statement or the Acknowledgements.
  - Authors should document the tool's purpose, whether it influenced key arguments or conclusions, and how they verified its output.
  - Generative AI may not create or alter research data or results.
  - AI use alone should not cause automatic rejection.
  - Prompt-engineering tips appear in Wiley's book-author guidance, not in the research guidance.

  — [Wiley press release](https://newsroom.wiley.com/press-releases/press-release-details/2025/New-Wiley-Guidelines-Give-Researchers-Clear-Path-Forward-in-Responsible-AI-Use/default.aspx); [Research Information summary](https://www.researchinformation.info/news/wiley-guidelines-give-researchers-clear-path-forward-in-ai-use) (summary details are secondary)
- **OUP (JFEc's publisher).** The journals policy was last updated 13 July 2026. AI use "must be disclosed both in cover letters to editors and in the Methods or Acknowledgements section of manuscripts"; full sourcing is in the existing note. Secondary summaries (Enago) mix in OUP's books guidance. — see `writeup/research_notes/AI layered paper writing system/economist_workflows_policies.md` lines 171–221; [OUP journals AI policy](https://academic.oup.com/pages/for-authors/journals/preparing-and-submitting-your-manuscript/ai-principles-for-oxford-journals/policy-on-ai-use-and-disclosure-for-oxford-journals-authors)
- **University writing centres.** Missouri State treats AI as useful for brainstorming and revision but not as a replacement for drafting, and asks students to revise AI output. Metro State coaches prompt revision and spotting hallucinated citations. UNC's five-part guide (Overview, Getting Started, Drafting, Revising, Citing) is known to me only through a 2023 blog summary. — [Missouri State Writing Center](https://writingcenter.missouristate.edu/resources-for-ai-writing-programs.htm); [Metro State GenAI writing support](https://www.metrostate.edu/academics/success/tutoring/resources/genai-writing-support); [CWCAACCR blog on UNC guides](https://cwcaaccr.com/2023/08/30/productive-and-ethical-)

### Inferences
- A prompt library is a set of entry points: one prompt per task, no constraints. Our skill's branches A–G are also task entry points, but each carries constraints (V2, V4, V5, V13) and a self-check. Reviewer responses are a task the libraries rarely cover. Our skill has no branch for them either (branch G is critic review, not a response letter).
- "Rewrite in an academic voice" is the global style instruction that V12 and §8.6 ban. The libraries are evidence of what not to give a drafting subagent.
- Wiley's threshold puts our scrub (S2–S8 on locked layers) in "substantial editing" territory, not "language polishing", because it rewrites sentences. Disclosure therefore applies under both Wiley's and OUP's wording.

### Gaps
- I did not read Elsevier's or Springer Nature's author AI pages directly in this round. Springer Nature's ban on generative AI creating or modifying research data is known to me only through a library summary. — [Reichman University library summary](https://www.runi.ac.il/en/library/library-services/faculty/artificial-intelligence-guidelines-from-publishers)
- I found no publisher prompt guide for journal articles (as opposed to books).

## Q3. Anti-slop lists, AI-tell catalogs and humanizer prompts: what they contain, overlap with AP-V and AP-S, and evidence on quality and detector evasion

### Takeaway
Most public anti-slop tools derive from Wikipedia's "Signs of AI writing", a descriptive catalog that warns against treating its signs as problems in themselves. The most popular skills (blader/humanizer, stop-slop) turn it into prescriptive rewriting and add positive targets: "vary sentence length", "sounds human". A trade press article presented the humanizer as a detector-evasion tool. Our AP-V1 and AP-S lists overlap most of these on the core items: delve, underscore, showcase, pivotal, intricate, em dashes, participial "-ing" tails, "not X but Y", signposting, summary openers. What we have and they lack: a field whitelist, rate-based flags for weak signals, and an explicit non-evasion goal. The only quantitative evaluation of such a rule pack (agent-style) measures mechanical tell density only and says it is not evidence of quality.

### Cited Findings

**Wikipedia: Signs of AI writing (WikiProject AI Cleanup)**
- **Sections:** Caveats; Content; Language and grammar; Style; Communication intended for the user; Markup; Citations; Comment-specific; Edit summaries; Miscellaneous; Signs of human writing; Ineffective indicators; Historical indicators. — [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)
- **Caveats.** The patterns are "only potential signs of a problem, not the problem itself". Editors should not just fix surface signs, because doing so can make detection harder. The page is descriptive, not prescriptive, and is not a ban on words: one word can be coincidence, and overuse of a word does not mean its synonyms are overused. — [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)
- **Word lists by model era.** The fetch tool rendered the opening words of each list, verbatim: 2023–mid-2024 "Additionally, boasts, bolstered, crucial, delve"; mid-2024–mid-2025 "align with, bolstered, crucial, emphasizing"; mid-2025–mid-2026 "emphasizing, enhance, highlighting, showcasing"; mid-2026 onward "challenge, clearer, dependable, echoed". The page calls these rough cutoffs. — [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)
- **"Ineffective indicators"** (verbatim headings): "Perfect grammar"; "Mixed registers"; "'Bland' or 'robotic' prose"; "'Fancy', 'academic', or 'formal' prose" ("The correlation does not extend to all formal, academic, or 'fancy'-sounding prose"); "Transition words in isolation" ("This pattern also has precedence in essay-like writing by humans"); "Unsourced content". The em dash is "most useful when taken in combination with other indicators, not by itself", and the newest models are "hard coded to not use em dashes at all". — [Wikipedia: Signs of AI writing (later part of page)](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)

**blader/humanizer (Claude Code skill; v3.1.0, MIT)**
- **Description:** "Rewrite AI-sounding text so it reads like the writer without changing what it says."
- **Patterns.** 26 numbered patterns in six categories: A "Staging instead of stating" (1 Not X but Y; 2 One-line closers and dramatic fragments; 3 Sayings that sound deep; 4 Staged run-up; 5 Arguing with no one); B "Rhythm by rule" (6 Forced triads; 7 Repeated sentence openings; 8 Dashes; 9 Stacked qualifiers; 10 Hyphenated pairs; 11 Passive voice and missing subjects); C "Inflation and borrowed authority" (12 Overused AI words; 13 Inflated significance; 14 Vague connection; 15 Shallow -ing riders; 16 Sales language; 17 Borrowed authority; 18 Avoiding is/are/has); D Formatting (bold, headings, curly quotes); E chat leftovers; F "Re-explaining what the reader knows". Patterns are marked strong or "weak alone"; §1–§5 justify an edit on a single sighting.
- **Rules:** "Treat an unsupported addition as an error, and a lost claim as an error unless a pattern calls for cutting it." "A user's writing sample overrides the patterns, including the dash rule." "Keep the fact and drop the significance."
- **Process:** step 4 tells the model to "vary sentence length". The file ends with a "When not to act" section and makes no claim to evade detectors.

— [blader/humanizer SKILL.md](https://github.com/blader/humanizer/blob/main/SKILL.md)
- **Evasion framing in the press (OSFY, 22 Jan 2026).** OSFY described Humanizer as a plugin that "tells the model to avoid the traits that detectors use", built on 24 patterns from WikiProject AI Cleanup. Its author, Siqi Chen, posted: "It's really handy that Wikipedia went and collated a detailed list of 'signs of AI writing.'" The repository gained over 1,600 stars within days. — [Open Source For You](https://www.opensourceforu.com/?p=96392)

**hardikpandya/stop-slop**
- **Description:** "Remove AI writing patterns from prose. Use when drafting, editing, or reviewing text to eliminate predictable AI tells."
- **Core rules,** verbatim: "Cut filler phrases." "Use active voice." "Be specific." "Vary rhythm." "No em dashes." "Trust readers." "Cut quotables."
- **Scoring:** five dimensions rated 1–10 (Directness, Rhythm, Trust, "Authenticity: Sounds human?", Density), with "Below 35/50: revise."
- **`references/phrases.md`:** eight categories, among them "Emphasis Crutches" ("This matters because", "Let that sink in.") and "Vague Declaratives" ("The implications are significant"). It also has the blanket rule "Kill all adverbs. No -ly words. No softeners, no intensifiers, no hedges."
- **Missing:** no statement on preserving meaning or on technical-writing exceptions.

— [stop-slop SKILL.md](https://github.com/hardikpandya/stop-slop/blob/main/SKILL.md); [stop-slop references/phrases.md](https://github.com/hardikpandya/stop-slop/blob/main/references/phrases.md)

**Word lists in academic rules files**
- **anywhere-agents.** Its default AI-tell list has about 45 words, denied at write time: "encompass, burgeoning, pivotal, realm, keen, adept, endeavor, uphold, imperative, profound, ponder, cultivate, hone, delve, embrace, pave, embark, monumental, scrutinize, vast, versatile, paramount, foster, necessitates, provenance, multifaceted, nuance, obliterate, articulate, acquire, underpin, underscore, harmonize, garner, undermine, gauge, facet, bolster, groundbreaking, game-changing, reimagine, turnkey, intricate, trailblazing, unprecedented". The file calls it "a default AI-tell list; trim or extend in your fork". — [anywhere-agents AGENTS.md](https://github.com/yzhao062/anywhere-agents/blob/main/AGENTS.md)
- **Eric Luo.** Banned words: "delve, tapestry, testament, underscore, showcase, pivotal, crucial, seamless, meticulous, boasts, load-bearing, smoke test". It also cuts "editorializing adverbs as a category: notably, importantly, effectively, carefully, particularly, crucially." — [ericluo04 CLAUDE.md](https://github.com/ericluo04/claude-academic-workflow/blob/main/CLAUDE.md)
- **Sant'Anna `humanize-auditor`.** Ten categories:
  1. boilerplate transitions ("Moreover", "Additionally", "It is worth noting that", "Notably", "In summary")
  2. AI-cliché lexicon ("delve into", "tapestry of", "play a crucial/pivotal role", "shed light on", "underscore the importance", and "leverage" (as a verb, in non-finance / non-engineering contexts))
  3. em-dash overuse (> 3 per paragraph)
  4. symmetric paragraph shapes
  5. tricolon abuse
  6. hedging stacking
  7. "not only X but also Y"
  8. formulaic openers ("calibrate to discipline (AER abstracts rarely use it; APSR often does)")
  9. hyphenation excess
  10. sycophancy ("Our novel approach")

  — [humanize-auditor.md](https://github.com/pedrohcgs/claude-code-my-workflow/blob/main/.claude/agents/humanize-auditor.md)

**Vale packages for AI tells**
- **AiTells (krishnasunkam/vale-ai-tells; v1.0.0, 25 Aug 2026).** 17 rules: 6 at error level (CodeToken, Dash, FilePath, NeverTag, SectionCode, StatusBracket) and 11 at suggestion level (AbstractTriad, Adverb, Cliche, CopulaInflation, EpigramContrast, HiddenVerb, LinkText, NegParallel, Passive, VirtueHonest, WeakResume). Messages name the fix, e.g. "Em/en dash: recast with a period, comma, colon, or 'to' for a range." and "Abstract-noun triad '%s': name one concrete thing instead of three abstractions." The rules are called "benchmarked", but the page gives no corpus or precision figures. — [Vale package explorer: AiTells](https://vale.sh/explorer/AiTells)
- **Other packages.** `ai-tells` and jdkato's `Voices` (with a no-ai-slop set) also exist; I know them only through listings. — [Vale explorer: ai-tells](https://vale.sh/explorer/ai-tells); [Vale blog: voices](https://vale.sh/blog/voices); [jdkato/voices](https://github.com/jdkato/voices)

**Evidence on quality and on detectors**
- **agent-style's own bench.** 10 prose tasks, 2 generations per condition, with and without the rules loaded. Mechanical violations fell 47% for Claude Opus 4.7 (88 → 47), 46% for GPT-5.4 (48 → 26) and 86% for Gemini 3 Flash (65 → 9). Copilot CLI showed no change (52 → 52), treated as a noise control. Limits the authors state:
  - Scoring covered "7 of the 21 rules".
  - Three rules carried most of the movement: sentence length, em/en dashes and the jargon list.
  - "Both **critical**-severity rules are measured by nothing here."
  - The numbers "are not evidence about argument quality, clarity for a stated audience, or sourcing."

  — [agent-style README, "Does It Work?"](https://github.com/yzhao062/agent-style)
- **Detector evidence (vendor and secondhand).** Pangram reports 97% accuracy on humanized text against GPTZero's 46%; these are vendor figures. I found no independent test of the Humanizer skill against detectors. — [Pangram blog: humanizers](https://www.pangram.com/blog/humanizers-aug-25); [Pangram announcement](https://www.pangram.com/blog/humanizers-announcement)
- **Expert readers and humanized text (existing note).** Experts still caught humanized text, and the share of their explanations citing vocabulary fell from 57.1% to 42.3%. — `llm_prose_markers.md` Q4; [arXiv 2501.15654v2](https://arxiv.org/html/2501.15654v2)

**Overlap of the external lists with our catalogs**

| Our id | Item | Also flagged by |
|---|---|---|
| AP-V1 | delve, underscore, showcase, pivotal, intricate, meticulous, realm, tapestry, testament, multifaceted, garner | anywhere-agents (delve, underscore, pivotal, realm, intricate, multifaceted, garner); Luo (delve, tapestry, testament, underscore, showcase, pivotal, meticulous); Sant'Anna (delve, tapestry, pivotal, underscore); Wikipedia era lists (delve, showcasing) |
| AP-V2 | additionally, notably, crucial, comprehensive, insights, enhance, highlighting | Wikipedia era lists (Additionally, crucial, enhance, highlighting); Sant'Anna cat. 1 ("Additionally", "Notably"); Luo ("crucial", "notably") |
| AP-V3 | utilize; adjectives on own results | agent-style RULE-06 ("utilize" → "use"); Luo ("used not utilized"); Sant'Anna cat. 10 ("Our novel approach") |
| AP-S1 | em dashes | humanizer §8; stop-slop; anywhere-agents; Luo; AiTells.Dash (error); agent-style RULE-B. **Allowed** by Boeing's template |
| AP-S2 | participial tails ", highlighting …" | humanizer §15 "Shallow -ing riders"; Luo "No trailing editorial clauses" |
| AP-S3 / V6 | clipped verdict tails | humanizer §2 "One-line closers and dramatic fragments"; stop-slop "Cut quotables". No external rule targets the *quantifier or verdict* tail as such |
| AP-S4 | not X but Y; rather than | humanizer §1; agent-style RULE-07; anywhere-agents; Luo ("X rather than Y"); Sant'Anna cat. 7; AiTells.NegParallel |
| AP-S5 | signposting ("it is worth noting") | Sant'Anna cat. 1; stop-slop ("It's worth noting") |
| AP-S6 | summary openers outside the conclusion | Sant'Anna cat. 1 ("In summary"); agent-style RULE-E; humanizer §2 |
| AP-S7 | serves as, stands as | humanizer §18; Luo ("is/has, not 'serves as'"); AiTells.CopulaInflation |
| AP-S8 | triads | humanizer §6; Luo; Sant'Anna cat. 5; AiTells.AbstractTriad |
| AP-S10 | importance assertions; "This matters because" | humanizer §13; stop-slop "This matters because"; Sant'Anna cat. 10 |
| V5 boosters | clearly, crucially, importantly … | Luo (cut editorializing adverbs); stop-slop (all adverbs); Sant'Anna ("Cut intensifiers") |

### Inferences
- Our AP-V1 list (11 words) sits inside the consensus core. The external lists are longer and include words that are ordinary in econometrics prose: "undermine", "gauge", "acquire", "unprecedented", "nuance", "provenance" (anywhere-agents), and "leverage" → "use" (agent-style RULE-06). They have no field whitelist. Sant'Anna's "(in non-finance contexts)" and Luo's "'Significantly' stays when it means statistically significant" are the only exceptions I found. Our whitelist (leverage effect, robust standard errors, significant) is ahead of practice.
- Wikipedia's "ineffective indicators" support our design choice that AP-V2 words are density flags, not bans ("Transition words in isolation" have precedent in human essays). They also suggest that a formal academic register should not itself be scrubbed.
- The humanizer and stop-slop both pair their removal lists with positive targets: "vary sentence length", "Vary rhythm", "Authenticity: Sounds human?". Those targets point at detectors, not at information, and V12 rejects them.
- V6 (clipped verdict tails that hide an unchecked universal) and V4/V5 (verdict words backed by predicates) have no counterpart in any external list. The external lists judge how a sentence sounds; ours also check whether it is true.

### Gaps
- I found no independent study that measures whether a humanizer or anti-slop skill improves human-rated quality of academic prose. agent-style's bench is mechanical only.
- The fetch tool returned only the first words of each Wikipedia era list. I did not confirm that "multifaceted" is on the page's mid-2026 list, as AP-V1's source column says.

## Q4. Mechanisms that work better than a checklist: linters, severity, rule ids and suppressions, tests, CI; and what to avoid

### Takeaway
Mature prose tooling encodes each rule as a small file with an id, a severity that maps to an action, a message naming the fix, scoped matching, inline suppression by id, and, since Vale v3.24, unit tests that must fire and must not fire. Agent projects add write-time hooks and tiered enforcement: hook → linter → self-check → cross-model review. Vale does not lint LaTeX, so LaTeX projects use TeXtidote, chktex or a detex step. The failure modes to avoid are documented: positive style directives, long LLM-written rule files, blanket bans that flatten technical register, and rules that contradict each other across files.

### Cited Findings

**Vale**
- **Rule files.** One YAML file per rule. `extends` picks the check: existence, substitution, occurrence, repetition, consistency, conditional, capitalization, metric, readability, spelling, sequence or script. `level` is `suggestion`, `warning` (the default) or `error`, and "Only errors produce a non-zero exit code". Each rule has a required `message`, an optional `link`, and a `scope`. `occurrence` reports too few or too many matches; `metric` evaluates a formula over counts. Since v3.20 a rule can extend another with `tokens+`/`tokens-`. — [Vale docs: Styles](https://docs.vale.sh/topics/styles)
- **Tests (v3.24+).** "A rule is a small program, and a rule that matches nothing fails silently." `vale test` runs cases stored with the rule. Example:

  ```yaml
  tests:
    - name: flags a hedge
      input: Perhaps we should ship.
      want: |
        1:1:House.Hedging:Consider removing 'Perhaps'.
    - name: leaves code alone
      input: |
        Run `perhaps` first.
      want: ''
  ```

  — [Vale docs (llms-full.txt), "Testing"](https://docs.vale.sh/llms-full.txt)
- **Inline suppression.** In Markdown: `<!-- vale off -->` / `<!-- vale on -->`, `<!-- vale Style.Redundancy = NO -->`, and per-match `<!-- vale Style.Redundancy["ACT test","OTHER"] = NO -->`. `CommentDelimiters` (e.g. `{/*, */}`) adapt this to formats that do not accept HTML comments. `MinAlertLevel` sets the lowest severity reported. — [Vale docs (llms-full.txt), "CommentDelimiters" and Markdown section](https://docs.vale.sh/llms-full.txt)
- **No LaTeX.** The supported markup formats are AsciiDoc, DITA, HTML, Markdown (incl. R Markdown), MDX, MyST, Org, QDoc, Quarto, reStructuredText, Typst and XML. LaTeX is not on the list. — [Vale docs (llms-full.txt), supported formats](https://docs.vale.sh/llms-full.txt)

**Other linters**
- **proselint.** It aims to "aggregate knowledge about best practices in writing" (Garner, Strunk & White, Orwell). Checks have dotted ids (`hedging`, `weasel_words`, `cliches.misc`, `redundancy.ras_syndrome`, `typography.symbols`, `uncomparables` …). `proselint.json` toggles any id at any granularity and supports `per_file_checks` by glob. The current README does not state the precision claim of earlier versions. — [amperser/proselint](https://github.com/amperser/proselint)
- **TeXtidote.** "Spelling, grammar and style checking on LaTeX documents" (1,059 stars, pushed 9 Sep 2026). It passes the text to LanguageTool but "keeps track of the relative position" so that warnings map back to the `.tex` source. Each warning shows its rule id in brackets; `--ignore sh:001` suppresses a rule; ignore begin/end comments exclude regions; it follows `\input` and `\include`. — [sylvainhalle/textidote](https://github.com/sylvainhalle/textidote)
- **Write-good** keeps passive-voice warnings plus ThereIs and Weasel rules. Running several packages flags the same issue more than once, so one guide advises keeping one package per issue. (This is a third-party guide, not official docs.) — [Writing Articles with Vale](https://thedroneely.com/posts/writing-with-vale/)

**Agent-side enforcement**
- **anywhere-agents.** The `guard.py` PreToolUse hook denies a write when a banned word appears. A second "agent-style advisory" gate is "advisory only: up to 5 findings to the model and the user, no permission decision". `AGENT_STYLE_HOOK=off` is the escape hatch: "Use the narrowest escape only for a legitimate write that quotes a banned word as an example (a style guide, a CHANGELOG entry), and remove the override after the write." Text an agent carries rather than writes (dispatch prompts, captured reviews) goes under an exempt `agent-io` path. — [anywhere-agents AGENTS.md](https://github.com/yzhao062/anywhere-agents/blob/main/AGENTS.md)
- **agent-style severity rubric.** "critical — reader cannot understand or trust the prose"; "high — externally visible AI-tell"; "medium — local readability cost"; "low — polish or preference".
- **agent-style enforcement tiers.** "Tier-1 `ask`" (hook regex, e.g. "sentence > 30 words; suggest split"), "Tier-2" (ProseLint or a metric such as the variance of sentence length), "Tier-3 agent self-check", "Tier-4 Codex review" (cross-model). Judgment rules (RULE-07 positive form, RULE-08 calibration, RULE-11 stress position) are deliberately "Not Tier-1".
- **agent-style escape hatch.** Orwell's Rule 6: "Break any of these rules sooner than say anything outright barbarous". It adds: "A draft that obeys every default in every sentence reads as machine-made." Only a few rules are firm: "do not invent citations, evidence, or numbers, and do not leave placeholders in finished text".

  — [agent-style RULES.md](https://github.com/yzhao062/agent-style/blob/main/RULES.md); [README](https://github.com/yzhao062/agent-style)
- **Sant'Anna `humanize-auditor`.** Severity is tied to density and location: "HIGH if > 1 per 1000 words; MED if ~1 per 2000 words" for transitions; cliché words are HIGH "on first three pages (abstract, intro, opening of methods)". The auditor has a "Boundary": "You do NOT review grammar (that's `proofreader`) … substance … (that's `domain-referee`) … You do NOT rewrite — you flag." Its report is a per-category HIGH/MED/LOW table plus line-level findings with a suggested fix. — [humanize-auditor.md](https://github.com/pedrohcgs/claude-code-my-workflow/blob/main/.claude/agents/humanize-auditor.md)

**CI on prose**
- Boeing runs `chktex` in pre-commit. MDOcean runs a `paperlint` stage per paper with `always_run: true` inside its reproducible pipeline. — [gboeing/project-template AGENTS.md](https://github.com/gboeing/project-template/blob/main/AGENTS.md); [MDOcean calkit.yaml](https://github.com/symbiotic-engineering/MDOcean/blob/main/calkit.yaml)

**Guidance on writing skills and rule files**
- **Anthropic's skill-authoring guide:**
  - "Keep the `SKILL.md` body under 500 lines"; "Keep references one level deep from `SKILL.md`".
  - The `description` has a "Maximum 1,024 characters" and should give "what the Skill does and when to use it".
  - "Avoid time-sensitive information", or put it in an "Old patterns" section.
  - "Use consistent terminology."
  - "Create evaluations BEFORE writing extensive documentation"; "At least three evaluations".
  - "Run validator → fix errors → repeat"; "Prefer scripts for deterministic operations".

  — [Claude Docs: Skill authoring best practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices)

**Documented failure modes**
- **Positive directives.** Positive directives were the harmful rules in Zhang et al. (Q1). — [arXiv 2604.11088](https://arxiv.org/abs/2604.11088)
- **Long LLM-written context files** added cost and gave no gain. — [ICLR 2026, Evaluating AGENTS.md](https://iclr.cc/virtual/2026/10018749)
- **Flattening.**
  - stop-slop: "Kill all adverbs".
  - agent-style RULE-06: substitutes "leverage" → "use".
  - agent-style RULE-G: "Use Title Case for Section and Subsection Headings".
  - Luo's file names the risk: "Applied hard, the rules above flatten prose that was already fine." — [stop-slop phrases.md](https://github.com/hardikpandya/stop-slop/blob/main/references/phrases.md); [agent-style RULES.md](https://github.com/yzhao062/agent-style/blob/main/RULES.md); [ericluo04 CLAUDE.md](https://github.com/ericluo04/claude-academic-workflow/blob/main/CLAUDE.md)
- **Contradictions across files.** Boeing's template prescribes em dashes (`---`) for strong asides, while every anti-slop list bans them. Copilot warns: "avoid providing conflicting sets of instructions." — [gboeing AGENTS.md](https://github.com/gboeing/project-template/blob/main/AGENTS.md); [GitHub Docs](https://docs.github.com/en/copilot/how-tos/configure-custom-instructions/add-repository-instructions)

### Inferences
- Vale's model maps onto our catalog almost one to one. Pattern id ↔ rule name; Detection grep ↔ `tokens`; hard flag / density flag / review flag ↔ `error` / `occurrence`-or-`metric` / `suggestion`; Source ↔ `link`. Because Vale cannot read `.tex`, the planned `scrub.py` (S1 detex, then S2 and S3) is the right host. It should copy Vale's conventions: ids, levels, messages naming the fix, and per-rule must-fire and must-not-fire tests.
- TeXtidote's position mapping solves a problem our S1 detex step will meet: reporting a flag at the `.tex` line after macros are stripped.
- anywhere-agents learned that a deny hook needs an exemption for files that quote banned words. Our pattern files and research notes quote every AP-V word, so a hook on `writeup/**` would block them unless they are exempt.

### Gaps
- I found no measurement of false-positive rates for any AI-tell Vale package on academic or economics prose.
- I did not test TeXtidote or chktex on our `.tex` files.

## Q5. Comparison with the paper-prose skill: what others do better, what to borrow, what to avoid

### Takeaway
On content, our skill is ahead of everything public I found. Numbers from macros (V2), verdict words from predicates (V4), Rule NT (V5), clipped-tail checks (V6), author text kept verbatim (V13), no polish before lock (V14), a field whitelist, and named operations instead of global targets (V12) have at most partial analogues elsewhere. On mechanism, it is behind: the skill itself says "Most automatic checks … are not built yet, so this checklist is the check." Others have deny or advise hooks, severity levels tied to actions, inline suppression by id, rule tests, and a measured (if limited) evaluation. Borrow the mechanisms. Avoid the positive "sound human" and "vary rhythm" targets and the long generic word lists.

### Cited Findings
- **Our skill's own statements:**
  - SKILL.md says "Most automatic checks (C2, C5, C18, C25, C26, `scrub.py`) are not built yet, so this checklist is the check."
  - AP-S states "The corpus does not exist yet (§8.9 Q4), so there is no band to count against."
  - V12 bans "use more passives", "vary sentence length" and "be accurate about scope" as instructions.

  — `.claude/skills/paper-prose/SKILL.md` line 93–94; `writeup/style/patterns/anti-patterns/AP-S-structural-markers.md`; `writeup/style/patterns/sentences/V12-named-sentence-operations.md`
- **Size and format.** SKILL.md is 103 lines and routes to 18 pattern files (1,094 lines in total) by id, one level deep. Its `description` is 1,060 characters (measured 29 Oct 2026), above the 1,024-character maximum in Anthropic's skill-authoring guide. — local measurement; [Claude Docs: best practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices)
- **Dated state in SKILL.md.** Branch F says "(On 29 Oct 2026 the file does not exist, so no layer is locked.)". Anthropic's guide says "Avoid time-sensitive information". — SKILL.md line 57–58; [Claude Docs](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices)
- **Closest external analogues of our unique rules:**
  - V2/C2: Boeing, "Verify analysis code output matches the paper's reported numbers"; Sant'Anna, "Every number carries its units, its uncertainty, and its source" and "`[NUM]` rather than inventing it".
  - V4/V5: Luo, "Effects get a magnitude, a direction, and a benchmark"; agent-style RULE-08, "Do Not Linguistically Overstate or Understate Claims Relative to the Evidence" (judgment-only, "Not Tier-1").
  - V13: humanizer, "A user's writing sample overrides the patterns"; Luo, do not flag "an em dash in something I wrote".
  - V3: PatrikValabek's notation file and clash detection.

  — sources as cited in Q1–Q4

### Inferences

**What others do better, and what to borrow** (the stage and check ids are those of the report, §8)

| # | Mechanism (who) | Our gap | Concrete borrow |
|---|---|---|---|
| B1 | Write-time hook: deny on hard flags, advisory on the rest (anywhere-agents) | Checks are manual; S2/S3 not built | A PreToolUse hook on `writeup/rehaul/*.tex` and `writeup/make_*.py` that runs the AP-V1 and AP-S1 greps from the pattern files. Deny or warn on AP-V1/AP-V3; advisory only for AP-S, V5 and V6. It must exempt `writeup/style/**`, `writeup/research_notes/**` and `writeup/reports/**`, which quote the banned words, and it must skip author-text spans (V13/V15) |
| B2 | Rule tests (`vale test`) | Detection greps have no fixtures; the whitelist is untested | Give each pattern file's Detection grep must-flag and must-not-flag cases: "the leverage effect", "robust standard errors", "statistically significant" must not flag AP-V; ", none significantly" must flag V6. Run them in CI before `scrub.py` uses them |
| B3 | Severity mapped to action (Vale levels; agent-style four levels; Sant'Anna HIGH/MED/LOW by density and location) | Flag types are spread across prose (hard, density, review) | Add one `Severity:` line to each pattern header (error = blocks the L4 gate; warning = review flag; suggestion = rate only). Weight by location as Sant'Anna does (abstract, introduction) |
| B4 | Inline suppression by rule id (Vale comments; TeXtidote `--ignore`; proselint.json) | Author text is protected by provenance (V13/V15), with no marker in the file | A LaTeX comment convention such as `% prose-ok AP-S1: author 2026-10-29` that the scrub honours and logs. It fits the existing "dated `%` comment" house rule |
| B5 | Explicit "when not to act" or escape hatch (agent-style, Orwell Rule 6; Luo "Know when to stop cutting"; humanizer) | Spread across V12's band stop and V13–V15 | A short "Do not flag" list in SKILL.md Step 2: the field whitelist, quotations, a genuine three-item list, a long sentence carrying one argument, formal register (Wikipedia: "fancy/academic/formal prose" is an ineffective indicator) |
| B6 | Measured evaluation with stated limits (agent-style bench; Anthropic "Build evaluations first", at least three) | No evaluation of the skill | At least three fixed tasks (a branch B number sentence, a branch C table note, a branch F scrub on a locked fixture), each run with and without the skill and scored by the planned checks (C2, C5, C18, V6 grep, AP-V). Report which patterns the score covers, as agent-style does |
| B7 | House style measured from the author's own pre-AI writing, at rates (agent-style) | Per-section voice waits for the JFEc corpus (§8.9 Q7) | Add the author's own earlier papers or drafts, if any, as a second exemplar source for the S5 bands, recorded as rates, not rules |
| B8 | Separate read-only auditor roles with boundaries (Sant'Anna humanize-auditor, proofreader, referees) | Branch G already says "Do not rewrite" | Keep it. Copy the explicit "Boundary" list (style auditor ≠ claim checker ≠ referee) into branch G so that a critic does not drift into proposing new results (§8.6) |
| B9 | Skill hygiene (Anthropic guide) | The description is 1,060 characters, over the 1,024 maximum; dated state in branch F | Trim the description to under 1,024 characters. Replace the "On 29 Oct 2026 the file does not exist" text with "if `writeup/layer_state.json` is missing, no layer is locked" |

**What we do better** (keep):
- V2 numbers from macros and V4 verdict words from predicates. No public rules file ties wording to computed values.
- V13–V15 author text and provenance. The public skills either rewrite everything (stop-slop) or defer to a "writing sample".
- The field whitelist. External lists ban "leverage", "undermine", "gauge", "unprecedented".
- V12 named operations and no global targets. This is supported by Zhang et al.'s polarity finding and contradicted by humanizer, stop-slop, agent-style RULE-12 and anywhere-agents, which all prescribe "vary sentence length".
- V5 Rule NT tests a word for information content and truth. Luo's "superlatives read as human" and stop-slop's "Kill all adverbs" are aimed at how the text sounds.
- The non-evasion framing. It matches Wikipedia's caution and OUP's disclosure rule, while the humanizer was presented in the press as a detector-evasion tool.

**What to avoid:**
- **"Sounds human" or "authenticity" scores, and "vary rhythm/length" instructions.** These point at detectors and are global targets (stop-slop rubric; humanizer step 4; agent-style RULE-12).
- **Importing a generic 45-word list wholesale.** It would flag domain vocabulary. If AP-V1 grows, regenerate it from the corpus z-scores as AP-V already plans, not from anywhere-agents or stop-slop.
- **Blanket category bans.** "All adverbs", "all -ly words", Title Case headings and "leverage → use" would break JFEc register and V9's conventional titles.
- **Long always-loaded rule text.** The ETH study and Cursor's 500-line advice both argue against it. Our routing skill with one-level pattern files already follows progressive disclosure. Do not move pattern bodies into CLAUDE.md or AGENTS.md.
- **Conflicts across instruction layers.** The user's global CLAUDE.md, any project AGENTS.md and the skill must not disagree on dashes, hedges or banned words; Boeing's template versus the anti-slop lists shows the risk. Copilot's docs warn against "conflicting sets of instructions".

### Gaps
- No public skill or rules file I found targets financial econometrics or JFEc. The closest are Sant'Anna's general econometrics workflow and Luo's social-science CLAUDE.md, so the borrow list is transferred, not field-tested.
- Whether Claude Code truncates or rejects a skill description over 1,024 characters was not tested. The limit comes from the platform docs, and Claude Code's own limit may differ.
- None of the external rule packs reports false-positive rates on academic text, so the cost of borrowing their lists cannot be quantified.
