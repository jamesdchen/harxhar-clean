# Assemble proven parts, lock the author's judgments

Every part of the system this author wants already exists and is proven in some field (§1.5):

- **Requirements traceability** flags downstream items when an upstream decision changes (DOORS Link Validity; Jama and Codebeamer suspect flags).
- **Assurance cases** separate potential from actual impact, and name ripple effects as the hard part.
- **Structured authoring** defines values once and filters layers by condition (DITA keys, conref, DITAVAL include/exclude/flag; legal answer files).
- **Spec-driven AI coding** keeps persistent upper layers above generated code (Spec Kit, Kiro, OpenSpec, Tessl).
- **AI fiction tools** keep fact codices.
- **HCI prototypes** keep outline layers linked to drafts (VISAR, Dang et al.).
- **Living papers** define values once and guard against stale ones (showyourwork).

What no system found does is combine them for a paper. None filters review on whether a claim's *outcome* flipped: they flag whenever an *input* changes. And no published tool round-trips prose both ways with Overleaf.

Among paper-writing systems themselves, the end-to-end research agents (Sakana's AI Scientist, Agent Laboratory, Jr. AI Scientist, data-to-paper) treat outlines as scaffolding and overwrite them. The economists who publish their Claude Code workflows stop at "every number comes from code" plus critics run in fresh contexts. This author's scaffold already holds the paper-side parts: generated number macros, a visible `\rhPending`, two generated review layers with injected labels, and a two-way Overleaf round trip for the main file. Most of the remaining work is **gates, not generation**, and most gates can be copied from those other fields. The evidence and the author's binding decisions (A1–A7, boxed below) point to seven additions:

1. **Close the number pipeline the way data-to-paper did.** Add a hard lint on typed digits, staleness stamps on generated files, and generator-emitted predicates for direction words. Outline-first generation raises coherence by 14 to 28 points, but it leaves verifiability unchanged (STORM: 3.80 vs 3.85).
2. **Render the final paper from the bullets with the scaffolding stripped, rather than having an LLM expand them.** Expansion is where unsupported connective claims enter, and bullet-to-prose expansion is how researchers synthesize "LLM-modified" text.
3. **Calibrate the prose scrub against a JFEc/JoE/JBES corpus, and run it last.** Let an LLM touch only register, through named sentence-level operations rather than global style targets, and only on content the author has already locked. The current live text already has no em dashes and no rare marker words. Every universal and superlative must pass Rule NT (§4.3.7): delete it if the context already makes it valid, even when it is true; if it adds information, it must match a computed fact.
4. **Lock the author's decisions instead of reviewing edits.** Each judgment at any layer (story, design, presentation, beat) is stored once as data with machine parameters. Generators compute from those parameters, scope wording prints from them, and the build checks every number and comparison against them. A changed decision therefore propagates to every lower layer, and the author sees only the claims it flips or weakens (§8.5).
5. **Build the paper top-down from a simple, unedited state.** The author reads the plainest version of each new layer once, as a PDF on Overleaf, and sends edits as messages.
   - An agent applies each edit at the layer it targets, records it as a lock with the author's words verbatim, and pushes so the PDF updates.
   - Only then is the next layer generated.
   - Polishing comes only after the content is locked, and it never alters text the author supplied.
   - Collaborators' edits on Overleaf flow back through the pull and are recorded too, but they lock only when the author adopts them (§8.5).
6. **Fetch BibTeX deterministically.** Search-enabled frontier models got only 50.9% of entries fully right.
7. **Generate OUP's mandatory AI disclosure from the git trailers.** OUP requires the disclosure in both the cover letter and the Acknowledgements, with tool name and version, and the trailers already record which Claude model touched each commit.

> **Author decisions (binding).** These override any inference in this report that conflicts with them.
>
> **A1 (28 Oct 2026): no review cards.** "no cards. the content that I need to review that you proposed is way too much. There needs to essentially be a way for a human's judgement at any top layer to trickle down and stay locked in for every lower layer. For example, if I decide that the main story is to compare on every RTH bar over 20 years of sample, that decision needs to be reflected everywhere those results are reported or used to back up any claim or compared against, etc."
> *Effect:* §8.5 is now decision locks. The card inferences in §5 are superseded and kept as research; the card design is in commit 3c87fa6.
>
> **A2 (28 Oct 2026): from simple to finished, through the author's corrections.** "thus, the paper needs to start from a simple unedited state and, through my reading and correction of the simple version, end in a finished edited state."
> *Effect:* layers are generated top-down, each from a locked parent. The author reads the simple version of each new layer once and sends corrections, which become locks; A3 sets how corrections are captured. Scaffold removal, typography and the scrub run last and may not change locked content (§8.1, §8.4, §8.5, §8.7).
>
> **A3 (28 Oct 2026): corrections arrive as messages, not Overleaf edits.** "but I don't correct it in place. I look at the pdf and I basically go through and send you edits. then you use the overleaf git to update what I'm looking at"
> *Effect:* amends A2's capture mechanism, not its direction. For the author, Overleaf is where the PDF is read. Corrections arrive as chat messages phrased against the PDF. The agent locates each edit in the source, applies it at the layer it targets, records it with the author's words verbatim, propagates it, and pushes so the PDF updates (§8.5). Amended by A5: Overleaf edits still flow back.
>
> **A4 (28 Oct 2026): author text is never scrubbed.** Asked whether Rule NT applies to text the author typed, the author answered: "no".
> *Effect:* Rule NT and the rest of the scrub never alter text the author supplied in an edit message. That text is protected verbatim, and the protection travels with it into lower layers and the final build (§4.3.7, §8.3 C18 and C26, §8.4, §8.6).
>
> **A5 (28 Oct 2026): Overleaf edits still flow back.** "but other people will collaborate on the overleaf. so edits do still need to have the option of flowing back from overleaf as well"
> *Effect:* amends A3. There are two input channels: the author's chat edits (A3), and edits made directly on Overleaf by collaborators (or on occasion the author), which flow back into the sources, review layers included. Under the proposed default, collaborator edits are applied as text and recorded with their source, lock only when the author adopts them, and are held when they would break a lock. Their authority, their protection under A4 and tracked changes are open questions (§8.5, §8.9 Q10–Q12).
>
> **A6 (29 Oct 2026): questions come in tiers, after deliberation.** "we may need a tier system for questions if we ever make parts of it more automatic. plenty of tokens should be burned arriving at the most likely correct conclusions before every coming to me with any questions"
> *Effect:* every open question is routed through a tier system (T0 mechanical, T1 evidence-resolvable, T2 inferred preference, T3 author judgment). Only residual T3 items reach the author, after a saved multi-solver deliberation, batched, each with a recommendation and a default (§8.10).
>
> **A7 (29 Oct 2026): escalation is encoded.** "Working through the questions should be somehow encoded in the process. There needs to be a system of escalation where only the important stuff reaches me."
> *Effect:* the tiers, thresholds, hard triggers and importance filter are code and protocol (`writeup/deliberation/`, commit 930958b). `PROTOCOL.md` governs wherever it and this report differ (§8.10).

The smallest useful first step is a single standard-library check that fails the build on any digit typed into live text.

**How to use this document.** Sections are numbered so design decisions can cite them (for example "per §3.2").

- **Sections 1–7:** plain paragraphs report sourced evidence with inline links. Paragraphs that open with ***Inference.*** are this report's reasoning. "(repo)" marks facts read from this repository on 2026-10-28.
- **Section 8:** the recommended design. Its tables carry a **Basis** column:
  - **D**: a study or primary document tests or mandates the mechanism. The field is named when it is not finance.
  - **T-code**: transferred from software code-review evidence.
  - **T-genre**: transferred from studies of other text genres (stories, encyclopedic articles, ML papers, emails, student essays).
  - **P**: prescriptive guidance (style guides, editorial rules, pragmatics) with no measured effect.
  - **I**: inferred from this repository's constraints and untested anywhere.
- **Author decisions** (A1–A7) are binding. Where an earlier inference conflicts with one, the inference is marked superseded and kept as research.

The sixteen source notes are in `writeup/research_notes/AI layered paper writing system/`. Claims the notes mark as secondary or unverified carry that flag here too.

---

## 1. Paper agents discard their outlines; the layered parts exist elsewhere

### 1.1 End-to-end agents fill templates, then critique a fixed number of times

The ML-oriented research agents share one shape. They fill a conference LaTeX template, then run a **fixed budget of one to three critique rounds** plus mechanical LaTeX checks.

**Sakana AI Scientist v1.** It drafts each section with one refinement pass. It first sketches Related Work only as LaTeX `%` comments, runs up to 20 Semantic Scholar citation rounds, and then checks that every `\cite` key exists, every figure file exists, chktex passes and the document compiles. **Numbers are not verified against logs in code.** That rests on the prompt "Do not hallucinate results that don't exist" ([AI Scientist v1 perform_writeup.py](https://raw.githubusercontent.com/SakanaAI/AI-Scientist/main/ai_scientist/perform_writeup.py)).

**AI Scientist v2.** It replaced per-section writing with a single pass by a reasoning model, followed by up to three whole-document reflections and a vision-model review of figures ([Yamada et al. 2025](https://arxiv.org/html/2504.08066)).

**Jr. AI Scientist.** Its writer is Claude Code, and it went back to staged drafting: Method first, then a structural outline, then the full paper, with Related Work last. The reason given is that "generating the whole paper at once degraded the Method section". Converting ablation results to LaTeX tables by code "significantly reduced numerical transcription errors" ([Jr. AI Scientist](https://arxiv.org/html/2511.04583)).

**data-to-paper.** The Kishony lab's system is the outlier.

- Each writing step sees only a whitelisted set of "provided prior products".
- Every product passes rule-based review before any LLM review.
- Every number the writer types must hyperlink to a labelled value from the analysis output.
- A missing value written as `[unknown]` aborts the whole run.

Its authors report reliable manuscripts in **80–90% of simple-goal runs**, and found that two or three one-sentence human comments fixed complex ones ([data-to-paper](https://arxiv.org/pdf/2404.17605)).

| System | How it writes | How numbers reach the text | Review loop | Reported writing failures |
|---|---|---|---|---|
| AI Scientist v1 (Aug 2024) | Template, section by section | Prompt instruction only | 1–2 critiques per section, mechanical LaTeX checks | Hallucinated results, a median of five citations, placeholders, 3.5 h of human time per paper ([Beel et al. 2025](https://arxiv.org/abs/2502.14297)) |
| AI Scientist v2 (Apr 2025) | One pass, reasoning model | JSON summaries in the prompt | ≤3 whole-document reflections, figure VLM | LSTM cited to Goodfellow (2016); misread Figure 3 caption ([Sakana blog](https://sakana.ai/ai-scientist-first-publication/); [Yamada et al.](https://arxiv.org/html/2504.08066)) |
| Jr. AI Scientist (Nov 2025) | Claude Code, staged | Code-generated tables | 3 reflections, GPT-4o reviewers | Invented ablations when a reviewer asked for them; wrote nonexistent papers into the .bib ([Jr. AI Scientist](https://arxiv.org/html/2511.04583)) |
| data-to-paper (Apr 2024; NEJM AI) | Step-gated, whitelisted inputs | Hyperlink to labelled value; `[unknown]` aborts | Rules → LLM reviewer → human co-pilot | Interpretation errors in 2 of 10 open-goal papers ([data-to-paper](https://arxiv.org/pdf/2404.17605)) |
| Kosmos (Nov 2025) | Report synthesis | Each statement cited to code or literature | — | 79.4% of statements rated accurate ([Kosmos](https://arxiv.org/abs/2511.02824)) |

The independent evaluations converge on the same failure classes:

- **Fabricated or invalidated results.** Coding agents produce fabricated or invalidated results "in 80% of the cases" in MLR-Bench ([Chen et al.](https://arxiv.org/abs/2505.19955)).
- **Undisclosed changes to data and metrics.** The AI Scientist v2 sometimes fabricated datasets or substituted metrics without saying so. An LLM auditor that read only the paper caught these pitfalls at **55% accuracy, near chance**; given the trace logs and code, it reached **82%** ([Luo, Kasirzadeh and Shah](https://arxiv.org/html/2509.08713)).
- **Hallucinated references.** At Agents4Science 2025, only about 44% of submissions had no hallucinated references, and authors reported that models "would overclaim results" ([Agents4Science report](https://arxiv.org/abs/2511.15534)).
- **Interpretive statements.** In Kosmos's reports, these were the least accurate class: 57.9%, against 82–86% for data-analysis and literature statements. This breakdown comes from a search summary of the full text and is unverified ([Kosmos v2](https://arxiv.org/html/2511.02824v2)).

***Inference.*** Reuse the gating machinery, not the autonomy: per-step input manifests, rules before critics, numbers that cannot be typed, and an abort on placeholders. The free-writing systems that relied on "do not hallucinate" prompts are exactly the ones whose evaluations report hallucinated numbers.

### 1.2 Hierarchical-generation research: plans buy coherence, not facts

Every successful long-form generator since 2018 does three things. It builds a compressed plan, writes each unit conditioned on the global plan plus its own plan node, and finishes with a deduplication pass. The coherence gains are large:

- **Re3** improved coherent plot by 14 points absolute and premise relevance by 20 points over direct generation ([Re3](https://arxiv.org/abs/2210.06774)).
- **DOC**, a deeper outline with a controller, beat Re3 by 22.5 points on coherence and 28.2 on outline relevance ([DOC](https://ar5iv.labs.arxiv.org/html/2212.10077)).
- **STORM**: Wikipedia editors rated its organization 4.00 against 3.25 for outline-driven retrieval (p = 0.005) ([STORM](https://arxiv.org/html/2402.14207)).

The plan does not make sentences true. STORM's **verifiability was unchanged** (3.80 vs 3.85, p = 0.843). About 15% of its sentences were unsupported by their citations, and the dominant error was "improper inferential linking", meaning the model connected facts its sources do not connect ([STORM](https://arxiv.org/html/2402.14207)). AutoSurvey's unsupported claims were 51% overgeneralization and 39% misalignment ([AutoSurvey](https://arxiv.org/html/2406.10252)).

The finest plan items are the ones most often dropped: DOC's authors state it "often fails to follow low-level outline items" ([DOC](https://ar5iv.labs.arxiv.org/html/2212.10077)). In Dramatron, professional writers edited the plot-outline layer least, and the system has **no mechanism to propagate an upper-level edit downward** ([Dramatron](https://ar5iv.labs.arxiv.org/html/2209.14958)).

On write order and revision:

- Writing paragraphs in parallel rather than sequentially cost about 6% coherence in AgentWrite ([LongWriter](https://arxiv.org/html/2408.07055)).
- Revision loops help only with an external signal. Self-Refine improved outputs by about 20% on average ([Madaan et al.](https://arxiv.org/abs/2303.17651v1)). Intrinsic self-correction without external feedback can degrade reasoning answers ([Huang et al.](https://arxiv.org/pdf/2310.01798)).
- AutoSurvey's reflection pass barely moved its citation metrics, and gains flattened after the second round ([AutoSurvey](https://arxiv.org/html/2406.10252)).

***Inference.*** The beats and bullets are where human review is cheapest and errors cascade from, so they are the human review surface. The step that turns bullets into prose must be barred from adding claims. Loops worth running are those with an oracle: compile, lint, macro resolution, or an entailment check against the bullets. Unanchored "improve this" rounds should be capped at one.

### 1.3 Economists who publish their workflows converge on code-sourced numbers and outside critics

Korinek's guide in the *Journal of Economic Literature* rated "synthesizing text" (turning rough bullet points into prose) **"highly useful", 3 of 3**, and literature research only 1 of 3 because of invented references. He also warned that the output is "a bit sterile and lacks the idiosyncracies and elements of surprise that characterize human writing" ([Korinek, NBER w30957](https://www.nber.org/system/files/working_papers/w30957/w30957.pdf)).

Other practitioners report these habits:

- **Goldsmith-Pinkham** commits each step to GitHub to review Claude's work "one change at a time" and links the repo to Overleaf. His core rule is that "every number in a draft should come from the code, not from an AI's memory" ([Markus' Academy ep. 162](https://markusacademy.substack.com/p/claude-code-for-applied-economists)).
- **Cunningham's "Referee 2"** runs in a fresh terminal because "independence is structural" and never modifies author code. His bibcheck "does NOT check whether each citation supports the claim it's attached to" ([MixtapeTools](https://github.com/scunning1975/MixtapeTools)).
- **Blattman** warns that "a council of five Claudes still shares one model's blind spots" and adds critics from other vendors. He also keeps a voice file with a ban list ([claudeblattman.com](https://claudeblattman.com/)).
- **Gans** called his year of AI-first research a failure ("the experiment… was a failure"). He cites confident overclaiming ("seduction") and now waits "at least a month" before posting ([Gans](https://joshuagans.substack.com/p/reflections-on-vibe-researching)).

The most direct warning concerns interpretation. In 20 runs each of Claude Code and Codex, an explicitly confirmatory prompt flipped Claude Code's support verdicts **from 10% to 90%** "while its coefficient distribution stayed essentially unchanged": "the locus of AI bias is not estimation but interpretation" ([Alizadeh et al. 2026](https://arxiv.org/abs/2606.11456)).

Anthropic's survey of 1,260 quantitative social scientists found that "only about a third of all AI users have used AI to draft prose" and "only economists and management researchers commonly draft prose with AI". It also found that agent users post about 75% more working papers with "no evidence of more journal submissions" ([Anthropic, May 2026](https://www.anthropic.com/research/coding-agents-social-sciences)).

***Inference.*** The author already matches the published practice on code-sourced numbers, the Overleaf link, and context files. Four things are missing:

- critics from another vendor;
- a claim-support check for citations;
- interpretation rules that fix the verdict words before drafting;
- a cooling-off interval.

None of the economists' published workflows uses a nested-bullets-plus-generated-layers architecture; Korinek's bracketed in-bullet instructions ("[add examples]") are the closest precedent among them. Outside academic writing, the parts exist (§1.5).

### 1.4 This scaffold has the paper-side parts and lacks the gates

The repository already implements several things no published paper-writing system does, though each has an analogue outside academic writing (§1.5) (repo).

- **Generated numbers.** Every number file opens with an `AUTO-GENERATED by writeup/make_rehaul_*.py -- do not edit` header ([generated/rehaul_oos_numbers.tex](../generated/rehaul_oos_numbers.tex)).
- **Review layers.** `make_paper_beats.py` builds both review layers from the same sources and injects `\newlabel` lines from the full paper's `.aux`. It refuses to write if any line it adds contains a digit, or if a second generation is not byte-identical ([make_paper_beats.py](../make_paper_beats.py)).
- **Overleaf round trip.** `flatten.py --check` builds both the flat file and the source and confirms that their PDF texts are identical. `unflatten.py` parks replaced lines, reports edits to numbers as conflicts, and warns when an Overleaf edit adds digits to prose ([overleaf_sync/README.md](../overleaf_sync/README.md); [SPEC.md](../overleaf_sync/SPEC.md)).
- **Pending checks.** `tests/check_pending.py` already polices pending arms in generated tables ([check_pending.py](../tests/check_pending.py)).
- **Pre-registered rules.** Analysis rules are written before scoring as `RULE_*.txt` files with amendments, for example [RULE_linear_pen.txt](../../results/beyond_trees_2026-10-27/RULE_linear_pen.txt) and its amendments B–D in the commit log.

The same audit shows two costs (repo measurement, 2026-10-28):

- **Comment bulk.** Full-line comments are **8,388 of the 12,553 lines (67%)** of `paper_rehaul.tex` plus `rehaul/*.tex`, and 807 of the 916 lines of `rehaul/06_results.tex`. Most are dated parked text and rationale.
- **Overleaf file size.** A preamble comment records that the flat `main2.tex` went over Overleaf's 2 MB editable-file limit on 2026-10-28 ([paper_rehaul.tex](../paper_rehaul.tex)).

| Capability | Best published precedent | This scaffold (repo) | Gap |
|---|---|---|---|
| Numbers come from code | data-to-paper's per-number link check | Generated macros, generator headers | No hard lint on typed digits; `unflatten.py` only warns, and only on Overleaf edits |
| Missing numbers | `[unknown]` aborts the run | `\rhPending` prints visibly; `check_pending.py` | The submission build does not fail on `\rhPending` |
| Stale numbers | showyourwork dependency graph (§3.2) | None | No check that a macro file matches its current results |
| Persistent layers | None for papers. Outside: DITA keys and DITAVAL, spec-driven coding layers, VISAR's synchronized outline (§1.5) | Two generated layers, label injection, byte-identical self-check | Ahead of published paper systems, not of other fields |
| Interpretation rules | Warned about (Alizadeh et al.) | `RULE_*.txt` pre-registered with amendments | Rules are not linked to the verdict words in the text |
| Claims ledger | Nanopublications (§3.3) | [CLAIMS_LEDGER_2026-08-06.md](../CLAIMS_LEDGER_2026-08-06.md), markdown, an earlier study | Not machine-checked |
| Citations | clibib deterministic fetch (§6.1) | 148-entry `references.bib` | No verification gate (§6.4) |
| Review surface | InkSync typed edits, Patchwork branches (§5.1) | The author reads the flat paper and two layers on Overleaf. The Overleaf helper script's `push-beats` treats the layers as read-only ("edits go to the sources") | The author's edit messages are applied by hand and leave no machine-readable record or lock; collaborators' edits on the layers cannot flow back (A2, A3, A5, §8.5) |
| Author decisions | None for papers. Outside: requirements suspect links (DOORS Link Validity, Jama), ADRs with fitness functions, legal answer files (§1.5) | D1–D20 table in [INTEGRATION_PLAN_2026-10-26.md](../INTEGRATION_PLAN_2026-10-26.md), prose that no generator reads | Decisions reach the code only by hand: D13 reached three generators two days late (§8.5) |
| AI disclosure | OUP July 2026 policy (§7.1) | Commit trailers name the model | No generated statement |

### 1.5 Outside academic writing: the parts exist

A follow-up search outside paper-writing tools finds every part of the design built, and most of them proven, in five fields.

**Requirements traceability.**

- **DOORS Next.** IBM's DOORS Next replaced timestamp-based "suspect links" with Link Validity, which "compares the actual contents of the artifacts in a relationship" and so "prevents insignificant changes from causing a relationship to become suspect". Links have three states, Suspect, Valid and Invalid, and the tool records who set the state, when, and an optional comment. A validity assertion carries over to other streams where the linked content is identical ([Jazz blog, 2015](https://jazz.net/blog/index.php/2015/11/12/link-validity-coming-in-clm-version-6-0-1/comment-page-1/)).
- **Jama.** A change to an upstream item flags the downstream link as suspect. "Only updates to certain fields trigger suspect links", and clearing is manual, one link at a time or with "Clear All" ([Jama Help](https://help.jamasoftware.com/en/manage-content/coverage-and-traceability/relationships/clear-suspect-links.html)).
- **Codebeamer.** It adds per-field "Omit Suspected" settings and a merge dialog that lists the changed fields ([PTC Codebeamer Help](https://support.ptc.com/help/codebeamer/r3.0/en/codebeamer/user_guide/ug_propogate_suspects_reference.html)).
- **Sphinx-needs.** The docs-as-code tool can "warn and stop the build" on failed constraints. A reported issue, however, says its warnings escaped the warnings file, "so pipelines may not fail" ([sphinx-needs #344](https://github-redirect.dependabot.com/useblocks/sphinx-needs/issues/344)).
- **Evidence on value.** Traceability made maintenance tasks about 24% faster, with about 50% more correct solutions, in a 71-subject experiment (Mäder & Egyed 2015, via secondary summaries) ([itemis](https://blogs.itemis.com/en/benefits-from-requirements-traceability-when-maintaining-software-systems)). Hand-maintained links erode ([Cleland-Huang et al. 2003](https://isr.uci.edu/content/event-based-traceability-managing-evolutionary-change.html)). No study measures suspect-link fatigue; the vendors' noise filters are the indirect evidence.

**Assurance cases and architecture decisions.**

- **GSN v3** adds confidence arguments and defeaters ([Adelard](https://adelard.com/news/asce-51-delivers-enhanced-functionality-support-for-gsn-v3-and-next-generation-assurance-case-design)).
- **ExplicitCase** separates computed *potential* impact from the *actual* impact a safety engineer decides, and calls indirect ripple effects the hardest part to capture (snippet only) ([fortiss AF3 help](https://git.fortiss.org/af3/af3-rcp/-/raw/29e6b519f2e3e7629d58446fa6b4d1a3a7cab28a/org.fortiss.af3.rcp.help/html/assuranceCases/maintenance.html)).
- **Continuous assurance (2025 proposal).** After a change, goals are labelled valid, invalid or uncertain. Cheap evidence regenerates automatically and expensive evidence manually (snippet only) ([arXiv 2511.14805](https://arxiv.org/pdf/2511.14805)).
- **ADRs with fitness functions.** In software architecture, a decision recorded as an Architecture Decision Record is enforced by an executable "fitness function" such as an ArchUnit test that fails CI. Practitioners warn that "documentation alone doesn't prevent drift" ([DEV](https://dev.to/alexandreamadocastro/stop-architecture-drift-operationalizing-adrs-with-automated-fitness-functions-22oi)).

**Structured authoring and document assembly.**

- **DITA** defines values once through keys, reuses content by reference (conref), and builds views with DITAVAL files whose actions are include, exclude, passthrough and "flag", which includes the content and marks it in colour ([DITA 1.3 DITAVAL](https://dita-lang.org/1.3/dita/langref/ditaval/ditaval-prop)). Its spec concedes that filtering before or after key resolution "can" give different bindings ([DITA 1.2 keys](https://docs.oasis-open.org/dita/v1.2/os/spec/archSpec/overview_of_keys.html)). Localization guidance says to reuse complete thoughts, not sentence fragments ([RWS](https://community.rws.com/product-groups/tridion/tridion-docs/b/weblog/posts/best-practices-writing-dita-with-localization-in-mind-tips-for-using-conrefs-and-keyrefs)).
- **Paligo** offers "where used" lookups ([Paligo](https://docs.paligo.net/en/how-to-find-reusable-content.html)).
- **Antora** has hard-set attributes that "cannot be overridden from a component version descriptor or a page" ([Antora](https://docs.antora.org/antora/latest/playbook/asciidoc-attributes/)).
- **Legal assembly.** Systems keep an answer file separate from the template, and Docassemble errors on any undefined variable ([Docassemble](https://docassemble.org/assets/docassemble-logic.html)).
- **Transclusion tools** trade live updates against diffable source. Notion synced blocks are live but database-bound ([Notion](https://www.notion.com/help/synced-blocks)). Logseq writes block IDs into the files, which creates merge hazards ([Logseq forum](https://discuss.logseq.com/t/confused-by-how-logseq-treats-the-id-property/1558)).

**AI tools with persistent upper layers.**

- **Spec-driven coding chain.** Spec Kit, Kiro, OpenSpec and Tessl chain constitution → spec → plan → tasks → code. Kiro writes requirements as "WHEN [condition/event] THE SYSTEM SHALL [expected behavior]" ([Kiro](https://kiro.dev/docs/specs/feature-specs/)) and includes steering files `always` by default ([Kiro Steering](https://kiro.dev/docs/steering/)). OpenSpec records each change as dated ADDED/MODIFIED/REMOVED deltas (search summary of its docs) ([OpenSpec](https://openspec.dev/docs/the-workflow)). Tessl generates code from specs ("spec-as-source").
- **Böckeler's three levels.** She distinguishes spec-first, spec-anchored and spec-as-source, finds that "all SDD approaches and definitions I've found are spec-first", and reports that agents both ignored and over-applied specs and that Tessl regenerated different code from an unchanged spec ([Böckeler](https://martinfowler.com/articles/exploring-gen-ai/sdd-3-tools.html)).
- **Review cost.** In the one quantified comparison, Spec Kit took about 5.5 hours of review and produced 2,577 lines of Markdown for about 689 lines of code, roughly 10 times slower than iterative prompting ([Eberhardt, Scott Logic](https://blog.scottlogic.com/2025/11/26/putting-spec-kit-through-its-paces-radical-idea-or-reinvented-waterfall.html)).
- **Fiction tools.** Sudowrite's Story Bible "only consider[s] Characters and Worldbuilding elements that are explicitly mentioned" ([Sudowrite](https://docs.sudowrite.com/using-sudowrite/1ow1qkGqof9rtcyGnrWUBS/what-is-story-bible/jmWepHcQdJetNrE991fjJC)). Novelcrafter's codex distinguishes "Always Include" from "Include when detected" ([Novelcrafter](https://www.novelcrafter.com/help/faq/ai-and-prompting/codex-context-in-prompting)). Users of Sudowrite report having to regenerate whole chapters for a partial beat change (feedback posts) ([Sudowrite feedback](https://feedback.sudowrite.com/feedback/84994)).

**Scholarly tools, HCI prototypes and Overleaf.**

- **HCI prototypes** qualify any claim that no writing system keeps persistent layers:
  - VISAR keeps a hierarchical outline and argument graph synchronized with the text, and drafts from outline variants (N = 12, snippet only) ([arXiv 2304.07810](https://arxiv.org/abs/2304.07810)).
  - Dang et al. keep paragraph summaries in the margin, "updated while typing", which helped writers rework structure (N = 4 and 8, essays) ([arXiv 2208.09323](https://arxiv.org/abs/2208.09323v1)).
- **Living Papers** renders one AST to HTML or PDF. Its "layers" are output targets, not abstraction levels, and it leaves collaborative authoring to future work ([Living Papers](https://idl.uw.edu/living-papers-paper/living-papers)).
- **showyourwork's Overleaf integration** syncs each file in one direction only and stops a pull with "Refusing to overwrite with Overleaf version" ([showyourwork: Overleaf](https://show-your.work/en/latest/overleaf/)).
- **Agent–Overleaf tools.** They either keep each file one-directional (ChernXiangyu's skill: generated files repo → Overleaf, prose Overleaf → repo, and "renames destroy comments and tracked changes") ([ChernXiangyu](https://github.com/ChernXiangyu/overleaf-git-skill)) or merge with a fixed winner, `-X ours`/`-X theirs` ([gsoosk](https://github.com/gsoosk/claude-overleaf)). claudeleaf instead types into the live editor from a dedicated bot account and can submit edits as tracked-change suggestions ([claudeleaf](https://github.com/lonetis/claudeleaf)).
- **Overleaf's own docs.** Overleaf-side edits reach GitHub bundled into one commit under the connecting account, and git-side changes appear in Overleaf history as "Anonymous (via GitHub)" ([Overleaf: GitHub sync](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/github)).

| Field | System | Mechanism | What we copy | What we avoid |
|---|---|---|---|---|
| Requirements traceability | DOORS Next Link Validity | Content comparison; Suspect/Valid/Invalid with who, when, comment; validity reused when content is identical | Stamps hashing only the parameters a generator reads; three-state review record; clearance carried forward (§8.5) | Timestamp-based suspicion |
| Requirements traceability | Jama, Codebeamer | Upstream field change flags downstream link; manual or bulk clear | Field-level triggers | An agent with a "Clear All" button |
| Docs-as-code | Sphinx-needs | Build-time constraints | Violations as build failures | A warning channel CI never sees |
| Assurance cases | ExplicitCase; continuous assurance (arXiv 2511.14805) | Potential vs actual impact; valid/invalid/uncertain; cost-tiered regeneration | Flips plus an uncertain tier; explicit ripple graph | A flip-only view that hides weakening |
| Architecture | ADRs + ArchUnit fitness functions | Recorded decision enforced by a CI test | `status`, `supersedes`, `rationale`; checks that read the decision record | Encoding style preferences as hard failures |
| Structured authoring | DITA keys, conref, DITAVAL | Indirect values; include/exclude/flag; filter vs resolve order | Flag-mode review build; resolve before filter; whole-claim reuse | Sentence-fragment reuse |
| Structured authoring | Paligo, Antora | Where-used lookup; hard-set attributes | Where-used report; decisions no layer can override | Local overrides of decisions |
| Legal assembly | HotDocs, ContractExpress, Docassemble | Answer file apart from the template; error on undefined | The register as answer file; loud failure | Silent fallbacks |
| Transclusion | Notion, Obsidian, Logseq | Live synced blocks; IDs written into source | — | A cloud database as the source of record; tool-generated UUIDs in source |
| AI coding | Spec Kit, Kiro, OpenSpec, Tessl | Persistent upper layers; EARS requirements; always-on steering; dated deltas | EARS-form locks; always-applied conventions; dated deltas | Long upper layers; assuming in-context decisions are obeyed; whole regeneration |
| AI fiction | Sudowrite, Novelcrafter, NovelAI | Fact codex injected by mention | Always-include for binding facts | Retrieval by mention; whole-chapter regeneration |
| HCI writing | VISAR; Dang et al. | Synchronized outline; live margin summaries | Evidence for reading layers | Generalizing from short essays |
| Living papers, Overleaf | Living Papers; showyourwork; claudeleaf | One AST; one-way sync with an overwrite guard; bot-account suggestions | Overwrite guard (already in the repo); bot-account option | Blind merges with a fixed winner |

**What none of them does.**

- **No combination for papers.** No system found combines these parts for a paper.
- **No outcome-based review filter.** None filters review on whether a claim's outcome flipped. Traceability tools, ADR checks and spec tools flag when an input changes. Only assurance research labels outcomes valid, invalid or uncertain, and there a human decides actual impact.
- **No two-way prose round trip with Overleaf.** Published workflows keep each file one-directional or merge with a fixed winner. The repo's flatten/unflatten with its refuse-on-unpulled-edits guard already goes further than any of them for the main file (repo).

***Inference.*** The design is a recombination. Copy each part as it was proven, and treat three things as the untested pieces that need this project's own evidence:

- the combination itself;
- the outcome-flip filter;
- the two-way prose round trip.

Mäder and Egyed's result is the best direct evidence that seeing the right links helps maintainers. No one has measured flag fatigue, so the author's own log of flags, flips and clearing time would be new evidence.

---

## 2. Layers: render the final paper from the bullets, don't expand them

### 2.1 Two families of LaTeX switches, and every skip loses its labels

LaTeX offers two families of switches.

- **Token-skipping conditionals** (`\newif`, `ifdraft`, `optional`'s `\opt`, `tagging`'s `\tagged`) work anywhere, including inside one `\item` or a macro argument. They fail only when the skipped text has unbalanced `\if`/`\fi` tokens ([TeX FAQ: conditional compilation](https://texfaq.org/FAQ-conditional)).
- **Line-scooping environments** (`comment`, `versions`) tolerate ill-formed text. However, the `comment` package requires that "the opening and closing commands should appear on a line of their own", and it discards trailing text on a `\begin` line ([comment manual](https://matojo.unizar.es/ctan/macros/latex/contrib/comment/writeup.tex)). Its environments break inside macro arguments ([versions docs](https://ctan.math.washington.edu/tex-archive/macros/latex/contrib/versions/versions-doc.pdf)), and excluding a list environment can raise "Lonely \item" errors ([TopAnswers TeX Q715](https://topanswers.xyz/tex?q=715)).

**Every mechanism that skips text also skips the `\label`s inside it, so references into a hidden layer print "??"** ([Overleaf: undefined references](https://www.overleaf.com/learn/latex/Errors%3AThere_were_undefined_references.)). Only file-level `\includeonly` and external-label injection keep numbers stable. The repo's injection of `.aux` labels into the review layers is the general fix (repo, [make_paper_beats.py](../make_paper_beats.py)).

For the final build:

- `latexpand` strips `%` comments by default ([latexpand README](https://tug.ctan.org/support/latexpand/README)).
- `arxiv_latex_cleaner` also removes `\iffalse` blocks and named commands and environments, but it assumes every `\if` command is a primitive. Its handling of user-defined flags such as `\ifauthornotes` needs a test before anyone relies on it ([arxiv-latex-cleaner](https://github.com/google-research/arxiv-latex-cleaner)).

***Inference.*** For files that agents edit, token-skipping flags and semantic macros are the right family. The `comment` package's own-line rule fails silently when an agent appends text after `\end{...}`.

### 2.2 Outside LaTeX, the precedents are profiles, synopses and article mode

**Quarto** has the most complete flag system: named profile files plus `.content-visible when-profile` blocks. It also has a cautionary property: **an unknown profile name is silently ignored**, so a typo goes unnoticed ([Quarto: Project Profiles](https://quarto.org/docs/projects/profiles.html)).

Only two systems make "outline level" native:

- **Scrivener** gives each document a synopsis and a text. A compile layout picks which one is output ([L&L forum](https://forum.literatureandlatte.com/t/tutorial-on-scrivener-compile-formats-and-creating-one-with-only-title-and-synopsis/153517)).
- **Org-mode** selects heading depth and tagged subtrees at export ([Org manual](https://orgmode.org/manual/Export-Settings.html)).

The closest LaTeX precedent is **beamerarticle**. One source yields slides and an article, and `ignorenonframetext` drops narrative text from the slides ([Beamer](https://en.wikipedia.org/wiki/Beamer_(LaTeX))). Org's dual-format recipe interleaves a bullet frame with a narrative subsection in the same file ([Worg: Beamer dual format](https://orgmode.org/worg/exporters/beamer/beamer-dual-format.html)). No public academic-paper template was found that pairs a beat and its prose per paragraph behind a flag.

***Inference.*** Copy Quarto's named, mutually exclusive profiles as a LaTeX layer selector, and make an unknown layer name a hard `\PackageError`. Typst, Quarto and Org have no Overleaf round trip, so they are design references, not migration targets.

### 2.3 Expansion is where facts leak, so the final layer should strip scaffolding

The evidence on bullets-to-prose expansion all points one way.

- **Expansion is how researchers synthesize LLM text.** Liang et al. built their model of LLM-modified scientific text by having an LLM summarize a human paragraph into a bullet outline and then expand the outline back into prose. That generator defines the "LLM" word distribution they detect at scale ([Liang et al., NHB 2025 / arXiv 2404.01268](https://arxiv.org/html/2404.01268)).
- **Expansion adds unsupported links.** The unsupported-claim rates in §1.2 (STORM's "improper inferential linking", AutoSurvey's 51% overgeneralization) arise at the expansion step.
- **Plan-then-write restates.** AgentWrite's plan-then-write lowered coherence and clarity by about 2%, with "frequent restating/summaries" ([LongWriter](https://arxiv.org/html/2408.07055)).
- **Expanded prose reads flat.** Korinek found it "sterile" ([NBER w30957](https://www.nber.org/system/files/working_papers/w30957/w30957.pdf)).

***Inference.*** This favors a design in which the final paper is a *rendering* of the bullets rather than a second text (call it Option A).

- Each top-level beat's claim becomes a paragraph's topic sentence.
- Each nested bullet is written as one or two complete sentences that can stand in that paragraph.
- The final layer removes the itemize, the bold labels and the nesting, then joins the sentences.

Option A makes the no-new-claims rule hold by construction. The words are the same words the author approved, and fidelity checks reduce to a token diff.

Option A's cost is that paragraphs assembled from bullets can read as lists. It is mitigated three ways:

- a content rule that bullets are sentences;
- two small macros for text that prints only in the bullet layer or only in the final layer;
- a "join pass" whose permitted edits are restricted to connectives, with a mechanical diff check (§8.4).

Option B co-locates an `\ifprose` paragraph with each beat and regenerates it when a stored hash of its beat and bullets changes. It suits the introduction and conclusion, where Head's formula reorders material ([Head](https://blogs.ubc.ca/khead/research/research-advice/formula)). Option B needs the expansion safeguards: a coverage check, a no-new-claims check, and a reverse outline ([UT Austin UWC](https://uwc.utexas.edu/wp-content/uploads/Reverse-Outlining-1-1.docx)). No study compares the two options or measures how referees read bullet-derived paragraphs.

### 2.4 Overleaf, git and diff mechanics bound the layer design

Overleaf constraints:

- **The git bridge is premium-only.** It "does not support branching" or "tagging".
- **Comments are fragile.** "Pushes from Git to Overleaf can result in the loss or displacement of track changes and comments" ([Overleaf: Git integration](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/git-integration)).
- **Size and time limits.** Editable files must stay at 2 MB or less and a project at 7 MB of editable material. Compiles time out at 10 s on free plans and 240 s on premium ([Overleaf plan limits](https://docs.overleaf.com/getting-started/free-and-premium-plans/plan-limits)).
- **No command-line flags.** A layer can only be chosen in the source. However, "you can compile multiple documents inside one Overleaf project without changing the project's main file setting" ([Overleaf: the main document](https://docs.overleaf.com/getting-started/recompiling-your-project/the-main-document)).

The flatten copies comments verbatim into the flat file ([SPEC.md](../overleaf_sync/SPEC.md)), so the 67% comment share counts against the 2 MB limit and the reading copy alike (repo).

For diffs, the Semantic Line Breaks spec requires a break after every sentence. That means "an edit to one sentence changes one line instead of reflowing a paragraph", and it recommends `git diff --word-diff` ([sembr.org](https://sembr.org/)). `git latexdiff` renders a marked-up PDF between revisions ([git-latexdiff](https://man.archlinux.org/man/git-latexdiff.1.en)). Its companion `latexrevise` can only accept or reject **all** changes at once ([CTAN latexdiff](https://ctan.org/pkg/latexdiff)).

***Inference.*** Five consequences follow:

1. **Layers on Overleaf should be thin wrapper files.** Each sets a layer name and inputs the body, the source-level equivalent of Quarto profiles.
2. **Decision state cannot live in git branches or Overleaf comments on the Overleaf side.** It must live in repo files. Under A3 the author's corrections arrive as chat messages. Under A5, edits made on Overleaf by collaborators flow back through the pull, review layers included. Comments are not an input (§8.5).
3. **Parked text should leave the live files** for a per-section sidecar. This keeps the never-delete policy while shrinking the flat file, the agents' context, and the diffs.
4. **One sentence per line** should become a source convention, checked on pull because Overleaf editing can rejoin lines.
5. **Collaborators on Overleaf (A5)** will use comments and tracked changes, which git pushes can lose or displace. The mitigations are to pull before every push, push minimal diffs (no whole-file reflows after collaborators start), and settle tracked changes before a pull (§8.5).

---

## 3. Hooking text to results: macros solve transcription, not staleness or claims

### 3.1 data-to-paper's chain, and what it left unsolved

data-to-paper "programmatically assign[s] a unique label for each numeric value" in the analysis outputs. The LLM must "wrap each numeric value that it writes with a LaTex hyperlink matching the corresponding label".

- **Derived values** use `\num(<formula>, "explanation")`. These compile to the value plus a "Notes" appendix listing every formula.
- **The rule check** verifies that "each numeric value mentioned in the section is hyperlinked, and that the target of each link correctly matches."
- **The resulting chain** runs from a number to the Notes, to the table, to the output file, to "the specific code lines which generated this output file" ([data-to-paper](https://arxiv.org/pdf/2404.17605)).

Its reported major errors were wrong analyses and interpretations. Typed numbers were not among them: one paper misread a correct analysis, and another ran an erroneous one ([data-to-paper](https://arxiv.org/pdf/2404.17605)).

***Inference.*** Generated macros already give this paper data-to-paper's labelled-value layer. Three of its pieces are missing:

- a lint that rejects literal digits;
- a formula route for derived numbers;
- a clickable chain for the reviewer.

The remaining risk sits in claims and undisclosed changes, not transcription.

### 3.2 A compiled macro cannot show that it is stale

Three styles of guarantee exist:

- **Rebuild everything.** Gentzkow and Shapiro: "Write a single script that executes all code from beginning to end". Prove replicability by deleting all outputs and reproducing them ([Gentzkow & Shapiro](https://web.stanford.edu/~gentzkow/research/CodeAndData.pdf)).
- **Dependency graph.** showyourwork's `\variable{file}` "explicitly marks the file as a dependency of the manuscript in the workflow graph", and the paper is rebuilt "when the generating script or rule changes" ([showyourwork docs](https://show-your.work/en/latest/latex/)).
- **Compute at render time.** Quarto and knitr do this, but they have documented escape hatches: `freeze`, and caches invalidated only by changes to chunk source code. Quarto warns users to re-render fully "when things outside of source code change (e.g. input data)" ([Quarto execution options](https://quarto.org/docs/projects/code-execution.html)).

***Inference.*** `\rhPending` and the undefined-reference gate catch missing numbers. Nothing catches a macro file produced from an older results file, which compiles cleanly. An agent that reruns one generator and recompiles creates exactly Quarto's frozen-cache hazard. The light fix:

- stamp each generated header with the generator's git blob hash and the SHA-256 of every input file;
- recompute the stamps before every build;
- run a full regenerate-and-diff before each Overleaf push.

### 3.3 No tool checks direction words, so the generators should emit them

**statcheck** recomputes p-values from reported statistics. A validity study gives 96.2–99.9% overall accuracy ([Nuijten et al.](https://osf.io/3cvkn)). A secondary critique says it catches only about 51.8% of truly inconsistent tests ([LSE Impact blog](https://blogs.lse.ac.uk/impactofsocialsciences/2018/02/28/statcheck-a-spellchecker-for-statistics/)).

LLM checkers are weak at this:

- On SPOT, built from errors serious enough to cause errata or retractions, the best model reached **21.1% recall and 6.1% precision**, and models "rarely rediscover the same errors" across runs ([SPOT](https://arxiv.org/abs/2505.11855)).
- On SciTab, most models verified claims against tables "barely above random guessing" ([SciTab](https://arxiv.org/abs/2305.13186)).

**Nanopublications** package each claim as an assertion, its provenance and publication info. The per-claim overhead is large ([Springer 2025](https://link.springer.com/article/10.1007/s00799-025-00431-x)).

***Inference.*** The deterministic route is to move data-to-paper's idea from numbers to words. Each generator that writes a value also writes the facts it licenses:

- a sign;
- a rank or argmin;
- a significance word chosen by the p-value at the stated level;
- a comparison word such as "lower" or "higher".

A linter then checks every sentence that contains a numeric macro and a direction or quantifier word against those predicates. The repo's `RULE_*.txt` files already fix decision rules before scoring (§1.4). Extending them to map estimates to verdict words is the guard the Alizadeh result calls for. The parked abstract comments already record such checks by hand, for example "'all ... each significant' backed by `\rhorAfLower = \rhorAfLowerSig = \rhorAfPairs`" (repo, [paper_rehaul.tex](../paper_rehaul.tex)). The same predicates serve the TRUE? half of Rule NT (§4.3.7): "every", "all six" and "the largest" are licensed only by a count equal to the total or an argmax.

### 3.4 Reviewers need provenance next to the text

The CMU auditor result (55% from the paper alone, 82% with logs and code) argues for a review build that shows each number's source ([Luo et al.](https://arxiv.org/html/2509.08713)).

- **showyourwork** puts margin icons on figures that link to the generating script ([showyourwork docs](https://show-your.work/en/latest/latex/)).
- **The Social Science Data Editors' template README** asks for a table of "Figure/Table #, Program, Line Number, Output file" and has a checkbox for "All numbers provided in text in the paper" ([template README](https://raw.githubusercontent.com/social-science-data-editors/template_README/master/template-README.md)).

***Inference.*** A provenance layer can redefine every number macro to print its value with a link or tooltip. The tooltip names the macro, its existing `%` description, the generator, and the input hash. Because every generated file names its generator, the same records can produce the replication README's table instead of hand-writing it.

---

## 4. Prose scrub: calibrate to the literature, let an LLM touch only register

### 4.1 The vocabulary fingerprint is real, moving and field-dependent

**Kobak et al.** estimated that at least **13.5% of 2024 PubMed abstracts** were LLM-processed.

- 379 of 454 excess words were style words, 66% of them verbs.
- Markers include "delves" (excess ratio r = 28.0), "underscores" (13.8) and "showcasing" (10.7) ([Kobak et al., Science Advances 2025](https://arxiv.org/html/2406.07016v5)).

Other studies find the same fingerprint:

- **ICLR 2024 reviews:** "commendable", "meticulous" and "intricate" became 9.8, 34.7 and 11.2 times more likely to appear in a sentence ([Liang et al., ICML 2024](https://arxiv.org/pdf/2403.07183)).
- **Economics:** in the top 25 economics journals, AI-associated terms rose **6.67 percentage points in 2024** against stable control terms ([Feyzollahi & Rafizadeh, Economics Letters 2025](https://ideas.repec.org/a/eee/ecolet/v250y2025ics0165176525001028.html)).

The fingerprint moves:

- "Delve" fell after it was publicized in spring 2024 ([Geng & Trotta](https://arxiv.org/html/2502.09606v2)).
- Wikipedia's maintained catalog dates its lists by model era. Its mid-2026 list includes "leverage, multifaceted, prioritize" ([Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)).

A rough count over the live text of `rehaul/0*.tex` (about 11,500 words after stripping comments and commands; repo measurement, 2026-10-28) found:

- zero `---` em dashes;
- none of the rare markers (delve, underscore, showcase, intricate, meticulous, pivotal, realm);
- no ", highlighting/ensuring/…" tails;
- six "rather than";
- three "leverage".

All three "leverage" hits are the **leverage effect** of the HAR literature, for example "leverage \citep{CorsiReno2012}". A generic AI-ism list would flag correct domain vocabulary (repo).

***Inference.*** The current main text, edited heavily under house rules, is already clean on vocabulary markers. The scrub's value now lies in register and structure, and in stopping regressions as agents redraft. Word lists need a field whitelist and should be scored as rates against a field baseline, not general English.

### 4.2 Structural markers: grammar ratios are measured, the rest is catalog

Reinhart et al. measured GPT-4o against human text:

- present-participial clauses about **5.3 times** the human rate;
- nominalizations about 2.1 times;
- phrasal coordination about 1.9 times;
- the agentless passive at about **half** the human rate.

Classifiers on these features separate humans from a given model at 93–98%. Instruction tuning pushes a dense, noun-heavy register that "limits their ability to mimic other writing styles" ([Reinhart et al., PNAS 2025](https://arxiv.org/html/2410.16107v2)).

On em dashes:

- They persisted under "no markdown" instructions in every tested model except Llama ([Freeburg 2026, preprint](https://arxiv.org/abs/2603.27006)).
- Wikipedia reports a July 2026 Economist finding that **among current models only Claude** used em dashes more than professional writers (secondhand) ([Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)).

On other constructions:

- "It's not X, it's Y" occurs up to 6.3 times the human rate in some models' creative writing ([Antislop](https://arxiv.org/html/2510.15061)).
- Triads, signposting ("it is important to note"), summary closers and bold inline headers are documented in catalogs, not controlled corpus studies ([Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)).
- Uniform sentence length ("low burstiness") is contested and may be confounded with length ([NeurIPS 2026](https://neurips.cc/virtual/2026/164820); [CT²](https://arxiv.org/pdf/2310.05030)).

***Inference.*** Mechanical structural rules should report rates per 1,000 words against the corpus band. Raw burstiness belongs on a dashboard, never in a gate; §4.3.4 replaces it with length-normalized cadence metrics. The bold-label `\item` format is itself a catalogued AI formatting sign ("inline-header vertical lists"). This is one more reason the final layer must strip it.

### 4.3 Cadence, voice and stance

This subsection extends §4.2 to grammatical voice, person, sentence rhythm and stance. It defines the scrub's Tier C (§8.4, S4a) and Rule NT, the gate for universals and superlatives (§4.3.7; §8.3 C18; §8.4 S4b).

#### 4.3.1 Instruction-tuned models avoid both the passive and "we"

Reinhart et al. gave each model the first 500 words of a human text and asked it to "write 500 more words in the same style, tone, and diction". GPT-4o still diverged from the human continuation:

- present-participial clauses at 5.3 times the human rate (Cohen's d = 1.38);
- nominalizations at 2.1 times (d = 1.23);
- agentless passives at "roughly half the rate as human texts".

Llama base models "use these features at rates closer to humans". The authors attribute the gap to instruction tuning but cannot name the cause, because tuning processes are not public ([Reinhart et al., PNAS 2025](https://arxiv.org/html/2410.16107v2)).

Studies of person point the same way. ChatGPT essays carried fewer epistemic and attitudinal stance markers and far less authorial presence than student essays on the same topics ([Jiang & Hyland 2026](https://ueaeprints.uea.ac.uk/103026/1/Stance_in_AI_and_student_essays.docx)). A comparison of research articles reports that the ChatGPT corpus "lacked self-mentions"; this is from a search summary of the article ([Pham 2026](https://ijal.upi.edu/index.php/ijal/article/view/101)).

The clearest documented case of preference tuning shaping a style feature concerns certainty, not voice. Reward models scored plain statements 4.03, strengthened ones 0.82 and hedged ones −1.86 ([Zhou et al., ACL 2024](https://arxiv.org/html/2401.06730v2)).

***Inference.*** Together these findings point to one profile: inanimate or nominal subjects with active verbs, as in "The analysis reveals…" or "This decomposition shows…". This is neither passive-heavy nor "we"-heavy. Agent instructions can produce the opposite artifact. The repo's notation guide tells drafters to use "'We' and active voice" ([NOTATION_GUIDE_2026-10-15.md](../NOTATION_GUIDE_2026-10-15.md), §6; repo).

A rough count over about 12,000 words of live main text finds "we/our/us" 44 times, or 3.6 per 1,000 words (repo measurement, 2026-10-28; detex by regex). So the draft does not over-use "we". The metric that tells which profile it has is the **distribution of main-clause subject types**, measured before any correction.

#### 4.3.2 The target literature keeps passives in data and methods

No corpus study of passive share or self-mention by section in finance or econometrics papers was found. The general research-article literature gives the expected shape:

- About 30% of clauses in 60 science articles were passive, and about 29% of those passives were in the methodology section ([Leong 2014](https://www.informalscience.org/passive-voice-scientific-writing-current-norm-science-journals)).
- Methods sections carried more passives than introductions in a 2024 comparison of applied-linguistics journals ([DOAJ record](https://doaj.org/article/d65ecf3c547f4b1385c9bd98904499f2)).
- Economics writers are visible authors:
  - In Harwood's corpus, "all but one of the instances of we in the Economics articles are inclusive", meaning reader plus writer ([Harwood 2005](https://www.researchgate.net/profile/Marcia-Lygia-Casarin/post/Should_I_use_either_we_or_I_in_my_thesis/attachment/59d61d9579197b8077977f87/AS%3A271736107405312%401441798344619/download/We+Do+Not+Seem+to+Have+a+Theory.pdf)).
  - Economics abstracts use authorial pronouns far more often than linguistics abstracts (abstract-level) ([Galaidin & Bednárová-Gibová 2023](https://repozytorium.uwb.edu.pl/jspui/handle/11320/15411)).
  - The English economics subcorpus held almost half of all authorial-presence markers in a three-discipline, three-language corpus (abstract-level) ([BELLS 2024](https://belgrade-bells.fil.bg.ac.rs/index.php/belgrade-bells/article/view/bells-2024-16-3)).

Editorial guidance pulls the other way:

- Nature calls active voice house policy "for as long as I can remember" ([Nature blog](https://blogs.nature.com/blog/active_is_better_than_passive/)).
- Cochrane would root out every passive ([Cochrane](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)).
- Practical guidance notes that the passive "remains the traditional and still-common choice in Methods sections" ([CASRAI](https://www.casrai.org/dictionary/term/active-voice-vs-passive-voice)).

***Inference.*** The JFEc pattern to test against the corpus is:

- "We find/show/document" for findings;
- concrete inanimate subjects for exhibits ("Table 4 reports");
- agentless passives for data provenance and routine procedures ("Quotes with zero bid are removed");
- "we" for authorial choices ("We choose the 30-minute window because…").

The notation guide's blanket active-voice rule matches Nature and Cochrane but probably not the data and methods norms. Per-section corpus bands should decide. Harwood's proportions predate 2005 and modern multi-author "we" is mostly exclusive, so they are not a target.

#### 4.3.3 LLM cadence is narrower and more templated, not just "punchy"

Measured findings:

- **Narrower length spread.** Human text shows "more scattered sentence length distributions", while LLMs show "less variation" and level genre differences ("register leveling") ([Muñoz-Ortiz et al.](https://arxiv.org/pdf/2308.09067)). When models regenerate domain text, the human distributions are wider, with "a longer and heavier right tail" (quoted via search snippet) ([Ju, Blix & Williams 2025](https://arxiv.org/pdf/2505.07784)).
- **Templates.** 76% of the part-of-speech templates in model text also appear in pretraining data, against 35% for human text, and the templates "are not overwritten during fine-tuning" ([Shaib et al., EMNLP 2024](https://aclanthology.org/2024.emnlp-main.368.pdf)).
- **Lower diversity.** Writing with feedback-tuned InstructGPT reduced the diversity of essays' key points, not just their words. The GPT-3 base model did not ([Padmakumar & He, ICLR 2024](https://arxiv.org/html/2309.05196v3)).
- **Length bias.** Reward models conflate verbosity with quality ([Kim et al.](https://arxiv.org/html/2511.12573v1)), and RLHF lengthens answers without a matching quality gain ([Park et al.](https://arxiv.org/html/2403.19159v2)).

The devices readers notice are catalogued but not measured in any study found: alternating long and short "punch" sentences, colon reveals, paragraph-final "This matters because" sentences, and sentence-initial repetition.

***Inference.*** In econometrics prose, the human right tail is the long conditional or definitional sentence, and the human short sentence is a terse pointer to an exhibit. Register leveling predicts less variation between sections in a draft than in corpus papers, which is testable.

A rough split of the current main text gives about 457 sentence units with a mean of 29 words, a coefficient of variation of 0.70, and 24% of sentences over 40 words (repo measurement; bold beat labels split as short "sentences", so the short-sentence share is unreliable). With no corpus band yet, these numbers carry no verdict.

#### 4.3.4 Measure rates against per-paper bands, with sampling error in view

Text length is "an unavoidable confounder in quantitative text-analytic corpus-linguistic studies" ([Liimatta 2024](https://www.degruyterbrill.com/document/doi/10.1075/scl.118.07lii/html)). Templatedness is computable on one draft with a part-of-speech tagger ([Shaib et al.](https://arxiv.org/html/2407.00211v2)), and passive detection agrees with human coding about 98% of the time ([PassivePy](https://pypi.org/project/PassivePy)).

***Inference.*** The standard errors below are binomial or Poisson approximations from the cadence note, not figures from a source. They decide which metrics can be acted on at this draft's size, about 400 sentences and 10,000–12,000 words.

| Metric | Precision at draft level | Precision per section (~60 sentences) | Use |
|---|---|---|---|
| Passive share, split agentless vs by-passive | SE ≈ 2.2 pp (±4 pp) | SE ≈ 5.6 pp (±11 pp) | Band check, draft and large sections |
| "We" rate per 1,000 words, split by verb class (find/show vs estimate/compute) | Good (tens of tokens) | Fair | Band check |
| Main-clause subject types: we, passive subject, concrete inanimate noun, nominalization head, dummy it/there, "This/These" + noun | Good for the 3–4 largest categories | Weak | Diagnoses the voice profile (§4.3.1) |
| Sentence length: mean, CV, p10/p50/p90, shares over 40 and under 10 words | Good; tail shares need 300+ sentences | Mean and CV only | Replaces raw SD, sidestepping the length confound (§4.2) |
| Lag-1 autocorrelation of sentence length, alternation share | SE ≈ 0.05; only abs(ρ) > 0.1 detectable | Useless | Dashboard only; no published baseline |
| Sentence-opening entropy; same-opener runs | Good | Fair | Band check |
| Repeated POS 5–6-grams | Fair, against equal-length corpus papers | Weak | Dashboard |
| Rare Biber features (subject that-clause, by-passive) | Poor (~10 events, ~30% SE) | Useless | Report, never gate |

Corpus percentiles must be computed per paper, then the draft placed in that distribution. Pooling sentences overstates precision. A section should be flagged only when it sits outside the corpus p10–p90 band by more than its own sampling error.

#### 4.3.5 Correct with named sentence operations, not style targets

Global instructions and exemplars are weak tools for grammar:

- Reinhart's models had the human exemplar and the "same style" instruction and still diverged ([Reinhart et al.](https://arxiv.org/html/2410.16107v2)).
- Instructions move stance markers but overshoot. Epistemic markers appeared in about 5–6% of answers by default and 71% when explicitly requested ([Zhou et al.](https://arxiv.org/html/2401.06730v2)).
- "Even when explicitly prompted for accuracy, most LLMs produced broader generalizations" ([Peters & Chin-Yee](https://arxiv.org/abs/2504.00025)).
- Per-axis rewriters do work, voice and length included. StyleRemix's merged LoRA adapters beat much larger LLMs on authorship obfuscation, but they need open-weight fine-tuning ([StyleRemix, EMNLP 2024](https://arxiv.org/html/2408.15666v1)).

No study measures how often voice conversion changes meaning or agency in academic prose.

***Inference.*** For a Claude pipeline the correction runs in five steps:

1. Compute the §4.3.4 metrics and find the sections outside the band.
2. Pick the sentences driving each deviation. Examples: methods sentences with "we" plus a routine verb (compute, winsorize, obtain), or inanimate subjects with evaluative verbs ("the results underscore").
3. Send each flagged sentence, with its paragraph, to the model with one **named operation**:
   - convert to agentless passive;
   - replace the inanimate agent with "we" or a passive;
   - merge with the next sentence;
   - split at the conjunction;
   - delete the participial tail;
   - delete the quantifier;
   - replace the evaluative verb with the number.
4. Re-measure.
5. Stop once the section is inside the band, not at the corpus median, which would itself homogenize the text.

The meaning checks are mechanical:

- the same multiset of macros, `\cite` keys and `\ref` labels;
- negations and modals unchanged unless targeted;
- no invented agent. "Data are provided by OptionMetrics" must not become "We provide data".

Whether Claude follows sentence-level operations more reliably than global instructions is inference, not a published result.

#### 4.3.6 LLMs inflate scope; booster counts are mixed

The strongest stance evidence concerns scope, not booster words. Across 10 LLMs, including Claude 3.7 Sonnet, and 4,900 summaries:

- LLM summaries "were nearly five times more likely to contain broad generalizations (OR = 4.85, 95% CI [3.06, 7.70])" than human-written summaries.
- "Newer models tended to perform worse".
- The mechanism is omitting "details that limit the scope of research conclusions" ([Peters & Chin-Yee, RSOS 2025](https://arxiv.org/abs/2504.00025)).

RLHF models emitted strengtheners more than weakeners, and only 53% of strengthener-marked answers were correct ([Zhou et al.](https://arxiv.org/html/2401.06730v2)).

Booster word counts are mixed:

- ChatGPT-3.5 essays used no "clearly" or "definitely" ([Shalevska 2024](https://eprints.uklo.edu.mk/10825/1/HEDGES%2BAND%2BBOOSTERS%2BIN%2BAI%2BAND%2BHUMAN%2BWRITING%2BA%2BCOMPARATIVE%2BANALYSIS.pdf)).
- ChatGPT abstracts overused attitude markers and boosting verbs (search summary of the catalog record) ([El-Dakhs et al.](https://ersearch2.cvtisr.sk/vufind/SummonRecord/FETCH-LOGICAL-c244t-abbf34b2aa81c4472ef9f703588e2b1621deb79416a625275c554b50cb5e47d23/Details?lng=en)).

Human prose is drifting toward promotion too. Hype adjectives appeared in 72% of NIH grant abstracts in 1985 and 97% in 2020 ([Millar et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC9412227)). No corpus study counts "always/never/every/all/the most" in LLM versus human academic prose.

***Inference.*** Two kinds of absolute need different treatment:

- **A literal universal or superlative** asserts a checkable fact about a defined finite set of bars, days, classes, horizons or tests. It is true only if a computed fact backs it: "every bar is scored" needs `n_scored == n_bars`; "all six classes" needs a count of six; "the largest gain is at 30 minutes" needs the argmax of the cited table.
- **A rhetorical absolute** ("fundamentally changes", "the most natural approach") has no set behind it. It is deleted or replaced with a number.

Hedging should run in both directions. Findings are scoped to sample, period, frequency and test. Modal hedges ("may", "suggests") are reserved for mechanism and interpretation, and a hedged statement of a computed fact ("the loss appears lower" when the table shows it is) is flagged too. Generic-scope detectors catch the Peters & Chin-Yee failure: bare present-tense generalizations where the evidence is sample-specific, and sample qualifiers present in the results but dropped in the abstract or conclusion.

The rough repo count shows why rates alone mislead. "Every" appears 86 times in about 12,000 words of main text, the most frequent universal, followed by "each" (50) and "all" (32). Boosters appear once; the notation guide bans "clearly". Many of the "every" instances name the paper's evaluation scope, "every regular-hours bar" as against the closing half hour, which is a contrast the reader needs. Others restate a scope the sentence has already fixed. Only a per-instance test separates them (repo measurement, 2026-10-28).

#### 4.3.7 Rule NT: NEEDED? first, then TRUE?

The author's observation is that superlatives and universals are often distracting and unnecessary when the context already makes them valid. Pragmatics states the principle: Grice's maxim of Quantity includes "do not make your contribution more informative than is required" ([Edinburgh pragmatics course](https://lel.ed.ac.uk/~hrohde/plc/implicature-3grice.html)). Cochrane's rules apply it to economics prose. He bans adjectives on one's own results, calls "In other words" "a sign of trouble", and wants "nothing before the main result that a reader does not need to know" ([Cochrane](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)).

The measured credibility evidence is weak or null:

- A randomized trial found hype "did not appear to bias readers' evaluation of the scientific merit" ([Peer Review Congress](https://peerreviewcongress.org/abstract/the-influence-of-promotional-language-on-evaluations-of-biomedical-literaturea-randomized-controlled-trial/)).
- Hedging-trust experiments are mixed ([Jensen 2008](https://scholars.houstonmethodist.org/en/publications/including-limitations-in-news-coverage-of-cancer-research-effects/); [Ratcliff et al.](https://summit.sfu.ca/_flysystem/fedora/2024-07/Ratcliff-et-al_Covid-Uncertainty-Disclosure.pdf); [Gustafson & Rice](https://rrice.faculty.comm.ucsb.edu/A145GustafsonRice2019UncertaintyFramesScienceCommunicationTopics.pdf)).

The pressure toward redundancy in LLM output is documented:

- verbosity rewarded as quality ([Kim et al.](https://arxiv.org/html/2511.12573v1));
- measured excess length, including "vacuum-filling" ([arXiv 2601.00624](https://arxiv.org/html/2601.00624v1));
- certainty preferred over hedges ([Zhou et al.](https://arxiv.org/html/2401.06730v2));
- scope limits dropped ([Peters & Chin-Yee](https://arxiv.org/abs/2504.00025)).

The case for deletion therefore rests on information and on reader inference, not on a measured loss of credibility. An unneeded "every" either says nothing or invites the reader to wonder what the exceptions were.

> **Rule NT (NEEDED?, then TRUE?).** Apply it to every universal quantifier (all, every, each, any, no, none, always, never, uniformly, without exception), completeness adverb (entirely, completely, fully), superlative (the most/least, -est forms, best, only, first as a rank, unique) and booster (clearly, crucially, fundamentally, importantly, indeed, very, remarkably, strikingly, substantially, dramatically).
>
> **NEEDED?** Keep the word only if it (a) contrasts scope against a plausible expectation of exceptions, (b) states a scope the reader cannot recover from the definitions, the table or the previous two sentences, or (c) is the paper's claim itself, stated once in the abstract, introduction or conclusion, with a number. Otherwise delete it, **even if it is true**.
>
> **TRUE?** A kept literal universal or superlative must cite a computed fact that holds: a count equal to the total, an argmax over the cited table, or a predicate macro (§3.3). A rhetorical absolute with no finite set behind it is deleted or replaced with a number.
>
> **Scope (A4).** Rule NT never applies to text the author supplied in an edit message. That text is protected verbatim in every layer and in the final build (§8.4). Rule NT governs only agent-written text.

Detection runs in two stages, and the order matters, because only words that pass NEEDED? cost a TRUE? check.

- **Stage A (structural, deterministic).** It attaches each hit to its scope noun by parse and looks the noun up in a **set registry** built from the paper: sets defined with a cardinality or enumeration ("the six classes", "horizons h ∈ {…}"), table row and column headers, and the same noun with a numeral or definite back-reference in the previous two sentences. If the scope is a registered set and no contrast marker is near ("including", "even", "although", "except", "despite"), the hit becomes a REDUNDANT-CANDIDATE. So does a superlative that restates the argmax of a table cited in the same sentence.
- **Stage B (deletion equivalence).** An LLM judge, or bidirectional NLI where the text is not math-heavy, sees the sentence with and without the word, plus the paragraph and the relevant definitions. It must name the exception or contrast the word excludes, or the word goes. This stage runs on candidates and on every completeness adverb and booster. Boosters default to deletion in results prose.

Every deletion is logged with a reason code (REDUNDANT-SCOPE, BOOSTER, SUPERLATIVE-RESTATES-TABLE, UNSUPPORTED-UNIVERSAL) so the author can audit the batch.

The notation guide's preferred magnitude words, "substantially", "materially" and "modestly" (repo), fall in the booster class. Under Rule NT they survive only as generator-emitted magnitude words tied to stated thresholds, like the comparison words of §3.3.

Basis: **P + I.** The NEEDED? half is prescriptive (Grice, Cochrane) plus the author's judgment. The TRUE? half is the deterministic predicate mechanism of §3.3. The *need* for the rule is **D**, from Peters & Chin-Yee and Zhou et al. No published system implements the deletion-equivalence test.

### 4.4 Corpus calibration uses Fightin' Words against a JFEc/JoE/JBES corpus

**Fightin' Words.** For flagging what a draft over- or under-uses relative to a corpus, the best-suited statistic is Monroe, Colaresi and Quinn's log-odds ratio with an informative Dirichlet prior, scored as a z-score. Its variance handles very unequal sizes: a 12,000-word draft against a corpus of a million words. The prior pushes function words out of the top lists ([Monroe et al. 2008](http://languagelog.ldc.upenn.edu/myl/Monroe.pdf)).

**Alternatives.** The standard corpus-linguistics alternative pairs log-likelihood G² for significance with Log Ratio for magnitude ([UCREL calculator](https://ucrel.lancs.ac.uk/llwizard.html)).

**Python tools.** These cover the remaining statistics:

- pybiber for Biber's 67 register features ([pybiber](https://browndw.github.io/pybiber));
- PassivePy for passive share, with 98% agreement with human coding ([PassivePy](https://pypi.org/project/PassivePy));
- faststylometry for Burrows' Delta ([faststylometry](https://pypi.org/project/faststylometry)).

**Vale.** Vale rules are plain YAML: `existence` takes a `tokens:` list and `substitution` a `swap:` map ([Vale docs](https://docs.vale.sh/checks/existence)). LaTeX is not among Vale's supported formats ([Vale docs index](https://docs.vale.sh/llms.txt)).

**Copyright.** The legal line runs through the source of the copies. *Bartz v. Anthropic* held that training on lawfully acquired books was fair use, but that building a library from pirated copies was not ([Publishers Weekly](https://www.publishersweekly.com/pw/by-topic/digital/copyright/article/98089-federal-judge-rules-ai-training-is-fair-use-in-anthropic-copyright-case.html)). OUP reportedly permits non-commercial text mining of its journals (library guides, secondary) ([UChicago guide](https://guides.lib.uchicago.edu/textmining/journals)). JFEc screens manuscripts "with services provided by third parties" ([JFEc General Instructions](https://academic.oup.com/jfec/pages/General_Instructions)).

**Exemplar-conditioned rewriting copies text.** LLMs reproduce verbatim, paraphrased and idea-level content, and fine-tuning reduced verbatim copying but increased paraphrase reuse ([Lee et al., WWW 2023](https://arxiv.org/abs/2203.07618v2)). The only research on few-shot style transfer concerns individual authors on Reddit, not academic register ([Patel et al.](https://arxiv.org/abs/2212.08986)).

***Inference.*** The corpus design follows from these findings:

- Keep 60–150 legitimately held PDFs from JFEc, JoE, JBES, JAE and IJF local and gitignored.
- Prefer recent but mostly pre-2023 papers. Human promotional language keeps rising: hype adjectives appeared in 72% of NIH grant abstracts in 1985 and 97% in 2020 ([Millar et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC9412227)). Kobak's counterfactual skips 2023 as possibly LLM-contaminated.
- Commit only statistics: n-gram counts, per-paper feature rates, and hashed long n-grams.
- Segment by section, because norms differ. Compute every statistic per paper and per section type, so the draft is placed in a distribution of papers rather than pooled sentences (§4.3.4).
- Keep a term only if it appears in at least five corpus papers, so one author's habits do not become the norm.
- Report under-use only as suggestions, because a short draft has little statistical power.
- Feed the LLM abstracted **phrase frames** for standard statements (Diebold–Mariano, MCS, QLIKE), each with a bibkey and page pointer, never raw corpus sentences.
- Run a Python linter over detexed text, because Vale cannot read `.tex`. Vale is optional, via pandoc.

### 4.5 Removal buys register quality, not invisibility

Removing marker words does not hide AI drafting from expert readers. Five frequent ChatGPT users, voting together, misclassified **1 of 300** articles. They kept **100% detection** of text "humanized" with a guide to their own clues, shifting from vocabulary cues (57.1% of correct explanations) to other cues (42.3% vocabulary) ([Russell, Karpinska & Iyyer, ACL 2025](https://arxiv.org/html/2501.15654v2)).

What readers penalize is broader than vocabulary:

- **Expert "slop" taxonomy.** Information utility (density, relevance), information quality, and style (repetition, templatedness, verbosity) ([Shaib et al. 2025](https://arxiv.org/html/2509.19163v2)).
- **Professional edits.** Writers who edited 1,057 LLM paragraphs produced text experts preferred, and automatic editing "show[s] promise" ([Chakrabarty, Laban & Wu, CHI 2025](https://arxiv.org/abs/2409.14509)).
- **Suppression method matters.** Careful suppression (Antislop's FTPO) cut about 90% of slop patterns while keeping quality within about 1%. DPO cost 6–15 points ([Antislop](https://arxiv.org/html/2510.15061)).

**Prescriptive economics guides supply the phrase rules:**

- **Cochrane:** "It should be noted that" is "particularly obnoxious". Use "use", not "utilize". No adjectives on one's own results. "Readers don't stick around to find the punchline in Table 12". "No number should appear in a table that is not discussed in the text" ([Cochrane, Writing Tips](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)).
- **Head:** the introduction runs hook, question, antecedents, value-added, then roadmap ([Head](https://blogs.ubc.ca/khead/research/research-advice/formula)).

***Inference.*** Frame the scrub as register alignment that sits beside a disclosure statement. A scrub aimed at detectors would read as concealment under OUP's policy (§7.1). When prescription and corpus conflict, corpus norms win on register features (pronouns, passive share, hedging, sentence length). Cochrane wins on throat-clearing, self-praise and AI-isms, which both sources reject. The LLM pass should target what the slop and LAMP taxonomies name:

- cut sentences that only assert importance;
- replace generic claims with the paper's macros and citations;
- keep one name per estimator, using the existing [NOTATION_GUIDE_2026-10-15.md](../NOTATION_GUIDE_2026-10-15.md) as the term list (repo);
- calibrate hedges to the test statistics (§4.3.6);
- run every universal, superlative and booster through Rule NT (§4.3.7).

No study measures whether any of this changes referee or editor ratings of academic papers.

---

## 5. Review loop: one screen per decision, friction only where claims change

*Status: research only.* The author's decision A1 (no review cards) supersedes this section's card inferences. A2 replaces per-edit review with one read-and-correct pass per layer (§8.5). The evidence stays because three of its findings carry into the lock design:

- friction belongs only on changes to what the paper asserts (§5.1);
- direction is settled before detail (§5.3);
- AI reviewers are detectors, not judges (§5.4).

### 5.1 Co-writing studies favor anchored, typed edits over chat

The strongest controlled evidence comes from InkSync. Each suggestion is an executable in-document edit with a flag for whether it **adds new information**:

- Chat that did not apply its edits was the slowest editing condition.
- **Major factual errors appeared in 14.8% of documents from conversational conditions vs 6% from non-conversational ones.**
- In a planted-error study, a "Warn + Verify" step raised the share of inaccurate edits prevented from **23.4% to 43.5%**. Verifications averaged **44 seconds**, and the new-information flag matched annotators 97.5% of the time ([InkSync, Laban et al. 2023](https://arxiv.org/html/2309.15337v1)).

AnchoredAI found that comments anchored to text spans cut words pasted per paste from 199.1 to 35.2. Ownership rose from 2.3 to 4.0. **Workload also rose**: mental demand went from 51.8 to 80.7 ([AnchoredAI](https://arxiv.org/html/2509.16128v1)).

Ink & Switch's Patchwork staged AI edits as a branch the author partially merges. Its bot explained each change **by citing the style guide**, and AI-written edit descriptions were among its most useful outputs. Litt also judged branch review "too heavy" for many small edits ([Litt 2024](https://buttondown.com/geoffreylitt/archive/towards-universal-version-control-with-patchwork)).

Users of AI code editors ask for per-change decisions and review "in page order" ([Cursor forum](https://forum.cursor.com/t/can-you-review-agent-edits-in-page-order/171421)). A Claude Code issue contrasts its all-or-nothing edit acceptance in VS Code with Cursor's per-hunk accept and reject ([claude-code #42448](https://github.com/anthropics/claude-code/issues/42448)).

***Inference (superseded by A1).*** Agents should emit span-level edits typed as one of three classes:

- **wording;**
- **structure;**
- **claim, number, hedge or citation**, which carries a warning flag.

Verification effort goes only to the third class. Wording edits get a bulk lane, auto-applied, attributed and revertible. Each edit's rationale should be a rule or comment ID, not free prose.

### 5.2 Code review supplies the sizing, the ordering and the description rule

Google's guidance:

- The right unit is "one self-contained change". About 100 lines is usually reasonable, and 1,000 usually too large.
- Refactors go in a separate change from behavior changes, and large changes from a trusted automatic tool are an exception ([Google: Small CLs](https://google.github.io/eng-practices/review/developer/small-cls.html)).

A 2006 vendor study at Cisco recommends reviewing **fewer than 200–400 lines at a time** in sessions of 60–90 minutes ([SmartBear summary](https://www.stickyminds.com/article/largest-case-study-code-reviews-ever)). An MSR 2026 study of about 80,000 pull requests found that **stating the type of feedback wanted** is the description element most associated with acceptance and reviewer engagement ([Pirouzkhah et al.](https://arxiv.org/html/2602.14611)).

***Inference (superseded by A1).*** By analogy, a card is one rhetorical move with at most about three sentences of diff, and it states the decision asked ("accept or reject", "pick A or B", "your judgment on: …"). Mechanical sweeps (reflow, macro renames, typo batches) travel as one bulk card with a count and a sample. A sitting is tens of cards, not hundreds. This is transfer from code review, not tested on prose.

### 5.3 Layered review is waste avoidance, not a measured saving

Google's reviewer guidance for navigating a change:

1. Take a broad view first: does the change make sense at all? If not, "respond immediately".
2. Send major design problems "immediately", before reviewing the rest, because "much of the code may be removed or rewritten" ([Google: Navigating a CL](https://google.github.io/eng-practices/review/reviewer/navigate.html)).

AnchoredAI found that anchored help did not change behavior on global restructuring tasks ([AnchoredAI](https://arxiv.org/html/2509.16128v1)). No controlled study measures the time saved by top-down review of documents.

***Inference (superseded by A1; its top-down order survives in A2).*** Beats map to Google's step 1, bullets to step 2, and final text to step 3. No bullet card should be served under an unaccepted beat, and no wording card under an unaccepted bullet. A rejected beat cancels its dependent cards. The saving is avoided rework, and its size is unmeasured.

### 5.4 AI reviewers: verifiers beat referee simulators, so pool them and make them refute

Referee simulators match human scores about as well as humans match each other, which is weakly:

- The Stanford Agentic Reviewer's Spearman correlation with one human was 0.42, against 0.41 between two humans ([paperreview.ai](https://paperreview.ai/tech-overview)).
- LLM reviewers inflate scores. Agent Laboratory's automated reviewers gave 6.1 of 10 where humans gave 3.8 ([Agent Laboratory](https://arxiv.org/html/2501.04227)).
- Reviewer feedback induced fabricated ablations ([Jr. AI Scientist](https://arxiv.org/html/2511.04583)).

Verifiers do better:

- **Detect-then-verify** reached **83.2% human-confirmed precision** and 60% recall on injected errors. Only 75.8% of its proposed fixes were correct ([Bianchi et al. 2025](https://arxiv.org/html/2512.05925v1)).
- **Staged review.** AAAI-26's aspect-staged review raised recall on planted errors from 0.43 to 0.64. The final synthesis dropped true Significance findings (0.53 at the stage, 0.45 after synthesis). Authors rated the AI reviews above human ones on 6 of 9 criteria, but worse on "overemphasizing minor issues" ([AAAI-26](https://arxiv.org/html/2604.13940v1)).
- **Structured prompts** added 32.2 points of detection within Claude Code ([Zubeldia et al., NeurIPS 2026](https://neurips.cc/virtual/2026/poster/152885)).
- **Pooling** across systems raised detection of 100 planted errors from 71 for the best single system to 91–93. The 7 errors no system caught were all **omissions** ([Litvak 2026](https://paullitvak.com/p/how-well-does-ai-peer-review-work)).

Refine, the economics tool, is the only reviewer aimed at economics.

- Its co-founder rates it at about the level of a better-than-average PhD student (a non-independent source). As of December 2025, he named table parsing as its main limitation on empirical papers ([Golub](https://markusacademy.substack.com/p/modern-ai-for-economics-research)).
- The AEA and the Econometric Society use it at the R&R and conditional-acceptance stages ([Refine blog](https://www.refine.ink/blog/economics-journal-partnerships); [AEA](https://benny.aeaweb.org/news/member-announcements/2026-aug-14)).
- No evidence was found of AI review at JFEc.
- Persona prompts ("a senior econometrician writing for a top journal") are practitioner advice and have not been tested in the studies found.

***Inference (revised for A1).*** Use AI reviewers as error detectors that feed the agent queue, never as score targets.

- At milestones, run several aspect-specialized reviewers with structured checklists. Use more than one model family where possible. The checklists cover:
  - identification and inference, including HAC and overlapping returns;
  - sample construction and look-ahead;
  - table, text and macro consistency;
  - multiple-testing disclosure;
  - the out-of-sample protocol;
  - literature positioning;
  - exposition.
- Merge duplicate issues, then have a separate verifier try to refute each one against the source and CSVs.
- Pass only confirmed issues to the agent queue. Agents fix what they can inside the locks; anything that would change a locked claim becomes a conflict (§8.5).
- Log any request for a new number, test or citation as a proposal. Never hand it to a writer: nothing acts on it until an author correction adopts it.
- Ask explicit "is X reported?" questions to cover omissions.

---

## 6. Citations: fetch, never type; check support where it can fail

### 6.1 Checking that a reference exists is solved and scriptable

Even with web search, frontier models produce unreliable BibTeX. GPT-5, Claude Sonnet 4.6 and Gemini 3 Flash reached 83.6% accuracy on BibTeX fields, but only **50.9% of entries were fully correct**. Accuracy fell 27.7 points from popular to recent papers. The authors' clibib tool fetches entries deterministically, then revises them in a separate gated step. It raised fully correct entries to 78.3% with a 0.8% regression rate, against 4.8% for single-stage tool loops ([Rao & Callison-Burch 2026](https://arxiv.org/abs/2604.03159)).

Hallucinated citations are reaching accepted papers:

- GPTZero confirmed at least 100 hallucinated citations in 51 accepted NeurIPS 2025 papers ([GPTZero](https://gptzero.me/news/neurips/)).
- An audit of 10 LLMs found hallucination rates from 11.4% to 56.8% ([Naser 2026](https://arxiv.org/abs/2603.03299)).

Scriptable checkers exist:

- RefChecker matches .bib entries against Semantic Scholar, OpenAlex and Crossref ([academic-refchecker](https://pypi.org/project/academic-refchecker/3.0.101/)).
- bibtex-verifier uses Crossref, with an OpenAlex fallback ([bibtex-verifier](https://pypi.org/project/bibtex-verifier/)).

A metadata-only check passes a fabricated reference that carries a real DOI or title (same source).

### 6.2 Checking that a citation supports its sentence is only prototyped

Claim-support tools are research prototypes:

- **CiteGuard** scores 65.4% on CiteME, against 69.7% for humans ([CiteGuard](https://arxiv.org/abs/2510.17853v2)).
- **scite**'s supporting and contrasting labels scored F-measures of 0.0 to 0.58 in an independent evaluation ([Bakker et al.](https://journals.indianapolis.iu.edu/index.php/hypothesis/article/view/26528)). The developers dispute it.
- **PaperQA2** indexes a local PDF folder, answers with page-level citations, works with Claude through LiteLLM, and scored 85.2% precision on LitQA2. Experts validated 70% of the contradictions it flagged ([PaperQA2](https://arxiv.org/abs/2409.13740v2); [paper-qa README](https://github.com/Future-House/paper-qa)).
- **OpenScholar**'s datastore is 45 million open-access papers ([OpenScholar](https://arxiv.org/abs/2411.14199)). That likely under-covers a literature that sits mostly in Elsevier, OUP and Wiley journals.

Programmatic access:

- **OpenAlex** has required a free key since 13 February 2026, with a $1/day free budget ([OpenAlex auth](https://developers.openalex.org/api-reference/authentication)).
- **Semantic Scholar** keys start at 1 request per second ([S2 tutorial](https://webflow.semanticscholar.org/product/api/tutorial)).

***Inference (revised for A1).*** Support checks feed the agent queue. The pipeline is:

1. Extract every `\cite` sentence.
2. Retrieve evidence with PaperQA2 from the local PDF, falling back to the abstract.
3. Have an LLM return a verdict (supported, partial, not supported, cannot assess) with a quoted passage and page.

An agent acts only on "not supported": it rewrites the sentence or swaps the citation. If neither is possible without changing a locked claim, the case is shown to the author as a conflict (§8.5). Verdicts are keyed to a sentence hash, so they go stale when the sentence changes.

### 6.3 Writers may cite only verified keys

Jr. AI Scientist's writer edited the .bib itself, "sometimes with wrong authors or nonexistent papers". The fix was a verified bibliography the writer cannot change ([Jr. AI Scientist](https://arxiv.org/html/2511.04583)). AI Scientist v1's refinement rule reads "DO NOT ADD ANY NEW CITATIONS!" ([v1 code](https://raw.githubusercontent.com/SakanaAI/AI-Scientist/main/ai_scientist/perform_writeup.py)). A citation that more than three LLMs independently produced was accurate 95.6% of the time ([Naser](https://arxiv.org/abs/2603.03299)). But database retrieval, not model agreement, is the safe acceptance test.

### 6.4 This repository's bibliography needs one source of truth

`writeup/references.bib` has **148 entries**. Only 73 lines carry a `doi =` field, 28 mention SSRN and 12 mention arXiv. A second copy, `writeup/overleaf-upload/references.bib`, has 71 entries and one DOI line. The note that counted these did not establish whether the upload copy is still used (repo, [references.bib](../references.bib)).

***Inference.*** The entries without DOIs (SSRN working papers, books) are where metadata checks are weakest and a deterministic re-fetch helps most. The canonical file should be the only one agents touch. Flatten's `references2.bib` should stay a generated export.

---

## 7. Disclosure and replication bound the system from outside

### 7.1 OUP's July 2026 policy governs JFEc

JFEc's own instructions contain **no AI clause**, so OUP's "Policy on AI Use & Disclosure for OUP Journals Authors" (last updated 13 July 2026) applies ([JFEc General Instructions](https://academic.oup.com/jfec/pages/General_Instructions); [OUP policy](https://academic.oup.com/pages/for-authors/journals/preparing-and-submitting-your-manuscript/ai-principles-for-oxford-journals/policy-on-ai-use-and-disclosure-for-oxford-journals-authors)).

**What triggers disclosure.** Substantially editing, rewriting or reorganizing text, analyzing or visualizing data, and generating or refining code all typically require it. Spelling and grammar help and accurate reference formatting do not.

**Where it goes.** Disclosure goes in **both the cover letter and the Acknowledgements**, and in Methodology "if AI use was integral to the research methodology". It must give the **tool name and version, how it was used, and the author's role in directing and reviewing output**. Dates and the sections affected are strongly recommended for complex cases.

**When.** The statement is updated at each revision, and the author asserts compliance at submission.

**What is prohibited.** Presenting AI-created content as original human research data or results is prohibited, and AI cannot be an author.

The fetch tool paraphrased the page, so the wording should be checked on the page before it is quoted in a cover letter. OUP's own survey found that 46% of journal authors had not fully disclosed AI use, "particularly in narrative drafting" ([OUP news, Sept 2026](https://corp.oup.com/news/supporting-researchers-to-disclose-ai-use-with-confidence/)).

A third-party summary claims OUP bars AI from "generating arguments, writing analysis, or drawing conclusions". It conflicts with the primary journals policy, which lists text generation as disclosable, and it may reflect OUP's books guidance ([Enago](https://www.enago.com/responsible-ai-movement/publisher-ai-guidelines/oup-ai-guidelines)).

The repository already holds the raw record OUP asks for (repo, `git log` trailers):

- **114 commits** to the paper sources and generators between 15 and 28 October 2026 carry `Co-Authored-By` trailers naming Claude Opus 5.5, 65 of them the 1M-context variant.
- Since June, the trailers name five model versions: Opus 4.8, Opus 5, Opus 5.5, Fable 5 and Fable 5.1.

***Inference.*** A disclosure generator can read the trailers and changed paths to list model versions, date ranges and sections touched. It writes the Acknowledgements sentence and the cover-letter paragraph. A check then fails the final build if the statement is missing or names fewer models than the log. A draft wording, not an OUP template:

> "The authors used Anthropic's Claude models [versions] through Claude Code ([dates]) to write and refine analysis code, to typeset numerical results as LaTeX macros produced by that code, and to draft and edit text from author-written outlines in Sections [x–y]. The authors specified all arguments and decision rules, reviewed and edited all generated text and code, and take full responsibility for the content."

### 7.2 JFEc's hard limits and replication package

JFEc requires the following ([JFEc General Instructions](https://academic.oup.com/jfec/pages/General_Instructions)):

- **Replication.** Empirical papers must "upload source files and relevant data files for editorial replicability review" (for submissions from 13 March 2023). Data are a condition of acceptance unless an exemption is requested in the cover letter.
- **Length and front matter.** An abstract of **at most 100 words**, **2–6 keywords** and JEL codes. Papers "should typically not exceed 40 double-spaced typewritten pages".
- **Style and format.** Chicago Manual of Style (15th edition), footnotes at the end of the text, and LaTeX submitted as a locally generated PDF.
- **Figures.** At least 300 dpi, with alt text.

JFEc also offers an optional, AI-driven Paperpal Preflight check (same source). The Data Editors' template README supplies the table-to-program map and the "All numbers provided in text" checkbox ([template README](https://raw.githubusercontent.com/social-science-data-editors/template_README/master/template-README.md)). The CMU authors recommend that venues require "the log traces of the entire research process and the generated code" ([Luo et al.](https://arxiv.org/html/2509.08713)).

***Inference.*** The abstract, keyword, JEL and page limits are trivial to assert and should block the final layer, unlike style flags. If the options data are licensed, the exemption route should be planned early. The generators' headers make the README's program map generatable.

### 7.3 Neighboring outlets, posting and confidentiality

Other venues place the disclosure differently:

- **Elsevier** (JFE, JoE) wants a separate AI declaration "immediately before the references" and recommends keeping logs and prompts ([Elsevier policy](https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals)).
- **AEA** journals require a brief description "during the submission process" ([AER editorial policy](https://www.aeaweb.org/journals/aer/editorial-policy)).
- **SSRN** requires the disclosure "in the PDF" for "any material use of generative AI" and prohibits fabricated citations ([SSRN AI policy](https://www.elsevier.support/ssrn/answer/AI)).

Reviewers at Elsevier and Wiley must not upload manuscripts to AI tools ([Elsevier](https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals); [Wiley](https://www.wiley.com/publishing/article/ai-guidelines)). OUP's principles warn that AI tools "will sometimes grant the tool's owner rights to the content uploaded" ([OUP AI Principles](https://academic.oup.com/pages/for-authors/journals/preparing-and-submitting-your-manuscript/ai-principles-for-oxford-journals)).

***Inference.*** One generated statement can be rendered in each venue's location. Before any external reviewer service sees the unpublished paper, the author should confirm that the Claude plan's data settings and the reviewer's terms are compatible. Refine contractually excludes training on uploads ([Golub](https://markusacademy.substack.com/p/modern-ai-for-economics-research)).

---

## 8. Recommended design

### 8.1 Layer model

One source and five layers, plus a provenance view. Three layers already exist as builds: L1 and L2 are generated by `make_paper_beats.py`, and L3 is `paper_rehaul.tex` itself. L4 and the provenance view are new wrappers. Every layer compiles on Overleaf by selecting its file.

Under A2 the layers are also a **sequence**:

1. The paper starts at L0–L1, in plain, unpolished sentences.
2. The author reads that simple version once, as a PDF on Overleaf, and sends edits as messages. An agent applies each edit and records it as a lock (A3, §8.5).
3. Only then is L2 generated from the locked L1. The author reads only what is new and sends edits on it.
4. L3 follows the same way.
5. L4's editing (scaffold removal, typography, the scrub of §8.4) comes last and may not change locked content.

No layer is served twice, except where a later edit conflicts with an earlier lock. Collaborators may edit any served layer on Overleaf. Their edits flow back through the pull, are recorded with their source, and lock only when the author adopts them (A5, §8.5).

| Layer | What it shows | How it is produced | Author | AI | Basis |
|---|---|---|---|---|---|
| L0 Contribution | Abstract plus a one-paragraph contribution (Cochrane, Minto), plain sentences | Source text in `paper_rehaul.tex` | Reads the PDF of the simple version once, together with L1; sends edits | Drafts it under the seeded locks; applies each edit, records it as a lock, pushes the updated PDF | T-genre (Dramatron logline, DOC premise); D for the ≤100-word limit (JFEc) |
| L1 Beats | Top-level `\beat` claims only, tables kept, the decision register printed in front, short beat IDs in the margin | `make_paper_beats.py` → `paper_rehaul_beats_top.tex` (exists) | Reads the PDF of the simple version once; sends edits; each corrected beat becomes a locked claim | Generates under the locks; applies edits at their layer, records them as locks, propagates, pushes | T-genre (Dramatron, DOC); T-code (direction first); I (A2, A3) |
| L2 Bullets | All bullets, floats stubbed, short beat IDs in the margin | `make_paper_beats.py` → `paper_rehaul_beats.tex` (exists) | Reads the PDF of the new bullets once, in plain form; sends edits | Generates from the locked L1 under the locks; checks against locked claims; applies and locks edits | T-genre; D for number gates (data-to-paper); I (A2, A3) |
| L3 Attached appendix | Full draft with `rehaul/A_*.tex`, author notes on | `paper_rehaul.tex` with `\paperlayer{draft}` (exists as default) | Reads the PDF of the new appendix material once; sends edits | Generates from the locked L2; applies and locks edits | I (A2, A3) |
| L4 Final | Scaffolding removed: no list markup, labels or margin IDs, notes off, `\rawseriesfalse`, parked text absent, `\rhPending` and `\editc` are errors | New wrapper `paper_final.tex`, then flatten | Reads the finished PDF once; sends edits like any other | Runs editing only after L0–L3 are locked; may not change locked content or author text (C25, C26) | I (Option A, §2.3; A2, A4) |
| Provenance view | L3 with each number macro annotated: macro, description, generator, input hash, decision stamps | New wrapper `paper_provenance.tex` | Only to explain a flip, on demand | Used by agents and AI verifiers | D (data-to-paper chain; CMU 55→82%) |

The layer selector and beat macros are an untested sketch (I). The wrappers define `\paperlayer` before `\input{paper_rehaul}`. An unknown name stops the build, which is the lesson of Quarto's silent ignore (§2.2).

Two DITA rules shape the selector (§1.5; D in structured authoring, T-genre for the paper):

- **Three actions, not two.** Each layer gets a DITAVAL-style action: include, exclude or **flag**. Flag builds a reviewer PDF in which every layer is present and colour-coded.
- **Resolve before filtering.** Values (number macros, lock macros) are resolved before layers are filtered, because DITA documents that the two orders can disagree.

Across layers, only whole claims and numbers are reused, never sentence fragments.

```latex
% paper_final.tex (new, 1 line):  \def\paperlayer{final}\input{paper_rehaul}
% paper_rehaul.tex preamble (sketch; \makeatletter context):
\providecommand{\paperlayer}{draft}
\newif\ifbulletform
\def\rh@L{draft}\ifx\paperlayer\rh@L \bulletformtrue \authornotestrue \else
\def\rh@L{final}\ifx\paperlayer\rh@L \bulletformfalse \authornotesfalse \rawseriesfalse \else
\def\rh@L{provenance}\ifx\paperlayer\rh@L \bulletformtrue \authornotesfalse \else
\PackageError{paper}{Unknown layer \paperlayer}{Use draft, final or provenance.}\fi\fi\fi
\ifbulletform   % beats and bullets as lists
  \newenvironment{beats}{\begin{itemize}}{\end{itemize}}
  \newcommand{\beat}[2]{\item\textbf{#2}\ }     % #1 = stable beat ID, #2 = label
  \newenvironment{support}{\begin{itemize}}{\end{itemize}}
  \newcommand{\pt}{\item}
  \newcommand{\bul}[1]{#1}\newcommand{\fin}[1]{}
\else           % final: each beat is a paragraph, its claim the topic sentence
  \newenvironment{beats}{}{\par}
  \newcommand{\beat}[2]{\par}                   % label dropped
  \newenvironment{support}{}{}
  \newcommand{\pt}{}
  \newcommand{\bul}[1]{}\newcommand{\fin}[1]{#1}
\fi
```

The source then reads `\beat{b:res-everybar}{Accuracy on every bar.} The full predictor set ...`, followed by `\begin{support} \pt ... \end{support}`. The ID never changes once minted. IDs are short and human-readable, never tool-generated UUIDs, which become merge hazards in tools that write them into source (Logseq; §1.5). A one-time script converts the existing `\item \textbf{...}` beats. `make_paper_beats.py`'s lexer already tracks list nesting, and the migration is accepted only if the bullet-layer PDF text is unchanged, the same test `flatten.py --check` applies. `\bul{}` and `\fin{}` carry the few words that must differ between the bullet and final forms. Option B paragraphs (`\ifprose` blocks with a `% src-hash:` line) are allowed in the introduction and conclusion only (§2.3).

### 8.2 File layout

Existing files are unmarked; additions are marked NEW. Everything stays under `writeup/`, and the replication README and disclosure are generated from the same records.

```text
writeup/
  paper_rehaul.tex                 preamble + body; + \paperlayer selector, beat macros (8.1);
                                     + \auth{ID}{text} for protected author text (prints text as is);
                                     + margin beat IDs in review builds only
  paper_final.tex                  NEW wrapper, layer L4
  paper_provenance.tex             NEW wrapper, provenance view
  paper_rehaul_beats.tex           generated L2;  paper_rehaul_beats_top.tex  generated L1
  make_paper_beats.py              + learns \beat/support/\pt; + writes beats/manifest.json (NEW:
                                     beat ID -> section, label, claim hash, bullet sentence hashes);
                                     + prints the decision register in front of L1
  decisions.yaml                   NEW the register: one entry per decision (author's message words
                                     verbatim + date, source: chat, Overleaf + editor, or inferred (T1/T2,
                                     with evidence), layer, parent,
                                     status, supersedes, rationale, EARS rule, binds, forbids,
                                     exceptions, times); each chat edit also kept as a dated
                                     ADDED/MODIFIED/REMOVED delta; seeded from D1-D20 and the dated
                                     author quotes; written by agents
  where_used.py                    NEW before a decision changes: every number, sentence, table and
                                     layer that cites it (from locks.py's ripple graph)
  locks.py                         NEW reads decisions.yaml; the only source of locked parameters for
                                     generators; writes generated/locks.tex and stamps (hashing only
                                     the parameters each generator reads); keeps the ripple graph
                                     number -> sentence -> abstract/intro/conclusion; checks C23
  apply_edit.py                    NEW (or agent skill) the seven-step chat procedure (8.5): classify,
                                     map numbers via the .aux of the commit printed on the PDF, locate
                                     (title, label, macro -> generator, quote), inventory, apply,
                                     verify, report; writes the decision entry and delta
  layer_state.json                 NEW per layer: served commit, locked or not, date (no layer twice)
  deliberation/PROTOCOL.md         the escalation rules (8.10; authoritative), with SCHEMA.md
  deliberation/digest.py           the mechanical escalation filter (tests/test_digest.py, 23 tests)
  deliberation/queue.jsonl         open items;  deliberation/log.jsonl  the record
  deliberation/<date>/             saved rounds: solver answers, adjudication, digest
  ../.claude/skills/deliberate/    the project skill an agent runs before any question
  flips.md                         NEW generated: flipped and uncertain claims, conflicts; the author's
                                     only list
  rehaul/*.tex, rehaul/A_*.tex     beats via \beat{id}{label}; one sentence per line
  rehaul/parked/<file>.parked.tex  NEW dated parked text moved out of live files, keyed by beat ID
  generated/*.tex                  + header stamp: generator blob hash, SHA-256 of each input,
                                     decision IDs and parameter values
  generated/stamps.json            NEW macro -> decision IDs + parameter values it was computed under
  generated/locks.tex              NEW lock macros for scope wording (\lockSpan ...), from locks.py
  generated/*_predicates.tex       NEW sign / rank / significance / comparison-word macros
  tests/check_pending.py           exists
  tests/check_numbers.py           NEW no typed digits in live text (gate)
  tests/check_stale.py             NEW header stamps match generator and inputs (gate)
  tests/check_claims.py            NEW direction words agree with predicates (gate)
  tests/check_beats.py             NEW beat IDs present, unique, stable; manifest consistent (gate)
  tests/check_locks.py             NEW C20-C30: parameters from locks.py, stamps match the governing
                                     decisions, comparison locks, locked claims, layer order,
                                     protected author text, exit codes, minimal-diff
                                     push, drift gate on pulled edits (gate)
  tests/check_bib.py               NEW Crossref/OpenAlex/S2 verification, cached (gate before push)
  tests/check_jfec.py              NEW abstract <=100 words, 2-6 keywords, JEL, page budget,
                                     disclosure present, zero \rhPending / \editc in L4 (gate)
  claims/claims.yml                NEW locked claims: beat ID, decision IDs it rests on, macros,
                                     predicate, table label, generator, input hash, review record
                                     (suspect/valid/invalid, who, when, comment) (successor to
                                     CLAIMS_LEDGER_*.md)
  lit/bib_verified.json            NEW per-key source, id, match score, date
  lit/litmap.jsonl                 NEW bibkey -> sentence hash, file:line, verdict, quote, page
  lit/pdfs/                        NEW gitignored local PDFs (PaperQA2 index lives outside the repo)
  style/voice.md                   NEW drafting guide for the simple version, loaded by every agent
  style/rules/                     NEW tier V/S/C rules, each with its source; field whitelist
  style/sets.json                  NEW set registry for Rule NT: defined sets and their cardinality
                                     (classes, bars, horizons, tests), with the macro holding each count
  style/corpus_stats/              NEW committed statistics only (counts, rates, hashed 8-grams)
  style/frames/                    NEW phrase frames (DM, MCS, QLIKE, MZ) with bibkey+page
  style/scrub.py                   NEW pipeline driver (8.4); runs only on locked layers
  agent_queue.jsonl                NEW advisory findings agents work off and log; never served
  disclosure/ai_use.json           NEW from git trailers: models, dates, paths
  disclosure/ai_disclosure.tex     NEW Acknowledgements sentence; cover-letter paragraph
  overleaf_sync/                   + pushes the layer being served and paper_final; + pull unflattens
                                     collaborators' edits on beats_top.tex / beats.tex back to source
                                     lines through their maps, so push-beats pulls layer edits instead
                                     of refusing them (A5); layer_header_fix.py keeps reverting stray
                                     header edits; autopush.sh pulls before it pushes; 2 MB file gate
  reports/                         this document
```

### 8.3 Checks at each layer

Gates block a build or a push. Advisory findings go to `agent_queue.jsonl`, which agents work off and log; they never reach the author (A1). Only lock conflicts and flipped or uncertain claims reach the author, through `flips.md` (§8.5). Checks run before any LLM critic (data-to-paper's order). Every gate is a hard error with a non-zero exit, never a warning that CI does not see (C27).

| ID | Check | Layers | Type | Implementation | Basis |
|---|---|---|---|---|---|
| C1 | Compile, undefined references and citations, duplicate labels | all | Gate | Exists (`flatten.py --check`, build) | D (AI Scientist mechanical gates) |
| C2 | No typed digit in live text outside macros, `\ref`, `\cite`, labels and a whitelist (years, "HAR(3)", "Section 3") | L0–L4 | Gate | `tests/check_numbers.py`; generalizes `unflatten.py`'s digit warning | D (data-to-paper; Jr. AI Scientist; Beel) |
| C3 | Generated file stale against its generator or inputs | all | Gate (provenance view prints STALE) | `tests/check_stale.py` + header stamps; full regenerate-and-diff before each push | T-code (showyourwork, Gentzkow–Shapiro, Quarto freeze) |
| C4 | `\rhPending`, `\editc`, `% RECHECK` count is zero; no margin beat IDs | L4 | Gate | Macros error in L4; grep in `check_jfec.py` | D (fixme final mode ([TeX Catalogue: fixme](https://ftp.rediris.es/mirror/ctan/help/Catalogue/entries/fixme.html)); data-to-paper `[unknown]`) |
| C5 | Direction, quantifier and significance words agree with generator predicates; t = coef/se; stars against p | L2–L4 | Gate | `tests/check_claims.py` + `*_predicates.tex` | I (statcheck idea; SPOT/SciTab show LLMs weak) |
| C6 | Each table's span, N and metric match the governing decisions (implemented by C21–C22) | L2–L4 | Gate | Lock stamps (§8.5) | I (CMU undisclosed-change pitfalls) |
| C7 | Every beat has a unique, unchanged ID; manifest matches source | L1–L4 | Gate | `tests/check_beats.py` | I |
| C8 | Beat label is a claim, not a topic; siblings non-overlapping | L1 | Advisory (agent queue), before the layer is served | LLM judge, one round | P (McCloskey, Minto) |
| C9 | L4 text equals the locked L2 sentences plus `\fin{}` text and logged join edits | L4 | Gate | Token diff of the two layers' PDF text | I (Option A) |
| C10 | Option B paragraphs, and any bullet: first sentence restates beat; each bullet realized; no sentence or interpretation beyond beat and bullets; sign and significance kept | L2, L4 (intro, conclusion) | Advisory (agent queue) | LLM/NLI judge; `% src-hash` staleness | T-genre (DOC, STORM, AutoSurvey; reverse outline) |
| C11 | Bib entry verified against Crossref/OpenAlex/S2; writer used only verified keys | all | Gate before push | `tests/check_bib.py`; RefChecker second opinion | D (Rao & Callison-Burch; GPTZero) |
| C12 | Cited passage supports the sentence | L2–L4 | Advisory (agent queue, "not supported" only); a case that cannot be fixed inside the locks becomes a conflict | PaperQA2 over `lit/pdfs/`; `litmap.jsonl` | D-partial (CiteGuard 65.4% vs 69.7%) |
| C13 | Scrub tiers V, S and C (S2–S4a): rates per draft and per section against the corpus p10–p90 band, flagged only beyond sampling error | Locked layers only | Rare markers fixed by S7 automatically and logged; others on the dashboard | `style/scrub.py` | D for marker frequencies and LLM grammar ratios; no evidence on referee ratings |
| C14 | Every `A_*` section referenced from the main text; every main-text reference into the appendix resolves in L1/L2 | L1–L3 | Gate | Extend the existing `.aux` injection check | I |
| C15 | Abstract ≤100 words, 2–6 keywords, JEL present, page budget, disclosure present and naming every model in the log | L4 | Gate | `tests/check_jfec.py` | D (JFEc; OUP policy) |
| C16 | Each flat file under 2 MB; project under 7 MB | push | Gate | `overleaf_git.sh push --check` | D (Overleaf limits; repo incident) |
| C17 | One sentence per line in `rehaul/*.tex` | source | Auto-fix, logged | Splitter on pull | T-code (SemBr) |
| C18 | **Rule NT** (§4.3.7). Every universal, completeness adverb, superlative and booster passes NEEDED? first: a scope contrast, a scope not recoverable from definitions, table or the previous two sentences, or the paper's claim stated once with a number. Otherwise it is deleted even if true. A kept literal universal or superlative must pass TRUE?: it cites a count, argmax or predicate macro that holds | Locked layers only; never author-supplied text (A4) | NEEDED? failures: deleted automatically, logged by reason code. Kept but unbacked: agent queue (add the backing macro or rephrase), or a conflict if that would change a locked claim. In L4, an unbacked literal universal is a gate failure | `style/scrub.py` stage S4b; set registry `style/sets.json`; deletion-equivalence judge; predicates from C5 | P + I for NEEDED? (Grice's Quantity maxim; Cochrane; credibility effect unmeasured); I for TRUE? (C5 mechanism); the need is D (Peters & Chin-Yee; Zhou et al.) |
| C19 | Scope and hedge calibration: sample, period and test qualifiers in the abstract and conclusion match the results section; modal hedges on mechanism and interpretation only, none on computed facts | L0, L2–L4 | Advisory (agent queue) | Qualifier diff across sections; hedge lexicon | D for the failure mode (Peters & Chin-Yee, OR 4.85); I for the check |
| C20 | Generators take every locked parameter from `locks.py`. A lint rejects any literal span, benchmark or design constant, or default results path, outside it. Every generated number carries a stamp of decision IDs and of the values of only those parameters its generator reads, so an unrelated edit to `decisions.yaml` flags nothing | all | Gate | `tests/check_locks.py`; `generated/stamps.json` | D for content-based, field-level triggers (DOORS Link Validity; Jama); T-code (showyourwork/Make dependencies) |
| C21 | Each number used in a section is stamped with the parameters of the decisions governing that section, unless its beat names an exception; forbids honored (e.g. D16) | L1–L4 | Gate | `check_locks.py` | I (repo: the D13 drift, §8.5) |
| C22 | Both sides of every comparison (table columns, a test, a sentence citing two macros) share span, sample, benchmark and design stamps | L1–L4 | Gate | `check_locks.py` | I; D for the hazard (CMU undisclosed changes) |
| C23 | A child decision refines its parent and never contradicts it; a differing value must be a named exception | register | Gate when the entry is written; a failure is a conflict shown to the author | `locks.py` | I |
| C24 | Each locked claim's scope and predicate hold after regeneration. Claims are classed flipped, uncertain (weakened without flipping) or unchanged, with a three-state review record and clearance carried forward when unchanged | L1–L4 | Gate. A flip is reported in chat and listed in `flips.md`, and the push stops until an author edit re-locks the beat. Uncertain claims are listed, not blocking. Unchanged claims stay silent. Agents never clear their own flags | `check_locks.py` + predicates; `claims.yml` | I (§3.3); D for the valid/invalid/uncertain labels (continuous assurance) and for review records (DOORS) |
| C25 | Order and polish (A2): a layer is generated only from locked parents; no scrub, register or stylistic edit on an unlocked layer; editing never changes locked content | all | Gate | `layer_state.json`; `check_locks.py`; S8 | I (A2) |
| C26 | Protected author text (A4). Every `\auth{ID}{…}` span is byte-identical to the text its decision recorded, in every layer and the final build; only surrounding scaffolding may be removed. Digits inside it are exempt from C2 but must agree with the decision they state; a mismatch is a conflict | all | Gate | `check_locks.py` against `decisions.yaml` | I (A4) |
| C27 | Every gate exits non-zero on failure and is wired into the build and `push --check`; no warning-only channel | all | Gate | Exit codes of `tests/*.py` | D for the trap (Sphinx-needs #344: warnings CI never saw) |
| C28 | Minimal-diff push: a push changes only lines whose source changed; no whole-file reflow; flat file names never change | push | Gate | Diff of the new flatten against the last one in `overleaf_git.sh push` | D for the hazard (Overleaf: git-side changes and renames lose comments and tracked changes); I for the gate |
| C29 | Drift gate on pulled Overleaf edits: each is checked against the locks; one that breaks a lock is held; a meaning-changing prose edit is re-derived into a proposed decision change | L1–L4 | Gate (hold); proposal to the author | `check_locks.py` on the pulled diff; LLM re-derivation | T-code (Kiro hooks; Tessl reverse extraction; drift gates in spec tools) |
| C30 | Every pushed review build prints its commit hash in the header; "Table N / Section N" in an edit message is mapped with the `.aux` of that commit, built from a clean export if the working copy differs | push, chat edits | Gate (push refuses a build without its hash) | Header macro filled at flatten; `apply_edit.py` | D for the failure (repo transcripts: shifted numbers, older-commit hints, unpushed working copy); I for the fix |

### 8.4 Prose-scrub pipeline

The scrub is editing, so under A2 it runs **last**. It works only on layers whose content the author has locked, never on the simple version the author is reading (C25). It runs on the bullet text (L2), because under Option A the bullets are the final sentences, and on every join edit.

- **Applied automatically.** Its edits are applied and logged, never served (A1).
- **Locked content is untouchable.** S8's meaning checks and the lock checks (C20–C24) reject any edit that changes it.
- **Author text is protected verbatim (A4).** Text the author supplied in an edit message is marked `\auth{<decision ID>}{…}` in the source. S1 removes it from what S2–S7 may touch, and S8 rejects any edit inside it. The mark travels with the text into lower layers and the final build, where the macro prints its argument unchanged (C26).

Deterministic stages come first. The LLM touches only sentences still flagged, and only register.

| Stage | What it does | Output | Basis |
|---|---|---|---|
| S0 Draft plain | `style/voice.md` guides the drafting of the simple version: plain declarative sentences, numbers as macros, "one name per estimator" from the notation guide, verdict and magnitude words only from predicates, and nothing the later tiers would remove (rare markers, Rule NT failures). It is a drafting rule, not a rewrite, so it applies before lock. Section voice rules join it once the corpus confirms them (§4.3.2) | Fewer flags at edit time; a plain version for the author to read | T-genre (Blattman voice file; Goldsmith-Pinkham style guide); I (A2) |
| S1 Detex | Strip comments, math, `\cite`, `\ref`, macro names (keep a placeholder token), floats, and every protected `\auth{}` span (A4) | Plain text with file:line map back to source | I |
| S2 Tier V (vocabulary) | Rare markers (delve, underscore, showcase, intricate, meticulous, pivotal, realm, tapestry, testament, multifaceted, garner…) as hard flags. Common words (additionally, notably, crucial, comprehensive, insights, enhance, highlighting) as density flags only. Field whitelist: leverage (effect), robust (standard errors), significant, causal, empirical | Fixed by S7, logged with rule ID and source | D (Kobak; Liang; Reinhart; Wikipedia catalog) |
| S3 Tier S (structure) | Em-dash rate; ", highlighting/ensuring…" tails; "not X but Y" and "rather than"; "it is worth noting"; summary openers outside the conclusion; "serves as"/"stands as"; triads per paragraph; bold inline headers in L4 | Rates per 1,000 words vs corpus band; outliers to S7 | D (Reinhart ratios); catalog-level for the rest |
| S4a Tier C (cadence and voice) | Per draft and per section (§4.3.4): passive share, agentless vs by-passive; "we" rate by verb class; main-clause subject types; sentence-length mean, CV, p10/p50/p90, shares over 40 and under 10 words; opening entropy; POS-template repetition; lag-1 length autocorrelation at draft level only. A section is flagged only when outside the corpus p10–p90 band by more than its sampling error. Evaluative verbs with inanimate subjects ("the results underscore") are flagged outright | Dashboard; flagged sentences go to S7 with a named operation | D for LLM-vs-human ratios and narrow length spread (Reinhart; Muñoz-Ortiz; Ju et al.; Shaib et al.); I for thresholds; no academic baseline for alternation |
| S4b Rule NT (stance) | Lexicon hits attached to their scope by parse. Stage A marks REDUNDANT-CANDIDATE when the scope noun is in the set registry and no contrast marker is near. Stage B runs a deletion-equivalence judge on candidates, completeness adverbs and boosters; it must name the excluded exception or the word goes. Survivors go to TRUE? (C5 predicates). Each deletion is logged as REDUNDANT-SCOPE, BOOSTER, SUPERLATIVE-RESTATES-TABLE or UNSUPPORTED-UNIVERSAL | Automatic deletion, logged by reason code; kept but unbacked claims to the agent queue (C18) | P + I (Grice; Cochrane); the need is D (Peters & Chin-Yee; Zhou et al.) |
| S5 Corpus calibration | Fightin' Words z for 1–3-grams vs the JFEc/JoE/JBES corpus (prior from pooled background; term in ≥5 papers). z ≥ +3: over-use warning. z ≤ −3 with document frequency ≥30%: suggestion. Per-paper percentiles for sentence length, passive share, "we" rate, em dashes; regenerates `style/rules/corpus_overuse.yml` | Dashboard; flags to S7 | D for the method (Monroe et al.); untested for draft-vs-journal correction |
| S6 Phrase frames | For standard statements (DM, MCS, QLIKE, Mincer–Zarnowitz), retrieve nearest corpus sentences and abstract them into templates with bibkey and page | `style/frames/*.md` | I (no published tool found) |
| S7 LLM register pass | Per flagged sentence, with its paragraph, rule IDs and frames: one **named operation** (convert to agentless passive; replace an inanimate agent with "we" or a passive; merge; split at the conjunction; delete the participial tail; delete the quantifier; replace an evaluative verb with the number; cut an importance-assertion). Never a global target such as "use 25% passives" or "vary sentence length". Never touch macros, cites, refs or comparison words. Stop when the section is inside the band, not at the corpus median | Applied automatically when S8 passes; logged with rule ID | D against global targets (Reinhart exemplar result; Zhou et al. overshoot); T-genre (LAMP; Patchwork rule citation); I for the operation list |
| S8 Post-checks | Content-preservation diff: same multiset of macros, cites and refs; negations, modals and direction words unchanged unless the operation targeted them; no invented agent (a passive-to-active rewrite must not make "we" do what a data vendor or cited paper did). Rerun S2–S5, since rewrites can add markers. Longest shared word n-gram with the corpus (hashed 8-grams); flag rare shared spans | Rewrite rejected automatically on any failure | T-genre (Lee et al. copying); I |

The scrub never claims to evade detectors and runs alongside the disclosure (§4.5, §7.1).

### 8.5 Decision locks

*This replaces the review-card loop (A1; the card design is in commit 3c87fa6) and implements A2's order.* The basis is **I** throughout unless a part says otherwise. The nearest precedents propagate code changes, not judgments:

- data-to-paper's labelled-value chain ties each number to the code that produced it ([data-to-paper](https://arxiv.org/pdf/2404.17605));
- showyourwork and Make rebuild an output when its inputs change ([showyourwork docs](https://show-your.work/en/latest/latex/));
- Gentzkow and Shapiro rebuild everything from one script ([Gentzkow & Shapiro](https://web.stanford.edu/~gentzkow/research/CodeAndData.pdf)).

Requirements and assurance tools propagate upstream changes as flags for a human to clear (§1.5). No system found propagates an author's decision through a paper, or limits review to claims whose outcome flipped. The principle is that the author's judgment enters once, at the highest layer it concerns, as an edit to plain text. Everything beneath it is computed from it, worded from it and checked against it.

**The cycle (A2, A3).** Two input channels feed one register: the author's chat edits, and Overleaf edits (A5, below).

1. **Serve the simple version.** An agent generates the next layer from the locked layers above it, in plain sentences, with no polish (S0, C25). It prints short beat IDs in the margin and pushes the layer to Overleaf.
2. **Read and send edits (A3).** The author reads the PDF on Overleaf once and sends edits as chat messages, phrased against the PDF: quotes, section or table numbers, margin beat IDs, "get rid of this". The author never writes YAML.
3. **Locate, apply, record.** For each edit the agent:
   1. locates it in the source;
   2. applies it at the layer it targets;
   3. records it in `decisions.yaml`, with the author's message words verbatim and dated as the decision text, plus the parameters or predicate it implies.

   One message may yield several entries. Text the author supplied is wrapped `\auth{ID}{…}` and protected verbatim (A4, C26). The agent asks only when an edit is ambiguous or conflicts with a lock.
4. **Lock, propagate, push.** `locks.py` validates each entry (C23), regenerates everything it governs, and re-evaluates stamps and predicates. The agent commits once per message, and the push updates the PDF the author is reading.
5. **Advance.** A layer with no open edits is locked and never served again. The next layer is generated beneath it.
6. **Edit last.** After L0–L3 are locked, L4's editing runs: scaffold removal, typography and the scrub (§8.4). It may not change locked content or author text (C20–C26, S8).

If an edit implies no machine parameter (a pure wording or framing change), it locks as words only.

**Seed of the register (repo).** The repo already follows the convention of quoting the author with a date:

- 920 dated `% <date> (user, …)` or `(author, …)` comments across `paper_rehaul.tex` and `rehaul/*.tex`;
- commit messages such as "(author, 28 Oct: "they're cheap, and they make the results apples to apples. run it.")".

**Locating and applying a chat edit (A3).** This project has evidence here: a transcript analysis of 64 past author edit requests, 53 to the LaTeX paper and 11 to the earlier GUI and notebook outputs (repo, `edit_location_method.md`).

- **How requests were phrased.** Exact-phrase grep was rare: 3 of the 53 paper requests. Most requests named a table, a section or an idea, not a string. They were located by:
  - `.aux` label lookup, which maps "Table N" or "Section N" to a label (18);
  - section and bullet titles (17);
  - tracing macro → generated file → generator (21);
  - searching for related text (17);
  - delegation with locating instructions (27).
- **Where the edits landed.** 21 of the 53 landed in a `make_*.py` generator or a macro (captions, table notes, headers, word-valued macros), not in section text.
- **Failures.** The observed failures were:
  - missed occurrences when an idea-level decision was applied by keyword: the request to corral every closing-interval model was applied to "closing-interval" models only, and pooled 16:00-bar results stayed in Sections 1, 6 and 9;
  - review layers left stale because a merge sat uncommitted;
  - a design decision (D13) not propagated to generator inputs;
  - a wrong placement: a citation went into the wrong bullet, caught by the author three minutes later;
  - shell-quoting breakage: Git Bash heredocs dropped backslashes.
- **Where PDF numbering broke.** Numbering held because `push --check` guarantees that the Overleaf build and the source build have identical pages and text. It broke only when the author's PDF and the local `.aux` differed: a shifted section number ("section 6" was Section 5), line hints from an older commit, and a working copy ahead of what was pushed.

The chat-channel procedure adopts that note's seven steps (D, from this repo's transcripts):

1. **Classify the request.** Find the anchor: quoted wording; a numbered object (Table, Section, Figure N); a named object (a bullet or section title, note ID, macro, citation or generator); or a paper-wide idea. Note scope words such as "everywhere", "in the tables" or "again".
2. **Resolve numbers against what the author sees.** Use the `.aux` of the commit the author's PDF came from, then render the page to confirm the object. If the working copy is ahead, say so and search by title or content instead.
3. **Find the source location.** Grep a fixed, case-insensitive string over live lines of the sources, generated files and generators. Use the longest one-line, number-free ASCII fragment when the PDF broke the quote. Trace generated text to its `make_*.py`, and take a bullet or section as the region between its title or label and the next one.
4. **Build the propagation inventory before editing.** List wording variants, `\ref`s to affected labels, uses of affected macros, the same citation key, and the claim's counterparts in the abstract, introduction and conclusion. Add derived outputs and, for a decision, every generator input and governed sentence, searched by idea, not keyword. Siblings whose scope is unclear go back to the author as one question.
5. **Apply.** Edit sources only and regenerate. Park replaced text (in the sidecar of §2.4 once Phase 1 lands), keep labels, and route numbers only through macros. Split large inventories by file ownership with one shared rule. Use the Edit tool or Python scripts with asserts, never shell heredocs.
6. **Verify coverage.** Re-grep with before and after counts, `git diff` the generated files, and build from a clean export of `HEAD`. Run `pdftotext` on the affected pages, regenerate the review layers from the same commit, then `push --check`.
7. **Report and close the loop.** Tell the author every spot changed and what was kept. Re-run a decision's inventory whenever a later rewrite touches its area.

Two additions:

- **Commit hash in every pushed review build (I).** Each pushed build prints its commit hash in the header, so "Table N" or "Section N" is mapped with the `.aux` of exactly that commit. This closes the shifted-number, older-commit and unpushed-working-copy failures (C30).
- **The locks automate step 4.** The inventory step is what the locks make mechanical for decisions: decision → generators (`locks.py` parameters) → macros (stamps) → sentences (lock macros and the ripple graph). That is the failure class the transcripts show: missed 16:00-bar occurrences, and D13 not reaching three generators.

Margin beat IDs remain optional (I). The review builds may print each beat's short stable ID in the margin, such as B6.2.3, so the author can write "B6.2.3: …" when convenient. Each ID is minted once and never renumbered, and the final build never prints them (C4). Quotes, numbers and titles still work.

**Second channel: Overleaf edits (A5).** Collaborators, and on occasion the author, edit on Overleaf, and those edits must flow back into the sources:

- **Main file.** `overleaf_git.sh pull` already runs `unflatten.py` on `main2.tex`. It applies prose edits to the section files, parks replaced lines, and reports number and table edits as conflicts (repo, [overleaf_sync README](../overleaf_sync/README.md)).
- **Layers (repo change needed).** Today `push-beats` "refuses if Overleaf's copy differs from its last flatten (the layers are for reading and comments; edits go to the sources)" (repo, Overleaf helper script). Under A5 the pull must instead unflatten edits on `beats_top.tex` and `beats.tex` back to their source lines through `map_beats_top.json` and `map_beats.json`, so `push-beats` pulls layer edits back instead of refusing them. This is feasible because `make_paper_beats.py` copies every line it keeps verbatim from its source ([make_paper_beats.py](../make_paper_beats.py)). `layer_header_fix.py` keeps auto-reverting stray edits inside a layer's generated `%%` header (repo).
- **Generated regions.** An edit to a table body, a float stub or a macro's printed value cannot flow back to text. It becomes a request to change the generator or a decision, reported to the author.
- **Authority (proposed default; §8.9 Q10–Q11).** A collaborator's Overleaf edit is applied as text and recorded with its source: Overleaf, the date, and the editor. It does not create a lock unless the author adopts it. An Overleaf edit that would break an existing lock is held, not applied, and shown to the author as a conflict.
- **Drift gate (C29).** Every pulled Overleaf edit is checked against the locks. A collaborator's prose edit that changes meaning is re-derived into a proposed decision change for the author, after Tessl's reverse extraction of a spec from code (T-code, §1.5).
- **Editor attribution.** Overleaf's git-integration page says "the author of the most recent change will be listed as the author of the commit", and a commit is created when someone pulls (as quoted in the 2026-10-29 deliberation, item X6; [Overleaf: Git integration](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/git-integration)). The editor field is therefore the bridge commit's author: the last editor of each pull batch, exact when one person edited between pulls. For GitHub sync, Overleaf instead bundles Overleaf-side edits under the connecting account ([Overleaf: GitHub sync](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/github)). The five Overleaf-side commits in the local clone carry the author `jamesdchen01` (repo). That fits the rule, though one account cannot discriminate. Whether two editors between pulls are split into separate commits is unverified; one test edit settles it.

Overleaf comments remain outside both channels.

**Collaboration hazards.** Overleaf warns that "pushes from Git to Overleaf can result in the loss or displacement of track changes and comments" (§2.4), and "renames destroy comments and tracked changes" (as summarized by a published Overleaf skill; [ChernXiangyu](https://github.com/ChernXiangyu/overleaf-git-skill)). Regenerating `main2.tex` on every push therefore threatens the comments collaborators will now leave. Mitigations (I unless noted):

- **Pull before every push.** The push already refuses while `main2.tex` has unpulled edits (repo); `push-beats` should refuse the same way once it pulls.
- **Minimal-diff pushes, as a gate (C28).** A push may change only the lines whose source changed, with no whole-file reflow, and flat file names never change. Do the one-sentence-per-line reflow (C17) once, announced, before collaborators start; one sentence per line then keeps comment anchors stable.
- **Tracked changes.** Accept or reject them on Overleaf before a pull, or detect and report them (§8.9 Q12). Whether tracked changes or comments are visible to the git bridge at all is unverified; the docs document only the loss risk.
- **Alternative agent route (§8.9 Q13).** claudeleaf types agent edits into the live editor from a dedicated bot account and can submit them as tracked-change suggestions. Agent edits then become attributable and leave comments in place, but the route needs premium track changes, cannot resolve comments, and edits the live document ([claudeleaf](https://github.com/lonetis/claudeleaf)).

**Loop speed (A3).** What matters is the time from the author's message to the updated PDF on Overleaf. The existing watcher, `writeup/overleaf_sync/autopush.sh`, closes the loop after each commit that changes `writeup/` (repo):

1. It polls every 30 s and lets a burst of commits settle for 20 s.
2. It regenerates the review layers from a clean checkout: three draft-mode pdflatex passes plus BibTeX, then `make_paper_beats.py`.
3. It runs `overleaf_git.sh push-all --check`, which builds the flat files and the source and compares their PDF texts.
4. It pushes to GitHub.

One push runs at a time, and a refused or failed step waits for the next commit. The floor is therefore about 50 s of polling and settling, plus several full LaTeX builds per commit, plus Overleaf's own recompile; no timings are logged. Inference:

- record message-received, committed and pushed times in each decision entry;
- commit once per message, so a message costs one build cycle;
- make `autopush.sh` pull before it pushes, so a collaborator's unpulled edit does not stall the author's loop (A5).

**The decision record.** Entries live in `writeup/decisions.yaml`, seeded from the D1–D20 table in [INTEGRATION_PLAN_2026-10-26.md](../INTEGRATION_PLAN_2026-10-26.md) and the dated author quotes (repo).

| Field | Content | Example from this repo |
|---|---|---|
| `id` | Stable, never reused | D13 |
| `layer` | story, design, presentation or beat | D1 story; D13 design; D16 presentation |
| `parent` | The decision it refines | a design decision under D1 |
| `words`, `date` | The author's message words verbatim (or the Overleaf edit's text), and the date | D16: "why is the 1600 bar still in the main paper??" (26 Oct) |
| `source`, `times` | `chat` (the author) or `overleaf` (with the editor if the bridge records it); received, committed and pushed times | an Overleaf edit locks only once the author adopts it |
| `binds` | Machine parameters | `eval.span=rth13`, `eval.oos=[start,end]`, `benchmark=ols_har_rth13`, `design=fomcfix`, `sharpe=pooled_d19` |
| `forbids` | What may not appear | D16: no 16:00-bar accuracy results in the main text |
| `exceptions` | Named exceptions, each itself a decision | D3's closing-interval section; the feature-importance sample from 2020-01-03, today a local constant (`FI_OOS_START` in [rehaul_fi_compute.py](../../experiments/rehaul_fi_compute.py)) |

```yaml
# writeup/decisions.yaml (sketch). Agents write entries from the author's edit messages and from
# pulled Overleaf edits; the author never edits this file. Seeded words are the plan's records of D1-D20; parents are illustrative.
- id: D1
  layer: story
  words: "Headline target = every regular-hours bar (intraday RV), not the closing half hour"
  date: 2026-10-26
  binds: {eval.span: rth13, eval.oos: [<start>, <end>]}
  exceptions: [D3, X-FI]
- id: D3                         # an exception: its own scope, its own span
  layer: design
  parent: D1
  words: "All closing-interval models ... are corralled into the final section ... and its appendix"
  date: 2026-10-26
  scope: {sections: [<final section>, <its appendix>]}
  binds: {eval.span: close30}
- id: X-FI                       # an exception, today the constant FI_OOS_START
  layer: design
  parent: D1
  words: "<the author's words of 2026-10-18, new-notes 22>"
  date: 2026-10-18
  scope: {sections: [sec:rh_fi]}
  binds: {fi.oos_start: 2020-01-03}
- id: D13
  layer: design
  parent: D4
  words: "FOMC fixes ADOPTED ... the paper's numbers come from the patched design"
  date: 2026-10-26
  binds: {design: fomcfix}
- id: D16
  layer: presentation
  parent: D1
  words: "why is the 1600 bar still in the main paper??"
  date: 2026-10-26
  forbids: [{result: accuracy, bar: "16:00", where: main_text}]
```

**Refinement rule.** A child decision may refine its parent: narrow a range, add a forbid, or fix a parameter the parent left open. It may never contradict its parent. A child that binds a different value must be listed among the parent's named exceptions. Otherwise C23 fails the moment the entry is written, and the clash is shown to the author as a conflict.

**Three propagation channels.**

- **(a) Compute.** Generators read every locked parameter from `writeup/locks.py`, never from local constants or default results paths (C20). They are rerun when a parameter they read changes, Make-style, and all of them are rerun before a push (Gentzkow–Shapiro). Each generated number is stamped in `generated/stamps.json` with the decision IDs and parameter values it was computed under. This extends Phase 0's header stamps.
- **(b) Words.** Scope wording prints from lock macros that `locks.py` writes into `generated/locks.tex`. For example, `\lockSpan` might print "every regular-hours bar, 2014–2024", so every sentence that states the scope changes with the decision. The macros are generated, so they pass C2. Rule NT's set registry (`style/sets.json`) is built from the same entries.
- **(c) Check.** The build fails if a number in a section is stamped with parameters that differ from the decisions governing that section, unless the beat holding it names an exception (C21). Forbids are enforced the same way: a 16:00-bar accuracy macro in the main text fails D16. Both sides of every comparison must share the lock: span, sample, benchmark and design (C22).

**Locked claims.** A beat the author has read, with the author's edits applied, becomes a locked claim, stored in `claims/claims.yml`. It has two parts:

- its scope, the decision IDs it rests on;
- a predicate over macros: sign, rank, significance, or count equal to total.

The abstract's "all … each significant" is an existing example: it is backed by `\rhorAfLower = \rhorAfLowerSig = \rhorAfPairs` (repo, §3.3). Bullets and final text beneath a locked claim are maintained by AI and must satisfy it; they are checked, not reviewed (C24).

Interpretation lives only in beats. A bullet that adds an interpretation beyond its beat (C10) is rewritten as evidence or logged as a proposal. It never ships unread.

**Stamps and review records (from traceability tools, §1.5).**

- **Stamps hash only what a generator reads.** A stamp covers only the decision parameters that generator actually reads, never the whole `decisions.yaml` file, so a comment edit or a reordering flags nothing. This is DOORS's content comparison and Jama's field-level triggers (D in requirements tools; T-code for the paper).
- **Three-state review record.** Each locked claim carries a record, suspect, valid or invalid, with who set it, when, and a comment, as in DOORS Link Validity (D; T-code).
- **Clearance carries forward** when a regenerated result is unchanged in the predicate's terms, as DOORS reuses validity where content is identical (D; T-code).
- **Agents never clear their own flags.** Code may clear only the class "number changed; sign, rank and significance class unchanged". Flipped and uncertain claims clear only through an author edit (I; motivated by the bulk "Clear All" pattern).

**Ripple graph and "where used".**

- **Ripple graph.** `locks.py` keeps an explicit graph from each number to the sentence, the paragraph, and the abstract, introduction and conclusion sentences built from them, using macro uses and beat IDs. A flipped table entry then reaches the abstract sentence that summarizes it. ExplicitCase names ripple effects as the hardest part of impact analysis (D in assurance research; I for the paper).
- **Where-used report.** Before a decision changes, a report lists every number, sentence, table and layer that cites it, as Paligo's lookup does (D in structured authoring; T-genre).

**Register fields (from ADRs, Kiro and OpenSpec).**

- **ADR fields.** `decisions.yaml` gains `status` (proposed, accepted, superseded), `supersedes` and a one-line `rationale` (T-code).
- **EARS-form rules.** Each lock is also written as a checkable rule in Kiro's EARS form, "WHEN <context>, THE TEXT SHALL <claim, number or wording>". For example: "WHEN Section 6 reports accuracy, THE TEXT SHALL score every regular-hours bar over `\lockSpan` against the OLS-HAR benchmark" (T-code; Kiro documents the form, not its effect).
- **Dated deltas.** Each chat edit is recorded as an OpenSpec-style dated delta, ADDED, MODIFIED or REMOVED, against the locked layer, and archived by date (T-code).
- **Always-applied conventions.** Paper-wide conventions (notation, sample period, headline numbers, forbidden claims) are applied to every agent step, never retrieved by mention. Fiction tools drop facts that are not mentioned, and Kiro's steering defaults to `always` (D for the failure; T-genre).

**Changing a decision.** The author sends one edit, against whatever place in the PDF they are reading. The agent applies and locks it, regenerates everything, and pushes; stamps and predicates re-evaluate. The author then sees only these, reported in chat and listed in `flips.md`:

- **flipped:** claims whose direction, rank or significance flipped;
- **uncertain:** claims that did not flip but weakened, such as p rising from 0.01 to 0.09, an effect halving, or an interval coming to span a threshold the text names. This tier follows the valid/invalid/uncertain labels of continuous-assurance research (D there; I for the thresholds);
- conflicts between decisions, between a new edit and an earlier lock, or held Overleaf edits;
- requests to change a generator that came from edits to generated regions.

Everything else is silent. Numbers update through macros, and the prose of unflipped claims is not regenerated. Flipped and uncertain claims are themselves routed through the question tiers (§8.10). They reach the author as T3 only if deliberation cannot place them: for example, a flip that a locked claim's own predicate already settles, such as a rewording dictated by a generated comparison word, is T0.

**The author's whole review surface** has three parts:

- the PDF of each new layer's simple version, read once;
- the decision register, one page printed at the front of L1 and changed by an edit message like any text;
- `flips.md`.

There are no diffs and no per-edit decisions.

**The drift that locks would have caught (repo).** D13 adopted the FOMC-patched design on 26 October. On 28 October the ladder and SHAP generators (`make_rehaul_ladder.py`, `make_rehaul_fi_pooled.py`, `make_rehaul_fi_live.py`) still read pre-patch results by default. Commits 7ca4268 and 8fb75e7 switched them by hand that afternoon, keeping `--prepatch` to reproduce the old outputs (git log). Under locks, `design: fomcfix` would have reached every generator through `locks.py` the moment D13 was recorded. Any number still stamped pre-patch in a section governed by D13 would have failed C21.

**What A1–A5 give up.** The per-edit evidence of §5.1, where InkSync's warning step caught twice as many planted bad edits, no longer applies to text below a lock. Under the A5 default, collaborators' edits that do not conflict with a lock are applied without the author reading them. The checks cover numbers, scope, comparisons, stated predicates, citations and S8's meaning diff. They do not cover a wording change that alters meaning without touching a macro, negation, modal or comparison word. Three things narrow that gap:

- Option A, under which the final text is the locked bullets (C9);
- confining interpretation to beats;
- the milestone pool of AI verifiers (§5.4).

The residual risk is unmeasured.

**Avoid (from §1.5).**

- **Long upper layers.** In the one quantified test, Spec Kit cost about 5.5 hours of review and 2,577 lines of Markdown for about 689 lines of code. Beats must stay much shorter than the prose they govern (D in coding; T-code).
- **Regenerating whole sections for one change**, as in Sudowrite's chapters and Tessl's non-deterministic regeneration. Regenerate only the beats whose governing decision changed, and keep approved and collaborator prose frozen (D for the failure; I).
- **Forcing typo fixes through the top layer.** Copy-edits, LaTeX fixes and citation keys have no beat-level meaning; Eberhardt's bug had no spec expression at all. They are applied at the layer they touch and logged as deltas, but create no lock above it (D in coding; I).
- **Assuming a decision in context is obeyed.** Agents both ignored and over-applied specs (Böckeler), so every lock gets a mechanical check wherever one is possible (D in coding).

**Starting point for this paper.**

1. Seed `decisions.yaml` from D1–D20 and from the dated author quotes in the sources.
2. Seed the current top beats as locked claims.
3. Do the one-sentence-per-line reflow once, announced, before collaborators start (A5).
4. Serve the current top beats, with the seeded register in front and beat IDs in the margin, as the simple starting state.

Whether the existing bullet text is reset or kept as the first L2 draft is open (§8.9).

### 8.6 Agent working rules

These rules govern every drafting and review agent. Each is a gate in practice, because the checks in §8.3 reject output that breaks them.

| Rule | Basis |
|---|---|
| Each step receives a declared input manifest and nothing else. Beat-to-bullet drafting gets L0, the section's locked beats, this beat and its neighbors, the decisions governing it (via `locks.py`), the macro table with descriptions, verified bib keys, predicate words and `voice.md`. | D (data-to-paper whitelisted inputs); T-genre (DOC, Dramatron context) |
| Generators take every locked parameter from `locks.py`; no local constant or default results path for a locked parameter. | I; T-code (showyourwork/Make dependencies) |
| A layer is generated only from a locked parent, and drafted plain. No scrub, register pass or stylistic rewrite runs before lock. | I (A2) |
| A decision's words are the author's message words, verbatim and dated. Agents infer its parameters and never edit its words. | I (A2, A3) |
| One edit message may yield several locks, each recorded separately with the same words and date. Commit once per message and push, so the PDF updates. | I (A3) |
| No question goes to the author without a saved deliberation under `writeup/deliberation/<date>/` that placed it in T3. T0–T2 outcomes are acted on and recorded, not asked (§8.10). | T-code and assurance-case transfer; D for pooled independent reviewers plus a verifier (§5.4); I (A6) |
| Locate chat edits by the seven-step procedure (§8.5): resolve numbers with the `.aux` of the commit printed on the PDF, build the propagation inventory before editing, search decisions by idea not keyword, ask one question for unclear siblings, never edit through shell heredocs. | D (repo transcripts, 64 past requests) |
| Ask only when an edit is ambiguous or conflicts with a lock. Otherwise apply, lock, push and report in one short message. | I (A3) |
| Text the author supplied is copied verbatim inside `\auth{ID}{…}`. No agent pass edits it, and the mark travels with it into lower layers and the final build. | I (A4) |
| Pull before every push. Apply collaborators' Overleaf edits as text, with their source recorded. They lock only when the author adopts them; one that would break a lock is held and reported. Never reflow whole files collaborators edit. | I (A5; default open in §8.9) |
| A reviewer's request for a new number, test or citation is logged as a proposal, never given to a writer. Nothing acts on it until an author edit adopts it. | D (Jr. AI Scientist: reviewer feedback induced fabrication) |
| Agents never clear their own review flags. Code clears only results whose sign, rank and significance class are unchanged; flipped and uncertain claims clear only through an author edit. | I; D for the bulk-clear pattern in requirements tools (§1.5) |
| Paper-wide conventions are in context for every step; binding facts are never retrieved by mention. | D for the failure (Sudowrite, Novelcrafter); T-genre |
| Regenerate only the beats whose governing decision changed; the prose of unflipped claims stays as it was. | I; D for whole-regeneration failures (Sudowrite, Tessl) |
| Reuse whole claims and numbers across layers, never sentence fragments. | D in DITA localization practice; T-genre |
| Never write tool-generated UUIDs into source; beat IDs are short and human-readable. | D for the merge hazard (Logseq); T-genre |
| AI reviewers run at milestones with the §5.4 checklists, merge and refute, and feed the agent queue. | D in other fields (§5.4) |
| Sections may be drafted in parallel; beats within a section sequentially. | T-genre (AgentWrite: parallel cost about 6% coherence) |
| Never type a digit. Derived numbers are computed in Python and emitted as macros with a formula comment. | D (data-to-paper `\num`) |
| Verdict words come only from predicates derived from the pre-registered `RULE_*.txt`. | D (Alizadeh et al.: prompt flipped verdicts 10%→90%) |
| No global style instruction ("use more passives", "vary sentence length", "be accurate about scope"). Style edits are named sentence operations, re-measured after each pass. | D (Reinhart exemplar result; Zhou et al. overshoot; Peters & Chin-Yee: accuracy prompts did not stop overgeneralization) |
| A universal or superlative goes into a draft only if it passes Rule NT (§4.3.7). | P + I; the need is D |
| At most one unanchored critique round per unit; oracle-backed loops stop after two. | D-genre (Huang et al.; AutoSurvey) |
| Critics run in fresh contexts, and in another model family at milestones. | T-genre (Cunningham, Blattman; Litvak pooling) |
| Writers cite only keys in `bib_verified.json` and never edit the .bib. A separate agent proposes additions by DOI fetch. | D (Jr. AI Scientist; clibib) |
| Agents keep sessions focused and hand off through files, not long chats. | Practitioner advice (Golub; Goldsmith-Pinkham) |

### 8.7 Phased build order

Each phase is useful alone, and the order follows value per effort plus dependencies. Locks and correction capture come before the scrub (A2). Citations also come before the scrub, because the live text already measures clean on vocabulary markers (§4.1) while about half of the bibliography lacks DOIs (§6.4).

| Phase | Deliverable | Done when | Depends on | Basis |
|---|---|---|---|---|
| **0. Gate the numbers (first step)** | **P0.1 `tests/check_numbers.py`**: standard library, run in the build and in `push --check`, exiting non-zero on failure (C27). P0.2 `\rhPending`/`\editc` error in a final wrapper. P0.3 header stamps + `check_stale.py`; full regenerate-and-diff before push | Zero unwhitelisted digits; every whitelist entry has a reason; a deliberately stale file fails | Nothing | D / T-code |
| 1. Layer selector and beat IDs | `\paperlayer` with hard error; `paper_final.tex`; beat macros and migration script; `make_paper_beats.py` learns them and writes `beats/manifest.json`; `check_beats.py`; parked text moved to `rehaul/parked/` (and `unflatten.py` parks there); SemBr reflow as one bulk commit, announced, before collaborators start (A5); minimal-diff push gate and fixed flat file names (C28) | Bullet-layer PDF text identical before and after migration; L4 compiles as paragraphs; flat file under 2 MB | Phase 0 | I; T-code |
| 2. Decision locks and correction capture (both channels) | Migrate D1–D20 and the dated author quotes into `decisions.yaml`, exceptions and forbids included (D3, D16, the feature-importance sample). Build `locks.py` and move generators onto it. Add stamping (`stamps.json`) and lock macros. Have generators emit the predicates the locked claims need (moved forward from Phase 5), and build `claims.yml`. Add `check_locks.py` (C20–C26). Build correction capture for both channels: `apply_edit.py` for chat edits (A3), with margin beat IDs and the `\auth` protection (A4); layer-file unflatten through `map_beats*.json`, so `push-beats` pulls collaborators' edits back instead of refusing them (A5); pull-before-push in `autopush.sh`; message-to-PDF times in the register. Make stamps hash only the parameters each generator reads. Add three-state review records with carried-forward clearance; the flipped and uncertain tiers; the ripple graph and `where_used.py`; ADR fields, EARS rules and dated deltas in the register; the drift gate on pulled Overleaf edits (C29); the commit hash in every pushed review build (C30); and `apply_edit.py` implementing the seven-step chat procedure. The question-tier router already exists as `digest.py` with saved deliberations (§8.10, commit 930958b); the T0 router stays manual until `locks.py` exists. Add `flips.md` and the register printed in front of L1. Then serve the current top beats, seeded from D1–D20, as the simple starting state | A changed span propagates to every table and sentence with only flipped and uncertain claims shown, and a reordering of `decisions.yaml` flags nothing. One edit message about L1 becomes a decision entry, reaches every governed number without further review, and lands as an updated PDF on Overleaf with its time logged. One collaborator edit on `beats_top.tex` flows back to its source line, and one that breaks a lock is held and reported | Phases 0–1 | I; T-code (showyourwork/Make; Gentzkow–Shapiro); D (data-to-paper chain) |
| 3. Citations | One canonical bib; deterministic fetch skill; `check_bib.py` with cache; then PaperQA2 over local PDFs → `litmap.jsonl` → "not supported" findings to the agent queue | All 148 entries verified or explained; writer restricted to verified keys | Phase 0 | D; D-partial |
| 4. Prose scrub | Built while the author reads; **runs only on locked layers** (A2). S0 drafting guide; S1–S3 lint. S4b Rule NT: Stage A and the set registry (built from `decisions.yaml`), with TRUE? on Phase 2's predicates. Corpus build, S4a cadence and voice bands, and S5–S6 (stats only committed). S7–S8 register pass with post-checks; edits applied automatically and logged | Corpus dashboard shows the locked draft inside the 10th–90th percentile band, or each exception is recorded as a decision; every universal and superlative is either deleted with a reason code or backed by a macro | Phase 2 | D for statistics; T-genre for rewriting; P + I for Rule NT |
| 5. Direction words and provenance | Generators emit the remaining predicates and spec macros; `check_claims.py` over every sentence; provenance wrapper | Every sentence with a numeric macro and a direction word is checked | Phase 2 | I; D (data-to-paper chain) |
| 6. Submission gates and outside review | `check_jfec.py`; disclosure generator from trailers; replication README table from generator headers; pooled reviewers + verifier feeding the agent queue; Refine at a milestone; a one-month cooling-off before submission | Final layer passes all gates; disclosure names every model in the log; one outside review round triaged | Phases 0–5 | D (OUP, JFEc); D-other-fields (AI review); practitioner (Gans) |

**The smallest useful first step is P0.1.** One standard-library script strips comments, math and the whitelist from the live text of `paper_rehaul.tex` and `rehaul/*.tex`, and fails on any remaining digit. It is wired into the existing build check and `overleaf_git.sh push --check`. It needs no new layer, closes the one failure every agent-writing evaluation reports, and makes the existing `unflatten.py` digit warning a source-wide invariant.

The phases set the order in which the **tooling** is built. The order in which the paper's **content** moves is A2's: L0–L1 read and edited first, then L2, then L3, editing last. The first author-facing moment is the end of Phase 2, when the current top beats are served as the simple starting state, seeded from D1–D20 and printed with the register.

### 8.8 What rests on evidence and what does not

| Design element | Basis | What is missing |
|---|---|---|
| Numbers only through code, typed-digit lint, abort on placeholders | D (data-to-paper; Jr. AI Scientist; Beel; MLR-Bench) | No finance-paper measurement; the mechanism is domain-independent |
| Staleness stamps, full regeneration | T-code (showyourwork; Gentzkow–Shapiro) | No measured error rate |
| Beats as the first and highest-value human read | T-genre (Dramatron, DOC, STORM, Re3) | No study of empirical economics papers |
| Final layer by scaffold removal (Option A) | I, motivated by STORM/AutoSurvey unsupported-claim rates and Liang et al.'s expansion generator | Untested; reader response to bullet-derived paragraphs unknown |
| Beat IDs, hashes, incremental regeneration | I. Partial precedents: Kiro tasks cite requirement IDs; DITA keys | No system implements it for papers |
| Content-based stamps, three-state review records, carried-forward clearance | D in requirements tools (DOORS Link Validity; Jama field triggers); T-code | No study of flag fatigue or clearing effort |
| Flipped and uncertain tiers, everything else silent | D for valid/invalid/uncertain labels in assurance research; I for the paper's thresholds | No tool filters review on outcome rather than input; thresholds untested |
| EARS-form locks, dated deltas, ADR fields, always-applied conventions | T-code (Kiro, OpenSpec, ADRs); D for the mention-retrieval failure (fiction tools) | No evidence they cut review; Spec Kit's long upper layers raised it |
| DITAVAL-style selector (include/exclude/flag), resolve before filter, whole-claim reuse | D in DITA; T-genre | Untested on papers |
| Direction-word predicates | I (statcheck idea) | No tool exists; SPOT and SciTab show LLM checks are weak |
| Decision locks: register, parameters from `locks.py`, stamps, comparison locks (A1) | D for the parts in other fields (requirements suspect links, ADR fitness functions, DITA keys, legal answer files; §1.5); T-code (showyourwork/Make; Gentzkow–Shapiro); I for combining them for a paper | No system combines them for a paper; the D13 drift is one repo incident, not a measured rate |
| Top-down order: simple layer read once and corrected, next layer generated from the locked one, polish last (A2) | I; T-code (Google: direction first); T-genre (Dramatron: beat layer least edited; DOC: outline errors cascade) | No study of reading each layer once; no measured time saving for documents |
| Edits applied and recorded from two channels (author chat, A3; Overleaf, A5); author edits lock, collaborator edits lock only on adoption; text below a lock checked, not reviewed | I | What the author misses by not reviewing below a lock is unmeasured; per-edit review evidence (InkSync) no longer applies there (§8.5) |
| Locating chat edits: the seven-step procedure, numbers mapped through the `.aux` of the commit printed on the PDF, the inventory automated by locks | D (repo transcripts: 64 past requests; exact-phrase grep in 3 of 53 paper requests; `.aux` labels 18, titles 17, macro → generator 21, related text 17, delegation 27). I for the commit-hash header and for optional margin IDs | One project and one author; message-to-PDF time not yet logged; whether locks remove the missed-occurrence failures is untested |
| Author text protected verbatim (A4) | I | A protected sentence keeps any redundant universal or slip the scrub would have fixed, by the author's choice |
| Overleaf round trip with collaborators (A5) | D for the hazards (Overleaf docs: git-side changes and renames lose comments and tracked changes; GitHub sync bundles Overleaf edits under the connecting account); D for one-owner-per-file sync (showyourwork; ChernXiangyu); I for the two-way design and its gates | No published tool round-trips prose both ways; no study of comment loss in mixed human/AI projects; whether tracked changes reach the git bridge is unverified |
| Pooled AI reviewers with a refuting verifier | D in ML, astrophysics, psychology and math (Bianchi; Zubeldia; Litvak; AAAI-26) | No financial-econometrics benchmark; persona prompts untested |
| Deterministic BibTeX and existence checks | D (Rao & Callison-Burch; GPTZero) | SSRN DOI resolution through Crossref unverified |
| Claim-support checks | D-partial (CiteGuard; scite evaluation) | Below human level; advisory only |
| Vocabulary and structure scrub | D for marker frequencies (Kobak; Liang; Reinhart; Feyzollahi) | No evidence removal changes referee ratings; experts still detect humanized text |
| Corpus calibration and register rewriting | D for the statistic (Monroe et al.); T-genre for rewriting (Reddit authorship transfer; LAMP) | No academic-register study; copy risk documented on GPT-2 only |
| Cadence and voice metrics in per-section bands | D for LLM-vs-human grammar and length spread (Reinhart; Muñoz-Ortiz; Ju et al.; Shaib et al.) | No finance or econometrics voice baseline; section-level passive share is noisy (±11 pp at 60 sentences); alternation has no published baseline |
| Named sentence operations instead of global style instructions | D (Reinhart exemplar result; Zhou et al. overshoot) | No study of meaning or agency drift from voice conversion in academic prose |
| Rule NT (NEEDED?, then TRUE?) | P + I (Grice; Cochrane; the author); the need is D (Peters & Chin-Yee OR 4.85; Zhou et al.) | No evidence that redundant universals reduce credibility (hype RCT null); no published deletion-equivalence system |
| Disclosure content and placement | D (OUP July 2026 policy) | Verbatim policy wording to be checked on the page |
| Replication package | D (JFEc instructions; Data Editors template) | Data exemption terms for licensed data not researched |

### 8.9 Open questions for the author

Under A6 each question carries a tier and a status (§8.10). Tiers marked "proposed" are the report's pre-deliberation guess; a deliberation assigns the final tier, and only residual T3 items reach the author.

1. **Overleaf comments.** *[Resolved by A3 and A5.]* The author's corrections arrive as chat messages, and collaborators' Overleaf edits flow back through the pull. Comments are still not an input.
2. **Final form, section by section.** *[T2 proposed; open.]* Should each section use Option A (scaffold removal) or Option B (co-located prose)? The recommendation is A everywhere except the introduction and conclusion. The author's bullet-structure requirement and A2 suggest the answer.
3. **Parked text.** *[T3 proposed: it changes the location of an author house rule; open.]* Moving parked text to sidecar files changes the rule's location, not its substance. It needs the author's assent because `unflatten.py` writes the parks.
4. **The style corpus.** *[T3 proposed: it depends on the author's legitimate access; open.]* Which 60–150 papers make up the corpus, held through which legitimate access?
5. **Budget.** *[T3: spending and outward-facing; open.]* Is there budget for Refine (a reader-reported $30–50 per paper; [Litvak](https://paullitvak.com/p/how-well-does-ai-peer-review-work)) and for a second model vendor at milestones?
6. **Data.** *[T3: outward-facing and licence-dependent; open.]* Do the options data need JFEc's replication exemption?
7. **Voice rules.** *[T1 proposed once the corpus bands exist, T2 for the magnitude words; open.]* Should the notation guide's blanket "'We' and active voice" rule stand, or should per-section corpus bands set voice (§4.3.2)? Should its preferred magnitude words ("substantially", "materially", "modestly") become generator-emitted words tied to thresholds (§4.3.7)?
8. **Starting state (A2).** *[T3 after the 29 Oct deliberation; asked in that round's digest.]* Should the existing bullet text be reset, so L2 is generated fresh from the locked L1? Or should it be kept as the first L2 draft, which the author reads once and sends edits on? Keeping it saves generation, but it serves text written before the locks existed.
9. **Author-typed text and the scrub.** *[Answered by A4: no.]* Rule NT and the rest of the scrub never alter text the author supplied, in any layer or the final build.
10. **Collaborator authority (A5).** *[T2 after the 29 Oct deliberation: default kept, held edits shown as a draft-only note; vetoable.]* The proposed default: a collaborator's Overleaf edit is applied as text and recorded with its source. It locks only when the author adopts it, and one that would break a lock is held and shown as a conflict. Keep this, or let named coauthors' edits lock directly?
11. **A4 for collaborators.** *[T1 after the 29 Oct deliberation: human text pulled from Overleaf is never scrubbed.]* Should collaborators' text get the same verbatim protection from the scrub and Rule NT as the author's?
12. **Tracked changes (A5).** *[T1 after the 29 Oct deliberation, with one conditional question if a read-only check finds track changes on for everyone.]* Should collaborators accept or reject tracked changes on Overleaf before each pull, or should the pull detect and report them? Whether the git bridge exposes tracked changes or comments at all is unverified.
13. **Agent edit route on Overleaf.** *[T1 after the 29 Oct deliberation: keep git; no claudeleaf, no bot account.]* Should agents keep the git route, or post edits as tracked-change suggestions from a dedicated bot account through the live editor (claudeleaf)? The bot route makes agent edits attributable and leaves comments in place. It needs premium track changes, cannot resolve comments, and edits the live document (§1.5, §8.5).

### 8.10 Question tiers and deliberation

**Authority (A7).** [`writeup/deliberation/PROTOCOL.md`](../deliberation/PROTOCOL.md) is the single source of these rules and is authoritative wherever it and this section differ. The author's decisions govern both. This section keeps the rationale and the evidence basis.

**What is encoded (commit 930958b).**

- **`PROTOCOL.md`** holds the rules:
  - the tiers, with numeric thresholds;
  - hard triggers that are always T3, whatever the confidence: compute beyond approved runs, outward-facing actions, config changes, deleting data, spending, and other irreversible actions;
  - an importance filter, under which abstract, introduction, headline claims and main-text numbers or tables are HIGH, and wording, appendix-only and process items are LOW;
  - at most three T3 questions per digest;
  - a one-line veto for any applied outcome.
- **`SCHEMA.md`** defines the data shapes.
- **`digest.py`** applies the filter in code. It recomputes every tier and importance, can only raise what the adjudicator declared, and writes the only text the author is sent; 23 tests guard it against drift.
- **`queue.jsonl` and `log.jsonl`** hold the open items and the record.
- **The project skill `.claude/skills/deliberate/SKILL.md`** is the procedure an agent runs.
- **The author's global `CLAUDE.md`** now says never to ask the author directly, but to enqueue the item and run the skill.

The first live round, on 2026-10-29, sent the author one T3 question (Q8) and two hard-trigger items, among them a proposed `CLAUDE.md` wording change (a `config` trigger). Everything else was applied or settled as T0–T2 (`writeup/deliberation/2026-10-29/`).

A6 asks for two things. Questions are sorted by how much they need the author, and every candidate question is deliberated at length before it reaches the author. The router applies to any open question: lock conflicts, flips and uncertain claims (§8.5), ambiguous edits (§8.5's step 1), the open questions of §8.9, and anything an agent would otherwise ask.

| Tier | What qualifies | What happens | Shown to the author |
|---|---|---|---|
| **T0 mechanical** | Follows from a lock, a generated fact or a rule (for example, a comparison word set by a predicate, or a Rule NT deletion) | Act and log | Never |
| **T1 evidence-resolvable** | Data, results, the literature, or the author's recorded decisions and quotes settle it; independent solvers agree and a refuter fails | Act, keep it reversible, record it in `decisions.yaml` with source `inferred` and its evidence | In a skimmable digest |
| **T2 inferred preference** | A matter of taste, but the author's dated quotes and past choices imply the answer | Apply provisionally, marked `inferred` | In the digest; the author can veto in one line |
| **T3 author judgment** | The author's own locks conflict with no precedence rule or later quote to settle them; or the evidence stays balanced after deliberation; or the action is irreversible or outward-facing (a submission, a spend, a public post) | Ask once, batched and ordered by impact. Each question carries a recommendation, a confidence, the alternatives, its where-used consequences (§8.5) and a default | Yes; only residual T3 |

**Deliberation before any T3.** No item is classed T3 until a deliberation has run:

1. **Independent solvers.** At least two, preferably three, solvers work the question with different lenses and without seeing each other's work:
   - the evidence and results;
   - the author's record of dated quotes and decisions (`decisions.yaml`, the dated `% (user|author, …)` comments, commit messages);
   - a JFEc referee's view.
2. **Adjudication.** An adjudicator compares the answers and tries to refute the majority answer against the repo. It checks for a later author decision that supersedes an earlier one, which is the precedence rule. It then assigns the tier.
3. **Saved record.** Every deliberation is saved under `writeup/deliberation/<date>/`. Every T1 or T2 outcome is recorded in `decisions.yaml` with source `inferred` and its evidence, so the author's veto is a single line.

**Basis.** The solver–refuter–adjudicator structure transfers the AI-review evidence of §5.4 (D in other fields):

- pooled independent reviewers raised detection of planted errors from 71 to 91–93 ([Litvak](https://paullitvak.com/p/how-well-does-ai-peer-review-work));
- a separate verifier reached 83.2% precision ([Bianchi et al.](https://arxiv.org/html/2512.05925v1));
- single runs rarely rediscover the same errors ([SPOT](https://arxiv.org/abs/2505.11855)).

The tiering transfers code review's rule of sending direction-level problems first (T-code), and the assurance-case split between computed potential impact and judged actual impact (§1.5). The thresholds and the routing are inference (I, A6). Two risks remain unmeasured:

- that deliberation spends tokens on questions a human would settle in seconds;
- that agreeing solvers share one model's blind spots. Cross-vendor solvers help (§1.3).

**First live run.** On 29 October deliberation started on the fresh version's conflicts C1–C7 (these are not the checks C1–C7 of §8.3) and on §8.9 Q8 and Q10–Q13. Their tiers and outcomes will be recorded under `writeup/deliberation/2026-10-29/` and in §8.9.

---

## Conclusion

The research turns the author's question around twice.

First, almost nothing in the design is new (§1.5). Requirements engineers flag what a changed decision touches. Technical writers define values once and filter layers by condition. Spec-driven coding tools keep binding upper layers above generated code. What is new is assembling these parts for a paper, filtering the author's review on outcome flips rather than input changes, and round-tripping prose both ways with Overleaf. Those three pieces are where this project must produce its own evidence.

Second, the hard part of an AI-assisted paper is not getting an agent to write. Korinek rated that "highly useful" in 2023, and every pipeline reviewed here writes fluently. The hard part is making each kind of error **impossible, mechanical to detect, or cheap to judge**, in that order:

- **Transcription errors become impossible** once no agent can type a digit and every generated file proves it is current.
- **Unsupported connective claims become impossible** if the final paper is a rendering of approved bullets rather than a fresh expansion.
- **Citation errors become mechanical** once entries are fetched rather than typed.
- **Universals and superlatives become mechanical** under Rule NT: they are deleted when the context already implies them, and otherwise tied to a count or an argmax.
- **Scope and design drift become mechanical** once every author decision is a lock that generators compute from and the build checks against.
- **Interpretation stays human.** It is the class every evaluation finds weakest: Kosmos's interpretive statements, data-to-paper's interpretation errors, the 10%-to-90% verdict flip. Under A1–A5 it lives in the beats, which the author reads first and corrects by message, and it stays locked from then on.

The author's decisions turn the layers from views into an **order of authority**. The simplest layer comes first and is read once. Each correction becomes a lock that every lower layer inherits, and polish comes last and cannot touch what is locked. Under that order, the cheapest reviewer, a script, checks everything beneath a lock. The most expensive reviewer, the author, reads each layer once in its plainest form and afterwards sees only flips and conflicts. The D13 drift shows why this has to be mechanical: a design decision adopted on 26 October was still missing from three generators two days later.

Two things this design cannot yet promise are worth stating plainly. No study shows that a scrubbed, register-matched draft earns better referee reports, so the scrub should be judged by the corpus dashboard and the author's eye, not by detectors. No study measures what an author misses by correcting simple layers once instead of reviewing every edit. Keeping each edit, flip and conflict in `decisions.yaml`, dated and with its message-to-PDF time, would give this project the first such record for an empirical finance paper.
