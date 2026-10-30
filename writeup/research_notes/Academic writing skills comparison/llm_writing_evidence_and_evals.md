# LLM writing assistants and style instructions: what works, what fails, how to evaluate, and what it means for paper-prose

Scope note. These notes skip sources that `writeup/reports/AI layered paper writing system.md` §4 and §5 already cite (Kobak, Liang ICLR reviews, Reinhart, Shaib, Padmakumar & He, Kim, Park, Peters & Chin-Yee, Zhou, Patchwork, InkSync, AnchoredAI, Russell et al., Chakrabarty CHI 2025, Antislop; and OUP, Elsevier, AEA, SSRN and Wiley in §7). They are cited here only as cross-references ("report §x"). Most findings below were read from abstracts, search summaries or the arXiv HTML. A figure taken from a secondary summary is marked "(secondary)", and a claim from memory with no fetched source is marked "(memory)". Repo measurements were made on 2026-10-29 with `wc -w` over the current files.

## 1. Instruction-following for style: long rule lists, position effects, checklists and self-critique

### Takeaway
Models follow a few explicit constraints almost perfectly. Satisfying all of them at once gets harder as the rules multiply. Later items in a list are dropped more often, and the typical failure is silent omission rather than a wrong attempt. Checklists help both generation and judging by a few points. Self-critique works when it has an external signal, such as a checklist, a tool or a verifier. Without one it is unreliable, and the model favors its own text.

### Cited Findings
- **Density degradation (IFScale).** IFScale gives 20 models a report-writing task with 10 to 500 keyword-inclusion instructions. Even the best frontier models reach only about 68% at 500 instructions ([Jaroslawicz et al., NeurIPS 2025](https://arxiv.org/html/2507.11538v1)). Accuracy by model:

  | Model | 10 | 50 | 100 | 250 | 500 |
  |---|---|---|---|---|---|
  | claude-opus-4 | 100% | 100% | 94.6% | 67.9% | 44.6% |
  | claude-sonnet-4 | 100% | 98.0% | 94.4% | 77.2% | 42.9% |
  | claude-3.7-sonnet | 100% | 99.6% | 94.8% | 72.9% | 52.7% |
  | gemini-2.5-pro | 100% | 99.6% | 98.4% | 84.8% | 68.9% |
  | o3 (high) | 100% | 99.6% | 98.2% | 97.8% | 62.8% |

  — [IFScale](https://arxiv.org/html/2507.11538v1)
- **Three degradation shapes.** "Threshold" models (o3, gemini-2.5-pro) stay near perfect through about 150 instructions. "Linear" models (gpt-4.1, claude-3.7-sonnet) decline steadily. "Exponential" models (llama-4-scout and others) collapse early. The paper is internally inconsistent about which models fall in the exponential group. — [IFScale](https://arxiv.org/html/2507.11538v1)
- **Primacy bias.** Instructions later in the list are violated more often. The ratio of last-third to first-third error rates peaks at around 150–200 instructions; for gpt-4.1-mini it is 1.50 at 50, 3.37 at 100 and 1.62 at 500. The authors' only design advice is that "packing more important instructions towards the beginning of a prompt may help". — [IFScale](https://arxiv.org/html/2507.11538v1)
- **Failures are omissions.** "Models overwhelmingly err toward omission errors as instruction density increases." At 500 instructions the omission-to-modification ratio is 6.3 for o3, 7.0 for gemini-2.5-pro and 31.7 for claude-3.5-haiku. — [IFScale](https://arxiv.org/html/2507.11538v1)
- **Small counts already hurt the all-constraints rate.** At small instruction counts, the probability of satisfying every constraint falls as instructions are added, even when each one is simple. Summaries report that Claude 3.5 and Gemini 1.5 Pro fall well below GPT-4o in prompt-level accuracy at six simultaneous instructions (secondary). — [Harada et al., "When Instructions Multiply", arXiv 2509.21051](https://arxiv.org/pdf/2509.21051); [secondary summary](https://viblo.asia/p/the-curse-of-instructions-khi-llm-ngop-tho-vi-qua-nhieu-yeu-cau-cung-luc-ymJXDDo6Jkq)
- **Conflicting instructions.** "Instruction Boosting", a post-generation revision step, raises the instruction-following rate by up to 7 points with two instructions and up to 4 points with ten (SCALEDIF benchmark). The authors name rising tension or conflict between instructions as a main driver of degradation, and they release a conflict-scoring tool. — [Instruction Boosting, arXiv 2510.14842](https://arxiv.org/abs/2510.14842)
- **Position in long contexts (lost in the middle).** Accuracy is U-shaped in the position of the relevant information: best at the start or end of the context, worse in the middle. The pattern holds for long-context models. — [Liu et al., TACL 2024](https://aclanthology.org/2024.tacl-1.9/)
- **Multi-turn sessions.** Top models lose on average 39% of their performance across six generation tasks when instructions arrive over several turns rather than in one prompt. Most of the loss is unreliability rather than lost aptitude, and models "make assumptions early" and over-rely on them. — [Laban et al., arXiv 2505.06120](https://arxiv.org/pdf/2505.06120)
- **Negative instructions are weaker.** Larger models did not get better at following negated instructions ([Jang et al., NeurIPS 2022 workshop](https://mlanthology.org/neuripsw/2022/jang2022neuripsw-large)). In one model (Qwen2.5-7B), a "do not use X" instruction lowers the probability of X, but the suppression is about 4.4 times weaker in the failing cases ([Semantic Gravity Wells, arXiv 2601.08070](https://arxiv.org/html/2601.08070v1)). The em-dash persistence result is already in report §4.2.
- **Checklists help judging and refinement (TICK).** Model-generated, instruction-specific yes/no checklists raised LLM-judge exact agreement with human preferences from 46.4% to 52.2% over direct scoring. Self-refinement against the checklist (STICK) gained +7.8 points absolute on LiveBench reasoning, and Best-of-N selection with STICK +6.3 on WildBench. Giving the checklists to human annotators raised their inter-annotator agreement from 0.194 to 0.256. — [Cook et al., TICK, arXiv 2410.03608](https://arxiv.org/html/2410.03608v1)
- **Checklists as an RL reward (RLCF).** Rewarding against instruction-specific checklists, with items scored by AI judges and by verifier programs, was the only method that improved all five benchmarks for Qwen2.5-7B: +4 points on FollowBench hard satisfaction, +6 on InFoBench and +3 on Arena-Hard (NeurIPS abstract; arXiv v2 reports relative gains of 5.4%, 6.9% and 6.4%). — [Viswanathan et al., arXiv 2507.18624](https://arxiv.org/html/2507.18624v2)
- **Self-Refine.** Generate, self-critique and revise improved seven tasks by about 20 points absolute on average over one-shot output, by human preference and automatic metrics (GPT-3.5 and GPT-4). — [Madaan et al., NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract.html)
- **Self-correction without external feedback.** Without external feedback, LLMs "struggle to self-correct their responses ... and at times, their performance even degrades after self-correction". Reported cause: models rarely spot flaws in their own reasoning. Secondary figures, unverified: GPT-4 on GSM8K falls from 95.5% to 91.5% after one round. — [Huang et al., ICLR 2024](https://arxiv.org/html/2310.01798v2); [secondary](https://beancount.io/bean-labs/research-logs/2026/04/28/llms-cannot-self-correct-reasoning-yet)
- **Constitutional AI (memory).** The critique-and-revise loop draws one principle from the constitution for each critique pass, rather than presenting the whole rule list at once (memory). — [Bai et al. 2022, arXiv 2212.08073](https://arxiv.org/abs/2212.08073)
- **Verifiable instructions (IFEval, memory).** IFEval is built on instructions a program can check ("verifiable instructions") and reports strict and loose accuracy (memory). — [Zhou et al. 2023, arXiv 2311.07911](https://arxiv.org/abs/2311.07911)

### Inferences
- Rules a program can check (word lists, em dashes, digits outside macros, macro multisets, byte-identity) should be checked by code after generation. They should not be trusted to the model's memory of a long rule list, because density, position and omission effects all predict silent misses.
- Ordering matters, but only modestly: put the rules whose violation is costliest first. paper-prose already orders scope, then claims, then numbers, then anti-patterns. IFScale's primacy result supports that order for drafting prompts.
- Self-critique works when the critic has something external to check against: a yes/no checklist item, a grep hit, a verifier. Free-form "review your work" by the same model in the same context is the weak case (Huang; and see §4 on self-preference).
- Giving one principle per critique pass (Constitutional AI) or one operation per sentence (report §4.3.5) is the low-density design that the IFScale and Harada results favor.

### Gaps
- No study found tests adherence to a long prose style guide (thousands of words of rules with examples) as opposed to keyword or format constraints. IFScale's keyword-inclusion task is a proxy.
- Harada et al.'s per-count accuracy figures could not be extracted (the PDF was compressed), so the six-instruction gap is secondary.
- No study found compares "rule plus rewrite table" against "rule only" formats for style adherence.
- No Claude-5-era model appears in IFScale; degradation shapes for current models are unknown.

## 2. Meaning preservation in LLM editing: drift, factual and numeric changes, and guards

### Takeaway
Asked to "correct" or "improve" text, LLMs make more edits than were asked for (over-correction). They also introduce factual errors at non-trivial rates. Surface-similarity metrics (ROUGE, BERTScore) do not detect meaning changes. Bidirectional entailment is the standard automatic paraphrase test, but it is weak on numbers. The guards that hold up are minimal-edit instructions, explicit checks on entities, numbers and negation, and human review of changes flagged as substantive.

### Cited Findings
- **Over-correction in grammatical error correction (GEC).** GPT models "have a tendency to over-correct and make extraneous fluency edits". This explains high scores on fluency benchmarks (JFLEG) and weak scores on minimal-edit benchmarks. — [Coyne et al., arXiv 2303.14342](https://arxiv.org/pdf/2303.14342); [Fang et al., arXiv 2304.01746](https://arxiv.org/pdf/2304.01746)
- **Over-correction is version-dependent.** In Brazilian Portuguese GEC, GPT-3.5 over-corrected 54 of 484 sentences and GPT-4 six of 484. — [Penteado & Perez, arXiv 2306.15788](https://arxiv.org/pdf/2306.15788)
- **Minimal-edit prompting.** A 2026 paper reports that batched inference on top of minimal-edit instructions "systematically" lowers the edit rate across LLM families (abstract only; details unverified). — [arXiv 2609.10810](https://arxiv.org/pdf/2609.10810)
- **Factual errors in AI-revised abstracts.** Researchers found 10 minor and 3 major factual errors in ChatGPT-4's rewrites of 24 conference abstracts. — [Gravel et al., medRxiv 2024](https://medrxiv.org/content/10.1101/2024.02.09.24302591v1.full.txt) (search summary)
- **Similarity metrics miss meaning changes.** ROUGE and BERTScore-type metrics track revision quality poorly (258 human-revised scientific paragraph pairs). The authors recommend a hybrid: an LLM judge for instruction-following plus a separate meaning-preservation metric, such as ParaPLUIE, a perplexity-based yes/no check. — [Jourdan et al., arXiv 2506.04772](https://arxiv.org/html/2506.04772v5)
- **Source overlap is not preserved meaning.** In LLM simplification of scientific summaries, high source similarity did not guarantee that meaning, terminology or claim strength survived. Expert post-editing was still needed. — [arXiv 2607.25630](https://arxiv.org/pdf/2607.25630)
- **Bidirectional entailment, and where it breaks.** Mutual NLI entailment is the usual operational test for paraphrase. A plain "rephrase preserving meaning" prompt is not enough, because generators produce non-equivalent paraphrases that must be filtered. These two points come from a search summary that did not clearly say which of the listed papers made each one; the likeliest source is [Verma et al., "Evaluating Paraphrastic Robustness in Textual Entailment Models"](https://arxiv.org/html/2306.16722v1). Check it before citing.
  - In biomedical NLI, rewrites with numbers failed on fraction, decimal and percent conversions and on time and measurement units (search summary). — [FZI-WIM, SemEval-2024](https://arxiv.org/pdf/2406.10040)
- **Prompt-judge alternative.** One prompt-based equivalence judge checks entities, numbers, dates, units, polarity and negation, and causal or temporal relations. It treats any mismatch in numbers, entities, dates or negation as a severe error (search summary). — [arXiv 2510.08616](https://arxiv.org/pdf/2510.08616v1)
- **Already in the report.** InkSync: conversational editing produced major factual errors in 14.8% of documents versus 6%, and "Warn + Verify" raised prevented inaccurate edits from 23.4% to 43.5% (report §5.1). Peters & Chin-Yee: LLMs drop scope limits, OR = 4.85 for broad generalizations (report §4.3.6).

### Inferences
- paper-prose's V16 ("same multiset of macros, cites, refs; negations, modals, direction words unchanged; no invented agent") is stronger than NLI on the failure that matters most here. Numbers are macros, so a multiset diff catches any numeric change exactly, where NLI is weakest. V16's list matches the prompt-judge rubric above (entities, numbers, negation, polarity), plus agency.
- What V16 does not cover: claim-strength drift that keeps every macro but changes the verb or quantifier, as when "lowers" becomes "substantially lowers". V4 and V5 cover part of this. A deletion or entailment check in both directions (report §4.3.7 Stage B) is the natural automatic complement for non-math sentences.
- The GEC evidence is a direct analog for V14 (no polish outside the target): an LLM asked to fix one thing also makes fluency edits elsewhere. An "edit only the targeted sentence" constraint needs a diff check, not just an instruction.

### Gaps
- No study found measures how often LLM style editing changes numbers, hedges or claim strength in academic prose specifically. Studies cover GEC, simplification and summarization.
- No published pipeline found combines bidirectional NLI with explicit numeric and unit diffing, or reports its precision and recall on rewrites.
- The over-correction rates of current Claude models on minimal-edit tasks were not found.

## 3. AI-edited academic text: reader ratings, homogenization, credibility and publisher positions (to Oct 2026)

### Takeaway
Where measured, authors who use AI suggestions on their own drafts produce better abstracts than AI-only drafts. AI-only drafts score below the originals and contain factual errors. At scale, LLM adoption raises output and prose complexity but breaks the link between complex prose and acceptance. AI suggestions homogenize style toward a dominant norm. Disclosure carries a measurable trust penalty, yet policies do not change behavior and almost nobody discloses. Publishers agree on three things: no AI authorship, human accountability, and disclosure for anything beyond copy editing.

### Cited Findings
**Reader and reviewer ratings**
- **Author plus AI suggestions beats AI alone (Gravel et al.).** Twenty-four researchers revised their own Pediatric Academic Societies abstracts with ChatGPT-4's suggestions. Blinded co-investigators rated the final versions above the originals (mean difference 1.10 points; 95% CI 0.54–1.66). The ChatGPT-only draft scored lowest (summary means: original 82, ChatGPT 65, final 90). The authors' conclusion: ChatGPT does not write better abstracts than researchers but "offers suggestions to help them improve". Three raters; preprint. — [Gravel et al., medRxiv 2024](https://medrxiv.org/content/10.1101/2024.02.09.24302591v1.full.txt)
- **Generated abstracts score below originals.** Blinded reviewers scored ChatGPT-generated abstracts about 4.7 against 8.1 for originals and guessed which were AI with 93% accuracy (224/240) (secondary summary) ([Su, JMIR 2023](https://jmir.org/article/export/ris/jmir_v25i1e51229)). Reviewers flagged 68% of generated abstracts and 86% of originals correctly ([Gao et al., npj Digital Medicine 2023](https://www.nature.com/articles/s41746-023-00819-6.pdf)). Both studies concern generated, not edited, abstracts.

**Scale effects**
- **Kusumegi et al. (Science, Dec 2025).** About 2.1M preprints (arXiv, bioRxiv, SSRN, 2018–2024), 28k review reports and 246M online accesses:
  - LLM adoption raised output by 23.7–89.3% depending on field and author background, with the largest gains for authors facing language barriers.
  - LLM-assisted papers had higher writing complexity.
  - For human-written papers, complexity predicted acceptance. For LLM-assisted papers the relationship reversed: complex AI-flagged prose was less likely to be accepted.
  - — [Kusumegi et al., via ScienceDaily](https://sciencedaily.com/releases/2025/12/251224032347.htm); [EurekAlert](https://www.eurekalert.org/news-releases/1109808)

**Homogenization**
- **Homogenization toward a dominant style.** In a 118-participant India/US experiment with GPT-4o autocomplete, AI suggestions made Indian participants write "more like Americans", both in content and in "more ingrained elements of how it's written". Indian participants gained less speed because they had to correct suggestions more often. — [Agarwal, Naaman & Vashistha, CHI 2025](https://arxiv.org/html/2409.11360v3)
- **Review (Trends in Cognitive Sciences).** LLMs homogenize expression and reasoning through two mechanisms, training-data mirroring and convergence from shared use. No effect sizes are quantified. — [Sourati, Ziabari & Dehghani](https://arxiv.org/html/2508.01491v2)
- **Ownership and voice (ghostwriter effect).** Writers do not feel they own AI-generated text but still declare themselves its authors (n = 30 and n = 96). Felt ownership rises with personal involvement in the writing. — [Draxler et al., ACM TOCHI 2024](https://arxiv.org/abs/2303.03283v1)

**Credibility**
- **Disclosure penalty.** Thirteen experiments: disclosing AI use lowers trust in the discloser, by 16% for a professor's grading, 18% for investors in firms and 20% for designers' clients. The penalty survives "proofreading only" and "human-reviewed" framings and holds for voluntary and mandatory disclosure. Being exposed for undisclosed use is penalized more. — [Schilke & Reimann, OBHDP 2025](https://news.arizona.edu/news/disclosing-ai-use-can-backfire-research-shows); [record](https://experts.arizona.edu/en/publications/the-transparency-dilemma-how-ai-disclosure-erodes-trust/)
- **Simpler summaries read as more credible.** GPT-4 plain-language significance statements were simpler, improved lay comprehension and made scientists seem more credible and trustworthy but less intelligent. Some effect sizes were small. — [Markowitz, PNAS Nexus 2024](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11406778/)

**Publisher positions (to Oct 2026)**
- **Policies do not curb use (He & Bu, PNAS 2026).** Across 5,114 journals and 5.2M papers, 70% of journals adopted AI policies, yet AI-assisted writing grew with no significant difference between journals with and without policies. Of about 75k post-2023 full texts, only about 0.1% (about 76) disclosed AI use (secondary summaries). — [PNAS 10.1073/pnas.2526734123 (Nature Index listing)](https://www.nature.com/nature-index/article/10.1073/pnas.2526734123); [InfoDocket](https://www.infodocket.com/2026/02/24/journal-article-academic-journals-ai-policies-fail-to-curb-the-surge-in-ai-assisted-academic-writing/)
- **ICMJE.** Authors disclose AI-assisted technologies "in both the cover letter and the submitted work in the appropriate section if applicable". "Humans are responsible for any submitted material that included the use of AI-assisted technologies", and authors should "carefully review and edit the AI-generated content as the output can be incorrect, incomplete, or biased". They must ensure attribution and full citations, and AI cannot be an author. The page is © 2026; CASRAI says the recommendations were updated in January 2026 (secondary). — [ICMJE, AI use by authors](https://www.icmje.org/recommendations/browse/artificial-intelligence/ai-use-by-authors.html); [CASRAI](https://casrai.org/guides/can-ai-be-listed-as-an-author)
- **COPE.** The position statement "Authorship and AI tools" (2023; page © 2024) says AI cannot be an author and authors are responsible for all content. No 2025–2026 revision was found. A July 2025 COPE forum covered disclosure and detection, and COPE relaunched its paper-mills working group in June 2026 (secondary). — [COPE](https://publicationethics.org/guidance/cope-position/authorship-and-ai-tools); [CASRAI](https://casrai.org/news/cope-relaunches-paper-mills-working-group-2026)
- **Springer Nature.**
  - The group guidance says AI used for copy editing need not be declared, and the author stays accountable (search summary of the group page, which returned 404 on fetch). — [Springer Nature AI guidance](https://group.springernature.com/br/group/ai/ai-guidance-for-our-researchers-and-communities)
  - The Nature Portfolio wording defines "AI assisted copy editing" as improvements to human-generated text "for readability and style" and for grammar, spelling, punctuation and tone. It excludes "generative editorial work and autonomous content creation", and authors must agree that "the edits reflect their original work" (memory; the policy page is behind a login redirect).
  - The current top-level Springer Nature policy states "human accountability cannot be transferred to AI systems". — [Springer Nature editorial policies](https://www.springernature.com/gp/policies/editorial-policies)
- **STM classification.** STM's nine-category classification of AI use in manuscript preparation (final, September 2025) is a template that publishers can build their disclosure policies on, not a policy itself. — [NISO](https://niso.org/niso-io/2025/09/stm-releases-recommendations-classification-ai-use-manuscript-preparation); [STM draft](https://stm-assoc.org/new-stm-draft-report-classifying-ai-use-in-manuscript-preparation/)
- **Already in the report.** OUP (governs JFEc): substantial editing or rewriting needs disclosure; grammar and spelling help does not (report §7.1). Elsevier, AEA, SSRN and Wiley are in report §7.3.

### Inferences
- The pattern that measurably helps is the author's text, improved by the author choosing among AI suggestions. AI-only rewriting does worse. This supports V13 and V15 (the author's words are kept) and the layer design in which the author locks text before any edit.
- The Kusumegi reversal suggests that LLM-raised prose complexity is not read as quality and may be read as the opposite. A register pass should aim at plain, specific prose (V1, V10), not at "journal-sounding" complexity.
- The Nature Portfolio copy-editing exemption (memory) covers readability and style edits to human text, not generative work. The agent-drafted layers of this paper are generative under any publisher's definition, so disclosure is needed either way (report §7.1). The disclosure penalty is a cost to accept, not a reason to make the edits harder to detect.

### Gaps
- No study found rates LLM-edited (rather than generated) economics or finance papers with blinded referees.
- No study found measures homogenization across authors in a single field's journal prose after 2023 at the sentence-feature level; Kobak, Liang and Geng, already cited in the report, measure vocabulary.
- The primary Springer Nature and Nature Portfolio pages could not be fetched (login redirects, 404). The copy-editing wording should be checked on the live page before it is quoted.
- He & Bu's figures are from secondary summaries; the PNAS full text was not read.

## 4. Evaluating a writing skill: LLM-as-judge reliability, rubrics, pairwise preference, small blinded A/B

### Takeaway
LLM judges reach human-level agreement on general chat preference. They are close to chance on writing quality, and they carry position, length, style-over-substance and self-preference biases. Decomposing the judgment into yes/no checklist items raises agreement and makes it auditable. The credible, cheap design for a style skill therefore combines three things: deterministic checks for verifiable rules, a checklist judge from a different model, used to detect violations rather than to score quality, and a small blinded pairwise test with the author as rater.

### Cited Findings
- **Chat-preference agreement (MT-Bench).** GPT-4 judges reach over 80% agreement with human preferences, the same as human–human agreement, on MT-Bench and Chatbot Arena votes. The paper documents position bias, verbosity bias (tested with a "repetitive list" attack) and self-enhancement bias; GPT-3.5 and Claude-v1 showed strong position bias (secondary). — [Zheng et al., NeurIPS 2023](https://arxiv.org/html/2306.05685v4)
- **Twelve judge biases (CALM).** The CALM framework quantifies twelve biases, among them position, verbosity and self-enhancement. Each is tested by perturbing the judged answer and checking whether the verdict changes; position bias grows with the number of candidates. — [Ye et al., ICLR 2025](https://arxiv.org/html/2410.02736v2)
- **Self-preference.** LLM evaluators score their own outputs above others' relative to human judgments. Self-preference rises linearly with self-recognition ability, and the gap is largest against human-written text. — [Panickssery, Bowman & Feng, NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/hash/7f1f0218e45f5414c79c0679633e47bc-Abstract-Conference.html)
  - The preference persists when the judge is not told which text is its own, and tracks perplexity. — [Wataoka et al., arXiv 2410.21819](https://arxiv.org/html/2410.21819v1)
- **Style over substance.** Both LLM and human judges ranked answers with factual errors above answers that were too short or had grammatical errors. Human judges rarely fact-checked unless the error was glaring. Scoring separate dimensions (Multi-Elo) helped LLM judges on factual accuracy but did not help crowd raters. — [Wu & Aji, COLING 2025](https://arxiv.org/abs/2307.03025v3)
- **Writing quality is hard to judge.** On a 4,729-judgment Writing Quality benchmark, "most competitive baselines, including state-of-the-art LLMs" performed close to random. Reward models trained on expert edits (edited text preferred over the draft) reached 74%. Nine experienced writers preferred samples the reward model selected 66% of the time, and 72.2% when the reward gap exceeded one point. — [Chakrabarty, Laban & Wu, COLM 2025](https://arxiv.org/html/2504.07532v3)
- **Creative-writing pairwise preference (LitBench).** The best zero-shot judge, Claude-3.7-Sonnet, agreed with human pairwise preferences 73% of the time on 2,480 debiased Reddit story pairs. Trained Bradley–Terry and generative reward models reached 78%, and distilled chain-of-thought hurt. — [LitBench, arXiv 2507.00769](https://arxiv.org/pdf/2507.00769v1)
- **Validate before use (Judge-Bench).** Eleven LLMs on 20 human-annotated NLP evaluation datasets showed large variance by model, by the property judged, by annotator expertise and by whether the text was human- or model-written. The authors conclude that LLM judges must be validated against human labels for each task before use. — [Bavaresco et al., ACL 2025](https://aclanthology.org/2025.acl-short.20)
- **Checklists raise agreement (CheckEval).** Replacing Likert scores with decomposed yes/no checklist questions raised average agreement across 12 evaluator models by 0.45 and lowered score variance. Each judgment traces to a binary item. — [Lee et al., CheckEval v3](https://arxiv.org/html/2403.18771v2)
- **TICK checklists** also raised human inter-annotator agreement, from 0.194 to 0.256 (§1). — [TICK](https://arxiv.org/html/2410.03608v1)
- **Already in the report.** Referee simulators correlate with humans about as weakly as humans with each other (0.42 vs 0.41). Detect-then-verify reaches 83.2% precision, and pooling systems raises detection of planted errors (report §5.4).

### Inferences
**A cheap, credible eval suite for paper-prose.** Power arithmetic below is my own: normal approximation to the sign test, two-sided α = 0.05, power 0.8.

1. **Deterministic regression set (no LLM judge).**
   - Build a fixture of about 40–80 sentences, each labeled with the pattern it violates or labeled clean. Seed it from live repo cases, such as the V6 tails at `08_feature_importance.tex:224` and `:261` (V14 file).
   - Run the skill's branch G critic, or the Detection greps, and report recall and precision for each pattern id.
   - Use it as a regression test whenever a pattern file changes.
2. **Meaning-preservation tests for rewrites (V16).**
   - Over N agent rewrites, the pass condition is: macro, `\cite` and `\ref` multisets equal; no change to negations, modals or direction words; no new "we" agent. Target zero failures; any failure is a bug.
   - Add a two-way deletion and entailment check on non-math sentences for claim-strength drift.
3. **Over-edit rate (V14, V13, V15).**
   - On "apply this one edit" tasks, count sentences changed outside the target and any byte change to author or Overleaf text.
   - This is the GEC over-correction metric transferred to this setting. Target zero.
4. **Ablation against a no-skill baseline.**
   - Draft the same beats with the skill, with a plain "write this beat" prompt, and with the skill plus automatic greps.
   - Compare violations per 1,000 words, using pattern ids as the unit.
   - This measures whether routing to pattern files lowers violations beyond what greps alone achieve.
5. **Blinded pairwise author A/B on quality.**
   - Present pairs (skill vs baseline, same beat, order randomized, condition hidden). The author picks one and gives a reason code.
   - Required pairs: about 29 to detect a true 75% preference, about 85 for 65%, about 194 for 60%. With one rater and about 30 pairs, only a large effect is detectable.
6. **LLM judge only as a violation detector.**
   - Use a different model family than the writer (self-preference), run both A/B orders (position bias), control length, and ask per-item yes/no questions (CheckEval/TICK) rather than "which is better written".
   - Validate on the item 1 fixture against the author's labels before trusting it (Judge-Bench).
   - Never use an LLM quality score as the target: judges are near chance on writing quality (WQ benchmark) and are biased toward fluent text with errors (Wu & Aji).
7. **The author's own corrections as an edit-based signal.**
   - Log the author's edits per 1,000 words on each locked layer, tagged by pattern id where one applies.
   - A falling rate over layers, or a falling rate for one pattern after it changes, is the WQRM idea (edits mark the better text) applied to this one-author setting.

### Gaps
- No study found validates LLM judges on economics or finance prose, or on rule-violation detection for a house style guide. The item 1 fixture would be the first validation here.
- No published standard was found for sample size in single-rater writing A/B tests. The power numbers above are arithmetic, not a field norm.
- The exact MT-Bench verbosity-attack and position-consistency figures were not retrieved (memory: GPT-4's position consistency was well below 100%).

## 5. What to avoid: detector-evasion framing, voice-flattening targets, rule bloat, unverifiable rules, rewriting author text

### Takeaway
Each item on the avoid list has evidence behind it. Detectors are unreliable and biased, so evading them is a meaningless target. Expert readers detect AI drafting regardless of markers (report §4.5). Global style targets cause over-editing and homogenization. Long rule lists lose rules silently, and judgment-only rules cannot be checked by the same model reliably. Replacing the author's words removes the one measured source of quality and ownership.

### Cited Findings
- **Detectors are biased.** Seven GPT detectors flagged non-native TOEFL essays as AI-written at an average false-positive rate of 61.3%, against near-zero for US eighth-grade essays. Asking ChatGPT to enrich the TOEFL essays' vocabulary cut the rate to 11.6%: detector scores track perplexity, not authorship. — [Liang et al., Patterns 2023](https://arxiv.org/html/2304.02819v3)
- **Expert readers.** Expert human readers kept 100% detection on "humanized" text (report §4.5, Russell et al.).
- **Disclosure penalty.** Concealment that is later exposed costs more trust than disclosure. — [Schilke & Reimann 2025](https://news.arizona.edu/news/disclosing-ai-use-can-backfire-research-shows)
- **Global targets cause extra edits and dominant-norm drift.** Global "correct/improve" prompts produce extraneous fluency edits ([Coyne et al.](https://arxiv.org/pdf/2303.14342)), and AI suggestions pull writers toward a dominant style ([Agarwal et al., CHI 2025](https://arxiv.org/html/2409.11360v3)). Exemplar-style and accuracy prompts overshoot or fail (report §4.3.5).
- **Rule bloat.** Adherence falls with instruction count, later rules are dropped more, and failures are mostly silent omissions ([IFScale](https://arxiv.org/html/2507.11538v1)). Rising conflict between rules is a named cause ([arXiv 2510.14842](https://arxiv.org/abs/2510.14842)).
- **Unverifiable rules.** Same-model self-correction without external feedback can degrade output ([Huang et al.](https://arxiv.org/html/2310.01798v2)), and judges favor their own text ([Panickssery et al.](https://papers.nips.cc/paper_files/paper/2024/hash/7f1f0218e45f5414c79c0679633e47bc-Abstract-Conference.html)). Verifiable checks are what make checklist methods work ([RLCF](https://arxiv.org/html/2507.18624v2)).
- **Negative phrasing.** Negatively phrased rules are weaker than positive ones ([Jang et al.](https://mlanthology.org/neuripsw/2022/jang2022neuripsw-large); [arXiv 2601.08070](https://arxiv.org/html/2601.08070v1)).
- **Rewriting author text.** Abstracts improved when authors kept control and took AI suggestions, while AI-only rewrites were worse and introduced factual errors ([Gravel et al.](https://medrxiv.org/content/10.1101/2024.02.09.24302591v1.full.txt)). Ownership falls as personal involvement falls ([Draxler et al.](https://arxiv.org/abs/2303.03283v1)).

### Inferences
- For each rule, the question to ask is "can code or a yes/no question check it?" A rule that cannot be checked should be few in number, placed early, and reviewed by a separate critic, not by the drafter.
- A forbidden-word rule should be stated as a rewrite ("instead of X, write Y") and enforced by a grep, not just as "do not use X".

### Gaps
- No study found measures the optimal number or length of style rules for current frontier models in a prose-writing task.

## 6. Implications for paper-prose: what to borrow, what to avoid

### Takeaway
paper-prose already embodies most of the evidence-backed choices:
- task routing to a few rules, with costly-first ordering;
- named single operations instead of global style targets (V12);
- a content-preservation diff stronger than NLI for numbers (V16);
- protection of author and Overleaf text (V13, V15);
- a lock gate against premature polish (V14);
- a positive rewrite table in every pattern.

Its weak points:
- load: 6,000–6,900 words per branch;
- a self-check run by the drafting agent in its own context;
- Detection greps written as prose, not as a script;
- no evaluation of whether the skill lowers violations.

### Cited Findings (repo measurements and evidence links)
- **Size.** The catalog is 18 pattern files totalling about 8,760 words: V1–V16, AP-V and AP-S, between 357 and 683 words each. `SKILL.md` is 1,283 words and `voice.md` 860 (repo, `wc -w`, 2026-10-29; `writeup/style/patterns/`, `.claude/skills/paper-prose/SKILL.md`).
- **Branch load.** Branch A (drafting a claim) has an agent read SKILL.md plus nine pattern files, about 4,700 words of patterns and about 6,000 in all. Branch F (edit-stage pass) reads eleven files, about 5,600 words of patterns and about 6,860 in all (repo).
- **Self-check.** Step 2 is a 16-line self-check. SKILL.md says "Most automatic checks (C2, C5, C18, C25, C26, `scrub.py`) are not built yet, so this checklist is the check" (`.claude/skills/paper-prose/SKILL.md`, Step 2).
- **Lock gate.** `writeup/layer_state.json` does not exist, so V14 blocks every edit-stage pass (repo, 2026-10-29; V14 file).
- **Evidence matches.** The evidence lines up with the following existing design choices:
  - Routing and ordering: [IFScale](https://arxiv.org/html/2507.11538v1).
  - One operation per sentence (V12): Constitutional AI's one principle per pass ([memory](https://arxiv.org/abs/2212.08073)) and the GEC over-correction results ([Coyne et al.](https://arxiv.org/pdf/2303.14342)).
  - Macro-multiset check (V16): NLI's numeric weakness ([SemEval-2024 FZI-WIM](https://arxiv.org/pdf/2406.10040)).
  - Author text protected (V13, V15): [Gravel et al.](https://medrxiv.org/content/10.1101/2024.02.09.24302591v1.full.txt) and [Draxler et al.](https://arxiv.org/abs/2303.03283v1).
  - Rule NT's TRUE? backed by computed facts rather than a judge: judges do not fact-check ([Wu & Aji](https://arxiv.org/abs/2307.03025v3)).

### Inferences
**Borrow (adopt or strengthen)**
1. **Script the Detection sections now.** Put the greps already written in V2, V5, V6, V10, V16, AP-V and AP-S into one script that runs after every drafting task. Its output becomes the external feedback for the Step 2 self-check. This turns most checklist lines into verifiable instructions (IFEval/RLCF logic) and removes their dependence on the model's recall of a 6,000-word load (IFScale omission and primacy effects).
2. **Split the self-check.** Keep the drafter's checklist, but answer it per item as yes/no against the script's hits (TICK/CheckEval). Run branch G in a fresh-context subagent, ideally a different model than the drafter, for the judgment-only lines (V1, V4, V5 NEEDED?, V7, V8). This follows Huang et al. on self-correction without feedback and Panickssery et al. on self-preference.
3. **A compact card per pattern for drafting.** For branches A–D, give the intent plus the "Instead of / Write" table (about 100–150 words per pattern) and keep the full GoF file for edge cases and the critic. This cuts the branch load by perhaps two-thirds while keeping the positive rewrite format, which the negation evidence favors. The token saving is an estimate, not a measurement.
4. **Make V16 a code check before the edit stage opens.** It is the strongest guard in the catalog and the cheapest to automate. Add a two-way deletion and entailment check for claim-strength drift in non-math sentences (report §4.3.7 Stage B).
5. **Build the §4 eval suite before adding patterns:**
   - a labeled fixture of violations, with per-pattern recall and precision;
   - over-edit and V16 failure counts;
   - a skill vs no-skill ablation in violations per 1,000 words;
   - a ~30-pair blinded author A/B for quality, read as detecting only large effects;
   - logged author corrections per layer as the long-run signal.
6. **Order within branches by cost of violation.** The scope-first order already does this. Inside branch A, V1 is read before V4, V6 and V5. If claim errors cost more than sentence shape, IFScale's primacy result suggests moving V4 and V5 ahead of V1. This is a small, testable change.

**Avoid**
1. **Rule growth without retirement.** Each new pattern adds load to every branch that lists it. Gate new patterns on (a) a Detection method a script can run, or (b) a fixture showing the critic catches violations of it. Retire or merge patterns that the fixture shows are never violated.
2. **Judgment-only rules enforced by the drafter.** V1 "one claim a sentence", V8 and V5's NEEDED? need a separate critic or the author. The drafter's own "[x]" on these lines is weak evidence (Huang; Panickssery).
3. **LLM quality scores as targets.** Judges are near chance on writing quality and biased toward fluent text. Use LLMs to find violations, never to rank drafts for the author.
4. **Detector-evasion framing.** Keep the AP-V and AP-S rationale as register and information, as V5's "case rests on information" and report §4.5 already do. Detectors are biased and experts are not fooled.
5. **Global "make it read like JFEc" targets.** These are already banned by V12 and SKILL.md's Subagents section. Kusumegi's complexity reversal and the homogenization studies are added reasons.
6. **Any path that restyles author or Overleaf text.** V13 and V15 are well supported. Keep them absolute.
7. **Long multi-turn drafting sessions.** These carry early-assumption drift ([Laban et al.](https://arxiv.org/pdf/2505.06120)). Fresh subagents with the full input manifest per task (report §8.6) are the better pattern.

### Gaps
- Whether Claude follows sentence-level named operations more reliably than global instructions remains untested (report §4.3.5 says the same). The ablation in §4 item 4 would test it for this skill.
- The token and adherence effect of compact cards versus full pattern files is untested; it is an inference from IFScale.
- No evidence was found on how a GoF-style pattern catalog compares with flat rule lists for LLM adherence.
