# LLM prose markers ("LLM-isms") in academic writing: evidence, catalogs, and scrub tooling

Research date: 2026-10-28. Scope: 2023-2026 evidence. Context: an automated scrub-and-style-align step for a Journal of Financial Econometrics paper drafted mostly by Claude agents.

## Q1. Excess-vocabulary studies: Kobak et al., Liang et al. (ICML 2024 and Nature Human Behaviour 2025), Gray, Geng & Trotta, and the economics-specific work

### Takeaway
Population-level word-frequency studies agree on one vocabulary fingerprint. In 2024 the excess words were almost all *style* words, mostly verbs and adjectives: delve(s), underscore(s), showcasing, intricate, meticulous(ly), commendable, pivotal, realm, crucial, notably, additionally, comprehensive, enhancing, insights. The frequency jumps are 10-35x for the rare markers. These methods estimate shares across a corpus; they do not label individual papers. The fingerprint is also moving: "delve" fell after it was publicized, newer models have newer favorite words, and economics papers show the same signal (+6.67 pp in 2024 in the top 25 journals).

### Cited Findings

**Kobak, González-Márquez, Horvát & Lause, "Delving into LLM-assisted writing in biomedical publications through excess vocabulary" (Science Advances 11(27), 2 July 2025; arXiv 2406.07016)**
- Corpus: over 15 million PubMed abstracts, 2010-2024. Main estimate: at least 13.5% of 2024 abstracts were processed with LLMs, reaching 40% in some subcorpora. Earlier arXiv v4 gave ≥10% and 30%; cite the published figures. — [arXiv abs](https://arxiv.org/abs/2406.07016v5); [search summary of versions](https://arxiv.org/abs/2406.07016v4)
- Method: frequency p = (a+1)/(b+1), the share of abstracts containing the word. The counterfactual 2024 frequency q = p₂₀₂₂ + 2·max{p₂₀₂₂ − p₂₀₂₁, 0} extrapolates linearly from 2021-22 and skips 2023 because it may already be contaminated. The excess gap is δ = p − q and the ratio is r = p/q. A word counts as "excess" if δ > 0.01 or log₁₀ r > (log₁₀2/4)·log₁₀ p, and it must exceed frequency 10⁻⁴ in both years. — [arXiv HTML v5](https://arxiv.org/html/2406.07016v5)
- Counts: 454 excess words in 2024 (343 lemmas), against a Covid-era peak of 190 in 2021. 379 of them were style words. Of the 2024 style words, 66% were verbs and 14% adjectives. Covid-era excess words were content nouns such as "respiratory". — [arXiv HTML v5](https://arxiv.org/html/2406.07016v5)
- Named markers with ratios: delves (r = 28.0), underscores (r = 13.8), showcasing (r = 10.7). Other named markers: potential, findings, crucial, intricate, notably, pivotal, realm. — [arXiv HTML v5](https://arxiv.org/html/2406.07016v5)
- The "common set" of 10 words used for the lower bound: across, additionally, comprehensive, crucial, enhancing, exhibited, insights, notably, particularly, within. The "rare set" has 291 style words with 2024 frequency below 0.02 (Fig. S6), including accentuates, adept, burgeoning, commendable, intricacies, meticulous, multifaceted, pivotal, showcasing. — [arXiv HTML v5](https://arxiv.org/html/2406.07016v5)
- Lower bound: Δ_G = P_G − Q_G is the excess share of abstracts containing at least one word from group G. Δ_rare = 0.136 and Δ_common = 0.134, averaging to 13.5%. — [arXiv HTML v5](https://arxiv.org/html/2406.07016v5)
- Heterogeneity:
  - By country: UK/Australia ≈ 0.05; China, South Korea and Taiwan ≈ 0.20.
  - By journal: Sensors ≈ 0.25, Cureus ≈ 0.20; Nature/Science/Cell 0.07; Nature-family 0.10.
  - By field: computational fields ≈ 0.20.
  - Intersections: Chinese computation papers 0.41; deep-learning object detection ≈ 0.50.
  - The authors note that native speakers may be better at editing out unnatural LLM words, so the bound undercounts them. — [arXiv HTML v5](https://arxiv.org/html/2406.07016v5)
- Code: GitHub berenslab/llm-excess-vocab. A July 2025 update adds six months of PubMed data and computes excess at monthly rather than yearly frequency. This comes from a secondary listing that I did not check against the repo. — [science.ecosyste.ms listing](https://science.ecosyste.ms/projects/142567)

**Liang et al., "Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews" (ICML 2024; arXiv 2403.07183)**
- In ICLR 2024 reviews, "commendable", "meticulous" and "intricate" became 9.8x, 34.7x and 11.2x more likely to appear in a sentence. Validation showed these adjectives are disproportionately frequent in AI-generated text. — [arXiv 2403.07183](https://arxiv.org/pdf/2403.07183)
- Estimated share of review sentences substantially modified by ChatGPT: 10.6% at ICLR 2024 and 16.9% at EMNLP, with a similar trend at NeurIPS. There was no significant evidence in Nature-portfolio reviews. — [arXiv 2403.07183](https://arxiv.org/pdf/2403.07183)
- Press coverage adds "innovative" and "comprehensive" to the flagged adjectives. — [The Register, 19 Mar 2024](https://www.theregister.com/2024/03/19/ai_researchers_reviewing_peers/)

**Liang et al., "Mapping the increasing use of LLMs in scientific papers" (arXiv 2404.01268), published as "Quantifying large language model usage in scientific papers," Nature Human Behaviour, 4 Aug 2025 (DOI 10.1038/s41562-025-02273-8)**
- Method ("distributional GPT quantification"): the corpus is modeled as the mixture (1−α)P + αQ, where P is the human word distribution and Q the LLM-modified one, and α is estimated by maximum likelihood at the sentence level.
  - Q is built with a two-stage generator. An LLM first summarizes a human paragraph into a bullet outline, then an LLM expands the outline back into prose. This keeps the content fixed while simulating LLM-assisted writing.
  - The full vocabulary is used. Adjectives, adverbs and verbs alone also validated well, but the full vocabulary was more stable. Population-level error stayed below 3.5% under temporal shift. — [arXiv 2404.01268 HTML](https://arxiv.org/html/2404.01268)
- Top LLM-skewed words shown: realm, intricate, showcasing, pivotal. All were flat in arXiv CS abstracts from 2010 to 2022 and surged from 2023. — [arXiv 2404.01268 HTML](https://arxiv.org/html/2404.01268)
- Preprint estimates to Feb 2024 (abstracts): CS 17.5%, Electrical Engineering & Systems Science 14.4%, Mathematics 4.9%, Nature portfolio 6.3%. Some intro-section figures are reported inconsistently across sections of the preprint. — [arXiv 2404.01268 HTML](https://arxiv.org/html/2404.01268)
- Published version: 1,121,912 papers, Jan 2020-Sep 2024. CS grew fastest, up to 22% (a news write-up gives 22.5% of CS abstract sentences). Mathematics and Nature portfolio reached up to 9%. — [Nature Index page](https://www.nature.com/nature-index/article/10.1038/s41562-025-02273-8); [Tribune India](https://www.tribuneindia.com/news/science-technology/computer-science-research-papers-show-fastest-uptake-of-ai-use-in-writing-analysis-finds)
- Correlates of higher LLM modification:
  - First authors who post more preprints: 19.3% vs 15.6% in abstracts.
  - Crowded areas: 22.2% vs 14.7%.
  - Papers under 5,000 words: 17.7% vs 13.6%. — [arXiv 2404.01268 HTML](https://arxiv.org/html/2404.01268)
- Disclosure: only 2 of 200 sampled CS preprints from Feb 2024 disclosed LLM use. — [search summary of NHB/arXiv](https://openreview.net/pdf/eff0ea8ed41a89a9e267752af56df912888fee65.pdf)
- Economics: the arXiv field breakdown I retrieved covers CS, EESS, math, physics, stats, bioRxiv and Nature. Economics is not reported separately. — [arXiv 2404.01268 HTML](https://arxiv.org/html/2404.01268)

**Gray, "ChatGPT 'contamination': estimating the prevalence of LLMs in the scholarly literature" (arXiv 2403.16887, Mar 2024)**
- Uses the Dimensions database. In 2023, "intricate" rose 117% and "meticulously" 137%. Papers with at least one of the four strongest indicator words rose 87%, and papers with two or more rose 468%. Estimate: at least 60,000 papers in 2023 (just over 1%) were LLM-assisted, a floor. Gray thinks wholly ChatGPT-written papers are rare and that polishing is the usual use. — [Computing.co.uk](https://computing.co.uk/news/4190311/meticulously-commendable-ais-fingerprints-recent-academic-papers); [Moonlight review](https://www.themoonlight.io/review/chatgpt-contamination-estimating-the-prevalence-of-llms-in-the-scholarly-literature)

**Geng & Trotta, "Human-LLM Coevolution: Evidence from Academic Writing" (Findings of ACL 2025; arXiv 2502.09606)**
- In arXiv abstracts, ChatGPT-favored words such as "delve" fell from about March-April 2024, after they were publicly identified. Others, such as "significant", kept rising. Writers are adapting, which makes detection by obvious markers harder, but word-frequency tracking still measures LLM influence. Leiter et al. (Dec 2024) independently found "delve" declining. — [arXiv 2502.09606 v2](https://arxiv.org/html/2502.09606v2); [ACL Anthology](https://preview.aclanthology.org/setup/2025.findings-acl.657)
- Follow-up (Geng, Dong, Poibeau 2026, arXiv 2603.25638, "Beyond Via"): arXiv titles use "beyond" and "via" more and "the" and "of" less. Real-world LLM usage is "heterogeneous and dynamic". — [arXiv 2603.25638](https://arxiv.org/pdf/2603.25638)

**Juzek & Ward, "Why Does ChatGPT 'Delve' So Much?" (COLING 2025; arXiv 2412.11385)**
- They identify 21 focal words whose rise in scientific abstracts is likely LLM-driven; "delve", "intricate" and "underscore" are named. They find no evidence that architecture, algorithm or training data cause the overuse. Model comparisons are *consistent with* RLHF, but the evidence is only suggestive. — [arXiv 2412.11385](https://arxiv.org/abs/2412.11385)
- A separate study (arXiv 2508.01930) links "delve"-type overuse in Llama models to human-feedback preferences. — [arXiv 2508.01930](https://arxiv.org/pdf/2508.01930v1)

**Reinhart et al., "Do LLMs write like humans?" (PNAS 122(8), 25 Feb 2025; arXiv 2410.16107): word level**
- Most overused words in GPT-4o: camaraderie (162x), tapestry (155x), intricate (119x), palpable (95x). The paper says these and others run above 100x the human rate. — [arXiv 2410.16107v2](https://arxiv.org/html/2410.16107v2)

**Economics-specific evidence**
- Feyzollahi & Rafizadeh, "The adoption of Large Language Models in economics research" (Economics Letters 250, 2025, DOI 10.1016/j.econlet.2025.112265).
  - Design: top 25 economics journals, 2001-2024. The frequency of 25 AI-associated terms is compared with control terms that are stable economics vocabulary (e.g., "coefficient").
  - Named AI terms: "underscore", "nuance" and "leverage" (UMass release); "intricate", "delve" and "foster" (blog summary). The full list is in Table 2, which I could not access.
  - Estimates: +4.76 pp pooled post-ChatGPT; +2.85 pp in 2023; +6.67 pp in 2024. — [UMass news](https://www.umass.edu/news/article/study-finds-accelerating-adoption-ai-language-models-economics-research-papers); [blog summary](https://sex-drugs-economics.blogspot.com/2025/06/how-prevalent-is-large-language-model.html); [IDEAS/RePEc](https://ideas.repec.org/a/eee/ecolet/v250y2025ics0165176525001028.html)
- Renault (Paris-Saclay, working paper): uses OpenAlex data on millions of economics articles and working papers. Adoption is measured three ways: with a classifier, with linguistic markers, and with text complexity. He finds a sharp rise and large heterogeneity across researchers, fields, institutions and countries, and uses the 2023 Italian ChatGPT ban as a natural experiment. I saw the seminar abstract only. — [RITM seminar page](https://www.ritm.universite-paris-saclay.fr/?p=13184)
- Askitas, "The Behavioral Signature of GenAI in Scientific Communication" (IZA DP 18062, 2025): event-study methods on academic writing and GPT adoption. — [EconPapers](https://econpapers.repec.org/RePEc:iza:izadps:dp18062)

### Inferences
- Every one of these methods measures *relative* frequency against a pre-2023 human baseline. A scrub step for a JFEc paper should therefore compute log-ratios of a draft's word frequencies against a corpus of target-journal papers (for example JFEc, JoE and JBES papers from before 2023), not against general English. This is Kobak's δ/r logic applied to one document against a field-matched baseline. It handles field vocabulary automatically: "significant", "empirical" and "coefficient" are common in econometrics and would not be flagged.
- Kobak's common-set words (across, within, particularly, potential, findings, comprehensive) are normal academic words. They are useful only as a *density* signal (rate per 1,000 words against the target-journal baseline), never as hard bans. The rare-set words (delve, underscore, showcase, intricate, meticulous, commendable, pivotal, realm, tapestry, multifaceted, burgeoning, adept, accentuate, notably-as-opener) can be hard-flagged at near-zero false-positive cost in econometrics prose.
- Coevolution means a static list goes stale; newer model favorites are listed in Q3. Claude has its own idiolect (Q2/Q3), so the list should be refreshed by profiling the drafting model's own outputs against the target corpus (slop-forensics style), not only by importing GPT-era lists.

### Gaps
- Kobak's full 291-word rare set (Fig. S6) and the 900 annotated words are in the supplement and the GitHub repo; I did not retrieve the files.
- The Feyzollahi & Rafizadeh 25-term list (Table 2) is paywalled. Neither it nor the published NHB version gives finance or econometrics rates; Liang's arXiv breakdown has no economics category.
- I did not retrieve the first Geng & Trotta paper ("Is ChatGPT transforming academics' writing style?", arXiv 2404.08627) or its estimates.
- The full 21 focal words of Juzek & Ward were not retrieved.

## Q2. Structural markers beyond vocabulary: em dashes, "not X but Y", triads, signposting, hedging, summary closers, uniform sentence length, bolded inline headers, generic transitions

### Takeaway
The strongest *quantitative* structural evidence is grammatical (Reinhart et al., PNAS 2025). Instruction-tuned models use present-participial clauses at 2-5x the human rate, nominalizations at 1.5-2x, and phrasal coordination at about 1.9x. Em-dash overuse is model-specific: by 2026 Claude is reportedly the main remaining over-user, which matters for a Claude-drafted paper. "It's not X, it's Y" is measured at up to 6.3x the human rate in some models. Triads, signposting ("it's important to note"), summary closers, bolded inline headers and title-case headings are documented mainly in practitioner catalogs (Wikipedia, expert-annotator studies), not in controlled corpus studies. Uniform sentence length ("low burstiness") is contested and possibly confounded with text length.

### Cited Findings

**Grammar and syntax (Reinhart et al., PNAS 2025)**
- Corpora: the Human-AI Parallel English (HAP-E) corpus across six genres including academic, the COCA AI Parallel corpus, and M4 arXiv abstracts as a check. Models: Llama 3 variants and GPT-4o, using Biber's lexical, grammatical and rhetorical features. — [arXiv 2410.16107v2](https://arxiv.org/html/2410.16107v2)
- Feature rates in GPT-4o vs humans:
  - Present participial clauses about 5.3x. These are the ", highlighting X" / ", ensuring Y" tails.
  - "That"-clauses as subject about 2.6x.
  - Nominalizations about 2.1x.
  - Phrasal coordination about 1.9x.
  - Agentless passive at about half the human rate.
  - Clausal coordination avoided; downtoners used more.
  - Llama 3 Instruct: present participial clauses 2-5x, nominalizations 1.5-2x. — [arXiv 2410.16107v2](https://arxiv.org/html/2410.16107v2)
- Detectability: classifiers on Biber features separate humans from a specific LLM at 93-98%. Instruction-tuned models are *less* human-like than their base models; tuning pushes a dense, noun-heavy style that limits genre mimicry. — [arXiv 2410.16107v2](https://arxiv.org/html/2410.16107v2)
- Copula avoidance ("serves as", "stands as", "features" instead of "is/are"): the Wikipedia catalog cites a study finding a drop of more than 10% in "is"/"are" in 2023 academic writing, and the same effect in GPT-3.5 revisions of 10,000 abstracts. — [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)
- Superficial analyses: trailing "-ing" phrases ("highlighting", "ensuring", "fostering", "enhancing", "valuable insights") that attach unattributed significance claims. — [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)

**Em dashes**
- Freeburg, "The Last Fingerprint: How Markdown Training Shapes LLM Prose" (arXiv 2603.27006, 27 Mar 2026; single-author preprint, not peer reviewed).
  - Scope: about 240k words from 12 instruction-tuned models across 5 providers (Anthropic, OpenAI, Meta, Google, DeepSeek).
  - Rates: em dashes range from 0.0 per 1,000 words (Llama) to 9.1 (GPT-4.1, under suppression instructions).
  - Suppression: when told to avoid markdown, headers, bullets and bold disappear, but em dashes persist in every model except Llama. Even explicit "no em dash" instructions fail for some models.
  - Origin: base-vs-instruct comparisons show the tendency exists before RLHF. The author's hypothesis is that it is a remnant of markdown structure. — [arXiv 2603.27006](https://arxiv.org/abs/2603.27006)
- Secondary report: GPT-4.1 essays use em dashes at 3.28x the human rate (unverified against the preprint). — [McGill OSS](https://www.mcgill.ca/oss/article/critical-thinking-student-contributors-technology/why-did-llms-steal-our-em-dashes)
- Wikipedia (Oct 2026) now lists em-dash overuse as a *historical* indicator (2022-Sept 2026).
  - GPT-5.1 was reportedly tuned to suppress them.
  - A July 2026 Economist study reportedly found that among contemporary models *only Claude* used em dashes more than professional writers, while ChatGPT used fewer.
  - From Sept 2026 some models were reportedly hard-coded to avoid them, so excess commas and periods are now their own sign.
  - AI em dashes are often spaced (" — "), against common typographic practice. — [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)
- Post-editing study (arXiv 2604.24444, "Can You Make It Sound Like You?"): participants removed 23% of the 254 em dashes in LLM drafts. — [arXiv 2604.24444](https://arxiv.org/pdf/2604.24444)
- Informal counts: about 1 em dash per 130 words in 24 AI-first drafts vs 1 per 850 in 15 human pieces (small and uncontrolled). — [Lilach Bullock blog](https://www.lilachbullock.com/does-ai-writing-use-a-lot-of-dashes/)

**"Not X but Y" / negative parallelism**
- Antislop (Paech et al., arXiv 2510.15061, Oct 2025; ICLR 2026 per one listing):
  - "It's not X, it's Y" constructions appear up to 6.3x more often than in human writing in some models.
  - The paper ships a regex blocklist for "not xx, but yy" constructions (Appendix L). Regex-ban sampling cuts the pattern to 0.00 hits per 1k characters (qwen3-4b). — [arXiv 2510.15061 HTML](https://arxiv.org/html/2510.15061); [arXiv abs](https://arxiv.org/abs/2510.15061)
- Wikipedia lists the forms "Not only... but also", "It's not X, it's Y" and "Y rather than X", noting they also occur in human listicles. "Rather than" disclaimers are a newer sign from mid-2026. — [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)
- Users report that banning the phrase in prompts produces paraphrases ("rather than X, you're really looking at Y"). This is anecdotal. — [hsu.cy note](https://hsu.cy/notes/202602272252/); [essay on "It's not X, it's Y"](https://news.lavx.hu/article/why-it-s-not-x-it-s-y-isn-t-the-problem-it-s-the-metrics-we-use-to-police-language)

**Triads, signposting, hedging, summary closers, formatting**
- Rule of three: overuse of "adjective, adjective, adjective" or "short phrase, short phrase, and short phrase", which makes a superficial analysis look comprehensive. — [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)
- Signposting and hedging, by period:
  - 2023-24 models used didactic "it's important to note".
  - 2024-25 retrieval chatbots wrote "not widely documented" or "based on available information".
  - Mid-2026 models use "rather than", "does not establish" and "should not be confused with". — [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)
- Summary closers: "In summary", "In conclusion" and "Overall", plus paragraphs that end by restating their point. These are now listed as historical, mainly in older models. — [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)
- Formatting signs:
  - Title-case headings.
  - Mechanical boldface ("key takeaways" style).
  - Inline-header vertical lists (bullet + **bold header** + colon + text).
  - Emoji bullets.
  - Markdown left in the text. — [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)
- Other rhetorical signs:
  - Undue emphasis on significance: "stands as", "is a testament", "pivotal", "underscores its importance", "setting the stage for".
  - Vague attributions: "Experts argue", "Observers have cited".
  - Formulaic challenge sections: "Despite these challenges", "Future Outlook". — [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)
- Elegant variation: cycling synonyms for one referent. Huang et al. (2025) found it more frequent in post-2023 Wikipedia text and in GPT-4o-mini and Gemini-1.5-Flash output. Wikipedia attributes it to repetition penalties in older models. — [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)
- Generic transitions: Wikipedia classes sentence-initial "Additionally", "Consequently" and "Notably" as *weak* evidence on their own because humans use them too. "Additionally" is still on its AI-vocabulary list. — [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)
- Expert annotators' clue categories (Russell, Karpinska & Iyyer, ACL 2025), by share of explanations: vocabulary 53.1%, sentence structure 35.9%, grammar & punctuation 24.8%, originality 23.7%, quotes 22.3%, clarity 19.5%. Formulaic structure and conclusions were also common clues. — [arXiv 2501.15654v2](https://arxiv.org/html/2501.15654v2)
- Syntactic templates: Shaib et al. study repeated part-of-speech templates as a templatedness measure; their 2025 slop taxonomy scores templatedness with automatic metrics. — [Measuring AI "Slop" in Text, arXiv 2509.19163](https://arxiv.org/html/2509.19163v2)

**Uniform sentence length / burstiness (contested)**
- One study finds human sentences vary more in length and that GPT output is consistently longer and less "bursty". — [arXiv 2408.04647](https://arxiv.org/pdf/2408.04647)
- Against it:
  - CT² finds the burstiness gap shrinks as models get larger. — [arXiv 2310.05030](https://arxiv.org/pdf/2310.05030)
  - A NeurIPS 2026 paper argues raw sentence-length variance grows with mean length, so the usual metric is confounded with length (abstract only). — [NeurIPS 2026: "Burstiness Was Measured Wrong..."](https://neurips.cc/virtual/2026/164820)
  - A non-peer-reviewed analysis of about 5,900 length-matched pairs finds burstiness AUROC 0.521, near chance. — [opace.agency](https://opace.agency/tools/ai/content-verification-integrity/research/ai-writing-myths-measured/)
- Wikipedia's mid-2026 note describes newer models (GPT-6 era) as writing sentences of roughly 8-20 words with more periods and commas. — [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)

### Inferences
- For a Claude-drafted paper, the highest-yield *mechanical* structural checks are, in order:
  1. Em-dash rate per 1,000 words against the target-journal baseline, given the reported Claude-specific persistence (Freeburg; the Economist via Wikipedia).
  2. Sentence-final present-participial tails (`, (highlighting|underscoring|emphasizing|reflecting|ensuring|showcasing|demonstrating|suggesting|indicating) `).
  3. "not (only|just|merely) ... but (also)?" and "(it's|it is) not X(,|;) (it's|it is)" contrasts.
  4. Signposting stock phrases ("it is (worth|important) (noting|to note)", "notably,", "importantly,").
  5. Summary openers ("In summary", "Overall,", "In conclusion,") outside the Conclusion section.
  6. Copula substitutes ("serves as", "stands as", "acts as a").
  7. Triads: count "X, Y, and Z" lists per paragraph; flag paragraphs with two or more.
  8. LaTeX `\textbf{...}:` or `\paragraph{...}` inline headers and title-case `\subsection` titles, against JFEc house style.
- Rates matter more than presence. Kobak's and Liang's evidence is about population frequencies, so each check should report a per-1,000-word rate against a target-literature baseline and flag outliers. Zero tolerance is warranted only for the rare markers.
- Do not enforce uniform-sentence-length metrics as a hard gate. The evidence is contested and length-confounded. Report sentence-length SD for information only.
- Nominalization density (-tion/-ment/-ness/-ity nouns per 100 words) and phrasal-coordination density are measurable with a POS tagger (spaCy). Reinhart's 2x ratios make them plausible soft metrics. However, econometrics prose is itself nominal-heavy, so they need the field baseline.

### Gaps
- No peer-reviewed corpus study measures triads, signposting phrases, hedging stacks or summary closers in LLM vs human *academic* prose; the evidence is catalog-level (Wikipedia, annotator studies).
- No peer-reviewed human baseline for em-dash rates by academic genre was found; the Freeburg abstract gives none, and the Economist study (July 2026) is cited only secondhand via Wikipedia.
- The 6.3x "It's not X, it's Y" figure comes from creative-writing outputs, not academic prose.
- I could not view the Russell et al. guidebook tables (Tables 11-12, full AI-vocab list).

## Q3. Practical catalogs and tools usable as an automated scrub step

### Takeaway
Usable building blocks exist:
- The Wikipedia "Signs of AI writing" catalog: the most complete, actively maintained, and era-dated.
- Paech's slop-forensics and auto-antislop: MIT-licensed code that *derives* over-represented words, n-grams and regex patterns for a given model against a human baseline.
- Several Vale packages (vale-ai-tells x2, deslop, jdkato/voices, an Ammil ruleset built from the Wikipedia page): drop-in prose-lint rules.

Commercial detectors are moving away from text-only scoring: Writefull withdrew its GPT detector, and Grammarly Authorship tracks the writing process. None of these is a style aligner. The right design combines a regex/word-list lint (Vale or a Python script), a target-corpus frequency comparison, and an LLM revision pass for the semantic patterns.

### Cited Findings

**Wikipedia: Signs of AI writing (WikiProject AI Cleanup), as of Oct 2026**
- Status: a descriptive guide, not policy. Detection tools have "non-trivial error rates". Fixing only the surface signs can make detection harder without fixing the underlying problems. — [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)
- AI-vocabulary list: "delve, crucial, pivotal, tapestry, showcase, underscore, intricate, additionally". One such word is weak evidence; many in one edit is strong. — [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)
- Era-specific lists (claims of the page):
  - 2023-mid 2024 (GPT-4): "delve, garner, testament".
  - Mid 2024-mid 2025 (GPT-4o): "align with, fostering, highlighting".
  - Mid 2025-mid 2026 (GPT-5): "emphasizing, enhance, showcasing".
  - Mid 2026 onward (GPT-6): "leverage, multifaceted, prioritize", with more periods and commas and sentences of about 8-20 words.
  - Grok: "causal, empirical, correlate, underscore". — [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)
- Category list:
  - Undue emphasis on significance or legacy.
  - Superficial -ing analyses.
  - Promotional language ("boasts a", "vibrant", "rich", "groundbreaking", "diverse array").
  - Vague connections ("in connection with", "associated with").
  - Vague attributions.
  - Formulaic challenge sections.
  - Copula avoidance and negative parallelisms.
  - Rule of three.
  - Title case, overused boldface, inline-header lists.
  - Hedging disclaimers.
  - Collaborative chatbot phrasing ("Certainly!", "I hope this helps").
  - Section summaries.
  - Elegant variation.
  - Citation problems (invalid DOIs, mismatched DOIs, `utm_source=` parameters).
  - Model-specific idiolects. — [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)
- Ineffective indicators: perfect grammar, formal or "academic" prose in general, "bland" prose, and transition words alone. — [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)
- Signs of *human* writing (useful as rewrite targets):
  - Simple "is/has" phrasing ("there is a", "it has a").
  - Plain words over stiff synonyms ("used" over "utilized", "wrote" over "authored").
  - Definite superlatives ("is the only", "was the first").
  - Ordinary qualifiers ("perhaps", "tends to").
  - Even wordy constructions ("in order to", "the fact that"). — [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)
- Model differences: each model has a distinct idiolect. Gemini and Claude responses tend to be more concise than ChatGPT and Grok. — [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)

**Sam Paech: slop-forensics, auto-antislop, Antislop sampler / FTPO**
- slop-forensics (MIT) runs four stages:
  - Generates about 100 outputs per model through OpenRouter or another OpenAI-compatible API.
  - Profiles single words, bigrams and trigrams (stop words and numbers excluded) and scores repetition, vocabulary complexity and a "Slop Index".
  - Merges cross-model over-represented terms into `slop_list.json`, `slop_list_bigrams.json`, `slop_list_trigrams.json` and `slop_list_phrases.jsonl`.
  - Clusters models into phylogenetic trees.
  - The default prompts are creative writing. — [GitHub: sam-paech/slop-forensics](https://github.com/sam-paech/slop-forensics)
- Antislop paper (arXiv 2510.15061):
  - Over-representation ratio ρ(p) = f_LLM(p)/f_human(p). Words are baselined on the `wordfreq` library; n-grams on Reddit creative writing plus Project Gutenberg.
  - Some patterns occur more than 1,000x more often than in human text.
  - The sampler backtracks on banned strings or regexes and resamples. It suppresses more than 8,000 patterns while keeping quality, whereas token banning becomes unusable at about 2,000 patterns.
  - FTPO fine-tuning cuts slop by about 90% while staying within about 1% of baseline writing quality (67.80 → 68.93 at 2k patterns). DPO drops quality 6-15 points and lexical diversity to 74-92% of baseline.
  - Code: [github.com/sam-paech/auto-antislop](https://github.com/sam-paech/auto-antislop) (MIT). — [arXiv 2510.15061 HTML](https://arxiv.org/html/2510.15061)

**Vale and prose-lint rule sets**
- krishnasunkam/vale-ai-tells: 17 rules derived from real editing passes and scored against a labeled benchmark. 6 are gated as errors (mechanical tells); the rest are suggestions. MIT. — [EveryDev listing](https://www.everydev.ai/tools/vale-ai-tells)
- tbhb/vale-ai-tells: 25 YAML rules covering vocabulary fingerprints, structural patterns and rhetorical tells (license not confirmed). — [TomeVault](https://tomevault.io/tome/tbhb/vale-ai-tells)
- JMill/deslop: a Vale package for stock phrases, hedges and structural tics (e.g., "plethora", "delve"). — [TomeVault](https://tomevault.io/tome/JMill/deslop)
- jdkato/voices, from Vale's author: turns AI-writing instructions into Vale rules. Its word lists come from Peter Yang's "no-ai-slop" (MIT). — [GitHub: jdkato/voices](https://github.com/jdkato/voices); [Vale blog](https://vale.sh/blog/voices)
- Ammil Industries: a personal Vale ruleset modeled on the Wikipedia page, split into warning and suggestion levels. — [Ammil post (via reader)](https://hn.jakechampion.name/reader/https://ammil.industries/signs-of-ai-writing-a-vale-ruleset)
- Vale packages install through `.vale.ini` + `vale sync`. — [Vale custom styles guide](https://www.mintlify.com/errata-ai/vale/guides/custom-styles)

**Commercial tools**
- Writefull's GPT Detector has been taken offline. Writefull says detection becomes less accurate as AI improves, which makes it unreliable for academic-integrity decisions. — [Writefull GPT Detector page](https://x.writefull.com/gpt-detector)
- Grammarly Authorship tracks the writing process and "isn't a form of AI detection". Since April 2025 Authorship reports include AI and plagiarism flags, and Grammarly's separate AI detector gives a percentage of text that appears AI-generated. I found no independent accuracy data. — [Grammarly Authorship](https://www.grammarly.com/authorship?id=81); [Grammarly AI detector](https://grammarly.com/ai-detector); [Grammarly blog](https://www.grammarly.com/blog/academic-writing/grammarly-authorship-just-got-a-major-update/)
- Russell et al. benchmarked Pangram, GPTZero, Binoculars, Fast-DetectGPT and RADAR. Many dropped sharply on humanized text, while Pangram's humanizer-robust model kept 96.7% TPR. A GPT-4o prompt-based detector given the expert guide had the highest TPR of the LLM-judge detectors. — [arXiv 2501.15654v2](https://arxiv.org/html/2501.15654v2)

### Inferences
Proposed implementable scrub stack, synthesized from the sources above:
- **Tier-1 hard flags** (rare markers with large ratios; near-zero legitimate use in econometrics prose): delve(s/d), underscore(s/d) (verb), showcase/showcasing, intricate/intricacies, meticulous(ly), commendable, pivotal, realm, tapestry, testament, multifaceted, burgeoning, adept, accentuate(s), garner, palpable, camaraderie, vibrant, boasts, groundbreaking, "plethora", "a diverse array", "stands as", "serves as a testament", "setting the stage for", "indelible". Sources: Kobak, Liang ICML/NHB, Gray, Reinhart, Wikipedia, deslop.
- **Tier-2 density flags** (normal words, flag only when the rate is well above the target-journal baseline): additionally (sentence-initial), notably, crucial, comprehensive, enhance/enhancing, insights, particularly, potential, findings, exhibited, foster(ing), leverage, align with, highlighting, emphasizing, nuance(d), prioritize, robust (non-statistical sense), landscape, seamless. Whitelist statistical "significant(ly)", "robust (standard errors)", "causal", "empirical", "correlate" for an econometrics paper.
- **Tier-3 regex structural rules**: em dash rate (and spaced " — "), present-participial tails, not-X-but-Y and "rather than" contrasts, "it is worth/important noting", summary openers, copula substitutes, triads per paragraph, `\textbf{}`-colon inline headers, title-case section titles, vague attributions ("researchers have argued", "it is widely recognized" without a citation).
- **Tier-4 model-specific refresh**: run slop-forensics/auto-antislop-style profiling of the drafting model (Claude) on prompts that mirror the paper's sections. Compute ρ against a corpus of JFEc/JoE/JBES papers rather than wordfreq, so the lists capture *this* model's idiolect against *this* literature.
- **Tier-5 LLM pass** for what regexes cannot see: superficial significance claims, unnecessary exposition, elegant variation, vague attribution, hedging calibrated to the actual evidence, restated-summary paragraphs, and generic claims that need a specific number or citation. Feed it the Wikipedia categories and target-journal exemplar paragraphs.
- Vale is the cheapest integration for LaTeX if its markup scoping handles `.tex` adequately. Otherwise a Python linter (regex + spaCy POS for participial clauses and nominalizations) over detexed text matches the repo's existing Python tooling.

### Gaps
- The full content of each Vale package's rules, and whether any handles LaTeX, was not verified.
- I could not confirm how the EQ-Bench Creative Writing "slop score" is computed; the Antislop paper uses EQ-Bench only as an out-of-distribution check.
- I did not confirm the existence or contents of a separate "antislop-sampler" repo; the paper names auto-antislop and antislop-vllm.
- No catalog targets *academic economics/finance* LLM style specifically; all lists are general, biomedical or creative-writing derived.

## Q4. Does removing these markers improve human-rated quality, or only evade detectors? Evidence from economics/finance referees and editors

### Takeaway
Removing marker *vocabulary* mainly reduces detectability by automatic detectors. It does not reliably make text read as human to experienced readers, and it does not by itself fix what readers dislike. Experts still caught "humanized" text, and expert-defined "slop" is mostly about information utility, relevance, coherence, templatedness and verbosity rather than individual words. Edits aimed at those deeper idiosyncrasies (clichés, unnecessary exposition) do improve alignment with human preferences, though expert human edits still beat automatic ones. Pattern suppression at generation time can preserve or slightly raise rated quality when done carefully (Antislop/FTPO). Crude suppression (token bans, DPO) degrades it. Economics-specific evidence is thin: one word-frequency study of the top 25 journals, editor anecdotes that extra submissions do not look conspicuously AI-written, and disclosure requirements at the Journal of Econometrics and the AEA.

### Cited Findings

**Detector evasion vs human perception**
- Russell, Karpinska & Iyyer (ACL 2025):
  - Five frequent ChatGPT users, voting together, misclassified 1 of 300 non-fiction articles. They beat most commercial and open-source detectors even under paraphrasing and humanization.
  - Paraphrasing (GPT-4o): expert majority vote stayed at 100% TPR and 0% FPR.
  - Humanization (o1-Pro, prompted with a guidebook of expert clues to avoid): the majority vote still reached 100% TPR. The share of correct explanations citing vocabulary fell from 57.1% to 42.3%, so experts switched to other cues.
  - One annotator was fooled consistently. Many automatic detectors dropped sharply on humanized text. o1-Pro was hardest for experts and GPT-4o easiest. — [arXiv 2501.15654v2](https://arxiv.org/html/2501.15654v2); [arXiv abs](https://arxiv.org/abs/2501.15654)
- Wikipedia's guidance states that fixing surface signs may only make detection harder without fixing the underlying problems. — [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)
- Kobak et al. note that native speakers may edit out unnatural LLM words, which hides their LLM use from word-based estimators. Geng & Trotta show "delve" declining after publicity, i.e., writers scrub markers. Both point to evasion of word-count detection, not to quality change. — [Kobak arXiv HTML v5](https://arxiv.org/html/2406.07016v5); [Geng & Trotta arXiv](https://arxiv.org/html/2502.09606v2)

**What readers actually object to**
- Shaib, Chakrabarty, Garcia-Olano & Wallace, "Measuring AI 'Slop' in Text" (arXiv 2509.19163, Sept 2025).
  - Taxonomy from interviews with 19 experts in NLP, writing, linguistics and philosophy, in three themes:
    - Information Utility: density, relevance.
    - Information Quality: factuality, bias.
    - Style Quality: repetition, templatedness, coherence, fluency, verbosity, word complexity, tone.
  - Validation: span annotations on 150 news articles and 100 QA passages; about 35% of samples were judged slop.
  - Binary slop judgments are somewhat subjective but correlate with coherence and relevance. — [arXiv 2509.19163](https://arxiv.org/html/2509.19163v2); [Northeastern Khoury](https://www.khoury.northeastern.edu/?p=58466)
- Chakrabarty, Laban & Wu, "Can AI writing be salvaged?" (CHI 2025, Best Paper nominee/Honorable Mention).
  - Taxonomy: seven categories of LLM idiosyncrasies (e.g., clichés, unnecessary exposition).
  - LAMP corpus: 1,057 LLM paragraphs edited by 18 professional writers (8,035 fine-grained edits).
  - GPT-4o, Claude-3.5-Sonnet and Llama-3.1-70b show no quality difference between them, i.e., limitations common to all.
  - Experts largely prefer expert-edited text, but automatic editing methods "show promise" in improving alignment between LLM and human text. — [arXiv 2409.14509](https://arxiv.org/abs/2409.14509); [Stony Brook](https://ai.stonybrook.edu/node/257305)
- Antislop: FTPO suppression of about 90% of slop patterns kept writing quality within about 1% of baseline (67.80 → 68.93 at 2k patterns). DPO reduced quality by 6-15 points and diversity to 74-92%. Token banning collapsed quality to 28 at 8k patterns. Quality was scored on creative writing. — [arXiv 2510.15061 HTML](https://arxiv.org/html/2510.15061)
- Reinhart et al.: instruction-tuned style is a dense, noun-heavy register that "limits their ability to mimic other writing styles" (genre misalignment). The problem is register, not just words. — [arXiv 2410.16107v2](https://arxiv.org/html/2410.16107v2)

**Economics/finance editors and journals**
- Paul Goldsmith-Pinkham, "AI in journals, conferences, and submissions" (Substack, 28 Sept 2026).
  - He surveyed editors of 16 journals. Most did not feel swamped by AI-generated submissions, and several said the extra submissions did not look conspicuously AI-written; he calls this "editorial intuition, not hard data".
  - Submission growth: notable at the QJE, Journal of Finance, JME, JBES and Journal of Econometrics. The AER was flat. Journals with low fees grew more.
  - Disclosure rules: the Journal of Econometrics requires an AI-use declaration at first submission. The AEA requires a brief description of AI use and covers editing; JoLE focuses on substantial rewriting. He found no AI policy at Econometrica.
  - He advises checking publisher-level policies. — [Goldsmith-Pinkham Substack](https://paulgp.substack.com/p/ai-in-journals-conferences-and-submissions)
- Development economics journals: editors report no large wave of fully AI-generated submissions but growing use of AI to improve writing, especially among non-native English speakers. — [fundsforNGOs news, June 2026](https://news.fundsforngos.org/2026/06/09/development-economics-journals-receive-over-16000-submissions-in-2026/)
- Outside economics: Organization Science editors reportedly ran an AI detector and found about a quarter of submissions looked mostly AI-written. Those papers tended to be lower quality and more often desk-rejected. I have this secondhand only. — [World Bank Development Impact weekly links](https://blogs.worldbank.org/en/impactevaluations/weekly-links-july-31--ai-use-in-journal-submissions-and-refereei)
- Michael Cameron's summary of Feyzollahi & Rafizadeh (2025):
  - He sees word-frequency measures as a proxy and likely an undercount, because authors rewrite LLM text and LLMs avoid the known tells.
  - He has not found reason to doubt the results of papers where LLM use is evident.
  - He cites a John List LinkedIn poll in which only 14% said Claude's use in a paper need not be disclosed. — [blog](https://sex-drugs-economics.blogspot.com/2025/06/how-prevalent-is-large-language-model.html)

### Inferences
- The scrub step should be framed as *quality and register alignment*, with detector evasion as a side effect, not the goal. The evidence suggests:
  - Word swaps alone leave the experts' other cues intact: sentence structure, formulaic organization, summary conclusions, lack of originality and specificity.
  - Word swaps alone fix none of the slop dimensions readers penalize: low information density, irrelevance, verbosity, templatedness.
- Highest-value LLM-pass instructions, drawn from the slop and LAMP taxonomies:
  - Cut sentences that only assert importance.
  - Replace generic claims with the paper's specific estimates and citations.
  - Remove unnecessary exposition and restated summaries.
  - Keep consistent terminology. Avoid elegant variation, which is harmful in econometrics, where one estimator or variable should have one name.
  - Calibrate hedges to the actual test statistics.
- Because the JoE and AEA require AI-use disclosure, and because a scrub that only defeats detectors could look like concealment, the pipeline should sit next to a disclosure statement rather than replace one. Whether JFEc (Oxford University Press) requires disclosure was not checked; see Gaps.
- Measuring success: compare the scrubbed draft against target-journal papers on the metric suite (tier-2 densities, em-dash rate, participial-tail rate, nominalization rate). Optionally have a blinded LLM judge, given the Russell-style clue guide, rate "AI-likeness" and "information density" before and after. Detector scores (Pangram, GPTZero) are a weak secondary check, given their documented brittleness.

### Gaps
- No study found measures whether removing LLM markers changes *referee or editor ratings* of academic papers, in economics or elsewhere; the quality evidence comes from creative writing (Antislop, LAMP) and news/QA (Shaib et al.).
- I did not check JFEc's or Oxford University Press's AI-use disclosure policy, and found no finance-journal (JF, JFE, RFS) editorial on AI prose style; Goldsmith-Pinkham's post covers submission volumes, not prose quality.
- The full seven LAMP categories were not verified in this session. From memory, unverified: clichés, unnecessary/redundant exposition, purple prose, poor sentence structure, lack of specificity, awkward word choice, tense inconsistency.
- I did not retrieve the Organization Science detector finding from its primary source.
